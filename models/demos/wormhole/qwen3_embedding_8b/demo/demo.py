# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B Performance Demo

Measures embedding throughput and latency on Tenstorrent hardware.
Supports data parallelism (DP) to run independent model instances on
separate devices — ideal for embedding workloads which are prefill-only.

Metrics reported:
  - Compile time: time to capture the first prefill trace
  - Prefill time: wall-clock time to embed the full batch
  - Embeddings/s: batch_size / prefill_time
  - Tokens/s:     total_tokens / prefill_time

Usage (standalone, single device):
    python models/demos/wormhole/qwen3_embedding_8b/demo/demo.py

Usage (pytest, picks device from MESH_DEVICE env):
    pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py -sv -k "dp32"
    pytest .../demo.py -sv -k "dp1-batch1-seqlt512"   # batch=1, seq length < 512
"""

import json
import math
import os
import time

import pytest
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
    determine_device_name,
)


def _qwen_embedding_optimizations(model_args):
    """Build a DecodersPrecision config layered on top of `performance` with optional
    BFP8 -> BFP4 weight promotions for Qwen3-Embedding workloads.

    Defaults match upstream `performance` (FF1/FF3 already BFP4 + LOFI).
    The following env vars opt extra weight classes into BFP4 + LOFI fidelity:

      QWEN_FF2_BFP4=1   FF2 (down_proj) weights go BFP4
      QWEN_QKV_BFP4=1   QKV projection weights go BFP4
      QWEN_WO_BFP4=1    WO (attention output) weights go BFP4

    BFP4 cuts each weight's DRAM footprint in half (vs BFP8). Per-knob measured
    on Qwen3-Embedding-0.6B / P150 (dp1-batch{1,8,32}-isl512, real text bs=8):

                 bs=1   bs=8   bs=32   AI/AI   AI/weather (lower=better)
      base       12.1   23.8   247.1   0.685   0.166
      +FF2       12.0   23.3   243.6   0.703   0.243   <- biggest accuracy hit
      +QKV       11.7   23.2   242.2   0.701   0.183
      +WO        12.0   23.6   244.5   0.673   0.181
      +QKV+WO    11.6   22.5   239.4   0.698   0.200   <- best perf/accuracy
      +all 3     11.6   22.6   235.8   0.710   0.261

    QKV+WO is the recommended profile: ~3% wall-time win across all bs with
    only +0.034 AI/weather drift; FF2 doubles the gain but accelerates the
    weather/AI confusion (+0.061 from +QKV+WO baseline). Left opt-in so the
    default test path stays bit-identical for CI/QA.
    """
    base = DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    promote_ff2 = os.getenv("QWEN_FF2_BFP4", "0") == "1"
    promote_qkv = os.getenv("QWEN_QKV_BFP4", "0") == "1"
    promote_wo = os.getenv("QWEN_WO_BFP4", "0") == "1"
    # Companion env vars (read elsewhere in the model):
    #
    #   QWEN_FF13_OUT_BFP8=1   (mlp.py)     - FF1/FF3 matmul output -> BFP8.
    #                                          Halves the activation DRAM write/read
    #                                          bandwidth between FF1/FF3 and FF2.
    #
    #   QWEN_FFNORM_IN_BFP8=1  (decoder.py) - Post-attn residual add output -> BFP8.
    #                                          Propagates through ff_norm and into
    #                                          FF1/FF3 inputs, halving the DRAM read
    #                                          on the [seq, dim] activation. Does NOT
    #                                          touch the rotary path (final residual
    #                                          add still emits bf16).
    #
    #   QWEN_ROPE_PREFILL_L1=1 (rope.py)    - cos/sin/trans_mat tables -> L1.
    #                                          Rotary kernel was MATH=4% on bs=32 due
    #                                          to redundant DRAM reads of cos/sin
    #                                          tables (~128 KB each) on every
    #                                          (head, seq_tile) inner-loop iteration.
    #                                          L1-resident tables hit the ~3 TB/s L1
    #                                          path instead of ~500 GB/s DRAM.
    #                                          Auto-disables if max_seq_len*head_dim*2
    #                                          > 8 MB (still safely fits in L1, but
    #                                          conservative for big-context models).
    #
    #   QWEN_LN_BLOCK_SHARDED=1 (model_config.py / distributed_norm.py)
    #                                        - Block-shard the prefill RMSNorm input
    #                                          across an 8x8 (64-core) grid instead
    #                                          of running the interleaved kernel on
    #                                          16 cores. Cuts the full-dim LN kernel
    #                                          from ~16.6 us -> ~13.0 us (matches
    #                                          BGE-M3); even after I2S/S2I overhead
    #                                          (~1 us each) nets ~1.5 us per LN.
    #                                          56 full-dim LNs/iter (28 attn_norm +
    #                                          28 ff_norm) -> ~85 us / iter on bs=1.
    #                                          q_norm/k_norm already use 130 cores
    #                                          so they're untouched. Auto-disables
    #                                          when block_h*block_w > 64 (i.e. for
    #                                          batched prefill where the per-core
    #                                          shard exceeds the L1 CB budget).
    #
    #   QWEN_RESIDUAL_BFP8=1 (decoder.py)    - post-FFN final residual add -> BFP8.
    #                                          Sibling of QWEN_FFNORM_IN_BFP8 (which
    #                                          covers the post-attn add). Together
    #                                          they keep the ENTIRE residual stream
    #                                          BFP8 across all 28 layers, halving
    #                                          the activation read of every QKV /
    #                                          FF1 / FF3 matmul. Safe wrt rotary
    #                                          (which asserts BF16 inputs): QKV's
    #                                          minimal_matmul / linear path now
    #                                          forces output dtype=bf16 regardless
    #                                          of input dtype, so Q/K split into
    #                                          rotary still get bf16. Cosine-sim
    #                                          actually IMPROVED slightly in the
    #                                          measured run (0.694 -> 0.700 on
    #                                          AI/AI), likely because the BFP8
    #                                          accumulation across more ops cancels
    #                                          some round-toward-zero bias.
    #
    # Cumulative measured impact on Qwen3-Embedding-0.6B / P150:
    #
    #               bs=1   bs=8   bs=32   AI/AI (real text bs=8)   AI/weather
    #   baseline    12.1   23.8   247.1   0.685                    0.166
    #   +QKV+WO     11.7   23.3   239.6   0.698                    0.200
    #   +FF13_OUT   11.3   22.5   232.2   0.694                    0.193
    #   +FFNORM_IN  11.3   21.8   228.5   0.694                    0.196
    #   +ROPE_L1    11.2   21.1   217.9   0.694                    0.196
    #   +LN_BSHARD   9.8   20.8   213.6   0.694                    0.196
    #   +LN_S2I_L1   9.5   20.8   213.4   0.694                    0.196 (rmsnorm fix)
    #   +RESID_BF8   9.4   19.8   208.9   0.700                    0.208
    #   +SDPA_CHK    9.4   19.8   208.3   (untested)                       <- recommended
    #
    # Recommended steady-state config:
    #   QWEN_QKV_BFP4=1 QWEN_WO_BFP4=1 QWEN_FF13_OUT_BFP8=1 QWEN_FFNORM_IN_BFP8=1 \
    #   QWEN_RESIDUAL_BFP8=1 QWEN_ROPE_PREFILL_L1=1 QWEN_LN_BLOCK_SHARDED=1
    #
    # NlpCreateHeads (currently 16 cores / 42 us / call on bs=1) is at a structural
    # ceiling — the kernel splits work along the seq tile axis (16 tiles for
    # ISL=512), and the Sharded variant caps at 8 cores for Qwen3's GQA layout
    # (n_kv_heads=8). Further wins here need a new kernel.
    # SDPA on bs=1 is similarly at a structural ceiling: 16 batch-heads × 1 batch
    # = 16 work units, so 16 cores is the max grid utilization.

    if not (promote_ff2 or promote_qkv or promote_wo):
        return base

    # DecodersPrecision shares a single ModelOptimizations instance across all
    # decoder ids by default (see DecodersPrecision.__init__), so mutating any
    # entry mutates them all. Dedupe by id() to handle the future case where a
    # JSON-driven path gives each layer its own instance.
    seen = set()
    for decoder_id in range(model_args.n_layers):
        opt = base.decoder_optimizations[decoder_id]
        if id(opt) in seen:
            continue
        seen.add(id(opt))
        tp = opt._opt_settings["TensorPrecision"]
        of = opt._opt_settings["OpFidelity"]
        if promote_ff2:
            tp[TensorGroup.FF2] = PrecisionSetting.BFP4
            of[OpGroup.LI_FF2] = MathFidelitySetting.LOFI
        if promote_qkv:
            tp[TensorGroup.WQKV] = PrecisionSetting.BFP4
            of[OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_QKV_DECODE] = MathFidelitySetting.LOFI
        if promote_wo:
            tp[TensorGroup.WO] = PrecisionSetting.BFP4
            of[OpGroup.LI_O_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_O_DECODE] = MathFidelitySetting.LOFI
    base._update_full_name()
    return base


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BLOCK_SIZE = 32

INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

SAMPLE_TEXTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "AI is changing the way humans use computers and machines.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "The weather is sunny today with clear blue skies.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "Baking bread requires flour, water, yeast, and patience.",
    "Neural networks mimic the human brain's structure and function.",
    "Natural language processing enables computers to understand text.",
]

MESH_SHAPES = {
    1: (1, 1),
    2: (1, 2),
    4: (1, 4),
    8: (1, 8),
    32: (8, 4),
}


def load_input_texts(input_file, batch_size):
    """Load input texts from a JSON file or generate synthetic ones."""
    if input_file and os.path.exists(input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        texts = [item["text"] if isinstance(item, dict) else item for item in data]
    else:
        texts = [INSTRUCTION + t for t in SAMPLE_TEXTS]

    while len(texts) < batch_size:
        texts = texts * 2
    return texts[:batch_size]


def get_default_mesh_device_param():
    if ttnn.using_distributed_env():
        try:
            n = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
            return MESH_SHAPES.get(n, n)
        except Exception:
            pass
    n = len(ttnn.get_device_ids())
    return MESH_SHAPES.get(n, n)


def _submesh_has_local_devices(submesh):
    view = submesh.get_view()
    return any(
        view.is_local(ttnn.MeshCoordinate(row, col))
        for row in range(submesh.shape[0])
        for col in range(submesh.shape[1])
    )


def prepare_embedding_model(
    mesh_device,
    global_batch_size,
    max_seq_len,
    optimizations,
    page_params,
    data_parallel=1,
):
    """Build TT model(s), generator, and KV cache for embedding workloads.

    When data_parallel > 1, creates independent model instances on submeshes.
    """
    batch_per_dp = global_batch_size // data_parallel

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    all_submeshes = create_submeshes(mesh_device, data_parallel)
    local_indices = (
        [i for i, s in enumerate(all_submeshes) if _submesh_has_local_devices(s)]
        if isinstance(mesh_device, ttnn.MeshDevice) and data_parallel > 1
        else list(range(len(all_submeshes)))
    )
    submeshes = [all_submeshes[i] for i in local_indices]

    if not submeshes:
        raise RuntimeError("No local submeshes available on this host rank")

    if len(submeshes) != len(all_submeshes):
        logger.info(f"Distributed mode: using {len(submeshes)}/{len(all_submeshes)} local submeshes")

    models = []
    model_args_list = []
    kv_caches = []
    state_dict = None

    for submesh in submeshes:
        model_args_i, model_i, kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=False,
            max_batch_size=batch_per_dp,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        models.append(model_i)
        model_args_list.append(model_args_i)
        kv_caches.append([layer.attention.layer_past for layer in model_i.layers])

    tokenizer = model_args_list[0].tokenizer

    generator = Generator(
        models,
        model_args_list,
        mesh_device,
        tokenizer=tokenizer,
    )

    local_dp = len(submeshes)
    local_batch = batch_per_dp * local_dp

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation).repeat(local_dp)
    page_table = reverse_permutation.reshape(local_batch, paged_attention_config.max_num_blocks // batch_per_dp)

    return generator, model_args_list[0], tokenizer, kv_caches, page_table


def tokenize_and_pad(tokenizer, texts, max_seq_len):
    """Tokenize texts, returning padded input_ids and original lengths."""
    encoded = tokenizer(texts, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    original_lens = attention_mask.sum(dim=1).tolist()
    return input_ids, [int(l) for l in original_lens]


def generate_synthetic_inputs(tokenizer, batch_size, seq_len):
    """Generate random token sequences of exactly seq_len for ISL benchmarking."""
    vocab_size = tokenizer.vocab_size
    low = max(100, 0)
    high = min(vocab_size, 50000)
    input_ids = torch.randint(low, high, (batch_size, seq_len), dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, prompt_lens


def run_embedding_prefill(
    generator, input_ids, page_table, kv_cache, prompt_lens, enable_trace, return_hidden_states, warmup_prefill=True
):
    """Run a single embedding prefill pass and return the result."""
    return generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=enable_trace,
        return_hidden_states=return_hidden_states,
        warmup_prefill=warmup_prefill,
    )


def clear_all_kv_caches(generator):
    """Zero out the KV cache for every DP model instance."""
    for model_instance in generator.model:
        for layer in model_instance.layers:
            k_cache, v_cache = layer.attention.layer_past
            k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
            v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, max_seq_len, input_seq_len, page_params, num_iterations, enable_trace, data_parallel",
    [
        (  # dp1-batch1-short: single text, 1024 tokens, single device
            1,
            1024,
            None,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            5,
            True,
            1,
        ),
        (  # dp1-batch1-seqlt512: batch=1, max_seq_len and ISL both < 512 (synthetic tokens)
            1,
            256,
            256,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            1,
            True,
            10,
        ),
        (  # dp1-batch1-seqlt1024: batch=1, max_seq_len and ISL both < 1024 (synthetic tokens)
            1,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl32
            1,
            128,
            32,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl64
            1,
            128,
            64,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl128
            1,
            128,
            128,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl256
            1,
            256,
            256,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl512
            1,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl1024
            1,
            1024,
            1024,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl2048
            1,
            2048,
            2048,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl4096
            1,
            4096,
            4096,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch1-isl8192
            1,
            8192,
            8192,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl32
            32,
            128,
            32,
            {"page_block_size": 32, "page_max_num_blocks": 128},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl64
            32,
            128,
            64,
            {"page_block_size": 32, "page_max_num_blocks": 128},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl128
            32,
            128,
            128,
            {"page_block_size": 32, "page_max_num_blocks": 128},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl256
            32,
            256,
            256,
            {"page_block_size": 32, "page_max_num_blocks": 256},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl512
            32,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            10,
            True,
            1,
        ),
        (  # dp1-batch32-isl1024
            32,
            1024,
            1024,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl2048
            32,
            2048,
            2048,
            {"page_block_size": 32, "page_max_num_blocks": 2048},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl4096
            32,
            4096,
            4096,
            {"page_block_size": 32, "page_max_num_blocks": 4096},
            3,
            True,
            1,
        ),
        (  # dp1-batch32-isl8192
            32,
            8192,
            8192,
            {"page_block_size": 32, "page_max_num_blocks": 8192},
            3,
            True,
            1,
        ),
        (  # dp1-batch2-isl512
            2,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            10,
            True,
            1,
        ),
        (  # dp1-batch4-isl512
            4,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            10,
            True,
            1,
        ),
        (  # dp1-batch8-isl512
            8,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            10,
            True,
            1,
        ),
        (  # dp1-batch9-isl512
            9,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 9 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch10-isl512
            10,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 10 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch11-isl512
            11,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 11 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch12-isl512
            12,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 12 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch13-isl512
            13,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 13 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch14-isl512
            14,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 14 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch15-isl512
            15,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 15 * 64},
            10,
            True,
            1,
        ),
        (  # dp1-batch16-isl512
            16,
            512,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            10,
            True,
            1,
        ),
        (  # dp1-batch8-short: 8 SAMPLE_TEXTS tokenized as real text and padded
            # by `tokenize_and_pad` up to max_seq_len; with ~30 real tokens per
            # text the per-user padded prefill seq lands at 128 (next power of 2),
            # so total tokens = 8*128 = 1024. Used for the bs>1 cosine-similarity
            # smoke test (synthetic random tokens give meaningless cosine numbers).
            8,
            1024,
            None,
            {"page_block_size": 32, "page_max_num_blocks": 1024},
            3,
            True,
            1,
        ),
        (  # dp32-isl512
            32,
            8192,
            512,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl1024
            32,
            8192,
            1024,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl2048
            32,
            8192,
            2048,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl4096
            32,
            8192,
            4096,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
        (  # dp32-isl8192
            32,
            8192,
            8192,
            {"page_block_size": 32, "page_max_num_blocks": 512},
            5,
            True,
            32,
        ),
    ],
    ids=[
        "dp1-batch1-short",
        "dp1-batch1-seqlt512",
        "dp1-batch1-seqlt1024",
        "dp1-batch1-isl32",
        "dp1-batch1-isl64",
        "dp1-batch1-isl128",
        "dp1-batch1-isl256",
        "dp1-batch1-isl512",
        "dp1-batch1-isl1024",
        "dp1-batch1-isl2048",
        "dp1-batch1-isl4096",
        "dp1-batch1-isl8192",
        "dp1-batch32-isl32",
        "dp1-batch32-isl64",
        "dp1-batch32-isl128",
        "dp1-batch32-isl256",
        "dp1-batch32-isl512",
        "dp1-batch32-isl1024",
        "dp1-batch32-isl2048",
        "dp1-batch32-isl4096",
        "dp1-batch32-isl8192",
        "dp1-batch2-isl512",
        "dp1-batch4-isl512",
        "dp1-batch8-isl512",
        "dp1-batch9-isl512",
        "dp1-batch10-isl512",
        "dp1-batch11-isl512",
        "dp1-batch12-isl512",
        "dp1-batch13-isl512",
        "dp1-batch14-isl512",
        "dp1-batch15-isl512",
        "dp1-batch16-isl512",
        "dp1-batch8-short",
        "dp32-isl512",
        "dp32-isl1024",
        "dp32-isl2048",
        "dp32-isl4096",
        "dp32-isl8192",
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [_qwen_embedding_optimizations],
    ids=["performance"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 200000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), get_default_mesh_device_param())
    ],
    indirect=True,
)
def test_embedding_perf(
    mesh_device,
    batch_size,
    max_seq_len,
    input_seq_len,
    page_params,
    num_iterations,
    enable_trace,
    data_parallel,
    optimizations,
    is_ci_env,
    request,
):
    """
    Embedding performance demo: measures compile time, prefill latency, and throughput.

    max_seq_len:   model capacity (rotary embeddings, KV cache allocation)
    input_seq_len: actual tokens per input (None = use real sample texts)
    """
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if data_parallel > num_devices:
        pytest.skip(f"data_parallel={data_parallel} requires {data_parallel} devices, only {num_devices} available")
    if batch_size % data_parallel != 0:
        pytest.skip(f"batch_size={batch_size} not evenly divisible by data_parallel={data_parallel}")

    # Allow disabling hardware trace capture for reporting/profiling runs.
    # TTNN_CONFIG_OVERRIDES with enable_logging=true issues per-op device syncs and
    # reads, which are illegal inside ttnn.begin_trace_capture. Set QWEN_DISABLE_TRACE=1
    # to fall back to the no-trace execution path for those runs.
    if os.environ.get("QWEN_DISABLE_TRACE", "0") == "1":
        logger.info("QWEN_DISABLE_TRACE=1 set; disabling hardware trace capture for this run.")
        enable_trace = False

    # Optional: capture a single-iteration ttnn graph report for the ttnn-visualizer.
    # Set QWEN_GRAPH_CAPTURE=/abs/path/to/report.json to wrap ONE benchmark iteration
    # in ttnn.graph.full_graph_capture(...). Produces a JSON file that can then be
    # imported offline with: python -m ttnn.graph_report <report.json> <visualizer_db/>
    # This path also forces enable_trace=False (graph capture is incompatible with
    # hardware trace capture).
    graph_capture_path = os.environ.get("QWEN_GRAPH_CAPTURE")
    if graph_capture_path:
        logger.info(f"QWEN_GRAPH_CAPTURE set; will write graph report to {graph_capture_path}")
        enable_trace = False

    skip_warmup = data_parallel > 1
    isl = input_seq_len or max_seq_len

    # Embedding workloads never read the KV cache back, so skip the paged_fill_cache
    # pipeline (+ the K/V bf16->bfp8 typecasts that feed it). Users can still opt out
    # by exporting TT_SKIP_KV_CACHE_FILL=0 before invoking pytest.
    os.environ.setdefault("TT_SKIP_KV_CACHE_FILL", "1")

    profiler = BenchmarkProfiler()
    profiler.start("run")

    test_id = request.node.callspec.id
    tt_device_name = determine_device_name(mesh_device)

    # ---- Build model ----
    batch_per_dp = batch_size // data_parallel
    logger.info(
        f"Building model: global_batch={batch_size}, batch_per_dp={batch_per_dp}, "
        f"dp={data_parallel}, max_seq_len={max_seq_len}, ISL={isl}, device={tt_device_name}"
    )
    profiler.start("build_model")
    generator, model_args, tokenizer, kv_caches, page_table = prepare_embedding_model(
        mesh_device,
        global_batch_size=batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        page_params=page_params,
        data_parallel=data_parallel,
    )
    profiler.end("build_model")
    logger.info(f"Model built in {profiler.get_duration('build_model'):.1f}s (dp={generator.data_parallel})")

    # ---- Prepare inputs ----
    profiler.start("loading_inputs")
    if input_seq_len is not None:
        input_ids, prompt_lens = generate_synthetic_inputs(tokenizer, batch_size, input_seq_len)
    else:
        texts = load_input_texts(None, batch_size)
        input_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, max_seq_len)
    profiler.end("loading_inputs")

    total_input_tokens = sum(prompt_lens)
    logger.info(f"Prepared {batch_size} inputs, ISL={isl}, total tokens = {total_input_tokens}")

    # ---- Warmup / compile ----
    logger.info("Compiling (first prefill)...")
    profiler.start("compile_prefill")
    _ = run_embedding_prefill(
        generator,
        input_ids,
        page_table,
        kv_caches,
        prompt_lens,
        enable_trace,
        return_hidden_states=True,
        warmup_prefill=not skip_warmup,
    )
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    # ---- Benchmark iterations ----
    # QWEN_NUM_ITER overrides the per-test num_iterations (used for clean
    # single-iteration profiler captures, e.g. QWEN_NUM_ITER=1).
    _iter_override = os.environ.get("QWEN_NUM_ITER")
    if _iter_override is not None:
        try:
            override_n = int(_iter_override)
        except ValueError:
            override_n = num_iterations
        if override_n > 0:
            logger.info(f"QWEN_NUM_ITER override active: {num_iterations} -> {override_n}")
            num_iterations = override_n
    logger.info(f"Running {num_iterations} benchmark iterations...")
    iteration_times = []
    embeddings = None

    for i in range(num_iterations):
        # NOTE: clear_all_kv_caches intentionally NOT called here.
        # paged_fill_cache in attention.forward_prefill unconditionally overwrites
        # the page range for this prompt, and SDPA is causal (reads only [0, seq_len)),
        # so there is no stale-data path for an embedding workload. Clearing was
        # profiling at ~31% of per-iteration kernel time for a 4 MB DRAM->DRAM write
        # that gets immediately overwritten by the much-cheaper (0.2 ms) paged_fill.
        generator.prev_page_table = None

        profiler.start(f"inference_prefill_{i}")
        if graph_capture_path and i == 0:
            # Capture exactly one iteration into a ttnn-visualizer report JSON.
            import pathlib as _pl

            _out = _pl.Path(graph_capture_path)
            _out.parent.mkdir(parents=True, exist_ok=True)
            with ttnn.graph.full_graph_capture(str(_out)):
                result = run_embedding_prefill(
                    generator,
                    input_ids,
                    page_table,
                    kv_caches,
                    prompt_lens,
                    enable_trace,
                    return_hidden_states=True,
                    warmup_prefill=False,
                )
            logger.info(f"ttnn graph report written to {_out}")
            logger.info(f"Import it with: python -m ttnn.graph_report {_out} {_out.with_suffix('')}_db/")
        else:
            result = run_embedding_prefill(
                generator,
                input_ids,
                page_table,
                kv_caches,
                prompt_lens,
                enable_trace,
                return_hidden_states=True,
                warmup_prefill=False,
            )
        profiler.end(f"inference_prefill_{i}")

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

        if embeddings is None:
            embeddings = result

    # ---- Compute metrics ----
    avg_prefill_time = sum(iteration_times) / len(iteration_times)
    best_prefill_time = min(iteration_times)

    embeddings_per_sec_avg = batch_size / avg_prefill_time
    embeddings_per_sec_best = batch_size / best_prefill_time
    tokens_per_sec_avg = total_input_tokens / avg_prefill_time
    tokens_per_sec_best = total_input_tokens / best_prefill_time

    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_prefill_time,
        "best_prefill_time": best_prefill_time,
        "embeddings/s_avg": embeddings_per_sec_avg,
        "embeddings/s_best": embeddings_per_sec_best,
        "prefill_t/s_avg": tokens_per_sec_avg,
        "prefill_t/s_best": tokens_per_sec_best,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": data_parallel,
        "input_seq_len": isl,
        "max_seq_len": max_seq_len,
        "total_input_tokens": total_input_tokens,
    }

    # ---- Print results ----
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Qwen3-Embedding-0.6B Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {data_parallel}")
    logger.info(f"  Global batch size:    {batch_size}")
    logger.info(f"  Batch per DP group:   {batch_per_dp}")
    logger.info(f"  Input seq length:     {isl}")
    logger.info(f"  Max seq length:       {max_seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_prefill_time * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_prefill_time * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {embeddings_per_sec_avg:.1f}")
    logger.info(f"  Best embeddings/s:    {embeddings_per_sec_best:.1f}")
    logger.info(f"  Avg tokens/s:         {tokens_per_sec_avg:.0f}")
    logger.info(f"  Best tokens/s:        {tokens_per_sec_best:.0f}")
    logger.info("=" * 60)

    # ---- Cosine similarity sanity check (only for real text inputs) ----
    if data_parallel <= 1 and embeddings is not None and batch_size >= 2:
        emb_np = embeddings.float().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        elif emb_np.ndim > 2:
            emb_np = emb_np.reshape(batch_size, -1)

        sim = cosine_similarity(emb_np)
        logger.info(f"  Cosine similarity [0,1] = {sim[0, 1]:.4f} (should be high, both AI-related)")
        if batch_size >= 4:
            logger.info(f"  Cosine similarity [0,3] = {sim[0, 3]:.4f} (should be low, AI vs weather)")

    # ---- CI benchmark data ----
    profiler.end("run")

    if is_ci_env:
        model_name = model_args.base_model_name if hasattr(model_args, "base_model_name") else "Qwen3-Embedding-0.6B"
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": data_parallel, "tensor_parallel": num_devices // data_parallel},
            input_sequence_length=isl,
            output_sequence_length=0,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B performance demo")
    parser.add_argument("--batch-size", type=int, default=1, help="Global batch size")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID (single-device mode)")
    args = parser.parse_args()

    page_max = max(512, math.ceil(args.max_seq_len / BLOCK_SIZE) * args.batch_size * 2)
    page_params = {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=32768, trace_region_size=50000000, num_command_queues=1
    )

    try:
        profiler = BenchmarkProfiler()
        profiler.start("run")

        texts = load_input_texts(None, args.batch_size)
        optimizations = _qwen_embedding_optimizations

        generator, model_args, tokenizer, kv_caches, page_table = prepare_embedding_model(
            device,
            global_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            optimizations=optimizations,
            page_params=page_params,
            data_parallel=1,
        )

        input_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, args.max_seq_len)
        total_tokens = sum(prompt_lens)

        logger.info("Compile run...")
        _ = run_embedding_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, True, True)

        logger.info(f"Benchmarking {args.iterations} iterations...")
        times = []
        for i in range(args.iterations):
            # See note in test_embedding_perf: paged_fill_cache overwrites, so no
            # clear is needed for embedding prefill.
            generator.prev_page_table = None
            t0 = time.perf_counter()
            _ = run_embedding_prefill(generator, input_ids, page_table, kv_caches, prompt_lens, True, True)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            logger.info(f"  Iter {i}: {(t1 - t0) * 1000:.1f}ms")

        avg_t = sum(times) / len(times)
        best_t = min(times)
        logger.info("")
        logger.info(
            f"Avg: {avg_t * 1000:.1f}ms | {args.batch_size / avg_t:.1f} emb/s | {total_tokens / avg_t:.0f} tok/s"
        )
        logger.info(
            f"Best: {best_t * 1000:.1f}ms | {args.batch_size / best_t:.1f} emb/s | {total_tokens / best_t:.0f} tok/s"
        )

        profiler.end("run")

    finally:
        ttnn.close_device(device)
