# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the pplx-embed-v1-4B perf entry points.

pplx-embed-v1-4B (https://huggingface.co/perplexity-ai/pplx-embed-v1-4b) is a
bidirectional Qwen3-4B derivative from Perplexity AI.  It shares the
Qwen3-Embedding-4B backbone dimensions but differs in two embedding-recipe
ways (identical to the 0.6B pplx model):

  1. **Bidirectional attention** (is_causal=False) — handled by
     PplxBidirectionalAttention in tt/attention.py.
  2. **Mean-token pooling** — instead of last-token extraction, all non-padding
     token hidden states are averaged. Output is an (optionally L2-normalized)
     2560-d vector.

Architecture (4B vs the 0.6B pplx model):
    hidden_size        = 2560   (0.6B: 1024)
    num_hidden_layers  = 36     (0.6B: 28)
    num_attention_heads= 32     (0.6B: 16)
    num_key_value_heads= 8      (both)
    intermediate_size  = 9728   (0.6B: 3072)
    head_dim           = 128    (both)
    GQA ratio          = 4:1    (0.6B: 2:1)

Memory placement — "highest use of L1 + DRAM" (P150, single device):

    The P150 single-user prefill keeps activations in L1 when the per-user
    sequence is <= TT_SHORT_SEQ_L1_PREFILL_MAX (default 512).  Activation bytes
    (bf16) = bs * seq * 2560 * 2:

        bs1  ISL=512 :  2.5 MB  -> L1 single-user path (fastest)
        bs1  ISL=1024:  5.0 MB  -> DRAM (per-user seq > 512)
        bs8  ISL=512 :   20 MB  -> DRAM
        bs32 ISL=512 :   80 MB  -> DRAM

    When activations spill to DRAM, the per-core L1 budget is freed, so we widen
    the MinimalMatmul grid to the full 130-core (13x10) Blackhole grid via
    ``QWEN_MM_GRID=13,10`` — this is the single biggest DRAM-path win for the 4B
    model (matmuls dominate ~60% of device time). For the bs=1 L1 path the
    matmuls stay on the standard grid (the activation is already L1-resident, so
    the minimal-matmul path is intentionally off).

4B-specific optimization notes (carried over from the Qwen3-Embedding-4B
reference):
    - LN block sharding: dim=2560 on an 8x8 grid gives block_h*block_w = 2*10 =
      20 which exceeds the 16-tile per-core cap, so QWEN_LN_BLOCK_SHARDED auto-
      disables. The env var is set but inert for this model.
    - Head-split NLP ops: n_kv_heads=8 (same as 0.6B), so head_groups=8 applies
      identically; bs=1 ISL=512 gets 16*8=128 work units.
    - RoPE L1: 512 * 128 * 2 = 128 KB cos/sin tables, well within budget.
    - With 32 Q-heads (vs 16) SDPA has more batch-head parallelism.

All env vars use ``os.environ.setdefault`` so any single knob can still be
overridden from the shell for A/B comparisons.
"""

import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.attention import PplxBidirectionalAttention
from models.demos.blackhole.pplx_embed_4b.tt.mlp import PplxFusedSwigluMLP
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, copy_host_to_device, get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
    determine_device_name,
)

try:
    from tracy import signpost as _tracy_signpost
except ModuleNotFoundError:

    def _tracy_signpost(*_args, **_kwargs):
        pass


MODEL_NAME = "perplexity-ai/pplx-embed-v1-4b"
BLOCK_SIZE = 32
# Hidden size of the 4B backbone — used for activation-size / L1-fit heuristics.
HIDDEN_DIM = 2560
# 130-core (13x10) MinimalMatmul grid for DRAM-resident workloads on Blackhole.
DRAM_MM_GRID = "13,10"


# ---------------------------------------------------------------------------
# ModelArgs subclass for pplx-embed (trust_remote_code + safetensors loading)
# ---------------------------------------------------------------------------


class PplxModelArgs(ModelArgs):
    """ModelArgs variant that enables ``trust_remote_code`` for the custom
    ``bidirectional_pplx_qwen3`` HuggingFace model type and loads weights
    directly from safetensors to avoid the custom modeling.py requiring
    a newer transformers version.
    """

    def _set_hf_params(self, checkpoint_dir):
        self.trust_remote_code_hf = True
        return super()._set_hf_params(checkpoint_dir)

    def get_max_prefill_chunk_size(self):
        """Same as Qwen3-Embedding-4B (identical architecture)."""
        chunk_sizes = {
            "N150": 4,
            "N300": 64,
            "T3K": 128,
            "TG": 128,
            "P150": 128,
            "P300": 128,
            "P150x4": 128,
            "P150x8": 128,
        }
        div1024 = chunk_sizes.get(self.device_name, 4)
        return div1024 * 1024

    def get_trace_prefill_supported_seq_lens(self):
        """ISL sweep range capped at ``max_seq_len`` to avoid warmup assertion failures."""
        all_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
        return [s for s in all_lens if s <= self.max_seq_len]

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        """Cap warmup sequence lengths at max_seq_len."""
        return [s for s in to_warmup_seq_lens if s <= self.max_seq_len]

    def load_state_dict(self):
        """Load weights from safetensors, bypassing AutoModel/AutoModelForCausalLM.

        pplx-embed's custom HF modeling.py imports ``TransformersKwargs``
        which may not exist in the installed transformers.  Since the model
        weights are standard Qwen3 format, we load them directly from the
        safetensors file(s) and add the ``model.`` prefix that the downstream
        ``standardize_hf_keys`` / ``convert_hf_to_meta`` pipeline expects.

        The 4B checkpoint is sharded across multiple safetensors files (an
        ``model.safetensors.index.json`` weight map), unlike the single-file
        0.6B checkpoint — both layouts are handled here.
        """
        if self.dummy_weights:
            return super().load_state_dict()

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        local_only = os.getenv("CI") == "true"
        logger.info(f"Loading pplx-embed-4B weights from {self.CKPT_DIR} via safetensors...")

        raw_sd = {}
        try:
            # Sharded checkpoint: read the index, then pull each shard once.
            index_path = hf_hub_download(self.CKPT_DIR, "model.safetensors.index.json", local_files_only=local_only)
            import json

            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shard_files = sorted(set(weight_map.values()))
            for shard in shard_files:
                shard_path = hf_hub_download(self.CKPT_DIR, shard, local_files_only=local_only)
                raw_sd.update(load_file(shard_path))
        except Exception:
            # Single-file checkpoint fallback.
            safetensor_path = hf_hub_download(self.CKPT_DIR, "model.safetensors", local_files_only=local_only)
            raw_sd = load_file(safetensor_path)

        # Add ``model.`` prefix so the state dict matches what
        # ``AutoModelForCausalLM.from_pretrained().state_dict()`` would produce.
        # Also create ``lm_head.weight`` from tied embeddings.
        state_dict = {f"model.{k}": v for k, v in raw_sd.items()}
        if "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        from models.tt_transformers.tt.load_checkpoints import (
            convert_hf_to_meta,
            convert_hf_to_meta_no_qkv_permute,
            standardize_hf_keys,
        )

        self.fuse_qkv = any("qkv" in k for k in state_dict)
        self.fuse_mlp = any("gate_up" in k for k in state_dict)
        state_dict = standardize_hf_keys(state_dict)
        if self.use_hf_rope:
            state_dict = convert_hf_to_meta_no_qkv_permute(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)
        else:
            state_dict = convert_hf_to_meta(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in range(self.n_layers, self.full_model_n_layers)]
        for k in keys_dict:
            if any(r in k for r in remv):
                state_dict.pop(k)

        return state_dict


# ---------------------------------------------------------------------------
# Environment / optimizations
# ---------------------------------------------------------------------------


def apply_recommended_env(batched_l1: bool) -> None:
    """Set the recommended optimization env vars (4B backbone).

    Precision: QKV/WO/FF2 weights BFP4, BFP8 residual stream, plus the pplx-safe
    LoFi wins (SDPA + RoPE).  For embedding workloads accuracy is measured by
    cosine similarity, which is robust to low-precision matmuls.

    FF2 (down_proj) BFP4 is enabled by default after on-device validation on this
    machine (P150, ISL=512):

        STS-B Spearman (masked-attn pool):  0.8287 (FF2 BFP8) -> 0.8276 (FF2 BFP4)
        Latency:  bs1 31.2 -> 30.4 ms (-2.6%);  bs32 727.8 -> 685.6 ms (-5.8%)

    The 0.0011 Spearman delta is within run-to-run noise while the throughput gain
    is real (the profiler showed FF2 as the single most expensive matmul — 224us at
    HiFi2/BFP8 vs ~100us for the BFP4/LoFi matmuls).  Opt back out for a precision
    baseline with ``QWEN_FF2_BFP4=0`` from the shell.
    """
    os.environ.setdefault("HF_MODEL", MODEL_NAME)
    # Weight precision — QKV + WO + FF2 to BFP4 (all validated for 4B above).
    os.environ.setdefault("QWEN_QKV_BFP4", "1")
    os.environ.setdefault("QWEN_WO_BFP4", "1")
    os.environ.setdefault("QWEN_FF2_BFP4", "1")
    # Activation precision — full BFP8 residual stream.
    os.environ.setdefault("QWEN_FF13_OUT_BFP8", "1")
    os.environ.setdefault("QWEN_FFNORM_IN_BFP8", "1")
    os.environ.setdefault("QWEN_RESIDUAL_BFP8", "1")
    # Architecture-level TM optimizations.
    os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_ROPE_PREFILL_L1", "1")
    # Lift the historical in0_block_w cap of 8 in find_largest_divisor. FF2 has
    # k=9728, which over a grid_y=8 row is 38 K-tiles whose only divisors are
    # 1, 2, 19 and 38 — the cap pinned it at 2, the value tt-perf-report OPT-004
    # calls out as a floor rather than a target. Measured on P150 bs=1 ISL=512
    # (best prefill, head-splits on):
    #   cap  8 (default) 33.2 ms   FF2 in0_block_w=2
    #   cap 10           32.8 ms
    #   cap 16           32.6 ms
    #   cap 19           29.0 ms   FF2 in0_block_w=19
    #   cap 38           28.8 ms   FF2 in0_block_w=38 (whole K row per block)
    os.environ.setdefault("QWEN_MM_MAX_DIVISOR", "38")
    # RoPE is a cos/sin rotation (operands in [-1,1]); the rotary_embedding_llama
    # op defaults to HiFi4, but LoFi is accuracy-neutral and cheaper. See
    # tt/attention.py:_mllama_rope_prefill.
    os.environ.setdefault("QWEN_ROPE_FIDELITY", "lofi")
    # Inert for 4B (dim=2560 exceeds per-core LN budget) but harmless to set.
    # Open only the chip this single-device demo uses. Without this, tt-metal
    # builds its cluster over every chip on the host: on a 32-chip Galaxy that
    # takes the CHIP_IN_USE lock on all 32 (blocking anyone else on the box) and,
    # observed 2026-09-23, can fail outright in cluster construction with
    # "IndexError: unordered_map::at" while the single chip opens fine.
    # setdefault, so dp32_multiprocess.py's per-process TT_VISIBLE_DEVICES and any
    # explicit choice of chip still win.
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("QWEN_LN_BLOCK_SHARDED", "1")
    # Fused head-split + per-head Q/K RMSNorm (tt/custom_ops/fused_qkv_heads_norm):
    # one generic_op with a compute kernel replaces nlp_create_qkv_heads + q_norm +
    # k_norm, three DRAM-bound passes over the same Q/K/V tensors. Measured e2e
    # (full-pipeline, ISL=512): bs1 25.2->25.0, bs8 155.6->144.7 (-7.0%),
    # bs16 288.7->276.8 (-4.1%), bs32 543.5->519.2 (-4.5%); STS-B 0.8125->0.8135.
    os.environ.setdefault("QWEN_FUSED_HEADS_NORM", "1")
    # ...and RoPE in the same pass (rotated = x @ T, x*cos + rotated*sin per tile), removing
    # the two rotary_embedding_llama ops per layer. E2E: bs1 25.0->23.9 (-4.4%), bs8
    # 144.7->143.4, bs16 276.8->263.5 (-4.8%), bs32 519.2->495.0 (-4.7%); STS-B 0.8134.
    os.environ.setdefault("QWEN_FUSED_ROTARY", "1")
    # ...and emit Q and K/V in bfp8 straight from that op (precise bfp8 packing, same
    # error as the stock Typecast), deleting the per-layer Q Typecast and halving the
    # SDPA operand bytes. "force" = bfp8 Q at bs1 too (the bs1 shortcut used bf16 Q).
    # E2E: bs1 23.9->23.7, bs8 143.4->135.3 (-5.6%), bs16 263.5->250.4 (-5.0%),
    # bs32 495.0->474.3 (-4.2%); STS-B 0.8134->0.8190 (Q-only 0.8164).
    os.environ.setdefault("QWEN_FUSED_Q_BFP8", "force")
    os.environ.setdefault("QWEN_FUSED_KV_BFP8", "1")
    # ...and let the QKV projection itself write bfp8 (the fused op is its only reader;
    # upstream pinned bf16 only for the stock rotary). E2E: bs8 135.3->127.5 (-5.8%),
    # bs16 250.4->240.9 (-3.8%), bs32 474.3->456.7 (-3.7%), bs1 23.7 unchanged; STS-B 0.8161.
    os.environ.setdefault("QWEN_QKV_OUT_BFP8", "1")
    # The block-sharded LN path aims for an 8x8 grid, which is inherited from
    # BGE-M3 / 0.6B. On 4B that silently disables it: k_tiles = dim/32 = 80, so
    # gx=8 gives block_w=10 and block_h*block_w = 20, over the 16-tile per-core
    # cap -> the helper returns None and LN stays interleaved. gx=10 divides 80
    # and lands exactly on the cap. Measured on P150 bs=1 ISL=512 (best prefill):
    #   gx<=8  (inert, interleaved LN) 28.8 ms
    #   gx<=10 (sharded LN, 80 cores)  26.9 ms   <- default
    #   gx<=10 gy<=10                  27.0 ms
    #   gx<=8  with cap raised to 20   27.1 ms
    os.environ.setdefault("QWEN_LN_GRID_MAX_X", "10")
    # Embedding-specific: skip KV cache fill (prefill-only, no decode).
    os.environ.setdefault("TT_SKIP_KV_CACHE_FILL", "1")
    # Bigger SDPA chunks for bs=1 — more work per SDPA launch.
    os.environ.setdefault("QWEN_SDPA_BIG_CHUNK_BS1", "1")
    # SDPA chunk. The 128 that QWEN_SDPA_BIG_CHUNK_BS1 selects was tuned on 0.6B
    # (16 q heads); 4B has 32 q heads over the same head_dim, so it wants more
    # work per SDPA launch. Measured on P150 bs=1 ISL=512 (best prefill):
    #   chunk  64   29.2 ms
    #   chunk 128   26.9 ms  (previous default)
    #   chunk 256   25.7 ms  <- default
    #   chunk 512   fails: statically allocated circular buffers exceed L1
    os.environ.setdefault("QWEN_SDPA_CHUNK", "256")
    # Direct q/k chunk. QWEN_SDPA_CHUNK only reaches the short-seq branch, so it
    # cannot affect batched prefill at all: at bs>=8 the flattened seq is
    # >= 2048, which pins q_chunk=256 unconditionally. Setting q/k directly is
    # worth ~2-4% at every batch size and is neutral at bs=1 (where q=512 is
    # simply the whole 512-token sequence in one chunk). Measured on P150,
    # best prefill / best tok/s:
    #   bs1   default 25.9 ms          q512/k256 25.9 ms   (neutral)
    #   bs8   default 189.2 ms / 21.6k q512/k256 181.9 ms / 22.5k
    #   bs16  default 362.9 ms / 22.6k q512/k256 355.7 ms / 23.0k
    #   bs32  default 706.6 ms / 23.2k q512/k256 682.1 ms / 24.0k
    # q=768 and q=1024 overflow L1 ("statically allocated circular buffers");
    # k=128 ties k=256 at bs32 (681.1 vs 682.1 ms) so k=256 is kept as the safer
    # of the two.
    # MinimalMatmul output subblock. ttnn.MinimalMatmulConfig defaults subblock_h
    # and subblock_w to 1 in its Python binding and the callers in model_config
    # never set them, so every batched prefill matmul was running a 1x1 output
    # subblock -- one tile at a time in DST. That is what made those rows come
    # back SLOW at ~48% of BFP4/LoFi peak with only 16-31% DRAM utilisation:
    # neither compute nor bandwidth saturated, just per-tile overhead. 1x8 uses
    # the whole 8-tile DST budget (2x8 fails "subblock_h * subblock_w must be <=
    # max_dest_volume"). Measured on P150, best prefill / best tok/s:
    #   bs8    1x1 182.0 ms / 22.5k    1x8 160.3 ms / 25.6k   (-11.9%)
    #   bs16   1x1 355.6 ms / 23.0k    1x8 308.7 ms / 26.5k   (-13.2%)
    #   bs32   1x1 684.2 ms / 23.9k    1x8 559.0 ms / 29.3k   (-18.3%)
    # bs=1 is unaffected: it uses the legacy MatmulMultiCoreReuseMultiCast path,
    # which derives its own out_subblock_w via get_out_subblock_w.
    os.environ.setdefault("QWEN_MM_SUBBLOCK", "1,8")
    os.environ.setdefault("QWEN_SDPA_Q_CHUNK", "512")
    os.environ.setdefault("QWEN_SDPA_K_CHUNK", "256")
    if batched_l1:
        os.environ.setdefault("TT_BATCHED_L1_PREFILL", "1")
        # Lift the batched-L1 activation cap from the 8 MiB default to 12 MiB so
        # bs=4 ISL=512 (10.5 MB activation) qualifies for the L1 prefill path.
        # Measured on this P150: bs=4 goes 90.8 ms (DRAM) -> 74.8 ms (L1), a 21%
        # throughput win (22.6k -> 27.4k tok/s — the highest of any batch size).
        # The per-op memory snapshot showed the DRAM path leaving ~1.45 MB/core of
        # L1 idle (peak 8 KB of 1464 KB) while activations round-tripped through
        # DRAM. 12 MiB still excludes bs=8 (20 MB), whose 9728-wide FF intermediate
        # overflows L1 ("static CB region clashes with L1 buffer") — bs>=8 stays on
        # the DRAM + 130-core matmul grid path.
        os.environ.setdefault("TT_BATCHED_L1_PREFILL_MAX_BYTES", str(12 * 1024 * 1024))


# ---------------------------------------------------------------------------
# Per-workload optimization configs
# ---------------------------------------------------------------------------

# Maps (batch_size, seq_len) to optimized settings.  The dp32 multiprocess
# script and individual demo files both look up from here so every combination
# is tuned in exactly one place.
#
# "batched_l1"  – whether the total activation (bs * seq * 2560 * 2 B) fits in
#                 the P150 batched-L1 prefill cap (12 MiB) AND per-user seq<=512.
#                 When True we set TT_BATCHED_L1_PREFILL=1. For the 4B model the
#                 hidden dim is large enough that only small batches at ISL=512 fit
#                 (bs<=4); bs>=8 OOMs L1 (9728-wide FF intermediate) -> DRAM path.
# "dram_grid"   – when activations are DRAM-resident, widen the MinimalMatmul
#                 grid to the full 130-core (13x10) Blackhole grid via
#                 QWEN_MM_GRID=13,10 (the dominant DRAM-path speedup).

WORKLOAD_CONFIGS = {
    # bs=1: single-user L1 path when seq<=512, else DRAM + big grid.
    (1, 512): {"batched_l1": False, "dram_grid": False},  # 2.5 MB -> L1
    (1, 1024): {"batched_l1": False, "dram_grid": True},  # 5 MB -> DRAM
    (1, 2048): {"batched_l1": False, "dram_grid": True},  # 10 MB -> DRAM
    # bs=4 ISL=512: 10.5 MB activation. The batched-L1 placement clashes with the fused ops' static circular
    # buffers since the 2026-09-24 landings (program 309: L1 buffer at 1440128 inside the CB region), and the
    # DRAM path is faster with them anyway: 65.3 ms best of 10 (2026-09-24) against the 74.8 ms the L1 path
    # read before. TT_BATCHED_L1_PREFILL=1 opts back in if the CB budget changes.
    (4, 512): {"batched_l1": False, "dram_grid": True},  # 10.5 MB -> DRAM + full grid
    # DRAM-resident batched activations -> 130-core matmul grid (bs>=8: L1 OOMs).
    (8, 512): {"batched_l1": False, "dram_grid": True},  # 20 MB
    (16, 512): {"batched_l1": False, "dram_grid": True},  # 40 MB
    (8, 1024): {"batched_l1": False, "dram_grid": True},  # 40 MB
    (8, 2048): {"batched_l1": False, "dram_grid": True},  # 80 MB
    (32, 512): {"batched_l1": False, "dram_grid": True},  # 80 MB
    (32, 1024): {"batched_l1": False, "dram_grid": True},  # 160 MB
    (32, 2048): {"batched_l1": False, "dram_grid": True},  # 320 MB
}


def _model_is_bidirectional() -> bool:
    """pplx-embed is a bidirectional Qwen3-4B; Qwen3-Embedding-4B (same backbone) is causal."""
    return "pplx" in os.getenv("HF_MODEL", MODEL_NAME).lower()


def apply_workload_env(batch_size: int, seq_len: int) -> None:
    """Apply optimized env vars for a specific (batch_size, seq_len) workload.

    Looks up ``WORKLOAD_CONFIGS`` for the exact pair.  Falls back to a heuristic
    for unseen combinations: batched-L1 if the activation fits the ~8 MiB cap
    AND seq<=512, DRAM + 130-core matmul grid otherwise.
    """
    # Attention direction follows the checkpoint: causal for Qwen3-Embedding-4B, bidirectional for pplx-embed.
    if not _model_is_bidirectional():
        os.environ.setdefault("QWEN_SDPA_CAUSAL", "1")
    cfg = WORKLOAD_CONFIGS.get((batch_size, seq_len))
    if cfg is None:
        activation_bytes = batch_size * seq_len * HIDDEN_DIM * 2
        fits_l1 = activation_bytes <= 12 * 1024 * 1024 and seq_len <= 512
        if batch_size == 1:
            # Single-user L1 path handles seq<=512 automatically.
            cfg = {"batched_l1": False, "dram_grid": seq_len > 512}
        else:
            cfg = {"batched_l1": fits_l1, "dram_grid": not fits_l1}

    # Batched-prefill SDPA: all 120 workers and one K chunk. Standalone at the model's exact
    # config (LoFi, exp approx, Q/K/V bfp8, non-causal): bs32 932 -> 806 us, bs8 308 -> 262 us
    # per call. bs1 keeps 8x8 / k256: its activations are L1-resident and the k512 SDPA CBs
    # clash with them (TT_THROW), and 32 work units cannot fill 120 cores anyway. Must run
    # before apply_recommended_env, whose setdefault would pin k256. Opt out:
    # QWEN_SDPA_BATCHED_WIDE=0.
    if batch_size > 1 and os.getenv("QWEN_SDPA_BATCHED_WIDE", "1") == "1":
        # bs8: 12x8 q512/k512 beats 12x10 (256 work units take 3 waves on 96 cores as on 120, with
        # less DRAM contention): standalone 272 -> 234 us per call, e2e 122.8 -> 121.7 (chip 7).
        # bs32 is neutral e2e (435.1 -> 436.4) and keeps 12x10; bs16 sits inside its run-to-run band.
        if batch_size == 8 and os.getenv("QWEN_SDPA_BS8_12X8", "1") == "1":
            os.environ.setdefault("QWEN_SDPA_GRID", "12,8")
            os.environ.setdefault("QWEN_SDPA_Q_CHUNK", "512")
        else:
            os.environ.setdefault("QWEN_SDPA_GRID", "12,10")
        os.environ.setdefault("QWEN_SDPA_K_CHUNK", "512")
    # SDPA writes its output straight in the [B, 1, S, H*d] layout (output_heads_concat=True, a tile-id
    # remap in the SDPA writer), so the concat-heads pass (142 MB per layer at bs32) is skipped. Bit-identical
    # to concat_heads(SDPA). Cold e2e: bs8 117.7 -> 115.4, bs16 226.2 -> 221.9, bs32 433.8 -> 430.1;
    # sustained bs32 445.8 -> 440.3. bs1 keeps the model-local concat op (4 us). Opt out: QWEN_SDPA_CONCAT_OUT=0.
    if batch_size > 1:
        os.environ.setdefault("QWEN_SDPA_CONCAT_OUT", "1")
    # bs1 too (QWEN_SDPA_CONCAT_OUT_BS1=0 opts out): the q256 8x8 SDPA drains faster into [1, 1, S, H*d]
    # (standalone 57.3 -> 54.1 us) and the 4.6 us model-local concat op per layer disappears.
    elif os.getenv("QWEN_SDPA_CONCAT_OUT_BS1", "1") == "1":
        os.environ.setdefault("QWEN_SDPA_CONCAT_OUT", "1")
    # Fused residual add + RMSNorm with each row split over R cores (multi-wave row-split kernel). The
    # row-granular kernel leaves most of the 120 cores idle in the last wave at 128-512 tile-rows;
    # standalone stock add + rms_norm -> split: M=4096 184 -> 141 us (R=5), M=8192 320 -> 242 (R=5),
    # M=16384 597 -> 459 (R=4). e2e bs8 123.1 -> 119.9 (chip 7, MIN_ROWS lowered to 4096 so bs8 takes the
    # fused path at all), bs32 449.4 -> 443.2 (chip 10). Per-call PCC vs the stock ops >= 0.9998 at bs8
    # (QWEN_FUSED_ADD_NORM_VERIFY=1). Opt out: QWEN_FUSED_ADD_NORM_R=0.
    if batch_size == 8 and seq_len == 512:
        os.environ.setdefault("QWEN_FUSED_ADD_NORM_MIN_ROWS", "4096")
        os.environ.setdefault("QWEN_FUSED_ADD_NORM_R", "5")
    if batch_size == 16 and seq_len == 512:
        # standalone 320 -> 242 us per call; e2e sits inside bs16's 217/229 ms two-mode band (alternating
        # 2-pair run: 228.3 / 231.6 -> 216.8 / 216.5), never measured worse.
        os.environ.setdefault("QWEN_FUSED_ADD_NORM_R", "5")
    if batch_size == 32 and seq_len == 512:
        os.environ.setdefault("QWEN_FUSED_ADD_NORM_R", "4")
        # minimal_matmul streams interleaved bfp4 weights 2-3% faster than width-sharded ones for
        # QKV, WO and W1/W3 at M=16384 (FF2 prefers sharded): e2e 441.9 -> 436.9 (chip 6). At M=4096 the
        # sharded layout is faster, so bs8 keeps it; the bs1 legacy kernel needs it.
        for k in (
            "QWEN_WEIGHT_INTERLEAVED_K2560_N6144",
            "QWEN_WEIGHT_INTERLEAVED_K4096_N2560",
            "QWEN_WEIGHT_INTERLEAVED_K2560_N9728",
        ):
            os.environ.setdefault(k, "1")
    # bs1 SDPA: q_chunk 256 doubles the work units (32 -> 64) so the 8x8 grid is full; k stays 256.
    # Standalone at the model's config (LoFi, fp32 acc off = streaming kernel, exp approx, bfp8
    # Q/K/V in L1): q512/k256 79.1 us -> q256/k256 55.0 us (-30%); q256/k512 71.5, q128/k128 72.2,
    # 12x10 grids slower (67.6). e2e 23.3 -> 22.4 ms (-3.9%, chip 4). Opt out: QWEN_SDPA_BS1_Q256=0.
    if batch_size == 1 and os.getenv("QWEN_SDPA_BS1_Q256", "1") == "1":
        os.environ.setdefault("QWEN_SDPA_Q_CHUNK", "256")
        os.environ.setdefault("QWEN_SDPA_K_CHUNK", "256")
    # bs1 legacy 2D-multicast matmul blocks (8x8 grid, DRAM width-sharded bfp4 weights), from a
    # standalone in0_block_w x out_subblock sweep at M=512 on the model's operand placement:
    # FF1/FF3 (K=2560,N=9728) in0_bw 10->8 + subblock 1x2->2x2: 115.9->111.7 us (-3.7%);
    # FF2 (K=9728,N=2560) subblock 1x2->1x5: 113.1->111.0 (-1.9%); WO (K=4096,N=2560) 1x2->1x5:
    # 52.3->51.3 (-2.0%); QKV (K=2560,N=6144) in0_bw 10->8 + 1x4->1x6: 71.0->70.1 (-1.2%).
    # e2e 22.4 -> 21.4 ms (-4.5%, chip 4, same-chip A/B on top of the q256 SDPA). Keyed by (K, N)
    # so they only reach these projections; the guard in _legacy_block_overrides skips shapes
    # they do not divide (shorter warm-up seq_lens). Opt out: QWEN_LEGACY_BS1_BLOCKS=0.
    # bs1 legacy 2D-multicast matmuls on 12x8 = 96 cores instead of 8x8 = 64. The DRAM width-sharded
    # bfp4 weights live in 8 banks; the 2D factory's per-column bank walk used to hand a column a
    # whole bank stripe, so any grid wider than 8 columns computed garbage.
    # With the capped walk every wide grid is bit-identical to 8x8. Standalone, traced, M=512:
    # QKV 67.5 -> 49.8 us, WO 48.7 -> 40.1, FF1/FF3 110.9 -> 79.5 each, FF2 101.8 -> 78.8
    # (about -112 us/layer). per_core_N 7 needs a 2x1 subblock (the derived 1x1 is slower than 8x8).
    # Opt out with QWEN_LEGACY_BS1_WIDE=0, which falls back to the tuned 8x8 blocks below.
    wide = batch_size == 1 and os.getenv("QWEN_LEGACY_BS1_WIDE", "1") == "1"
    if wide:
        for k, v in (
            ("QWEN_QKV_GRID_X", "12"),
            ("QWEN_LEGACY_GRID_FF13", "12,8"),
            ("QWEN_LEGACY_GRID_FF2", "12,8"),
            ("QWEN_LEGACY_GRID_WO", "12,8"),
            ("QWEN_LEGACY_TIGHT_PER_CORE_N", "1"),
            ("QWEN_LEGACY_SUBBLOCK_K2560_N6144", "1,4"),
            ("QWEN_LEGACY_SUBBLOCK_K2560_N9728", "2,2"),
            ("QWEN_LEGACY_SUBBLOCK_K9728_N2560", "2,1"),
            ("QWEN_LEGACY_SUBBLOCK_K4096_N2560", "2,1"),
        ):
            os.environ.setdefault(k, v)
    if batch_size == 1 and not wide and os.getenv("QWEN_LEGACY_BS1_BLOCKS", "1") == "1":
        for k, v in (
            ("QWEN_LEGACY_IN0_BW_K2560_N9728", "8"),
            ("QWEN_LEGACY_SUBBLOCK_K2560_N9728", "2,2"),
            ("QWEN_LEGACY_SUBBLOCK_K9728_N2560", "1,5"),
            ("QWEN_LEGACY_SUBBLOCK_K4096_N2560", "1,5"),
            ("QWEN_LEGACY_IN0_BW_K2560_N6144", "8"),
            ("QWEN_LEGACY_SUBBLOCK_K2560_N6144", "1,6"),
        ):
            os.environ.setdefault(k, v)
    # Fused-SwiGLU minimal_matmul blocks at bs16 (M=8192): 4,8,8 / 1x4 was -6% standalone and
    # -2.2% e2e vs the 8,8,8 / 1x8 default (237.0 -> 231.8, chip 8); a wider sweep then found
    # K_block 20 (4 K steps over the 80 K tiles) another -3.8% standalone (2988 -> 2874 us) and
    # -1.9% e2e (232.8 -> 228.3, chip 8). bs8's default is already the best of the sweep; bs32
    # runs the unfused path, where larger K steps are slower for every projection.
    if batch_size == 16 and seq_len == 512:
        os.environ.setdefault("QWEN_MM_BLOCK_FF13", "4,20,8")
        os.environ.setdefault("QWEN_MM_SUBBLOCK_FF13", "1,4")
    # Plain minimal_matmul blocks at bs8 (M=4096), same sweep: FF2 16,8,8 (-16% standalone),
    # QKV 8,4,8 (-15%), WO 16,8,8 (-12%); e2e 126.4 -> 123.4 ms (-2.4%, chip 7). At bs16/bs32
    # the 8,8,8 defaults are the best of the sweep for these projections.
    if batch_size == 8 and seq_len == 512:
        os.environ.setdefault("QWEN_MM_BLOCK_FF2", "16,8,8")
        os.environ.setdefault("QWEN_MM_BLOCK_QKV", "8,4,8")
        os.environ.setdefault("QWEN_MM_BLOCK_WO", "16,8,8")
    apply_recommended_env(batched_l1=cfg["batched_l1"])
    # Fused SwiGLU (tt/mlp.py PplxFusedSwigluMLP) folds FF1 + FF3 + the silu*mul
    # BinaryNg into one minimal_matmul(fuse_swiglu=True). It is a win at moderate
    # batch and a loss at bs=32, where doubling the packed weight width to
    # 2*hidden_dim=19456 costs more in block shape than the removed BinaryNg
    # saves. Measured on P150 ISL=512 (best prefill / best tok/s):
    #   bs8   off 160.8 ms / 25.5k   on 158.3 ms / 25.9k   -1.6%
    #   bs16  off 308.4 ms / 26.6k   on 290.9 ms / 28.2k   -5.7%
    #   bs32  off 558.4 ms / 29.3k   on 570.5 ms / 28.7k   +2.2%  <- regression
    # bs=1 never reaches it (legacy MatmulMultiCoreReuseMultiCast path).
    # STS-B Spearman identical either way (0.8125).
    os.environ.setdefault("QWEN_FUSE_SWIGLU", "1" if cfg.get("fuse_swiglu", 1 < batch_size <= 16) else "0")
    if cfg["dram_grid"]:
        os.environ.setdefault("QWEN_MM_GRID", DRAM_MM_GRID)


def pplx_optimizations(model_args):
    """Aggressive performance precision for pplx-embed embedding workloads.

    Starts from ``DecodersPrecision.performance`` (FF1/FF3 already LoFi) and:
      - forces SDPA LoFi (safe for embedding — cosine similarity is robust to
        low-precision attention),
      - opt-in promotes QKV / WO / FF2 weights to BFP4 (gated by the
        QWEN_*_BFP4 env vars set in apply_recommended_env).
    """
    base = DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    # NOTE: FF1/FF3 weights are already BFP4 here — DecodersPrecision.performance
    # sets TensorGroup.FF1_FF3 to BFP4 before this function runs, so there is no
    # QWEN_FF13_BFP4 knob to add. Verified by reading the resolved policy:
    # FF1_FF3=bfp4 with LI_FF1_FF3 fidelity=lofi. All five prefill matmul roles
    # (QKV, WO, FF1, FF3, FF2) therefore run BFP4/LoFi, which puts the whole
    # matmul roofline at 3.72 TFLOP / 580.9 TFLOPS = 6.40 ms for bs=1 ISL=512.
    promote_ff2 = os.getenv("QWEN_FF2_BFP4", "0") == "1"
    promote_qkv = os.getenv("QWEN_QKV_BFP4", "0") == "1"
    promote_wo = os.getenv("QWEN_WO_BFP4", "0") == "1"

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
        # SDPA LoFi — safe for embedding (cosine similarity insensitive).
        of[OpGroup.SDPA_PREFILL] = MathFidelitySetting.LOFI
        of[OpGroup.SDPA_DECODE] = MathFidelitySetting.LOFI
        # MLP LoFi across the board (FF1/FF3 already LoFi from performance base).
        of[OpGroup.LI_FF1_FF3] = MathFidelitySetting.LOFI
    base._update_full_name()
    return base


# ---------------------------------------------------------------------------
# Model build / inputs
# ---------------------------------------------------------------------------


def _page_params_for(batch_size: int, seq_len: int) -> dict:
    """Page-table sizing for the 4B model.

    KV cache per token per layer: 2 * n_kv_heads * head_dim * 2 bytes
      = 2 * 8 * 128 * 2 = 4 KB/token/layer  (x36 layers = 144 KB/token total).
    Page sizing is in block units (not bytes), so the 0.6B formula carries over.
    """
    if batch_size == 1:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 512}
    if batch_size == 8:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 1024}
    page_max = max(512, math.ceil(seq_len / BLOCK_SIZE) * batch_size * 2)
    return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}


def build_single_device_model(mesh_device, batch_size: int, seq_len: int):
    """Build one pplx-embed-4B model instance + Generator + page table."""
    page_params = _page_params_for(batch_size, seq_len)
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    padded_seq_len = get_padded_prefill_len(seq_len)

    model_args = PplxModelArgs(
        mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=pplx_optimizations,
        max_seq_len=padded_seq_len,
        prefetcher=None,
    )

    state_dict = model_args.load_state_dict()

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        attention_class=PplxBidirectionalAttention,
        mlp_class=PplxFusedSwigluMLP,
    )
    if os.getenv("QWEN_FUSED_ADD_NORM", "1") == "1":
        # Residual add + RMSNorm pairs -> one generic_op each (bs16+ prefill; see tt/decoder_fusion.py).
        # E2E same-chip A/B: bs16 239.6->234.8 (-2.0%), bs32 450.6->443.5 (-1.6%); bs8 regressed
        # (+1.7%) and bs1 is slower standalone, both keep the stock ops via the row threshold.
        from models.demos.blackhole.pplx_embed_4b.tt.decoder_fusion import install_decoder_fusion

        install_decoder_fusion(model)

    kv_caches = [[layer.attention.layer_past for layer in model.layers]]
    generator = Generator(
        [model],
        [model_args],
        mesh_device,
        tokenizer=model_args.tokenizer,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)

    return generator, model_args, kv_caches, page_table


def generate_synthetic_inputs(tokenizer, batch_size: int, seq_len: int):
    """Random tokens of exactly ``seq_len``."""
    vocab_size = tokenizer.vocab_size
    high = min(vocab_size, 50000)
    input_ids = torch.randint(100, high, (batch_size, seq_len), dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, prompt_lens


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_perf(
    mesh_device,
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    *,
    emit_signposts: bool,
    is_ci_env: bool = False,
    full_pipeline: bool = True,
):
    """Build, compile, and benchmark pplx-embed-v1-4B."""
    profiler = BenchmarkProfiler()
    profiler.start("run")
    tt_device_name = determine_device_name(mesh_device)

    logger.info(f"Building pplx-embed-v1-4B: bs={batch_size}, seq_len={seq_len}, device={tt_device_name}")

    profiler.start("build_model")
    generator, model_args, kv_caches, page_table = build_single_device_model(
        mesh_device, batch_size=batch_size, seq_len=seq_len
    )
    profiler.end("build_model")
    logger.info(f"Built in {profiler.get_duration('build_model'):.1f}s")

    input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, batch_size, seq_len)
    total_input_tokens = sum(prompt_lens)

    logger.info("Compiling (first prefill captures hardware trace + runs warmup)...")
    profiler.start("compile_prefill")
    _ = generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_caches,
        prompt_lens=prompt_lens,
        enable_trace=True,
        return_hidden_states=True,
        warmup_prefill=True,
    )
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    # Locate the captured trace for direct replay — bypasses the Generator's
    # per-iteration Python overhead (page table reset, prefill_forward_text
    # loop, process_hidden_states_after_prefill_trace, D2H copy). This gives
    # us pure device-execution + sync latency.
    # The Generator keys prefill traces as f"{seq_len}_{model_id}_{batch_size}_{use_start_pos}";
    # this lookup used the older 3-part key and never matched, so use_direct_trace was
    # always False and every timed iteration went through the Generator path: eager
    # post-processing ops (slice + norm + to_layout) dispatched outside any trace, four
    # H2D copies and a blocking readback -- ~3.4 ms/iter at bs=1 (13%), ~3.8 ms at bs=32.
    # Match on the 3-part prefix so the suffix format cannot silently break it again.
    trace_key = f"{seq_len}_0_{batch_size}"
    _prefix = trace_key + "_"
    trace_id = next(
        (v for k, v in generator.trace_id_prefill.items() if k == f"{seq_len}_0_{batch_size}" or k.startswith(_prefix)),
        None,
    )
    use_direct_trace = (trace_id is not None) and not full_pipeline

    # --- Optimized full-pipeline path ---
    # Capture an *extended* trace that includes post-processing ops (slice +
    # norm + to_layout) so they execute as part of trace replay rather than as
    # individually dispatched ops.  Pre-compute host inputs once to skip
    # Generator Python overhead in the hot loop.
    ext_trace_id = None
    ext_trace_output = None
    ext_device_inputs = None
    ext_host_inputs = None
    if full_pipeline and trace_id is not None:
        model = generator.model[0]
        last_token_idx = seq_len - 1
        is_batched = batch_size > 1
        padded_batch = model_args.max_batch_size

        # Release all Generator traces — we'll capture a new extended trace that
        # includes post-processing ops.  Releasing all avoids "unsafe allocation"
        # warnings from warmup traces at shorter seq_lens.
        for key, tid in list(generator.trace_id_prefill.items()):
            if tid is not None:
                ttnn.release_trace(mesh_device, tid)
                generator.trace_id_prefill[key] = None

        if is_batched:
            prefill_ids = torch.zeros(padded_batch, seq_len, dtype=torch.long)
            padded_last_token_idx = [0] * padded_batch
            for slot in range(batch_size):
                prefill_ids[slot] = input_ids[slot]
                padded_last_token_idx[slot] = last_token_idx
            get_last_token = (last_token_idx // 32) * 32
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": 0}
        else:
            prefill_ids = input_ids
            get_last_token = (last_token_idx // 32) * 32
            prefill_kwargs = {"page_table": page_table[0:1]}

        host_inputs_full = model.prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        rot_mats_global = host_inputs_full[1]
        rot_mats_local = host_inputs_full[2]
        ext_host_inputs = (host_inputs_full[0], host_inputs_full[3], host_inputs_full[4])

        fwd_kwargs = dict(
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            kv_cache=kv_caches[0],
        )
        if is_batched:
            fwd_kwargs["batch_size"] = batch_size
            fwd_kwargs["user_id"] = 0

        def _forward_and_postprocess(dinputs):
            transformed = model.transform_and_embed_prefill_inputs_device(*dinputs, tt_chunk_start_idx=None)
            tt_out = model.ttnn_prefill_forward(
                x=transformed[0],
                page_table=transformed[1],
                chunk_page_table=transformed[2],
                **fwd_kwargs,
            )
            if is_batched:
                return model.process_hidden_states_after_prefill_trace_batched(tt_out, get_last_token)
            return model.process_hidden_states_after_prefill_trace(tt_out, last_token_idx)

        # Warm-run to ensure all post-processing kernels are compiled.
        device_inputs = copy_host_to_device(ext_host_inputs, mesh_device=mesh_device)
        _ = _forward_and_postprocess(device_inputs)
        ttnn.synchronize_device(mesh_device)

        # Capture extended trace: forward + post-processing in one replay unit.
        device_inputs = copy_host_to_device(ext_host_inputs, mesh_device=mesh_device)
        ext_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        ext_trace_output = _forward_and_postprocess(device_inputs)
        ttnn.end_trace_capture(mesh_device, ext_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ext_device_inputs = device_inputs
        logger.info("Captured extended trace (forward + post-processing)")

    if use_direct_trace:
        logger.info(f"Running {num_iterations} iterations via direct trace replay (key={trace_key})...")
    elif ext_trace_id is not None:
        logger.info(f"Running {num_iterations} iterations via extended trace (forward + post-proc)...")
    else:
        logger.info(f"Running {num_iterations} iterations via generator (no direct trace)...")

    iteration_times = []
    last_iter_idx = num_iterations - 1
    for i in range(num_iterations):
        sig = emit_signposts and i == last_iter_idx

        profiler.start(f"inference_prefill_{i}")
        if sig:
            _tracy_signpost("start")
        try:
            if use_direct_trace:
                _t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                _t1 = time.perf_counter()
                ttnn.synchronize_device(mesh_device)
                if os.getenv("QWEN_ITER_TIMING", "0") == "1":
                    logger.info(
                        f"  iter-timing trace_issue={(_t1-_t0)*1000:.2f} sync={(time.perf_counter()-_t1)*1000:.2f} ms"
                    )
            elif ext_trace_id is not None:
                # QWEN_ITER_TIMING=1 logs where the per-iteration host time goes. The
                # device profile shows ~3.4 ms/iter at bs=1 of device idle inside the
                # first ops of each replay (waiting on the host), for inputs of a few KB.
                _tt = os.getenv("QWEN_ITER_TIMING", "0") == "1"
                _t = [time.perf_counter()]
                copy_host_to_device(ext_host_inputs, device_tensors=ext_device_inputs, mesh_device=mesh_device)
                _t.append(time.perf_counter())
                ttnn.execute_trace(mesh_device, ext_trace_id, cq_id=0, blocking=False)
                _t.append(time.perf_counter())
                hidden_host = ext_trace_output.cpu(blocking=False)
                _t.append(time.perf_counter())
                ttnn.synchronize_device(mesh_device)
                _t.append(time.perf_counter())
                _ = ttnn.to_torch(ttnn.get_device_tensors(hidden_host)[0])
                _t.append(time.perf_counter())
                if _tt:
                    d = [(_t[j + 1] - _t[j]) * 1000 for j in range(len(_t) - 1)]
                    logger.info(
                        f"  iter-timing h2d={d[0]:.2f} trace_issue={d[1]:.2f} readback_enq={d[2]:.2f} "
                        f"sync={d[3]:.2f} to_torch={d[4]:.2f} ms"
                    )
            else:
                generator.prev_page_table = None
                generator.prefill_forward_text(
                    input_ids,
                    page_table=page_table,
                    kv_cache=kv_caches,
                    prompt_lens=prompt_lens,
                    enable_trace=True,
                    return_hidden_states=True,
                    warmup_prefill=False,
                )
                ttnn.synchronize_device(mesh_device)
        finally:
            if sig:
                _tracy_signpost("stop")
        profiler.end(f"inference_prefill_{i}")

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

    avg_t = sum(iteration_times) / len(iteration_times)
    best_t = min(iteration_times)
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_t,
        "best_prefill_time": best_t,
        "embeddings/s_avg": batch_size / avg_t,
        "embeddings/s_best": batch_size / best_t,
        "prefill_t/s_avg": total_input_tokens / avg_t,
        "prefill_t/s_best": total_input_tokens / best_t,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "input_seq_len": seq_len,
        "total_input_tokens": total_input_tokens,
    }

    if ext_trace_id is not None:
        mode_label = "full pipeline (extended trace: forward + pooling + I/O)"
    elif use_direct_trace:
        mode_label = "direct trace (device-only forward replay)"
    else:
        mode_label = "generator fallback (eager post-processing, no trace reuse)"
    time_label = "full pipeline time" if full_pipeline else "prefill time"

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {os.getenv('HF_MODEL', MODEL_NAME).split('/')[-1]} Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Input seq length:     {seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info(f"  Mode:                 {mode_label}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg {time_label}:     {avg_t * 1000:.1f}ms")
    logger.info(f"  Best {time_label}:    {best_t * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {measurements['embeddings/s_avg']:.1f}")
    logger.info(f"  Best embeddings/s:    {measurements['embeddings/s_best']:.1f}")
    logger.info(f"  Avg tokens/s:         {measurements['prefill_t/s_avg']:.0f}")
    logger.info(f"  Best tokens/s:        {measurements['prefill_t/s_best']:.0f}")
    logger.info("=" * 60)

    profiler.end("run")

    if is_ci_env:
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name="pplx-embed-v1-4B",
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": 1, "tensor_parallel": 1},
            input_sequence_length=seq_len,
            output_sequence_length=0,
        )

    return measurements


# ---------------------------------------------------------------------------
# Standalone (no-pytest) entry point
# ---------------------------------------------------------------------------


def standalone_main(
    batch_size: int, seq_len: int, iterations: int, device_id: int = 0, full_pipeline: bool = True
) -> None:
    """`python <entry_file>` path — opens its own device, no pytest fixture."""
    apply_workload_env(batch_size, seq_len)

    logger.info(f"Opening device {device_id}...")
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=200_000_000,
        num_command_queues=1,
    )
    try:
        t0 = time.perf_counter()
        run_perf(
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            num_iterations=iterations,
            emit_signposts=False,
            full_pipeline=full_pipeline,
        )
        logger.info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
    finally:
        ttnn.close_device(device)
