# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
"""Device-performance test for the HunyuanImage-3.0 recaption DECODE step (tracy).

The AR (recaption) stage is one KV-cache prefill of the prompt prefix, then
`max_new_tokens` single-token decode steps. This test measures ONE decode step in
isolation — the small-M (M == 1 token, padded to 32; 64 with a 2-row batch),
memory-bound single-token forward that appends to the per-layer K/V cache
(`decode_step=True`). The one-time prefill cost is measured separately by
test_recaption_prefill_perf.py.

Decode is the term multiplied by max_new_tokens in the stage estimate, so per-step
regressions dominate end-to-end cost:

    stage latency ≈ prefill_cost  +  per_token_decode_cost × max_new_tokens
                    (prefill perf test)  ^^^^^^^^^^^^^^^^^^^^  (this file)

Small-M decode matmuls are memory-bound (bf8 weight reads dominate), so they use the
1D decode_mm_program_config, not the 2D wide (prefill) config — this test guards that
path against regression.

Runs on the 2x2 mesh in the production recaption config (demo_i2i._build_backbone
with recap_sp=1): expert-parallel MoE, TP=2, sp_factor=1 (KV cache forbids sp>1).

Two tests, the standard tt-metal device-perf split:
  * test_recaption_decode_device_ops — prefill once (warms + populates the KV cache),
    warm one decode step (program-cache warm), then a SECOND decode step bracketed by
    tracy signpost("start")/signpost("stop") so only ONE decode step is measured.
  * test_recaption_decode_perf_device — the gate. Re-invokes the workload under the
    device profiler, sums DEVICE KERNEL duration between the signposts, reports it.

Env knobs: HY_NUM_LAYERS (backbone layers), HY_PREFILL_ISL (prefix length before the
decode step, default 512).

Run the raw op profile (workload only, writes ops_perf_results_*.csv):
    HY_NUM_LAYERS=2 HY_PREFILL_ISL=512 python_env/bin/python -m tracy -r -p -v -m \
      "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_decode_perf.py::test_recaption_decode_device_ops -s"

Run the device-perf gate (spawns the workload under the profiler for you):
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_decode_perf.py::test_recaption_decode_perf_device
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_dit.parallel.manager import CCLManager

from models.experimental.hunyuan_image_3_0.ref.attention.mask import (
    build_attention_mask,
    build_attention_mask_query_row,
    to_additive,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.kv_cache import HunyuanTtKvCache
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

PROMPT = "a cat on a mat"
BOT_TASK = "recaption"
SEQUENCE_TEMPLATE = "instruct"
# Prefix length the decode step attends over (the KV cache is populated to this ISL).
PREFILL_ISL = int(os.environ.get("HY_PREFILL_ISL", "512"))

use_signpost = True
try:
    from tracy import signpost
except ModuleNotFoundError:
    use_signpost = False


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _full_num_layers() -> int:
    return int(json.load(open(h.I2I_WEIGHTS / "config.json"))["num_hidden_layers"])


def _build_workload(mesh_device):
    """Build the production 2x2 recaption backbone (sp_factor=1, TP=2, expert-parallel)
    and return a (decode_step closure, prefill closure) pair. Prefill must run once to
    populate the KV cache before the decode step is measured."""
    num_layers = int(os.environ.get("HY_NUM_LAYERS", str(_full_num_layers())))
    c = h.model_cfg()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tok = HunyuanTokenizer.from_model_dir(h.I2I_WEIGHTS, sequence_template=SEQUENCE_TEMPLATE)
    bundle = prepare_recaption_inputs(tok, PROMPT, bot_task=BOT_TASK, sequence_template=SEQUENCE_TEMPLATE)

    wte = h.load_tensor("model.wte.weight")
    ln_f_w = h.load_tensor("model.ln_f.weight")

    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    backbone = HunyuanTtModel(
        mesh_device,
        num_layers=num_layers,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV_HEADS"],
        head_dim=c["HEAD_DIM"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        use_qk_norm=c["USE_QK_NORM"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=False,
        layer_loader=layer_loader,
        embed_state_dict={"model.wte.weight": wte},
        norm_state_dict={"model.ln_f.weight": ln_f_w},
        apply_final_norm=True,
        weight_dtype=ttnn.bfloat8_b,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=1,  # KV cache requires sp_factor=1
        bf16_layers=[],
        model_cache_name="hunyuan-image-3.0",
    )
    lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": h.load_tensor("lm_head.weight")})

    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]
    wte_host = wte.detach().float()

    prefix_len = int(bundle.input_ids.shape[1])
    seq_len = max(PREFILL_ISL, prefix_len)
    pad = bundle.input_ids[:, -1:].repeat(1, seq_len - prefix_len)
    ids = torch.cat([bundle.input_ids, pad], dim=1) if seq_len > prefix_len else bundle.input_ids
    next_id = bundle.input_ids[:, -1:]
    # RoPE table sized with headroom past the prefill: the warm + measured decode steps
    # each append a token, so positions reach seq_len + a few. Cap at the HF ceiling.
    DECODE_HEADROOM = 8
    max_pos = min(
        seq_len + DECODE_HEADROOM, int(json.load(open(h.I2I_WEIGHTS / "config.json"))["max_position_embeddings"])
    )

    replicate_to_mesh = ttnn.ReplicateTensorToMesh(mesh_device)
    rope = backbone.layers[0].self_attn.rope
    cos_full, sin_full = rope.prepare_cos_sin(max_pos, image_infos=image_infos)
    kv_cache = HunyuanTtKvCache(len(backbone.layers))

    def _upload(host, mm=None):
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm or replicate_to_mesh,
        )

    def prefill_once():
        """Populate the KV cache with a prefill at ISL seq_len (not measured)."""
        emb = torch.nn.functional.embedding(ids.long(), wte_host)
        hidden_tt = _upload(emb)
        cos_tt = ttnn.slice(cos_full, [0, 0, 0, 0], [1, 1, seq_len, cos_full.shape[-1]])
        sin_tt = ttnn.slice(sin_full, [0, 0, 0, 0], [1, 1, seq_len, sin_full.shape[-1]])
        mask_add = to_additive(build_attention_mask(seq_len, attn_slices, bsz=1), dtype=torch.bfloat16).reshape(
            1, 1, seq_len, seq_len
        )
        mask_tt = _upload(mask_add)
        hidden = backbone.forward(
            inputs_embeds=hidden_tt,
            seq_len=seq_len,
            image_infos=image_infos,
            attention_mask=mask_tt,
            kv_cache=kv_cache,
            use_cache=True,
            decode_step=False,
            cos_sin=(cos_tt, sin_tt),
        )
        kv_cache.seq_len = seq_len
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(mask_tt)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        ttnn.deallocate(hidden)

    def decode_once():
        """One single-token decode step appending to the KV cache; returns hidden.

        Each call appends a token, so the cache (and K/query positions) advance — the
        mask/rope are built at the LIVE cache length so K and mask stay in sync. The op
        shapes are identical step to step (M == 1 token, K grows by 1), so a warm step
        followed by a measured step gives a representative single-token decode cost."""
        query_pos = kv_cache.seq_len  # live position (advances each call)
        total = query_pos + 1
        emb = torch.nn.functional.embedding(next_id.long(), wte_host)
        hidden_tt = _upload(emb)
        cos_tt, sin_tt = rope.slice_cos_sin(cos_full, sin_full, query_pos)
        mask_add = to_additive(
            build_attention_mask_query_row(total, query_pos, attn_slices, bsz=1), dtype=torch.bfloat16
        ).reshape(1, 1, 1, total)
        mask_tt = _upload(mask_add)
        hidden = backbone.forward(
            inputs_embeds=hidden_tt,
            seq_len=total,
            image_infos=image_infos,
            attention_mask=mask_tt,
            kv_cache=kv_cache,
            use_cache=True,
            decode_step=True,
            cos_sin=(cos_tt, sin_tt),
        )
        kv_cache.seq_len = total
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(mask_tt)
        return hidden

    logger.info(
        f"recaption decode workload: num_layers={num_layers} prefix_len={prefix_len} "
        f"prefill_ISL={seq_len} decode_pos={seq_len}"
    )
    return prefill_once, decode_once, seq_len


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_recaption_decode_device_ops(mesh_device):
    """One warm KV-cache decode step, bracketed by signposts for the perf gate."""
    mesh_device.enable_program_cache()

    prefill_once, decode_once, seq_len = _build_workload(mesh_device)

    # Populate the KV cache (prefill) — required before any decode step.
    prefill_once()

    # Warm the decode program cache (not measured).
    hidden = decode_once()
    ttnn.deallocate(hidden)

    # Measured: a second, warm decode step — only its device ops fall between signposts.
    if use_signpost:
        signpost(header="start")
    hidden = decode_once()
    if use_signpost:
        signpost(header="stop")

    logger.info(f"recaption decode hidden shape={tuple(hidden.shape)} (prefill_ISL={seq_len})")
    ttnn.deallocate(hidden)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    # Summed DEVICE KERNEL DURATION [ns] for ONE decode step at the profiled prefix ISL.
    # PLACEHOLDER — re-baseline on first green run (assert_on_fail=False reports only).
    # Per-token cost: multiply by max_new_tokens and add prefill for the stage estimate.
    "expected_device_kernel_duration_ns",
    [1_000_000_000],
)
def test_recaption_decode_perf_device(expected_device_kernel_duration_ns):
    batch_size = 1
    subdir = "hunyuan_recaption_decode"
    num_iterations = 3
    margin = 0.05
    cols = ["DEVICE KERNEL"]

    command = "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_decode_perf.py::test_recaption_decode_device_ops"

    duration_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command,
        subdir=subdir,
        num_iterations=num_iterations,
        cols=cols,
        batch_size=batch_size,
        has_signposts=True,
        op_support_count=20000,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=margin,
        expected_perf_cols={duration_key: expected_device_kernel_duration_ns},
        assert_on_fail=False,
    )
    prep_device_perf_report(
        model_name="hunyuan_recaption_decode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"decode_step_ISL{PREFILL_ISL}",
    )
