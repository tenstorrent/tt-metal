# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
"""Device-performance test for the HunyuanImage-3.0 recaption PREFILL stage (tracy).

The AR (recaption) stage is one KV-cache prefill of the whole prompt prefix, then
`max_new_tokens` single-token decode steps. This test measures the ONE-TIME prefill
cost in isolation — the large-M, compute-bound full-sequence forward that populates
the per-layer K/V cache. The per-token decode cost is measured separately by
test_recaption_ar_perf.py.

This is a CLEAN prefill number: unlike test_recaption_ar_perf.py (which runs a
full-sequence, no-KV forward but reports it as a per-token cost multiplied by
max_new_tokens), this runs the actual KV-cache prefill once and reports it as a
one-time latency. In the end-to-end stage model:

    stage latency ≈ prefill_cost  +  per_token_decode_cost × max_new_tokens
                    ^^^^^^^^^^^^^     (this file)             (AR perf test)

Runs on the 2x2 mesh in the PRODUCTION recaption config (demo_i2i._build_backbone):
expert-parallel MoE, TP=2 (tp_axis=1), and sp_factor=1. The KV-cache path forbids
sequence parallel (sp>1) because each device must hold the whole K/V sequence, so
the demo sets recap_sp = 1 when RECAPTION_KV is on — this test matches that (all 4
devices still used, via TP + expert parallel instead of SP).

Two tests, the standard tt-metal device-perf split (same shape as
test_recaption_ar_perf.py / test_vae_decode_perf.py):

  * test_recaption_prefill_device_ops — the raw workload. Builds the backbone once,
    warms the program cache with one prefill at the profiled ISL, then runs a SECOND
    (warm) prefill bracketed by tracy signpost("start") / signpost("stop") so only
    ONE prefill's device ops are measured.
  * test_recaption_prefill_perf_device — the gate. Re-invokes the workload under the
    device profiler, sums DEVICE KERNEL duration between the signposts, reports it.

Env knobs: HY_NUM_LAYERS (backbone layers), HY_PREFILL_ISL (profiled ISL, default 512).

Run the raw op profile (workload only, writes ops_perf_results_*.csv):
    HY_NUM_LAYERS=2 HY_PREFILL_ISL=512 python_env/bin/python -m tracy -r -p -v -m \
      "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_prefill_perf.py::test_recaption_prefill_device_ops -s"

Run the device-perf gate (spawns the workload under the profiler for you):
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_prefill_perf.py::test_recaption_prefill_perf_device
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

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.kv_cache import HunyuanTtKvCache
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

PROMPT = "a cat on a mat"
BOT_TASK = "recaption"
SEQUENCE_TEMPLATE = "instruct"
# Profiled prefill ISL. Text-only recaption uses a pure causal mask, so any ISL >= the
# real prompt prefix is valid (the prefix is padded with a repeated token; values don't
# affect timing). Default to a practical CI length.
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
    + LM head + a padded ids tensor, and return a (prefill closure, ids, seq_len)."""
    num_layers = int(os.environ.get("HY_NUM_LAYERS", str(_full_num_layers())))
    c = h.model_cfg()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tok = HunyuanTokenizer.from_model_dir(h.I2I_WEIGHTS, sequence_template=SEQUENCE_TEMPLATE)
    bundle = prepare_recaption_inputs(tok, PROMPT, bot_task=BOT_TASK, sequence_template=SEQUENCE_TEMPLATE)

    wte = h.load_tensor("model.wte.weight")
    ln_f_w = h.load_tensor("model.ln_f.weight")

    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    # Matches demo_i2i._build_backbone with recap_sp=1 (the RECAPTION_KV path): bf8
    # experts, expert-parallel MoE, TP=2 on axis 1, and sp_factor=1 (KV cache forbids
    # sequence parallel — each device must hold the whole K/V sequence).
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
        sp_factor=1,  # KV-cache prefill requires sp_factor=1 (demo recap_sp when KV on)
        bf16_layers=[],
        model_cache_name="hunyuan-image-3.0",
    )
    lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": h.load_tensor("lm_head.weight")})

    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]
    wte_host = wte.detach().float()

    # Build ids at the profiled ISL: real prompt prefix padded with a repeated token.
    prefix_len = int(bundle.input_ids.shape[1])
    seq_len = max(PREFILL_ISL, prefix_len)
    pad = bundle.input_ids[:, -1:].repeat(1, seq_len - prefix_len)
    ids = torch.cat([bundle.input_ids, pad], dim=1) if seq_len > prefix_len else bundle.input_ids

    replicate_to_mesh = ttnn.ReplicateTensorToMesh(mesh_device)

    def prefill_once():
        """One KV-cache prefill at ISL seq_len; returns the backbone hidden [B, S, H]."""
        emb = torch.nn.functional.embedding(ids.long(), wte_host)
        hidden_tt = ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_to_mesh,
        )
        mask_add = to_additive(build_attention_mask(seq_len, attn_slices, bsz=1), dtype=torch.bfloat16).reshape(
            1, 1, seq_len, seq_len
        )
        mask_tt = ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_to_mesh,
        )
        kv_cache = HunyuanTtKvCache(len(backbone.layers))
        hidden = backbone.forward(
            inputs_embeds=hidden_tt,
            seq_len=seq_len,
            image_infos=image_infos,
            attention_mask=mask_tt,
            kv_cache=kv_cache,
            use_cache=True,
            decode_step=False,
        )
        kv_cache.seq_len = seq_len
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(mask_tt)
        kv_cache.clear()
        return hidden

    logger.info(
        f"recaption prefill workload: num_layers={num_layers} prefix_len={prefix_len} " f"profiled_ISL={seq_len}"
    )
    return prefill_once, ids, seq_len


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)  # full-layer bf8 backbone build + 2 prefills; overrides the 300s default
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_recaption_prefill_device_ops(mesh_device):
    """One warm KV-cache prefill, bracketed by signposts for the perf gate."""
    mesh_device.enable_program_cache()

    prefill_once, ids, seq_len = _build_workload(mesh_device)

    # Warm: populate the program cache with one prefill (not measured).
    hidden = prefill_once()
    ttnn.deallocate(hidden)

    # Measured: a second, warm prefill — only its device ops fall between the signposts.
    if use_signpost:
        signpost(header="start")
    hidden = prefill_once()
    if use_signpost:
        signpost(header="stop")

    logger.info(f"recaption prefill hidden shape={tuple(hidden.shape)} at seq_len={seq_len}")
    ttnn.deallocate(hidden)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    # Summed DEVICE KERNEL DURATION [ns] for ONE prefill at the profiled ISL.
    # PLACEHOLDER — re-baseline on first green run (assert_on_fail=False reports only).
    # One-time cost: add to (per-token AR cost × max_new_tokens) for the stage estimate.
    "expected_device_kernel_duration_ns",
    [1_000_000_000],
)
def test_recaption_prefill_perf_device(expected_device_kernel_duration_ns):
    batch_size = 1
    subdir = "hunyuan_recaption_prefill"
    num_iterations = 3
    margin = 0.05
    cols = ["DEVICE KERNEL"]

    command = "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_prefill_perf.py::test_recaption_prefill_device_ops"

    duration_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command,
        subdir=subdir,
        num_iterations=num_iterations,
        cols=cols,
        batch_size=batch_size,
        has_signposts=True,  # measure only between signpost("start") and signpost("stop")
        # The full 32-layer MoE backbone emits far more ops than the default marker
        # buffer (1333) holds; undersizing drops markers and crashes the report step.
        op_support_count=20000,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=margin,
        expected_perf_cols={duration_key: expected_device_kernel_duration_ns},
        assert_on_fail=False,  # placeholder baseline — report only until re-baselined
    )
    prep_device_perf_report(
        model_name="hunyuan_recaption_prefill",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"prefill_ISL{PREFILL_ISL}",
    )
