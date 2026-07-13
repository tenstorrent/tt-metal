# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-performance test for the HunyuanImage-3.0 recaption (AR) stage (tracy).

The recaption stage rewrites the prompt by autoregressively generating text: one
full backbone forward per generated token (T2I uses ``make_backbone_logits_fn`` —
a full-sequence forward each step, no KV cache). In an end-to-end T2I run this
stage is ~44% of wall-clock (512 tokens x ~0.8s), so the per-token backbone
forward is the unit worth profiling.

The backbone is built with the SAME config the demo's recaption stage uses (2x2
mesh, bf8 experts, TP=2 / SP=2, expert-parallel MoE, first/last 3 layers bf16) so
the profiled op mix matches the real run. (The single-device path is not used —
it routes through ``HunyuanTtMoE``, a code path that is currently broken.)

Two tests, the standard tt-metal device-perf split (same shape as
``test_vae_decode_perf.py``):

  * ``test_recaption_ar_device_ops`` — the workload. Builds the backbone + LM head
    once, warms the program cache with one forward at the profiled sequence length,
    then runs a SECOND (warm) forward bracketed by tracy ``signpost("start")`` /
    ``signpost("stop")`` so only ONE AR token's device ops are measured. Multiply the
    reported duration by ``max_new_tokens`` for the stage estimate (the T2I path
    re-runs a full-sequence forward per token, so per-token cost also grows with S —
    see HY_PERF_TOKEN_POS).

  * ``test_recaption_ar_perf_device`` — the gate. Re-invokes the workload under the
    device profiler, sums DEVICE KERNEL duration between the signposts, reports it.

Env knobs:
  HY_NUM_LAYERS        backbone layers to build (default: full num_hidden_layers).
                       Set e.g. 6 for a fast smoke run; use the full count for a
                       representative per-token number.
  HY_PERF_TOKEN_POS    how many tokens past the prompt to profile at (default 128),
                       i.e. profiled S = prefix_len + HY_PERF_TOKEN_POS. Simulates a
                       mid-generation token; the T2I full-seq forward cost scales
                       with S, so this picks the representative point.

Run the raw op profile (workload only, writes ops_perf_results_*.csv):
    python3 tools/tracy/profile_this.py -n hunyuan_recaption_ar \
      -c "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_ar_perf.py::test_recaption_ar_device_ops -s"

Run the device-perf gate (spawns the workload under the profiler for you):
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_ar_perf.py::test_recaption_ar_perf_device
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
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import enrich_bundle_attention
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

PROMPT = "a cat on a mat"
BOT_TASK = "recaption"
SEQUENCE_TEMPLATE = "instruct"
PERF_TOKEN_POS = int(os.environ.get("HY_PERF_TOKEN_POS", "128"))
SP_FACTOR = 2  # must match sp_factor passed to HunyuanTtModel below
SP_TILE_ALIGN = SP_FACTOR * 32  # model.py shards the seq dim in TILE(32)-aligned chunks

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
    """Build the demo-config backbone + LM head + a padded ids tensor, and return a
    warm per-token forward closure plus the profiled sequence length."""
    num_layers = int(os.environ.get("HY_NUM_LAYERS", str(_full_num_layers())))
    c = h.model_cfg()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tok = HunyuanTokenizer.from_model_dir(h.I2I_WEIGHTS, sequence_template=SEQUENCE_TEMPLATE)
    proc = HunyuanImage3ImageProcessor(json.load(open(h.I2I_WEIGHTS / "config.json")))
    bundle = prepare_recaption_inputs(tok, PROMPT, bot_task=BOT_TASK, sequence_template=SEQUENCE_TEMPLATE)
    bundle = enrich_bundle_attention(bundle, proc)

    wte = h.load_tensor("model.wte.weight")
    ln_f_w = h.load_tensor("model.ln_f.weight")

    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    # Matches demo/_build_backbone: bf8 experts, TP=2/SP=2 on the 2x2 mesh,
    # expert-parallel MoE (ccl_manager set), first/last 3 layers in bf16.
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
        sp_factor=SP_FACTOR,
        bf16_layers=[],
        model_cache_name="hunyuan-image-3.0",
    )
    lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": h.load_tensor("lm_head.weight")})

    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]

    def attention_mask_fn(S: int):
        mask_bool = build_attention_mask(S, attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, S, S)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    forward_logits_fn = make_backbone_logits_fn(
        backbone,
        lm_head,
        mesh_device,
        attention_mask_fn=attention_mask_fn,
        image_infos=image_infos,
    )

    # Build a token sequence at the profiled length: the real prompt prefix padded to
    # prefix_len + PERF_TOKEN_POS with a valid token id (values don't affect timing).
    prefix_len = int(bundle.input_ids.shape[1])
    seq_len = prefix_len + max(PERF_TOKEN_POS, 0)
    # Round up to the sp-shard tile alignment so model.forward()'s internal
    # sp_pad is 0 — the pad/fill_pad ops it would otherwise emit are pure
    # alignment overhead (masked-out, discarded on exit) and add nothing to a
    # representative per-token timing; padding token values don't affect timing.
    seq_len = ((seq_len + SP_TILE_ALIGN - 1) // SP_TILE_ALIGN) * SP_TILE_ALIGN
    pad = bundle.input_ids[:, -1:].repeat(1, seq_len - prefix_len)
    ids = torch.cat([bundle.input_ids, pad], dim=1) if seq_len > prefix_len else bundle.input_ids

    logger.info(
        f"recaption AR workload: num_layers={num_layers} prefix_len={prefix_len} "
        f"profiled_seq_len={seq_len} (token_pos={PERF_TOKEN_POS})"
    )
    return forward_logits_fn, ids, seq_len


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)  # full-layer bf8 backbone build + 2 forwards; overrides the 300s default
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_recaption_ar_device_ops(mesh_device):
    """One warm AR-token backbone forward, bracketed by signposts for the perf gate."""
    mesh_device.enable_program_cache()

    forward_logits_fn, ids, seq_len = _build_workload(mesh_device)

    logits = forward_logits_fn(ids)  # warmup: compile programs / fill cache
    assert logits is not None and logits.shape[-1] > 0, "forward produced no logits"

    if use_signpost:
        signpost(header="start")
    logits = forward_logits_fn(ids)  # profiled run (warm)
    if use_signpost:
        signpost(header="stop")

    logger.info(f"recaption AR forward logits shape={tuple(logits.shape)} at seq_len={seq_len}")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    # Summed DEVICE KERNEL DURATION [ns] between signposts for ONE AR-token forward.
    # PLACEHOLDER — re-baseline on first green run (assert_on_fail=False reports only).
    # Stage estimate = this value x max_new_tokens (512), plus prefill; note the T2I
    # full-seq path also grows this per-token cost with sequence length.
    "expected_device_kernel_duration_ns",
    [1_000_000_000],
)
def test_recaption_ar_perf_device(expected_device_kernel_duration_ns):
    batch_size = 1  # one AR token per profiled forward
    subdir = "hunyuan_recaption_ar"
    num_iterations = 3
    margin = 0.05
    cols = ["DEVICE KERNEL"]

    command = "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_recaption_ar_perf.py::test_recaption_ar_device_ops"

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
        # Bump well above the observed per-forward op count.
        op_support_count=20000,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=margin,
        expected_perf_cols={duration_key: expected_device_kernel_duration_ns},
        assert_on_fail=False,  # placeholder baseline — report only until re-baselined
    )
    prep_device_perf_report(
        model_name="hunyuan_recaption_ar",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"AR_token_forward_pos{PERF_TOKEN_POS}",
    )
