# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview FFN modules on the Galaxy vs the HF ``hy_v4`` reference, on real weights (#65890, #65923).

Layout: the SC1 placement (TP32 / DP1, ``tt/hy4/layout.py``'s tp32) on the PHYSICAL 8x4 mesh: SP 8 on mesh axis 0,
TP 4 on mesh axis 1, i.e. the hidden over the 4-chip axis and the intermediate / tokens over the 8-chip axis, the
residual ``[S / 8, 6144 / 4]`` (#65884: hidden over its 4 rows). It is the dev layout's mesh too; only the shared
expert / dense FFN placement differs (2D here). ``HY4_MOE_SP_AXES=0,1`` adds the SP-on-axis-1 mesh (H on the 8-chip
axis), which the module also supports.

  * ``test_hy4_dense_ffn``: layer 0's dense FFN (intermediate 18432, unclamped) as ``TtMlp2D``: hidden over TP,
    intermediate over SP, no replication. Prefill form (tokens over SP, the block's layout) and decode form (tokens
    replicated). PCC >= 0.99 (#65923).
  * ``test_hy4_moe``: a MoE layer as ``TtMoe`` with the 2D-TP shared expert: sigmoid router with bias-corrected top-8
    (n_group 1), weights normalised x 2.827, EP32 routed experts (8 per chip, clamped SwiGLU limit 10 in the
    unified kernel), shared expert unclamped. Checked: router picks (recall), shared, routed and final outputs.
    Final PCC >= 0.971 (GLM-5.3's adapter threshold; #65923). Routed expert precision: BFP8 (the MXFP8 release's
    match, #65884's baseline), BFP4 (#65894's candidate), BF16.
  * ``clamp`` cases scale the real FFN input by 12 so ~1% of the routed gate and ~7% of the up values pass the limit (the real
    prompt reaches it on ~1e-6 of values) and compute the reference on the fly; the shared expert stays unclamped.

Inputs and goldens come from ``reference/hy4_moe_golden.py`` (the #65919 fp32 CPU reference run over the 4K
LongBench golden prompt): ``$HY4_MOE_REF_DIR/layer_{i}.safetensors``. Weights from ``$HY4_HF_MODEL`` (BF16 release).
Each missing input skips. Weight caches go under ``$HY4_MOE_TTNN_CACHE`` (default: next to the goldens).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference import hy4_moe as ref
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4TP32Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    extract_mesh_config,
    get_sp_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_mlp_2d import TtMlp2D
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import (
    GateComputeMode,
    TtMoEGateConfig,
    TtMoEGatePrefill,
    gate_bias_centered,
    gate_bias_fp32,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tt_py_test_utils_common.utils_for_testing import comp_pcc

CFG = Hy4TP32Config
REF_DIR = Path(os.environ.get("HY4_MOE_REF_DIR", "/mnt/tt-data/dwadhwani/hy4-moe-ref/longbench4k"))
HF_MODEL = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
CACHE_ROOT = Path(os.environ.get("HY4_MOE_TTNN_CACHE", str(REF_DIR.parent / "ttnn-cache")))
REPORT = os.environ.get("HY4_MOE_REPORT")  # optional JSON-lines file the results are appended to

DENSE_PCC = 0.99  # #65923: dense FFN PCC >= 0.99
MOE_PCC = 0.971  # #65923: MoE PCC >= GLM-5.3's adapter threshold
SHARED_PCC = 0.99  # the shared expert is a dense MLP; same bar as the dense FFN
GATE_RECALL = 0.97  # fraction of HF's top-8 picks the device router reproduces (near-ties may swap)
# The x12 clamp input pushes many sigmoid scores toward 1, so bias-corrected near-ties are far more common than on
# real activations (measured 0.964 at every expert precision; the final PCC stays >= 0.998, so the swaps are ties).
GATE_RECALL_CLAMP = 0.95
CLAMP_SCALE = float(os.environ.get("HY4_MOE_CLAMP_SCALE", "12"))
DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}


def _galaxy(sp_axis):
    return pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=moe_fabric_payload(CFG)),
        sp_axis,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="sc1-8x4-sp0" if sp_axis == 0 else "alt-8x4-sp1",
    )


MESHES = [_galaxy(int(a)) for a in os.environ.get("HY4_MOE_SP_AXES", "0").split(",")]


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not found at {path}")
    return path


def _golden(layer: int) -> dict:
    from safetensors.torch import load_file

    return load_file(str(_need(REF_DIR / f"layer_{layer}.safetensors", f"layer {layer} FFN golden")))


def _ckpt():
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    _need(HF_MODEL / "model.safetensors.index.json", "Hy4 checkpoint")
    return Hy4Checkpoint(HF_MODEL)


def _pcc(a, b) -> float:
    return float(comp_pcc(a.float(), b.float())[1])


def _report(**row):
    logger.info(f"HY4_MOE_RESULT {json.dumps(row)}")
    if REPORT:
        with open(REPORT, "a") as f:
            f.write(json.dumps(row) + "\n")


def _x_mapper(mesh_device, sp_axis):
    """Host ``[SP, S/SP, H]``: tokens over the SP axis, hidden over the TP axis (the prefill block's layout)."""
    return ttnn.ShardTensor2dMesh(
        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, -1) if sp_axis == 0 else (-1, 0)
    )


def _upload(x: torch.Tensor, mesh_device, sp_axis) -> ttnn.Tensor:
    sp = mesh_device.shape[sp_axis]
    return ttnn.from_torch(
        x.reshape(sp, x.shape[0] // sp, x.shape[-1]).to(torch.bfloat16),
        mesh_mapper=_x_mapper(mesh_device, sp_axis),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )


def _download(t: ttnn.Tensor, mesh_device, sp_axis) -> torch.Tensor:
    out = ttnn.to_torch(t, mesh_composer=get_tp_mesh_composer(mesh_device, sp_axis), dtype=torch.bfloat16)
    return out.reshape(-1, out.shape[-1]).float()


# ----------------------------------------------------------------------------------------------------- dense FFN
@pytest.mark.parametrize("mesh_device, device_params, sp_axis", MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("weights", ["bfp8", "bf16"])
def test_hy4_dense_ffn(mesh_device, device_params, sp_axis, weights):
    golden = _golden(0)
    ckpt = _ckpt()
    w = {p: ckpt._read(f"model.layers.0.mlp.{p}.weight") for p in ("gate_proj", "up_proj", "down_proj")}
    tp_axis = 1 - sp_axis
    topology = per_axis_topology(device_params["fabric_config"])
    ffn = TtMlp2D(
        mesh_device,
        emb_dim=CFG.EMB_SIZE,
        hidden_dim=CFG.INTERMEDIATE_SIZE,
        hidden_axis=tp_axis,
        torch_weights=w,
        weights_dtype=DTYPES[weights],
        activation=CFG.DENSE_FFN_ACTIVATION,
        num_links=2,
        topology=topology,
    )
    # bf16-rounded input through the fp32 reference, so the PCC measures the module, not the input rounding.
    x = golden["hn"].to(torch.bfloat16)
    expect = ref.swiglu_mlp(x, w["gate_proj"], w["up_proj"], w["down_proj"])
    got = _download(ffn(_upload(x, mesh_device, sp_axis)), mesh_device, sp_axis)
    pcc = _pcc(expect, got)
    pcc_hf = _pcc(golden["dense_out"], got)

    # Decode form: 32 tokens replicated over the SP axis; one all-reduce per axis, output replicated over SP.
    xd = x[:32].reshape(1, 32, -1)
    xd_tt = ttnn.from_torch(
        xd,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, -1) if sp_axis == 0 else (-1, None)
        ),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    yd = ttnn.to_torch(
        ffn(xd_tt, tokens_sharded=False), mesh_composer=get_tp_mesh_composer(mesh_device, sp_axis), dtype=torch.bfloat16
    ).float()  # [SP, 32, H]: every SP replica must hold the full answer
    pcc_decode = min(_pcc(expect[:32], yd[i]) for i in range(yd.shape[0]))
    _report(
        test="dense_ffn",
        mesh=f"{tuple(mesh_device.shape)} sp_axis={sp_axis}",
        weights=weights,
        tokens=x.shape[0],
        pcc=pcc,
        pcc_vs_hf_golden=pcc_hf,
        pcc_decode=pcc_decode,
    )
    assert pcc >= DENSE_PCC, f"dense FFN PCC {pcc:.5f} < {DENSE_PCC}"
    assert pcc_decode >= DENSE_PCC, f"dense FFN decode-form PCC {pcc_decode:.5f} < {DENSE_PCC}"


# ---------------------------------------------------------------------------------------------------------- MoE
def _moe_weights(ckpt, layer: int) -> dict:
    p = f"model.layers.{layer}.mlp."
    gate_up = ckpt._read(p + "experts.gate_up_proj")  # [E, 2I, H]
    down = ckpt._read(p + "experts.down_proj")  # [E, H, I]
    inter = CFG.MOE_INTERMEDIATE_SIZE
    return {
        "gate": {
            "weight": ckpt._read(p + "gate.weight"),
            "e_score_correction_bias": ckpt.get(p + "gate.e_score_correction_bias").float(),
        },
        "gate_up": gate_up,
        "down": down,
        "routed": [
            {"gate_proj": gate_up[e, :inter], "up_proj": gate_up[e, inter:], "down_proj": down[e]}
            for e in range(gate_up.shape[0])
        ],
        "shared": {q: ckpt._read(p + f"shared_experts.{q}.weight") for q in ("gate_proj", "up_proj", "down_proj")},
    }


def _recall(tt_idx: torch.Tensor, ref_idx: torch.Tensor) -> float:
    k = ref_idx.shape[-1]
    hits = (tt_idx.long().unsqueeze(-1) == ref_idx.long().unsqueeze(-2)).any(-1).sum()
    return float(hits) / (ref_idx.shape[0] * k)


@pytest.mark.parametrize("mesh_device, device_params, sp_axis", MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("experts", ["bfp8", "bfp4", "bf16"])
@pytest.mark.parametrize("case", ["real", "clamp"])
@pytest.mark.parametrize("layer", [1])
@pytest.mark.timeout(3600)
def test_hy4_moe(mesh_device, device_params, sp_axis, experts, case, layer):
    golden = _golden(layer)
    ckpt = _ckpt()
    w = _moe_weights(ckpt, layer)
    x = golden["hn"].to(torch.bfloat16)
    if case == "clamp":
        x = (x.float() * CLAMP_SCALE).to(torch.bfloat16)
    S, H = x.shape

    # Reference: HF HYV4MoE on the bf16-rounded input (router picks, routed half with the clamp, unclamped shared).
    if case == "real":
        # The golden ran on the fp32 hn; re-run both halves on the bf16 input the device sees, with the golden picks.
        r = {"topk_idx": golden["topk_idx"].long()}
        clamp = golden["clamp_frac"].tolist()
        r["routed"] = ref.routed_moe(
            x, golden["topk_idx"].long(), golden["topk_w"], w["gate_up"], w["down"], limit=CFG.SWIGLU_LIMIT
        )
        r["shared"] = ref.swiglu_mlp(x, w["shared"]["gate_proj"], w["shared"]["up_proj"], w["shared"]["down_proj"])
    else:
        r = ref.moe(
            x,
            gate_w=w["gate"]["weight"],
            bias=w["gate"]["e_score_correction_bias"],
            gate_up=w["gate_up"],
            down=w["down"],
            shared=w["shared"],
            top_k=CFG.NUM_EXPERTS_PER_TOKEN,
            limit=CFG.SWIGLU_LIMIT,
            routed_scaling_factor=CFG.ROUTE_SCALE,
        )
        clamp = list(ref.clamp_fractions(x, r["topk_idx"], w["gate_up"], limit=CFG.SWIGLU_LIMIT))
    expect_final = r["routed"] + r["shared"]

    sp = mesh_device.shape[sp_axis]
    seq_len_per_chip = S // sp
    mesh_config = extract_mesh_config(mesh_device, sp_axis)
    capacity = 8  # worst case: every token of the dispatch group sends all 8 picks to one chip's 8 experts
    experts_per_chip, metadata_len, max_buffer, max_per_expert = compute_constants(
        seq_len_per_chip,
        CFG.NUM_ROUTED_EXPERTS,
        CFG.NUM_EXPERTS_PER_TOKEN,
        mesh_device.get_num_devices(),
        mesh_config.dispatch_group_size,
        capacity,
    )
    assert experts_per_chip == 8, f"EP32 means 8 routed experts per chip, got {experts_per_chip}"
    cache = CACHE_ROOT / f"layer{layer}_{experts}_{mesh_device.shape[0]}x{mesh_device.shape[1]}_sp{sp_axis}"
    cache.mkdir(parents=True, exist_ok=True)

    moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=mesh_config.dispatch_group_size,
        num_dispatch_groups=mesh_config.num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=CFG.NUM_ROUTED_EXPERTS,
        num_experts_per_tok=CFG.NUM_EXPERTS_PER_TOKEN,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_per_expert,
        max_dispatch_buffer_token_size=max_buffer,
        seq_len_per_chip=seq_len_per_chip,
        gate_weights=w["gate"],
        emb_dim=H,
        hidden_dim=CFG.MOE_INTERMEDIATE_SIZE,
        n_expert_groups=CFG.NUM_EXPERT_GROUPS,
        n_limited_groups=CFG.NUM_LIMITED_GROUPS,
        route_scale=CFG.ROUTE_SCALE,
        dispatch_fwd_vc_pages=CFG.MOE_FWD_VC_PAGES,
        combine_fwd_vc_pages=CFG.MOE_FWD_VC_PAGES,
        num_links=2,
        topology=per_axis_topology(device_params["fabric_config"]),
        routed_expert_weights=w["routed"],
        shared_expert_weights=w["shared"],
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DTYPES[experts],
        routed_expert_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[CFG.ROUTED_EXPERT_ACTIVATION],
        routed_expert_hybrid_token_threshold=CFG.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        shared_expert_activation=CFG.SHARED_EXPERT_ACTIVATION,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        gate_bias_centered=gate_bias_centered(CFG),
        gate_bias_fp32=gate_bias_fp32(CFG),
        weight_cache_path=cache,
        layer_idx=layer,
        moe_transport="direct",
        sp_axis=sp_axis,
        shared_expert_2d_tp=True,
    )
    del w
    out, inter = moe(_upload(x, mesh_device, sp_axis), return_intermediates=True)
    ttnn.synchronize_device(mesh_device)

    got_final = _download(out, mesh_device, sp_axis)
    got_shared = _download(inter.shared_output, mesh_device, sp_axis)
    got_routed = _download(inter.routed_output, mesh_device, sp_axis)
    tt_idx = ttnn.to_torch(
        inter.gate_indices, mesh_composer=get_sp_mesh_composer(mesh_device, sp_axis), dtype=torch.int32
    )
    tt_idx = tt_idx.reshape(S, -1)[:, : CFG.NUM_EXPERTS_PER_TOKEN]

    row = dict(
        test="moe",
        case=case,
        layer=layer,
        mesh=f"{tuple(mesh_device.shape)} sp_axis={sp_axis}",
        experts=experts,
        tokens=S,
        clamp_frac_gate_up=clamp,
        gate_recall=_recall(tt_idx, r["topk_idx"]),
        pcc_shared=_pcc(r["shared"], got_shared),
        pcc_routed=_pcc(r["routed"], got_routed),
        pcc_final=_pcc(expect_final, got_final),
    )
    if case == "real":
        row["pcc_final_vs_hf_golden"] = _pcc(golden["routed_out"] + golden["shared_out"], got_final)
    _report(**row)
    assert row["gate_recall"] >= (GATE_RECALL if case == "real" else GATE_RECALL_CLAMP), row
    assert row["pcc_shared"] >= SHARED_PCC, row
    assert row["pcc_final"] >= MOE_PCC, row


# ---------------------------------------------------------------------------------------------------- router bias
BIAS_LAYERS = [int(n) for n in os.environ.get("HY4_ROUTER_BIAS_LAYERS", "1,2,20,40,77").split(",")]


@pytest.mark.parametrize("mesh_device, device_params, sp_axis", MESHES[:1], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("layer", BIAS_LAYERS)
@pytest.mark.timeout(1800)
def test_hy4_router_bias_precision(mesh_device, device_params, sp_axis, layer):
    """Expert picks of the device router (TtMoEGatePrefill, DEVICE_FP32) with the correction bias in bf16 (before) and
    fp32 (MOE_GATE_BIAS_FP32, after) against the fp32 HF router on the same bf16 input. The input is the layer's real
    FFN input where a golden has it (layers 1, 2), else layer 2's (the bias and the gate weight are the layer's own).
    Recorded, and the fp32 bias must not lose picks."""
    ckpt = _ckpt()
    p = f"model.layers.{layer}.mlp."
    gate_w = ckpt._read(p + "gate.weight")
    bias = ckpt.get(p + "gate.e_score_correction_bias").float()
    src = layer if (REF_DIR / f"layer_{layer}.safetensors").exists() else 2
    x = _golden(src)["hn"].to(torch.bfloat16)
    S = x.shape[0]
    ref_idx, _ = ref.route(
        x.float(), gate_w.float(), bias, top_k=CFG.NUM_EXPERTS_PER_TOKEN, routed_scaling_factor=CFG.ROUTE_SCALE
    )
    sp = mesh_device.shape[sp_axis]
    row = dict(test="router_bias", layer=layer, input_layer=src, tokens=S)
    row["bias_bf16_max_abs_err"] = float((bias.to(torch.bfloat16).float() - bias).abs().max())
    for name, fp32 in (("bf16", False), ("fp32", True)):
        config = TtMoEGateConfig.from_model_cfg(CFG, sp_dim=S // sp, bias_fp32=fp32)
        config.ccl_config["DISPATCH_AXIS"] = sp_axis
        config.ccl_config["TP_AXIS"] = 1 - sp_axis
        config.ccl_config["NUM_LINKS"] = 2
        config.ccl_config["TOPOLOGY"] = per_axis_topology(device_params["fabric_config"])[1 - sp_axis]
        gate = TtMoEGatePrefill(
            config, mesh_device, weight=gate_w, bias=bias, fallback_mode=GateComputeMode.DEVICE_FP32
        )
        assert gate.bias.dtype == (ttnn.float32 if fp32 else ttnn.bfloat16)
        x_tt = _upload(x, mesh_device, sp_axis)
        _, idx, _ = gate(ttnn.view(x_tt, (x_tt.shape[0] * x_tt.shape[1], x_tt.shape[2])))
        tt_idx = ttnn.to_torch(idx, mesh_composer=get_sp_mesh_composer(mesh_device, sp_axis), dtype=torch.int32)
        tt_idx = tt_idx.reshape(S, -1)[:, : CFG.NUM_EXPERTS_PER_TOKEN]
        row[f"recall_{name}"] = _recall(tt_idx, ref_idx)
        row[f"rows_exact_{name}"] = float(
            (torch.sort(tt_idx.long(), -1).values == torch.sort(ref_idx.long(), -1).values).all(-1).float().mean()
        )
        del gate
    _report(**row)
    assert row["recall_fp32"] >= row["recall_bf16"] - 1e-3, row
    assert row["recall_fp32"] >= GATE_RECALL, row
