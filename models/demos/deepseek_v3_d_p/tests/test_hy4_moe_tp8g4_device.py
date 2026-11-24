# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tp8g4 routed experts on the 8x4 Galaxy with stock ttnn ops (``tt/moe/tt_routed_tp8g4.py``) against prefill's EP32
``TtMoe`` on the same 4K golden FFN input (phase 1 of the shared expert layout, #65884).

  * ``test_tp8g4_routed``: one MoE layer's routed half on the split (BFP4 slices as plain ttnn tensors, golden
    routing), output PCC against the fp32 reference, wall time per stage (eager, each stage closed by a device
    synchronize; median of ``HY4_TP8G4_ITERS`` after one warm-up).
  * ``test_ep32_moe``: ``TtMoe`` as ``tests/pcc/test_hy4_moe.py`` builds it (EP32 BFP4, its own device router, 2D-TP
    shared expert), with the same per-stage timing (gate, routing setup, all-gather, dispatch, experts, combine,
    reduce, shared expert); routed / final PCC against the reference.

Inputs: ``tests/test_hy4_moe_tp8g4_cpu.py``'s golden packs (``$HY4_TP8G4_WORK/golden/layer_{L}.pt``: the FFN input
from the 4K ropeint reference streams, the fp32 router picks and outputs). Weight caches under
``$HY4_TP8G4_WORK/ttnn-cache``. Rows to the log (``HY4_TP8G4``) and ``HY4_TP8G4_REPORT``.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference import hy4_moe_tp8g4 as t8
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4TP32Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_tp8g4 import SyncTimer, TtRoutedTp8g4

CFG = Hy4TP32Config
HF_MODEL = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
WORK = Path(os.environ.get("HY4_TP8G4_WORK", "/mnt/tt-data/dwadhwani/hy4-test/moe-tp8g4"))
REPORT = os.environ.get("HY4_TP8G4_REPORT")
LAYERS = [int(v) for v in os.environ.get("HY4_TP8G4_DEVICE_LAYERS", "1").split(",")]
ITERS = int(os.environ.get("HY4_TP8G4_ITERS", "3"))

MESH = pytest.param(
    (8, 4),
    torus_xy_device_params(fabric_payload_size=moe_fabric_payload(CFG)),
    marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
    id="sc1-8x4",
)


def _pack(layer):
    path = WORK / "golden" / f"layer_{layer}.pt"
    if not path.is_file():
        pytest.skip(f"no golden pack {path} (run test_hy4_moe_tp8g4_cpu.py -k golden first)")
    return torch.load(path, weights_only=True)


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _emit(row):
    logger.info(f"HY4_TP8G4 {json.dumps(row)}")
    if REPORT:
        with open(REPORT, "a") as f:
            f.write(json.dumps(row) + "\n")


def _upload(x, mesh_device):
    """Host [S, H] -> the residual layout [S/8, H/4] per chip (tokens over axis 0, hidden over axis 1)."""
    return ttnn.from_torch(
        x.reshape(8, x.shape[0] // 8, x.shape[-1]).to(torch.bfloat16),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(0, -1)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )


def _download(t, mesh_device):
    out = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(8, 4), dims=(0, -1)), dtype=torch.bfloat16
    )
    return out.reshape(-1, out.shape[-1]).float()


def _ckpt():
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    return Hy4Checkpoint(HF_MODEL)


def _median_times(runs):
    return {k: round(statistics.median(r[k] for r in runs) * 1e3, 3) for k in runs[0]}


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True)
@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.timeout(7200)
def test_tp8g4_routed(mesh_device, device_params, layer):
    pk = _pack(layer)
    S = pk["hn"].shape[0]
    cache = WORK / "ttnn-cache" / f"layer{layer}_bfp4_tp8g4"
    have_cache = cache.is_dir() and len(list(cache.glob("tp8g4.local_*"))) == 3 * t8.EPG
    t0 = time.time()
    if have_cache:
        gate_up = down = None
    else:
        cache.mkdir(parents=True, exist_ok=True)
        ckpt = _ckpt()
        p = f"model.layers.{layer}.mlp."
        gate_up, down = ckpt._read(p + "experts.gate_up_proj"), ckpt._read(p + "experts.down_proj")
    nd = os.environ.get("HY4_TP8G4_ND", "1") == "1"
    moe = TtRoutedTp8g4(mesh_device, gate_up, down, cache_path=cache, nd_sharded=nd)
    del gate_up, down
    logger.info(f"HY4_TP8G4 weights placed in {time.time() - t0:.0f}s (cache hit {have_cache})")
    plans = t8.plan(pk["top_i"], pk["top_w"])
    plan = moe.upload_plan(plans, S)
    x = _upload(pk["hn"], mesh_device)
    for combine in os.environ.get("HY4_TP8G4_COMBINE", "pcr,embedding").split(","):
        runs = []
        out = None
        for it in range(ITERS + 1):
            if out is not None:
                ttnn.deallocate(out)
            timer = SyncTimer(mesh_device)
            t_start = time.perf_counter()
            out = moe.forward(x, plan, timer=timer, combine=combine)
            ttnn.synchronize_device(mesh_device)
            timer.times["total"] = time.perf_counter() - t_start
            if it:
                runs.append(timer.times)
        # one more pass for the end-to-end wall time without per-stage synchronizes
        walls = []
        for _ in range(ITERS):
            ttnn.deallocate(out)
            ttnn.synchronize_device(mesh_device)
            t_start = time.perf_counter()
            out = moe.forward(x, plan, combine=combine)
            ttnn.synchronize_device(mesh_device)
            walls.append(time.perf_counter() - t_start)
        got = _download(out, mesh_device)
        ttnn.deallocate(out)
        row = dict(
            test="device_tp8g4",
            layer=layer,
            combine=combine,
            weights_nd_sharded=nd,
            tokens=S,
            R_per_group=[p.R for p in plans],
            K_per_group=[p.K for p in plans],
            padded_R=plan["R"],
            K=plan["K"],
            max_tokens_per_expert=plan["max_tokens"],
            pcc_routed=_pcc(got, pk["routed"]),
            rel_err=float((got - pk["routed"]).norm() / pk["routed"].norm()),
            ms=_median_times(runs),
            ms_total_no_stage_sync=round(statistics.median(walls) * 1e3, 3),
        )
        _emit(row)
        assert row["pcc_routed"] > 0.98, row
    x_bf16 = pk["hn"].to(torch.bfloat16).float()
    if os.environ.get("HY4_TP8G4_HOST_CHECK"):  # host BFP4 emulation of the same split on the bf16 input (slow)
        ckpt = _ckpt()
        p = f"model.layers.{layer}.mlp."
        emu = t8.tp8g4_routed(
            x_bf16,
            pk["top_i"],
            pk["top_w"],
            ckpt._read(p + "experts.gate_up_proj"),
            ckpt._read(p + "experts.down_proj"),
            weights_fn=t8.bfp4_slices,
            plans=plans,
        )
        _emit(dict(test="device_tp8g4_vs_host_bfp4", layer=layer, pcc=_pcc(got, emu)))


# ------------------------------------------------------------------------------------------------------- EP32
def _timed(obj, attr, timer, name):
    """Wrap obj.attr (a callable) so its time lands in timer.times[name] (synchronize before and after)."""
    fn = getattr(obj, attr)

    def wrapped(*a, **k):
        timer(None)
        r = fn(*a, **k)
        timer(name)
        return r

    setattr(obj, attr, wrapped)


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True)
@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.timeout(7200)
def test_ep32_moe(mesh_device, device_params, layer):
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        compute_constants,
        extract_mesh_config,
        get_sp_mesh_composer,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import (
        GateComputeMode,
        gate_bias_centered,
        gate_bias_fp32,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    pk = _pack(layer)
    S, H = pk["hn"].shape
    sp_axis = 0
    ckpt = _ckpt()
    p = f"model.layers.{layer}.mlp."
    gate_up, down = ckpt._read(p + "experts.gate_up_proj"), ckpt._read(p + "experts.down_proj")
    inter = CFG.MOE_INTERMEDIATE_SIZE
    mesh_config = extract_mesh_config(mesh_device, sp_axis)
    experts_per_chip, metadata_len, max_buffer, max_per_expert = compute_constants(
        S // 8, CFG.NUM_ROUTED_EXPERTS, CFG.NUM_EXPERTS_PER_TOKEN, 32, mesh_config.dispatch_group_size, 8
    )
    cache = WORK / "ttnn-cache" / f"layer{layer}_bfp4_ep32"
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
        seq_len_per_chip=S // 8,
        gate_weights={
            "weight": ckpt._read(p + "gate.weight"),
            "e_score_correction_bias": ckpt.get(p + "gate.e_score_correction_bias").float(),
        },
        emb_dim=H,
        hidden_dim=inter,
        n_expert_groups=CFG.NUM_EXPERT_GROUPS,
        n_limited_groups=CFG.NUM_LIMITED_GROUPS,
        route_scale=CFG.ROUTE_SCALE,
        dispatch_fwd_vc_pages=CFG.MOE_FWD_VC_PAGES,
        combine_fwd_vc_pages=CFG.MOE_FWD_VC_PAGES,
        num_links=2,
        topology=per_axis_topology(device_params["fabric_config"]),
        routed_expert_weights=[
            {"gate_proj": gate_up[e, :inter], "up_proj": gate_up[e, inter:], "down_proj": down[e]}
            for e in range(gate_up.shape[0])
        ],
        shared_expert_weights={
            q: ckpt._read(p + f"shared_experts.{q}.weight") for q in ("gate_proj", "up_proj", "down_proj")
        },
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
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
    del gate_up, down
    x = _upload(pk["hn"], mesh_device)
    runs = []
    timer = SyncTimer(mesh_device)
    stages = [
        (moe.gate, "gate"),
        (moe.routing_setup, "routing_setup"),
        (moe.shared_expert, "shared_expert"),
        (moe.dispatch_module, "dispatch"),
        (moe.routed_expert, "experts"),
        (moe.combine_module, "combine"),
        (moe.reduce_module, "reduce_weighted_rs_axis1"),
    ]
    for obj, name in stages:
        _timed(obj, "forward", timer, name)
    for it in range(ITERS + 1):
        timer.times = {}
        timer(None)
        t_start = time.perf_counter()
        out, _ = moe(x)
        ttnn.synchronize_device(mesh_device)
        timer.times["total"] = time.perf_counter() - t_start
        timer.times["rest(ag_axis1,final add,...)"] = timer.times["total"] - sum(
            v for k, v in timer.times.items() if k != "total"
        )
        if it:
            runs.append(dict(timer.times))
        ttnn.deallocate(out)
    for obj, _ in stages:  # unwrap: end-to-end wall time without per-stage synchronizes
        del obj.forward
    walls = []
    for _ in range(ITERS):
        ttnn.synchronize_device(mesh_device)
        t_start = time.perf_counter()
        out, _ = moe(x)
        ttnn.synchronize_device(mesh_device)
        walls.append(time.perf_counter() - t_start)
        ttnn.deallocate(out)
    out, inter_t = moe(x, return_intermediates=True)
    got_routed = ttnn.to_torch(
        inter_t.routed_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(8, 4), dims=(0, -1)),
        dtype=torch.bfloat16,
    )
    got_routed = got_routed.reshape(-1, got_routed.shape[-1]).float()
    got_final = _download(out, mesh_device)
    tt_idx = ttnn.to_torch(
        inter_t.gate_indices, mesh_composer=get_sp_mesh_composer(mesh_device, sp_axis), dtype=torch.int32
    )
    tt_idx = tt_idx.reshape(S, -1)[:, : CFG.NUM_EXPERTS_PER_TOKEN].long()
    recall = float((tt_idx.unsqueeze(-1) == pk["top_i"].long().unsqueeze(-2)).any(-1).float().mean())
    row = dict(
        test="device_ep32",
        layer=layer,
        tokens=S,
        gate_recall=recall,
        pcc_routed=_pcc(got_routed, pk["routed"]),
        rel_err=float((got_routed - pk["routed"]).norm() / pk["routed"].norm()),
        pcc_final=_pcc(got_final, pk["routed"] + pk["shared"]),
        ms=_median_times(runs),
        ms_total_no_stage_sync=round(statistics.median(walls) * 1e3, 3),
    )
    _emit(row)
