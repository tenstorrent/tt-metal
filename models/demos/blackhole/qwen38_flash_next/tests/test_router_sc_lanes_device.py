# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_ROUTER_SC_LANES=1 (one fast top-k group per occupied sort pass inside moe_dense) on the TP=8 mesh.

Correctness, rows in {2, 8, 9, 17, 32}: for every row r < rows, the flag-on moe_dense's scores (bf16 bits) and indices (ten
slots in order), on every die, equal (a) the flag-off moe_dense's row r (today's lane form) and (b) router_tail_sc's rows == 1
program run on that row alone (batch invariance).  Inputs: bf16-exact random logits, plain fp32 random logits, rows with
12..60 forced ties at the top.  A mismatch on a row whose sort keys (logit + 128 in fp32) tie while the raw logits differ is
the documented raw-logit case: it is printed (RTLANES_TIE) and not a failure.  Timing: traced moe_dense at rows 8 and 32,
flag 0 vs 1 (RTLANES_TIMING).

Run: see the runner R/go.sh (MESH_DEVICE=P150x8, TT_VISIBLE_DEVICES=0..7)."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_dense as md
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc as rt_sc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import shared_expert as se

BF16 = ttnn.bfloat16
HIDDEN = se.HIDDEN
FLAG = rt_sc.LANES_ENV
PARAMS = {**DEVICE_PARAMS, "trace_region_size": 64 * 1024 * 1024}
ROWS = [2, 8, 9, 17, 32]
_SETUP: dict = {}
# the mesh _SETUP's tensors live on: a reopened mesh starts a fresh cache
_SETUP_MESH: list = []


def per_die(tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _logit_rows(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    kinds = []
    out = torch.empty(rows, rt.EXPERTS)
    for r in range(rows):
        kind = ("bf16", "fp32", "ties")[r % 3]
        v = torch.randn(rt.EXPERTS, generator=g) * 2.0
        if kind == "bf16":
            v = v.to(torch.bfloat16).float()
        elif kind == "ties":  # 12..60 columns share the maximum (ties inside the top 12), bf16-exact tail
            v = v.to(torch.bfloat16).float()
            n = int(torch.randint(12, 61, (1,), generator=g))
            v[torch.randperm(rt.EXPERTS, generator=g)[:n]] = 5.0
        out[r] = v
        kinds.append(kind)
    return out, kinds


def _raw_tie_case(row: torch.Tensor) -> bool:
    """The documented case: two of the top-12 sort keys (logit + 128, fp32) are equal while the logits differ."""
    top = torch.topk(row, 12).values
    key = top + 128.0
    return any(bool(key[i] == key[i + 1]) and bool(top[i] != top[i + 1]) for i in range(11))


def _setup(mesh, rows):
    if not _SETUP_MESH or _SETUP_MESH[0] is not mesh:
        _SETUP.clear()
        _SETUP_MESH[:] = [mesh]
    if rows in _SETUP:
        return _SETUP[rows]
    shared = se.gate_configs(mesh)
    gen = torch.Generator().manual_seed(5100 + rows)
    weights = se.upload_weights(
        mesh,
        torch.randn(HIDDEN, se.INTERMEDIATE, generator=gen) * 0.02,
        torch.randn(HIDDEN, se.INTERMEDIATE, generator=gen) * 0.02,
        torch.randn(HIDDEN, 1, generator=gen) * 0.02,
        torch.randn(se.INTERMEDIATE, HIDDEN, generator=gen) * 0.02,
    )
    hidden_dram = ttnn.from_torch(
        (torch.randn(1, 1, rows, HIDDEN, generator=gen) * 2).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    full_hidden = ttnn.to_memory_config(hidden_dram, shared["hidden_act_memory_config"])
    gate_up = ttnn.linear(
        full_hidden,
        weights["gate_up_scalar"],
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=shared["gate_up_scalar_program_config"],
        compute_kernel_config=shared["compute_kernel_config"],
    )
    _SETUP[rows] = (weights, full_hidden, gate_up, md.configs(mesh))
    return _SETUP[rows]


def _upload(mesh, values):
    return ttnn.from_torch(
        values.float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _outputs(mesh, rows):
    return (
        fp.allocate((1, 1, rows, md.TOP_K), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh),
        fp.allocate((1, 1, rows, md.TOP_K), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh),
    )


def _moe_dense(mesh, rows, logits, flag, outs=None):
    """moe_dense's full result tuple; (scores, indices) are ``outs`` (allocated here when None)."""
    weights, full_hidden, gate_up, cfg = _setup(mesh, rows)
    os.environ[FLAG] = "1" if flag else "0"
    scores, indices = _outputs(mesh, rows) if outs is None else outs
    return md.moe_dense(logits, gate_up, weights["down"], full_hidden, scores=scores, indices=indices, **cfg)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", ROWS)
def test_lanes_equal_lane_form_and_rows1_form(mesh_device, rows):
    mesh = mesh_device
    rt_sc.prepare(mesh)
    values, kinds = _logit_rows(rows, 7000 + rows)
    logits = _upload(mesh, values.reshape(1, 1, rows, rt.EXPERTS))
    try:
        off_s, off_i = _moe_dense(mesh, rows, logits, False)[:2]
        on_s, on_i = _moe_dense(mesh, rows, logits, True)[:2]
        a = [[x.reshape(rows, md.TOP_K) for x in per_die(t)] for t in (off_s, off_i, on_s, on_i)]
        # (b) the rows == 1 form on each row alone
        os.environ[FLAG] = "1"
        single = []
        for r in range(rows):
            one = _upload(mesh, values[r].reshape(1, 1, 1, rt.EXPERTS))
            s1 = fp.allocate((1, 1, 1, md.TOP_K), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh)
            i1 = fp.allocate((1, 1, 1, md.TOP_K), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh)
            rt_sc.router_tail_sc_into(one, s1, i1)
            single.append(([x.reshape(md.TOP_K) for x in per_die(s1)], [x.reshape(md.TOP_K) for x in per_die(i1)]))
            for t in (one, s1, i1):
                ttnn.deallocate(t)
    finally:
        os.environ.pop(FLAG, None)
    report = {"rows": rows, "vs_lane_form_bad": [], "vs_rows1_bad": [], "tie_info": [], "dies": len(a[0])}
    for d in range(len(a[0])):
        off_s, off_i, on_s, on_i = (x[d] for x in a)
        for r in range(rows):
            same_a = torch.equal(on_s[r].view(torch.int16), off_s[r].view(torch.int16)) and torch.equal(
                on_i[r], off_i[r]
            )
            s1, i1 = single[r][0][d], single[r][1][d]
            same_b = torch.equal(on_s[r].view(torch.int16), s1.view(torch.int16)) and torch.equal(on_i[r], i1)
            if not (same_a and same_b) and "first_diff" not in report:
                report["first_diff"] = {
                    "die": d,
                    "row": r,
                    "on_s": on_s[r].float().tolist(),
                    "off_s": off_s[r].float().tolist(),
                    "s1": s1.float().tolist(),
                    "on_i": on_i[r].tolist(),
                    "off_i": off_i[r].tolist(),
                    "i1": i1.tolist(),
                }
            if not same_b:  # batch invariance: exact for every row
                report["vs_rows1_bad"].append({"die": d, "row": r, "kind": kinds[r]})
            if same_a:
                continue
            # a lane-form mismatch is a tie case only when the new form equals the rows == 1 form, the index sets are
            # equal and every slot whose index differs has a bitwise-equal score
            slots = [k for k in range(md.TOP_K) if int(on_i[r][k]) != int(off_i[r][k])]
            scores_equal = all(int(on_s[r].view(torch.int16)[k]) == int(off_s[r].view(torch.int16)[k]) for k in slots)
            same_set = sorted(on_i[r].tolist()) == sorted(off_i[r].tolist())
            if same_b and same_set and scores_equal:
                report["tie_info"].append(
                    {
                        "die": d,
                        "row": r,
                        "kind": kinds[r],
                        "slots": slots,
                        "indices_new": [int(on_i[r][k]) for k in slots],
                        "indices_lane": [int(off_i[r][k]) for k in slots],
                    }
                )
            else:
                report["vs_lane_form_bad"].append({"die": d, "row": r, "kind": kinds[r]})
    for item in report["tie_info"]:
        print("RTLANES_TIE", json.dumps(item), flush=True)
    print("RTLANES_RESULT", json.dumps(report), flush=True)
    assert not report["vs_lane_form_bad"], report
    assert not report["vs_rows1_bad"], report


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [8, 32])
def test_lanes_traced_timing(mesh_device, rows):
    mesh = mesh_device
    rt_sc.prepare(mesh)
    values, _ = _logit_rows(rows, 8000 + rows)
    logits = _upload(mesh, values.reshape(1, 1, rows, rt.EXPERTS))
    times = {}
    try:
        for flag in (False, True):
            outs = _outputs(mesh, rows)

            def call(flag=flag, outs=outs):
                for t in _moe_dense(mesh, rows, logits, flag, outs)[2:]:
                    if t is not None:
                        ttnn.deallocate(t)

            times[flag] = round(traced_wall_us(mesh, call, replays=50)[0], 2)
    finally:
        os.environ.pop(FLAG, None)
    print(f"RTLANES_TIMING rows={rows} off_us={times[False]} on_us={times[True]}", flush=True)
