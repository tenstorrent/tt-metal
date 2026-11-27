# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf bake-off: progressive output (several blocks per core so the output drains during the K-walk).

Runs the REAL genify_matmul op with its plan() monkeypatched to a forced per-core Blocking on the
op's own grid (the kernels already walk nb_m x nb_n blocks). Nothing in the op is edited.

Variant string (PMB_VARIANT env, or the parametrize id):
    base                 the op's own plan, untouched
    M<nb>  / N<nb>       split the core region into nb M-blocks / N-blocks
    ...Q<q>              N quantum override for an N split (default: the grid's q_n)
    ...K<bk>             force block_k (default: the largest that fits the L1 budget, "auto")
    ...Kb                keep the baseline block_k
    ...W<w>              operand CB depth (cb_act_tiles and cb_weight_tiles, in K-blocks; default the op's)
    ...D<d>              cb_output_tiles depth in row groups (default: the whole block's row groups,
                         so block i's epilogue never waits on its own writes)
Mode (PMB_MODE env): real (full op, PCC checked) | anything else is passed as GENIFY_ABLATE (e.g. epi:
epilogue compute stubbed; reads,mcast,writes: compute only), output wrong by design, no check.

Skipped unless GENIFY_MATMUL_PERF=1. Profile:
  GENIFY_MATMUL_PERF=1 PMB_CASE=focus PMB_VARIANT=M2 PMB_MODE=real \
    scripts/run_safe_pytest.sh --profile <this file>
"""
import os
import re
from dataclasses import replace

import pytest
import torch
import ttnn

import eval.golden_tests.genify_matmul.helpers as helpers
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as pd

pytestmark = [
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench: set GENIFY_MATMUL_PERF=1"),
]

_BF16, _BFP4 = ttnn.bfloat16, ttnn.bfloat4_b
_LOFI, _HIFI2, _HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4

# name -> (inputs, dtype, weight_dtype, fidelity, epilogue)
CASES = {
    "focus": (((1, 2048, 5120), (5120, 8704)), _BF16, _BFP4, _LOFI, "swiglu"),
    "q4160": (((1, 2048, 5120), (5120, 4160)), _BF16, _BFP4, _LOFI, "none"),
    "fx_addcmul": (((1, 2048, 6144), (6144, 3072)), _BF16, _BF16, _HIFI2, "addcmul"),
    "sq4096": (((1, 4096, 4096), (4096, 4096)), _BF16, _BF16, _HIFI4, "none"),
    "fx_ff1_txt": (((1, 256, 6144), (6144, 18432)), _BF16, _BF16, _HIFI2, "swiglu"),
    "q4096_swiglu": (((1, 4096, 5120), (5120, 8704)), _BF16, _BFP4, _LOFI, "swiglu"),
}


def _parse(variant):
    if variant == "base":
        return None
    m = re.fullmatch(r"([MN])(\d+)(?:Q(\d+))?(?:K(\d+|b))?(?:D(\d+))?(?:W(\d+|m))?", variant)
    assert m, f"bad variant {variant}"
    axis, nb, q, bk, d = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5)
    w = m.group(6)
    w = None if w is None else ("max" if w == "m" else int(w))
    return axis, nb, (int(q) if q else None), bk, (int(d) if d else None), w


def _forced_plan(orig_plan, spec, log):
    axis, nb, q_n, bk_spec0, depth_out, w0 = spec

    def plan(a, b, out, fp32_dest_acc_en, device, epilogue="none", row=None, residual=None):
        saved = pd.DEPTH_OUT
        pd.DEPTH_OUT = 2  # the op's own grid choice, untouched
        geom, grid, base_blk, gxy = orig_plan(
            a, b, out, fp32_dest_acc_en, device, epilogue=epilogue, row=row, residual=residual
        )
        pd.DEPTH_OUT = saved
        cm, cn = grid.core_m_tiles, grid.core_n_tiles
        if axis == "M":
            bm, bn = pd._cdiv(cm, nb), base_blk.block_n_tiles
        else:
            bm, bn = base_blk.block_m_tiles, pd._round_up(pd._cdiv(cn, nb), q_n or grid.q_n)
        sb_h, sb_w = pd.choose_subblock(bm, bn, geom.dest_limit)
        groups = bm // sb_h
        pd.DEPTH_OUT = depth_out if depth_out else groups
        budget = pd.l1_cb_budget(device, [t for t in (a, b, out, row, residual) if t is not None])
        depths = (base_blk.depth_act, base_blk.depth_weight)
        w, bk_spec = w0, bk_spec0
        if w == "max":
            # deepest equal operand depth that fits with the requested block_k (baseline block_k if Kb / auto)
            bk_probe = int(bk_spec) if bk_spec not in (None, "b") else base_blk.block_k_tiles
            bk_probe = pd._k_split(geom.tensor_k_tiles, bk_probe)[0]
            w = max(base_blk.depth_act, 2)
            while pd._footprint(geom, bm, bn, bk_probe, True, depth_weight=w + 1, depth_act=w + 1) <= budget and w < 8:
                w += 1
            bk_spec = str(bk_probe)
        if w:
            depths = (w, w)
        if bk_spec is None:
            bk_max = pd._block_k_max(geom, bm, bn, budget, depths)
        elif bk_spec == "b":
            bk_max = base_blk.block_k_tiles
        else:
            bk_max = int(bk_spec)
        blk = pd._make_blocking(geom, base_blk.block_batches, bm, bn, bk_max, depths)
        foot = pd._footprint(geom, bm, bn, blk.block_k_tiles, True, depth_weight=depths[1], depth_act=depths[0])
        log.update(
            grid=grid,
            base=base_blk,
            blk=blk,
            depth_out=pd.DEPTH_OUT,
            budget=budget,
            footprint=foot,
            nb=(pd._cdiv(cm, bm), pd._cdiv(cn, bn)),
            mode=pd.operand_mode(geom, grid, blk),
            base_mode=pd.operand_mode(geom, grid, base_blk),
        )
        assert foot <= budget, f"forced blocking does not fit: {foot} > {budget}"
        assert log["mode"] == log["base_mode"], f"operand mode changed {log['base_mode']} -> {log['mode']}"
        return geom, grid, blk, gxy

    return plan


def test_multiblock(device, monkeypatch):
    case = os.environ.get("PMB_CASE", "focus")
    variant = os.environ.get("PMB_VARIANT", "base")
    mode = os.environ.get("PMB_MODE", "real")
    inputs, dtype, wdtype, fid, epi = CASES[case]

    log = {}
    spec = _parse(variant)
    if spec is not None:
        monkeypatch.setattr(pd, "plan", _forced_plan(pd.plan, spec, log))
        monkeypatch.setattr(pd, "DEPTH_OUT", pd.DEPTH_OUT)  # restored after the test
    if mode != "real":
        monkeypatch.setenv("GENIFY_ABLATE", mode)  # epi | reads,mcast,writes | ... (comma list)
    else:
        monkeypatch.delenv("GENIFY_ABLATE", raising=False)

    orig_check = helpers.check_output

    def check(ttnn_output, expected, **kw):
        got = ttnn.to_torch(ttnn_output).float().flatten()
        exp = expected.float().flatten()
        pcc = torch.corrcoef(torch.stack([got, exp]))[0, 1].item()
        rms = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
        print(f"\nPMB_RESULT case={case} variant={variant} mode={mode} pcc={pcc:.6f} rel_rms={rms:.5f}")
        if mode == "real":
            return orig_check(ttnn_output, expected, **kw)

    monkeypatch.setattr(helpers, "check_output", check)
    if not log:
        # log the baseline plan too
        orig = pd.plan

        def logging_plan(*a, **k):
            r = orig(*a, **k)
            log.update(grid=r[1], blk=r[2], mode=pd.operand_mode(r[0], r[1], r[2]))
            return r

        monkeypatch.setattr(pd, "plan", logging_plan)
    helpers.run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epi,
        device=device,
        extras={"math_fidelity": fid},
    )
    print(f"\nPMB_PLAN case={case} variant={variant} mode={mode} " + " ".join(f"{k}={v}" for k, v in log.items()))
