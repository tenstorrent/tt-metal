# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf bake-off tail_writes_both_nocs: split the last block's output writes between the writer (BRISC /
NoC1) and the idle reader (NCRISC / NoC0).

Runs a COPY of the op (perf_experiments/tail_writes_both_nocs/pd_copy.py + kernels/) by swapping the op's
create_program_descriptor; the real op is untouched. Every variant in TWBN_MODES runs on the SAME inputs in
one process; the outputs must be bit-identical to the first variant's (correctness gate), and the first
variant is also checked against torch by the golden helper (real mode only).

Env:
  TWBN_CASE   case name (CASES)
  TWBN_MODES  comma list of variants, each "<mode>[:<pct>]" (mode 0 off, 1 column cut with pct % on the
              reader, 2 tile alternation, 3 row alternation inside a writer batch), e.g. "0,1:50,2"
  TWBN_ABL    GENIFY_ABLATE comma list for ablated runs ("" = real; e.g. "epi" or "reads,mcast,matmul,epi")
  TWBN_WB     optional OUT_WRITE_BATCH_ROW_GROUPS override (all variants)
Skipped unless GENIFY_MATMUL_PERF=1.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

import eval.golden_tests.genify_matmul.helpers as helpers
import ttnn.operations.genify_matmul.genify_matmul  # noqa: F401
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as pd_real

op_mod = sys.modules["ttnn.operations.genify_matmul.genify_matmul"]
pytestmark = pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench")

ROOT = Path(pd_real.__file__).resolve().parents[4]
EXP = ROOT / "ttnn/ttnn/operations/genify_matmul/perf_experiments/tail_writes_both_nocs"

_BF16, _BFP4, _BFP8 = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b
_LOFI, _HIFI2, _HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4

# name -> (inputs, dtype, weight_dtype, fidelity, epilogue, fp32_dest, extras)
CASES = {
    "focus": (((1, 2048, 5120), (5120, 8704)), _BF16, _BFP4, _LOFI, "swiglu", True, {}),
    "q4160": (((1, 2048, 5120), (5120, 4160)), _BF16, _BFP4, _LOFI, "none", True, {}),
    "fx_addcmul": (((1, 2048, 6144), (6144, 3072)), _BF16, _BF16, _HIFI2, "addcmul", True, {}),
    "fx_ff1_txt": (((1, 256, 6144), (6144, 18432)), _BF16, _BF16, _HIFI2, "swiglu", True, {}),
    "pdl_h": (((1, 32, 512, 128), (1, 32, 128, 1024)), _BFP8, _BFP8, _LOFI, "none", False, {"weight_buffer": "l1"}),
    "pdl_w_l1": (
        ((1, 32, 128, 256), (256, 1024)),
        _BFP8,
        _BFP8,
        _LOFI,
        "none",
        False,
        {"act_buffer": "l1", "out_buffer": "l1"},
    ),
    "sq4096": (((1, 4096, 4096), (4096, 4096)), _BF16, _BF16, _HIFI4, "none", True, {}),
}


def _load_copy():
    """TWBN_PD=grad loads the graduation candidate (grad/: graduate_tail_split.patch applied to the op, no
    ablation hooks, mode 1 only); default: the experiment copy (all modes + the ablation hooks)."""
    grad = os.environ.get("TWBN_PD") == "grad"
    path = EXP / "grad" / "pd_grad.py" if grad else EXP / "pd_copy.py"
    spec = importlib.util.spec_from_file_location("twbn_pd_copy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["twbn_pd_copy"] = mod
    spec.loader.exec_module(mod)
    assert mod.KERNEL_DIR == path.parent / "kernels"
    return mod


def _maps():
    """Per-core (physical x, y) -> (T_noc1_all, T_noc0_all) write-tail times (us) of the focus case."""
    t1, t0, cur = {}, {}, None
    xs = [1, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15]
    for line in open(EXP / "maps_focus_epi.txt"):
        if line.startswith("noc1_all"):
            cur = t1
        elif line.startswith("noc0_all"):
            cur = t0
        elif line.startswith("y=") and cur is not None:
            parts = line.split()
            y = int(parts[1]) if parts[0] == "y=" else int(parts[0][2:])
            vals = parts[2:] if parts[0] == "y=" else parts[1:]
            for x, v in zip(xs, vals):
                cur[(x, y)] = float(v) - 345.85
    return t1, t0


def _policy(name, gx, gy):
    """Mode-4 per-core ncut policies: (phys_x, phys_y, lx, ly, out_block_n) -> reader (NoC0) tiles per row."""
    if name.startswith("oracle"):
        t1, t0 = _maps()
        xs = [1, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15]  # profiler (physical NoC0) columns, logical order

        def f(px, py, lx, ly, nb):
            c = (xs[lx], ly + 2)
            return round(nb * t1[c] / (t1[c] + t0[c]))

        return f
    if name.startswith("lin"):  # lin<lo>_<hi> : f0 from lo % (logical origin) to hi % (far corner)
        lo, hi = (int(v) for v in name[3:].split("_"))

        def f(px, py, lx, ly, nb):
            u = (lx / max(gx - 1, 1) + ly / max(gy - 1, 1)) / 2
            return round(nb * (lo + (hi - lo) * u) / 100)

        return f
    raise ValueError(name)


def test_tail_split(device, monkeypatch):
    case = os.environ.get("TWBN_CASE", "focus")
    modes = [m for m in os.environ.get("TWBN_MODES", "0,1:50").split(",") if m]
    abl = os.environ.get("TWBN_ABL", "")
    inputs, dtype, wdtype, fid, epi, acc, extras = CASES[case]
    mod = _load_copy()
    if os.environ.get("TWBN_WB"):
        mod.OUT_WRITE_BATCH_ROW_GROUPS = int(os.environ["TWBN_WB"])
    monkeypatch.setattr(op_mod, "create_program_descriptor", mod.create_program_descriptor)
    if abl:
        monkeypatch.setenv("GENIFY_ABLATE", abl)
    else:
        monkeypatch.delenv("GENIFY_ABLATE", raising=False)

    orig_check = helpers.check_output
    outs = {}
    plan_log = {}
    orig_plan = mod.plan

    def plan(*a, **k):
        r = orig_plan(*a, **k)
        plan_log["plan"] = f"grid={r[1]} blk={r[2]} mode={mod.operand_mode(r[0], r[1], r[2])}"
        return r

    mod.plan = plan
    for i, m in enumerate(modes):
        mode, _, pct = m.partition(":")
        mod.TAIL_SPLIT_MODE = int(mode)
        if mode == "4":
            g = device.compute_with_storage_grid_size()
            mod.TAIL_NCUT_FN = _policy(pct, g.x, g.y)
        else:
            mod.TAIL_NC_PCT = int(pct) if pct else 50

        def check(ttnn_output, expected, _m=m, _i=i, **kw):
            got = ttnn.to_torch(ttnn_output)
            outs[_m] = got
            g, e = got.float().flatten(), expected.float().flatten()
            pcc = torch.corrcoef(torch.stack([g, e]))[0, 1].item()
            print(f"\nTWBN_RESULT case={case} variant={_m} abl={abl or 'real'} order={_i} pcc={pcc:.6f}")
            if not abl and _i == 0:
                orig_check(ttnn_output, expected, **kw)

        monkeypatch.setattr(helpers, "check_output", check)
        if hasattr(device, "clear_program_cache"):
            device.clear_program_cache()
        helpers.run_genify_matmul(
            inputs,
            dtype=dtype,
            weight_dtype=wdtype,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=acc,
            placement="interleaved",
            epilogue=epi,
            device=device,
            extras={"math_fidelity": fid, **extras},
        )
        ttnn.synchronize_device(device)
    print(f"\nTWBN_PLAN case={case} {plan_log.get('plan')}")
    ref = outs[modes[0]]
    for m in modes[1:]:
        same = torch.equal(outs[m], ref)
        print(f"\nTWBN_EXACT case={case} variant={m} vs {modes[0]} bit_identical={same}")
        if not abl:  # ablated outputs are garbage by design (stubbed compute); only real runs are gated
            assert same, f"variant {m} differs from {modes[0]}"
