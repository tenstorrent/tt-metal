# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated swiglu-epilogue bake-off (perf experiment, not a unit test of the op).

    SWG_SET=focus scripts/run_safe_pytest.sh --profile <this file>
Each case appends a JSON line (config + precision) to results_<set>.jsonl in the experiment dir, in
execution order; the profiler's ops CSV rows (one GenericOp per case, same order) carry the ns.
Correctness gate: PCC vs torch (silu(g) * u in fp64 from the stored inputs) > 0.999 -- perf is not asserted.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

EXP_DIR = (
    Path(__file__).resolve().parents[7] / "ttnn/ttnn/operations/genify_matmul/perf_experiments/swiglu_epilogue_sfpu"
)
_spec = importlib.util.spec_from_file_location("swg_bench", EXP_DIR / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SET = os.environ.get("SWG_SET", "focus")
F = dict(block_m=6, block_n=28)  # the focus block (R1a, 6 x 28 B-tiles -> 6 x 14 outputs)

CASES = {
    # focus shape: every variant, bf16 out, fp32 DEST, the op's block_size 4 (clamps to 2 lanes)
    "focus": [dict(F, variant=v) for v in ("baseline", "fused", "fast3", "fast2", "fast3_nr0", "floor")],
    "focus_u": [dict(F, variant=v) for v in ("baseline", "fusedU", "fast3U", "fast2U", "fast3_nr0U")],
    "sym": [dict(F, variant=v) for v in ("sym3U", "sym2U")],
    "symh": [dict(F, variant=v) for v in ("symh1U", "symh2U")],
    "sp": [dict(F, variant=v) for v in ("spU",)],
    "spi": [dict(F, variant=v) for v in ("spiU",)],
    "spi2": [dict(F, variant=v) for v in ("spi2U",)],
    "spik": [dict(F, variant=v) for v in ("spikU",)],
    "spi2k": [dict(F, variant=v) for v in ("spi2kU",)],
    # domain sweep: baseline vs the recommended spikU (order alternates per pair)
    "domain": [
        dict(F, variant="baseline", natural_pairs=True),
        dict(F, variant="spikU", natural_pairs=True),
        dict(F, variant="spikU", fp32_dest=False),
        dict(F, variant="baseline", fp32_dest=False),
        dict(F, variant="baseline", out_dtype="bf8"),
        dict(F, variant="spikU", out_dtype="bf8"),
        dict(block_m=1, block_n=2, variant="spikU"),
        dict(block_m=1, block_n=2, variant="baseline"),
        dict(block_m=2, block_n=8, variant="baseline"),
        dict(block_m=2, block_n=8, variant="spikU"),
        # fp32 output: the op runs the fp32-precision sigmoid; candidate = the same math fused
        dict(F, variant="fusedU", out_dtype="fp32"),
        dict(F, variant="baseline", out_dtype="fp32"),
        dict(F, variant="spikU", out_dtype="fp32"),
    ],
    "fp32cmp": [  # identical data (seed 0): precision comparison for fp32 outputs
        dict(F, variant="baseline", out_dtype="fp32"),
        dict(F, variant="fusedU", out_dtype="fp32"),
        dict(F, variant="spi4U", out_dtype="fp32"),
        dict(F, variant="spikU", out_dtype="fp32"),
    ],
    "fp32out": [
        dict(F, variant="spi4U", out_dtype="fp32"),
        dict(F, variant="spi4U"),
        dict(F, variant="spi4U", out_dtype="fp32", fp32_dest=False),
        dict(F, variant="baseline", out_dtype="fp32", fp32_dest=False),
    ],
    # the packaged header (graduation artifact): bf16 focus, fp32 out, 16-bit DEST, R4 natural pairs, UTD
    "grad": [
        dict(F, variant="grad"),
        dict(F, variant="grad", out_dtype="fp32"),
        dict(F, variant="grad", fp32_dest=False),
        dict(F, variant="grad", natural_pairs=True),
        dict(F, variant="grad", unpack_to_dest=True),
    ],
    # precision stress: heavy-negative / wide gate (g ~ N(-4, 8^2)), plus the specials
    "stress": [
        dict(F, variant="baseline", scale=8.0, shift=-4.0),
        dict(F, variant="spikU", scale=8.0, shift=-4.0),
        dict(F, variant="spi2kU", scale=8.0, shift=-4.0),
        dict(F, variant="baseline", scale=1.0),
        dict(F, variant="spikU", scale=1.0),
        dict(F, variant="spi2kU", scale=1.0),
    ],
    "focus_rev": [dict(F, variant=v) for v in ("floor", "fast3_nr0", "fast2", "fast3", "fused", "baseline")],
    # implementation knobs around the winner
    "knobs": [
        dict(F, variant="baseline", unpack_to_dest=True),
        dict(F, variant="sym3U", unpack_to_dest=True),
        dict(F, variant="sym2U", unpack_to_dest=True),
        dict(F, variant="floor", unpack_to_dest=True),
        dict(F, variant="sym3U", block_size=1),
        dict(F, variant="sym3U", block_size=2),
    ],
}


def _cases():
    return CASES[SET]


@pytest.mark.parametrize("idx", range(len(_cases())))
def test_swiglu_epilogue(device, idx):
    case = dict(_cases()[idx])
    case.setdefault("out_dtype", "bf16")
    case.setdefault("fp32_dest", True)
    case.setdefault("natural_pairs", False)
    case.setdefault("block_size", 4)
    case.setdefault("unpack_to_dest", False)
    out_dtype = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[case["out_dtype"]]
    bm, bn = case["block_m"], case["block_n"]
    host, _, _ = bench.make_block(
        bm,
        bn,
        case["natural_pairs"],
        seed=case.get("seed", 0),
        scale=case.get("scale", 3.0),
        shift=case.get("shift", 0.0),
    )
    kw = {k: v for k, v in case.items() if k not in ("out_dtype", "block_m", "block_n", "variant", "scale", "shift")}
    out = bench.run(device, host, block_m=bm, block_n=bn, variant=case["variant"], out_dtype=out_dtype, **kw)
    got = ttnn.to_torch(out).float()
    # reference from the inputs as stored on device (bf16 partials when fp32_dest is off)
    stored = host if case["fp32_dest"] else host.to(torch.bfloat16).float()
    half = bn // 2
    t = stored.double().reshape(bm * 32, bn, 32)
    if case["natural_pairs"]:
        gate, up = t[:, 0::2, :], t[:, 1::2, :]
    else:
        gate, up = t[:, :half, :], t[:, half:, :]
    ref = (torch.nn.functional.silu(gate) * up).reshape(bm * 32, half * 32)
    m = bench.metrics(got, ref) if case["variant"] != "floor" else {}
    rec = dict(set=SET, idx=idx, **case, **m)
    with open(EXP_DIR / f"results_{SET}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("SWG_RESULT", json.dumps(rec))
    if case["variant"] != "floor":
        assert m["pcc"] > 0.999, m
