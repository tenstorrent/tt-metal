# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment trid_pipelined_dm: transaction-id pipelined reader/writer vs the op's current DM.

Variant id = "r<R>_w<W>[_<ablation>]":
  R / W = 0 -> the op's own reader / writer kernel (baseline), else the trid kernel with depth R / W.
  ablation: "" full op, "rw" compute stubbed, "ro" read only (writes + compute stubbed),
            "wo" write only (reads + compute stubbed).
Correctness test: bit-exact vs the baseline op output (and vs torch) on aligned + ragged shapes.
Perf test: one profiled launch per variant; labels appended to perf_experiments/trid_pipelined_dm/labels.txt.
Run perf under scripts/run_safe_pytest.sh --profile. Never asserts a perf direction.
"""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

SRC = Path(pd.KERNEL_DIR)
HERE = Path(pd.__file__).parent / "perf_experiments" / "trid_pipelined_dm"
KDIR = HERE / "kernels"
GEN = HERE / "gen_kernels"
LO, HI = -1.0, 1.0


def _gen_baseline():
    """Op's own reader/writer, byte-identical except the ABLATE switch around the NoC call."""
    GEN.mkdir(exist_ok=True)
    r = (SRC / "clamp_square_practice_reader.cpp").read_text()
    r = r.replace(
        "noc_async_read_page(page0 + i, input_accessor, l1_write_addr);",
        "if constexpr (!ABLATE_READ) { noc_async_read_page(page0 + i, input_accessor, l1_write_addr); }",
    )
    r = r.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_READ\n#define ABLATE_READ 0\n#endif")
    w = (SRC / "clamp_square_practice_writer.cpp").read_text()
    w = w.replace(
        "noc_async_write_page(page0 + i, output_accessor, l1_read_addr);",
        "if constexpr (!ABLATE_WRITE) { noc_async_write_page(page0 + i, output_accessor, l1_read_addr); }",
    )
    w = w.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_WRITE\n#define ABLATE_WRITE 0\n#endif")
    assert "ABLATE_READ) {" in r and "ABLATE_WRITE) {" in w
    (GEN / "base_reader.cpp").write_text(r)
    (GEN / "base_writer.cpp").write_text(w)


ABL = {"": (0, 0, 0), "rw": (0, 0, 1), "ro": (0, 1, 1), "wo": (1, 0, 1)}


def run_variant(device, x, shape, rd, wr, abl=""):
    ar, aw, ac = ABL[abl]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = pd.create_program_descriptor(x, out, LO, HI)
    rk, wk, ck = desc.kernels
    block = rk.compile_time_args[1]
    page = rk.compile_time_args[2]
    in_blocks = desc.cbs[0].total_size // (page * block)
    out_blocks = desc.cbs[1].total_size // (page * block)
    if rd:
        rk.kernel_source = str(KDIR / "reader.cpp")
        rk.defines = [("ABLATE_READ", str(ar)), ("TRID_DEPTH", str(rd)), ("CB_BLOCKS", str(in_blocks))]
    else:
        rk.kernel_source = str(GEN / "base_reader.cpp")
        rk.defines = [("ABLATE_READ", str(ar))]
    if wr:
        wk.kernel_source = str(KDIR / "writer.cpp")
        wk.defines = [("ABLATE_WRITE", str(aw)), ("TRID_DEPTH", str(wr)), ("CB_BLOCKS", str(out_blocks))]
    else:
        wk.kernel_source = str(GEN / "base_writer.cpp")
        wk.defines = [("ABLATE_WRITE", str(aw))]
    ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", str(ac))]
    desc.kernels = [rk, wk, ck]
    ttnn.generic_op([x, out], desc)
    return out


def _input(device, shape, seed=0):
    torch.manual_seed(seed)
    t = (torch.randn(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


CORR_SHAPES = [
    (1, 1, 4096, 4096),
    (8, 1, 1024, 1024),
    (2, 4, 256, 512),
    (1, 1, 32, 32),
    (2, 64, 256),
    (1, 1, 32, 8192),
    (1, 1, 256, 256),
    (64, 128),
    (3, 5, 96, 160),
]
CORR_COMBOS = [(2, 0), (3, 0), (4, 0), (0, 2), (0, 3), (0, 4), (2, 2), (3, 3), (4, 4), (4, 2), (2, 4)]


@pytest.mark.parametrize("shape", CORR_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_correctness(device, shape):
    if os.environ.get("TP_PERF_ONLY"):
        pytest.skip("perf only")
    _gen_baseline()
    t, x = _input(device, shape)
    ref = (torch.clamp(t.float(), LO, HI) ** 2).to(torch.bfloat16)
    base = ttnn.to_torch(run_variant(device, x, shape, 0, 0))
    assert torch.equal(base, ref), "baseline != torch"
    for rd, wr in CORR_COMBOS:
        got = ttnn.to_torch(run_variant(device, x, shape, rd, wr))
        assert torch.equal(got, base), f"r{rd}_w{wr} not bit-exact vs baseline on {shape}"
    print(f"CORRECT shape={shape} combos={len(CORR_COMBOS)}")


FOCUS = (1, 1, 4096, 4096)
PERF_CASES = (
    # focus: full op
    [(FOCUS, r, w, "") for r, w in [(0, 0), (2, 0), (3, 0), (4, 0), (0, 2), (0, 3), (0, 4), (2, 2), (3, 3), (4, 4)]]
    # focus: read only / write only / read+write
    + [(FOCUS, r, 0, "ro") for r in (0, 2, 3, 4)]
    + [(FOCUS, 0, w, "wo") for w in (0, 2, 3, 4)]
    + [(FOCUS, r, w, "rw") for r, w in [(0, 0), (2, 2), (3, 3), (4, 4)]]
    # sweep
    + [
        (s, r, w, "")
        for s in [(8, 1, 1024, 1024), (2, 4, 256, 512), (1, 1, 32, 32)]
        for r, w in [(0, 0), (2, 2), (3, 3), (4, 4), (4, 0), (0, 4)]
    ]
)


def _pid(c):
    s, r, w, a = c
    return f"{'x'.join(map(str, s))}-r{r}_w{w}" + (f"_{a}" if a else "")


@pytest.mark.parametrize("case", PERF_CASES, ids=_pid)
def test_perf(device, case):
    flt = os.environ.get("TP_FILTER")
    if flt and not any(f in _pid(case) for f in flt.split(",")):
        pytest.skip("filtered")
    _gen_baseline()
    shape, rd, wr, abl = case
    _, x = _input(device, shape)
    run_variant(device, x, shape, rd, wr, abl)
    ttnn.synchronize_device(device)
    with open(HERE / "labels.txt", "a") as f:
        f.write(_pid(case) + "\n")
