# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 breakdown bench: whole-op + cumulative ablation of clamp_square_practice.

Variants (payload stubbed, sync kept; trip/transaction counts identical):
  full        - production kernels
  no_compute  - CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1
  no_comp_wr  - + writer NoC writes stubbed
  sync_only   - + reader NoC reads stubbed (everything stubbed at once)
  no_read     - reader NoC reads stubbed only
  no_write    - writer NoC writes stubbed only
Perf only; never asserts PCC. Run under scripts/run_safe_pytest.sh --profile.
"""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

HERE = Path(pd.__file__).parent / "perf_experiments" / "breakdown"
SRC = Path(pd.KERNEL_DIR)
GEN = HERE / "gen_kernels"


def _gen():
    GEN.mkdir(exist_ok=True)
    r = (SRC / "clamp_square_practice_reader.cpp").read_text()
    for call in (
        "noc_async_read_page(page, input_accessor, l1_write_addr);",
        "noc_async_read_page(page0 + i, input_accessor, l1_write_addr);",
    ):  # pre-Perf-2 reader
        r = r.replace(call, "if constexpr (!ABLATE_READ) { " + call + " }")
    r = r.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_READ\n#define ABLATE_READ 0\n#endif")
    (GEN / "reader.cpp").write_text(r)
    w = (SRC / "clamp_square_practice_writer.cpp").read_text()
    for call in (
        "noc_async_write_page(page, output_accessor, l1_read_addr);",
        "noc_async_write_page(page0 + i, output_accessor, l1_read_addr);",
    ):  # pre-Perf-2 writer
        w = w.replace(call, "if constexpr (!ABLATE_WRITE) { " + call + " }")
    w = w.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_WRITE\n#define ABLATE_WRITE 0\n#endif")
    (GEN / "writer.cpp").write_text(w)
    assert "ABLATE_READ) {" in r and "ABLATE_WRITE) {" in w


VARIANTS = {
    "full": (0, 0, 0),
    "baseline": (0, 0, 0),  # pre-Perf-1 op: helper Clamp->Square compute kernel + 2-block CBs
    "no_read": (1, 0, 0),
    "no_write": (0, 1, 0),
    "no_compute": (0, 0, 1),
    "no_comp_wr": (0, 1, 1),
    "sync_only": (1, 1, 1),
    "compute_only": (1, 1, 0),
    "write_only": (1, 0, 1),
}
SHAPES = [
    (1, 1, 4096, 4096),
    (8, 1, 1024, 1024),
    (2, 4, 256, 512),
    (1, 1, 32, 32),
    (1, 1, 32, 8192),
    (1, 1, 256, 256),
    (64, 128),
    (2, 64, 256),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("rep", range(int(os.environ.get("CS_REPS", "1"))))
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_breakdown(device, shape, variant, rep):
    # Env filters (run_safe_pytest word-splits -k): CS_VARIANTS=full,no_read  CS_SHAPES=1x1x4096x4096
    vs, ss = os.environ.get("CS_VARIANTS"), os.environ.get("CS_SHAPES")
    if (vs and variant not in vs.split(",")) or (ss and "x".join(map(str, shape)) not in ss.split(",")):
        pytest.skip("filtered")
    _gen()
    ar, aw, ac = VARIANTS[variant]
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16) * 2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    depth = {"input_depth_blocks": 2, "output_depth_blocks": 2} if variant == "baseline" else {}
    desc = pd.create_program_descriptor(x, out, -1.0, 1.0, **depth)
    rk, wk, ck = desc.kernels
    rk.kernel_source = str(GEN / "reader.cpp")
    wk.kernel_source = str(GEN / "writer.cpp")
    rk.defines = [("ABLATE_READ", str(ar))]
    wk.defines = [("ABLATE_WRITE", str(aw))]
    ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", str(ac))]
    if variant == "baseline":
        ck.kernel_source = str(HERE / "baseline_compute.cpp")
    desc.kernels = [rk, wk, ck]
    ttnn.generic_op([x, out], desc)
    ttnn.synchronize_device(device)
    print(f"BENCH_ROW variant={variant} shape={shape}")
