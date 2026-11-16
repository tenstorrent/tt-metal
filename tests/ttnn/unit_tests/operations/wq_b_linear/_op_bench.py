# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-only bench for wq_b_linear (run under `scripts/run_safe_pytest.sh --profile`).

WQB_VARIANT selects an ablation: full | no_compute | no_w_read | no_compute_no_w_read | sync_only | w_only | compute_only
(sync_only / w_only / compute_only shrink every x multicast to one page and drops the injector x prefetch).
Ablated variants produce wrong output by design; no correctness check here.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from ttnn.operations.wq_b_linear import wq_b_linear

VARIANT = os.environ.get("WQB_VARIANT", "full")
ITERS = int(os.environ.get("WQB_ITERS", "3"))

_orig_create = pd.create_program_descriptor
_tmp_dir = Path(tempfile.mkdtemp(prefix="wqb_bench_"))


def _gated_reader():
    src = (pd.KERNEL_DIR / "wq_b_linear_reader.cpp").read_text()
    needle = "noc.async_read(w, cb_w, w_page_bytes, {.page_id = row_page + n}, {.offset_bytes = offset});"
    assert needle in src
    src = src.replace(needle, "if (w_addr == 0xFFFFFFFF) { " + needle + " }")
    path = _tmp_dir / "wq_b_linear_reader_no_w_read.cpp"
    path.write_text(src)
    return str(path)


def _tiny_x_writer():
    src = (pd.KERNEL_DIR / "wq_b_linear_writer.cpp").read_text()
    send = "send_pipe->send(slot, slot, round_bytes);"
    read = "noc.async_read(x, cb_x, x_page_bytes, {.page_id = row_page + kk}, {.offset_bytes = offset});"
    assert send in src and read in src
    src = src.replace(send, "send_pipe->send(slot, slot, x_page_bytes);")
    src = src.replace(read, "if (x_addr == 0xFFFFFFFF) { " + read + " }")
    path = _tmp_dir / "wq_b_linear_writer_tiny_x.cpp"
    path.write_text(src)
    return str(path)


def _patched_create(*args, **kwargs):
    desc = _orig_create(*args, **kwargs)
    skip_compute = VARIANT in ("no_compute", "no_compute_no_w_read", "sync_only", "w_only")
    skip_w = VARIANT in ("no_w_read", "no_compute_no_w_read", "sync_only", "compute_only")
    tiny_x = VARIANT in ("sync_only", "w_only", "compute_only")
    kernels = list(desc.kernels)
    for k in kernels:
        if skip_compute and k.kernel_source.endswith("wq_b_linear_compute.cpp"):
            k.defines = [("SKIP_COMPUTE", "1")]
        if skip_w and k.kernel_source.endswith("wq_b_linear_reader.cpp"):
            k.kernel_source = _gated_reader()
        if tiny_x and k.kernel_source.endswith("wq_b_linear_writer.cpp"):
            k.kernel_source = _tiny_x_writer()
    desc.kernels = kernels
    return desc


@pytest.mark.parametrize("variant", [VARIANT])
def test_bench(device, variant, monkeypatch):
    opmod = sys.modules["ttnn.operations.wq_b_linear.wq_b_linear"]
    monkeypatch.setattr(opmod, "create_program_descriptor", _patched_create)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(ITERS):
        y = wq_b_linear(tx, tw)
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
