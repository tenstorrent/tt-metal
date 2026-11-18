# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-only bench for wq_b_linear (run under `scripts/run_safe_pytest.sh --profile`).

WQB_VARIANT selects an ablation: full | no_compute | no_w_read | no_compute_no_w_read | sync_only | w_only | compute_only
| mcast_only (full-size x multicast of unread slots, no x prefetch, no W, no compute)
| w_compute (W reads + compute; x multicast shrunk to 1 page/round, no x prefetch)
| no_y_write (full op minus the y store NoC writes)
(sync_only / w_only / compute_only shrink every x multicast to one page and drops the injector x prefetch).
Ablated variants produce wrong output by design; no correctness check here.
WQB_KNOBS overrides program-descriptor knobs, e.g. WQB_KNOBS="X_ROUND_K_BLOCKS=2,X_MCAST_PROTOCOL=chain_flag".
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

for _kv in filter(None, os.environ.get("WQB_KNOBS", "").split(",")):
    _k, _v = _kv.split("=")
    setattr(pd, _k, type(getattr(pd, _k))(_v) if not isinstance(getattr(pd, _k), bool) else _v == "True")

_orig_create = pd.create_program_descriptor
_tmp_dir = Path(tempfile.mkdtemp(prefix="wqb_bench_"))


_W_READ_HEADER = "ttnn/ttnn/operations/wq_b_linear/kernels/wq_b_linear_w_read.hpp"


def _gated_w_header():
    src = (pd.KERNEL_DIR / "wq_b_linear_w_read.hpp").read_text()
    for needle in (
        "noc.async_read(PrecomposedUnicastEndpoint{}, cb_w, bytes, {.noc_addr = src}, {.offset_bytes = offset});",
        "noc.async_read(w, cb_w, bytes, {.page_id = page_id}, {.offset_bytes = offset});",
    ):
        assert needle in src
        src = src.replace(needle, "if (page_id == 0xFFFFFFFF) { " + needle + " }")
    path = _tmp_dir / "wq_b_linear_w_read_gated.hpp"
    path.write_text(src)
    return str(path)


def _with_gated_w(src):
    assert _W_READ_HEADER in src
    return src.replace(_W_READ_HEADER, _gated_w_header())


def _gated_reader():
    src = _with_gated_w((pd.KERNEL_DIR / "wq_b_linear_reader.cpp").read_text())
    path = _tmp_dir / "wq_b_linear_reader_no_w_read.cpp"
    path.write_text(src)
    return str(path)


def _tiny_x_writer(shrink_send=True, gate_w=False):
    src = (pd.KERNEL_DIR / "wq_b_linear_writer.cpp").read_text()
    if gate_w:
        src = _with_gated_w(src)
    send = "send_pipe.send(slot, slot, round_bytes);"
    read = "noc.async_read(x, cb_x, x_page_bytes, {.page_id = row_page + kk}, {.offset_bytes = offset});"
    assert send in src and read in src
    if shrink_send:
        src = src.replace(send, "send_pipe.send(slot, slot, x_page_bytes);")
    src = src.replace(read, "if (x_addr == 0xFFFFFFFF) { " + read + " }")
    path = _tmp_dir / f"wq_b_linear_writer_tiny_x_{int(shrink_send)}_{int(gate_w)}.cpp"
    path.write_text(src)
    return str(path)


_Y_STORE_HEADER = "ttnn/ttnn/operations/wq_b_linear/kernels/wq_b_linear_y_store.hpp"


def _no_y_write_kernel(name):
    hdr = (pd.KERNEL_DIR / "wq_b_linear_y_store.hpp").read_text()
    write = "            noc.async_write(\n                cb_y,"
    assert write in hdr
    gated_hdr = _tmp_dir / "wq_b_linear_y_store_gated.hpp"
    gated_hdr.write_text(hdr.replace(write, "            if (m == 0xFFFFFFFF) noc.async_write(\n                cb_y,"))
    src = (pd.KERNEL_DIR / name).read_text()
    assert _Y_STORE_HEADER in src
    path = _tmp_dir / name.replace(".cpp", "_no_y_write.cpp")
    path.write_text(src.replace(_Y_STORE_HEADER, str(gated_hdr)))
    return str(path)


def _patched_create(*args, **kwargs):
    desc = _orig_create(*args, **kwargs)
    skip_compute = VARIANT in ("no_compute", "no_compute_no_w_read", "sync_only", "w_only")
    skip_w = VARIANT in ("no_w_read", "no_compute_no_w_read", "sync_only", "compute_only")
    tiny_x = VARIANT in ("sync_only", "w_only", "compute_only", "w_compute")
    no_x_prefetch = VARIANT == "mcast_only"
    if VARIANT == "mcast_only":
        skip_compute = skip_w = True
    kernels = list(desc.kernels)
    for k in kernels:
        if skip_compute and k.kernel_source.endswith("wq_b_linear_compute.cpp"):
            k.defines = [("SKIP_COMPUTE", "1")]
        if skip_w and k.kernel_source.endswith("wq_b_linear_reader.cpp"):
            k.kernel_source = _gated_reader()
        if k.kernel_source.endswith("wq_b_linear_writer.cpp"):
            if tiny_x:
                k.kernel_source = _tiny_x_writer(gate_w=skip_w)
            elif no_x_prefetch:
                k.kernel_source = _tiny_x_writer(shrink_send=False, gate_w=skip_w)
            elif skip_w:
                src = _with_gated_w(Path(k.kernel_source).read_text())
                path = _tmp_dir / "wq_b_linear_writer_no_w_read.cpp"
                path.write_text(src)
                k.kernel_source = str(path)
        if VARIANT == "no_y_write":
            for name in ("wq_b_linear_writer.cpp", "wq_b_linear_reader.cpp"):
                if k.kernel_source.endswith(name):
                    k.kernel_source = _no_y_write_kernel(name)
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
