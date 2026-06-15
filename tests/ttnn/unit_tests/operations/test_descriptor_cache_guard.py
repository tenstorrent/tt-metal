# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Third axis of the descriptor-cache regression suite (#46506): the PERF / NO-REBUILD guard.

The companion modules (test_descriptor_cache*.py) cover the two CORRECTNESS axes — NOT STALE
(results track per-call data on a cache hit) and NOT OVER-CACHING (entry count stays bounded when
only input DATA varies). Neither of those catches the actual #46506 perf bug directly: a
descriptor op can be perfectly correct AND mint a single cache entry yet STILL re-run
create_descriptor() on every program-cache HIT (because it baked a buffer address as a raw uint32
instead of registering a patchable Buffer* binding). That per-dispatch host cost is what regressed
ResNet50 ~20x on the non-trace path.

The mesh-device-operation adapter has an env-gated guard for exactly this
(ttnn/api/ttnn/mesh_device_operation_adapter.hpp ~624): when
TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT is set, an op that falls to the slow-path rebuild
on a cache HIT raises TT_FATAL instead of silently rebuilding. A TT_FATAL aborts the process, so a
SLOW-PATH op makes the run exit non-zero.

The guard is read ONCE from a namespace-scope `inline const bool ... = std::getenv(...) != nullptr`
initialized at library LOAD time, so the env var must be set BEFORE ttnn is imported/loaded. We
therefore run each op in an ISOLATED SUBPROCESS that sets the env in its own os.environ before importing
ttnn, runs the op TWICE (1st = cache miss, 2nd = cache HIT where the guard fires), prints a success
marker, and exits. The parent test asserts returncode == 0 AND the marker is in stdout (so a clean
exit that skipped the op cannot masquerade as a pass).

This module itself never imports ttnn nor opens a device (a second device opener would crash the
single in-use GPU) — all device work happens in the child process.

Pattern modeled on tests/ttnn/unit_tests/operations/test_descriptor_narrow.py (build a script
string, run via subprocess.run with the env, assert returncode 0 + success marker in stdout).
"""

import subprocess
import sys

import pytest

# Marker the child prints on success; the parent requires it in stdout so a no-op clean exit
# (e.g. the op was skipped/not reached) cannot pass as green.
_SUCCESS_MARKER = "DESCRIPTOR_NO_REBUILD_OK"

# Header shared by every child script: set the guard env BEFORE importing ttnn, open a
# program-cache-enabled device, and provide a _run_twice helper (2nd call = cache hit -> guard
# fires on a slow-path rebuild). The op-specific body is appended per test.
_CHILD_HEADER = f"""
import os
# Must be set before ttnn is imported/loaded: the adapter reads it via getenv into a
# namespace-scope `inline const bool` initialized at library load time.
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import torch
import ttnn


def _tile(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def _rm(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=dtype)


def _run_twice(dev, fn):
    # 1st call: cache miss -> resolve_bindings() validates every declared Buffer* binding.
    # 2nd call: cache HIT -> the guard TT_FATALs (non-zero exit) if the op rebuilds its descriptor.
    out = None
    for _ in range(2):
        out = fn()
        ttnn.synchronize_device(dev)
    return out


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        dev.enable_program_cache()
        _body(dev)
    finally:
        ttnn.close_device(dev)
    print("{_SUCCESS_MARKER}")


"""


# Per-op bodies. Each defines `_body(dev)` running the op via _run_twice; shapes mirror the
# existing descriptor-cache / no-rebuild tests. Kept minimal — the assertion under test is "the
# subprocess exits 0", i.e. no slow-path rebuild on the cache hit.
_OP_BODIES = {
    # eltwise / unary
    "relu": """
def _body(dev):
    a = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.relu(a))
""",
    # eltwise / binary_ng tensor-tensor
    "add_tensor": """
def _body(dev):
    a = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), dev)
    b = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.add(a, b))
""",
    # eltwise / binary_ng tensor-scalar (arith)
    "add_scalar": """
def _body(dev):
    a = _tile(torch.randn(1, 1, 256, 256, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.add(a, 3.0))
""",
    # eltwise / binary_ng tensor-scalar (relational)
    "gt_scalar": """
def _body(dev):
    a = _tile(torch.randint(-100, 100, (1, 1, 256, 256), dtype=torch.int32), dev, dtype=ttnn.int32)
    _run_twice(dev, lambda: ttnn.gt(a, 5))
""",
    # pool / max_pool2d (NHWC row-major flat input)
    "max_pool2d": """
def _body(dev):
    batch, in_h, in_w, ch = 16, 16, 16, 32
    flat = torch.randn(batch, in_h, in_w, ch, dtype=torch.bfloat16).reshape(1, 1, batch * in_h * in_w, ch)
    a = _rm(flat, dev)
    _run_twice(
        dev,
        lambda: ttnn.max_pool2d(
            input_tensor=a,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=ch,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
        ),
    )
""",
    # normalization / softmax
    "softmax": """
def _body(dev):
    a = _tile(torch.randn(1, 1, 64, 256, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.softmax(a, dim=-1))
""",
    # data_movement / transpose (HC tiled)
    "transpose": """
def _body(dev):
    a = _tile(torch.randn(1, 4, 64, 64, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.transpose(a, 1, 2))
""",
    # data_movement / slice (tiled)
    "slice": """
def _body(dev):
    a = _tile(torch.randn(1, 1, 128, 128, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.slice(a, [0, 0, 0, 0], [1, 1, 64, 64]))
""",
    # pool / rotate (NHWC row-major)
    "rotate": """
def _body(dev):
    a = _rm(torch.randn(1, 16, 16, 64, dtype=torch.bfloat16), dev)
    _run_twice(dev, lambda: ttnn.rotate(a, angle=30.0, interpolation_mode="nearest"))
""",
}


def _run_guard_subprocess(op_key):
    script = _CHILD_HEADER + _OP_BODIES[op_key] + "\n\nif __name__ == '__main__':\n    main()\n"
    # Inherit the parent env (PYTHONPATH etc.); the script sets the FORBID env itself before ttnn.
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result


@pytest.mark.parametrize(
    "op_key",
    ["relu", "add_tensor", "add_scalar", "gt_scalar", "max_pool2d", "softmax", "transpose", "slice", "rotate"],
)
def test_descriptor_no_rebuild_guard(op_key):
    """Run `op_key` in an isolated subprocess with the no-rebuild guard armed. A slow-path
    descriptor rebuild on the cache HIT (the #46506 bug) TT_FATALs in the child -> non-zero exit.
    Pass requires exit 0 AND the success marker in stdout."""
    result = _run_guard_subprocess(op_key)
    assert result.returncode == 0, (
        f"{op_key}: descriptor slow-path rebuild on cache hit (or other failure), "
        f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert _SUCCESS_MARKER in result.stdout, (
        f"{op_key}: subprocess exited 0 but did not reach the success marker — the op may have been "
        f"skipped or not dispatched.\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
