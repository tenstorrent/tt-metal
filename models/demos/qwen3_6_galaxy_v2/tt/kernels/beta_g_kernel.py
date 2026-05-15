# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang authoring script for the fused DeltaNet beta/g kernel.

Math (per qwen36_delta_attention._compute_beta_g):
    beta = sigmoid(b)
    a_biased = a + dt_bias
    sp = softplus(a_biased)  # = log(1 + exp(a_biased))
    A_exp = exp(A_log)
    g = -A_exp * sp

Inputs : b, a, dt_bias_broadcast, A_log_broadcast, ones  (5 tensors, same tile shape)
Outputs: beta, g                                          (2 tensors)

Total 7 tensors. The 6-op TTNN chain on Tenstorrent fans out 6 dispatches; this
kernel issues a single launch (1 compute + 2 DM threads).

Run with `python_env_312/bin/python` to compile and copy the emitted C++ kernels
into ./beta_g/ for the 3.10 ttnn loader.

The 3.12 venv has both tt-lang and tt-lang-sim installed; the sim marker is
present, so the `import ttl` re-export path is sim-only. We bypass that by
importing the underlying module `ttl.ttl` directly, which exposes the hardware
operation/compute/datamovement decorators regardless of the marker.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

# Bypass the sim-only re-export gate in ttl.__init__: bind the hardware
# decorators / helpers onto the package object so source text can refer to
# `ttl.<thing>` and `ttl.math.<thing>` literally (the AST resolver checks
# `_is_ttl_module_access` / `_is_ttl_math_access`, which both inspect the
# textual access pattern -- the name `ttl` must appear in source).
import ttl as _ttl_pkg
from ttl.ttl import compute, copy, datamovement, make_dataflow_buffer_like
from ttl.ttl import math as ttl_math
from ttl.ttl import operation

import ttnn

_ttl_pkg.operation = operation
_ttl_pkg.compute = compute
_ttl_pkg.datamovement = datamovement
_ttl_pkg.make_dataflow_buffer_like = make_dataflow_buffer_like
_ttl_pkg.copy = copy
_ttl_pkg.math = ttl_math
ttl = _ttl_pkg  # noqa: F841 (used in kernel source text)

# Hardware operates on 32x32 tiles.
TILE_SIZE = 32

# Kernel-validation shape: 2x2 tiles per tensor. Real model uses
# `[B=32, T=1, n_v_per_row=6]` which tilizes to `[32, 32, 32]` -- one tile.
# We pick a 2x2 layout to exercise multi-tile iteration without inflating the
# compile budget.
ROWS = 2
COLS = 2

OUT_DIR = Path(__file__).resolve().parent / "beta_g"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@ttl.operation(grid=(1, 1))
def __beta_g_op(
    b: ttnn.Tensor,
    a: ttnn.Tensor,
    dt_bias_b: ttnn.Tensor,
    A_log_b: ttnn.Tensor,
    ones: ttnn.Tensor,
    beta: ttnn.Tensor,
    g: ttnn.Tensor,
):
    rows = b.shape[0] // TILE_SIZE
    cols = b.shape[1] // TILE_SIZE

    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    dt_dfb = ttl.make_dataflow_buffer_like(dt_bias_b, shape=(1, 1), block_count=2)
    al_dfb = ttl.make_dataflow_buffer_like(A_log_b, shape=(1, 1), block_count=2)
    ones_dfb = ttl.make_dataflow_buffer_like(ones, shape=(1, 1), block_count=2)
    beta_dfb = ttl.make_dataflow_buffer_like(beta, shape=(1, 1), block_count=2)
    g_dfb = ttl.make_dataflow_buffer_like(g, shape=(1, 1), block_count=2)

    @ttl.compute()
    def beta_g_compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    b_dfb.wait() as b_blk,
                    a_dfb.wait() as a_blk,
                    dt_dfb.wait() as dt_blk,
                    al_dfb.wait() as al_blk,
                    ones_dfb.wait() as ones_blk,
                    beta_dfb.reserve() as beta_blk,
                    g_dfb.reserve() as g_blk,
                ):
                    # beta = sigmoid(b)
                    beta_blk.store(ttl.math.sigmoid(b_blk))

                    # softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
                    a_biased = a_blk + dt_blk
                    sp = ttl.math.log(ones_blk + ttl.math.exp(a_biased))

                    # g = -exp(A_log) * sp
                    A_exp = ttl.math.exp(al_blk)
                    g_blk.store(ttl.math.neg(A_exp) * sp)

    @ttl.datamovement()
    def beta_g_read():
        for r in range(rows):
            for c in range(cols):
                with (
                    b_dfb.reserve() as b_blk,
                    a_dfb.reserve() as a_blk,
                    dt_dfb.reserve() as dt_blk,
                    al_dfb.reserve() as al_blk,
                    ones_dfb.reserve() as ones_blk,
                ):
                    tx_b = ttl.copy(b[r, c], b_blk)
                    tx_a = ttl.copy(a[r, c], a_blk)
                    tx_dt = ttl.copy(dt_bias_b[r, c], dt_blk)
                    tx_al = ttl.copy(A_log_b[r, c], al_blk)
                    tx_ones = ttl.copy(ones[r, c], ones_blk)
                    tx_b.wait()
                    tx_a.wait()
                    tx_dt.wait()
                    tx_al.wait()
                    tx_ones.wait()

    @ttl.datamovement()
    def beta_g_write():
        for r in range(rows):
            for c in range(cols):
                with beta_dfb.wait() as beta_blk, g_dfb.wait() as g_blk:
                    tx_beta = ttl.copy(beta_blk, beta[r, c])
                    tx_g = ttl.copy(g_blk, g[r, c])
                    tx_beta.wait()
                    tx_g.wait()


def _host_tile_zeros(shape, dtype=torch.bfloat16):
    """Build a host-resident tilized ttnn tensor without opening a device."""
    return ttnn.from_torch(
        torch.zeros(shape, dtype=dtype),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def main():
    # Compile-only: do not attempt to launch on a device.
    os.environ.setdefault("TTLANG_COMPILE_ONLY", "1")
    # Emit a ready-to-use Python runner alongside the C++ kernels.
    runner_path = str(OUT_DIR / "_runner_emitted.py")
    os.environ["TTLANG_EMIT_RUNNER"] = runner_path

    shape = (ROWS * TILE_SIZE, COLS * TILE_SIZE)
    b = _host_tile_zeros(shape)
    a = _host_tile_zeros(shape)
    dt = _host_tile_zeros(shape)
    al = _host_tile_zeros(shape)
    ones = _host_tile_zeros(shape)
    beta = _host_tile_zeros(shape)
    g = _host_tile_zeros(shape)

    # Trigger compilation. The decorator caches the result on the wrapper.
    __beta_g_op(b, a, dt, al, ones, beta, g)

    # The tt-lang pipeline writes the emitted C++ files under
    # `/tmp/$USER/ttlang_kernel_<name>_<hash>.cpp` and logs each path as
    # `=== <name> kernel written to <path> ===`. We tee them into the
    # repo-local beta_g/ directory for the 3.10 runner.
    user = os.environ.get("USER", "default")
    tmp_dir = Path(f"/tmp/{user}")
    if not tmp_dir.exists():
        print(f"ERROR: expected kernels in {tmp_dir}, directory missing.")
        sys.exit(2)

    expected_threads = {"beta_g_compute", "beta_g_read", "beta_g_write"}
    found = {}
    for path in tmp_dir.glob("ttlang_kernel_*.cpp"):
        name = path.name
        # ttlang_kernel_<name>_<hash>.cpp -> strip prefix + suffix
        stem = name[len("ttlang_kernel_") : -len(".cpp")]
        # `_<hash>` is exactly 9 trailing chars (underscore + 8 hex)
        thread_name = stem[:-9]
        if thread_name in expected_threads:
            # Pick newest for each thread name (mtime-based).
            prev = found.get(thread_name)
            if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
                found[thread_name] = path

    if not expected_threads.issubset(found):
        missing = expected_threads - set(found)
        print(f"ERROR: missing emitted kernels for: {missing}")
        print(f"  found: {sorted(found)}")
        sys.exit(3)

    for thread_name, path in found.items():
        dest = OUT_DIR / f"{thread_name}.cpp"
        shutil.copy(path, dest)
        size = dest.stat().st_size
        print(f"  copied {path.name} -> {dest} ({size} bytes)")

    print(f"OK -- 3 kernels written to {OUT_DIR}")


if __name__ == "__main__":
    main()
