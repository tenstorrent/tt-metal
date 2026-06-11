# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the descriptor-framework slow-path rebuild perf bug (#46506).

This PR keeps only the broadly-validated core on the descriptor cache-hit FAST PATH:
  * move (MULTI_CORE_SHARDED): get_dynamic_runtime_args re-derives the address-derived
    reader chunk args (num_chunks / move_chunk_size_bytes / remainder) on every dispatch.
  * pool  (Pool2D): empty get_dynamic_runtime_args (CB-bound-stable) opts onto the fast
    path with no descriptor rebuild.

The check uses TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1: if either op rebuilds
its descriptor on a program-cache hit, the adapter raises (TT_FATAL). The guard is read
ONCE per process by the C++ adapter, so it MUST be set before ttnn first dispatches and
it leaks to the whole interpreter. To avoid polluting unrelated tests collected in the
same pytest worker (which would FATAL when they legitimately rebuild), each op is run in
an ISOLATED SUBPROCESS with the env var set. The subprocess runs the op 3x with a fresh
(address-varying) input each call and checks PCC; it exits non-zero if the guard fires
(rebuild on cache hit) or the data is corrupted (stale baked address) -- the test asserts
the subprocess succeeded.
"""

import os
import subprocess
import sys

import pytest

_MOVE_SCRIPT = r"""
import os
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"
import torch, ttnn
from tests.ttnn.utils_for_testing import check_with_pcc

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
dev.enable_program_cache()
try:
    torch.manual_seed(0)
    shape = (1, 1, 256, 64)
    shard_shape = (32, 64)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})
    mem_cfg = ttnn.create_sharded_memory_config(
        shard_shape, core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True)
    torch_in = torch.randn(shape, dtype=torch.bfloat16)
    out = None
    for _ in range(3):
        # Fresh interleaved->sharded input each call so the move op's input/output buffer
        # addresses (and thus the baked chunk geometry) change across program-cache hits.
        t = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        t = ttnn.to_memory_config(t, mem_cfg)
        out = ttnn.move(t)
        ttnn.synchronize_device(dev)
    res = ttnn.to_torch(out).float().reshape(shape)
    ok, pcc = check_with_pcc(torch_in.float(), res, 0.9999)
    assert ok, "move_sharded corrupted on cache hit (stale chunk args?): " + str(pcc)
finally:
    ttnn.close_device(dev)
print("MOVE_OK")
"""

_POOL_SCRIPT = r"""
import os
os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"
import torch, ttnn
from tests.ttnn.utils_for_testing import check_with_pcc

N, H, W, C = 16, 16, 16, 32
K, S, P, D = (3, 3), (2, 2), (1, 1), (1, 1)
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
dev.enable_program_cache()
try:
    torch.manual_seed(0)
    nhwc = torch.randn(N, H, W, C, dtype=torch.bfloat16)
    flat = nhwc.reshape(1, 1, N * H * W, C)
    nchw = nhwc.permute(0, 3, 1, 2).float()
    out = None
    for _ in range(3):
        inp = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        out = ttnn.max_pool2d(input_tensor=inp, batch_size=N, input_h=H, input_w=W, channels=C,
                              kernel_size=K, stride=S, padding=P, dilation=D)
        ttnn.synchronize_device(dev)
    oh = (H + 2 * P[0] - D[0] * (K[0] - 1) - 1) // S[0] + 1
    ow = (W + 2 * P[1] - D[1] * (K[1] - 1) - 1) // S[1] + 1
    res = ttnn.to_torch(out).float().reshape(N, oh, ow, C).permute(0, 3, 1, 2)
    ref = torch.nn.functional.max_pool2d(nchw, kernel_size=K, stride=S, padding=P, dilation=D)
    ok, pcc = check_with_pcc(ref, res, 0.999)
    assert ok, "max_pool2d corrupted on cache hit: " + str(pcc)
finally:
    ttnn.close_device(dev)
print("POOL_OK")
"""


def _run_guarded(script, marker):
    """Run script in an isolated subprocess so the process-global guard env var does not
    leak into other tests. Fails if the subprocess errors (guard FATAL or PCC mismatch)."""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0 and marker in proc.stdout, (
        f"guarded subprocess failed (rc={proc.returncode}).\n"
        f"--- stdout tail ---\n{proc.stdout[-2000:]}\n--- stderr tail ---\n{proc.stderr[-3000:]}"
    )


def test_move_sharded_no_rebuild():
    _run_guarded(_MOVE_SCRIPT, "MOVE_OK")


def test_max_pool2d_no_rebuild():
    _run_guarded(_POOL_SCRIPT, "POOL_OK")
