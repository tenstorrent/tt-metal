# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Stage bench for the strips path's per-wave YUV readback on the 4x8 mesh, no weights: today's three uint8 shards with
28-byte pages ((1, 192, 168, 28) + 2 x (1, 96, 84, 28) per device) against the same bytes with wide pages, one concatenated
read, and preallocated host tensors; plus the device-side share (reads queued behind a ~50 ms dummy op).
    pytest models/tt_dit/tests/models/minimax_h3/tools/yuv_d2h_bench.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

MESH = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8",
    )
]
T = 28
Y_SHAPE, UV_SHAPE = (1, 192, 168, T), (1, 96, 84, T)


def _timed(mesh_device, fn, n=20):
    fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return best * 1e3


def _rep(mesh_device, shape):
    t = torch.randint(0, 255, shape, dtype=torch.uint8)
    return ttnn.from_torch(
        t, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _read3(mesh_device, tensors):
    hosts = [t.cpu(blocking=False) for t in tensors]
    ttnn.synchronize_device(mesh_device)
    return hosts


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_yuv_d2h_bench(mesh_device):
    torch.manual_seed(0)
    today = [_rep(mesh_device, Y_SHAPE), _rep(mesh_device, UV_SHAPE), _rep(mesh_device, UV_SHAPE)]
    wide = [_rep(mesh_device, (1, Y_SHAPE[1], Y_SHAPE[2] * T)), _rep(mesh_device, (1, UV_SHAPE[1], UV_SHAPE[2] * T)), _rep(mesh_device, (1, UV_SHAPE[1], UV_SHAPE[2] * T))]
    total = Y_SHAPE[1] * Y_SHAPE[2] * T + 2 * UV_SHAPE[1] * UV_SHAPE[2] * T
    one = [_rep(mesh_device, (1, 1, total))]
    results = {}
    results["today 3 reads, 28 B pages"] = _timed(mesh_device, lambda: _read3(mesh_device, today))
    results["wide pages, 3 reads"] = _timed(mesh_device, lambda: _read3(mesh_device, wide))
    results["one concatenated read"] = _timed(mesh_device, lambda: _read3(mesh_device, one))

    def _reshape_then_read():
        wide_now = [ttnn.reshape(t, (1, t.shape[1], t.shape[2] * t.shape[3])) for t in today]
        _read3(mesh_device, wide_now)
        for t in wide_now:
            ttnn.deallocate(t)

    results["device reshape to wide, then 3 reads (the MINIMAX_H3_YUV_WIDE path)"] = _timed(mesh_device, _reshape_then_read)
    try:
        hosts = [ttnn.allocate_tensor_on_host(t.spec, mesh_device) for t in wide]

        def _prealloc():
            for d, h in zip(wide, hosts):
                ttnn.copy_device_to_host_tensor(d, h, blocking=False)
            ttnn.synchronize_device(mesh_device)

        results["wide pages, preallocated hosts"] = _timed(mesh_device, _prealloc)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"YUVD2H preallocated-host variant unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    # Device-side share: queue the reads behind a dummy op so the host-side work overlaps it.
    a = ttnn.from_torch(torch.randn(1, 4096, 4096) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    b = ttnn.from_torch(torch.randn(4096, 4096) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

    def _dummy():
        for _ in range(12):
            out = ttnn.matmul(a, b)
            ttnn.deallocate(out)

    t_dummy = _timed(mesh_device, _dummy, n=5)

    def _dummy_then_read(tensors):
        _dummy()
        _read3(mesh_device, tensors)

    t_both_today = _timed(mesh_device, lambda: _dummy_then_read(today), n=5)
    t_both_wide = _timed(mesh_device, lambda: _dummy_then_read(wide), n=5)
    results["device-side share, today (behind dummy)"] = t_both_today - t_dummy
    results["device-side share, wide (behind dummy)"] = t_both_wide - t_dummy
    logger.info(f"YUVD2H dummy op alone: {t_dummy:.2f} ms (bytes per device per wave: {total / 1e6:.2f} MB)")
    for name, ms in results.items():
        logger.info(f"YUVD2H {name}: {ms:.2f} ms")
