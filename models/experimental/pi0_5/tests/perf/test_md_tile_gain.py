# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""matmul_decode 16x32 vs 32x32 tile device-kernel gain for the denoise MLP matmuls.

Two parts:
  * test_md_op  — inner op runner: one ttnn.matmul_decode at a given (M, K, N) in the
    full-width mode the model uses (PI0_MD_DENOISE), B width-sharded over n//32 cores
    halved to fit the grid (gate/up 4096->64, down 1024->32). Also checks PCC.
  * test_md_tile_gain — wrapper: runs the inner op under the device profiler for M=32
    and M=16 (per shape), reads DEVICE KERNEL DURATION, and asserts the 16x32 tile is
    faster by >= MIN_GAIN.

Reference (device 9, full-width): gate/up 5359->4498 ns (1.19x), down 11331->8973 ns (1.26x).

Run:
  # gain assertion — plain pytest; run_device_perf wraps the inner op with `python -m tracy`:
  pytest models/experimental/pi0_5/tests/perf/test_md_tile_gain.py::test_md_tile_gain -v --device-id 9
  # inner op alone, profiled — needs tracy to emit device-kernel durations:
  python -m tracy -p -r -m pytest \
      models/experimental/pi0_5/tests/perf/test_md_tile_gain.py::test_md_op --device-id 9
"""
import pytest
import torch
import ttnn

from models.perf.device_perf_utils import run_device_perf
from tests.ttnn.utils_for_testing import assert_with_pcc

# (name, K, N) — Gemma-300M denoise expert MLP matmuls.
MLP_SHAPES = [("gate_up", 1024, 4096), ("down", 4096, 1024)]
NUM_INPUTA_CORES = 2
N_RUNS = 5  # op invocations per profiler run (averaged)
MIN_GAIN = 1.05  # 16x32 must beat 32x32 in device-kernel time (measured ~1.13x gate/up, ~1.25x down)
PCC = 0.99

_THIS = "models/experimental/pi0_5/tests/perf/test_md_tile_gain.py"
# id form is "<name>_m<M>" so -k selects exactly one inner case.
_INNER_IDS = [f"{name}_m{m}" for name, _, _ in MLP_SHAPES for m in (16, 32)]
_INNER_PARAMS = [(name, k, n, m) for name, k, n in MLP_SHAPES for m in (16, 32)]


def _tile_height(m):
    for th in (1, 2, 4, 8, 16, 32):
        if m <= th:
            return th
    return 32


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("name, k, n, m", _INNER_PARAMS, ids=_INNER_IDS)
def test_md_op(device, name, k, n, m):
    """Inner runner: full-width matmul_decode at (M=m, K=k, N=n). Used by the profiler
    wrapper below; also a standalone PCC check."""
    max_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    num_inputB_cores = n // 32
    while num_inputB_cores > max_cores:
        num_inputB_cores //= 2

    torch.manual_seed(0)
    ta = torch.randn((m, k), dtype=torch.bfloat16)
    tb = torch.randn((k, n), dtype=torch.bfloat16)
    ref = ta.to(torch.float32) @ tb.to(torch.float32)

    a_cfg = ttnn.create_sharded_memory_config(
        (m, k // NUM_INPUTA_CORES),
        core_grid=ttnn.num_cores_to_corerangeset(NUM_INPUTA_CORES, device.compute_with_storage_grid_size(), True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    b_cfg = ttnn.create_sharded_memory_config(
        (k, n // num_inputB_cores),
        core_grid=ttnn.num_cores_to_corerangeset(num_inputB_cores, device.compute_with_storage_grid_size(), True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a = ttnn.from_torch(
        ta, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((_tile_height(m), 32)), device=device, memory_config=a_cfg
    )
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, memory_config=b_cfg)

    out = None
    for _ in range(N_RUNS):
        out = ttnn.matmul_decode(a, b)
    assert out.shape == (m, n)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), PCC)


def _kernel_ns(request, inner_id):
    """Run the inner op under the device profiler; return avg DEVICE KERNEL DURATION [ns]."""
    dev_id = request.config.getoption("device_id")
    dev = f" --device-id {dev_id}" if dev_id is not None else ""
    command = f"pytest {_THIS}::test_md_op -k {inner_id}{dev}"
    res = run_device_perf(command, subdir="pi05_md_tile_gain", num_iterations=1, cols=["DEVICE KERNEL"], batch_size=1)
    return res["AVG DEVICE KERNEL DURATION [ns]"]


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("name, k, n", MLP_SHAPES, ids=[s[0] for s in MLP_SHAPES])
def test_md_tile_gain(request, name, k, n):
    """16x32 (M=16) matmul_decode device-kernel time must beat 32x32 (M=32) by >= MIN_GAIN."""
    ns32 = _kernel_ns(request, f"{name}_m32")
    ns16 = _kernel_ns(request, f"{name}_m16")
    gain = ns32 / ns16
    print(f"\n  {name} (K={k} N={n}):  M32(32x32)={ns32:.0f} ns  M16(16x32)={ns16:.0f} ns  gain={gain:.2f}x")
    assert (
        gain >= MIN_GAIN
    ), f"{name}: 16x32 device-kernel gain {gain:.2f}x < {MIN_GAIN}x (M16={ns16:.0f} M32={ns32:.0f})"
