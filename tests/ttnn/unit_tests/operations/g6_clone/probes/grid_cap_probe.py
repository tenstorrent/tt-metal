# Verifier probe: grid-occupancy lamp on p1. Caps the grid the plan splits over (rows x cols of the
# compute grid) and reads device kernel time per variant under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/grid_cap_probe.py

import importlib

import pytest
import torch
import ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
pd_mod = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

REPEATS = 5
P1 = (16, 12, 128, 128)
GRIDS = [(11, 10), (11, 8), (11, 6), (8, 8), (11, 4)]


class _CappedDevice:
    def __init__(self, device, x, y):
        self._device, self._grid = device, ttnn.CoreCoord(x, y)

    def compute_with_storage_grid_size(self):
        return self._grid


@pytest.mark.parametrize("grid", GRIDS, ids=[f"{x}x{y}" for x, y in GRIDS])
def test_grid_cap(device, grid, monkeypatch):
    real_plan = pd_mod.compute_plan
    monkeypatch.setattr(
        op, "compute_plan", lambda dev, pages, page_bytes: real_plan(_CappedDevice(dev, *grid), pages, page_bytes)
    )
    op._DESCRIPTOR_CACHE.clear()
    torch.manual_seed(0)
    bits = torch.randint(-(1 << 31), 1 << 31, P1, dtype=torch.int64).to(torch.int32)
    x = ttnn.from_torch(
        bits.view(torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    plan = op.g6_clone_plan(x)
    print(f"PLAN grid={grid} cores={plan['num_cores']} g1={plan['core_pages_g1']} g2={plan['core_pages_g2']}")
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).view(torch.int32), bits)
    op._DESCRIPTOR_CACHE.clear()
