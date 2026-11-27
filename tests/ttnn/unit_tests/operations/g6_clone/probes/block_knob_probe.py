# Verifier probe: force the multi-block path (block_pages < core assignment, cb_depth_blocks = 2,
# ragged last block) on p1 by turning BLOCKS_PER_CORE_KNOB, check bit-exactness,
# and read device kernel time per variant under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/block_knob_probe.py
# Phase 0 never takes this path on INPUTS (one block per core), so this is the only coverage of the
# depth-2 ring, the nominal-count push/pop on a ragged block and the writer's flush-before-pop.

import importlib

import pytest
import torch
import ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
pd_mod = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

REPEATS = 5
P1 = (16, 12, 128, 128)

# blocks per core -> block_pages on p1 (28 pages on group-1 cores): 28, 14, 10, 7, 4, 1.
BLOCKS_PER_CORE = [1, 2, 3, 4, 7, 28]


@pytest.mark.parametrize("blocks_per_core", BLOCKS_PER_CORE, ids=[f"blocks{b}" for b in BLOCKS_PER_CORE])
def test_block_knob(device, blocks_per_core, monkeypatch):
    monkeypatch.setattr(pd_mod, "BLOCKS_PER_CORE_KNOB", blocks_per_core)
    op._DESCRIPTOR_CACHE.clear()
    torch.manual_seed(0)
    bits = torch.randint(-(1 << 31), 1 << 31, P1, dtype=torch.int64).to(torch.int32)
    x_host = bits.view(torch.float32)
    x = ttnn.from_torch(
        x_host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    plan = op.g6_clone_plan(x)
    print(
        f"PLAN block_pages={plan['block_pages']} depth={plan['cb_depth_blocks']} "
        f"blocks_max={plan['num_blocks_max']} g1={plan['core_pages_g1']} g2={plan['core_pages_g2']}"
    )
    assert plan["block_pages"] == -(-28 // blocks_per_core)
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).view(torch.int32), bits)
    op._DESCRIPTOR_CACHE.clear()
