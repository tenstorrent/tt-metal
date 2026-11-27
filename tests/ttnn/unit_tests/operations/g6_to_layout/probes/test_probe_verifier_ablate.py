# Verifier probe: device-ns of one ablated kernel variant on the flagged perf case 308 (and its R2 twin 286).
# The kernel is edited in place for the run, then restored. Run under scripts/run_safe_pytest.sh --profile.
# No value check: an ablated kernel moves wrong data by construction.
import pytest
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout

SHAPES = [(1, 2048, 2048, 3), (1, 1024, 1024, 3)]
REPEATS = 2


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_ablate(device, shape):
    tt_x = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(REPEATS):
        g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(device)
