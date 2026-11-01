# Probe: cost of the 128-on-110 tail (18 two-row cores) vs one tile-row per core. Not collected by default;
# run: scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/layer_norm_tile/probes/probe_tail.py
import pytest
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import run_and_check, DRAM, L1


@pytest.mark.parametrize(
    "shape", [(110, 1, 32, 768), (128, 1, 32, 768), (8, 1, 512, 768)], ids=["T110", "T128_b", "T128"]
)
@pytest.mark.parametrize(
    "bufs", [(L1, L1, L1), (L1, DRAM, DRAM), (DRAM, DRAM, DRAM)], ids=["all_l1", "in_l1", "all_dram"]
)
def test_tail(device, shape, bufs):
    run_and_check(device, shape, input_buf=bufs[0], residual_buf=bufs[1], output_buf=bufs[2], measure=True)
