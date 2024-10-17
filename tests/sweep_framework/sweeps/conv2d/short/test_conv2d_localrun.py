import pytest
from tests.sweep_framework.sweeps.conv2d.short.conv2d_short_sweep import parameters
from tests.sweep_framework.sweep_utils.conv2d_common import run_short


@pytest.mark.parametrize("input_spec", parameters["short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_localrun(device, input_spec):
    run_short(
        input_spec,
        device,
    )
