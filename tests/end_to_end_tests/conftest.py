import pytest

import tt_lib

@pytest.fixture(scope="function")
def first_grayskull_device(silicon_arch_name, silicon_arch_grayskull):
    return tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
