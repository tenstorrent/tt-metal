import pytest

import tt_lib


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_program_cache():
    tt_lib.program_cache.disable_and_clear()
    tt_lib.program_cache.enable()
    assert tt_lib.program_cache.num_entries() == 0, f"Unused program cache has non-zero entries?"
    tt_lib.program_cache.disable_and_clear()
    pass

@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_arch():
    assert tt_lib.device.Arch.GRAYSKULL.name == "GRAYSKULL"
    assert tt_lib.device.Arch.WORMHOLE_B0.name == "WORMHOLE_B0"
    pass


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_host():
    host = tt_lib.device.Host()
    pass
