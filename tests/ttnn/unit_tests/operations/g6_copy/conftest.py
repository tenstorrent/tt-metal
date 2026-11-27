import pytest


def pytest_collection_modifyitems(items):
    # A module that sets its own use_module_device(device_params) (the trace test) keeps it.
    for item in items:
        if item.get_closest_marker("use_module_device") is None:
            item.add_marker(pytest.mark.use_module_device)
