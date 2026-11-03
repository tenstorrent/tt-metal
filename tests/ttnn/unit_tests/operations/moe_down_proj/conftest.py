import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


@pytest.fixture(scope="module", autouse=True)
def _fresh_module_caches(request):
    # The precompile collect pass runs bodies with stand-in torch.randn / from_torch values; a module-level
    # lru_cache (e.g. host_weight) would carry them into the real pass. Every fixture is torn down between
    # passes, so clearing here gives each pass its own cache.
    def clear():
        for value in vars(request.module).values():
            if callable(getattr(value, "cache_clear", None)):
                value.cache_clear()

    clear()
    yield
    clear()
