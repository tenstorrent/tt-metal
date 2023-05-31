import time
import random

import pytest

from loguru import logger


def pytest_collection_modifyitems(session, config, items):
    random_state = random.getstate()

    seed = int(time.time())

    logger.info(f"Using seed {seed} for test ordering (llrt)")

    for pytest_item in items:
        assert (
            "llrt" in pytest_item.name
        ), f"{pytest_item} doesn't seem to be an llrt test"

    random.seed(seed)

    random.shuffle(items)

    random.setstate(random_state)


@pytest.fixture(scope="session")
def llrt_to_dos():
    logger.warning("Need to do timeouts via external thing, not within cmd")
    logger.warning("Need to do multiple threads for build_kernels")


@pytest.fixture(scope="function")
def llrt_fixtures(reset_tensix, llrt_to_dos):
    yield
