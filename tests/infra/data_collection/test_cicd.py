import pytest
from loguru import logger


def test_pass():
    pass


@pytest.mark.parametrize("arg1", (1, 2, 3))
@pytest.mark.parametrize("arg2", (4, 5, 6))
def test_parametrized(arg1, arg2):
    if arg1 == 2:
        raise Exception("Failure in parametrized")
    logger.info(f"{arg1} {arg2}")


def test_fail():
    raise Exception("Failure")


@pytest.mark.skip(reason="Problem with this test")
def test_skipped():
    pass


@pytest.fixture
def cause_error():
    raise Exception("Error in fixture")


def test_error(cause_error):
    pass
