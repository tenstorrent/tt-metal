import enum


class InfraErrorV1(enum.Enum):
    GENERIC_SET_UP_FAILURE = enum.auto()
    JOB_UNIT_TIMEOUT_FAILURE = enum.auto()
    JOB_CUMULATIVE_TIMEOUT_FAILURE = enum.auto()
    GENERIC_FAILURE = enum.auto()
    DISK_SPACE_FAILURE = enum.auto()
    RUNNER_COMM_FAILURE = enum.auto()
    RUNNER_SHUTDOWN_FAILURE = enum.auto()
    API_RATE_LIMIT_FAILURE = enum.auto()
    RUNNER_CARD_IN_USE_FAILURE = enum.auto()
    JOB_HANG = enum.auto()


class TestErrorV1(enum.Enum):
    PY_TEST_FAILURE = enum.auto()
    CPP_TEST_FAILURE = enum.auto()
    UNKNOWN_TEST_FAILURE = enum.auto()
