import enum


class InfraErrorV1(enum.Enum):
    GENERIC_SET_UP_FAILURE = enum.auto()
    JOB_TIMEOUT_FAILURE = enum.auto()
    GENERIC_FAILURE = enum.auto()
