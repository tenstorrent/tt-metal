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
    TT_TRIAGE_JOB_HANG = enum.auto()
    DOCKER_REGISTRY_FAILURE = enum.auto()
    ARTIFACT_UPLOAD_FAILURE = enum.auto()
    CHECKOUT_FAILURE = enum.auto()
    DOCKER_CONTAINER_ID_NULL_FAILURE = enum.auto()
    ACTION_DOWNLOAD_FAILURE = enum.auto()
    TEST_REPORTER_FAILURE = enum.auto()
    TEST_REPORTER_NO_REPORTS_FAILURE = enum.auto()
    GENERIC_EXIT_CODE_FAILURE = enum.auto()
    GIT_PROCESS_FAILURE = enum.auto()
    ARTIFACT_FINALIZE_FAILURE = enum.auto()
    ARTIFACT_DOWNLOAD_CONNECTION_FAILURE = enum.auto()
    ARTIFACT_DOWNLOAD_NOT_FOUND_FAILURE = enum.auto()
    ARTIFACT_DOWNLOAD_FORBIDDEN_FAILURE = enum.auto()
    ARTIFACT_UPLOAD_STALLED_FAILURE = enum.auto()
    REQUEST_CANCELLED_FAILURE = enum.auto()
    GITHUB_API_MALFORMED_REQUEST_FAILURE = enum.auto()


class TestErrorV1(enum.Enum):
    PY_TEST_FAILURE = enum.auto()
    CPP_TEST_FAILURE = enum.auto()
    UNKNOWN_TEST_FAILURE = enum.auto()


class CodeQualityErrorV1(enum.Enum):
    CLANG_TIDY_VIOLATION = enum.auto()
