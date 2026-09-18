"""Strict optional native bridge timeout values shared by plan validation and serialization."""

from runner_support import require

DEFAULT_INBOUND_TIMEOUT_MS = 600_000
DEFAULT_BRIDGE_TIMEOUT_MS = 1_500_000
UINT32_MAX = 2**32 - 1


def _positive_uint32(plan, key, default):
    value = plan.get(key, default)
    require(type(value) is int and 0 < value <= UINT32_MAX, key + " must be a positive uint32 integer")
    return value


def bridge_timeouts(plan):
    return {
        "inbound_timeout_ms": _positive_uint32(plan, "inbound_timeout_ms", DEFAULT_INBOUND_TIMEOUT_MS),
        "bridge_timeout_ms": _positive_uint32(plan, "bridge_timeout_ms", DEFAULT_BRIDGE_TIMEOUT_MS),
    }


def validate_transfer_timeout(plan):
    values = bridge_timeouts(plan)
    phase = plan.get("transfer_phase_timeout_seconds")
    maximum = values["inbound_timeout_ms"] // 1000 - 10
    require(
        type(phase) is int and 0 < phase <= maximum,
        "Python transfer wait must retain ten seconds inside the native inbound deadline",
    )
    return values
