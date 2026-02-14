# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value


def is_default_value(values, default):
    """Check if values match a default, handling None, scalar, and iterable inputs."""
    if values is None:
        return True
    if isinstance(values, (int, float)):
        return values == default
    return all(value == default for value in values)


def filter_none(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}
