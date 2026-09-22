# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Legacy sampling public surface with lazy compatibility exports.

Importing this package must not eagerly load the TTTv1 sampling generator,
penalties, or trace state. Common TTTv2 code imports neutral value modules
directly; legacy callers retain the aggregate API and trigger only the module
that owns the requested attribute.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TTSampling": (".tt_sampling", "TTSampling"),
    "TTPenalties": (".tt_penalties", "TTPenalties"),
    "apply_penalties": (".tt_penalties", "apply_penalties"),
    "LogProbsCalculator": (".tt_log_probs", "LogProbsCalculator"),
    "LogProbsResult": (".tt_log_probs", "LogProbsResult"),
    "SamplingGenerator": (".generator", "SamplingGenerator"),
    "SamplingParams": (".generator", "SamplingParams"),
    "SAMPLING_PARAM_FIELDS": (".generator", "SAMPLING_PARAM_FIELDS"),
    "format_sampling_params": (".generator", "format_sampling_params"),
    "broadcast_sampling_params": (".generator", "broadcast_sampling_params"),
    "scatter_sampling_params_to_slots": (".generator", "scatter_sampling_params_to_slots"),
    "slice_sampling_params": (".generator", "slice_sampling_params"),
    "chunk_sampling_params": (".generator", "chunk_sampling_params"),
    "SeedManager": (".generator", "SeedManager"),
    "split_list": ("._utils", "split_list"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
