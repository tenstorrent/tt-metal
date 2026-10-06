# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)


def _extract_tt_config(
    config: dict[str, Any], config_name: str
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object")
    if "tt" not in config:
        return {}, False
    tt_config = config["tt"]
    if not isinstance(tt_config, dict):
        raise ValueError(f"{config_name}['tt'] must be a JSON object")
    return tt_config, True


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT config from vLLM's generic additional config namespace."""
    additional_config, _ = _extract_tt_config(
        getattr(vllm_config, "additional_config", {}) or {}, "additional_config"
    )
    return dict(additional_config)


# Internal key recording the resolved TT lane count. Stored at the top level of
# additional_config -- deliberately outside the user "tt" namespace -- so it
# never collides with user config and reads as platform-derived state rather
# than user input. Written by store_tt_lane_count, read by
# get_tt_data_parallel_size.
_RESOLVED_LANE_COUNT_KEY = "_tt_resolved_lane_count"


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int:
    """Effective TT lane count for batching, KV sizing, and merged execution.

    With gathered multi-process DP (``data_parallel_size > 1``) this is just
    ``data_parallel_size`` (one engine per rank). With a single engine
    (``data_parallel_size == 1``) it is the lane count resolved by the Galaxy
    gather-DP-to-lanes conversion (see ``platform.py``) and recorded via
    ``store_tt_lane_count``; absent that, the count is 1. Not user-facing.
    """
    if vllm_config.parallel_config.data_parallel_size > 1:
        return vllm_config.parallel_config.data_parallel_size
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_RESOLVED_LANE_COUNT_KEY, 1))


def store_tt_lane_count(vllm_config: "VllmConfig", lanes: int) -> None:
    """Record the resolved in-process TT lane count on the config.

    Writes an internal, top-level key into ``additional_config`` (kept out of
    the user "tt" namespace) so ``get_tt_data_parallel_size`` observes it both
    here and in the worker subprocess -- ``additional_config`` is a declared
    config field, so it survives the copy/pickle to that process. Internal
    handoff from the Galaxy gather-DP-to-lanes conversion; not user-facing.
    """
    if lanes < 1:
        raise ValueError(f"resolved TT lane count must be >= 1, got {lanes}")
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_RESOLVED_LANE_COUNT_KEY] = lanes


def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Return the global TT batch capacity for model/KV sizing.

    Gathered multi-process DP keeps the historical contract: each rank receives
    ``max_num_seqs`` requests and the TT model is initialized for the gathered
    DP batch. Single-process lane mode is different: vLLM sees one engine, so
    ``max_num_seqs`` is already the global engine capacity and lanes are only an
    internal partition.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    if vllm_config.parallel_config.data_parallel_size > 1:
        return max_num_seqs * vllm_config.parallel_config.data_parallel_size
    return max_num_seqs


def get_tt_per_lane_max_num_seqs(vllm_config: "VllmConfig") -> int:
    """Return the per-lane/per-rank scheduling and wire-format capacity.

    Outside lane mode the global ``max_num_seqs`` is already the per-rank
    capacity. In single-process lane mode it is the validated per-lane split
    (see ``validate_tt_lane_config``).
    """
    if not uses_tt_lane_coordinator(vllm_config):
        return int(vllm_config.scheduler_config.max_num_seqs)
    return validate_tt_lane_config(vllm_config)


def validate_tt_lane_config(vllm_config: "VllmConfig") -> int:
    """Validate single-process lane-mode batch sizing; return per-lane capacity.

    Lane mode partitions the global ``max_num_seqs`` evenly across the lanes
    (one in-process DP replica each), so the global value must be a positive
    multiple of the lane count; raises ``ValueError`` otherwise. Assumes lane
    mode is active (callers gate on ``uses_tt_lane_coordinator``).

    Exposed as a named helper so ``platform.check_and_update_config`` can run
    this check at config time -- calling it for its raising side effect so a
    misconfiguration fails fast with a clear message -- rather than calling the
    per-lane getter and discarding its result.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    lanes = get_tt_data_parallel_size(vllm_config)
    if max_num_seqs % lanes != 0:
        raise ValueError(
            "max_num_seqs must be divisible by the TT lane count in "
            f"single-process lane mode; got max_num_seqs={max_num_seqs}, "
            f"lanes={lanes}."
        )
    per_lane = max_num_seqs // lanes
    if per_lane < 1:
        raise ValueError(
            "max_num_seqs must provide at least one request per TT lane; got "
            f"max_num_seqs={max_num_seqs}, lanes={lanes}."
        )
    return per_lane


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )
