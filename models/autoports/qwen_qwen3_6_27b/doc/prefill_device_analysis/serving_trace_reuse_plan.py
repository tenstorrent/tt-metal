# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only policy prototype for a reviewed, closed C1 trace-reuse experiment.

This module neither imports TTNN nor modifies the serving adapter. A future
adapter must implement the returned actions and supply observed ownership and
allocation evidence. Matching shapes alone never authorizes reuse.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContract:
    # Include physical/logical token extents, allocated batch, active slot,
    # prefill implementation/configuration, page-table spec and tensor layouts.
    prefill_signature: tuple
    active_slots: tuple[int, ...]
    sampling_signature: tuple
    # Each entry must identify both tensor ownership and buffer address/spec.
    cache_bindings: tuple
    page_table_binding: tuple
    seeded: bool = False
    penalties: bool = False
    logprobs: bool = False
    nonidentity_remap: bool = False


@dataclass(frozen=True)
class CapturedContract:
    request: RequestContract
    # Model/sampler trace IDs and persistent token/position/mask/logit owners.
    trace_bindings: tuple
    prefill_warmed_before_capture: bool


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def before_prefill(captured: CapturedContract | None, requested: RequestContract) -> Decision:
    """Decide before reset/state rebinding whether trace handles may survive."""
    if captured is None:
        return Decision("release_and_capture", "no captured contract")
    if not captured.prefill_warmed_before_capture:
        return Decision("release_and_capture", "prefill program envelope was not warmed before capture")
    if len(requested.active_slots) != 1:
        return Decision("release_and_capture", "prototype supports one active request")
    if requested.seeded or requested.penalties or requested.logprobs:
        return Decision("release_and_capture", "sampler mode is outside the initial prototype")
    if requested.nonidentity_remap:
        return Decision("release_and_capture", "slot permutation is outside the initial prototype")
    if requested != captured.request:
        return Decision("release_and_capture", "request shape, mode, slot or persistent ownership changed")
    return Decision("preserve_until_prefill_finishes", "matching prewarmed request contract")


def after_prefill(
    captured: CapturedContract,
    requested: RequestContract,
    *,
    current_trace_bindings: tuple,
    program_entries_before: int,
    program_entries_after: int,
    allocation_evidence_clear: bool,
    inputs_reload_authorized: bool,
    request_prefill_complete: bool,
) -> Decision:
    """Decide before any replay; uncertain ownership always forces recapture.

    allocation_evidence_clear must come from runtime inspection of surviving
    post-capture allocations; a matching program-cache count is insufficient.
    The adapter must still refresh page-table contents, token and position via
    the existing explicit reload contract. Sampler history/RNG ordering remains
    owned by the current adapter, with no calls added or removed by this plan.
    """
    initial = before_prefill(captured, requested)
    if initial.action != "preserve_until_prefill_finishes":
        return initial
    if not inputs_reload_authorized:
        return Decision("reject", "new request requires authoritative reload_inputs")
    if not request_prefill_complete:
        return Decision("reject", "reset request slot has not completed prefill")
    if current_trace_bindings != captured.trace_bindings:
        return Decision("release_and_capture", "trace IDs or persistent trace I/O owners changed")
    if program_entries_after != program_entries_before:
        return Decision("release_and_capture", "eager prefill created or replaced program-cache entries")
    if not allocation_evidence_clear:
        return Decision("release_and_capture", "surviving post-capture allocations are unverified")
    return Decision("refresh_and_replay", "resident trace ownership and prefill envelope verified")
