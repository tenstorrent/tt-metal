# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-host validation for the generator's ordinary decode boundary.

Implements metadata rules from vllm-tt-plugin commit
042a94863ad27d88207bcf71abdf67122c6f4c27, docs/DECODE_RELOAD_CONTRACT.md,
MODEL_EXECUTION.md and MODEL_LIFECYCLE.md. This is not a vLLM adapter and does
not declare tested device execution, async support or a capability marker.

Commands are authoritative: never derive tokens, positions, RoPE, active rows,
or RNG counters from host token/position arguments when reload_inputs=False.
The generator must apply a full-capacity slot permutation exactly once, before
sampling parameter reload/reset, including dormant sampler and off-batch state.
None means no permutation. A remap is data, not permission to infer reloads.

Submission handles retain tensor references, but references alone do not prevent
a subsequent trace from overwriting their backing storage. The owner must copy
into prepared per-submission output storage or drain a pending read before that
storage is reused. Host logits must own writable, independent full-vocab storage.
Readback is observational: no sampling, position advance, or RNG mutation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from numbers import Integral
from typing import Any, Literal


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _slot_tuple(value, name):
    # The pinned plugin sends slot_remap as a CPU torch.Tensor. Accept its
    # host-only list protocol without importing torch, and reject a device
    # transfer hidden inside validation. NumPy arrays have no device attribute.
    if not isinstance(value, Sequence) and callable(getattr(value, "tolist", None)):
        device = getattr(value, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cpu":
            raise ValueError(f"{name} must be host-resident metadata")
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integer slot IDs")
    result = tuple(value)
    if not result or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in result):
        raise ValueError(f"{name} must contain nonnegative integer slot IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not repeat slots")
    return tuple(int(i) for i in result)


@dataclass(frozen=True)
class DecodeCommands:
    reload_inputs: bool = True
    reload_page_table: bool = False
    reload_sampling_params: bool = False
    reset_sampling_state: bool = False
    sampling_state_slots: tuple[int, ...] | None = None
    slot_remap: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ("reload_inputs", "reload_page_table", "reload_sampling_params", "reset_sampling_state"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool; commands are not truthiness hints")
        if self.reload_inputs and self.reload_page_table:
            raise ValueError("reload_inputs includes page tables; reload_page_table must be False")
        if self.reset_sampling_state and not self.reload_inputs:
            raise ValueError("reset_sampling_state requires authoritative host inputs (reload_inputs=True)")
        for name in ("sampling_state_slots", "slot_remap"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _slot_tuple(value, name))
        if self.sampling_state_slots is not None and not self.reset_sampling_state:
            raise ValueError("sampling_state_slots only applies to an explicit sampling-state reset")

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]):
        """Parse ONLY command kwargs; normal forward arguments are explicit elsewhere."""
        if "reset_batch" in kwargs:
            raise TypeError("Legacy reset_batch is unsupported by decode command contract version 1")
        allowed = {item.name for item in fields(cls)}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError("Unknown decode command keyword(s): " + ", ".join(sorted(unknown)))
        return cls(**kwargs)

    def validate_context(
        self,
        *,
        mode: Literal["host", "device"],
        slot_capacity: int,
        batch_size: int,
        forward_initialized: bool,
        sampling_params_initialized: bool = False,
        sampling_state_initialized: bool = False,
        sampling_params_available: bool = False,
        pending_submission: bool = False,
    ):
        """Validate prior state explicitly; never infer commands or reset on equality.

        Initialization flags describe the submitted binding/layout, not whether
        an unrelated prior request once used a sampler. A first None seed still
        needs parameter upload and RNG/state initialization. The generator must
        verify sampling_state_slots have authoritative position/history metadata.
        """
        _positive_integer(slot_capacity, "slot_capacity")
        _positive_integer(batch_size, "batch_size")
        if batch_size > slot_capacity:
            raise ValueError("Decode batch exceeds the configured slot capacity")
        if mode not in ("host", "device"):
            raise ValueError("Decode mode must be host or device")
        if self.slot_remap is not None and (
            len(self.slot_remap) != slot_capacity or set(self.slot_remap) != set(range(slot_capacity))
        ):
            raise ValueError("slot_remap must be a bijection over the FULL configured slot capacity")
        if self.sampling_state_slots is not None and any(i >= slot_capacity for i in self.sampling_state_slots):
            raise ValueError("sampling_state_slots exceeds the configured slot capacity")
        if not self.reload_inputs and not forward_initialized:
            raise ValueError("Resident decode requires initialized forward state for this binding/layout")
        mutates_layout = self.slot_remap is not None and self.slot_remap != tuple(range(slot_capacity))
        if pending_submission and (
            self.reload_inputs or self.reload_sampling_params or self.reset_sampling_state or mutates_layout
        ):
            raise ValueError("Drain pending submissions before input/layout/sampling-state transitions")
        if mode == "host":
            if not self.reload_inputs:
                raise ValueError("Host sampling requires authoritative inputs on every decode")
            if self.reload_sampling_params or self.reset_sampling_state:
                raise ValueError("Host sampling supplies no device sampling parameters/state-reset request")
            return
        if self.reload_sampling_params and not sampling_params_available:
            raise ValueError("reload_sampling_params requires supplied device sampling parameters")
        if not sampling_params_initialized and not self.reload_sampling_params:
            raise ValueError("Initialize sampling parameters explicitly, including an unspecified/None seed")
        if not sampling_state_initialized and not self.reset_sampling_state:
            raise ValueError("Initialize RNG/penalty state explicitly before the first device sample")


@dataclass(frozen=True)
class DecodeCompletion:
    """An already completed host read; creation performs no transfer or sampling."""

    host_output: Any = field(repr=False)
    row_order: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DecodeSubmission:
    """Own an exact submission's outputs until its read completes.

    ``row_order[i]`` is the state slot for submitted output row i, including
    inactive rows. Return [B,1] IDs or [B,1,V] full logits without dropping gaps.
    ``slot_remap`` instead covers the full capacity and must not be confused
    with this row list. ``keepalive`` retains temporary transfer/output storage.
    """

    submission_id: int
    mode: Literal["tokens", "logits"]
    batch_size: int
    row_order: tuple[int, ...]
    slot_capacity: int
    token_tensor: Any = field(default=None, repr=False)
    logits_tensor: Any = field(default=None, repr=False)
    vocab_size: int | None = None
    keepalive: tuple[Any, ...] = field(default=(), repr=False)
    active_rows: tuple[bool, ...] | None = None
    completion: DecodeCompletion | None = field(default=None, init=False)

    def __post_init__(self):
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.slot_capacity, "slot_capacity")
        if isinstance(self.submission_id, bool) or not isinstance(self.submission_id, int) or self.submission_id < 0:
            raise ValueError("submission_id must be a nonnegative integer")
        object.__setattr__(self, "row_order", _slot_tuple(self.row_order, "row_order"))
        object.__setattr__(self, "keepalive", tuple(self.keepalive))
        if self.active_rows is not None and (
            len(self.active_rows) != self.batch_size or any(type(value) is not bool for value in self.active_rows)
        ):
            raise ValueError("active_rows must describe every submitted row")
        if len(self.row_order) != self.batch_size or any(i >= self.slot_capacity for i in self.row_order):
            raise ValueError("row_order must match submitted batch rows within full slot capacity")
        if self.mode == "tokens":
            if self.token_tensor is None:
                raise ValueError("Token submission must own its sampled token tensor")
        elif self.mode == "logits":
            if self.logits_tensor is None:
                raise ValueError("Logits submission must own its full-vocabulary logits tensor")
            _positive_integer(self.vocab_size, "vocab_size")
        else:
            raise ValueError("Submission mode must be tokens or logits")

    @property
    def expected_output_shape(self):
        return (self.batch_size, 1) if self.mode == "tokens" else (self.batch_size, 1, self.vocab_size)

    def complete_read(self, host_output, *, shape, row_order):
        """Record completed read metadata only; repeat of the same read is idempotent.

        The owner must wait for its own transfer events BEFORE calling this. Shape
        is explicit metadata so this helper never invokes tensor/device methods.
        This does not certify ownership, writable storage, or device completion.
        """
        shape = tuple(shape)
        row_order = _slot_tuple(row_order, "row_order")
        if shape != self.expected_output_shape or row_order != self.row_order:
            raise ValueError("Readback shape/order differs from the exact submitted output")
        if host_output is None:
            raise ValueError("Completed read requires its host output")
        if self.completion is not None:
            if self.completion.host_output is not host_output:
                raise ValueError("Cannot replace an already completed submission with a different read")
            return self.completion
        object.__setattr__(self, "completion", DecodeCompletion(host_output, row_order, shape))
        return self.completion
