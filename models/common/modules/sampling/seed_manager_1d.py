# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned request seed state for :mod:`sampling_1d`.

``Sampling1D`` owns a stable seed ``LazyBuffer`` because traces capture its
device handle.  Request seeds, however, are invocation state.  ``SeedManager1D``
bridges those two lifetimes without storing request state on the module or on
the manager: every mutable lifecycle field lives in a caller-owned
``SeedState`` passed to each operation.

This module is intentionally independent of the legacy sampling generator and
state classes.  It only borrows the ``Sampling1DConfig.seeds`` ``LazyBuffer``.
"""

from __future__ import annotations

import copy
import random
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import torch

from models.common.modules.lazy_buffer import LazyBuffer

if TYPE_CHECKING:
    from models.common.modules.sampling.sampling_1d import Sampling1DConfig


# The device reserves UINT32_MAX as its "advance the existing RNG" sentinel.
# Seeds generated here are always ordinary positive seed values.
MAX_UINT32 = 2**32 - 1
DEVICE_SEED_MAX = 1_000_000
_UINT64_MASK = (1 << 64) - 1


def _hash_request_seed_to_device_seed(seed: int, counter: int, salt: int = 0) -> int:
    """Return a stable, bounded device seed for one request token.

    ``salt`` distinguishes simultaneous requests carrying the same explicit
    request seed.  A unique request keeps salt zero, preserving the familiar
    slot-independent ``(seed, counter)`` stream.
    """

    value = (int(seed) & _UINT64_MASK) ^ ((int(counter) + 0x9E3779B97F4A7C15) & _UINT64_MASK)
    value ^= (int(salt) * 0xD1B54A32D192ED03) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value = (value ^ (value >> 31)) & _UINT64_MASK
    return (value % DEVICE_SEED_MAX) + 1


@dataclass(frozen=True)
class SeedSlotState:
    """Immutable checkpoint for a request temporarily removed from a lane.

    Callers use :meth:`SeedManager1D.suspend` before preemption and pass the
    checkpoint back to :meth:`SeedManager1D.resume`.  The Python RNG state is
    included so an unseeded request continues its varied stream instead of
    being reinitialized after resume.
    """

    request_seed: int | None
    token_counter: int
    salt: int
    unseeded_rng_state: object
    last_absolute_position: int | None
    current_device_seed: int | None


@dataclass(frozen=True)
class SeedStateSnapshot:
    """Immutable, assertion-friendly view of a mutable :class:`SeedState`."""

    active: tuple[bool, ...]
    request_seeds: tuple[int | None, ...]
    token_counters: tuple[int, ...]
    salts: tuple[int, ...]
    unseeded_rng_states: tuple[object, ...]
    last_absolute_positions: tuple[int | None, ...]
    current_device_seeds: tuple[int | None, ...]
    buffer_is_default: bool

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(slot for slot, active in enumerate(self.active) if active)


@dataclass
class SeedState:
    """Mutable per-lane seed state owned by the runtime caller.

    The parallel lists are slot indexed.  ``request_seeds`` deliberately uses
    ``None`` for an unseeded request, so ``active`` separately distinguishes an
    admitted unseeded request from an empty slot.
    """

    active: list[bool]
    request_seeds: list[int | None]
    token_counters: list[int]
    salts: list[int]
    unseeded_rngs: list[random.Random]
    last_absolute_positions: list[int | None]
    current_device_seeds: list[int | None]
    buffer_is_default: bool = True

    def __post_init__(self) -> None:
        lengths = {
            len(self.active),
            len(self.request_seeds),
            len(self.token_counters),
            len(self.salts),
            len(self.unseeded_rngs),
            len(self.last_absolute_positions),
            len(self.current_device_seeds),
        }
        if len(lengths) != 1:
            raise ValueError("all SeedState slot fields must have the same capacity")
        if any(counter < 0 for counter in self.token_counters):
            raise ValueError("SeedState token counters must be nonnegative")
        if any(salt < 0 for salt in self.salts):
            raise ValueError("SeedState salts must be nonnegative")

    @classmethod
    def create(
        cls,
        capacity: int,
        *,
        entropy_factory: Callable[[int], int] = secrets.randbits,
    ) -> "SeedState":
        """Create empty caller-owned state with independent unseeded RNGs."""

        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("SeedState capacity must be positive")
        return cls(
            active=[False] * capacity,
            request_seeds=[None] * capacity,
            token_counters=[0] * capacity,
            salts=[0] * capacity,
            unseeded_rngs=[random.Random(entropy_factory(64)) for _ in range(capacity)],
            last_absolute_positions=[None] * capacity,
            current_device_seeds=[None] * capacity,
        )

    @property
    def capacity(self) -> int:
        return len(self.active)

    def snapshot(self) -> SeedStateSnapshot:
        """Return an immutable view without exposing mutable RNG objects."""

        return SeedStateSnapshot(
            active=tuple(self.active),
            request_seeds=tuple(self.request_seeds),
            token_counters=tuple(self.token_counters),
            salts=tuple(self.salts),
            unseeded_rng_states=tuple(rng.getstate() for rng in self.unseeded_rngs),
            last_absolute_positions=tuple(self.last_absolute_positions),
            current_device_seeds=tuple(self.current_device_seeds),
            buffer_is_default=bool(self.buffer_is_default),
        )


class SeedManager1D:
    """Update a ``Sampling1DConfig`` seed buffer from caller-owned state.

    The manager retains only the borrowed buffer, its construction defaults,
    and an entropy provider.  It never retains a ``SeedState``.
    """

    def __init__(
        self,
        sampling_config: "Sampling1DConfig",
        *,
        entropy_factory: Callable[[int], int] = secrets.randbits,
    ) -> None:
        seed_buffer = getattr(sampling_config, "seeds", None)
        if seed_buffer is None or not callable(getattr(seed_buffer, "update", None)) or not callable(
            getattr(seed_buffer, "get_device_buffer", None)
        ):
            raise TypeError("Sampling1DConfig.seeds must be a mutable LazyBuffer-compatible object")
        source = getattr(seed_buffer, "source", None)
        if not isinstance(source, torch.Tensor):
            raise TypeError("Sampling1DConfig.seeds.source must be a torch.Tensor")

        capacity = int(getattr(sampling_config, "max_batch_size", 0))
        if capacity <= 0:
            raise ValueError("Sampling1DConfig.max_batch_size must be positive")
        if source.numel() != capacity:
            raise ValueError(
                "Sampling1DConfig seed-buffer source size does not match max_batch_size "
                f"({source.numel()} != {capacity})"
            )

        self._seed_buffer = seed_buffer
        self._default_source = source.detach().clone()
        self._default_values = tuple(int(value) for value in self._default_source.reshape(-1).tolist())
        self._capacity = capacity
        self._entropy_factory = entropy_factory

    @property
    def max_batch_size(self) -> int:
        return self._capacity

    @property
    def default_values(self) -> tuple[int, ...]:
        return self._default_values

    @property
    def seed_buffer(self) -> LazyBuffer:
        """Return the borrowed persistent buffer specification."""

        return self._seed_buffer

    def create_state(self) -> SeedState:
        """Convenience constructor; ownership of the returned state is the caller's."""

        return SeedState.create(self._capacity, entropy_factory=self._entropy_factory)

    def get_seed_device_buffer(self):
        """Return the stable device handle captured by eager/trace execution."""

        return self._seed_buffer.get_device_buffer()

    def admit(
        self,
        state: SeedState,
        request_seeds: int | None | Sequence[int | None] | torch.Tensor,
        slots: Iterable[int],
    ) -> None:
        """Register a simultaneous prefill admission in request order.

        Admission always starts a new request stream, even if a target slot
        previously held the same integer seed.  Equal-seed salts are allocated
        after all target slots have been cleared, so the simultaneous admission
        set and surviving live requests determine collision-free salts.
        """

        self._validate_state(state)
        normalized_slots = self._normalize_slots(slots, label="admission slot")
        seeds = self._request_ordered_seeds(request_seeds, len(normalized_slots))

        for slot in normalized_slots:
            self._clear_slot(state, slot)
        for slot, seed in zip(normalized_slots, seeds):
            self._register_slot(state, slot, seed)

    def synchronize(
        self,
        state: SeedState,
        slot_seeds: int | None | Sequence[int | None] | torch.Tensor,
        active_slots: Iterable[int],
        *,
        reset_batch: bool,
    ) -> None:
        """Synchronize slot-indexed decode state at a batch boundary.

        With ``reset_batch=False`` every active slot must already be registered
        with the same request seed; the call may only remove completed slots.
        With ``reset_batch=True`` new or changed slots are admitted, while an
        unchanged running request preserves its counter, salt, RNG, and current
        seed.  This prevents an unrelated admission from splicing a survivor
        onto a finished equal-seed sibling's stream.
        """

        self._validate_state(state)
        active = self._normalize_slots(active_slots, label="active slot")
        desired = {slot: self._slot_indexed_seed(slot_seeds, slot) for slot in active}
        changed = [slot for slot in active if not state.active[slot] or state.request_seeds[slot] != desired[slot]]
        if changed and not reset_batch:
            raise RuntimeError(
                "new or changed active seed slots require reset_batch=True or an explicit admit() call: "
                f"{changed}"
            )

        live = set(active)
        removed = [slot for slot in range(self._capacity) if state.active[slot] and slot not in live]
        for slot in removed:
            self._clear_slot(state, slot)

        # Clear all replacements before assigning any salt, making allocation
        # depend on the complete surviving/admission set rather than mutation order.
        for slot in changed:
            self._clear_slot(state, slot)
        for slot in active:
            if slot in changed:
                self._register_slot(state, slot, desired[slot])

        if removed or changed:
            self._write_current_values(state)

    def refresh(
        self,
        state: SeedState,
        active_slots: Iterable[int],
        *,
        positions=None,
    ) -> tuple[int, ...]:
        """Advance active streams and refresh the persistent seed buffer.

        ``positions`` is slot-indexed and denotes the zero-based absolute decode
        position.  An explicit request seed is hashed with ``position + 1``.
        Repeating a refresh for the same absolute position is idempotent and
        does not double-advance counters.  Unseeded rows reuse their cached draw
        at the same position; at a new position their caller-owned RNG advances
        exactly once.

        If ``positions`` is omitted, each call consumes the slot's next
        sequential token counter.
        """

        self._validate_state(state)
        active = self._normalize_slots(active_slots, label="active slot")
        for slot in active:
            if not state.active[slot]:
                raise RuntimeError(f"seed slot {slot} must be admitted or synchronized before refresh")

        values = list(self._default_values)
        for slot in active:
            position = self._position_for_slot(positions, slot)
            if positions is not None and position is None:
                raise ValueError(f"absolute positions do not cover active seed slot {slot}")
            request_seed = state.request_seeds[slot]
            if request_seed is not None:
                if position is None:
                    counter = state.token_counters[slot]
                    state.token_counters[slot] = counter + 1
                    state.last_absolute_positions[slot] = None
                else:
                    if position < 0:
                        raise ValueError("active seed positions must be nonnegative")
                    counter = position + 1
                    # Absolute-position refresh is deliberately idempotent.
                    state.token_counters[slot] = counter + 1
                    state.last_absolute_positions[slot] = position
                device_seed = _hash_request_seed_to_device_seed(request_seed, counter, state.salts[slot])
            else:
                if position is not None and position < 0:
                    raise ValueError("active seed positions must be nonnegative")
                repeated_position = (
                    position is not None
                    and state.last_absolute_positions[slot] == position
                    and state.current_device_seeds[slot] is not None
                )
                if repeated_position:
                    device_seed = state.current_device_seeds[slot]
                else:
                    device_seed = state.unseeded_rngs[slot].randint(1, DEVICE_SEED_MAX)
                    state.token_counters[slot] += 1
                state.last_absolute_positions[slot] = position

            state.current_device_seeds[slot] = device_seed
            values[slot] = device_seed

        self._write_values(state, values)
        return tuple(values)

    def refresh_prefill_replicated(
        self,
        state: SeedState,
        slot: int,
        *,
        position: int | None = None,
    ) -> int:
        """Advance one admitted prefill stream and replicate its device seed.

        Single-request prefill logits may occupy a sequence-tile row rather
        than the request's persistent decode slot.  Replicating the one draw
        makes every physical sampling row observe that request stream while
        keeping the caller-owned counter attached to its real slot.
        """

        slot = self._validate_slot(slot, label="prefill slot")
        positions = None if position is None else {slot: int(position)}
        values = self.refresh(state, (slot,), positions=positions)
        device_seed = int(values[slot])
        self._write_values(state, (device_seed,) * self._capacity)
        return device_seed

    def apply_slot_remap(self, state: SeedState, remap: Sequence[int] | torch.Tensor) -> None:
        """Move complete seeded and unseeded request state during compaction.

        ``remap[new_slot] = old_slot``.  Identity entries are no-ops.  A moved
        source that is not also a move destination is vacated.
        """

        self._validate_state(state)
        flat = remap.reshape(-1).tolist() if isinstance(remap, torch.Tensor) else list(remap)
        if len(flat) != self._capacity:
            raise ValueError(f"slot remap must contain {self._capacity} entries")
        normalized = [int(slot) for slot in flat]
        if any(slot < 0 or slot >= self._capacity for slot in normalized):
            raise ValueError("slot remap source is outside the seed-state capacity")
        moves = [(old_slot, new_slot) for new_slot, old_slot in enumerate(normalized) if old_slot != new_slot]
        if not moves:
            return
        moved_sources_list = [old_slot for old_slot, _ in moves]
        if len(set(moved_sources_list)) != len(moved_sources_list):
            raise ValueError("slot remap cannot copy one seed stream into multiple destinations")

        snapshots = [
            self._checkpoint_slot(state, slot) if state.active[slot] else None for slot in range(self._capacity)
        ]
        moved_sources = {old_slot for old_slot, _ in moves}
        moved_destinations = {new_slot for _, new_slot in moves}
        for old_slot, new_slot in moves:
            checkpoint = snapshots[old_slot]
            if checkpoint is None:
                self._clear_slot(state, new_slot)
            else:
                self._restore_checkpoint(state, new_slot, checkpoint)
        for old_slot in moved_sources - moved_destinations:
            self._clear_slot(state, old_slot)
        self._write_current_values(state)

    def suspend(self, state: SeedState, slot: int) -> SeedSlotState:
        """Detach and return an immutable checkpoint for a preempted request."""

        self._validate_state(state)
        slot = self._validate_slot(slot, label="suspend slot")
        if not state.active[slot]:
            raise RuntimeError(f"cannot suspend inactive seed slot {slot}")
        checkpoint = self._checkpoint_slot(state, slot)
        self._clear_slot(state, slot)
        self._write_current_values(state)
        return checkpoint

    def resume(self, state: SeedState, slot: int, checkpoint: SeedSlotState) -> None:
        """Restore a checkpoint into an empty slot without changing its stream."""

        self._validate_state(state)
        slot = self._validate_slot(slot, label="resume slot")
        if state.active[slot]:
            raise RuntimeError(f"cannot resume into active seed slot {slot}")
        if checkpoint.request_seed is not None:
            collision = any(
                other != slot
                and state.active[other]
                and state.request_seeds[other] == checkpoint.request_seed
                and state.salts[other] == checkpoint.salt
                for other in range(self._capacity)
            )
            if collision:
                raise RuntimeError("cannot resume an equal-seed checkpoint after its salt was reused")
        self._restore_checkpoint(state, slot, checkpoint)
        self._write_current_values(state)

    def cleanup(self, state: SeedState, live_slots: Iterable[int]) -> None:
        """Remove completed/inactive slot state and clear stale buffer rows."""

        self._validate_state(state)
        live = set(self._normalize_slots(live_slots, label="live slot"))
        removed = [slot for slot in range(self._capacity) if state.active[slot] and slot not in live]
        if not removed:
            return
        for slot in removed:
            self._clear_slot(state, slot)
        self._write_current_values(state)

    def restore_defaults(self, state: SeedState) -> None:
        """Restore construction defaults without discarding request streams."""

        self._validate_state(state)
        self._write_values(state, self._default_values)

    def reset(self, state: SeedState) -> None:
        """Clear every request stream and restore construction defaults."""

        self._validate_state(state)
        for slot in range(self._capacity):
            self._clear_slot(state, slot, reseed_unseeded=True)
        self._write_values(state, self._default_values)

    # Internal state operations -------------------------------------------------

    def _validate_state(self, state: SeedState) -> None:
        if not isinstance(state, SeedState):
            raise TypeError("state must be a caller-owned SeedState")
        if state.capacity != self._capacity:
            raise ValueError(
                f"SeedState capacity {state.capacity} does not match seed-buffer capacity {self._capacity}"
            )

    def _validate_slot(self, slot: int, *, label: str) -> int:
        slot = int(slot)
        if slot < 0 or slot >= self._capacity:
            raise ValueError(f"{label} {slot} is outside the seed-state capacity")
        return slot

    def _normalize_slots(self, slots: Iterable[int], *, label: str) -> tuple[int, ...]:
        normalized = tuple(self._validate_slot(slot, label=label) for slot in slots)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{label}s must be unique")
        return normalized

    def _request_ordered_seeds(self, seeds, count: int) -> tuple[int | None, ...]:
        if isinstance(seeds, torch.Tensor):
            if seeds.ndim == 0:
                seeds = seeds.item()
            else:
                seeds = seeds.reshape(-1).tolist()
        if isinstance(seeds, (list, tuple)):
            if len(seeds) != count:
                raise ValueError(f"expected {count} request seeds, got {len(seeds)}")
            return tuple(self._normalize_seed(seed) for seed in seeds)
        return tuple(self._normalize_seed(seeds) for _ in range(count))

    def _slot_indexed_seed(self, seeds, slot: int) -> int | None:
        if seeds is None:
            return None
        if isinstance(seeds, torch.Tensor):
            if seeds.ndim == 0:
                return self._normalize_seed(seeds.item())
            flat = seeds.reshape(-1)
            return None if slot >= flat.numel() else self._normalize_seed(flat[slot].item())
        if isinstance(seeds, (list, tuple)):
            return None if slot >= len(seeds) else self._normalize_seed(seeds[slot])
        return self._normalize_seed(seeds)

    @staticmethod
    def _normalize_seed(seed) -> int | None:
        if seed is None:
            return None
        if isinstance(seed, torch.Tensor):
            if seed.numel() != 1:
                raise ValueError("each request seed tensor must contain exactly one value")
            seed = seed.item()
        return int(seed)

    @staticmethod
    def _position_for_slot(positions, slot: int) -> int | None:
        if positions is None:
            return None
        if isinstance(positions, dict):
            value = positions.get(slot)
            return None if value is None else int(value)
        if isinstance(positions, torch.Tensor):
            if positions.ndim == 0:
                return int(positions.item())
            flat = positions.reshape(-1)
            return None if slot >= flat.numel() else int(flat[slot].item())
        if isinstance(positions, (list, tuple)):
            if slot >= len(positions) or positions[slot] is None:
                return None
            return int(positions[slot])
        return int(positions)

    def _next_free_salt(self, state: SeedState, slot: int, seed: int) -> int:
        taken = {
            state.salts[other]
            for other in range(self._capacity)
            if other != slot and state.active[other] and state.request_seeds[other] == seed
        }
        salt = 0
        while salt in taken:
            salt += 1
        return salt

    def _register_slot(self, state: SeedState, slot: int, seed: int | None) -> None:
        state.active[slot] = True
        state.request_seeds[slot] = seed
        state.token_counters[slot] = 0
        state.salts[slot] = 0 if seed is None else self._next_free_salt(state, slot, seed)
        state.unseeded_rngs[slot].seed(self._entropy_factory(64) if seed is None else int(seed))
        state.last_absolute_positions[slot] = None
        state.current_device_seeds[slot] = None

    def _clear_slot(self, state: SeedState, slot: int, *, reseed_unseeded: bool = False) -> None:
        state.active[slot] = False
        state.request_seeds[slot] = None
        state.token_counters[slot] = 0
        state.salts[slot] = 0
        if reseed_unseeded:
            state.unseeded_rngs[slot].seed(self._entropy_factory(64))
        state.last_absolute_positions[slot] = None
        state.current_device_seeds[slot] = None

    @staticmethod
    def _checkpoint_slot(state: SeedState, slot: int) -> SeedSlotState:
        return SeedSlotState(
            request_seed=state.request_seeds[slot],
            token_counter=state.token_counters[slot],
            salt=state.salts[slot],
            unseeded_rng_state=copy.deepcopy(state.unseeded_rngs[slot].getstate()),
            last_absolute_position=state.last_absolute_positions[slot],
            current_device_seed=state.current_device_seeds[slot],
        )

    @staticmethod
    def _restore_checkpoint(state: SeedState, slot: int, checkpoint: SeedSlotState) -> None:
        state.active[slot] = True
        state.request_seeds[slot] = checkpoint.request_seed
        state.token_counters[slot] = checkpoint.token_counter
        state.salts[slot] = checkpoint.salt
        rng = random.Random()
        rng.setstate(checkpoint.unseeded_rng_state)
        state.unseeded_rngs[slot] = rng
        state.last_absolute_positions[slot] = checkpoint.last_absolute_position
        state.current_device_seeds[slot] = checkpoint.current_device_seed

    def _write_current_values(self, state: SeedState) -> None:
        values = list(self._default_values)
        for slot in range(self._capacity):
            if state.active[slot] and state.current_device_seeds[slot] is not None:
                values[slot] = state.current_device_seeds[slot]
        self._write_values(state, values)

    def _write_values(self, state: SeedState, values: Sequence[int]) -> None:
        if len(values) != self._capacity:
            raise ValueError(f"expected {self._capacity} seed values, got {len(values)}")
        try:
            normalized = [int(value) for value in values]
        except (TypeError, ValueError) as error:
            raise ValueError("device seed values must be integer-like") from error
        if any(value < 0 or value > MAX_UINT32 for value in normalized):
            raise ValueError("device seed values must be in [0, UINT32_MAX]")

        # Materialize before update.  LazyBuffer.update() only replaces the
        # future source when unmaterialized, but request values must reach the
        # stable handle captured by Sampling1D eager/trace execution now.
        self.get_seed_device_buffer()
        source = torch.tensor(
            normalized,
            dtype=self._default_source.dtype,
            device=self._default_source.device,
        ).reshape(self._default_source.shape)
        try:
            self._seed_buffer.update(source)
        finally:
            # Request state must never become the construction source used after
            # Sampling1D.release() and later rematerialization.
            self._seed_buffer.source = self._default_source.detach().clone()
        state.buffer_is_default = tuple(normalized) == self._default_values
