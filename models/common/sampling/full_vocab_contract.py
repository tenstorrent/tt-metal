"""Host-side request contract for a future full-vocabulary device sampler.

This module does not advertise a device capability or execute sampling.  It
preserves the user-visible distinction that the existing ``TTSampling`` path
currently loses when it clamps ``top_k <= 0`` to its 32-candidate kernel.
Keeping classification separate from execution lets the device implementation
fail closed until mixed-row, RNG, and numerical contracts are qualified.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Callable, Literal, Sequence


_UINT64_MASK = (1 << 64) - 1
_INACTIVE_DEVICE_SEED = (1 << 32) - 1
_USABLE_DEVICE_SEEDS = _INACTIVE_DEVICE_SEED - 1
_TOKEN_COUNTER_STRIDE = 0x9E3779B1
_MAX_MANAGED_SUBDRAWS = 16


SamplingRowMode = Literal["inactive", "greedy", "bounded", "unrestricted"]


@dataclass(frozen=True)
class SamplingRowContract:
    slot: int
    mode: SamplingRowMode
    temperature: float
    top_p: float
    top_k: int
    seed: int | None


@dataclass(frozen=True)
class FullVocabularyBatchContract:
    rows: tuple[SamplingRowContract, ...]

    @property
    def active_modes(self) -> frozenset[SamplingRowMode]:
        return frozenset(row.mode for row in self.rows if row.mode != "inactive")

    @property
    def needs_full_vocabulary(self) -> bool:
        return "unrestricted" in self.active_modes

    @property
    def is_mixed(self) -> bool:
        return len(self.active_modes) > 1

    @property
    def unrestricted_slots(self) -> tuple[int, ...]:
        return tuple(row.slot for row in self.rows if row.mode == "unrestricted")

    @property
    def unrestricted_nucleus_slots(self) -> tuple[int, ...]:
        """Rows that additionally require a full-vocabulary nucleus boundary.

        This is deliberately only a requirement classification.  It does not
        imply that the runtime has an exact full-vocabulary sort or CDF path.
        """

        return tuple(
            row.slot for row in self.rows if row.mode == "unrestricted" and row.top_p < 1.0
        )


@dataclass(frozen=True)
class ManagedDrawSeedPlan:
    """Per-subdraw device seeds; inactive rows carry the TTNN skip sentinel."""

    seeds_by_subdraw: tuple[tuple[int, ...], ...]
    draws_per_slot: tuple[int, ...]


def _managed_draw_seed(root: int, token_counter: int, salt: int, subdraw: int) -> int:
    """Domain-separated uint32 seed for one request token/subdraw.

    The token counter is an affine permutation over the 2**32-2 usable device
    seeds (the stride is coprime to that modulus), so one request/subdraw does
    not repeat a seed until that counter space wraps.  Different request roots,
    salts, or subdraw streams can still collide: a 32-bit device seed cannot
    provide a global collision-free mapping from those larger domains.
    """

    value = int(root) & _UINT64_MASK
    value ^= ((int(salt) + 1) * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= ((int(subdraw) + 1) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value = (value ^ (value >> 31)) & _UINT64_MASK
    # Zero means "do not reseed" and all-ones means "inactive" to TTNN.
    base = value % _USABLE_DEVICE_SEEDS
    return ((base + int(token_counter) * _TOKEN_COUNTER_STRIDE) % _USABLE_DEVICE_SEEDS) + 1


class ManagedPerSlotDrawSeeds:
    """Opt-in RNG state for composed device sampling algorithms.

    One request-owned root follows each slot through remaps.  A token advances
    its counter once regardless of how many internal device draws its selected
    algorithm needs; subdraw seeds are domain-separated from that token root.
    This prevents a bounded row's stream from changing merely because another
    row in the same batch needs an unrestricted multi-draw algorithm.

    The class only produces scalar seeds.  Random variates and all probability
    math remain device operations.  Callers must explicitly admit/reset a slot
    before drawing from it and explicitly deactivate it on request departure.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        entropy: Callable[[], int] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._entropy = entropy or (lambda: secrets.randbits(64))
        self._roots: list[int | None] = [None] * batch_size
        self._salts: list[int] = [0] * batch_size
        self._token_counters: list[int] = [0] * batch_size

    def reset_slot(self, slot: int, request_seed: int | None, *, salt: int = 0) -> None:
        self._validate_slot(slot)
        if request_seed is not None:
            if isinstance(request_seed, bool) or int(request_seed) != request_seed:
                raise ValueError("request_seed must be an integer or None")
            root = int(request_seed) & _UINT64_MASK
        else:
            root = int(self._entropy()) & _UINT64_MASK
        if salt < 0:
            raise ValueError("salt must be non-negative")
        self._roots[slot] = root
        self._salts[slot] = int(salt)
        self._token_counters[slot] = 0

    def deactivate_slots_except(self, live_slots: Sequence[int]) -> None:
        live = self._validated_slots(live_slots)
        for slot in set(range(self.batch_size)) - live:
            self._clear_slot(slot)

    def remap(self, remap: Sequence[int]) -> None:
        if len(remap) != self.batch_size:
            raise ValueError("remap must contain exactly batch_size entries")
        if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in remap):
            raise ValueError("remap entries must be integer slot indices")
        if any(slot < 0 or slot >= self.batch_size for slot in remap):
            raise ValueError("remap entries must be in range")
        moves = [(old, new) for new, old in enumerate(remap) if old != new]
        if not moves:
            return
        old_roots = list(self._roots)
        old_salts = list(self._salts)
        old_counters = list(self._token_counters)
        sources = {old for old, _ in moves}
        destinations = {new for _, new in moves}
        for old, new in moves:
            self._roots[new] = old_roots[old]
            self._salts[new] = old_salts[old]
            self._token_counters[new] = old_counters[old]
        for old in sources - destinations:
            self._clear_slot(old)

    def align_token_counters(self, positions: Sequence[int], slots: Sequence[int], *, offset: int = 1) -> None:
        if len(positions) != self.batch_size:
            raise ValueError("positions must contain exactly batch_size entries")
        for slot in self._validated_slots(slots):
            if self._roots[slot] is None:
                raise RuntimeError(f"slot {slot} has no admitted request seed root")
            position = positions[slot]
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("active positions must be non-negative integers")
            self._token_counters[slot] = position + offset

    def next_plan(self, draws_per_slot: Sequence[int]) -> ManagedDrawSeedPlan:
        if len(draws_per_slot) != self.batch_size:
            raise ValueError("draws_per_slot must contain exactly batch_size entries")
        draws = []
        for slot, count in enumerate(draws_per_slot):
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= _MAX_MANAGED_SUBDRAWS:
                raise ValueError(
                    f"draws_per_slot[{slot}] must be an integer in [0, {_MAX_MANAGED_SUBDRAWS}]"
                )
            if count and self._roots[slot] is None:
                raise RuntimeError(f"slot {slot} has no admitted request seed root")
            draws.append(count)

        seeds_by_subdraw = []
        for subdraw in range(max(draws, default=0)):
            values = []
            for slot, count in enumerate(draws):
                if subdraw >= count:
                    values.append(_INACTIVE_DEVICE_SEED)
                    continue
                values.append(
                    _managed_draw_seed(
                        self._roots[slot], self._token_counters[slot], self._salts[slot], subdraw
                    )
                )
            seeds_by_subdraw.append(tuple(values))
        for slot, count in enumerate(draws):
            if count:
                self._token_counters[slot] += 1
        return ManagedDrawSeedPlan(tuple(seeds_by_subdraw), tuple(draws))

    def _validate_slot(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < self.batch_size:
            raise ValueError("slot must be an in-range integer")

    def _validated_slots(self, slots: Sequence[int]) -> set[int]:
        for slot in slots:
            self._validate_slot(slot)
        result = set(slots)
        if len(result) != len(slots):
            raise ValueError("slots must be distinct")
        return result

    def _clear_slot(self, slot: int) -> None:
        self._roots[slot] = None
        self._salts[slot] = 0
        self._token_counters[slot] = 0


def classify_sampling_batch(
    *,
    temperature: Sequence[float],
    top_p: Sequence[float],
    top_k: Sequence[int],
    seeds: Sequence[int | None],
    active_slots: Sequence[int],
    batch_size: int,
    vocab_size: int,
    max_bounded_top_k: int,
) -> FullVocabularyBatchContract:
    """Classify rows without normalizing unrestricted requests to bounded k.

    ``top_k <= 0`` and ``top_k >= vocab_size`` both mean unrestricted in the
    vLLM-facing contract.  Greedy is classified first because its distribution
    is exactly argmax regardless of an unrestricted top-k sentinel.
    """

    if batch_size <= 0 or vocab_size <= 0 or max_bounded_top_k <= 0:
        raise ValueError("batch_size, vocab_size, and max_bounded_top_k must be positive")
    fields = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seeds": seeds,
    }
    for name, values in fields.items():
        if len(values) != batch_size:
            raise ValueError(f"{name} must contain exactly batch_size={batch_size} entries")

    if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in active_slots):
        raise ValueError("active_slots must contain integer slot indices")
    active = set(active_slots)
    if len(active) != len(active_slots) or any(slot < 0 or slot >= batch_size for slot in active):
        raise ValueError("active_slots must contain distinct in-range slot indices")

    rows = []
    for slot in range(batch_size):
        temp = float(temperature[slot])
        probability = float(top_p[slot])
        raw_k = top_k[slot]
        if isinstance(raw_k, bool):
            raise ValueError(f"top_k[{slot}] must be an integer")
        try:
            k = int(raw_k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"top_k[{slot}] must be an integer") from exc
        if k != raw_k:
            raise ValueError(f"top_k[{slot}] must be an integer")
        seed = seeds[slot]
        if seed is not None:
            if isinstance(seed, bool):
                raise ValueError(f"seeds[{slot}] must be an integer or None")
            try:
                normalized_seed = int(seed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"seeds[{slot}] must be an integer or None") from exc
            if normalized_seed != seed:
                raise ValueError(f"seeds[{slot}] must be an integer or None")
            seed = normalized_seed
        if not math.isfinite(temp) or temp < 0:
            raise ValueError(f"temperature[{slot}] must be finite and non-negative")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"top_p[{slot}] must be finite and in [0, 1]")

        if slot not in active:
            mode: SamplingRowMode = "inactive"
        elif temp == 0 or k == 1 or probability == 0:
            mode = "greedy"
        elif k <= 0 or k >= vocab_size:
            mode = "unrestricted"
        elif k <= max_bounded_top_k:
            mode = "bounded"
        else:
            raise ValueError(
                f"top_k[{slot}]={k} exceeds bounded device limit {max_bounded_top_k} "
                "but is smaller than vocab_size; no exact device algorithm is declared"
            )
        rows.append(SamplingRowContract(slot, mode, temp, probability, k, seed))
    return FullVocabularyBatchContract(tuple(rows))
