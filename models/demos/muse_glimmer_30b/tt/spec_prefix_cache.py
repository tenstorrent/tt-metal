# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host bookkeeping for prefix caching on the DFlash2 (speculative decode) rail. No ttnn, no device.

The spec adapter owns its KV: engine slot u owns target page row u and drafter ring u, and vLLM's block ids are ignored
(generator_vllm.py module docstring), so a vLLM prefix hit restores nothing by itself. A cached prefix is therefore kept
by the engine slot that computed it: a finished conversation leaves its slot idle with
  - the full-attention KV of its prompt (and output) in the slot's target page row,
  - a shadow copy of the 39 target sliding rings and of the 5 drafter rings, taken at the end of its prefill (the live
    rings are overwritten by decode),
and a later prompt that extends it is given that engine slot back (the adapter's phys map is a permutation of idle
slots, no KV moves) and resumes at floor(P / ALIGN) * ALIGN from the shadow rings.

This module decides which slot, where to resume and what the drafter misses. Design and the device work still to do:
profile/claude/projects/muse-glimmer-30b/AGENTX_SPEC_CACHE_ANALYSIS.md.
"""

from dataclasses import dataclass, field

from models.demos.muse_glimmer_30b.tt.prefix_cache import CACHE_ALIGN, chain_hashes

DRAFTER_WINDOW = 2048  # the drafter's context ring holds the last sliding_window rows (tap_window, DFlash2 config)
DRAFTER_ROW_ALIGN = 32  # context rows are written 32 at a time (prefill_with_taps: s0 is 32-aligned)


def drafter_rows_needed(prompt_len: int) -> tuple:
    """Context rows [lo, hi) the drafter needs after a prompt of prompt_len tokens (what a from-scratch prefill writes)."""
    return max(0, (prompt_len - DRAFTER_WINDOW) // DRAFTER_ROW_ALIGN * DRAFTER_ROW_ALIGN), prompt_len


def drafter_gap(old_prompt_len: int, start: int, new_prompt_len: int) -> int:
    """Context rows the drafter lacks after a resume at `start`: rows needed for the new prompt, minus the rows of the
    suffix taps [start, new_prompt_len) and of the shadow drafter ring [lo_old, old_prompt_len). Missing rows only lower
    the draft acceptance; the target verifies every token, so outputs are unchanged."""
    lo, hi = drafter_rows_needed(new_prompt_len)
    have_lo_old, _ = drafter_rows_needed(old_prompt_len)
    shadow = (have_lo_old, min(old_prompt_len, start))
    gap = 0
    for r0 in range(lo, min(hi, start), DRAFTER_ROW_ALIGN):
        if not (shadow[0] <= r0 < shadow[1]):
            gap += DRAFTER_ROW_ALIGN
    return gap


@dataclass
class _Retained:
    digests: list
    prompt_len: int
    boundary: int  # floor(prompt_len / ALIGN) * ALIGN: the only resume position the shadow rings serve
    tick: int


@dataclass
class SpecResumePlan:
    engine_slot: int
    start: int
    drafter_gap_rows: int


@dataclass
class SlotPrefixCache:
    """Which idle engine slot holds which prompt, with an LRU order for choosing a slot to overwrite."""

    n_slots: int
    retained: dict = field(default_factory=dict)  # engine slot -> _Retained (only while idle)
    _tick: int = 0
    hits: int = 0
    misses: int = 0

    def record_prefill(self, engine_slot: int, tokens) -> None:
        """Call after a prefill (full or resumed) finished in engine_slot: its shadow rings now match `tokens`."""
        self._tick += 1
        t = len(tokens)
        self.retained[engine_slot] = _Retained(chain_hashes(tokens), t, t // CACHE_ALIGN * CACHE_ALIGN, self._tick)

    def plan(self, tokens, cached_tokens: int, idle_slots) -> "SpecResumePlan | None":
        """The idle engine slot to resume in, or None (prefill from scratch). A slot qualifies when the new prompt
        repeats its prompt up to the boundary (>= ALIGN) and vLLM reports at least that many tokens cached."""
        if cached_tokens < CACHE_ALIGN:
            return None
        digests = chain_hashes(tokens)
        best = None
        for e in idle_slots:
            r = self.retained.get(e)
            if r is None or r.boundary < CACHE_ALIGN or r.boundary > min(cached_tokens, len(tokens) - 1):
                continue
            k = r.boundary // CACHE_ALIGN
            if len(digests) < k or digests[k - 1] != r.digests[k - 1]:
                continue
            if best is None or r.boundary > self.retained[best].boundary:
                best = e
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        r = self.retained[best]
        return SpecResumePlan(best, r.boundary, drafter_gap(r.prompt_len, r.boundary, len(tokens)))

    def victim(self, idle_slots) -> int:
        """Idle engine slot to overwrite for a prompt that resumes nothing: one with no retained prompt first, then
        the least recently used."""
        idle = list(idle_slots)
        empty = [e for e in idle if e not in self.retained]
        if empty:
            return min(empty)
        return min(idle, key=lambda e: self.retained[e].tick)

    def drop(self, engine_slot: int) -> None:
        self.retained.pop(engine_slot, None)


def swap_phys(phys: list, plugin_slot: int, engine_slot: int) -> list:
    """New plugin-slot -> engine-slot map in which plugin_slot uses engine_slot; the engine slot it replaces goes to
    the plugin slot that had engine_slot (both must be idle: no live request, no pending verify)."""
    out = list(phys)
    other = out.index(engine_slot)
    out[other], out[plugin_slot] = out[plugin_slot], engine_slot
    return out
