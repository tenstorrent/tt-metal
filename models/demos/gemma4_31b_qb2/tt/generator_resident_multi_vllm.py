# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: the resident decode engine serving up to 32 users, with per-request speculative state (S0).

Plan: bench/resident/M5B_PLAN_v2.md §8 lane S, step S0. Contract: M5_DESIGN.md §4, M5_DESIGN_ADDENDUM_1.md §2 (e, h),
M5_DESIGN_v2_ADDENDUM_1.md §3, m5/M1_INTERFACE_M5_ADDENDUM.md, m5/K2A_DESIGN_v2_ADDENDUM_1.md §2 and §4. Results and
open items: bench/resident/s0/S0_RESULTS.md. On top of the M1 adapter (pools, table translation, ttnn prefill, the
sampling trace, the L1 guard):

- Stable engine slots. The plugin's lane batch exists only with tt_data_parallel_size > 1 (model_runner.py
  `_is_lane_mode`); this engine is TP4 DP1, so every step is front-packed and rows move whenever the persistent batch
  condenses. A request keeps one engine slot for its life instead, keyed by its first full-layer block id (M1's
  prefill-history key: stable across prefill chunks, reorders and condense; unique, because Gemma4 QB2 refuses prefix
  caching). The plugin's own state slots (`empty_slots`, `slot_remap`, `note_state_slots_moved`) are followed only to
  map release_request(slot) to a request, and every reload checks them against the keys. No per-slot state is copied.
- Per-request state (SlotState): step, offer, regime, suppressed, exhausted, TapBook, policy and counters, created
  fresh at the request's first prefill chunk and dropped at its release or preemption.
- decode_forward over B rows: launch row r is decode row r, in its request's engine slot; empty rows are inert
  (position -1). A live slot's tables go to its row of the engine's [users, 6, P] only when they changed.
- The host guard (M1_INTERFACE_M5_ADDENDUM §5): a launch never carries more users than its program serves; checked
  for max_num_seqs at warmup and before every launch.
- Per-user status (§1): a flag names the flagged users' requests and quarantines their slots. The plugin has no
  per-request failure path, so the step still raises.
- propose_draft_tokens for every row: tap bookkeeping for the requests holding a tap slot (the first TAP_SLOTS
  admitted, M6a's policy), then the request's own policy. Above DRAFT_ROWS live rows every row declines: SpecPlan's
  k_by_rows has no K = 0 (vllm_tt_plugin/spec_decode.py, SpecPlan.__post_init__).
- The tap upkeep of a declining request is deferred into the next launch's device window: launch s's taps stay in
  taps_buffer(s % 2) through launch s + 1 (bench/resident/m2_1/M2_1_DESIGN.md, taps overlapped after the launch).

The drafter and the tap store are interfaces (DraftSource and TapStore below), supplied by M6's resident drafter and
M6a's tap layout. Until then S0 runs them as host fakes (bench/resident/s0/fakes.py).
"""

import os
from collections import Counter
from dataclasses import dataclass, field

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt import resident_taps
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import (
    _SAMPLING_FIELDS,
    PLACEHOLDER_TOKEN_ID,
    Gemma4ResidentSpecForCausalLM,
    Offer,
    ResidentSpecPolicy,
    _Params,
)
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import (
    DEFAULT_L1_TOP,
    RAW_WEIGHT_CACHE,
    Gemma4ResidentForCausalLM,
    _PrefillGenerator,
)
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model

MAX_USERS = 32  # one tile of feed rows (M1_INTERFACE ROWS)
TAP_SLOTS = 16  # M6a slot policy (M5B_PLAN_v2 §8): the first 16 admitted keep tap slots; the others never draft
DRAFT_ROWS = 16  # above this many live rows every row declines (M5B_PLAN_v2 §8 S2)
# Draft caps by live rows (M5B_PLAN_v2 §8 S2: blocks of 8 / 4 / 2 rows at U <= 4 / 8 / 16). The (32, 1) entry only
# covers max_num_seqs for admission: the adapter declines every row above DRAFT_ROWS.
K_BY_ROWS = ((4, 7), (8, 3), (16, 1), (32, 1))
ENGINE_STATUS_USER = 8  # status word of user block 0's nibble (M1_INTERFACE_M5_ADDENDUM §1)
VERIFY_BLOCKS = (1, 2, 4, 8)  # user-block rows of the R = 32 verify launches (1 + the K_BY_ROWS caps)


def launch_policy():
    """GEMMA4_RESIDENT_LAUNCH_ROWS: fixed (option D: every launch at the largest R, so a request's numerics never
    depend on its co-tenants) or by_users (option S: the smallest R whose program serves the live users)."""
    mode = os.environ.get("GEMMA4_RESIDENT_LAUNCH_ROWS", "fixed")
    if mode not in ("fixed", "by_users"):
        raise ValueError(f"GEMMA4_RESIDENT_LAUNCH_ROWS={mode!r}: expected fixed or by_users")
    return mode


def draft_cap(live_rows):
    """K_BY_ROWS's cap for this many live rows, 0 above DRAFT_ROWS."""
    if live_rows > DRAFT_ROWS:
        return 0
    return next(k for rows, k in K_BY_ROWS if live_rows <= rows)


@dataclass
class SlotStep:
    """What a request's last decode step was, for the propose that follows it (M2's Step plus the launch row)."""

    kind: str  # "ordinary" or "verify"
    launch: int  # engine.launches before the launch: its taps are taps_buffer(launch % 2)
    position: int  # the request's first input position in that launch
    row: int  # the request's first launch row
    drafts: int = 0  # n on a verify
    host: bool = False  # the step was host-sampled (sampling_params is None)
    dropped: bool = False  # an offer was outstanding and the step did not verify it


@dataclass
class SlotState:
    """One request's state, keyed by its engine slot (M5_DESIGN_ADDENDUM_1 §2 e). Never reused across requests."""

    key: int  # first full-layer block id (scaled to the aliased full view, as M1's prefill-history key)
    slot: int  # engine slot
    tap_slot: object  # tap store slot, or None (never drafts)
    policy: ResidentSpecPolicy
    book: object  # resident_taps.TapBook, or None without a tap slot
    position: int = -1  # the next plain step's input position (host mirror)
    step: object = None
    offer: object = None
    regime: str = "plain"
    suppressed: bool = False
    exhausted: bool = False
    seeded: bool = False
    flagged: int = 0  # status flags the engine raised for this request
    deferred: int = 0  # tap upkeep entries waiting in deferred_taps
    counters: Counter = field(default_factory=Counter)


class SlotTable:
    """Engine slots <-> requests (by key) <-> the plugin's state slots. Lowest free slot first, so the assignment is a
    function of the admission order alone."""

    def __init__(self, users, tap_slots):
        self.users = users
        self.free = list(range(users))
        self.free_taps = list(range(tap_slots))
        self.by_key = {}  # key -> engine slot
        self.states = {}  # engine slot -> SlotState
        self.plugin = {}  # plugin state slot -> engine slot

    def admit(self, key, plugin_slot, make_state):
        if key in self.by_key:
            raise ValueError(f"A prefill from position 0 names block {key}, which engine slot {self.by_key[key]} holds")
        if not self.free:
            raise RuntimeError(f"No free engine slot for block {key}: {self.users} slots, owners {self.by_key}")
        slot = self.free.pop(0)
        tap_slot = self.free_taps.pop(0) if self.free_taps else None
        state = make_state(key, slot, tap_slot)
        self.by_key[key], self.states[slot] = slot, state
        self.bind(plugin_slot, slot)
        return state

    def resume(self, key, plugin_slot):
        """A later prefill chunk: its request keeps its slot whatever plugin slot this chunk was given."""
        slot = self.by_key.get(key)
        if slot is None:
            raise ValueError(f"A prefill chunk continues block {key}, which no engine slot holds")
        self.bind(plugin_slot, slot)
        return self.states[slot]

    def bind(self, plugin_slot, slot):
        other = self.plugin.get(plugin_slot)
        if other is not None and other != slot and other in self.states:
            raise RuntimeError(f"Plugin slot {plugin_slot} is bound to engine slot {other}, not {slot}: {self.plugin}")
        self.plugin = {p: s for p, s in self.plugin.items() if s != slot}
        self.plugin[plugin_slot] = slot

    def slot_of(self, key):
        slot = self.by_key.get(key)
        if slot is None:
            raise RuntimeError(f"Decode row of block {key} has no engine slot (no prefill seen): {self.by_key}")
        return slot

    def moved(self, moves):
        """The plugin's gather moved state slot old -> new for every pair (a permutation)."""
        self.plugin = {moves.get(p, p): s for p, s in self.plugin.items()}

    def release(self, plugin_slot):
        slot = self.plugin.pop(plugin_slot, None)
        if slot is None:
            return None
        state = self.states.pop(slot)
        del self.by_key[state.key]
        self.free = sorted(self.free + [slot])
        if state.tap_slot is not None:
            self.free_taps = sorted(self.free_taps + [state.tap_slot])
        return state


class TapStore:
    """The tap buffers of TAP_SLOTS requests (M6a's layout supplies it; S0 runs a host fake). Each call is
    resident_taps.TapBuffers' for one tap slot; `stage` takes the request's rows [row, row + rows) of the launch."""

    slots = 0

    def stage(self, tap_slot, launch_taps, row, rows):
        raise NotImplementedError

    def ring_insert(self, tap_slot, ring_slot):
        raise NotImplementedError

    def flush(self, tap_slot, page, entries):
        raise NotImplementedError

    def write_prefill(self, tap_slot, parts, first_page):
        raise NotImplementedError

    def insert_page(self, tap_slot, page, rows, ring_slot):
        raise NotImplementedError

    def read_window(self, tap_slot, first_page):
        raise NotImplementedError

    def commit(self, tap_slot, rows):
        raise NotImplementedError

    def reset(self, tap_slot):
        raise NotImplementedError

    def seal(self):
        raise NotImplementedError


class DraftSource:
    """The drafter for TAP_SLOTS requests (M6's resident drafter supplies it). `draft` takes every drafting request
    of a step at once: [(tap_slot, anchor, position, committed_start, committed)] -> {tap_slot: drafts}."""

    max_positions = 0
    tap_layers = ()

    def tap_window(self, end):
        raise NotImplementedError

    def seed(self, tap_slot, windows, start, valid):
        raise NotImplementedError

    def draft(self, jobs):
        raise NotImplementedError

    def reset(self, tap_slot):
        raise NotImplementedError


class Gemma4ResidentMultiForCausalLM(Gemma4ResidentForCausalLM):
    model_capabilities = dict(Gemma4ResidentSpecForCausalLM.model_capabilities)
    ROWS = (MAX_USERS,)
    BLOCK = Gemma4ResidentSpecForCausalLM.BLOCK
    K = Gemma4ResidentSpecForCausalLM.K
    defer_taps = True  # a declining request's tap upkeep waits for the next launch's device window

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None, tap_layers=()):
        super().__init__(generator, max_batch_size, max_seq_len, vllm_config=vllm_config)
        self.tap_layers = tuple(tap_layers)
        self.full_layer = next(i for i, layer in enumerate(generator.model.layers) if not layer.sliding)
        self.slots = SlotTable(max_batch_size, min(TAP_SLOTS, max_batch_size))
        self.row_slots = [None] * max_batch_size  # engine slot of each decode row, as the device feed holds them
        self.launch_users = []  # engine slot of each user block of the last launch (status nibbles)
        self.pending_rows = None  # {row: engine slot} of the last reload, checked when the plugin's moves arrive
        groups = max(self.groups) + 1
        self.group_layers = [self.groups.index(g) for g in range(groups)]  # a layer of each vLLM group
        self.full_groups = {g for g, layer in zip(self.groups, generator.model.layers) if not layer.sliding}
        # each slot's plugin table rows as last written to the engine, [groups, users, pages]; -1 = must write
        self.slot_raw = torch.full((groups, max_batch_size, max_seq_len // 128), -1, dtype=torch.int32)
        self.seen = None  # (rows, plugin group tables) of the last table write: the engine holds them
        self.deferred_taps = []  # (state, step, committed) whose tap upkeep runs after the next launch
        self.taps = None  # TapStore
        self.drafter = None  # DraftSource
        self.counters = Counter()
        self.launch_mode = launch_policy()

    # ------------------------------------------------------------------ construction

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        """M1's construction (generator_resident_vllm.initialize_vllm_model), for 1..32 users."""
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= MAX_USERS:
            raise ValueError(f"Resident Gemma4 QB2 (multi-user) serves TP4, DP1, 1..{MAX_USERS} requests")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        tile_pad = os.environ.get("GEMMA4_RESIDENT_TILE_PAD", "1") == "1"
        model = Gemma4Model(mesh_device, decode_weights=False, prefill_tile_pad=tile_pad)
        return cls(_PrefillGenerator(model, max_seq_len=max_seq_len), max_batch_size, max_seq_len,
                   tap_layers=Gemma4ResidentSpecForCausalLM.drafter_tap_layers())

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """K = 7 with M5B's caps by live rows; argmax_ids only until S2 adds the logits mode."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if not 1 <= max_num_seqs <= MAX_USERS:
            return SpecReject(reason=f"Resident Gemma4 QB2 speculation serves 1..{MAX_USERS} requests")
        if requested_k != cls.K:
            return SpecReject(
                reason=f"DFlash2 block {cls.BLOCK} drafts exactly {cls.K} tokens; num_speculative_tokens={requested_k}",
                supported_k=(cls.K,),
            )
        caps = tuple((rows, k) for rows, k in K_BY_ROWS if rows < max_num_seqs)
        caps += (next((rows, k) for rows, k in K_BY_ROWS if rows >= max_num_seqs),)
        n, page = 5, resident_taps.TILE * resident_taps.TAP_WIDTH * 2
        per_slot = n * (2 * resident_taps.HISTORY_PAGES + resident_taps.RING_SLOTS + resident_taps.HISTORY_PAGES) * page
        return SpecPlan(
            effective_k=cls.K,
            lanes_per_request=1,
            extra_bytes_per_seq=per_slot,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=caps,
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """M1's pools and engine, then the tap store and the drafter (after the engine, before any trace: §8.1)."""
        cache = super().allocate_kv_cache_per_layer(per_layer_specs)
        self.taps, self.drafter = self.build_taps(), self.build_drafter()
        if self.drafter is not None and tuple(self.drafter.tap_layers) != self.tap_layers:
            raise ValueError(f"Drafter taps {self.drafter.tap_layers} differ from the engine's {self.tap_layers}")
        return cache

    def build_taps(self):
        """M6a's tap store; None until it exists (requests then never draft)."""
        return None

    def build_drafter(self):
        """M6's resident drafter; None until it exists."""
        return None

    def _build_engine(self):
        """The K2a engine: one R = 32 program, user blocks of one row."""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m5_3")
        resident = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.engine")
        setup = self.generator.model.shared_setup
        cfg = resident.EngineConfig(
            weight_cache=os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE),
            rows=self.ROWS,
            max_users=self.max_batch_size,
            capacity=self.max_seq_len,
            tap_layers=self.tap_layers,
            l1_top=int(os.environ.get("GRES_L1_TOP", DEFAULT_L1_TOP)),
            logits=True,
            embedding="resident",
            block={R: 1 for R in self.ROWS},
        )
        pools = resident.KVPools(
            kv=list(self.cache.kv),
            layer_group=list(self.groups),
            scratch=self._group_scratch(),
            rope={
                kind: (setup[("rope", kind)]["decode_cos"], setup[("rope", kind)]["decode_sin"])
                for kind in ("sliding_attention", "full_attention")
            },
        )
        logger.info("Building the resident multi-user engine from {}, rows {}, users {}", package, cfg.rows,
                    cfg.max_users)
        return resident.ResidentEngine(self.mesh, cfg, pools)

    def _users_served(self, R):
        """Users R's program serves. An engine without users_for is resident_m5..m5_3, whose attention serves user
        0's block only (M1_INTERFACE_M5_ADDENDUM §5)."""
        users_for = getattr(self.engine, "users_for", None)
        return 1 if users_for is None else int(users_for(R))

    def _launch_rows(self, users):
        rows = sorted(self.ROWS)
        if self.launch_mode == "fixed":
            return rows[-1]
        return next((R for R in rows if self._users_served(R) >= users), rows[-1])

    def _launch(self, R, M=1):
        """R's program with M-row user blocks: the configured block, or one of the engine's blocks_for(R)."""
        if M == self.engine.cfg.block_for(R):
            self.engine.launch(R)
            return
        blocks = getattr(self.engine, "blocks_for", None)
        if blocks is None or M not in blocks(R):
            raise ValueError(f"The engine has no R = {R} program with {M}-row user blocks (S0_RESULTS.md, engine)")
        self.engine.launch(R, block=M)

    def _guard_users(self, R, slots):
        """The host guard: the distinct live slots of a launch against what R's program serves."""
        users = len({s for s in slots if s is not None})
        if users > self._users_served(R):
            raise ValueError(f"A launch at R = {R} would carry {users} users; its program serves "
                             f"{self._users_served(R)} (M1_INTERFACE_M5_ADDENDUM §5)")

    # ------------------------------------------------------------------ prefill

    def _key(self, value):
        """A vLLM full-layer block id as the key M1 keeps prefill history under (scaled to the aliased view)."""
        return max(0, int(value)) * getattr(self.cache, "full_page_scale", 2)

    def _new_state(self, key, slot, tap_slot):
        book = resident_taps.TapBook() if tap_slot is not None and self.taps is not None else None
        return SlotState(key, slot, tap_slot if book is not None else None, ResidentSpecPolicy(self.K), book)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, sampling_params=None,
                        page_tables_per_layer=None, start_pos=None, empty_slots=None, **kwargs):
        """M1's prefill (same steps) into each request's engine slot, admitting a request at its first chunk."""
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        starts = [0] * len(prompt_lens) if start_pos is None else [int(x) for x in start_pos]
        lengths = [int(x) for x in prompt_lens]
        plugin_slots = list(range(len(lengths))) if empty_slots is None else [int(s) for s in empty_slots]
        source = page_tables_per_layer[self.full_layer]
        keys = [self._key(source[i, 0]) for i in range(len(lengths))]
        states = []
        for key, plugin_slot, start in zip(keys, plugin_slots, starts):
            if start == 0:
                state = self.slots.admit(key, plugin_slot, self._new_state)
                if state.tap_slot is not None:
                    self.taps.reset(state.tap_slot)
                    if self.drafter is not None:
                        self.drafter.reset(state.tap_slot)
                self.counters["requests"] += 1
            else:
                state = self.slots.resume(key, plugin_slot)
            states.append(state)
        engine_slots = [state.slot for state in states]
        tables = self._tables(kv_cache, page_tables_per_layer, slots=engine_slots)
        self.prefill_slot_keys = {
            s: k for s, k in self.prefill_slot_keys.items() if s not in plugin_slots and k not in keys
        }
        self.prefill_slot_keys.update(zip(plugin_slots, keys))
        novel = (kv_cache.batch, kv_cache.capacity, tuple(lengths), tuple(starts), False) not in (
            self.generator.prefill_shapes
        )
        if novel:
            self._release_sampling_trace()
        taps = {}
        if self.drafter is not None and any(state.book is not None for state in states):
            windows = [self.drafter.tap_window(end) if s.book is not None else None for s, end in zip(states, lengths)]
            captured = [dict.fromkeys(self.tap_layers) if s.book is not None else None for s in states]
            taps = dict(taps=captured, tap_windows=windows)
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=engine_slots,
            return_device=sampling_params is not None,
            start_positions=starts,
            history_keys=keys,
            **taps,
        )
        if novel:
            self._l1_guard(f"prefill {tuple(lengths)} from {tuple(starts)}")
        for i, (state, start, end) in enumerate(zip(states, starts, lengths)):
            state.position = end
            if taps and state.book is not None:
                self._record_prefill_taps(state, taps["taps"][i], taps["tap_windows"][i], start, end)
        if sampling_params is None:
            return outputs
        return self._sample_prefill(outputs, sampling_params, lengths)

    def _record_prefill_taps(self, state, captured, window, start, end):
        """M2's prefill taps (generator_resident_spec_vllm._record_prefill_taps) into the request's tap slot."""
        lo = self.generator.tap_capture_range(start, window)
        state.book.note_prefill(end)
        if lo is not None:
            first_page = resident_taps.page_of(lo[0])
            for page in range(first_page, resident_taps.page_of(lo[1] - 1) + 1):
                state.book.check_write(page)
            self.taps.write_prefill(state.tap_slot, {i: captured[layer] for i, layer in enumerate(self.tap_layers)},
                                    first_page)
        if end % resident_taps.TILE:
            page = resident_taps.page_of(end)
            rows = end % resident_taps.TILE
            self.taps.insert_page(state.tap_slot, page, rows, state.book.insert_page(page, rows))

    # ------------------------------------------------------------------ decode

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, spec_mode=None, num_valid_drafts=None,
                       accepted_counts=None, slot_remap=None, enable_trace=True, read_from_device=True,
                       sampling_params=None, page_tables_per_layer=None, reload_inputs=True, reload_page_table=False,
                       reload_sampling_params=True, reset_sampling_state=True, **kwargs):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if not enable_trace:
            raise ValueError("Resident decode is always a replay")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if spec_mode is not None:
            return self._verify(tokens, start_pos, kv_cache, spec_mode, num_valid_drafts, slot_remap,
                                sampling_params, page_tables_per_layer)
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        if self.feed_stale and not reload_inputs:
            raise ValueError("The resident feed was reset (engine rebuild or a verify); the step needs current inputs")
        if slot_remap is not None and not reload_inputs:
            raise ValueError("A slot remap moves rows: the step needs current inputs (reload_inputs)")
        batch = self.max_batch_size
        if reload_inputs or reload_page_table:
            sources = self._group_tables(page_tables_per_layer)
        if reload_inputs:
            positions = [int(p) for p in start_pos.reshape(-1)[:batch]]
            rows = self._layout(sources, positions, slot_remap)
            self._write_slot_tables(sources, rows)
            done = {self.slots.states[s].key for s in rows if s is not None}
            for key in done:
                kv_cache.prefill_history.pop(key, None)
            self.prefill_slot_keys = {slot: key for slot, key in self.prefill_slot_keys.items() if key not in done}
            self.engine.write_rows([0 if s is None else s for s in rows], positions,
                                   [int(t) for t in tokens.reshape(-1)[:batch]])
            self.positions, self.row_slots, self.feed_stale = positions, rows, False
        elif reload_page_table:
            self._write_slot_tables(sources, self.row_slots)
        live = [r for r, s in enumerate(self.row_slots) if s is not None and self.positions[r] >= 0]
        over = [(r, self.positions[r]) for r in live if self.positions[r] >= self.max_seq_len]
        if over:
            raise ValueError(f"No decode steps remain within the cache capacity: rows (row, position) {over}")
        R = self._launch_rows(len(live))
        self._guard_users(R, [self.row_slots[r] for r in live])
        host = sampling_params is None
        temps = [] if host else self._rows(sampling_params.temperature, batch)
        greedy = [r for r in live if not host and temps[r] == 0]
        sampled = not host and len(greedy) < len(live)
        if not host and (reload_sampling_params or reset_sampling_state):
            params = self._sampling(sampling_params, batch, start_pos, reset_seed=reset_sampling_state)
            self.generator.configure_sampling(batch=batch, **params)
        mode = ("none", True) if host else ("external", True) if sampled else ("argmax", False)
        if sampled and self.sampling_trace is None:
            self._capture_sampling()
            self._guard_after_capture()
        if mode != self.mode:
            self.engine.set_mode(feed=mode[0], write_logits=mode[1])
            self.mode = mode
        launch = self.engine.launches
        for r in live:
            state = self.slots.states[self.row_slots[r]]
            dropped, state.offer = state.offer is not None, None
            state.step = SlotStep("ordinary", launch, self.positions[r], r, host=host, dropped=dropped)
        self.launch_users = [self.row_slots[r] if r in live else None for r in range(R)]
        self._launch(R)
        if not host:
            self.positions = [p + 1 if p >= 0 else p for p in self.positions]
            for r in live:
                self.slots.states[self.row_slots[r]].position = self.positions[r]
        if host:
            self._drain_taps()
            logits = self._host_logits(batch).unsqueeze(1)
            self._check_status(self._status_words())
            return logits
        if sampled:
            self._replay_sampling()
            if greedy:
                self._restore_greedy(greedy)
        self._drain_taps()  # enqueued behind the step; the host work runs while the device decodes
        output = self.engine.tokens
        return self.process_decode_output_host(output, is_tokens=True) if read_from_device else output

    def _layout(self, sources, positions, slot_remap):
        """Engine slot of each decode row, checked against the plugin's state slots: row r's request holds plugin
        slot slot_remap[r] before this step's gather (r without one)."""
        keys = sources[self.groups[self.full_layer]][:, 0].tolist()
        rows, remap = [], None if slot_remap is None else [int(s) for s in slot_remap.reshape(-1)]
        for r, position in enumerate(positions):
            if position < 0:
                rows.append(None)
                continue
            slot = self.slots.slot_of(self._key(keys[r]))
            plugin_slot = r if remap is None else remap[r]
            if self.slots.plugin.get(plugin_slot) != slot:
                raise RuntimeError(f"Decode row {r} (block {int(keys[r])}, engine slot {slot}) arrives from plugin "
                                   f"slot {plugin_slot}, which is bound to {self.slots.plugin.get(plugin_slot)}")
            if self.slots.states[slot].flagged:
                raise RuntimeError(f"Engine slot {slot} was flagged 0x{self.slots.states[slot].flagged:x}; its "
                                   "request must not decode again")
            rows.append(slot)
        if len(set(s for s in rows if s is not None)) != sum(s is not None for s in rows):
            raise RuntimeError(f"Two decode rows share an engine slot: {rows}")
        self.pending_rows = {r: s for r, s in enumerate(rows) if s is not None}
        return rows

    def _group_tables(self, page_tables_per_layer):
        """One plugin table per vLLM group ([rows, pages], vLLM block ids); the layers of a group share it (as M1's
        _group_sources requires, checked here by identity first)."""
        sources = [page_tables_per_layer[i] for i in self.group_layers]
        for i, g in enumerate(self.groups):
            table = page_tables_per_layer[i]
            if table is not sources[g] and not torch.equal(table, sources[g]):
                raise ValueError(f"Layer {i} does not share the page table of vLLM group {g}")
        return sources

    def _write_slot_tables(self, sources, rows):
        """The live rows whose plugin tables differ from what their engine slot holds, translated into those slots'
        rows of [users, groups, pages] (M1's _tables + _resident_tables, per row, with the slot's own scratch page)."""
        live = [r for r, s in enumerate(rows) if s is not None]
        seen, self.seen = self.seen, (tuple(rows), sources)
        if not live:
            return
        width = sources[0].shape[1]
        front = live == list(range(len(live)))
        r_idx = torch.tensor(live)
        s_idx = torch.tensor([rows[r] for r in live])
        changed = torch.zeros(len(live), dtype=torch.bool)
        if seen is not None and seen[0] == tuple(rows) and front:
            # the same layout as the last write, which the engine holds: compare with those rows (views, no gather)
            for src, old in zip(sources, seen[1]):
                changed |= (src[: len(live)] != old[: len(live)]).any(dim=1)
        else:
            for g, src in enumerate(sources):
                mine = src[: len(live)] if front else src[r_idx]
                changed |= (mine != self.slot_raw[g, s_idx, :width]).any(dim=1)
        if not bool(changed.any()):
            return
        pick = changed.nonzero().flatten()
        r_idx, s_idx = r_idx[pick], s_idx[pick]
        update = torch.empty(len(pick), len(sources), self.slot_raw.shape[2], dtype=torch.int32)
        scale = getattr(self.cache, "full_page_scale", 2)
        scratch = torch.tensor(self._group_scratch(), dtype=torch.int32)[:, s_idx]
        columns = torch.arange(1, width + 1)
        for g, src in enumerate(sources):
            raw = src[r_idx].to(torch.int32)
            self.slot_raw[g, s_idx, :width] = raw
            table = raw.clamp_min(0) * (scale if g in self.full_groups else 1)
            # vLLM's null block is 0: a full-group 0 is unallocated, and sliding zeros after a row's last live block
            # are unallocated; both go to the slot's write-only scratch page. Sliding zeros before it are evicted
            # blocks outside the window and stay 0 (M1 _resident_tables).
            live_cols = table != 0
            if g not in self.full_groups:
                live_cols = columns <= (live_cols * columns).amax(dim=1, keepdim=True)
            update[:, g, :width] = torch.where(live_cols, table, scratch[g][:, None])
            update[:, g, width:] = scratch[g][:, None]
        self.engine.write_tables(update, slots=s_idx.tolist())
        self.counters["table_writes"] += 1
        self.counters["table_rows_written"] += len(pick)

    def _status_words(self):
        import ttnn

        return ttnn.get_device_tensors(self.engine.status)[0].cpu()

    def _check_status(self, status):
        """M1_INTERFACE_M5_ADDENDUM §1: word 1 is the OR of every user's flags, words 8..11 a nibble per user block
        of the launches since the last clear. The flagged users' requests are named and their slots quarantined; the
        other users' outputs of the launch are valid, but the plugin cannot fail one request, so the step raises."""
        import ttnn

        words = ttnn.to_torch(status).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        if int(words[1]) == 0:
            return
        flagged = []
        for u, slot in enumerate(self.launch_users):
            nibble = int(words[ENGINE_STATUS_USER + u // 8] >> (4 * (u % 8))) & 0xF
            if nibble and slot is not None:
                state = self.slots.states.get(slot)
                if state is not None:
                    state.flagged |= nibble
                flagged.append((u, slot, None if state is None else state.key, nibble))
        signed = [w - (1 << 32) if w >= 1 << 31 else w for w in words.tolist()]
        message = (f"Resident engine flagged user blocks (block, engine slot, request key, flags) {flagged}: flags "
                   f"0x{words[1]:x}, first row {signed[2]}, position {signed[3]}, page {signed[4]}, entry {signed[5]}")
        logger.error(message)
        self.engine.clear_status()
        self.mode = None
        self.feed_stale = True
        raise RuntimeError(message)

    def _restore_greedy(self, rows):
        """A mixed step's greedy rows take the hub argmax, not TTSampling's capped-bf16 argmax, which can break a tie
        another way (K2A_DESIGN_v2 §7): tokens = where(greedy mask, argmax_ids, tokens), one fixed program."""
        import ttnn

        mask = torch.zeros(1, 1, 1, MAX_USERS, dtype=torch.int32)
        mask[..., rows] = 1
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.engine.greedy_mask)
        ttnn.where(self.engine.greedy_mask, self.engine.argmax_ids, self.engine.tokens,
                   output_tensor=self.engine.tokens)
        self.counters["greedy_restores"] += 1

    # ------------------------------------------------------------------ verify

    def _verify(self, tokens, start_pos, kv_cache, spec_mode, num_valid_drafts, slot_remap, sampling_params,
                page_tables_per_layer):
        """One argmax_ids verify: live row b's [1, 1 + n_b] block as user block u at launch rows [u M, u M + M), M the
        smallest engine block holding 1 + max n_b; rows past 1 + n_b of a block are inert (-1). The step after a
        multi-token commit is a verify even when no row drafts (model_runner._step_verifies, unresolved commit): it
        launches with M = 1, as a plain step does."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode != "argmax_ids":
            raise ValueError(f"Resident speculation serves argmax_ids verifies only, got {spec_mode!r}")
        batch = self.max_batch_size
        width = tokens.shape[1]
        positions_in = start_pos.reshape(start_pos.shape[0], -1)
        live = [b for b in range(batch) if int(positions_in[b, 0]) >= 0]
        cap = draft_cap(len(live))
        drafts = max(int(num_valid_drafts[b]) for b in live)
        if drafts > cap:
            raise ValueError(f"A verify of {len(live)} live rows carries {drafts} drafts, above the cap {cap}")
        M = next(m for m in VERIFY_BLOCKS if m > drafts)
        R = max(self.ROWS)
        if len(live) * M > R:
            raise ValueError(f"{len(live)} users x {M} rows do not fit R = {R}")
        sources = self._group_tables(page_tables_per_layer)
        rows = self._layout(sources, [int(p) for p in positions_in[:batch, 0]], slot_remap)
        self._write_slot_tables(sources, rows)
        self._guard_users(R, [rows[b] for b in live])
        self.row_slots = rows
        slots, positions, ids_in, blocks = [0] * R, [-1] * R, [0] * R, []
        for u, b in enumerate(live):
            state = self.slots.states[rows[b]]
            n = int(num_valid_drafts[b])
            block = [int(t) for t in tokens[b, : n + 1]]
            p = int(positions_in[b, 0])
            offer, state.offer = state.offer, None
            if n:
                if offer is None or offer.anchor != block[0] or offer.position != p or offer.drafts[:n] != block[1:]:
                    raise ValueError(f"Verify block {block} at {p} is not the retained proposal {offer}")
                if p + M > self._spec_limit():
                    raise ValueError(f"Verify block [{p}, {p + M}) passes the limit {self._spec_limit()}")
            for j in range(n + 1):
                slots[u * M + j], positions[u * M + j], ids_in[u * M + j] = rows[b], p + j, block[j]
            blocks.append((u, b, state, n, p, offer))
        self.engine.write_rows(slots, positions, ids_in)
        temps = None if sampling_params is None else self._rows(sampling_params.temperature, batch)
        sampled = temps is not None and any(temps[b] > 0 for b in live)
        if sampled:
            self._configure_verify_sampling(sampling_params, blocks, M, R)
            if self.sampling_trace is None:
                self._capture_sampling()
                self._guard_after_capture()
                self.engine.write_rows(slots, positions, ids_in)
        mode = ("none", sampled)
        if mode != self.mode:
            self.engine.set_mode(feed=mode[0], write_logits=mode[1])
            self.mode = mode
        launch = self.engine.launches
        self.launch_users = [rows[b] for b in live] + [None] * (R // M - len(live))
        self._launch(R, M)
        if sampled:
            self._replay_sampling()
            greedy = [u * M + j for u, b, _, n, _, _ in blocks if temps[b] == 0 for j in range(n + 1)]
            if greedy:
                self._restore_greedy(greedy)
        self._drain_taps()
        ids = self.engine.read_tokens(R)
        self._check_status(self._status_words())
        self.feed_stale = True  # the engine's positions are the blocks'; the next ordinary decode reloads
        out = torch.full((tokens.shape[0], width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for u, b, state, n, p, offer in blocks:
            out[b, : n + 1] = ids[u * M : u * M + n + 1].to(torch.int32)
            state.step = SlotStep("verify", launch, p, u * M, drafts=n, dropped=offer is not None and n == 0)
            state.counters["verifies"] += 1
            state.counters["verified_drafts"] += n
        self.counters["verifies"] += 1
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out)

    def _configure_verify_sampling(self, sampling_params, blocks, M, R):
        """Every row of user u's block sampled with u's parameters; row j's seed is u's plain seed at p + j (M2 §6)."""

        def column(name, default, b):
            value = getattr(sampling_params, name, default)
            value = value.tolist() if isinstance(value, torch.Tensor) else value
            return list(value)[b] if isinstance(value, (list, tuple)) else value

        values = {name: [default] * R for name, default in _SAMPLING_FIELDS.items()}
        values["temperature"] = [0.0] * R
        positions = [0] * R
        for u, b, _, _, p, _ in blocks:
            for j in range(M):
                for name, default in _SAMPLING_FIELDS.items():
                    values[name][u * M + j] = column(name, default, b)
                positions[u * M + j] = p + j
        params = self._sampling(_Params(values), R, positions, reset_seed=True)
        self.generator.configure_sampling(batch=R, **params)

    # ------------------------------------------------------------------ propose

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        k = int(num_drafts)
        if k != self.K:
            raise ValueError(f"Expected {self.K} draft columns, got {k}")
        rows = committed_tokens.shape[0]
        ids = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        counts = accepted_counts.reshape(-1).tolist()
        tokens, positions = committed_tokens.tolist(), committed_positions.tolist()
        live = [r for r in range(min(rows, len(self.row_slots)))
                if positions[r][0] >= 0 and self.row_slots[r] is not None]
        cap = draft_cap(len(live))
        if cap and any(positions[r][counts[r] - 1] + self.BLOCK > self._spec_limit() for r in live):
            # The capacity-edge guard (VERIFY_M3_3 §5): a verify launch puts every live row in an M-row block, and the
            # engine checks a block's whole extent against its capacity (M1_INTERFACE_M5_ADDENDUM §3), so a request
            # within BLOCK positions of the edge would be flagged even while it declines. No row offers on such a step.
            cap = 0
            self.counters["capacity_edge_steps"] += 1
        jobs = []
        for r in live:
            state = self.slots.states.get(self.row_slots[r])
            if state is None:
                continue  # released between the step and its propose
            step, state.step = state.step, None
            if step is None:
                continue
            c = counts[r]
            position = positions[r][c - 1]
            if position != step.position + c:
                raise ValueError(f"Row {r}: anchor position {position} is not the step's {step.position} plus {c}")
            job = self._propose(state, step, c, tokens[r][c - 1], position, cap)
            if job is not None:
                jobs.append((r, state, job))
        if jobs:
            drafted = self.drafter.draft([job for _, _, job in jobs])
            for r, state, job in jobs:
                drafts = [int(t) for t in drafted[state.tap_slot][:cap]]
                ids[r, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
                valid[r] = len(drafts)
                state.offer = Offer(job[1], job[2], drafts)
                state.regime = "hold"
                state.counters["drafts"] += 1
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    def _propose(self, state, step, c, anchor, position, cap):
        """M2's §5.2 for one request (generator_resident_spec_vllm._propose), or None to decline. A drafting request
        gets its tap upkeep now; a declining one's waits for the next launch."""
        if step.kind == "verify" and step.drafts and state.regime == "hold":
            state.policy.observe(c - 1, step.drafts)
            state.counters["accepted_drafts"] += c - 1
        state.counters["committed_tokens"] += c
        if step.host or step.dropped:
            state.counters["spec_nonspeculable_steps"] += 1
            if step.dropped:
                # A drop is the request's own only when the runner trimmed its offer for room under max_model_len
                # (model_runner.py _propose_model_drafts); any other drop is a co-tenant's (a host-sampled row, the
                # scheduler's budget) and never suppresses (M5_DESIGN_ADDENDUM_1 §2 d ii).
                own = position + self.K >= self.max_seq_len
                state.counters["spec_offers_dropped" if own else "spec_offers_dropped_cotenant"] += 1
                state.suppressed = state.suppressed or own
            return self._decline(state, step, c)
        if cap == 0:
            state.counters["spec_rows_over_cap"] += 1
            return self._decline(state, step, c)
        if state.book is None or self.drafter is None:
            state.counters["spec_no_tap_slot"] += 1
            return self._decline(state, step, c)
        if position + self.BLOCK > self._spec_limit():
            state.exhausted = True
        if state.suppressed or state.exhausted:
            return self._decline(state, step, c)
        if state.regime == "hold":
            if not state.policy.decide(position, entering=False):
                return self._decline(state, step, c)  # exit: nothing commits; the next entry reseeds
            self._tap_upkeep(state, step, c)
            self.taps.commit(state.tap_slot, c)
            return (state.tap_slot, anchor, position, step.position, c)
        if not state.policy.decide(position, entering=True):
            return self._decline(state, step, c, asked=True)
        self._tap_upkeep(state, step, c)
        self._enter(state, position)
        return (state.tap_slot, anchor, position, position, 0)

    def _spec_limit(self):
        limit = int(getattr(self.engine.cfg, "capacity", self.max_seq_len))
        return limit if self.drafter is None else min(limit, int(self.drafter.max_positions))

    def _decline(self, state, step, c, asked=False):
        state.regime = "plain"
        if not asked:
            state.policy.plain()
        state.counters["declines"] += 1
        if state.book is not None:
            if self.defer_taps:
                self.deferred_taps.append((state, step, c))
                state.deferred += 1
            else:
                self._tap_upkeep(state, step, c)
        return None

    def _tap_upkeep(self, state, step, c):
        """The step's committed tap rows into the request's ring and history (M2 §5.2 step 1), after any upkeep
        still deferred for the same request."""
        if state.deferred:
            self._drain_taps(only=state)
        self.taps.stage(state.tap_slot, self.engine.taps_buffer(step.launch), step.row, c)
        self.taps.ring_insert(state.tap_slot, state.book.insert(step.position, c))
        if (step.position + c) // resident_taps.TILE > step.position // resident_taps.TILE:
            # these rows end a page: it is the only one that can have completed (TapBook.complete_pages)
            for page in state.book.complete_pages():
                state.book.check_write(page)
                self.taps.flush(state.tap_slot, page, state.book.entries(page))
                state.book.flushed(page)

    def _drain_taps(self, only=None):
        """The deferred upkeep, in order: after a launch (inside its device window), before an engine rebuild, or for
        one request before its own upkeep."""
        if not self.deferred_taps:
            return
        if only is None:
            run, self.deferred_taps = self.deferred_taps, []
        else:
            run = [entry for entry in self.deferred_taps if entry[0] is only]
            self.deferred_taps = [entry for entry in self.deferred_taps if entry[0] is not only]
        for state, step, c in run:
            state.deferred -= 1
        for state, step, c in run:
            if self.slots.states.get(state.slot) is state:  # its request is still live
                self._tap_upkeep(state, step, c)
        self.counters["deferred_tap_steps"] += len(run)

    def _enter(self, state, position):
        """Seed the request's drafter rows from its history window of every position < `position` (M2 §5.2 step 4)."""
        page = resident_taps.page_of(position - 1)
        if state.book.entries(page):
            state.book.check_write(page)
            self.taps.flush(state.tap_slot, page, state.book.entries(page))
            state.book.flushed(page, partial=True)
        first, start, valid = resident_taps.window(position)
        windows = self.taps.read_window(state.tap_slot, first)
        self.drafter.seed(state.tap_slot, {layer: windows[i] for i, layer in enumerate(self.tap_layers)}, start, valid)
        state.counters["seeds"] += 1
        state.seeded = True

    # ------------------------------------------------------------------ lifecycle

    def _guard_after_capture(self):
        """M1's guard after the lazy sampling capture (generator_resident_vllm._guard_after_capture), with the feed a
        rebuild's warmup zeroed written back into each row's engine slot, not slot = row."""
        for _ in range(2):
            kept = {}

            def keep_feed():
                self._check_status(self._status_words())
                kept["tokens"] = self.engine.read_tokens(self.max_batch_size).tolist()

            self._l1_guard("sampling trace capture", before_rebuild=keep_feed)
            if kept:
                positions = [p if s is not None and p < self.max_seq_len else -1
                             for s, p in zip(self.row_slots, self.positions)]
                slots = [0 if s is None else s for s in self.row_slots]
                tokens = [t if p >= 0 else 0 for t, p in zip(kept["tokens"], positions)]
                self.engine.write_rows(slots, positions, tokens)
                self.feed_stale = False
            if self.sampling_trace is not None:
                return
            self._capture_sampling()  # the rebuild released the trace just captured
        raise RuntimeError("The sampling trace was released by an engine rebuild twice in a row")

    def _l1_guard(self, reason, before_rebuild=None):
        """M1's guard; deferred tap upkeep reads taps_buffer, which a rebuild's warmup overwrites, so it runs first."""
        self._drain_taps()
        super()._l1_guard(reason, before_rebuild=before_rebuild)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        """M1's decode warmup, after the profile checks: the engine serves max_num_seqs users in one launch, and a
        multi-user engine exposes the hub argmax a mixed step's greedy rows need."""
        if self.prefill_tokens is not None:
            return
        served = max(self._users_served(R) for R in self.ROWS)
        if served < self.max_batch_size:
            raise ValueError(f"The engine's programs serve at most {served} users; max_num_seqs is "
                             f"{self.max_batch_size} (M1_INTERFACE_M5_ADDENDUM §5)")
        if self.max_batch_size > 1 and getattr(self.engine, "argmax_ids", None) is None:
            raise ValueError("A multi-user engine must expose argmax_ids and greedy_mask (K2A_DESIGN_v2 §7)")
        super().warmup_model_decode(kv_cache, enable_trace, **kwargs)
        if self.taps is not None:
            self.taps.seal()

    def note_state_slots_moved(self, moves):
        """The plugin's gather moved its state slots; engine slots stay. Checked against the last reload's rows."""
        Gemma4ForCausalLM.note_state_slots_moved(self, moves)
        self.slots.moved(dict(moves))
        if self.pending_rows is not None:
            wrong = {r: (s, self.slots.plugin.get(r)) for r, s in self.pending_rows.items()
                     if self.slots.plugin.get(r) != s}
            if wrong:
                raise RuntimeError(f"After the plugin's moves {moves}, rows (engine slot, bound) disagree: {wrong}")
        self.pending_rows = None

    def release_request(self, slot):
        """A finished or preempted request (the plugin's current state slot): its engine slot, tap slot and state."""
        Gemma4ForCausalLM.release_request(self, slot)
        state = self.slots.release(slot)
        if state is None:
            self.counters["releases_unknown"] += 1
            return
        self.deferred_taps = [entry for entry in self.deferred_taps if entry[0] is not state]
        if self.engine is not None:
            self.engine.release_slot(state.slot)  # its tables back to scratch
        self.slot_raw[:, state.slot] = -1
        self.seen = None
        self.positions = [-1 if s == state.slot else p for s, p in zip(self.row_slots, self.positions)]
        self.row_slots = [None if s == state.slot else s for s in self.row_slots]
        if state.counters:
            logger.debug("Resident request done (slot {}): {}", state.slot, dict(state.counters))
        self.counters.update({f"done_{k}": v for k, v in state.counters.items()})
        self.counters["releases"] += 1

    def release_persistent_capture(self):
        self.deferred_taps = []
        super().release_persistent_capture()
