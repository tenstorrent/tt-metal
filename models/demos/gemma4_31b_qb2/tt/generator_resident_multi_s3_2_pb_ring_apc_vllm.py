# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: option C's ring class (generator_resident_multi_s3_2_pb_ring_vllm.py) with S1 prefix resume
(bench/resident/s1prefix/S1_DESIGN.md + ADDENDUM_1 / ADDENDUM_2; LEAD_DECISIONS_S1P Q1-Q3). Differences from the ring
class:
- capabilities: `supports_prefix_caching` and `prefix_resume` (vllm_tt_plugin.prefix_resume): resume points on the
  8,192 grid, at least 2,080 prompt tokens always recomputed (the ring's 1,152-row decode window and any drafter's
  2,056-row tap window lie in the recomputed suffix), GEMMA4_S1_TAIL_SLOTS tail slots (default 4), prompt-only
  caching, and the rows' request ids;
- get_kv_cache_spec: the ring class's spec without its refusal of prefix caching;
- request identity: engine slots and prefill histories are keyed by the plugin's request id (`req_ids`), because
  prefix caching shares full block 0 between requests. S0's prefill takes each row's key with exactly one `_key` call
  per row, in row order (generator_resident_multi_vllm.py:402); during a prefill with request ids, `_key` hands out
  the rows' request keys instead, and the call count is checked. S1's continuation rebind and S0's decode layout read
  the request keys directly (`_rebind_continuations`, `_layout`);
- admission past 0: a request whose first chunk starts at c > 0 (a prefix-cache hit, or a preempted request resumed
  on its own cached blocks) is admitted here with S0's start-0 resets, and its prefill history is the tail the
  scheduler ordered restored from the pool (s1_tail_pool);
- store orders: after the prefill, a row whose chunk ended where the scheduler ordered a store has its tail copied
  into the ordered pool slot;
- same-step order: a row that reads a block another row of the same step writes (its hit covers part of that row's
  chunk; vLLM caches blocks at schedule time) runs after it. Writes lie at columns >= the writer's start and reads
  below the reader's start, so sorting the rows by start orders every such pair; the outputs keep the plugin's order.
Without request ids (the bucket warmups, the continuation pre-warm) the ring class runs unchanged.
"""

import dataclasses
import os

import torch

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_vllm import (
    PAGE,
    Gemma4ResidentMultiS32PBRingForCausalLM,
)
from models.demos.gemma4_31b_qb2.tt.s1_tail_pool import TailPool, device_alloc, device_copy, token_digest

GRANULARITY = 8192  # the cold run's chunk grid (max_num_batched_tokens): resume points that are bitwise to cold
MIN_SUFFIX = 2080  # >= the ring's 1,152 decode-window rows and DFlash2's 2,080-row seed window (ADDENDUM_1 §1)
KEY_BASE = 1 << 40  # request keys: far above every block-derived key S0's warmups use


def tail_slots(default):
    value = int(os.environ.get("GEMMA4_S1_TAIL_SLOTS", str(default)))
    if value < 0:
        raise ValueError(f"GEMMA4_S1_TAIL_SLOTS must be >= 0, got {value}")
    return value


def prefix_resume(slots):
    return {"granularity": GRANULARITY, "min_suffix": MIN_SUFFIX, "tail_slots": slots, "prompt_only": True,
            "req_ids": True}


class Gemma4ResidentMultiS32PBRingAPCForCausalLM(Gemma4ResidentMultiS32PBRingForCausalLM):
    model_capabilities = {
        **Gemma4ResidentMultiS32PBRingForCausalLM.model_capabilities,
        "supports_prefix_caching": True,
        "prefix_resume": prefix_resume(tail_slots(4)),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_pool = None
        self._req_keys = {}  # request id -> key, for the request's life (dropped at release)
        self._key_reqs = {}  # key -> request id
        self._next_key = KEY_BASE
        self._prefill_keys = None  # the rows' keys while a prefill with request ids runs (_key)
        self._prefill_key_calls = 0
        self._decode_keys = None  # the rows' keys while a decode with request ids runs (_layout)

    # ------------------------------------------------------------------ vLLM's view

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """The ring class's spec (the full layers only) without the refusal of prefix caching."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if vllm_config.max_in_flight_tokens > 2 * cls.MAX_PREFILL_TOKENS:
            raise ValueError("Gemma4 QB2 supports at most 8192 batched tokens with async chunked prefill")
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != PAGE:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=block,
                num_kv_heads=config.num_global_key_value_heads,
                head_size=config.global_head_dim,
                dtype=torch.bfloat16,
            )
            for i, kind in enumerate(config.layer_types)
            if kind != "sliding_attention"
        }

    # ------------------------------------------------------------------ pools

    def allocate_pools(self, per_layer_specs, *, alias=True):
        state = super().allocate_pools(per_layer_specs, alias=alias)
        self.tail_pool = self._make_tail_pool(self.model_capabilities["prefix_resume"]["tail_slots"])
        return state

    def _make_tail_pool(self, slots):
        return TailPool(slots, self.generator.model.layers, alloc=device_alloc(self.mesh), copy=device_copy)

    # ------------------------------------------------------------------ request keys

    def _request_key(self, req_id):
        key = self._req_keys.get(req_id)
        if key is None:
            key, self._next_key = self._next_key, self._next_key + 1
            self._req_keys[req_id], self._key_reqs[key] = key, req_id
        return key

    def _key(self, value):
        if self._prefill_keys is None:
            return super()._key(value)
        if self._prefill_key_calls >= len(self._prefill_keys):
            raise RuntimeError(f"S0's prefill asked for key {self._prefill_key_calls + 1} of a "
                               f"{len(self._prefill_keys)}-row prefill")
        key = self._prefill_keys[self._prefill_key_calls]
        self._prefill_key_calls += 1
        return key

    def _admit(self, key, plugin_slot):
        """S0's start-0 admission (generator_resident_multi_vllm.py:406-412) for a request starting past 0."""
        state = self.slots.admit(key, plugin_slot, self._new_state)
        if state.tap_slot is not None:
            self.taps.reset(state.tap_slot)
            if self.drafter is not None:
                self.drafter.reset(state.tap_slot)
        self.counters["requests"] += 1
        self.counters["admissions_past_0"] += 1

    # ------------------------------------------------------------------ prefill

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, req_ids=None, tail_orders=None,
                        start_pos=None, empty_slots=None, sampling_params=None, page_tables_per_layer=None,
                        **kwargs):
        if req_ids is None:
            if tail_orders is not None:
                raise ValueError("Tail orders arrived without request ids")
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos=start_pos,
                                           empty_slots=empty_slots, sampling_params=sampling_params,
                                           page_tables_per_layer=page_tables_per_layer, **kwargs)
        n = len(prompt_lens)
        lengths = [int(x) for x in prompt_lens]
        starts = [0] * n if start_pos is None else [int(x) for x in start_pos][:n]
        plugin_slots = list(range(n)) if empty_slots is None else [int(s) for s in empty_slots]
        orders = [None] * n if tail_orders is None else list(tail_orders)
        if len(req_ids) != n or len(orders) != n or len(starts) != n or len(plugin_slots) != n:
            raise ValueError(f"{n} prefill rows with {len(req_ids)} request ids, {len(orders)} tail orders, "
                             f"{len(starts)} starts and {len(plugin_slots)} plugin slots")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        full = page_table if page_tables_per_layer is None else page_tables_per_layer[self.full_layer]
        keys = [self._request_key(r) for r in req_ids]
        if len(set(keys)) != n:
            raise ValueError(f"A request appears twice in one prefill: {list(req_ids)}")
        for i in range(n):
            restore = (orders[i] or {}).get("restore")
            new = keys[i] not in self.slots.by_key
            if restore is not None and (not new or starts[i] != restore[1]):
                raise ValueError(f"Request {req_ids[i]}: a restore at {restore[1]} for a row starting at {starts[i]} "
                                 f"({'new' if new else 'continuing'})")
            if new and starts[i] > 0:
                if restore is None:
                    raise ValueError(f"Request {req_ids[i]} starts at {starts[i]} with no tail to restore")
                self._admit(keys[i], plugin_slots[i])
            if restore is not None:
                slot, end, generation = restore
                layers = self.tail_pool.restore(slot, generation, end, token_digest(tokens[i], end))
                kv_cache.prefill_history[keys[i]] = {"end": end, "padded_end": end, "layers": layers}
        order = self._step_order(full, starts, lengths)
        args = (tokens, page_table, page_tables_per_layer, lengths, starts, plugin_slots, sampling_params)
        if order != list(range(n)):
            self.counters["prefill_reorders"] += 1
            args = self._permuted(order, *args)
        p_tokens, p_table, p_tables, p_lengths, p_starts, p_slots, p_sampling = args
        self._prefill_keys, self._prefill_key_calls = [keys[i] for i in order], 0
        try:
            out = super().prefill_forward(p_tokens, p_table, kv_cache, p_lengths, start_pos=p_starts,
                                          empty_slots=p_slots, sampling_params=p_sampling,
                                          page_tables_per_layer=p_tables, **kwargs)
            calls = self._prefill_key_calls
        finally:
            self._prefill_keys = None
        if calls != n:
            raise RuntimeError(f"S0's prefill took {calls} keys for {n} rows")
        for i in range(n):
            store = (orders[i] or {}).get("store")
            if store is None:
                continue
            slot, end, generation = store
            history = kv_cache.prefill_history.get(keys[i])
            if history is None or end != lengths[i] or history["end"] != end or history["padded_end"] != end:
                raise ValueError(f"Request {req_ids[i]}: a store at {end} after a chunk ending at {lengths[i]} "
                                 f"(history {None if history is None else (history['end'], history['padded_end'])})")
            self.tail_pool.store(slot, generation, end, token_digest(tokens[i], end), history["layers"])
        return out if order == list(range(n)) else self._unpermuted(order, out, n)

    @staticmethod
    def _step_order(full, starts, lengths):
        """The plugin's row order, or the rows sorted by start when a row reads a block a later row writes."""
        n = len(starts)
        reads = [{int(b) for b in full[i, : starts[i] // PAGE].tolist() if b > 0} for i in range(n)]
        writes = [{int(b) for b in full[i, starts[i] // PAGE: -(-lengths[i] // PAGE)].tolist() if b > 0}
                  for i in range(n)]
        if not any(writes[w] & reads[r] for r in range(n) for w in range(r + 1, n)):
            return list(range(n))
        order = sorted(range(n), key=lambda i: starts[i])
        position = {row: k for k, row in enumerate(order)}
        late = [(w, r) for r in range(n) for w in range(n) if w != r and writes[w] & reads[r]
                and position[w] > position[r]]
        if late:
            raise RuntimeError(f"Prefill rows (writer, reader) {late} share blocks in both directions")
        return order

    def _permuted(self, order, tokens, page_table, page_tables_per_layer, lengths, starts, slots, sampling):
        index = torch.tensor(order)
        n = len(order)

        def rows(t):
            if t is None:
                return None
            out = t.clone()
            out[:n] = t[index]
            return out

        tables = None
        if page_tables_per_layer is not None:
            cache = {}
            tables = [cache.setdefault(id(t), rows(t)) for t in page_tables_per_layer]
        return (rows(tokens), rows(page_table), tables, [lengths[i] for i in order], [starts[i] for i in order],
                [slots[i] for i in order], self._permuted_sampling(order, sampling))

    @staticmethod
    def _permuted_sampling(order, sampling):
        if sampling is None:
            return None
        n = len(order)
        changes = {}
        for field in dataclasses.fields(sampling):
            value = getattr(sampling, field.name)
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] >= n:
                moved = value.clone()
                moved[:n] = value[torch.tensor(order)]
                changes[field.name] = moved
            elif isinstance(value, (list, tuple)) and len(value) >= n:
                changes[field.name] = type(value)([value[i] for i in order] + list(value[n:]))
        return dataclasses.replace(sampling, **changes)

    @staticmethod
    def _unpermuted(order, out, n):
        if isinstance(out, torch.Tensor) and out.dim() > 0 and out.shape[0] >= n:
            result = out.clone()
            result[torch.tensor(order)] = out[:n]
            return result
        if isinstance(out, list) and len(out) == n:
            result = [None] * n
            for k, i in enumerate(order):
                result[i] = out[k]
            return result
        raise TypeError(f"Cannot restore the plugin's row order of a prefill output {type(out).__name__}")

    def _rebind_continuations(self, page_tables_per_layer, start_pos, empty_slots):
        """S1's rebind (generator_resident_multi_s1_vllm.py:163-188) over the rows' request keys: every row whose
        request already holds an engine slot (a continuation, or a request admitted past 0 by this prefill)."""
        if self._prefill_keys is None:
            return super()._rebind_continuations(page_tables_per_layer, start_pos, empty_slots)
        if start_pos is None or empty_slots is None:
            return
        moving = {}
        for key, plugin_slot in zip(self._prefill_keys, empty_slots):
            slot = self.slots.by_key.get(key)
            if slot is not None:
                moving[int(plugin_slot)] = slot
        if not moving or all(self.slots.plugin.get(p) == s for p, s in moving.items()):
            return
        engine = set(moving.values())
        self.slots.plugin = {p: s for p, s in self.slots.plugin.items() if s not in engine}
        for plugin_slot, slot in moving.items():
            other = self.slots.plugin.get(plugin_slot)
            if other is not None and other in self.slots.states:
                raise RuntimeError(f"Plugin slot {plugin_slot} is bound to engine slot {other}, not {slot}: "
                                   f"{self.slots.plugin}")
            self.slots.plugin[plugin_slot] = slot
        self.counters["prefill_rebinds"] += 1

    # ------------------------------------------------------------------ decode

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, req_ids=None, **kwargs):
        if req_ids is not None and kwargs.get("spec_mode") is not None:
            raise NotImplementedError("S1's batch class serves no speculation (GEMMA4_RESIDENT_SPEC_POLICY never)")
        self._decode_keys = None if req_ids is None else [self._req_keys.get(r) for r in req_ids]
        try:
            return super().decode_forward(tokens, start_pos, page_table, kv_cache, **kwargs)
        finally:
            self._decode_keys = None

    def _layout(self, sources, positions, slot_remap):
        """S0's layout (generator_resident_multi_vllm.py:548-573) over the rows' request keys."""
        if self._decode_keys is None:
            return super()._layout(sources, positions, slot_remap)
        rows, remap = [], None if slot_remap is None else [int(s) for s in slot_remap.reshape(-1)]
        for r, position in enumerate(positions):
            if position < 0:
                rows.append(None)
                continue
            key = self._decode_keys[r] if r < len(self._decode_keys) else None
            if key is None:
                raise RuntimeError(f"Decode row {r} (position {position}) has no request key: "
                                   f"{self._decode_keys}")
            slot = self.slots.slot_of(key)
            plugin_slot = r if remap is None else remap[r]
            if self.slots.plugin.get(plugin_slot) != slot:
                raise RuntimeError(f"Decode row {r} (request {self._key_reqs.get(key)}, engine slot {slot}) arrives "
                                   f"from plugin slot {plugin_slot}, which is bound to "
                                   f"{self.slots.plugin.get(plugin_slot)}")
            if self.slots.states[slot].flagged:
                raise RuntimeError(f"Engine slot {slot} was flagged 0x{self.slots.states[slot].flagged:x}; its "
                                   "request must not decode again")
            rows.append(slot)
        if len(set(s for s in rows if s is not None)) != sum(s is not None for s in rows):
            raise RuntimeError(f"Two decode rows share an engine slot: {rows}")
        self.pending_rows = {r: s for r, s in enumerate(rows) if s is not None}
        return rows

    # ------------------------------------------------------------------ release

    def release_request(self, slot):
        engine = self.slots.plugin.get(int(slot))
        state = self.slots.states.get(engine) if engine is not None else None
        super().release_request(slot)
        if state is not None:
            req_id = self._key_reqs.pop(state.key, None)
            if req_id is not None:
                self._req_keys.pop(req_id, None)
