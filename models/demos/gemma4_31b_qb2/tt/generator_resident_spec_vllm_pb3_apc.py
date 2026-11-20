# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: `single-user` (PB3: resident engine + DFlash2 + the paged-bucket prefill) with S1 prefix resume
(bench/resident/s1prefix/S1_DESIGN.md §5 + ADDENDUM_1 §1 / ADDENDUM_2; LEAD_DECISIONS_S1P Q1-Q3). Differences from
PB3:
- capabilities: `supports_prefix_caching` and `prefix_resume` (vllm_tt_plugin.prefix_resume): resume points on the
  8,192 grid, at least 2,080 prompt tokens always recomputed (DFlash2's seed reads the 65 tap pages ending at the
  first decode position, all of them inside that suffix: ADDENDUM_1 §1), GEMMA4_S1_TAIL_SLOTS tail slots (default
  2), prompt-only caching; no request ids (one request at a time);
- get_kv_cache_spec: generator_vllm's hybrid spec (50 sliding, 10 full) without its refusal of prefix caching. The
  profile serves with --prefix-cache-retention-interval 8192 so vLLM's sliding groups report hits on the grid;
- admission past 0: the first chunk of a request (no request is live) that starts at c > 0 runs the start-0 resets
  (`_reset_request`, `spec_policy.new_request`: the tap book, the drafter and the policy) and gets the tail the
  scheduler ordered restored as its prefill history (s1_tail_pool);
- store orders: after the prefill, the tail of a chunk that ended where the scheduler ordered a store is copied into
  the ordered pool slot.
The engine reads sliding K/V from vLLM's blocks and has no history of its own; the suffix rewrites every sliding page
and tap row the first decode reads.
"""

import torch

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_vllm import prefix_resume, tail_slots
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb3 import Gemma4ResidentSpecPB3ForCausalLM
from models.demos.gemma4_31b_qb2.tt.s1_tail_pool import TailPool, device_alloc, device_copy, token_digest


class Gemma4ResidentSpecPB3APCForCausalLM(Gemma4ResidentSpecPB3ForCausalLM):
    model_capabilities = {
        **Gemma4ResidentSpecPB3ForCausalLM.model_capabilities,
        "supports_prefix_caching": True,
        "prefix_resume": {**prefix_resume(tail_slots(2)), "req_ids": False},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_pool = None
        self.live = False  # a request's first chunk has run and it is not yet released

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """generator_vllm.Gemma4ForCausalLM.get_kv_cache_spec without the refusal of prefix caching."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        if vllm_config.max_in_flight_tokens > 2 * cls.MAX_PREFILL_TOKENS:
            raise ValueError("Gemma4 QB2 supports at most 8192 batched tokens with async chunked prefill")
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != 128:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        page_bytes = 2 * block * config.num_key_value_heads * config.head_dim * 2
        result = {}
        for i, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            common = dict(
                block_size=block,
                num_kv_heads=config.num_key_value_heads if sliding else config.num_global_key_value_heads,
                head_size=config.head_dim if sliding else config.global_head_dim,
                dtype=torch.bfloat16,
                page_size_padded=page_bytes,
            )
            result[f"model.layers.{i}.self_attn"] = (
                SlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**common)
            )
        return result

    def allocate_pools(self, per_layer_specs, *, alias=True):
        state = super().allocate_pools(per_layer_specs, alias=alias)
        self.tail_pool = self._make_tail_pool(self.model_capabilities["prefix_resume"]["tail_slots"])
        return state

    def _make_tail_pool(self, slots):
        return TailPool(slots, self.generator.model.layers, alloc=device_alloc(self.mesh), copy=device_copy)

    def _history_key(self, kv_cache, page_tables_per_layer, empty_slots):
        """The key PB's prefill_forward keeps the row's history under (generator_resident_spec_vllm_pb.py:79-81)."""
        slots = [0] if empty_slots is None else [int(s) for s in empty_slots]
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        full_layer = next(i for i, layer in enumerate(self.generator.model.layers) if not layer.sliding)
        return int(tables[full_layer][slots[0], 0])

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, tail_orders=None, start_pos=None,
                        empty_slots=None, page_tables_per_layer=None, **kwargs):
        if len(prompt_lens) != 1:
            raise ValueError("Speculative serving prefills one request at a time")
        start = 0 if start_pos is None else int(start_pos[0])
        end = int(prompt_lens[0])
        orders = (tail_orders or [None])[0] or {}
        restore, store = orders.get("restore"), orders.get("store")
        key = None
        if restore is not None:
            if self.live or start != restore[1]:
                raise ValueError(f"A restore at {restore[1]} for a chunk starting at {start} "
                                 f"({'a live request' if self.live else 'a new request'})")
            self._reset_request()
            self.spec_policy.new_request()
            self.counters["admissions_past_0"] += 1
            slot, at, generation = restore
            key = self._history_key(kv_cache, page_tables_per_layer, empty_slots)
            layers = self.tail_pool.restore(slot, generation, at, token_digest(tokens[0], at))
            kv_cache.prefill_history[key] = {"end": at, "padded_end": at, "layers": layers}
        elif start > 0 and not self.live:
            raise ValueError(f"A new request starts at {start} with no tail to restore")
        out = super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos=start_pos,
                                      empty_slots=empty_slots, page_tables_per_layer=page_tables_per_layer, **kwargs)
        self.live = True
        if store is not None:
            slot, at, generation = store
            key = self._history_key(kv_cache, page_tables_per_layer, empty_slots) if key is None else key
            history = kv_cache.prefill_history.get(key)
            if history is None or at != end or history["end"] != at or history["padded_end"] != at:
                raise ValueError(f"A store at {at} after a chunk ending at {end} (history "
                                 f"{None if history is None else (history['end'], history['padded_end'])})")
            self.tail_pool.store(slot, generation, at, token_digest(tokens[0], at), history["layers"])
        return out

    def release_request(self, slot):
        super().release_request(slot)
        self.live = False
