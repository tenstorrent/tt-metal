# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU simulation of KV packing with the real vLLM KV cache manager and the real table translation.

Gemma4 groups (five sliding groups of 128-token blocks, one full-attention group of 256-token blocks) share one global
block ID space; every ID is one wide backing row. The page tables the manager produces go through the model's own
`_tables`, and a shadow model of the wide rows checks that every attention read (the whole cached or computed prefix for full
attention, the window for sliding attention) returns exactly the content written for that position, under pool pressure,
preemption, prefix hits, eviction and reuse. A full block owns both half-pages of its row, a sliding page covers both halves.
"""

import math
import random
from types import SimpleNamespace

import numpy as np
import torch

from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM

PAGE, WINDOW, LAYERS, MAX_SEQ = 128, 1024, 60, 262144
PAGE_BYTES = 2 * PAGE * 16 * 256 * 2


class Memory:
    """Shadow content of the wide rows: [row, half, offset] -> (writer group, prefix fingerprint)."""

    def __init__(self, rows):
        self.tag = np.zeros((rows, 2, PAGE), dtype=np.int64)
        self.group = np.full((rows, 2, PAGE), -1, dtype=np.int8)
        self.reads = 0

    @staticmethod
    def locate(table_row, positions, full):
        pages, offsets = table_row[positions // PAGE], positions % PAGE
        return (pages // 2, pages % 2, offsets) if full else (pages, None, offsets)

    def write(self, table_row, positions, values, group, full):
        rows, halves, offsets = self.locate(table_row, positions, full)
        assert (rows > 0).all(), "write into the null block"
        for half in [None] if full else [0, 1]:
            hh = halves if full else np.full_like(rows, half)
            self.tag[rows, hh, offsets] = values
            self.group[rows, hh, offsets] = group

    def check(self, table_row, positions, values, group, full, label):
        if len(positions) == 0:
            return
        rows, halves, offsets = self.locate(table_row, positions, full)
        assert (rows > 0).all(), f"{label}: read of the null block"
        for half in [None] if full else [0, 1]:
            hh = halves if full else np.full_like(rows, half)
            ok = (self.tag[rows, hh, offsets] == values) & (self.group[rows, hh, offsets] == group)
            assert ok.all(), f"{label}: corrupt KV at position {int(positions[np.flatnonzero(~ok)[0]])}, group {group}"
        self.reads += len(positions)


def fingerprints(tokens):
    out, h = np.empty(len(tokens), dtype=np.int64), 1469598103934665603
    for i, t in enumerate(tokens):
        h = (h * 1099511628211 + t + 1) & 0x7FFFFFFFFFFFFFFF
        out[i] = h
    return out


def build(num_blocks, batch):
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils as U
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
    from vllm_tt_plugin.kv_cache_anchor import AnchoredSlidingWindowSpec

    # Importing the plugin registers its spec first, which makes vLLM skip its lazy registration of the built-in ones.
    register_all_kvcache_specs(None)
    specs = {}
    for i in range(LAYERS):
        if i % 6 == 5:
            spec = FullAttentionSpec(
                block_size=2 * PAGE, num_kv_heads=4, head_size=512, dtype=torch.bfloat16, page_size_padded=PAGE_BYTES
            )
        else:
            spec = AnchoredSlidingWindowSpec(
                block_size=PAGE,
                num_kv_heads=16,
                head_size=256,
                dtype=torch.bfloat16,
                page_size_padded=PAGE_BYTES,
                sliding_window=WINDOW,
            )
        specs[f"model.layers.{i}.self_attn"] = spec
    assert len({s.page_size_bytes for s in specs.values()}) == 1
    groups = U._get_kv_cache_groups_uniform_page_size(specs)
    sizes = [g.kv_cache_spec.block_size for g in groups]
    assert sorted(sizes) == [PAGE] * 5 + [2 * PAGE]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=MAX_SEQ,
        scheduler_block_size=math.lcm(*sizes),
        hash_block_size=math.gcd(*sizes),
        max_in_flight_tokens=16384,
        enable_caching=True,
    )
    layer_group = {
        int(n.split(".")[2]): g for g, group in enumerate(groups) for name in group.layer_names for n in [name]
    }
    U.init_none_hash(sha256)
    hasher = U.get_request_block_hasher(math.gcd(*sizes), sha256)
    model = object.__new__(Gemma4ForCausalLM)
    model.sub_caches, model.verify_caches = {}, {}
    model.full_layers = frozenset(i for i in range(LAYERS) if i % 6 == 5)
    model.cache = SimpleNamespace(
        batch=batch,
        kv=[None] * LAYERS,
        page_tables={i: torch.zeros(batch, MAX_SEQ // PAGE, dtype=torch.int32) for i in range(LAYERS)},
        columns=1,
    )
    return manager, groups, layer_group, hasher, model


def simulate(seed, blocks=500, conversations=10, admitted=8, steps=20000, max_context=14000):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request, RequestStatus

    rng, vocab = random.Random(seed), 50000
    manager, groups, layer_group, hasher, model = build(blocks, admitted)
    memory, sp = Memory(blocks), SamplingParams(max_tokens=1 << 20)
    convs = [
        dict(
            id=c,
            tokens=[rng.randrange(vocab) for _ in range(rng.randint(1500, 6000))],
            turn=0,
            turns=rng.randint(3, 8),
            ready_at=rng.randint(0, 200),
            done=False,
            active=None,
        )
        for c in range(conversations)
    ]
    stats = dict(steps=0, finished_turns=0, hit_tokens=0, preemptions=0, admission_waits=0, tables=0)
    waiting, running, meta = [], [], {}

    def new_request(conv):
        rid = f"c{conv['id']}-t{conv['turn']}"
        meta[rid] = dict(conv=conv, fp=fingerprints(conv["tokens"]), out=rng.randint(20, 300))
        return Request(rid, list(conv["tokens"]), sp, None, block_hasher=hasher)

    def tables_of(r):
        ids, per_group = manager.get_block_ids(r.request_id), []
        for g in range(len(groups)):
            t = torch.zeros(1, MAX_SEQ // PAGE, dtype=torch.int32)
            t[0, : len(ids[g])] = torch.tensor(ids[g], dtype=torch.int32) if ids[g] else t[0, :0]
            per_group.append(t)
        stats["tables"] += 1
        return model._tables(model.cache, [per_group[layer_group[i]] for i in range(LAYERS)], slots=[0])

    def execute(r, start, end):
        m = meta[r.request_id]
        if len(m["fp"]) < end:
            m["fp"] = fingerprints(r.all_token_ids)
        tables = tables_of(r)
        for layer in range(6):  # one pool position: layer 5 is its full-attention layer
            g, full = layer_group[layer], layer % 6 == 5
            row = tables[layer][0].numpy().astype(np.int64)
            first = 0 if full else max(0, start - (WINDOW - 1))
            memory.check(row, np.arange(first, start, dtype=np.int64), m["fp"][first:start], g, full, r.request_id)
            memory.write(row, np.arange(start, end, dtype=np.int64), m["fp"][start:end], g, full)
            pad_end = min(-(-end // 32) * 32, len(row) * PAGE)  # tile padding past the chunk, inside its page
            if pad_end > end and row[end // PAGE] > 0:
                memory.write(
                    row, np.arange(end, pad_end, dtype=np.int64), np.full(pad_end - end, -7, dtype=np.int64), g, full
                )

    for step in range(steps):
        stats["steps"] = step
        for conv in convs:
            if (
                not conv["done"]
                and conv["active"] is None
                and conv["ready_at"] <= step
                and not any(meta[w.request_id]["conv"] is conv for w in waiting)
            ):
                waiting.append(new_request(conv))
                conv["active"] = True
        budget = 8192
        if waiting and len(running) < admitted:
            r = waiting[0]
            blocks_hit, hits, _ = manager.get_computed_blocks(r) if r.num_computed_tokens == 0 else (None, 0, 0)
            chunk = min(budget, r.num_tokens - r.num_computed_tokens - hits)
            if (
                manager.allocate_slots(
                    r,
                    chunk,
                    num_new_computed_tokens=hits,
                    new_computed_blocks=blocks_hit,
                    has_scheduled_reqs=bool(running),
                )
                is not None
            ):
                waiting.pop(0)
                r.status = RequestStatus.RUNNING
                stats["hit_tokens"] += hits
                start = r.num_computed_tokens + hits
                execute(r, start, start + chunk)
                r.num_computed_tokens = start + chunk
                running.append(r)
                budget -= chunk
                if r.num_computed_tokens == r.num_tokens:
                    r.append_output_token_ids(rng.randrange(vocab))
            else:
                stats["admission_waits"] += 1
        for r in list(running):
            if r not in running or r.num_tokens - r.num_computed_tokens <= 0:
                continue
            remaining = r.num_tokens - r.num_computed_tokens
            n = min(remaining, budget) if remaining > 1 else 1
            if n <= 0:
                continue
            while manager.allocate_slots(r, n) is None:
                victim = running.pop()
                manager.free(victim)
                victim.status, victim.num_computed_tokens = RequestStatus.PREEMPTED, 0
                waiting.insert(0, victim)
                stats["preemptions"] += 1
                if victim is r:
                    break
            if r not in running:
                continue
            start = r.num_computed_tokens
            execute(r, start, start + n)
            r.num_computed_tokens = start + n
            budget -= n if remaining > 1 else 0
            if r.num_computed_tokens == r.num_tokens:
                m = meta[r.request_id]
                if r.num_output_tokens >= m["out"] or r.num_tokens >= max_context:
                    conv = m["conv"]
                    running.remove(r)
                    manager.free(r)
                    conv["tokens"] = list(r.all_token_ids) + [
                        rng.randrange(vocab) for _ in range(rng.randint(50, 1500))
                    ]
                    conv["turn"] += 1
                    conv["active"] = None
                    conv["ready_at"] = step + rng.randint(1, 200)
                    conv["done"] = conv["turn"] >= conv["turns"] or len(conv["tokens"]) >= max_context
                    stats["finished_turns"] += 1
                else:
                    r.append_output_token_ids(rng.randrange(vocab))
        if all(c["done"] for c in convs) and not running and not waiting:
            break
    stats.update(reads_checked=memory.reads, all_done=all(c["done"] for c in convs))
    return stats


def test_packed_blocks_never_corrupt_a_read_under_pressure_preemption_and_reuse():
    for seed in (3, 11):
        stats = simulate(seed)
        assert stats["all_done"], stats
        assert stats["reads_checked"] > 10_000_000, stats
        assert stats["hit_tokens"] > 0, "prefix hits must be exercised"
        assert stats["preemptions"] + stats["admission_waits"] > 0, "the pool must be under pressure"


def test_the_simulation_detects_a_wrong_translation(monkeypatch, expect_error):
    """Control: a full block mapped to one native page instead of its two half-pages makes rows shared between blocks."""

    def wrong(blocks, width):
        blocks = blocks[:, : width // 2]
        out = torch.zeros(blocks.shape[0], 2 * blocks.shape[1], dtype=blocks.dtype)
        out[:, ::2] = blocks
        out[:, 1::2] = blocks
        return out

    monkeypatch.setattr(Gemma4ForCausalLM, "_half_pages", staticmethod(wrong))
    with expect_error(AssertionError, "corrupt KV|null block"):
        simulate(3)
