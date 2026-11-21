# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache snapshots through the real vLLM adapter, compared with fresh runs on host tensors."""

import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt import prefix_cache
from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM
from models.demos.qwen38_27b_qb2.tt.prefix_cache import ALIGNMENT, PrefixSnapshots, prefix_snapshot_settings

P = 1_000_003
MULTIPLIERS = (31, 37, 41)


class HostTensor:
    def __init__(self, data, dtype, layout):
        self.data, self.dtype, self.layout = data, dtype, layout

    @property
    def shape(self):
        return tuple(self.data.shape)

    def __getitem__(self, index):
        return HostTensor(self.data[index].clone(), self.dtype, self.layout)


def host_copy(source, target):
    if source.shape != target.shape:
        raise ValueError(f"copy {source.shape} -> {target.shape}")
    target.data.copy_(source.data)


def host_concat(tensors, dim=0):
    return HostTensor(torch.cat([t.data for t in tensors], dim=dim), tensors[0].dtype, tensors[0].layout)


def host_allocate(shape, dtype, layout, device, memory_config):
    return HostTensor(torch.full(tuple(shape), 99991, dtype=torch.int64), dtype, layout)


class StateGenerator:
    def __init__(self, batch=4):
        def linear():
            return SimpleNamespace(
                recurrent=HostTensor(torch.zeros(batch, 2, 2, 2, dtype=torch.int64), ttnn.float32, ttnn.TILE_LAYOUT),
                conv=HostTensor(torch.zeros(batch, 3, 4, dtype=torch.int64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            )

        layers = [linear(), linear(), linear(), SimpleNamespace(recurrent=None, conv=None)]
        self.cache = SimpleNamespace(layers=layers, batch_size=batch)
        self.model = SimpleNamespace(_resident_decode_bucket=None)
        self.mesh = None
        self.page_host = torch.zeros(batch, 8, dtype=torch.int32)
        self.counters = Counter()
        self.computed = []
        self.cached = {}

    def reset_recurrent_slots(self, slots):
        for state in self.cache.layers[:3]:
            for slot in slots:
                state.recurrent.data[slot] = 0
                state.conv.data[slot] = 0

    def state(self, slot):
        return [(int(s.recurrent.data[slot, 0, 0, 0]), s.conv.data[slot, :, 0].tolist()) for s in self.cache.layers[:3]]

    def _advance(self, slot, row, start):
        if start > self.cached.get(slot, 0):
            raise AssertionError(f"slot {slot} resumed at {start}, beyond the cached prefix {self.cached[slot]}")
        recs = [int(s.recurrent.data[slot, 0, 0, 0]) for s in self.cache.layers[:3]]
        convs = [s.conv.data[slot, :, 0].tolist() for s in self.cache.layers[:3]]
        for token in row.tolist():
            for layer, multiplier in enumerate(MULTIPLIERS):
                recs[layer] = (recs[layer] * multiplier + token + sum(convs[layer]) * (layer + 1)) % P
                convs[layer] = convs[layer][1:] + [token]
        for state, rec, conv in zip(self.cache.layers[:3], recs, convs):
            state.recurrent.data[slot] = rec
            state.conv.data[slot] = torch.tensor(conv).reshape(3, 1)
        self.computed.append((slot, start, start + len(row)))
        self.cached[slot] = start + len(row)
        return sum(recs) % 1000

    def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, start_pos, slots, **kwargs):
        return [self._advance(slots[0], tokens[0, : prompt_lens[0]], start_pos[0])]

    def serving_prefill_tokens(self, tokens, *, page_table, kv_cache, prompt_lens, start_pos, slots):
        out = torch.zeros(32, 1, dtype=torch.int64)
        for row, (start, end, slot) in enumerate(zip(start_pos, prompt_lens, slots)):
            out[row, 0] = self._advance(slot, tokens[row, start:end], start)
        return out

    def set_batch_sampling_params(self, **kwargs):
        pass


def reference(prompt):
    gen = StateGenerator(1)
    token = gen._advance(0, torch.tensor(prompt), 0)
    return token, gen.state(0)


def mark(end):
    return (end - 16) // ALIGNMENT * ALIGNMENT


def hit(previous, prompt):
    common = 0
    while common < min(len(previous), len(prompt)) and previous[common] == prompt[common]:
        common += 1
    return min(common, len(prompt) - 1) // 32 * 32


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        ops = patch.multiple(
            prefix_cache.ttnn,
            copy=host_copy,
            concat=host_concat,
            allocate_tensor_on_device=host_allocate,
            Shape=tuple,
            create=True,
        )
        ops.start()
        self.addCleanup(ops.stop)
        env = patch.dict("os.environ", {"QWEN_VLLM_HOST_COMPATIBILITY": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.gen = StateGenerator()
        self.adapter = Qwen38ForCausalLM(self.gen, 4, 1024)
        self.adapter.cache = self.gen.cache
        self.adapter._decode_bound = True
        read = patch.object(self.adapter, "read_decode_output", side_effect=lambda tensor: tensor.clone())
        read.start()
        self.addCleanup(read.stop)
        convert = patch(
            "models.demos.qwen38_27b_qb2.tt.generator_vllm.ttnn.to_torch", side_effect=lambda tensor: tensor
        )
        convert.start()
        self.addCleanup(convert.stop)
        self.rng = torch.Generator().manual_seed(7)

    def words(self, n):
        return torch.randint(0, 5000, (n,), generator=self.rng).tolist()

    def enable(self, capacity=4):
        self.adapter.prefix_snapshots = PrefixSnapshots(self.gen, capacity, minimum=64, margin=16)
        self.gen.counters.clear()

    def serve(self, prompts, starts, slots):
        for slot, start in zip(slots, starts):
            self.gen.cached[slot] = start
        width = max(map(len, prompts))
        tokens = torch.zeros(len(prompts), width, dtype=torch.int32)
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt)
        before = len(self.gen.computed)
        result, _ = self.adapter.prefill_forward(
            tokens=tokens,
            page_table=torch.zeros(len(prompts), 8, dtype=torch.int32),
            kv_cache=self.gen.cache,
            prompt_lens=[len(p) for p in prompts],
            start_pos=torch.tensor(starts),
            sampling_params=SimpleNamespace(
                temperature=torch.zeros(len(prompts)),
                top_k=torch.ones(len(prompts), dtype=torch.int32),
                top_p=torch.ones(len(prompts)),
                seed=torch.tensor([7] * len(prompts)),
            ),
            empty_slots=slots,
        )
        for row, (prompt, slot) in enumerate(zip(prompts, slots)):
            token, state = reference(prompt)
            self.assertEqual(int(result[row, 0]), token)
            self.assertEqual(self.gen.state(slot), state)
        return self.gen.computed[before:]

    def test_settings_default_to_disabled_and_validate(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(prefix_snapshot_settings(), (0, 4096, 16))
        with patch.dict("os.environ", {"QWEN_PREFIX_SNAPSHOTS": "32", "QWEN_PREFIX_SNAPSHOT_MARGIN": "0"}):
            self.assertEqual(prefix_snapshot_settings(), (32, 4096, 1))
        for name, value in (("QWEN_PREFIX_SNAPSHOTS", "-1"), ("QWEN_PREFIX_SNAPSHOT_MIN", "16")):
            with patch.dict("os.environ", {name: value}), self.assertRaises(ValueError):
                prefix_snapshot_settings()

    def test_next_turn_resumes_from_the_previous_prompt(self):
        self.enable()
        first = self.words(300)
        self.assertEqual(self.serve([first], [0], [2]), [(2, 0, mark(300)), (2, mark(300), 300)])
        second = first[:-2] + self.words(130)
        start = hit(first, second)
        self.assertGreaterEqual(start, mark(300))
        next_mark = mark(len(second))
        self.assertEqual(self.serve([second], [start], [1]), [(1, mark(300), next_mark), (1, next_mark, len(second))])
        self.assertEqual(self.gen.counters["prefix_snapshot_restores"], 1)

    def test_hit_without_snapshot_restarts_from_zero(self):
        self.enable()
        prompt = self.words(50)
        self.assertEqual(self.serve([prompt], [32], [0]), [(0, 0, 50)])
        self.assertEqual(self.gen.counters["prefix_snapshot_saves"], 0)

    def test_hit_below_every_snapshot_restarts_from_zero(self):
        self.enable()
        first = self.words(300)
        self.serve([first], [0], [0])
        second = first[:100] + self.words(200)
        segments = self.serve([second], [hit(first, second)], [0])
        self.assertEqual(segments[0][1], 0)

    def test_rows_of_one_step_are_independent(self):
        self.enable()
        first = self.words(300)
        self.serve([first], [0], [3])
        second, other = first + self.words(160), self.words(70)
        segments = self.serve([second, other], [hit(first, second), 0], [0, 1])
        self.assertIn((0, mark(300), mark(460)), segments)
        self.assertIn((1, 0, 70), segments)

    def test_full_pool_replaces_superseded_snapshots_first(self):
        self.enable(capacity=2)
        conversations = [self.words(200), self.words(230)]
        previous = [None, None]
        for _ in range(3):
            for i, prompt in enumerate(conversations):
                start = hit(previous[i], prompt) if previous[i] else 0
                self.serve([prompt], [start], [i])
                previous[i] = prompt
                conversations[i] = prompt + self.words(40)
        self.assertEqual(self.gen.counters["prefix_snapshot_restores"], 4)
        self.assertEqual(len(self.adapter.prefix_snapshots.keys), 2)

    def test_restore_publishes_the_resident_decode_bucket_first(self):
        self.enable()
        first = self.words(300)
        self.serve([first], [0], [0])
        events = []
        self.gen.model._resident_decode_bucket = object()
        self.gen.model.suspend_decode_bucket = lambda discard_slots: events.append(list(discard_slots))
        snapshots = self.adapter.prefix_snapshots
        restore = snapshots.restore
        snapshots.restore = lambda slot, index: (events.append(("restore", slot)), restore(slot, index))
        second = first + self.words(20)
        self.serve([second], [hit(first, second)], [2])
        self.assertEqual(events[:2], [[2], ("restore", 2)])

    def test_snapshots_sit_on_sdpa_chunk_boundaries(self):
        self.enable()
        prompt = self.words(500)
        segments = self.serve([prompt], [0], [0])
        self.assertEqual(segments, [(0, 0, 384), (0, 384, 500)])
        self.assertTrue(all(position % ALIGNMENT == 0 for position, _ in self.adapter.prefix_snapshots.keys))

    def test_disabled_adapter_keeps_the_upstream_prefill(self):
        prompt = self.words(300)
        self.assertEqual(self.serve([prompt], [0], [0]), [(0, 0, 300)])
        self.assertEqual(self.gen.counters, Counter())


if __name__ == "__main__":
    unittest.main(verbosity=2)
