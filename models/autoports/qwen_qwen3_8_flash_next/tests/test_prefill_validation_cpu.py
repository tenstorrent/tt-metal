# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only real prefill orchestration with instrumented tensor operations.

python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu -v

The method and parameter setter are AST-extracted: this imports no TTNN and
opens no devices. Numerical decoder correctness remains a hardware gate.
"""

import ast
import copy
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import test_decode_validation_cpu as decode_validation


class PrefillValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        decode_validation.DecodeValidationTest.setUpClass.__func__(cls)
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        owner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        nodes = [
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("prefill_forward", "prefill_spans", "_prefill_chunk", "_write_prefill_last")
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), cls.namespace)
        for node in nodes:
            setattr(cls.gen_type, node.name, cls.namespace[node.name])

    snapshot = decode_validation.DecodeValidationTest.snapshot
    assert_unchanged = decode_validation.DecodeValidationTest.assert_unchanged

    def setUp(self):
        decode_validation.DecodeValidationTest.setUp(self)
        g = self.g
        g._event = lambda *args, **kwargs: None
        self.chunks = []
        self.histories = []
        g.prefill_inputs = {n: torch.zeros(1, n, dtype=torch.int32) for n in range(1, 33)}
        g.prefill_words = {n: torch.zeros(1, n, 16, dtype=torch.int32) for n in range(1, 33)}
        g.inactive_logits = torch.zeros(1, 1, 1, 64)
        g.sample_logits = torch.zeros(1, 1, 32, 64)
        g.prefill_cache = SimpleNamespace(position=0, layers=copy.deepcopy(g.caches[0].layers))
        g.prefill_output = torch.zeros_like(g.sample_logits)
        self.state_copies = []

        def copy_state(src, dst):
            self.state_copies.append((src, dst))
            dst.position = src.position
            for left, right in zip(src.layers, dst.layers, strict=True):
                if left.attention is not None:
                    right.attention.page_table.copy_(left.attention.page_table)
            from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_checkpoint import compact_state

            for (_, left), (_, right) in zip(compact_state(src), compact_state(dst), strict=True):
                right.copy_(left)

        g.slot_mover = SimpleNamespace(_copy=copy_state)

        def reset(cache):
            self.events.append(("cache_reset", next(i for i, owned in enumerate(g.caches) if owned is cache)))
            cache.position = 0
            g.prefill_checkpoints.invalidate(i for i, owned in enumerate(g.caches) if owned is cache)

        g._reset_cache = reset
        g.sampler._penalties_active = True
        g.sampler.reset_prompt_tokens = lambda history, slots: self.histories.append((history.clone(), slots))

        def prefill(n, slot):
            values = g.prefill_inputs[n][0].clone()
            self.chunks.append((slot, values.tolist()))
            g.prefill_output.zero_()
            g.prefill_output[0, 0, :n] = torch.nn.functional.one_hot(values.long(), 64).float()
            g.prefill_cache.position += n

        g._prefill_chunk = prefill
        g._read_logits = lambda output: output.reshape(32, 64).clone()

        def sample(slots):
            self.events.append(("sample", slots))
            for slot in slots:
                g.decode_tokens[..., slot] = g.sample_logits[0, 0, slot].argmax()

        g._sample = sample
        g._read_tokens = lambda: g.decode_tokens.flatten().tolist()
        self.namespace["ttnn"].concat = lambda values, dim: torch.cat(values, dim=dim)
        self.namespace["ttnn"].full_like = lambda tensor, value, optional_tensor: optional_tensor.fill_(value)
        self.namespace["write_token_chunk"] = lambda source, destination, slot: destination[
            :, :, slot : slot + 1
        ].copy_(source)

    def call(self, **changes):
        args = dict(
            tokens=torch.arange(128).remainder(64).repeat(2, 1),
            prompt_lens=[33, 68],
            start_pos=[0, 0],
            slots=[0, 1],
            kv_cache=self.g.caches,
            page_table=torch.arange(4).repeat(2, 1),
        )
        args.update(changes)
        return self.g.prefill_forward(**args)

    def test_bad_later_row_never_resets_first_row(self):
        cases = [
            dict(tokens=torch.ones(2, 128)),
            dict(tokens=torch.ones(128, dtype=torch.long)),
            dict(tokens=torch.ones(1, 128, dtype=torch.long)),
            dict(prompt_lens=[33]),
            dict(prompt_lens=[33, 68.5]),
            dict(prompt_lens=[33, 129]),
            dict(prompt_lens=[33, 0]),
            dict(start_pos=[0]),
            dict(start_pos=[0, -1]),
            dict(start_pos=[0, 68]),
            dict(start_pos=[0, 0.5]),
            dict(slots=[0]),
            dict(slots=[0, 0]),
            dict(slots=[0, 2]),
            dict(slots=[0, 1.0]),
            dict(kv_cache=[]),
            dict(kv_cache=[self.g.caches[0]]),
            dict(kv_cache=[self.g.caches[0], object()]),
            dict(page_table=torch.ones(2, 4)),
            dict(page_table=torch.arange(4).reshape(1, 4)),
            dict(page_table=torch.ones(2, 3, dtype=torch.long)),
            dict(page_table=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])),
            dict(page_table=torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])),
            dict(sampling_params=self.params(temperature=1.0, top_k=64, top_p=1.0)),
            dict(sampling_params=self.params(temperature=[0.0], top_k=1, top_p=1.0)),
            dict(sampling_params=self.params(temperature=0.0, top_k=1, top_p=1.0), return_all_logits=True),
            dict(prefill_sample_mask=[True]),
            dict(prefill_sample_mask=[1, 0]),
            dict(prefill_sample_mask=[[True, False]]),
            dict(prefill_sample_mask=[True, False], return_all_logits=True),
        ]
        for invalid in cases:
            with self.subTest(invalid=invalid):
                before = self.snapshot()
                with self.assertRaises((ValueError, TypeError)):
                    self.call(**invalid)
                self.assert_unchanged(before)
                self.assertEqual(self.histories, [])
                self.assertEqual(self.chunks, [])

    def test_intermediate_rows_update_cache_without_readback_or_rng(self):
        def unexpected_read(*args):
            self.fail("Intermediate-only prefill read outputs")

        self.g._read_logits = unexpected_read
        self.g._read_tokens = unexpected_read
        result = self.call(prefill_sample_mask=[False, False])
        self.assertIsNone(result)
        self.assertEqual([c.position for c in self.g.caches], [33, 68])
        self.assertEqual(self.histories, [])
        self.assertFalse(any(isinstance(e, tuple) and e[0] in ("sample", "seed_reset") for e in self.events))

    def test_mixed_final_rows_only_sample_and_initialize_rng(self):
        for mask in ((True, False), (False, True)):
            with self.subTest(mask=mask):
                self.setUp()
                result = self.call(
                    prefill_sample_mask=mask,
                    sampling_params=self.params(temperature=0.0, top_k=1, top_p=1.0),
                )
                expected = [[32 if mask[0] else -1], [3 if mask[1] else -1]]
                self.assertEqual(result.tolist(), expected)
                selected = [i for i, emit in enumerate(mask) if emit]
                self.assertEqual(self.histories[0][1], selected)
                samples = [e[1] for e in self.events if isinstance(e, tuple) and e[0] == "sample"]
                self.assertEqual(samples, [tuple(selected)])
                resets = [e[3] for e in self.events if isinstance(e, tuple) and e[0] == "seed_reset"]
                self.assertEqual(resets, [selected])
                self.assertEqual([c.position for c in self.g.caches], [33, 68])

    def test_inactive_penalties_skip_history_but_preserve_sample_and_seed(self):
        self.g.sampler._penalties_active = False
        result = self.call(
            prefill_sample_mask=[False, True],
            sampling_params=self.params(temperature=0.0, top_k=1, top_p=1.0),
        )
        self.assertEqual(result.tolist(), [[-1], [3]])
        self.assertEqual(self.histories, [])
        resets = [e[3] for e in self.events if isinstance(e, tuple) and e[0] == "seed_reset"]
        self.assertEqual(resets, [[1]])
        self.assertEqual([c.position for c in self.g.caches], [33, 68])

    def test_host_compatibility_reads_only_final_rows(self):
        reads = []
        original = self.g._read_logits

        def record(output):
            reads.append(True)
            return original(output)

        self.g._read_logits = record
        result = self.call(prefill_sample_mask=[False, True])
        self.assertEqual(len(reads), 1)
        self.assertEqual(result[1].argmax(-1).item(), 3)
        self.assertTrue((result[0] == 0).all())

    def test_intermediate_host_only_parameters_do_not_reject_final_device_rows(self):
        for mask in ((True, False), (False, True), (False, False)):
            with self.subTest(mask=mask):
                self.setUp()
                before = copy.deepcopy(self.g.formatted_sampling_params)
                result = self.call(
                    prefill_sample_mask=mask,
                    sampling_params=self.params(
                        temperature=[0.7, 0.7],
                        top_k=[20 if final else 40 for final in mask],
                        top_p=[0.9, 0.9],
                        presence_penalty=[0.0 if final else 0.5 for final in mask],
                        seed=[42, 43],
                    ),
                )
                if any(mask):
                    self.assertEqual(result.tolist(), [[32 if mask[0] else -1], [3 if mask[1] else -1]])
                else:
                    self.assertIsNone(result)
                for slot, final in enumerate(mask):
                    if not final:
                        for field in self.namespace["fields"](before):
                            old = getattr(before, field.name)
                            if isinstance(old, list):
                                self.assertEqual(getattr(self.g.formatted_sampling_params, field.name)[slot], old[slot])

    def test_invalid_prefix_token_and_logical_tail_reject_before_reset(self):
        for start, invalid_index, value in ((0, 67, 64), (0, 67, -1), (32, 2, 64), (32, 67, -1)):
            with self.subTest(start=start, invalid_index=invalid_index, value=value):
                tokens = torch.arange(128).remainder(64).repeat(2, 1)
                tokens[1, invalid_index] = value
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    self.call(tokens=tokens, start_pos=[0, start])
                self.assert_unchanged(before)

    def test_numpy_metadata_mixed_lengths_and_padding_are_valid(self):
        tokens = torch.arange(128).remainder(64).repeat(2, 1)
        tokens[0, 33:] = -999
        tokens[1, 68:] = 999
        result = self.call(
            tokens=tokens,
            prompt_lens=np.array([33, 68], dtype=np.int64),
            start_pos=np.array([0, 0], dtype=np.int32),
            slots=np.array([1, 0]),
            kv_cache=[self.g.caches[1], self.g.caches[0]],
        )
        self.assertEqual(result.shape, (2, 1, 64))
        self.assertEqual(result.argmax(-1).flatten().tolist(), [32, 3])
        self.assertEqual([c.position for c in self.g.caches], [68, 33])
        history, slots = self.histories[0]
        self.assertEqual(slots, [1, 0])
        self.assertTrue((history[1, 33:] == -1).all())
        self.assertTrue((history[0, 68:] == -1).all())

    def test_resumed_prefill_uses_absolute_chunk_grid_and_slices_outputs(self):
        result = self.call(start_pos=np.array([33, 68]), prompt_lens=np.array([68, 97]), return_all_logits=True)
        self.assertEqual(
            [(slot, len(tokens)) for slot, tokens in self.chunks],
            [(0, 16), (0, 8), (0, 4), (0, 2), (0, 1), (0, 4), (1, 16), (1, 8), (1, 4), (1, 1)],
        )
        self.assertEqual(result.shape, (2, 35, 64))
        self.assertEqual(result[0].argmax(-1).tolist(), [i % 64 for i in range(33, 68)])
        self.assertEqual(result[1, :29].argmax(-1).tolist(), [i % 64 for i in range(68, 97)])
        self.assertTrue((result[1, 29:] == 0).all())
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "cache_reset" for event in self.events))
        self.assertEqual(
            self.state_copies,
            [
                (self.g.caches[0], self.g.prefill_cache),
                (self.g.prefill_cache, self.g.caches[0]),
                (self.g.caches[1], self.g.prefill_cache),
                (self.g.prefill_cache, self.g.caches[1]),
            ],
        )

    def test_dyadic_spans_cover_all_phases_without_padding_or_grid_crossing(self):
        sizes = set()
        for start in range(64):
            for length in range(1, 97):
                spans = list(self.g.prefill_spans(start, start + length))
                self.assertEqual([p for lo, hi in spans for p in range(lo, hi)], list(range(start, start + length)))
                for lo, hi in spans:
                    self.assertEqual(lo // 32, (hi - 1) // 32)
                    self.assertIn(hi - lo, (1, 2, 4, 8, 16, 32))
                    sizes.add(hi - lo)
        self.assertEqual(sizes, {1, 2, 4, 8, 16, 32})

    def test_single_slot_reuses_own_cache_without_state_copy(self):
        self.g.prefill_cache = self.g.caches[0]
        result = self.call(
            tokens=torch.arange(31).reshape(1, -1),
            prompt_lens=[31],
            start_pos=[0],
            slots=[0],
            kv_cache=[self.g.caches[0]],
            page_table=torch.arange(4).reshape(1, -1),
        )
        self.assertEqual(result.argmax(-1).item(), 30)
        self.assertFalse(self.state_copies)

    def test_prefill_traces_capture_once_per_shape_across_slots_and_requests(self):
        captured, replayed = [], []
        self.g.prefill_traces = {}
        self.g.counters.update(captures=0, prefill_replays=0)
        self.g._prefill_body = lambda count: captured.append(count)

        def record(body, signature=None):
            body()
            return len(captured)

        self.g._record_trace = record
        self.namespace["ttnn"].execute_trace = lambda mesh, trace, **kw: replayed.append(trace)
        for request in range(2):
            for slot in range(32):
                for count in (32, 16, 8, 4, 2, 1):
                    self.gen_type._prefill_chunk(self.g, count, slot)
        self.assertEqual(captured, [32, 16, 8, 4, 2, 1])
        self.assertEqual(len(self.g.prefill_traces), 6)
        self.assertEqual(self.g.counters["captures"], 6)
        self.assertEqual(self.g.counters["prefill_replays"], 384)
        self.assertEqual(replayed[:6], replayed[-6:])

    def test_sampled_first_token_respects_empty_slots_and_row_order(self):
        result = self.call(
            slots=None,
            empty_slots=np.array([1, 0]),
            kv_cache=[self.g.caches[1], self.g.caches[0]],
            prompt_lens=[1, 31],
            sampling_params=self.params(temperature=0.0, top_k=1, top_p=1.0),
        )
        self.assertEqual(result.tolist(), [[0], [30]])
        self.assertIn(("sample", (1, 0)), self.events)
        self.assertIn(("seed_reset", [None, None], [1, 31], [1, 0]), self.events)

    def test_streamed_device_rows_survive_workspace_reuse_and_clear_inactive_rows(self):
        self.assertNotEqual(self.g.prefill_output.data_ptr(), self.g.sample_logits.data_ptr())
        for length in (1, 2, 4, 8, 16, 32):
            self.g.sample_logits.fill_(777)
            result = self.call(
                prompt_lens=[length, 47],
                slots=[1, 0],
                kv_cache=[self.g.caches[1], self.g.caches[0]],
                return_device=True,
            )
            self.assertIs(result, self.g.sample_logits)
            self.assertEqual(result[0, 0, :2].argmax(-1).tolist(), [46, length - 1])
            self.assertTrue((result[0, 0, 2:] == -100).all())

    def test_prepare_warms_actual_fill_and_write_for_every_prefill_bucket(self):
        path = Path(__file__).resolve().parents[1] / "tt/generator.py"
        tree = ast.parse(path.read_text())
        prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        loop = next(n for n in prepare.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "PREFILL_BUCKETS")
        function = ast.parse("def warm(self): pass").body[0]
        function.body = [loop]
        seen = []
        self.g.prefill_cache = self.g.caches[0]

        def body(count):
            seen.append(count)
            self.g.prefill_output.fill_(count)

        self.g._prefill_body = body
        namespace = dict(self.namespace, time=time, PREFILL_BUCKETS=(1, 2, 4, 8, 16, 32), print=lambda *a, **kw: None)
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        namespace["warm"](self.g)
        self.assertEqual(seen, [1, 2, 4, 8, 16, 32])
        self.assertTrue((self.g.sample_logits[0, 0, 0] == 32).all())
        self.assertTrue((self.g.sample_logits[0, 0, 1:] == -100).all())

    def test_real_checkpoint_continuation_arbitrary_slots_and_new_odd_tails(self):
        arguments = dict(slots=[1, 0], kv_cache=[self.g.caches[1], self.g.caches[0]], return_all_logits=True)
        self.call(prompt_lens=[31, 33], **arguments)
        self.chunks.clear()
        result = self.call(prompt_lens=[35, 37], start_pos=[31, 33], **arguments)
        self.assertEqual(result.shape, (2, 4, 64))
        self.assertEqual(result.argmax(-1).tolist(), [[31, 32, 33, 34], [33, 34, 35, 36]])
        self.assertEqual(
            [(slot, len(tokens)) for slot, tokens in self.chunks], [(1, 2), (1, 2), (1, 1), (0, 4), (0, 1)]
        )
        self.assertEqual([record.end for record in self.g.prefill_checkpoints.records], [37, 35])
        self.assertEqual([cache.position for cache in self.g.caches], [37, 35])

    def test_invalid_later_submission_cannot_restore_valid_earlier_checkpoint(self):
        self.call(prompt_lens=[31, 33])
        before = [
            (record.end, record.token, [tensor.clone() for _, tensor in record.tensors])
            for record in self.g.prefill_checkpoints.records
        ]
        copies = []
        original_copy = self.g.prefill_checkpoints.copy_tensor
        self.g.prefill_checkpoints.copy_tensor = lambda a, b: (copies.append((a, b)), original_copy(a, b))
        with self.assertRaises(ValueError):
            self.call(start_pos=[31, 33], prompt_lens=[35, 129])
        self.assertFalse(copies)
        for record, (end, token, tensors) in zip(self.g.prefill_checkpoints.records, before):
            self.assertEqual((record.end, record.token), (end, token))
            for (_, current), old in zip(record.tensors, tensors):
                self.assertTrue(torch.equal(current, old))

    def test_checkpoint_restore_keeps_changed_valid_page_ownership(self):
        self.call(prompt_lens=[31, 33])
        old_bind = self.g._bind

        def bind(tables, caches, slots, positions):
            old_bind(tables, caches, slots, positions)
            for row, cache in enumerate(caches):
                cache.layers[0].attention.page_table.copy_(tables[0][row : row + 1])

        self.g._bind = bind
        changed = torch.tensor([[3, 2, 1, 0], [2, 3, 0, 1]], dtype=torch.int32)
        result = self.call(start_pos=[31, 33], prompt_lens=[35, 37], page_table=changed, return_all_logits=True)
        self.assertEqual(result.argmax(-1).tolist(), [[31, 32, 33, 34], [33, 34, 35, 36]])
        for row, cache in enumerate(self.g.caches):
            self.assertTrue(torch.equal(cache.layers[0].attention.page_table, changed[row : row + 1]))

    def test_shared_pool_cow_replay_preserves_old_pages_and_rejects_overlap_first(self):
        from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool

        class CpuTile:
            dtype, layout = "bfloat16", "tile"

            def __init__(self, shape):
                self.shape = shape
                self.value = torch.zeros(shape, dtype=torch.bfloat16)

            def __getitem__(self, index):
                return self.value[index]

        pool = SharedKVPool(
            [((16, 1, 32, 256), "bfloat16", 0)],
            allocate_pair=lambda mesh, spec: (CpuTile(spec.shape), CpuTile(spec.shape)),
        )
        g = self.g
        g.shared_pool, g.page_tables = pool, {}
        for cache in [*g.caches, g.prefill_cache]:
            cache.position = 0
            cache.layers[0].attention.k_cache, cache.layers[0].attention.v_cache = pool.pair_for_attention_layer(0)

        def bind(tables, caches, slots, positions):
            for row, (cache, slot, position) in enumerate(zip(caches, slots, positions)):
                cache.position = position
                cache.layers[0].attention.page_table.copy_(tables[0][row : row + 1])
                g.page_tables[slot] = (tables[0][row : row + 1].clone(),)

        g._bind = bind
        original_chunk = g._prefill_chunk

        def chunk(n, slot):
            start = g.prefill_cache.position
            attention = g.prefill_cache.layers[0].attention
            values = g.prefill_inputs[n].flatten().tolist()
            for index, value in enumerate(values):
                position = start + index
                page = int(attention.page_table[0, position // 32])
                attention.k_cache[page, 0, position % 32].fill_(value)
                attention.v_cache[page, 0, position % 32].fill_(value)
            original_chunk(n, slot)

        g._prefill_chunk = chunk
        tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
        self.call(prompt_lens=[31, 33], kv_cache=pool, page_table=tables)
        before = pool.pair_for_attention_layer(0)[0][0].clone()
        # The scheduler copies a prefix page into new storage before changing ownership.
        for tensor in pool.pair_for_attention_layer(0):
            tensor[8].copy_(tensor[0])
        bad = tables.clone()
        bad[0, 0] = 4
        copies = []
        real_copy = g.prefill_checkpoints.copy_tensor
        g.prefill_checkpoints.copy_tensor = lambda a, b: (copies.append((a, b)), real_copy(a, b))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.call(start_pos=[31, 33], prompt_lens=[35, 37], kv_cache=pool, page_table=bad)
        self.assertFalse(copies)
        self.assertEqual([record.end for record in g.prefill_checkpoints.records], [31, 33])
        changed = tables.clone()
        changed[0, 0] = 8
        result = self.call(
            start_pos=[31, 33], prompt_lens=[35, 37], kv_cache=pool, page_table=changed, return_all_logits=True
        )
        self.assertEqual(result.argmax(-1).tolist(), [[31, 32, 33, 34], [33, 34, 35, 36]])
        key = pool.pair_for_attention_layer(0)[0]
        self.assertTrue(torch.equal(key[0], before))
        self.assertTrue(torch.equal(key[8, 0, :30], before[0, :30]))
        self.assertEqual(key[8, 0, 30:32, 0].tolist(), [30, 31])
        self.assertEqual(g.page_tables[0][0][0, 0].item(), 8)

    def test_context_boundary_and_single_token_tails(self):
        result = self.call(prompt_lens=torch.tensor([128, 7]), start_pos=torch.tensor([127, 6]))
        self.assertEqual(result.argmax(-1).flatten().tolist(), [63, 6])
        self.assertEqual([(slot, len(tokens)) for slot, tokens in self.chunks], [(0, 1), (1, 1)])
        self.assertEqual([c.position for c in self.g.caches], [128, 7])

    def test_empty_submission_rejects_without_device_work(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.call(tokens=torch.empty(0, 0, dtype=torch.long), prompt_lens=[], start_pos=[], slots=[], kv_cache=[])
        self.assert_unchanged(before)


if __name__ == "__main__":
    unittest.main()
