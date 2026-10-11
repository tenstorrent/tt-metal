# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the actual adapter/generator sampling state methods.

AST extraction avoids importing TTNN or opening devices. The fake TT primitive
boundary holds torch CPU tensors; model forwards are no-ops and sampling emits
the input seed as a sentinel token. Thus these checks establish reload, remap,
and admission state contracts, not device math, output quality, or performance.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tt/generator_vllm.py"
GENERATOR = ROOT / "tt/generator.py"


class CPUPrimitives:
    TILE_LAYOUT = "tile"
    ROW_MAJOR_LAYOUT = "row_major"

    @staticmethod
    def copy(source, target):
        target.copy_(source.reshape(target.shape))
        return target

    @staticmethod
    def reshape(value, shape):
        return value.reshape(shape)

    @staticmethod
    def to_layout(value, layout):
        return value.clone()

    @staticmethod
    def where(mask, left, right, output_tensor=None):
        result = torch.where(mask.bool(), left, right)
        if output_tensor is not None:
            output_tensor.copy_(result)
            return output_tensor
        return result

    @staticmethod
    def plus_one(value, skip_negative_entries=False):
        if skip_negative_entries:
            value[value >= 0] += 1
        else:
            value += 1
        return value


def extracted_class(path, name, methods, namespace):
    tree = ast.parse(path.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods), "Required source method is missing"
    module = ast.Module(
        body=[ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


NAMESPACE = {"torch": torch, "ttnn": CPUPrimitives, "os": os, "json": json}
Adapter = extracted_class(
    ADAPTER,
    "MuseGlimmerForConditionalGeneration",
    {
        "__init__",
        "_event",
        "_sampling_params",
        "_remap_slots",
        "_check_sampling",
        "prefill_forward",
        "decode_forward",
    },
    NAMESPACE,
)
Generator = extracted_class(
    GENERATOR,
    "MuseGlimmerGenerator",
    {
        "_slice_coordinates",
        "_sample_graph",
        "_save_admission_graph",
        "_restore_admission_graph",
        "set_sampling_params",
        "prefill_forward",
        "decode_forward",
        "decode_steps",
    },
    NAMESPACE,
)


class CPUGenerator(Generator):
    BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)

    def __init__(self, batch):
        self.batch_size = batch
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072), vocab_size=8)
        self.prepared = True
        self.per_layer_page_tables = False
        self._unique_page_table_indices = [0]
        self.kv_cache = object()
        self.counters = Counter()
        self.events = []
        self.writes = []
        self.sample_seeds = []
        self.tokens = torch.arange(batch, dtype=torch.int32).reshape(1, 1, 1, batch) + 200
        self.positions = torch.full((batch,), -1, dtype=torch.int32)
        self.rope_indices = torch.zeros((1, batch), dtype=torch.int32)
        self._host_positions = [-1] * batch
        self._layer_page_hosts = [torch.zeros((batch, 4096), dtype=torch.int32)]
        self.logits = torch.zeros((1, 1, batch, 8))
        self.prefill_logits_buffer = torch.zeros_like(self.logits)
        self.admission_mask = torch.zeros((1, 1, 1, batch), dtype=torch.int32)
        self.saved_tokens = torch.zeros_like(self.admission_mask)
        self.saved_seeds = torch.zeros_like(self.admission_mask)
        self.row_mask = torch.zeros((1, 1, batch, 1))
        self.k = torch.ones(batch, dtype=torch.int32)
        self.p = torch.zeros(batch)
        self.temperature = torch.ones(batch)
        self.sampler = SimpleNamespace(_seeds=torch.arange(batch, dtype=torch.int32), decode_forward=self._sample)
        self.slice_start = torch.zeros(512, dtype=torch.int32)
        self.slice_end = torch.zeros(512, dtype=torch.int32)
        self.last_indices = torch.zeros((1, batch), dtype=torch.int32)
        self.prefill = {
            size: {
                "tokens": torch.zeros((1, size), dtype=torch.int32),
                "rope": torch.zeros((1, size), dtype=torch.int32),
                "table": torch.zeros((1, 4096), dtype=torch.int32),
                "fill": torch.zeros((1, size // 32), dtype=torch.int32),
                "start": torch.zeros(1, dtype=torch.int32),
                "tail_start": torch.zeros(1, dtype=torch.int32) if size == 64 else None,
            }
            for size in self.BUCKETS
        }

    def _sample(self, logits, *, tt_out_tok, **kwargs):
        self.sample_seeds.append(self.sampler._seeds.tolist())
        tt_out_tok.copy_(self.sampler._seeds.reshape(tt_out_tok.shape))

    def _write(self, value, tensor, counter=None):
        self.writes.append((id(tensor), value.clone()))
        tensor.copy_(value.reshape(tensor.shape))
        if counter:
            self.counters[counter] += 1

    def _execute(self, key):
        self.counters["execute_" + str(key)] += 1
        if key == "sample":
            self._sample_graph()
        elif key == "argmax":
            self._sample_graph(force_argmax=True)
        elif key == "admission_save":
            self._save_admission_graph()
        elif key == "admission_restore":
            self._restore_admission_graph()
        elif key == "decode":
            CPUPrimitives.plus_one(self.positions, skip_negative_entries=True)
            CPUPrimitives.plus_one(self.rope_indices)

    def _check_cache(self, cache):
        assert cache is self.kv_cache

    def _refresh_table(self, page_table, page_tables_per_layer):
        tables = page_tables_per_layer or [page_table]
        self._layer_page_hosts = [table.clone() for table in tables]

    def _event(self, event, **kwargs):
        self.events.append((event, kwargs))

    def _read_tokens(self):
        self.counters["token_reads"] += 1
        return self.tokens.flatten().tolist()

    def _read_logits(self, value):
        self.counters["logit_reads"] += 1
        return value.clone()


def owner(batch=1):
    adapter = Adapter(batch_size=batch)
    adapter.allow_host_sampling = True
    adapter.generator = CPUGenerator(batch)
    adapter._tables = lambda *args: [torch.zeros((batch, 4096), dtype=torch.int32)]
    return adapter


def params(seeds, *, k=4, temperature=1.0):
    seeds = list(seeds)
    return SimpleNamespace(
        seed=seeds, top_k=[k] * len(seeds), top_p=[0.9] * len(seeds), temperature=[temperature] * len(seeds)
    )


def prefill(adapter, ends, seeds, *, starts=None, slots=None, intermediate=None):
    rows = len(ends)
    tokens = torch.arange(max(ends), dtype=torch.int32).repeat(rows, 1)
    return adapter.prefill_forward(
        tokens,
        None,
        adapter.generator.kv_cache,
        ends,
        start_pos=starts,
        empty_slots=slots,
        sampling_params=params(seeds),
        page_tables_per_layer=[],
        intermediate_prefill_mask=intermediate,
    )


def decode(adapter, positions, sampling_params, **commands):
    return adapter.decode_forward(
        torch.zeros((adapter.batch_size, 1), dtype=torch.int32),
        positions,
        None,
        adapter.generator.kv_cache,
        sampling_params=sampling_params,
        read_from_device=False,
        page_tables_per_layer=[],
        **commands,
    )


class SamplingStateTests(unittest.TestCase):
    def assert_ledger(self, adapter):
        self.assertEqual(adapter._sampling["seeds"], adapter.generator.sampler._seeds.tolist())

    def test_whole_and_chunked_prefill_seed_alignment(self):
        whole, chunked = owner(), owner()
        whole_out = prefill(whole, [129], [11])[0]
        prefill(chunked, [128], [11], intermediate=[True])
        chunked_out = prefill(chunked, [129], [11], starts=[128])[0]
        self.assertTrue(torch.equal(whole_out, chunked_out))
        self.assertEqual(whole.generator.sample_seeds, [[139]])
        self.assertEqual(chunked.generator.sample_seeds, [[139]])
        self.assertEqual(whole._sampling["seeds"], [140])
        self.assert_ledger(whole)
        self.assert_ledger(chunked)

    def test_final_prefill_and_reload_share_next_counter(self):
        adapter = owner()
        prefill(adapter, [129], [11])
        next_seed = adapter._sampling["seeds"][:]
        decode(adapter, [129], params([11]), reload_inputs=True, reload_sampling_params=True, reset_sampling_state=True)
        self.assertEqual(adapter.generator.sample_seeds[-1], next_seed)
        self.assertEqual(adapter._sampling["seeds"], [141])
        self.assert_ledger(adapter)

    def test_parameter_only_changed_base_ignores_stale_position(self):
        adapter = owner()
        prefill(adapter, [129], [11])
        decode(adapter, [-999], params([21], k=7), reload_inputs=False, reload_sampling_params=True)
        self.assertEqual(adapter.generator.sample_seeds[-1], [150])
        self.assertEqual(adapter._sampling["seeds"], [151])
        self.assertEqual(adapter._sampling["top_k"], [7])
        self.assertEqual(adapter.generator._host_positions, [130])
        self.assert_ledger(adapter)

    def test_state_only_reset_keeps_cached_parameters_and_base(self):
        adapter = owner()
        prefill(adapter, [129], [11])
        decode(
            adapter,
            [129],
            params([999], k=7, temperature=2.0),
            reload_inputs=True,
            reset_sampling_state=True,
            reload_sampling_params=False,
        )
        self.assertEqual(adapter.generator.sample_seeds[-1], [140])
        self.assertEqual(adapter._seed_bases, [11])
        self.assertEqual(adapter._sampling["top_k"], [4])
        self.assertEqual(adapter._sampling["temperature"], [1.0])
        self.assert_ledger(adapter)

    def test_seed_integer_boundary_matches_resident_increment(self):
        for seed in (2147483646, 2147483647, 2**63 - 1, -(2**63)):
            with self.subTest(seed=seed):
                adapter = owner()
                prefill(adapter, [1], [seed])
                resident = adapter._sampling["seeds"][:]
                decode(
                    adapter,
                    [1],
                    params([seed]),
                    reload_inputs=True,
                    reload_sampling_params=True,
                    reset_sampling_state=True,
                )
                self.assertEqual(adapter.generator.sample_seeds[-1], resident)
                self.assert_ledger(adapter)

    def test_intermediate_only_preserves_all_sampling_state(self):
        adapter = owner(2)
        before = copy.deepcopy(adapter._sampling)
        bases = adapter._seed_bases[:]
        tokens = adapter.generator.tokens.clone()
        out, _ = prefill(adapter, [64, 65], [11, None], intermediate=[True, True])
        self.assertEqual(adapter._sampling, before)
        self.assertEqual(adapter._seed_bases, bases)
        self.assertTrue(torch.equal(adapter.generator.tokens, tokens))
        self.assertEqual(adapter.generator.sample_seeds, [])
        self.assertEqual(adapter.generator.counters["token_reads"], 0)
        self.assertEqual(adapter.generator.counters["logit_reads"], 0)
        self.assertEqual(out.shape, (2, 1))
        self.assert_ledger(adapter)

    def test_first_mixed_final_intermediate_preserves_nonemitting_rows(self):
        adapter = owner(3)
        before = copy.deepcopy(adapter._sampling)
        bases = adapter._seed_bases[:]
        tokens = adapter.generator.tokens.flatten().clone()
        prefill(adapter, [65, 64], [11, None], slots=[0, 1], intermediate=[False, True])
        self.assertEqual(adapter._sampling["seeds"][0], 76)
        for slot in (1, 2):
            for key in before:
                self.assertEqual(adapter._sampling[key][slot], before[key][slot])
            self.assertEqual(adapter._seed_bases[slot], bases[slot])
            self.assertEqual(int(adapter.generator.tokens.flatten()[slot]), int(tokens[slot]))
        self.assertEqual(len(adapter.generator.sample_seeds), 1)
        self.assert_ledger(adapter)

    def test_admission_preserves_ongoing_request(self):
        adapter = owner(3)
        prefill(adapter, [65], [11], slots=[0])
        before = copy.deepcopy(adapter._sampling)
        tokens = adapter.generator.tokens.flatten().clone()
        prefill(adapter, [64, 65], [None, 22], slots=[1, 2], intermediate=[True, False])
        for slot in (0, 1):
            for key in before:
                self.assertEqual(adapter._sampling[key][slot], before[key][slot])
            self.assertEqual(int(adapter.generator.tokens.flatten()[slot]), int(tokens[slot]))
        self.assertEqual(adapter._sampling["seeds"][2], 87)
        self.assert_ledger(adapter)

    def test_dormant_host_remap_is_simultaneous_and_exactly_once(self):
        adapter = owner(3)
        prefill(adapter, [5, 9, 13], [11, 22, 33])
        before = copy.deepcopy(adapter._sampling)
        bases = adapter._seed_bases[:]
        permutation = [2, 0, 1]
        decode(adapter, [13, 5, 9], None, reload_inputs=True, slot_remap=permutation)
        self.assertEqual(adapter._sampling, {key: [values[i] for i in permutation] for key, values in before.items()})
        self.assertEqual(adapter._seed_bases, [bases[i] for i in permutation])
        self.assert_ledger(adapter)
        decode(adapter, [6, 10, 14], None, reload_inputs=True, slot_remap=[1, 2, 0])
        self.assertEqual(adapter._sampling, before)
        self.assertEqual(adapter._seed_bases, bases)
        self.assert_ledger(adapter)

    def test_remap_rejects_invalid_permutations_and_skips_identity_write(self):
        adapter = owner(3)
        for bad in ([0, 0, 2], [0, 1], [0, 1, 3], [-1, 1, 2]):
            with self.subTest(remap=bad), self.assertRaises(ValueError):
                adapter._remap_slots(bad)
        writes = len(adapter.generator.writes)
        adapter._remap_slots([0, 1, 2])
        self.assertEqual(len(adapter.generator.writes), writes)

    def test_unseeded_parameter_only_update_keeps_base_and_counter(self):
        adapter = owner()
        prefill(adapter, [65], [None])
        old_base = adapter._seed_bases[:]
        old_seed = adapter._sampling["seeds"][:]
        decode(adapter, [-123], params([None], k=8), reload_inputs=False, reload_sampling_params=True)
        self.assertEqual(adapter._seed_bases, old_base)
        self.assertEqual(adapter.generator.sample_seeds[-1], old_seed)
        self.assertEqual(adapter._sampling["seeds"], [old_seed[0] + 1])
        self.assert_ledger(adapter)

    def test_inactive_physical_row_reanchors_before_request_resume(self):
        adapter = owner(2)
        prefill(adapter, [65, 67], [11, 22])
        second_request_next_seed = adapter._sampling["seeds"][1]
        decode(
            adapter,
            [65, -1],
            params([11, 22]),
            reload_inputs=True,
            reload_sampling_params=True,
            reset_sampling_state=True,
        )
        # The canonical fixed-batch sampler changes even inactive physical
        # rows. Their discarded draws are not request-counter authority.
        self.assertNotEqual(adapter._sampling["seeds"][1], second_request_next_seed)
        self.assert_ledger(adapter)
        decode(
            adapter,
            [66, 67],
            params([11, 22]),
            reload_inputs=True,
            reload_sampling_params=True,
            reset_sampling_state=True,
        )
        self.assertEqual(adapter.generator.sample_seeds[-1][1], second_request_next_seed)
        self.assert_ledger(adapter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/host_sampling_state.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SamplingStateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "CPU AST-extracted adapter/generator methods with fake TT primitive boundary; no device evidence",
        "tests_run": result.testsRun,
        "failures": [{"test": str(case), "traceback": trace} for case, trace in result.failures],
        "errors": [{"test": str(case), "traceback": trace} for case, trace in result.errors],
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ADAPTER, GENERATOR)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status'].upper()}: {result.testsRun} host sampling-state tests; {args.output}")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
