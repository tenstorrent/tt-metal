# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run actual generator boundary methods with host storage and fake TT operations.

AST extraction avoids constructing/importing a TT model. The reload planner,
preflight, prefill/decode, seed reconstruction, execution bookkeeping, and
release/reset methods are production code; only device operations are replaced.
Run with python_env/bin/python -m unittest <this module>.
"""

import ast
import unittest
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt import sampling_state
from models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract import DecodeCommands

MODEL = Path(__file__).resolve().parents[1]
METHOD_NAMES = {
    "_sampling_histories",
    "_sampling_reload_progress",
    "_record_prefill_sampling_progress",
    "_reset_sampling_state",
    "prefill_forward",
    "_decode_preflight",
    "decode_forward",
    "_execute",
    "_validate_resident_window",
    "_refresh_resident_page_bounds",
    "_invalidate_resident",
    "reset_slots",
    "release_request",
    "reset",
}


@dataclass
class Parameters:
    seed: list


class PrefillHelper:
    @staticmethod
    def prepare(generator, **_kwargs):
        generator.events.append(("prefill_prepare",))
        return generator.prefill_helper

    def forward(self, tokens, *, validate_only=False, sample_on_device=False, **kwargs):
        rows = tuple(i for i, end in enumerate(kwargs["prompt_lens"]) if end)
        slots = tuple(kwargs["empty_slots"][i] for i in rows)
        if validate_only:
            return {"input_rows": rows, "active_slots": slots}
        if sample_on_device:
            return SimpleNamespace(input_rows=rows, active_slots=slots, logits=object())
        return torch.zeros(len(rows), 1, 128)

    def reset_slots(self, **_kwargs):
        pass


def production_integers():
    tree = ast.parse((MODEL / "tt/prefill_contract.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PrefillContract")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_integers")
    method.decorator_list = []
    namespace = {"torch": torch, "Integral": Integral}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "prefill_contract.py", "exec"), namespace)
    return namespace["_integers"]


PrefillHelper._integers = staticmethod(production_integers())


def production_methods(source=None):
    tree = ast.parse(source or (MODEL / "tt/generator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OrnithGenerator")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES]
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEVICE_SAMPLING_MAX_TOP_K" for target in node.targets)
    ]

    def format_parameters(params, size, *, max_top_k):
        assert max_top_k == 64, "Canonical reset must preserve the selected sampling capacity"
        return SimpleNamespace(seed=params.seed + [None] * (size - len(params.seed)))

    namespace = {
        "Integral": Integral,
        "DecodeCommands": DecodeCommands,
        "normalize_histories": sampling_state.normalize_histories,
        # Known initial seed 101 makes off-by-one failures easy to inspect.
        "_hash_request_seed_to_device_seed": lambda seed, counter: 100 + seed,
        "format_sampling_params": format_parameters,
        "PrefillContract": PrefillHelper,
        "ttnn": SimpleNamespace(
            execute_trace=lambda *_args, **_kwargs: None,
            copy=lambda *_args, **_kwargs: None,
            get_device_tensors=lambda tensor: [tensor],
            to_torch=lambda tensor: tensor,
        ),
    }
    exec(compile(ast.Module(body=constants + methods, type_ignores=[]), "generator.py", "exec"), namespace)
    return type("ProductionMethods", (), {name: namespace[name] for name in METHOD_NAMES if name in namespace})


class FakeGenerator(production_methods()):
    def __init__(self, capacity=3):
        self.cache = [[object() for _ in range(capacity)]]
        self.model = SimpleNamespace(layers=[SimpleNamespace(kind="linear_attention")], vocab_size=128)
        self.max_seq_len = 128
        self.closed = False
        self.submissions = {}
        self._sampling_progress = {}
        self.counters = {"model_replays": 0, "sampling_replays": 0, "readbacks": 0}
        self.device = object()
        self.sampling_params = Parameters([42] * 32)
        self.sampler = SimpleNamespace(_penalties_active=False)
        self.events = []
        self.reset_values = {}
        self.prefill_helper = PrefillHelper()
        self.page_table = torch.zeros(capacity, 4, dtype=torch.int32)
        self.prefill_runtime = SimpleNamespace(
            bindings={(id(self.cache), slot): {"context": 128} for slot in range(capacity)},
            remap_slots=lambda cache, remap: self.events.append(("runtime_remap", tuple(remap))),
            reset=lambda: None,
        )
        self.sampling_state = SimpleNamespace(
            reset_history=lambda **kw: self.events.append(("history", kw)),
            reset_seed_values=self._save_seeds,
            reset_parameters=self._parameters,
            refresh_active_slots=lambda slots: None,
            apply_slot_remap=lambda remap: self.events.append(("sampler_remap", tuple(remap))),
        )
        state = SimpleNamespace(
            cache=self.cache,
            slots=tuple(range(capacity)),
            tokens=torch.zeros(1, 1, 1, 32, dtype=torch.int32),
            positions=[object() for _ in range(capacity)],
            tables=[object() for _ in range(capacity)],
            host_tables=self.page_table.clone(),
            host_positions=(),
            position_advances=0,
            forward_initialized=False,
            sampling_params_initialized=False,
            sampling_state_initialized=False,
            batch_state=SimpleNamespace(
                set_active=lambda active: None,
                permute_slots=lambda remap: self.events.append(("batch_remap", tuple(remap))),
            ),
            masked=False,
            model_traces={False: "model", True: "masked_model"},
            trace_logits={False: object(), True: object()},
            sample_traces={(masked, penalties): "sample" for masked in (False, True) for penalties in (False, True)},
        )
        self._refresh_resident_page_bounds(state)
        self.state = state
        self.states = {(id(self.cache), state.slots): state}
        self.cache_ownership = SimpleNamespace(
            serving=SimpleNamespace(cache=self.cache),
            register_inferred=lambda *args, **kw: None,
            release_request=lambda slot, reset_slots: reset_slots([slot]),
        )
        self.zero_states = [None]

    def _save_seeds(self, *, slots, values):
        self.reset_values.update(zip(slots, values))
        self.events.append(("seed", tuple(values)))

    def _parameters(self, params, *, slots):
        for row, slot in enumerate(slots):
            self.sampling_params.seed[slot] = params.seed[row]

    def _validate_sampling(self, params):
        return params

    def configure_sampling(self, params):
        self.events.append(("params",))
        self.sampling_params = params

    def prepare_decode(self, **_kwargs):
        return self.state

    def _copy(self, value, target, counter, **_kwargs):
        self.events.append((counter,))

    def _remap_forward_inputs(self, state, remap):
        state.host_tables = state.host_tables[list(remap)].clone()
        state.mapped_prefix_pages = tuple(state.mapped_prefix_pages[source] for source in remap)
        if state.host_positions:
            state.host_positions = tuple(state.host_positions[source] for source in remap)
        self.events.append(("forward_remap", tuple(remap)))

    def _sample_logits(self, logits, tokens):
        tokens.fill_(7)

    def capture(self, state):
        self.events.append(("capture",))

    def _submit_read(self, state, **_kwargs):
        return "submission"

    def prefill(self, *, ends=(33,), slots=(0,), starts=None, device=True, originals=None, tokens=None):
        tokens = torch.zeros(len(ends), max(ends), dtype=torch.int64) if tokens is None else tokens
        params = Parameters([42] * len(ends)) if device else None

        def raw_parameters(params, capacity, *, max_top_k):
            assert capacity == 32, "Physical slot capacity remains 32"
            assert max_top_k == 64, "Canonical prefill must preserve the selected sampling capacity"
            return params

        with patch.object(sampling_state, "_raw_parameters", raw_parameters):
            return self.prefill_forward(
                tokens,
                page_table=self.page_table[: len(ends)],
                kv_cache=self.cache,
                prompt_lens=ends,
                empty_slots=slots,
                start_pos=starts,
                sampling_params=params,
                original_prompt_lens=originals,
            )

    def decode(self, positions=None, *, device=True, reset=True, remap=None, outputs=None, scoped=None):
        reload = positions is not None
        return self.decode_forward(
            torch.zeros(3, dtype=torch.int64) if reload else Poison(),
            torch.tensor(positions) if reload else Poison(),
            page_table=self.page_table if reload else Poison(),
            kv_cache=self.cache,
            sampling_params=Parameters([42] * 32) if device else None,
            reload_inputs=reload,
            reload_sampling_params=reset and device,
            reset_sampling_state=reset and device,
            slot_remap=remap,
            output_tokens=outputs,
            sampling_state_slots=scoped,
            read_from_device=False,
        )


class Poison:
    def __getattribute__(self, name):
        raise AssertionError("Resident host metadata was inspected: " + name)


class SeedContinuityTests(unittest.TestCase):
    def test_fresh_resume_collision_changes_real_history_split_and_seed(self):
        prefix = torch.tensor([[1, 2, 3, 9, 9]])
        for original, expected_seed, prompt, outputs in (
            (5, 143, (1, 2, 3, 9, 9), ()),
            (3, 145, (1, 2, 3), (9, 9)),
        ):
            with self.subTest(original=original):
                gen = FakeGenerator()
                gen.prefill(ends=(5,), originals=[original], tokens=prefix)
                history = [event[1] for event in gen.events if event[0] == "history"][-1]
                self.assertEqual(history["prompt_tokens"], (prompt,))
                self.assertEqual(history["output_tokens"], (outputs,))
                self.assertEqual(gen.reset_values[0], expected_seed)
                self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (original, 6 - original))
                gen.decode([5, -1, -1])
                self.assertEqual(gen.reset_values[0], expected_seed + 1)

    def test_original_boundary_crosses_chunks_before_resumed_outputs(self):
        gen = FakeGenerator()
        prefix = torch.arange(43).reshape(1, 43)
        gen.prefill(ends=(32,), originals=[40], device=False, tokens=prefix)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (40, 0))
        gen.prefill(ends=(41,), starts=(32,), originals=[40], device=False, tokens=prefix)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (40, 1))
        gen.prefill(ends=(43,), starts=(41,), originals=[40], tokens=prefix)
        history = [event[1] for event in gen.events if event[0] == "history"][-1]
        self.assertEqual(history["prompt_tokens"], (tuple(range(40)),))
        self.assertEqual(history["output_tokens"], ((40, 41, 42),))
        self.assertEqual(gen.reset_values[0], 146)
        gen.decode([43, -1, -1])
        self.assertEqual(gen.reset_values[0], 147)

    def test_host_resumed_prefill_to_device_reset_uses_original_boundary(self):
        gen = FakeGenerator()
        gen.prefill(ends=(43,), originals=[40], device=False)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (40, 3))
        gen.decode([43, -1, -1])
        self.assertEqual(gen.reset_values[0], 147)

    def test_original_boundaries_follow_packed_rows_and_slot_reuse(self):
        gen = FakeGenerator()
        tokens = torch.tensor([[1, 2, 3, 9, 9], [5, 6, 7, 8, 8]])
        gen.prefill(ends=(5, 5), slots=(2, 0), originals=[3, 4], tokens=tokens)
        history = [event[1] for event in gen.events if event[0] == "history"][-1]
        self.assertEqual(history["slots"], (2, 0))
        self.assertEqual(history["output_tokens"], ((9, 9), (8,)))
        self.assertEqual(gen.reset_values[2], 145)
        self.assertEqual(gen.reset_values[0], 144)
        gen.release_request(2)
        gen.prefill(ends=(5,), slots=(2,), originals=[5], tokens=tokens[:1])
        self.assertEqual(gen.reset_values[2], 143)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 2)], (5, 1))
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (4, 2))

    def test_invalid_original_metadata_rejected_before_prefill_preparation(self):
        for originals in ([], [1, 2], [-1], [129], [True], [1.5], torch.tensor([3.0])):
            with self.subTest(originals=originals):
                gen = FakeGenerator()
                with self.assertRaises((TypeError, ValueError)):
                    gen.prefill(originals=originals)
                self.assertEqual(gen.events, [])
                self.assertEqual(gen._sampling_progress, {})

    def test_first_decode_uses_prefill_output_without_histories(self):
        gen = FakeGenerator()
        gen.prefill()
        self.assertEqual(gen.reset_values[0], 143)
        before = len(gen.events)
        gen.decode([33, -1, -1])
        self.assertEqual(gen.reset_values[0], 144)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (33, 2))
        events = [event[0] for event in gen.events[before:]]
        self.assertEqual(events.count("token_refreshes"), 1)
        self.assertEqual(events.count("position_refreshes"), 3)
        # Preparation already uploaded this equal table; input reload still
        # refreshes token/positions and seed state, without table transfers.
        self.assertEqual(events.count("page_table_refreshes"), 0)
        self.assertEqual(events.count("params"), 1)
        self.assertEqual(events.count("seed"), 1)
        self.assertLess(events.index("params"), events.index("seed"))
        self.assertLess(events.index("seed"), events.index("capture"))

    def test_resident_steps_and_membership_reset_preserve_offset(self):
        gen = FakeGenerator()
        gen.prefill()
        gen.decode([33, -1, -1])
        before = len(gen.events)
        gen.decode(reset=False)
        gen.decode(reset=False)
        self.assertFalse(any(e[0] in ("token_refreshes", "position_refreshes", "seed") for e in gen.events[before:]))
        gen.decode([36, -1, -1])
        self.assertEqual(gen.reset_values[0], 147)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (33, 5))

    def test_resident_remap_and_page_reload_do_not_inspect_stale_inputs(self):
        gen = FakeGenerator()
        gen.prefill(ends=(33, 40), slots=(0, 2))
        gen.decode([33, -1, 40])
        gen.decode(reset=False, remap=(2, 1, 0))
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (40, 3))
        gen.decode_forward(
            Poison(),
            Poison(),
            page_table=gen.page_table,
            kv_cache=gen.cache,
            sampling_params=Parameters([42] * 32),
            reload_inputs=False,
            reload_page_table=True,
            read_from_device=False,
        )
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 0)], (40, 4))

    def test_remap_moves_active_and_dormant_progress_once(self):
        gen = FakeGenerator()
        gen.prefill(ends=(33, 40), slots=(0, 2))
        gen.decode([33, -1, -1])
        gen.decode(reset=False)
        gen.decode([40, -1, -1], remap=(2, 1, 0))
        self.assertEqual(gen.reset_values[0], 144)
        self.assertEqual(gen.reset_values[2], 146)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 2)], (33, 3))
        for event in ("batch_remap", "sampler_remap", "runtime_remap", "forward_remap"):
            self.assertEqual(sum(e[0] == event for e in gen.events), 1)
        gen.decode([41, -1, 35])
        self.assertEqual(gen.reset_values[2], 146)

    def test_host_chunks_and_host_interval_use_absolute_positions(self):
        gen = FakeGenerator()
        gen.prefill(ends=(32,), device=False)
        gen.prefill(ends=(40,), starts=(32,), device=False)
        gen.decode([40, -1, -1])
        self.assertEqual(gen.reset_values[0], 144)
        gen.decode([41, -1, -1], device=False, reset=False)
        gen.decode([42, -1, -1], device=False, reset=False)
        gen.decode([43, -1, -1])
        self.assertEqual(gen.reset_values[0], 147)

    def test_explicit_history_rewind_and_empty_override_inference(self):
        for outputs, offset in (([[7, 8]], 2), ([[]], 0)):
            with self.subTest(offset=offset):
                gen = FakeGenerator()
                gen.prefill()
                gen.decode([40, -1, -1], outputs=outputs, scoped=(0,))
                self.assertEqual(gen.reset_values[0], 143 + offset)
                gen.decode([41, -1, -1])
                self.assertEqual(gen.reset_values[0], 144 + offset)

    def test_missing_origin_rejected_before_writes_but_explicit_history_admitted(self):
        gen = FakeGenerator()
        with self.assertRaisesRegex(ValueError, "admitted prompt metadata"):
            gen.decode([40, -1, -1])
        self.assertEqual(gen.events, [])
        gen.decode([40, -1, -1], outputs=[[7, 8]], scoped=(0,))
        self.assertEqual(gen.reset_values[0], 145)

    def test_invalid_inferred_offset_rejected_before_writes(self):
        gen = FakeGenerator()
        gen.prefill()
        before = len(gen.events)
        with self.assertRaisesRegex(ValueError, "outside context"):
            gen.decode([10, -1, -1])
        self.assertEqual(len(gen.events), before)

    def test_release_reuse_reset_and_cache_identity(self):
        gen = FakeGenerator()
        gen.prefill(ends=(33, 40), slots=(0, 1))
        peer = gen._sampling_progress[(id(gen.cache), 1)]
        gen.release_request(0)
        self.assertNotIn((id(gen.cache), 0), gen._sampling_progress)
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 1)], peer)
        gen.prefill(ends=(10,))
        gen.decode([10, -1, -1])
        self.assertEqual(gen.reset_values[0], 144)
        other_cache = [[object(), object(), object()]]
        gen._record_prefill_sampling_progress(other_cache, (0,), (0,), (20,), sampled=False)
        gen.reset()
        self.assertEqual(gen._sampling_progress, {(id(other_cache), 0): (20, 0)})

    def test_scoped_reset_does_not_reconstruct_peer_seed(self):
        gen = FakeGenerator()
        gen.prefill(ends=(33, 40), slots=(0, 1))
        gen.decode([33, 40, -1])
        gen.reset_values.clear()
        gen.decode([34, 41, -1], outputs=[[]], scoped=(0,))
        self.assertEqual(gen.reset_values, {0: 143})
        self.assertEqual(gen._sampling_progress[(id(gen.cache), 1)], (40, 3))


if __name__ == "__main__":
    unittest.main()
