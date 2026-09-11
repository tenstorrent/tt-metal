# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only adapter lifecycle tests; AST loading avoids importing device libraries.

Run with ``python3 -m unittest discover -s models/autoports/qwen_qwen3_6_27b/tests
-p test_vllm_decode_reload.py``. Device operations are replaced at the generator
boundary; the adapter method, seed manager, seed hash and Qwen arguments are real.
"""

import ast
import copy
import random
import secrets
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path


class HostTensor:
    def __init__(self, values):
        self.values = list(values)

    def reshape(self, *_):
        return self

    def __getitem__(self, index):
        return HostTensor(self.values[index]) if isinstance(index, slice) else self.values[index]

    def __ge__(self, value):
        return HostTensor([item >= value for item in self.values])

    def tolist(self):
        return list(self.values)

    def clone(self):
        return HostTensor(self.values)

    def numel(self):
        return len(self.values)

    def __setitem__(self, index, value):
        # Capture setup fills tiled metadata; these tests inspect lifecycle
        # effects at the sampler boundary rather than emulating tensor layout.
        pass


class StaleHostInput:
    def __getattr__(self, name):
        raise AssertionError(f"Steady decode consumed stale host input: {name}")


def _load_definitions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    selected = [node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _runtime():
    root = Path(__file__).resolve().parents[4]
    fake_torch = types.SimpleNamespace(
        Tensor=HostTensor,
        as_tensor=lambda value: value,
        equal=lambda left, right: left.values == right.values,
    )
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "os": types.SimpleNamespace(environ={}),
        "torch": fake_torch,
        "ttnn": types.SimpleNamespace(Topology=types.SimpleNamespace(Linear="Linear")),
        "random": random,
        "secrets": secrets,
        "copy": copy,
        "MAX_UINT32": 2**32 - 1,
        "DEVICE_SEED_MAX": 1_000_000,
        "_UINT64_MASK": 2**64 - 1,
        "format_sampling_params": lambda params, _: params,
    }
    model_dir = root / "models/autoports/qwen_qwen3_6_27b/tt"
    _load_definitions(model_dir / "generator.py", {"SamplingArgs"}, namespace)
    _load_definitions(
        root / "models/common/sampling/generator.py", {"SeedManager", "_hash_request_seed_to_device_seed"}, namespace
    )
    _load_definitions(model_dir / "generator_vllm.py", {"decode_forward"}, namespace, "Qwen36ForCausalLM")
    return namespace


class SamplingBoundary:
    def __init__(self, runtime):
        args = runtime["SamplingArgs"](248320, 248320, 32)
        self.seed_manager = runtime["SeedManager"](
            types.SimpleNamespace(_sampling_dp=1),
            max_batch_size=32,
            salt_duplicate_seeds=getattr(args, "salt_duplicate_seeds", True),
            reseed_unseeded_each_step=getattr(args, "reseed_unseeded_each_step", False),
        )
        self.device_seeds = []
        self.seed_manager.write_device_seed_values = lambda values: self.device_seeds.append(list(values))
        self.parameter_refreshes = 0
        self.prompt_history = None
        self.output_history = []

    def apply_decode_state(self, params, **kwargs):
        self.parameter_refreshes += int(kwargs["refresh_sampling_params"])
        if kwargs["reset_batch"]:
            self.reset_prompt_tokens(kwargs["prompt_tokens"])
            self.reset_output_state(kwargs["output_tokens"])

    def reset_prompt_tokens(self, tokens):
        self.prompt_history = tokens

    def reset_output_state(self, tokens):
        self.output_history = list(tokens or [])


class GeneratorBoundary:
    def __init__(self, runtime):
        self.model = types.SimpleNamespace(batch=2)
        self.sampling = SamplingBoundary(runtime)
        self.setups = []
        self.history_preservation = []
        self.replays = []
        self.remaps = []

    def setup_token_out_decode(self, tokens, positions, **kwargs):
        self.setups.append((tokens.tolist(), positions.tolist()))
        self.history_preservation.append(kwargs["preserve_sampling_history"])
        # Real setup samples once to warm its kernels. It must not become part
        # of the request's history supplied by the scheduler.
        if not kwargs["preserve_sampling_history"]:
            self.sampling.output_history.append(999)

    def remap_decode_slots(self, remap):
        self.remaps.append(remap)

    def token_out_decode_step(self, *, page_table=None, readback=False):
        self.replays.append((list(self.sampling.output_history), page_table))
        self.sampling.output_history.append(700)
        return "device tokens"


class TestDecodeReload(unittest.TestCase):
    def setUp(self):
        self.runtime = _runtime()
        self.generator = GeneratorBoundary(self.runtime)
        self.adapter = types.SimpleNamespace(
            _require_generator=lambda: self.generator,
            _sampling_key=lambda params: params.key,
            _sampling_contract_key="unchanged",
            _decode_ready=False,
            _last_page_table=None,
            _active_seed_slots=[],
        )
        self.params = types.SimpleNamespace(seed=[42, None] + [None] * 30, key="unchanged")

    def decode(self, *, tokens=None, positions=None, page_table=None, **flags):
        return self.runtime["decode_forward"](
            self.adapter,
            HostTensor([10, 20]) if tokens is None else tokens,
            HostTensor([3, -1]) if positions is None else positions,
            HostTensor([1, 2]) if page_table is None else page_table,
            None,
            read_from_device=False,
            sampling_params=self.params,
            prompt_tokens=[11, 12],
            output_tokens=[21, 22],
            **flags,
        )

    def transition(self, **kwargs):
        return self.decode(
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=True,
            **kwargs,
        )

    def test_transition_restores_history_after_warmup_and_refreshes_equal_params(self):
        self.assertEqual(self.transition(), "device tokens")
        self.assertEqual(self.generator.sampling.parameter_refreshes, 1)
        self.assertEqual(self.generator.sampling.prompt_history, [11, 12])
        self.assertEqual(self.generator.replays[0][0], [21, 22])
        self.assertEqual(self.generator.setups, [([10, 20], [3, -1])])
        self.assertEqual(self.generator.history_preservation, [False])

    def test_overlap_does_not_consume_stale_inputs_or_repeat_device_seed(self):
        self.transition()
        for _ in range(2):
            self.decode(
                tokens=StaleHostInput(),
                positions=StaleHostInput(),
                page_table=StaleHostInput(),
                reload_inputs=False,
            )
        self.assertEqual(len(self.generator.setups), 1)
        self.assertEqual(self.generator.sampling.parameter_refreshes, 1)
        self.assertEqual(self.generator.replays[-1][0], [21, 22, 700, 700])
        actual = [values[0] for values in self.generator.sampling.device_seeds]
        expected = [self.runtime["_hash_request_seed_to_device_seed"](42, counter) for counter in (4, 5, 6)]
        self.assertEqual(actual, expected)

    def test_page_table_update_preserves_tokens_history_and_counter(self):
        self.transition()
        pages = HostTensor([3, 4])
        self.decode(
            tokens=StaleHostInput(),
            positions=StaleHostInput(),
            page_table=pages,
            reload_inputs=False,
            reload_page_table=True,
        )
        self.assertEqual(len(self.generator.setups), 1)
        self.assertIs(self.generator.replays[-1][1], pages)
        self.assertEqual(self.adapter._last_page_table.values, [3, 4])
        self.assertEqual(self.generator.replays[-1][0], [21, 22, 700])

    def test_slot_remap_and_new_prefill_rebuild_authoritative_state(self):
        self.transition()
        self.adapter._decode_ready = False  # prefill invalidates resident decode
        self.params.seed = [None, 42] + [None] * 30
        self.transition(positions=HostTensor([-1, 8]), slot_remap=HostTensor([1, 0]))
        self.assertEqual(self.generator.remaps, [[1, 0]])
        self.assertEqual(self.adapter._active_seed_slots, [1])
        self.assertIsNone(self.generator.sampling.seed_manager.seeds[0])
        self.assertEqual(self.generator.replays[-1][0], [21, 22])
        self.assertEqual(
            self.generator.sampling.device_seeds[-1][1],
            self.runtime["_hash_request_seed_to_device_seed"](42, 9),
        )

    def test_legacy_calls_keep_parameter_cache_and_reload_on_reset_batch(self):
        self.decode(reset_batch=True)
        self.decode(positions=HostTensor([4, -1]))
        self.assertEqual(len(self.generator.setups), 1)
        self.assertEqual(self.generator.sampling.parameter_refreshes, 0)
        self.assertEqual(self.generator.replays[0][0], [21, 22])
        self.assertEqual(self.generator.sampling.seed_manager.seed_counters[0], 6)

    def test_recapture_without_history_reset_preserves_live_history(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                self.setUp()
                self.transition()
                kwargs = {"reload_inputs": True} if explicit else {}
                self.decode(page_table=HostTensor([8, 9]), positions=HostTensor([4, -1]), **kwargs)
                self.assertEqual(self.generator.replays[-1][0], [21, 22, 700])
                self.assertEqual(self.generator.history_preservation, [False, True])

    def test_invalid_reload_commands_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires reload_inputs"):
            self.decode(reload_inputs=False, reset_sampling_state=True)
        with self.assertRaisesRegex(ValueError, "cannot accompany"):
            self.decode(reload_inputs=True, reload_page_table=True)
        with self.assertRaisesRegex(RuntimeError, "must be loaded"):
            self.decode(reload_inputs=False)

    def test_qwen_duplicate_seeds_follow_requests_across_admission_order(self):
        args = self.runtime["SamplingArgs"](248320, 248320, 32)
        self.assertFalse(getattr(args, "salt_duplicate_seeds", True))
        results = []
        for order in (("A", "B"), ("B", "A")):
            manager = SamplingBoundary(self.runtime).seed_manager
            manager.reset_seed([42, 42], [0, 1])
            results.append({name: manager._next_device_seed_for_slot(slot) for slot, name in enumerate(order)})
        self.assertEqual(results[0], results[1])

    def test_real_capture_warmup_preserves_live_penalty_history(self):
        runtime = self.runtime
        events = []

        def clone(tensor):
            events.append("clone")
            return tensor.clone()

        def copy_tensor(source, destination):
            events.append("restore")
            destination.values = list(source.values)

        def begin_trace(*args, **kwargs):
            events.append("begin trace")
            return "trace"

        runtime["torch"] = types.SimpleNamespace(
            uint32="uint32",
            zeros=lambda shape, **kwargs: HostTensor([0] * shape[-1]),
            ones=lambda count, **kwargs: HostTensor([1] * count),
            as_tensor=lambda value, **kwargs: value,
        )
        runtime["ttnn"] = types.SimpleNamespace(
            uint32="uint32",
            bfloat16="bfloat16",
            TILE_LAYOUT="tile",
            DRAM_MEMORY_CONFIG="dram",
            from_torch=lambda value, **kwargs: value,
            ReplicateTensorToMesh=lambda mesh: None,
            synchronize_device=lambda mesh: None,
            add=lambda *args, **kwargs: None,
            clone=clone,
            copy=copy_tensor,
            deallocate=lambda tensor: events.append("deallocate"),
            begin_trace_capture=begin_trace,
            end_trace_capture=lambda *args, **kwargs: None,
        )
        model_path = Path(__file__).resolve().parents[1] / "tt/generator.py"
        _load_definitions(model_path, {"_capture_token_out_trace"}, runtime, "Qwen36Generator")
        buffers = [HostTensor([21, 22]) for _ in range(3)]
        captured = []

        def sample(logits, *, count_tokens=True, **kwargs):
            events.append("counting warmup" if count_tokens else "noncounting warmup")
            if count_tokens:
                for buffer in buffers:
                    buffer.values.append(999)

        def capture_trace(logits, **kwargs):
            captured.append(kwargs)
            return "captured sampler"

        generator = types.SimpleNamespace(
            model=types.SimpleNamespace(batch=2, layers=[], decode_forward=lambda **kwargs: HostTensor([0, 0])),
            mesh_device=object(),
            kv_cache=[],
            _page_table=object(),
            _upload=lambda value, **kwargs: value,
            _sampling_logits=lambda logits: logits,
            _seed_token_out_trace=lambda *args, **kwargs: None,
            trace_reuse=types.SimpleNamespace(record_execution=lambda: None),
            sampling=types.SimpleNamespace(
                _penalties_active=True,
                tt_penalties=types.SimpleNamespace(
                    output_mask=buffers[0], output_counts=buffers[1], output_counts_gathered=buffers[2]
                ),
                tt_sampling=types.SimpleNamespace(max_batch_size=32),
                sample=sample,
                capture_trace=capture_trace,
            ),
        )
        runtime["_capture_token_out_trace"](generator, HostTensor([10, 20]), HostTensor([3, 4]))
        self.assertEqual([buffer.values for buffer in buffers], [[21, 22]] * 3)
        self.assertEqual(
            events, ["clone"] * 3 + ["counting warmup"] + ["restore"] * 3 + ["deallocate"] * 3 + ["begin trace"]
        )
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["skip_precompile"])
        self.assertNotIn("count_tokens", captured[0])  # real replay keeps its normal counting behavior
        events.clear()
        runtime["_capture_token_out_trace"](
            generator, HostTensor([10, 20]), HostTensor([3, 4]), preserve_sampling_history=False
        )
        self.assertEqual(events, ["counting warmup", "begin trace"])


class SeedRefreshPolicyTests(unittest.TestCase):
    def sampler(self, *, qwen_policy):
        runtime = _runtime()
        root = Path(__file__).resolve().parents[4]
        runtime["TTSampling"] = lambda **kwargs: types.SimpleNamespace(max_batch_size=2, _sampling_dp=1)
        runtime["TTPenalties"] = lambda **kwargs: object()
        _load_definitions(
            root / "models/common/sampling/generator.py",
            {"__init__", "apply_prefill_state"},
            runtime,
            "SamplingGenerator",
        )
        sampler_class = type("HostSamplingGenerator", (), {"__init__": runtime["__init__"]})
        args = runtime["SamplingArgs"](248320, 248320, 2) if qwen_policy else types.SimpleNamespace()
        sampler = sampler_class(args=args, mesh_device=None, tt_ccl=None)
        sampler.reset_sampling_params = lambda *args, **kwargs: None
        sampler.reset_prompt_tokens = lambda *args, **kwargs: None
        sampler.reset_output_state = lambda *args, **kwargs: None
        writes = []
        sampler.seed_manager.write_device_seed_values = lambda values: writes.append(list(values))
        entropy = iter(range(101, 1001))
        sampler.seed_manager._next_unseeded_device_seed = lambda: next(entropy)
        return sampler, runtime, writes

    def test_qwen_prefill_and_decode_refresh_without_explicit_seed_or_skip(self):
        sampler, runtime, writes = self.sampler(qwen_policy=True)
        manager = sampler.seed_manager
        self.assertTrue(manager.reseed_unseeded_each_step)
        runtime["apply_prefill_state"](
            sampler,
            sampling_params=types.SimpleNamespace(seed=[None]),
            prompt_tokens=None,
            empty_slots=[0],
            replicate_seeds=False,
        )
        for _ in range(3):
            manager.get_new_values([0])
            self.assertFalse(manager.has_active_request_seed())  # internal tracing remains available
        self.assertEqual(writes, [[101, 102], [103, 104], [105, 106], [107, 108]])
        self.assertFalse(manager._needs_skip)

    def test_shared_default_keeps_init_skip_and_no_steady_upload(self):
        sampler, _, writes = self.sampler(qwen_policy=False)
        manager = sampler.seed_manager
        self.assertFalse(manager.reseed_unseeded_each_step)
        manager.reset_seed([None], [0])
        for _ in range(4):
            manager.get_new_values([0])
        self.assertEqual(writes, [[101, 102], [2**32 - 1] * 2])
        self.assertFalse(manager.has_active_request_seed())

    def test_explicit_and_mixed_streams_match_legacy_then_return_to_unseeded(self):
        for seeds in ([42, 99], [42, None]):
            with self.subTest(seeds=seeds):
                streams = []
                for qwen_policy in (False, True):
                    sampler, runtime, writes = self.sampler(qwen_policy=qwen_policy)
                    manager = sampler.seed_manager
                    manager.reset_seed(seeds, [0, 1])
                    manager.rngs[1].seed(12345)  # match the unseeded member's entropy for comparison
                    for _ in range(3):
                        manager.get_new_values([0, 1])
                        self.assertTrue(manager.has_active_request_seed())
                    self.assertEqual(
                        [row[0] for row in writes],
                        [runtime["_hash_request_seed_to_device_seed"](42, counter) for counter in range(3)],
                    )
                    streams.append(list(writes))
                    if qwen_policy:
                        manager.deactivate_slots_except([])
                        manager.get_new_values([0])
                        manager.get_new_values([0])
                        self.assertEqual(writes[-2:], [[101, 102], [103, 104]])
                        self.assertFalse(manager.has_active_request_seed())
                self.assertEqual(streams[0], streams[1])


if __name__ == "__main__":
    unittest.main()
