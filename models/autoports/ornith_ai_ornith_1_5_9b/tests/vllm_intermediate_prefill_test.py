# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual prefill admission code with CPU-only runtime and device sentinels."""

import ast
import unittest
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_seed_continuity_test import FakeGenerator, Parameters
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan import (
    LINEAR_BUCKETS,
    plan_prefill,
    prefix_profiles_compatible,
)


def production_helper():
    path = Path(__file__).resolve().parents[1] / "tt/prefill_contract.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "PrefillContract"
    )
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("forward", "_integers")]
    namespace = {"torch": torch, "Integral": Integral, "prefix_profiles_compatible": prefix_profiles_compatible}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return type("ActualPrefillHelper", (), {name: namespace[name] for name in ("forward", "_integers")})


class Runtime:
    def __init__(self, cache):
        self.calls = []
        self.bindings = {(id(cache), slot): {"end": 0} for slot in range(2)}

    def prefill(self, ids, **kwargs):
        self.calls.append(kwargs)
        if not kwargs["start"] < kwargs["start"] + len(ids) <= kwargs["total_length"]:
            raise ValueError("Invalid horizon")
        if kwargs.get("validate_only"):
            return None
        if kwargs.get("discard_logits"):
            return None
        if kwargs["return_device_logits"]:
            return object()  # Any host conversion or assembly would fail.
        return torch.full((1, 1, 8), float(kwargs["slot"] + 3))


def production_runtime_prefill():
    path = Path(__file__).resolve().parents[1] / "tt/prefill_runtime.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "PrefillRuntime"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "prefill")
    namespace = {
        "asdict": asdict,
        "torch": torch,
        "ttnn": SimpleNamespace(Tensor=type("DeviceTensor", (), {})),
        "plan_prefill": plan_prefill,
        "prefix_profiles_compatible": prefix_profiles_compatible,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["prefill"]


class ValidatingRuntime(Runtime):
    """Production runtime validates profiles/tails; only device execution is mocked."""

    validate_prefill = production_runtime_prefill()

    def __init__(self, cache):
        super().__init__(cache)
        self.prepared = True
        self.gen = SimpleNamespace(cache=cache)
        self.model = SimpleNamespace(vocab_size=8, layers=[SimpleNamespace(kind="linear_attention")])
        self.profile_spec = {"linear": LINEAR_BUCKETS}
        for bound in self.bindings.values():
            bound.update(context=4096, fingerprint="fixed", total_length=None, layers=[{"history": {"boundary": None}}])

    def _fingerprint(self, cache, slot):
        return "fixed"

    def prefill(self, ids, **kwargs):
        self.calls.append(kwargs)
        self.validate_prefill(ids, **dict(kwargs, validate_only=True))
        if kwargs.get("validate_only"):
            return None
        bound = self.bindings[(id(kwargs["cache"]), kwargs["slot"])]
        end = kwargs["start"] + len(ids)
        plan = plan_prefill(
            total_length=kwargs["total_length"], start=kwargs["start"], end=end, kind="linear_attention"
        )
        # Model the metadata writes that follow successful device execution.
        bound["layers"][0]["history"]["boundary"] = plan.segments[-1].save_boundary_before
        bound.update(end=end, total_length=kwargs["total_length"])
        if kwargs.get("discard_logits"):
            return None
        if kwargs["return_device_logits"]:
            return object()
        return torch.full((1, 1, 8), float(kwargs["slot"] + 3))


class IntermediatePrefillTest(unittest.TestCase):
    def setUp(self):
        self.cache = object()
        self.helper = production_helper()()
        self.helper.gen = SimpleNamespace(max_seq_len=128)
        self.helper.model = SimpleNamespace(vocab_size=8, layers=[])
        self.helper.runtime = Runtime(self.cache)
        self.helper.prepared_bindings = set(self.helper.runtime.bindings)
        self.helper.events = []
        self.resets = []
        self.helper.reset_slots = lambda **kwargs: self.resets.append(kwargs)
        self.args = {
            "page_table": torch.zeros(2, 4, dtype=torch.int32),
            "kv_cache": self.cache,
            "prompt_lens": [3, 5],
            "request_lens": [11, 5],
            "reset_sampling_state": False,
        }
        self.tokens = torch.ones(2, 5, dtype=torch.int64)

    def test_mixed_rows_discard_only_intermediate_device_logits(self):
        result = self.helper.forward(self.tokens, discard_logits=[True, False], **self.args)
        calls = [call for call in self.helper.runtime.calls if not call.get("validate_only")]
        self.assertEqual([call["return_device_logits"] for call in calls], [True, False])
        self.assertEqual([call["discard_logits"] for call in calls], [True, False])
        self.assertEqual(tuple(result.shape), (2, 1, 8))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[0].tolist(), [[0.0] * 8])
        self.assertEqual(result[1].tolist(), [[4.0] * 8])
        self.assertEqual(len(self.resets), 1)

    def test_standalone_default_preserves_both_real_host_rows(self):
        result = self.helper.forward(self.tokens, **self.args)
        calls = [call for call in self.helper.runtime.calls if not call.get("validate_only")]
        self.assertEqual([call["return_device_logits"] for call in calls], [False, False])
        self.assertEqual(result[:, 0, 0].tolist(), [3.0, 4.0])

    def test_bad_discard_metadata_precedes_any_reset_or_runtime_call(self):
        for flags in ([True], [True, 1], torch.tensor([True, False]), [True, False, False]):
            with self.subTest(flags=flags), self.assertRaises((TypeError, ValueError)):
                self.helper.forward(self.tokens, discard_logits=flags, **self.args)
        for mode in ({"sample_on_device": True}, {"return_all_logits": True}):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.helper.forward(self.tokens, discard_logits=[True, False], **mode, **self.args)
        self.assertEqual(self.resets, [])
        self.assertEqual(self.helper.runtime.calls, [])

    def test_later_invalid_row_prevents_earlier_row_reset_or_execution(self):
        self.args["request_lens"] = [11, 4]
        with self.assertRaises(ValueError):
            self.helper.forward(self.tokens, discard_logits=[True, False], **self.args)
        self.assertEqual(self.resets, [])
        self.assertTrue(all(call.get("validate_only") for call in self.helper.runtime.calls))

    def test_generator_rejects_discard_modes_before_preparation(self):
        generator = FakeGenerator()
        args = {
            "page_table": generator.page_table[:2],
            "kv_cache": generator.cache,
            "prompt_lens": [3, 5],
        }
        for flags in ([True], [True, 1], [False, False, False]):
            with self.subTest(flags=flags), self.assertRaises((TypeError, ValueError)):
                generator.prefill_forward(self.tokens, discard_logits=flags, **args)
        for mode in ({"sampling_params": Parameters([42, 42])}, {"return_all_logits": True}):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                generator.prefill_forward(self.tokens, discard_logits=[True, False], **mode, **args)
        self.assertEqual(generator.events, [])

    def test_known_prompt_horizon_reuses_prefix_without_rebuild(self):
        self.helper.model.layers = [SimpleNamespace(kind="linear_attention")]
        for bound in self.helper.runtime.bindings.values():
            bound.update(end=32, total_length=100, layers=[{"history": {"boundary": None}}])
        args = dict(self.args, prompt_lens=[64, 64], start_pos=[32, 32], request_lens=[100, 100])
        self.helper.forward(torch.ones(2, 64, dtype=torch.int64), discard_logits=[True, True], **args)
        calls = [call for call in self.helper.runtime.calls if not call.get("validate_only")]
        self.assertEqual([call["start"] for call in calls], [32, 32])
        self.assertEqual(self.helper.events[-1]["rebuilt_prefix_rows"], [])
        self.assertEqual(self.resets, [])

    def test_unknown_replay_horizon_retains_prefix_rebuild(self):
        self.helper.model.layers = [SimpleNamespace(kind="linear_attention")]
        for bound in self.helper.runtime.bindings.values():
            bound.update(end=32, total_length=32, layers=[{"history": {"boundary": None}}])
        args = dict(self.args, prompt_lens=[65, 65], start_pos=[32, 32], request_lens=None)
        result = self.helper.forward(torch.ones(2, 65, dtype=torch.int64), discard_logits=[False, False], **args)
        calls = [call for call in self.helper.runtime.calls if not call.get("validate_only")]
        self.assertEqual([call["start"] for call in calls], [0, 32, 0, 32])
        self.assertEqual([call["return_device_logits"] for call in calls], [True, False, True, False])
        self.assertEqual([call["discard_logits"] for call in calls], [True, False, True, False])
        self.assertEqual(self.helper.events[-1]["rebuilt_prefix_rows"], [0, 1])
        self.assertEqual(result[:, 0, 0].tolist(), [3.0, 4.0])

    def _transition_from_unknown_to_known(self, first_end, second_end, original):
        self.helper.runtime = ValidatingRuntime(self.cache)
        self.helper.model = self.helper.runtime.model
        self.helper.gen.max_seq_len = 4096
        args = dict(self.args, page_table=torch.zeros(2, 128, dtype=torch.int32), request_lens=None)
        args["prompt_lens"] = [first_end, 130]
        self.helper.forward(torch.ones(2, max(first_end, 130), dtype=torch.int64), discard_logits=[True, False], **args)
        # Peer replay ends and the next solo admission now knows A's horizon.
        args.update(
            page_table=args["page_table"][:1],
            prompt_lens=[second_end],
            start_pos=[first_end],
            request_lens=[original],
        )
        tokens = torch.ones(1, second_end, dtype=torch.int64)
        self.helper.runtime.calls.clear()
        self.helper.forward(tokens, discard_logits=[True], **args)
        calls = [call for call in self.helper.runtime.calls if not call.get("validate_only")]
        self.assertEqual([call["start"] for call in calls], [0, first_end])
        self.assertEqual(self.helper.events[-1]["rebuilt_prefix_rows"], [0])

    def test_mixed_unknown_to_known_horizon_rebuilds_incompatible_profile(self):
        self._transition_from_unknown_to_known(first_end=512, second_end=1024, original=4096)

    def test_known_horizon_rebuilds_missing_unaligned_tail_with_compatible_profile(self):
        # Both horizons use bucket 511, but ending at the formerly final token
        # saved no boundary. Production runtime must see the rebuilt tail.
        self.assertTrue(
            prefix_profiles_compatible(
                cached_total_length=129, requested_total_length=200, prefix_end=129, kind="linear_attention"
            )
        )
        self._transition_from_unknown_to_known(first_end=129, second_end=160, original=200)


class ExecutingRuntime(ValidatingRuntime):
    """Run the complete production prefill method with observable fake kernels."""

    def __init__(self, cache):
        super().__init__(cache)
        self.staging = (object(), object())
        self.events, self.work, self.heads = [], [], []

    def reset(self, **kwargs):
        self.work.append(("reset", kwargs["slots"]))

    def _embed(self, ids, start):
        self.work.append(("embed", tuple(ids), start))

    def _save_boundary(self, index, bound, source, boundary):
        bound["layers"][index]["history"]["boundary"] = boundary
        self.work.append(("save", boundary))

    def _restore_boundary(self, index, bound, source, boundary, missing):
        self.work.append(("restore", boundary))

    def _execute(self, index, bound, source, destination, **kwargs):
        self.work.append(("layer", index, kwargs))

    def _head(self, source, start, count, **kwargs):
        self.heads.append((start, count, kwargs))
        return "device_logits" if kwargs["return_device_logits"] else "host_logits"


class RuntimeDiscardTests(unittest.TestCase):
    def runtime(self):
        cache = object()
        runtime = ExecutingRuntime(cache)
        return runtime, dict(cache=cache, page_table=torch.zeros(128, dtype=torch.int32), last_only=True)

    def test_discard_and_control_produce_same_continuation_work_and_metadata(self):
        states = []
        for discard in (False, True):
            runtime, args = self.runtime()
            first = runtime.validate_prefill([1] * 33, total_length=65, discard_logits=discard, **args)
            self.assertEqual(first, None if discard else "host_logits")
            self.assertEqual(runtime.bindings[(id(args["cache"]), 0)]["end"], 33)
            final = runtime.validate_prefill([2] * 32, start=33, total_length=65, **args)
            self.assertEqual(final, "host_logits")
            self.assertEqual(len(runtime.heads), 1 if discard else 2)
            bound = runtime.bindings[(id(args["cache"]), 0)]
            states.append((runtime.work, bound["end"], bound["total_length"], bound["layers"]))
        self.assertEqual(states[0], states[1])

    def test_discard_validation_precedes_all_state_mutation(self):
        for bad in (None, 1, "yes"):
            runtime, args = self.runtime()
            with self.assertRaises(TypeError):
                runtime.validate_prefill([1] * 33, total_length=33, discard_logits=bad, **args)
            self.assertEqual(runtime.work, [])
        runtime, args = self.runtime()
        args["last_only"] = False
        with self.assertRaises(ValueError):
            runtime.validate_prefill([1] * 33, total_length=33, discard_logits=True, **args)
        self.assertEqual(runtime.work, [])

    def test_default_device_host_and_all_logits_still_call_head(self):
        for last_only, device in ((True, False), (True, True), (False, False)):
            runtime, args = self.runtime()
            args.update(last_only=last_only, return_device_logits=device)
            output = runtime.validate_prefill([1] * 33, total_length=33, **args)
            self.assertEqual(output, "device_logits" if device else "host_logits")
            self.assertEqual(len(runtime.heads), 1)
            self.assertEqual(runtime.heads[0][2]["last_only"], last_only)


if __name__ == "__main__":
    unittest.main()
