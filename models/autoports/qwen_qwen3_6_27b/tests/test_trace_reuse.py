# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only allocation and lifecycle guards; no device-library imports.

The controller and input-copy method are loaded from their real ASTs. Device
boundaries expose allocation identities and unsafe maps so failures can be
injected before and between model and sampler replay.
"""

import math
import types
import unittest
from collections import Counter, namedtuple
from pathlib import Path

from test_vllm_decode_reload import _load_definitions


class HostTensor:
    def __init__(self, values, shape=None, dtype="uint32"):
        self.values = list(values)
        self.shape = shape or (len(self.values),)
        self.dtype = dtype

    def reshape(self, *_):
        return self

    def nonzero(self):
        return HostTensor([index for index, value in enumerate(self.values) if value])

    def tolist(self):
        return self.values.copy()

    def numel(self):
        return len(self.values)

    def repeat(self, count):
        return HostTensor(self.values * count)

    def to(self, dtype):
        return HostTensor(self.values, self.shape, dtype)

    def __setitem__(self, index, value):
        self.values[index[-1]] = value.values


class DeviceTensor:
    def __init__(self, shape=(1, 1, 1, 32)):
        self.shape = self.padded_shape = shape
        self.dtype, self.layout = "uint32", "tile"
        self.allocated = True
        self.values = []

    def is_allocated(self):
        return self.allocated

    def buffer_unique_id(self):
        return id(self)

    def buffer_address(self):
        return id(self)

    def memory_config(self):
        return "DRAM"


TraceKey = namedtuple("TraceKey", "penalties_on log_probs_on force_argmax bucket")


class Runtime:
    def __init__(self, *, tracking=True, skip_program_cache="0", enabled="1"):
        self.calls, self.queries, self.unsafe = [], [], {}
        self.entries, self.thread, self.releases = 42, 1, 0
        self.mesh = types.SimpleNamespace(num_program_cache_entries=lambda: self.entries)
        self.ttnn = types.SimpleNamespace(
            TRACE_ALLOC_TRACKING=tracking,
            synchronize_device=lambda mesh: None,
            uint32="uint32",
            bfloat16="bfloat16",
            ROW_MAJOR_LAYOUT="row_major",
            TILE_LAYOUT="tile",
            from_torch=lambda tensor, **kwargs: tensor,
            copy_host_to_device_tensor=lambda host, device: setattr(device, "values", host.tolist()),
            _ttnn=types.SimpleNamespace(
                operations=types.SimpleNamespace(
                    trace=types.SimpleNamespace(get_unsafe_tracked_ids=self.query, execute_trace=self.execute)
                )
            ),
        )
        self.namespace = {
            "os": types.SimpleNamespace(
                environ={"QWEN36_TRACE_REUSE": enabled, "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": skip_program_cache}
            ),
            "threading": types.SimpleNamespace(get_ident=lambda: self.thread),
            "Counter": Counter,
            "torch": types.SimpleNamespace(
                Tensor=HostTensor,
                uint32="uint32",
                bfloat16="bfloat16",
                as_tensor=lambda value, **kwargs: value if isinstance(value, HostTensor) else HostTensor(value),
                zeros=lambda shape, **kwargs: HostTensor([0] * math.prod(shape), shape),
            ),
            "ttnn": self.ttnn,
            "format_sampling_params": lambda params, batch: params,
        }
        source = Path(__file__).resolve().parents[1] / "tt"
        _load_definitions(source / "trace_reuse.py", {"tensor_binding", "DecodeTraceReuse"}, self.namespace)
        _load_definitions(source / "generator.py", {"_seed_token_out_trace"}, self.namespace, "Qwen36Generator")
        self.gen = types.SimpleNamespace(
            mesh_device=self.mesh,
            model=types.SimpleNamespace(
                batch=32, layers=[types.SimpleNamespace(layer_idx=0, caches={"recurrent": DeviceTensor()})]
            ),
            kv_cache=object(),
            page_table_host=HostTensor([0] * 256, (32, 8)),
            _page_table=DeviceTensor((32, 8)),
            _decode_trace_id=1,
        )
        for name in ("token", "position", "active_mask", "active_state_mask", "logits"):
            setattr(self.gen, f"_trace_{name}", DeviceTensor())
        self.gen._trace_page_table = self.gen._page_table
        self.gen.sampling = types.SimpleNamespace(
            _active_trace_bucket=None,
            _penalties_active=False,
            _log_probs_active=False,
            tt_sampling=types.SimpleNamespace(force_argmax_sampling=True),
            _trace_states={
                TraceKey(False, False, True, None): {
                    "id": 2,
                    "input": self.gen._trace_logits,
                    "output": self.gen._trace_token,
                }
            },
            _validate_trace_inputs=self.validate_inputs,
        )
        self.controller = self.namespace["DecodeTraceReuse"](self.gen)
        self.gen._release_traces = self.release
        self.params = types.SimpleNamespace(
            seed=[None] * 32,
            enable_log_probs=[False] * 32,
            presence_penalty=[0] * 32,
            frequency_penalty=[0] * 32,
            repetition_penalty=[1] * 32,
            top_k=[1] * 32,
            top_p=[1.0] * 32,
            temperature=[1.0] * 32,
        )

    def query(self, mesh, trace_id):
        self.queries.append(trace_id)
        return self.unsafe.get(trace_id, {})

    def execute(self, mesh, trace_id, **kwargs):
        self.calls.append(trace_id)

    def release(self):
        self.releases += 1
        self.controller.invalidate("traces released")

    def validate_inputs(self, slot, logits, token):
        if slot["input"] is not logits or slot["output"] is not token:
            raise RuntimeError("sampler input owner mismatch")

    def begin(self, length=128, slot=0):
        return self.controller.begin_prefill(
            physical_seq_len=length,
            prompt_lens=(length,),
            slots=(slot,),
            page_table=self.gen.page_table_host,
            kv_cache=self.gen.kv_cache,
            sampling_params=self.params,
        )

    def warm(self, length=128, slot=0):
        self.begin(length, slot)
        self.controller.end_prefill()
        self.controller.captured_setup(self.params)

    def reuse_setup(self, slot=0):
        return self.controller.can_reuse_setup(
            sampling_params=self.params,
            active_values=[int(index == slot) for index in range(32)],
            page_table=self.gen.page_table_host,
            kv_cache=self.gen.kv_cache,
        )


class TestTraceReuse(unittest.TestCase):
    def test_unknown_shape_releases_before_prefill_and_warmed_shapes_survive_release(self):
        runtime = Runtime()
        runtime.warm(128)
        self.assertFalse(runtime.begin(4096))
        self.assertIsNone(runtime.controller.execution)
        self.assertEqual(runtime.releases, 2)
        runtime.controller.end_prefill()
        runtime.controller.captured_setup(runtime.params)
        self.assertEqual(len(runtime.controller.warmed), 2)
        self.assertTrue(runtime.begin(128))
        self.assertTrue(runtime.controller.end_prefill())
        self.assertTrue(runtime.reuse_setup())

    def test_tracking_and_program_tracking_are_required(self):
        for kwargs in ({"tracking": False}, {"skip_program_cache": "1"}):
            with self.subTest(kwargs=kwargs):
                runtime = Runtime(**kwargs)
                self.assertFalse(runtime.begin())
                self.assertIsNone(runtime.controller.request_key)
                self.assertEqual(runtime.releases, 1)

    def test_control_disables_reuse_but_retains_checked_execution(self):
        runtime = Runtime(enabled="0")
        runtime.warm()
        runtime.controller.execute(runtime.mesh, 1)
        self.assertEqual(runtime.calls, [1])
        self.assertFalse(runtime.begin())

    def test_prefill_surviving_allocation_recaptures_before_decode(self):
        runtime = Runtime()
        runtime.warm()
        self.assertTrue(runtime.begin())
        runtime.unsafe[2] = {100: "ordinary prefill output"}
        self.assertFalse(runtime.controller.end_prefill())
        self.assertEqual(runtime.controller.counters["guard_recaptures"], 1)
        self.assertIsNone(runtime.controller.execution)
        self.assertFalse(runtime.reuse_setup())
        self.assertEqual(runtime.calls, [])

    def test_program_growth_or_cache_replacement_invalidates_before_replay(self):
        for changed in ("program", "cache"):
            with self.subTest(changed=changed):
                runtime = Runtime()
                runtime.warm()
                self.assertTrue(runtime.begin())
                if changed == "program":
                    runtime.entries += 1
                else:
                    runtime.gen.model.layers[0].caches["recurrent"] = DeviceTensor()
                self.assertFalse(runtime.controller.end_prefill())
                self.assertIsNone(runtime.controller.execution)
                self.assertEqual(runtime.calls, [])

    def test_sampler_only_unsafe_allocation_stops_first_model_execution(self):
        runtime = Runtime()
        runtime.warm()
        runtime.queries.clear()
        runtime.unsafe[2] = {101: "sampler-only hazard"}
        with self.assertRaisesRegex(RuntimeError, "unsafe allocations"):
            runtime.controller.execute(runtime.mesh, 1)
        self.assertEqual(runtime.queries, [1, 2])
        self.assertEqual(runtime.calls, [])

    def test_late_sampler_failure_aborts_without_repeating_model_step(self):
        runtime = Runtime()
        runtime.warm()
        runtime.controller.execute(runtime.mesh, 1)
        runtime.unsafe[2] = {102: "late allocation"}
        with self.assertRaisesRegex(RuntimeError, "unsafe allocations"):
            runtime.controller.execute(runtime.mesh, 2)
        self.assertEqual(runtime.calls, [1])
        self.assertEqual(runtime.releases, 1)

    def test_all_live_sampler_traces_are_checked(self):
        runtime = Runtime()
        runtime.gen.sampling._trace_states[TraceKey(True, False, True, None)] = {
            "id": 3,
            "input": DeviceTensor(),
            "output": DeviceTensor(),
        }
        runtime.warm()
        runtime.unsafe[3] = {103: "older sampler trace hazard"}
        with self.assertRaisesRegex(RuntimeError, "unsafe allocations"):
            runtime.controller.execute(runtime.mesh, 1)
        self.assertEqual(runtime.calls, [])
        self.assertIn(3, runtime.queries)

    def test_inactive_live_bucket_cannot_bypass_complete_tracking(self):
        runtime = Runtime()
        runtime.gen.sampling._trace_states[TraceKey(True, False, True, "old bucket")] = {
            "id": 3,
            "input": DeviceTensor(),
            "output": DeviceTensor(),
        }
        with self.assertRaisesRegex(RuntimeError, "missing sampler ownership"):
            runtime.controller.record_execution()
        self.assertIsNone(runtime.controller.execution)
        self.assertEqual(runtime.calls, [])

    def test_sampled_or_seeded_modes_keep_checked_execution_without_cross_request_reuse(self):
        for mode in ("sampled", "seeded"):
            with self.subTest(mode=mode):
                runtime = Runtime()
                if mode == "sampled":
                    runtime.params.top_k = [32] * 32
                else:
                    runtime.params.seed[0] = 42
                self.assertFalse(runtime.begin())
                runtime.controller.record_execution()
                runtime.controller.execute(runtime.mesh, 1)
                self.assertEqual(runtime.calls, [1])
                self.assertIsNone(runtime.controller.captured)

    def test_thread_change_and_unknown_trace_fail_before_submission(self):
        runtime = Runtime()
        runtime.warm()
        runtime.thread = 2
        with self.assertRaisesRegex(RuntimeError, "serialized device-submit thread"):
            runtime.controller.execute(runtime.mesh, 1)
        runtime.thread = 1
        with self.assertRaisesRegex(RuntimeError, "outside the resident pair"):
            runtime.controller.execute(runtime.mesh, 77)
        self.assertEqual(runtime.calls, [])

    def test_malformed_contract_releases_and_preserves_exception(self):
        runtime = Runtime()
        runtime.warm()
        runtime.params.top_k = None
        with self.assertRaises(TypeError):
            runtime.begin()
        self.assertIsNone(runtime.controller.execution)
        self.assertIsNone(runtime.controller.request_key)

    def test_malformed_slot_releases_existing_traces_before_raising(self):
        runtime = Runtime()
        runtime.warm()
        with self.assertRaises(ValueError):
            runtime.begin(slot="invalid slot")
        self.assertIsNone(runtime.controller.execution)
        self.assertIsNone(runtime.controller.request_key)
        self.assertEqual(runtime.controller.request_slots, ())
        self.assertEqual(runtime.releases, 2)

    def test_recording_reuse_eligibility_does_not_double_count_captures(self):
        runtime = Runtime()
        runtime.warm()
        self.assertIsNotNone(runtime.controller.captured)
        self.assertEqual(runtime.controller.counters["captures"], 0)

    def test_warmed_slot_change_refreshes_both_mask_contents_in_place(self):
        runtime = Runtime()
        runtime.warm(slot=7)
        runtime.warm(slot=0)
        self.assertTrue(runtime.begin(slot=7))
        self.assertTrue(runtime.controller.end_prefill())
        self.assertTrue(runtime.reuse_setup(slot=7))
        bindings = runtime.controller._trace_bindings()
        active = [int(index == 7) for index in range(32)]
        runtime.namespace["_seed_token_out_trace"](runtime.gen, [10] * 32, [128] * 32, active_mask=active)
        self.assertEqual(runtime.gen._trace_active_mask.values, active)
        self.assertEqual(runtime.gen._trace_active_state_mask.values, active)
        self.assertEqual(runtime.controller._trace_bindings(), bindings)


class TestSamplerExecutor(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        root = Path(__file__).resolve().parents[4]
        _load_definitions(
            root / "models/common/sampling/generator.py",
            {"_execute_trace", "sample"},
            self.runtime.namespace,
            "SamplingGenerator",
        )
        self.default_calls, self.checked_calls, self.eager_calls = [], [], []
        self.runtime.ttnn.execute_trace = lambda *args, **kwargs: self.default_calls.append((args, kwargs))
        self.sampler = self.runtime.gen.sampling
        self.sampler.mesh_device, self.sampler.cq_id = self.runtime.mesh, 0
        self.sampler._execute_trace = types.MethodType(self.runtime.namespace["_execute_trace"], self.sampler)
        self.sampler.seed_manager = types.SimpleNamespace(has_active_request_seed=lambda: False)
        key, slot = next(iter(self.sampler._trace_states.items()))
        self.sampler._trace_slot = lambda *args: (key, slot)
        self.sampler._run_sampling = lambda *args, **kwargs: self.eager_calls.append((args, kwargs)) or "eager output"

    def sample(self, **kwargs):
        return self.runtime.namespace["sample"](
            self.sampler, self.runtime.gen._trace_logits, tt_out_tok=self.runtime.gen._trace_token, **kwargs
        )

    def checked(self, *args, **kwargs):
        self.checked_calls.append((args, kwargs))

    def test_default_executor_behavior_is_unchanged(self):
        self.assertIs(self.sample(), self.runtime.gen._trace_token)
        self.assertEqual(self.default_calls, [((self.runtime.mesh, 2), {"cq_id": 0, "blocking": False})])
        self.assertEqual(self.checked_calls, [])

    def test_instance_callback_uses_same_trace_and_output(self):
        self.assertIs(self.sample(trace_executor=self.checked), self.runtime.gen._trace_token)
        self.assertEqual(self.checked_calls, [((self.runtime.mesh, 2), {"cq_id": 0, "blocking": False})])
        self.assertEqual(self.default_calls, [])

    def test_explicit_seed_keeps_eager_sampling_and_token_count_policy(self):
        self.sampler.seed_manager.has_active_request_seed = lambda: True
        self.assertEqual(self.sample(trace_executor=self.checked, count_tokens=False), "eager output")
        self.assertEqual(self.checked_calls, [])
        self.assertEqual(self.default_calls, [])
        self.assertFalse(self.eager_calls[0][1]["count_tokens"])


if __name__ == "__main__":
    unittest.main()
