# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capacity wiring against the accepted model API, with every native boundary replaced."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry
from models.demos.llama_3p1_8b_d_p.tt.runners.kv_layout import PrefillKVLayout
from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import (
    TtPrefillRuntime,
    TtPrefillRuntimeConfig,
    build_runtime,
    config_from_params,
)

PREFIX = "models.demos.llama_3p1_8b_d_p.tt."


def params(**overrides):
    values = dict(
        mesh_shape=(4, 8),
        num_layers=32,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=2048,
        chunk_size=1024,
        num_users=2,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    values.update(overrides)
    return PrefillRunParams(**values)


class RuntimeCapacityTests(unittest.TestCase):
    # Allocation, model, upload and table agree on capacity while H2D input stays borrowed and unmodified.
    def test_capacity_agrees_through_all_factories_and_two_slots(self):
        for capacity in (2048, 4096, 131072):
            with self.subTest(capacity=capacity):
                seen = {}
                events = []
                dtype = object()
                borrowed_upload = object()

                def allocate(mesh, mesh_config, *, num_users, num_layers, max_seq_len, cache_dtype):
                    seen["allocation"] = (num_users, num_layers, max_seq_len, cache_dtype)
                    return SimpleNamespace(num_users=num_users, num_layers=num_layers, max_seq_len=max_seq_len, sp=4)

                def model(mesh, checkpoint, *, num_layers, enable_lm_head, cache_dtype, max_seq_len):
                    seen["model"] = (num_layers, enable_lm_head, cache_dtype, max_seq_len)
                    return SimpleNamespace(
                        num_layers=num_layers,
                        max_seq_len=max_seq_len,
                        prefill_chunk=lambda tokens, cache, **kw: events.append(("forward", tokens, kw)),
                    )

                def upload(mesh, ids, *, actual_start, actual_end, max_seq_len):
                    PrefillGeometry(max_seq_len).validate_chunk_range(actual_start, actual_end)
                    seen.setdefault("uploads", []).append((ids, actual_start, actual_end, max_seq_len))
                    return borrowed_upload

                def table(**kwargs):
                    cache = kwargs["kv_cache"]
                    layout = PrefillKVLayout(
                        max_seq_len=cache.max_seq_len,
                        num_slots=cache.num_users,
                        num_layers=cache.num_layers,
                        num_banks=8,
                        chunk_size=kwargs["chunk_size"],
                    )
                    seen["table"] = layout
                    return kwargs["path"]

                stubs = {
                    "loguru": SimpleNamespace(logger=SimpleNamespace(info=lambda *args: None)),
                    "ttnn": SimpleNamespace(bfloat8_b=dtype, synchronize_device=lambda mesh: events.append(("sync",))),
                    PREFIX + "config": SimpleNamespace(MeshConfig=lambda shape, tp: (shape, tp)),
                    PREFIX + "kv_cache": SimpleNamespace(allocate_kv_cache=allocate),
                    PREFIX + "model": SimpleNamespace(PrefillModel=model),
                    PREFIX + "input": SimpleNamespace(upload_token_chunk=upload),
                    PREFIX + "runners.kv_chunk_table": SimpleNamespace(build_and_serialize_kv_chunk_table=table),
                }
                with patch.dict(sys.modules, stubs):
                    adapter = get_adapter("llama_3p1_8b")
                    request_params = params(max_seq_len=capacity)
                    cache = adapter.allocate_kv_cache(mesh_device="mesh", hf_config=None, params=request_params)
                    runtime = build_runtime("mesh", params=request_params, checkpoint_path="checkpoint")
                    self.assertIs(
                        runtime.make_chunk_input(list(range(32)), actual_start=capacity - 32), borrowed_upload
                    )
                    runtime.compiled = True
                    runtime.set_layer_completion_sink(lambda layer, request: events.append(("ack", layer, request)))
                    for slot in (0, 1):
                        tokens = object()
                        runtime.prefill_chunk(
                            tokens,
                            cache,
                            slot_id=slot,
                            actual_start=capacity - 32,
                            actual_end=capacity,
                            request_id=slot,
                        )
                        forward = [event for event in events if event[0] == "forward"][-1]
                        self.assertIs(forward[1], tokens)
                        self.assertEqual(forward[2]["slot_idx"], slot)
                    self.assertEqual(runtime.build_kv_chunk_table(cache, "table.pb"), "table.pb")
                self.assertEqual(seen["allocation"], (2, 32, capacity, dtype))
                self.assertEqual(seen["model"], (32, False, dtype, capacity))
                self.assertEqual(seen["uploads"], [(list(range(32)), capacity - 32, capacity, capacity)])
                self.assertEqual(seen["table"].max_seq_len, capacity)
                self.assertNotEqual(
                    seen["table"].locate(15, 31, capacity - 32, 0, 65536),
                    seen["table"].locate(15, 31, capacity - 32, 1, 65536),
                )
                self.assertEqual(
                    [event for event in events if event[0] == "ack"],
                    [("ack", layer, slot) for slot in (0, 1) for layer in range(32)],
                )
                self.assertEqual(sum(event[0] == "sync" for event in events), 2)

    # The shared geometry rejects malformed lengths and unsupported topology before touching a native factory.
    def test_invalid_geometry_fails_before_native_import(self):
        for change in (
            dict(max_seq_len=0),
            dict(max_seq_len=2016),
            dict(max_seq_len=132096),
            dict(max_seq_len=True),
            dict(chunk_size=512),
            dict(num_users=1),
            dict(use_trace=True),
            dict(tp_shard_kv=True),
            dict(dflash_enabled=True),
        ):
            with self.subTest(change=change):
                with self.assertRaises((TypeError, ValueError, NotImplementedError)):
                    build_runtime("mesh", params=params(**change), checkpoint_path="checkpoint")

    # Direct runtime configuration must share the model geometry ceiling, including Python-type validation.
    def test_runtime_config_rejects_out_of_range_capacity(self):
        for capacity in (-1024, 0, 2016, 132096, True, 4096.0):
            with self.subTest(capacity=capacity):
                with self.assertRaises((TypeError, ValueError)):
                    TtPrefillRuntimeConfig(max_seq_len=capacity, chunk_size=1024, num_users=2)
        for capacity in (2048, 4096, 131072):
            self.assertEqual(config_from_params(params(max_seq_len=capacity)).max_seq_len, capacity)

    # A mismatched model cannot be used with a different allocation stride or RoPE capacity.
    def test_model_capacity_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "model.*capacity"):
            TtPrefillRuntime(
                "mesh",
                config=TtPrefillRuntimeConfig(max_seq_len=2048, chunk_size=1024, num_users=2),
                model=SimpleNamespace(num_layers=32, max_seq_len=4096),
                synchronize=lambda mesh: self.fail("unexpected synchronization"),
                upload=lambda *args, **kwargs: self.fail("unexpected upload"),
            )

    # Cache mismatch is a validation error and emits no writes or acknowledgments.
    def test_cache_capacity_mismatch_fails_before_dispatch(self):
        calls = []
        runtime = TtPrefillRuntime(
            "mesh",
            config=TtPrefillRuntimeConfig(max_seq_len=4096, chunk_size=1024, num_users=2),
            model=SimpleNamespace(
                num_layers=32, max_seq_len=4096, prefill_chunk=lambda *args, **kwargs: calls.append("forward")
            ),
            synchronize=lambda mesh: calls.append("sync"),
            upload=None,
        )
        runtime.compiled = True
        runtime.set_layer_completion_sink(lambda *args: calls.append("ack"))
        with self.assertRaises(ValueError):
            runtime.prefill_chunk(
                object(),
                SimpleNamespace(num_users=2, num_layers=32, max_seq_len=2048, sp=4),
                slot_id=1,
                actual_start=3072,
                actual_end=4096,
                request_id=0,
            )
        self.assertEqual(calls, [])
        self.assertFalse(runtime._failed)


if __name__ == "__main__":
    unittest.main()
