# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of actual cache-init definitions, with an ordered TTNN double."""

import gc
import importlib.util
import sys
import traceback
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "tt"


def load_module(name):
    """Import complete production modules with only their hardware dependencies mocked."""
    qualified_name = f"models.demos.qwen38_27b_qb2.tt.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


class Tensor:
    next_id = 0

    def __init__(self, shape, dtype, mesh):
        Tensor.next_id += 1
        self.ident = Tensor.next_id
        self.shape = self.padded_shape = tuple(shape)
        self.dtype, self.layout, self.mesh = dtype, "tile", mesh
        self.pending = self.freed = False

    def buffer_unique_id(self):
        return self.ident

    def memory_config(self):
        return "dram"

    def device(self):
        return self.mesh


class Runtime:
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "tile", "row", "dram"
    bfloat8_b, bfloat16, float32 = "bf8", "bf16", "fp32"
    Shape = tuple
    Arch = NS(BLACKHOLE="blackhole")

    def __init__(self):
        self.mesh = NS(get_num_devices=lambda: 4, arch=lambda: "blackhole", shape=(1, 4))
        self.cluster = NS(ClusterType=NS(P300_X2="qb2"), get_cluster_type=lambda: "qb2")
        self.events, self.allocated = [], []
        self.fail_sync, self.fail_free = set(), set()
        self.syncs = 0
        self.fail_fill = self.fail_allocate = self.fail_host = False

    def allocate_tensor_on_device(self, shape, dtype, layout, mesh, memory):
        self.events.append("allocate")
        if self.fail_allocate:
            raise RuntimeError("allocation failed")
        assert layout == "tile" and memory == "dram"
        value = Tensor(shape, dtype, mesh)
        self.allocated.append(value)
        return value

    def zeros_like(self, value, optional_tensor):
        assert value is optional_tensor
        self.events.append("fill")
        value.pending = True
        if self.fail_fill:
            raise RuntimeError("fill failed")
        return value

    def synchronize_device(self, mesh):
        assert mesh is self.mesh
        self.events.append("sync")
        self.syncs += 1
        if self.syncs in self.fail_sync:
            raise RuntimeError("drain failed")
        for value in self.allocated:
            value.pending = False

    def deallocate(self, value):
        self.events.append(("free", value.ident))
        assert not value.pending, "freed before drain"
        if value.ident in self.fail_free:
            raise RuntimeError("release failed")
        assert not value.freed, "double free"
        value.freed = True

    def zeros(self, shape, *, dtype, layout, device, memory_config):
        self.events.append(("host_zero", layout))
        if self.fail_host:
            raise RuntimeError("host zero failed")
        value = Tensor(shape, dtype, device)
        value.layout = layout
        return value


class TestCacheInitialization(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        with patch.dict("sys.modules", {"ttnn": self.runtime, "models.common.tt_ccl": NS(TT_CCL=object)}):
            self.cache_initialization = load_module("cache_initialization")
            load_module("decode_conv")
            self.decoder_module = load_module("decoder")
            self.tp_module = load_module("decoder_tp")
        self.owner = NS(device=self.runtime.mesh, _cache_initialization_pending=None)

    def allocate(self, shape=(8192, 1, 32, 256), dtype="bf8"):
        return self.cache_initialization.allocate_cache_zero(self.owner, shape, dtype=dtype)

    def test_only_proven_shapes_are_eligible(self):
        yes = [
            ((8192, 1, 32, 256), "bf8"),
            ((8192, 1, 32, 256), "bf16"),
            ((1, 12, 128, 128), "fp32"),
            ((2, 12, 128, 128), "fp32"),
        ]
        no = [((n, 1, 32, 256), "bf8") for n in (1, 17, 129, 4096, 8191, 8193, 16384)]
        no += [
            ((3, 12, 128, 128), "fp32"),
            ((1, 12, 128, 128), "bf16"),
            ((True, 12, 128, 128), "fp32"),
            ((8192, 2, 32, 256), "bf8"),
        ]
        for shape, dtype in yes + no:
            with self.subTest(shape=shape, dtype=dtype):
                self.assertEqual(
                    self.cache_initialization.device_zero_eligible(shape, dtype, "tile", self.owner.device),
                    (shape, dtype) in yes and all(type(n) is int for n in shape),
                )
        self.assertFalse(self.cache_initialization.device_zero_eligible(yes[0][0], "bf8", "row", self.owner.device))
        self.assertFalse(
            self.cache_initialization.device_zero_eligible(yes[0][0], "bf8", "tile", NS(get_num_devices=lambda: 1))
        )

    def test_unqualified_topologies_retain_host_initialization(self):
        for target, attribute, value in (
            (self.runtime.mesh, "get_num_devices", lambda: 1),
            (self.runtime.mesh, "arch", lambda: "wormhole"),
            (self.runtime.mesh, "shape", (2, 2)),
            (self.runtime.cluster, "get_cluster_type", lambda: "p150_x4"),
        ):
            with self.subTest(attribute=attribute), patch.object(target, attribute, value):
                self.assertFalse(
                    self.cache_initialization.device_zero_eligible((8192, 1, 32, 256), "bf8", "tile", self.runtime.mesh)
                )
                self.runtime.events.clear()
                self.decoder().allocate_state(batch_size=1, num_pages=8192)
                self.assertEqual(self.runtime.events, [("host_zero", "tile")] * 2)

    def test_success_finishes_before_handoff(self):
        for shape, dtype in [
            ((8192, 1, 32, 256), "bf8"),
            ((8192, 1, 32, 256), "bf16"),
            ((1, 12, 128, 128), "fp32"),
            ((2, 12, 128, 128), "fp32"),
        ]:
            with self.subTest(shape=shape, dtype=dtype):
                self.runtime.events.clear()
                value = self.allocate(shape, dtype)
                self.assertEqual(self.runtime.events, ["allocate", "fill", "sync"])
                self.assertFalse(value.pending or value.freed)

    def test_allocation_failure_has_no_fill_or_cleanup(self):
        self.runtime.fail_allocate = True
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self.allocate()
        self.assertEqual(self.runtime.events, ["allocate"])

    def test_fill_failure_drains_then_frees(self):
        self.runtime.fail_fill = True
        with self.assertRaisesRegex(RuntimeError, "fill failed"):
            self.allocate()
        value = self.runtime.allocated[0]
        self.assertEqual(self.runtime.events, ["allocate", "fill", "sync", ("free", value.ident)])
        self.assertTrue(value.freed)
        self.cache_initialization.require_cache_initialization_ready(self.owner)

    def test_first_fence_failure_is_drained_and_released(self):
        self.runtime.fail_sync = {1}
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            self.allocate()
        self.assertEqual(self.runtime.syncs, 2)
        self.assertTrue(self.runtime.allocated[0].freed)
        self.cache_initialization.require_cache_initialization_ready(self.owner)

    def test_failed_drain_stays_owned_after_exception_is_discarded(self):
        self.runtime.fail_sync = {1, 2}
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.allocate()
        reference = weakref.ref(self.runtime.allocated[0])
        self.runtime.allocated.clear()
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(reference().pending)
        self.assertFalse(reference().freed)
        before = list(self.runtime.events)
        with self.assertRaisesRegex(RuntimeError, "close the model mesh"):
            self.allocate()
        self.assertEqual(self.runtime.events, before)

    def test_failed_release_is_retained_and_reuse_is_blocked(self):
        self.runtime.fail_fill = True
        self.runtime.fail_free = {Tensor.next_id + 1}
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.allocate()
        pending = self.owner._cache_initialization_pending
        self.assertIs(pending, self.runtime.allocated[0])
        self.assertFalse(pending.pending or pending.freed)
        with self.assertRaisesRegex(RuntimeError, "close the model mesh"):
            self.cache_initialization.require_cache_initialization_ready(self.owner)

    def test_cleanup_failure_keeps_original_failure_in_traceback(self):
        self.runtime.fail_fill = True
        self.runtime.fail_sync = {1}
        with self.assertRaisesRegex(RuntimeError, "cleanup failed") as caught:
            self.allocate()
        error = caught.exception
        self.assertEqual(str(error.__cause__), "drain failed")
        self.assertEqual(str(error.__cause__.__context__), "fill failed")
        rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.assertIn("fill failed", rendered)
        self.assertIn("drain failed", rendered)

    def test_unproven_geometry_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "qualified TP4"):
            self.allocate((129, 1, 32, 256))
        self.assertFalse(self.runtime.events)

    def decoder(self, kind="full_attention", policy=None):
        value = self.decoder_module.Qwen38Decoder()
        value.device, value.kind = self.runtime.mesh, kind
        value.policy = {"kv_dtype": "bfloat8_b", **(policy or {})}
        value.config = NS(
            num_key_value_heads=1,
            head_dim=256,
            linear_num_key_heads=4,
            linear_key_head_dim=128,
            linear_num_value_heads=12,
            linear_value_head_dim=128,
        )
        return value

    def test_actual_decoder_default_and_optout(self):
        enabled = self.decoder()
        state = enabled.allocate_state(batch_size=1, num_pages=8192)
        self.assertFalse(state.key.pending or state.value.pending)
        self.assertEqual(self.runtime.events, ["allocate", "fill", "sync"] * 2)
        self.runtime.events.clear()
        disabled = self.decoder(policy={"device_zero_cache": False})
        state = disabled.allocate_state(batch_size=1, num_pages=8192)
        self.assertEqual(self.runtime.events, [("host_zero", "tile")] * 2)
        self.assertEqual(state.key.shape, (8192, 1, 32, 256))

    def test_actual_decoder_convolution_and_tiny_kv_keep_original_path(self):
        state = self.decoder("linear_attention").allocate_state(batch_size=1)
        self.assertEqual(state.conv.layout, "row")
        self.assertFalse(state.recurrent.pending)
        self.assertEqual(self.runtime.events, ["allocate", "fill", "sync", ("host_zero", "row")])
        self.runtime.events.clear()
        self.decoder().allocate_state(batch_size=1, num_pages=129)
        self.assertEqual(self.runtime.events, [("host_zero", "tile")] * 2)

    def test_invalid_policy_rejected_before_state_allocation(self):
        for value in (0, 1, None, "false"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.decoder(policy={"device_zero_cache": value}).allocate_state(batch_size=1, num_pages=8192)
        self.assertFalse(self.runtime.events)

    def test_pending_cleanup_cannot_be_bypassed_by_optout(self):
        owner = self.decoder(policy={"device_zero_cache": False})
        owner._cache_initialization_pending = object()
        with self.assertRaisesRegex(RuntimeError, "close the model mesh"):
            owner.allocate_state(batch_size=1, num_pages=8192)
        self.assertFalse(self.runtime.events)

    def test_later_convolution_failure_sees_completed_recurrent_buffer(self):
        self.runtime.fail_host = True
        with self.assertRaisesRegex(RuntimeError, "host zero failed"):
            self.decoder("linear_attention").allocate_state(batch_size=1)
        self.assertEqual(len(self.runtime.allocated), 1)
        self.assertFalse(self.runtime.allocated[0].pending)
        self.assertEqual(self.runtime.events[-2:], ["sync", ("host_zero", "row")])

    def test_actual_tp_setup_failure_follows_completed_base_state(self):
        base = self.decoder(policy={"fused_input": True, "ccl_dtype": "bfloat16"})
        owner = self.tp_module.Qwen38TPDecoder()
        owner.__dict__.update(base.__dict__)

        def fail_extra(*args, **kwargs):
            self.assertEqual(len(self.runtime.allocated), 2)
            self.assertTrue(all(not value.pending for value in self.runtime.allocated))
            raise RuntimeError("TP extra setup failed")

        self.runtime.empty = fail_extra
        with self.assertRaisesRegex(RuntimeError, "TP extra setup failed"):
            owner.allocate_state(batch_size=1, num_pages=8192)
