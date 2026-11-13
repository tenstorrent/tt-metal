# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU execution of production preparation contracts; no device/numerical acceptance."""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def methods(path, cls, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"contextmanager": contextmanager, "nullcontext": nullcontext, **namespace}
    exec(compile("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), path, "exec"), ns)
    return {n.name: ns[n.name] for n in selected}


class Native:
    bfloat16 = "bf16"
    TILE_LAYOUT = "tile"
    ROW_MAJOR_LAYOUT = "row"
    DRAM_MEMORY_CONFIG = "dram"
    Shape = tuple

    def __init__(self):
        self.capturing = False
        self.syncs = 0
        self.allocated = []
        self.freed = []

    def synchronize_device(self, mesh):
        if self.capturing:
            raise RuntimeError("synchronize_device inside capture")
        self.syncs += 1

    def allocate_tensor_on_device(self, shape, *args):
        out = SimpleNamespace(shape=shape)
        self.allocated.append(out)
        return out

    def fill(self, x, *args, **kwargs):
        return x

    def deallocate(self, x):
        self.freed.append(x)


def manager(native):
    names = {"get_dim", "get_ag_ping_pong_buffer", "get_np_ping_pong_buffer", "transient_ping_pong_buffers",
             "audio_buffer_scope", "_check_audio_buffer", "_record_audio_buffer"}
    cls = type("Manager", (), methods("parallel/manager.py", "CCLManager", names, {"ttnn": native}))
    obj = cls()
    obj.mesh_device = SimpleNamespace(shape=(4, 8))
    obj._ping_pong_buffer_cache = {}
    obj._ping_pong_buffer_indices = {}
    return obj


def np_buffer(m, length=32):
    return m.get_np_ping_pong_buffer((1, length, 32), [1], [1], [1])


def vocoder(m, native):
    names = {"forward_device_BTC", "release_trace"}
    cls = type("Vocoder", (), methods("models/audio_vae/vocoder_ltx.py", "Vocoder", names, {}))
    obj = cls()
    obj.ccl_manager = m
    obj._prepared_audio_ccl = {}
    obj._tpad_mask_cache = {}
    obj.calls = []
    obj.fail = None
    obj.extra = False

    def forward(self, x, *, traced=False, tracer_trace_key=None):
        self.calls.append(traced)
        native.capturing = traced
        try:
            np_buffer(m, x.shape[1])
            m.get_ag_ping_pong_buffer(x.shape, 1, 1)
            if self.extra:
                np_buffer(m, 999)
            if self.fail:
                raise self.fail
            return x
        finally:
            native.capturing = False
    forward._tracers_keyed = {}
    cls._forward_device = forward
    obj._device_to_host = lambda x: x
    return obj


def tensor(length=32, dtype="bf16"):
    return SimpleNamespace(shape=(1, length, 32), get_dtype=lambda: dtype)


class PreparationTest(unittest.TestCase):
    def test_pipeline_retains_all_warmed_shapes_until_capture(self):
        native = Native()
        m = manager(native)
        names = {"warmup", "_warm_audio_decode", "_capture_audio", "_audio_ccl_transient"}
        ns = {"_skip_decode_warm": lambda: False, "_is_host_rank": lambda: False,
              "tqdm": SimpleNamespace(tqdm=lambda x, **kw: x), "sys": SimpleNamespace(stderr=None),
              "_TQDM_BAR_FORMAT": ""}
        cls = type("Pipeline", (), methods("pipelines/minimax_h3/pipeline_minimax_h3.py", "MiniMaxH3Pipeline", names, ns))
        p = cls()
        p.dynamic_length = False
        p.bucket_denoise = True
        p.trace_audio = True
        p.use_persistent_ccl_buffers = False
        p.audio_ccl_manager = m
        p.vae_output_type = "float"
        p.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
        p._warm_denoise_buckets = lambda *args: {}
        p._warm_prompt_encoder = lambda: None
        p._host_log = lambda *args: None
        p._audio_warm_inputs = lambda: [tensor(32), tensor(64)]
        v = vocoder(m, native)
        p._prepare_audio_decoder = lambda: lambda x, traced=False: v.forward_device_BTC(x, t_pad=0, traced=traced, trace_key=x.shape)
        p._capture_traces = lambda prompt, fitted, overrides, trace_audio: p._capture_audio(trace_audio)
        p.warmup()
        self.assertEqual(v.calls, [False, False, True, True])
        self.assertTrue(p.trace_audio)
        self.assertFalse(getattr(p, "_warming_audio_trace", False))
        self.assertEqual(native.freed, [])
        syncs = native.syncs
        p._capture_audio(True)
        self.assertEqual(native.syncs, syncs)

    def test_recorded_buffers_are_instance_and_shape_bound(self):
        native = Native()
        m = manager(native)
        a, b = vocoder(m, native), vocoder(m, native)
        x = tensor()
        a.forward_device_BTC(x, t_pad=0, trace_key="a")
        for target, value, pad, key in [(b, x, 0, "a"), (a, tensor(64), 0, "a"),
                                         (a, tensor(dtype="fp32"), 0, "a"), (a, x, 1, "a"), (a, x, 0, "b")]:
            with self.subTest(target=target, shape=value.shape, pad=pad, key=key):
                with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
                    target.forward_device_BTC(value, t_pad=pad, traced=True, trace_key=key)
        a.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")

    def test_removed_replaced_or_foreign_manager_rejected_before_capture(self):
        for mutation in ("removed", "replaced", "manager"):
            with self.subTest(mutation=mutation):
                native = Native()
                m = manager(native)
                v = vocoder(m, native)
                x = tensor()
                v.forward_device_BTC(x, t_pad=0, trace_key="a")
                if mutation == "manager":
                    v.ccl_manager = manager(native)
                else:
                    k = next(iter(m._ping_pong_buffer_cache))
                    if mutation == "removed":
                        m._ping_pong_buffer_cache.pop(k)
                    else:
                        m._ping_pong_buffer_cache[k] = [object(), object()]
                syncs = native.syncs
                with self.assertRaisesRegex(RuntimeError, "prepar|warm|buffer"):
                    v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
                self.assertEqual(v.calls, [False])
                self.assertEqual(native.syncs, syncs)

    def test_unwarmed_collective_rejected_without_sync_and_scope_restored(self):
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        v.extra = True
        syncs = native.syncs
        with self.assertRaisesRegex(RuntimeError, "prepar|warm|buffer"):
            v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(native.syncs, syncs)
        self.assertFalse(getattr(m, "_audio_buffer_capture", False))
        v.extra = False
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")

    def test_failed_or_interrupted_warm_does_not_prepare(self):
        for error in (RuntimeError("collective failed"), KeyboardInterrupt()):
            native = Native()
            m = manager(native)
            v = vocoder(m, native)
            x = tensor()
            v.fail = error
            with self.assertRaises(type(error)):
                v.forward_device_BTC(x, t_pad=0, trace_key="a")
            v.fail = None
            with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
                v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
            self.assertIsNone(getattr(m, "_audio_buffer_record", None))
            v.forward_device_BTC(x, t_pad=0, trace_key="a")
            v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")

    def test_release_invalidates_preparation(self):
        native = Native()
        v = vocoder(manager(native), native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        v.release_trace()
        with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
            v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")


if __name__ == "__main__":
    unittest.main()
