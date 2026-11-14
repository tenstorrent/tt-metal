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
        self.copies = []

    def synchronize_device(self, mesh):
        if self.capturing:
            raise RuntimeError("synchronize_device inside capture")
        self.syncs += 1

    def allocate_tensor_on_device(self, spec, *args):
        # `spec` is `tensor()`'s (*shape, dtype): the allocation has that shape, dtype and device, like ttnn's.
        out = SimpleNamespace(shape=spec[:-1], spec=spec, get_dtype=lambda: spec[-1], device=lambda: "mesh")
        self.allocated.append(out)
        return out

    def fill(self, x, *args, **kwargs):
        return x

    def deallocate(self, x):
        self.freed.append(x)

    def copy(self, src, dst):
        if self.capturing:
            raise RuntimeError("input copy compiled inside capture")
        self.copies.append((src, dst))


def manager(native):
    names = {
        "get_dim",
        "get_ag_ping_pong_buffer",
        "get_np_ping_pong_buffer",
        "transient_ping_pong_buffers",
        "audio_buffer_scope",
        "_check_audio_buffer",
        "_record_audio_buffer",
        "audio_buffers_current",
        "_audio_buffer_matches",
    }
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
    cls = type("Vocoder", (), methods("models/audio_vae/vocoder_ltx.py", "Vocoder", names, {"ttnn": native}))
    obj = cls()
    obj.ccl_manager = m
    obj.mesh_device = m.mesh_device
    obj.prepare_audio_ccl = True
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


def live_trace(v, key="z"):
    """Mark a captured trace at another key live on `v`: no eager warm may run then."""
    type(v)._forward_device._tracers_keyed[v] = {key: SimpleNamespace(trace_captured=True)}


def tensor(length=32, dtype="bf16"):
    return SimpleNamespace(
        shape=(1, length, 32), spec=(1, length, 32, dtype), get_dtype=lambda: dtype, device=lambda: "mesh"
    )


class PreparationTest(unittest.TestCase):
    def test_pipeline_retains_all_warmed_shapes_until_capture(self):
        native = Native()
        m = manager(native)
        names = {"warmup", "_warm_audio_decode", "_capture_audio", "_audio_ccl_transient"}
        ns = {
            "_skip_decode_warm": lambda: False,
            "_is_host_rank": lambda: False,
            "tqdm": SimpleNamespace(tqdm=lambda x, **kw: x),
            "sys": SimpleNamespace(stderr=None),
            "_TQDM_BAR_FORMAT": "",
        }
        cls = type(
            "Pipeline", (), methods("pipelines/minimax_h3/pipeline_minimax_h3.py", "MiniMaxH3Pipeline", names, ns)
        )
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
        p._prepare_audio_decoder = lambda: lambda x, traced=False: v.forward_device_BTC(
            x, t_pad=0, traced=traced, trace_key=x.shape
        )
        p._capture_traces = lambda prompt, fitted, overrides, trace_audio: p._capture_audio(trace_audio)
        p.warmup()
        # untraced warm per length, capture per length, then one replay per length
        self.assertEqual(v.calls, [False, False, True, True, True, True])
        self.assertTrue(p.trace_audio)
        self.assertFalse(getattr(p, "_warming_audio_trace", False))
        # the only frees are the eager warm's input-copy slots, one per length (no CCL pool is released)
        self.assertEqual(native.freed, [dst for _, dst in native.copies])
        self.assertEqual(len(native.copies), 2)
        syncs = native.syncs
        p._capture_audio(True)
        self.assertEqual(native.syncs, syncs)

        def fail_warm(*args):
            raise RuntimeError("warm failed")

        p._warm_denoise_buckets = fail_warm
        with self.assertRaisesRegex(RuntimeError, "warm failed"):
            p.warmup()
        self.assertTrue(p.trace_audio)
        self.assertFalse(p._warming_audio_trace)

    def test_skipped_decode_warm_keeps_the_audio_warm_a_capture_needs(self):
        # MINIMAX_H3_WARMUP_SKIP_DECODE_WARM=1 skips the VAE canvas warm; the audio length warm stays when audio
        # is traced, because each capture needs its length's eager warm and none may run once a trace is live.
        for trace_audio, expected in ((True, [False, False, True, True, True, True]), (False, [])):
            with self.subTest(trace_audio=trace_audio):
                native = Native()
                m = manager(native)
                names = {"warmup", "_warm_audio_decode", "_capture_audio", "_audio_ccl_transient"}
                ns = {
                    "_skip_decode_warm": lambda: True,
                    "_is_host_rank": lambda: False,
                    "tqdm": SimpleNamespace(tqdm=lambda x, **kw: x),
                    "sys": SimpleNamespace(stderr=None),
                    "_TQDM_BAR_FORMAT": "",
                }
                cls = type(
                    "Pipeline",
                    (),
                    methods("pipelines/minimax_h3/pipeline_minimax_h3.py", "MiniMaxH3Pipeline", names, ns),
                )
                p = cls()
                p.dynamic_length = False
                p.bucket_denoise = True
                p.trace_audio = trace_audio
                p.use_persistent_ccl_buffers = False
                p.audio_ccl_manager = m
                p.vae_output_type = "yuv420"
                p.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
                p._warm_denoise_buckets = lambda *args: {}
                p._warm_vae_decode = lambda: self.fail("the VAE canvas warm is skipped")
                p._warm_prompt_encoder = lambda: None
                p._host_log = lambda *args: None
                p._audio_warm_inputs = lambda: [tensor(32), tensor(64)]
                v = vocoder(m, native)
                live = type(v)._forward_device._tracers_keyed.setdefault(v, {})

                def decode(x, traced=False):
                    out = v.forward_device_BTC(x, t_pad=0, traced=traced, trace_key=x.shape)
                    if traced:
                        live[x.shape] = SimpleNamespace(trace_captured=True)  # what a real capture leaves
                    return out

                p._prepare_audio_decoder = lambda: decode
                p._capture_traces = lambda prompt, fitted, overrides, trace_audio: p._capture_audio(trace_audio)
                p.warmup()
                self.assertEqual(v.calls, expected)

    def test_untraced_nonpersistent_policy_still_releases_buffers(self):
        ns = methods("pipelines/minimax_h3/pipeline_minimax_h3.py", "MiniMaxH3Pipeline", {"_audio_ccl_transient"}, {})
        for trace, warming, persistent in [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]:
            native = Native()
            m = manager(native)
            p = SimpleNamespace(
                audio_ccl_manager=m,
                trace_audio=trace,
                _warming_audio_trace=warming,
                use_persistent_ccl_buffers=persistent,
            )
            with ns["_audio_ccl_transient"](p):
                np_buffer(m)
            self.assertEqual(bool(m._ping_pong_buffer_cache), trace or warming or persistent)

    def test_recorded_buffers_are_instance_and_shape_bound(self):
        native = Native()
        m = manager(native)
        a, b = vocoder(m, native), vocoder(m, native)
        x = tensor()
        a.forward_device_BTC(x, t_pad=0, trace_key="a")
        # With a trace live no eager warm can run, so a mismatch is rejected rather than re-warmed.
        live_trace(a)
        live_trace(b)
        for target, value, pad, key in [
            (b, x, 0, "a"),
            (a, tensor(64), 0, "a"),
            (a, tensor(dtype="fp32"), 0, "a"),
            (a, x, 1, "a"),
            (a, x, 0, "b"),
        ]:
            with self.subTest(target=target, shape=value.shape, pad=pad, key=key):
                with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
                    target.forward_device_BTC(value, t_pad=pad, traced=True, trace_key=key)
        a.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")

    def test_removed_replaced_or_foreign_manager_rejected_before_capture(self):
        for mutation in ("removed", "replaced", "element", "manager", "mesh"):
            with self.subTest(mutation=mutation):
                native = Native()
                m = manager(native)
                v = vocoder(m, native)
                x = tensor()
                v.forward_device_BTC(x, t_pad=0, trace_key="a")
                live_trace(v)
                if mutation == "mesh":
                    v.mesh_device = object()
                elif mutation == "manager":
                    v.ccl_manager = manager(native)
                else:
                    k = next(iter(m._ping_pong_buffer_cache))
                    if mutation == "removed":
                        m._ping_pong_buffer_cache.pop(k)
                    elif mutation == "element":
                        m._ping_pong_buffer_cache[k][0] = object()
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
        for error in (RuntimeError("collective failed"), KeyboardInterrupt(), SystemExit(37), GeneratorExit()):
            native = Native()
            m = manager(native)
            v = vocoder(m, native)
            x = tensor()
            v.fail = error
            with self.assertRaises(type(error)):
                v.forward_device_BTC(x, t_pad=0, trace_key="a")
            v.fail = None
            self.assertEqual(v._prepared_audio_ccl, {})
            self.assertIsNone(getattr(m, "_audio_buffer_record", None))
            # Nothing was prepared, so the traced call runs its own eager warm before the capture.
            v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
            self.assertEqual(v.calls, [False, False, True])

    def test_eager_after_capture_requires_release(self):
        native = Native()
        v = vocoder(manager(native), native)
        type(v)._forward_device._tracers_keyed[v] = {"a": SimpleNamespace(trace_captured=True)}
        with self.assertRaisesRegex(RuntimeError, "release"):
            v.forward_device_BTC(tensor(), t_pad=0, trace_key="a")
        self.assertEqual(v.calls, [])

    def test_buffer_scope_rejects_nesting_and_preserves_indices(self):
        native = Native()
        m = manager(native)
        with m.audio_buffer_scope() as used:
            np_buffer(m)
            m.get_ag_ping_pong_buffer((1, 32, 32), 1, 0)
            with self.assertRaisesRegex(RuntimeError, "nest"):
                with m.audio_buffer_scope():
                    pass
        indices = dict(m._ping_pong_buffer_indices)
        syncs = native.syncs
        with m.audio_buffer_scope(used):
            pass
        self.assertEqual(m._ping_pong_buffer_indices, indices)
        self.assertEqual(native.syncs, syncs)

    def test_release_invalidates_preparation(self):
        native = Native()
        v = vocoder(manager(native), native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        v.release_trace()
        self.assertEqual(v._prepared_audio_ccl, {})
        live_trace(v)
        with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
            v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(v.calls, [False])

    def test_traced_decode_after_release_warms_eagerly_before_capture(self):
        # A server guard calls the pipeline's release_traces() and then a traced decode (ltx-server
        # _h3_guard_audio_trace). The capture must not run over unprepared buffers: with no trace live the
        # traced call first runs the eager warm itself, on a copy of its input, then captures from the input.
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        v.release_trace()
        copies = len(native.copies)
        seen = []
        forward = type(v)._forward_device

        def record(self, x_in, *, traced=False, tracer_trace_key=None):
            seen.append((x_in, traced))
            return forward(self, x_in, traced=traced, tracer_trace_key=tracer_trace_key)

        record._tracers_keyed = forward._tracers_keyed
        type(v)._forward_device = record
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(v.calls, [False, True, False, True])
        (warm_in, warm_traced), (cap_in, cap_traced) = seen
        self.assertFalse(warm_traced)
        self.assertTrue(cap_traced)
        self.assertIsNot(warm_in, x)
        self.assertIs(cap_in, x)
        # the warm's input copy and its own replay-copy slot, both freed (Native raises on a sync or copy in capture)
        self.assertEqual([src for src, _ in native.copies[copies:]], [x, warm_in])
        self.assertIn(warm_in, native.freed)
        # prepared again: the next traced call replays without another warm
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(v.calls, [False, True, False, True, True])

    def test_new_shape_with_a_live_trace_is_still_rejected(self):
        native = Native()
        v = vocoder(manager(native), native)
        v.forward_device_BTC(tensor(32), t_pad=0, trace_key=(32,))
        live_trace(v, (32,))
        with self.assertRaisesRegex(RuntimeError, "prepar|warm"):
            v.forward_device_BTC(tensor(64), t_pad=0, traced=True, trace_key=(64,))
        self.assertEqual(v.calls, [False])

    def test_eager_warm_compiles_the_replay_input_copy(self):
        # A replay copies its input into the trace's slot (ttnn.copy); the eager warm builds that program first.
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        self.assertEqual(len(native.copies), 1)
        src, slot = native.copies[0]
        self.assertIs(src, x)
        self.assertEqual(slot.spec, x.spec)
        self.assertEqual(native.freed, [slot])
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(len(native.copies), 1)

    def test_stale_record_from_a_transient_untraced_decode_warms_and_captures(self):
        # ltx-server's warm gens decode untraced (trace_audio False) inside the pipeline's transient CCL scope,
        # which frees the buffers the eager pass recorded; the served traced decode then found a record naming
        # freed buffers and raised "audio CCL buffer ... is not prepared". The stale record is now dropped and
        # the traced call warms eagerly on a copy, then captures over the current buffers.
        ns = methods("pipelines/minimax_h3/pipeline_minimax_h3.py", "MiniMaxH3Pipeline", {"_audio_ccl_transient"}, {})
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        p = SimpleNamespace(
            audio_ccl_manager=m, trace_audio=False, _warming_audio_trace=False, use_persistent_ccl_buffers=False
        )
        x = tensor()
        with ns["_audio_ccl_transient"](p):
            v.forward_device_BTC(x, t_pad=0, traced=p.trace_audio, trace_key="a")
        self.assertEqual(m._ping_pong_buffer_cache, {})
        self.assertIn("a", {k[0] for k in v._prepared_audio_ccl})
        p.trace_audio = True
        with ns["_audio_ccl_transient"](p):
            v.forward_device_BTC(x, t_pad=0, traced=p.trace_audio, trace_key="a")
        self.assertEqual(v.calls, [False, False, True])
        # prepared again over the current buffers: the next traced call goes straight to the trace
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(v.calls, [False, False, True, True])

    def test_stale_record_with_its_own_trace_live_releases_and_recaptures(self):
        for mutation in ("removed", "replaced", "element"):
            with self.subTest(mutation=mutation):
                native = Native()
                m = manager(native)
                v = vocoder(m, native)
                x = tensor()
                v.forward_device_BTC(x, t_pad=0, trace_key="a")
                v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
                tracer = SimpleNamespace(trace_captured=True)
                tracer.release_trace = lambda t=tracer: setattr(t, "trace_captured", False)
                type(v)._forward_device._tracers_keyed[v] = {"a": tracer}
                v._tpad_mask_cache["mask"] = object()
                k = next(iter(m._ping_pong_buffer_cache))
                if mutation == "removed":
                    m._ping_pong_buffer_cache.pop(k)
                elif mutation == "element":
                    m._ping_pong_buffer_cache[k][0] = object()
                else:
                    m._ping_pong_buffer_cache[k] = [object(), object()]
                v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
                self.assertFalse(tracer.trace_captured)
                self.assertEqual(v._tpad_mask_cache, {})
                self.assertEqual(v.calls, [False, True, False, True])

    def test_valid_record_goes_straight_to_capture(self):
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        x = tensor()
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        prepared = dict(v._prepared_audio_ccl)
        copies, syncs = len(native.copies), native.syncs
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        self.assertEqual(v.calls, [False, True])
        self.assertEqual((len(native.copies), native.syncs), (copies, syncs))
        self.assertEqual(v._prepared_audio_ccl, prepared)


class LtxUnchangedTest(unittest.TestCase):
    """LTX's vocoder and BWE share these helpers; only the H3 decoder opts into prepared persistent buffers."""

    def test_ltx_vocoder_decodes_without_scope_or_preparation(self):
        native = Native()
        m = manager(native)
        v = vocoder(m, native)
        v.prepare_audio_ccl = False
        scoped = []

        def forward(self, x, *, traced=False, tracer_trace_key=None):
            scoped.append((traced, getattr(m, "_audio_buffer_record", None) is not None))
            return x

        type(v)._forward_device = forward
        x = tensor()
        # A traced call at a never-warmed shape still captures lazily, as on ltx-rt before this change.
        v.forward_device_BTC(x, t_pad=0, traced=True, trace_key="a")
        v.forward_device_BTC(x, t_pad=0, trace_key="a")
        self.assertEqual(scoped, [(True, False), (False, False)])
        self.assertEqual(native.copies, [])
        self.assertEqual(v._prepared_audio_ccl, {})
        self.assertIsNone(getattr(m, "_audio_buffer_record", None))

    def test_audio_gathers_persistent_only_inside_h3_scope(self):
        class TConfig:
            def __init__(self, a0, a1):
                self.axis0, self.axis1 = SimpleNamespace(mesh_axis=a0), SimpleNamespace(mesh_axis=a1)

        tree = ast.parse((ROOT / "layers/audio_ops.py").read_text())
        names = {"_audio_persistent", "_all_gather_t", "all_gather_channel"}
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in selected}, names)
        ns = {"ttnn": None, "AudioTParallelConfig": TConfig, "channel_axis": lambda pc: 1}
        exec(compile("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), "ops", "exec"), ns)
        m = manager(Native())
        seen = []
        m.all_gather = lambda x, **kw: seen.append(kw["use_persistent_buffer"]) or x

        def gather_all():
            ns["_all_gather_t"](m, "x", TConfig(0, 1))
            ns["_all_gather_t"](m, "x", SimpleNamespace(mesh_axis=1))
            ns["all_gather_channel"](m, "x", None, dim=2)

        gather_all()
        self.assertEqual(seen, [False] * 4)
        seen.clear()
        with m.audio_buffer_scope():
            gather_all()
        self.assertEqual(seen, [True] * 4)
        seen.clear()
        gather_all()
        self.assertEqual(seen, [False] * 4)


if __name__ == "__main__":
    unittest.main()
