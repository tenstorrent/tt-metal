# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contract for reset_for_shape and its pieces; no TTNN import.

The production functions are lifted out of their modules by AST and run against a fake ttnn,
like test_gemma_encode_trace_capture. Run directly with Python or with pytest -p no:cacheprovider.
"""

from __future__ import annotations

import ast
import gc
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _lift(path, names, namespace, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    nodes = [n for n in scope if isinstance(n, (ast.FunctionDef, ast.Assign)) and _name(n) in names]
    assert {_name(n) for n in nodes} == set(names), f"{path}: missing {set(names) - {_name(n) for n in nodes}}"
    for n in nodes:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _name(node):
    return node.name if isinstance(node, ast.FunctionDef) else getattr(node.targets[0], "id", None)


class FakeTensor:
    def __init__(self, nbytes, addr):
        self.nbytes, self.addr = nbytes, addr

    def volume(self):
        return self.nbytes

    def element_size(self):
        return 1


class FakeTTNN:
    BufferType = SimpleNamespace(DRAM="dram", TRACE="trace")

    def __init__(self, views=None):
        self.log = []
        self.views = views or {}

    def deallocate(self, t):
        self.log.append(("free", t))

    def synchronize_device(self, _mesh):
        self.log.append(("sync",))

    def get_memory_view(self, _mesh, kind):
        return self.views[kind]


def _view(allocated, free, largest, size, banks=2):
    per = lambda x: x // banks  # noqa: E731
    return SimpleNamespace(
        num_banks=banks,
        total_bytes_allocated_per_bank=per(allocated),
        total_bytes_free_per_bank=per(free),
        largest_contiguous_bytes_free_per_bank=per(largest),
        total_bytes_per_bank=per(size),
    )


class CCLManagerRelease(unittest.TestCase):
    def manager(self, ttnn):
        ns = _lift(
            "parallel/manager.py",
            ["_cached_buffer_entries", "cached_buffer_bytes", "release_shape_buffers"],
            {"ttnn": ttnn},
            cls="CCLManager",
        )
        resets = []
        m = SimpleNamespace(mesh_device="mesh", reset_global_semaphores=lambda: resets.append(1))
        for k in ("_cached_buffer_entries", "cached_buffer_bytes", "release_shape_buffers"):
            setattr(m, k, ns[k].__get__(m))
        rs = [[FakeTensor(10, 1), FakeTensor(5, 2)], [FakeTensor(10, 3), None]]
        ag = [FakeTensor(7, 4), FakeTensor(7, 5)]
        m._ping_pong_buffer_cache = {((1, 2), 3, 0): rs, ("ag", (4,), 0, 1, "bf16"): ag}
        m._ping_pong_buffer_indices = {((1, 2), 3, 0): 1, ("ag", (4,), 0, 1, "bf16"): 0}
        m._fused_norm_stats_buffer_cache = {"k": {"bufs": [FakeTensor(3, 6), FakeTensor(3, 7)], "idx": 1}}
        return m, resets

    def test_frees_every_cached_tensor_once_and_resets_semaphores(self):
        ttnn = FakeTTNN()
        m, resets = self.manager(ttnn)
        self.assertEqual(m.cached_buffer_bytes(), 45)
        self.assertEqual(m.release_shape_buffers(), (7, 45))
        freed = [t.addr for op, *a in ttnn.log if op == "free" for t in a]
        self.assertEqual(sorted(freed), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(ttnn.log[0], ("sync",))  # nothing freed while a collective may still read it
        self.assertEqual(ttnn.log[-1], ("sync",))
        self.assertEqual(resets, [1])
        caches = (m._ping_pong_buffer_cache, m._ping_pong_buffer_indices, m._fused_norm_stats_buffer_cache)
        self.assertEqual(caches, ({}, {}, {}))

    def test_empty_caches_are_a_no_op(self):
        ttnn = FakeTTNN()
        m, resets = self.manager(ttnn)
        m._ping_pong_buffer_cache, m._fused_norm_stats_buffer_cache = {}, {}
        self.assertEqual(m.release_shape_buffers(), (0, 0))
        self.assertEqual((ttnn.log, resets), ([], []))


class FakeTracer:
    def __init__(self, captured):
        self.trace_captured, self.released = captured, 0

    def release_trace(self):
        self.released += 1
        self.trace_captured = False


class Ctx:
    pass


class ReleaseTracedFunctions(unittest.TestCase):
    def test_releases_keyed_and_unkeyed_tracers_of_one_context_only(self):
        wrappers = weakref.WeakSet()
        ns = _lift(
            "utils/tracing.py",
            ["release_traced_functions"],
            {"weakref": weakref, "_TRACED_WRAPPERS": wrappers, "Any": object},
        )
        a, b = Ctx(), Ctx()
        w1 = lambda: None  # noqa: E731
        w2 = lambda: None  # noqa: E731
        ta, tb, k1, k2 = FakeTracer(True), FakeTracer(True), FakeTracer(True), FakeTracer(False)
        w1._tracers = weakref.WeakKeyDictionary({a: ta, b: tb})
        w1._tracers_keyed = weakref.WeakKeyDictionary()
        w2._tracers = weakref.WeakKeyDictionary()
        w2._tracers_keyed = weakref.WeakKeyDictionary({a: {"s1": k1, "s2": k2}})
        wrappers.add(w1)
        wrappers.add(w2)
        self.assertEqual(ns["release_traced_functions"](a), 2)  # k2 was never captured
        self.assertEqual((ta.released, k1.released, k2.released, tb.released), (1, 1, 1, 0))
        self.assertNotIn(a, w1._tracers)
        self.assertNotIn(a, w2._tracers_keyed)
        self.assertIn(b, w1._tracers)
        self.assertEqual(ns["release_traced_functions"](3), 0)  # not weak-referenceable: never a key


class StateTensorRelease(unittest.TestCase):
    def test_frees_device_data_and_drops_host_data(self):
        dev, host = SimpleNamespace(kind="DEVICE"), SimpleNamespace(kind="HOST")
        for t in (dev, host):
            t.storage_type = lambda t=t: t.kind
        ttnn = FakeTTNN()
        ttnn.Tensor = SimpleNamespace
        ttnn.StorageType = SimpleNamespace(DEVICE="DEVICE")
        ns = _lift("utils/tracing.py", ["release"], {"ttnn": ttnn}, cls="StateTensor")
        for data in (dev, host, None):
            st = SimpleNamespace(_data=data)
            ns["release"](st)
            self.assertIsNone(st._data)
        self.assertEqual(ttnn.log, [("free", dev)])


class DramByCategory(unittest.TestCase):
    def test_counts_each_buffer_once_in_precedence_and_rest_is_other(self):
        w1, w2, c1, io1, st1 = (FakeTensor(n, a) for n, a in ((100, 1), (50, 2), (30, 3), (20, 4), (10, 5)))
        shared = FakeTensor(30, 3)  # the same buffer reached as tracer I/O: already counted as CCL
        ttnn = FakeTTNN({"dram": _view(400, 600, 500, 1000), "trace": _view(40, 60, 50, 100)})
        state = SimpleNamespace(data=st1)
        ns = _lift(
            "utils/dram_accounting.py",
            ["CATEGORIES", "_view", "dram_by_category", "format_row"],
            {
                "ttnn": ttnn,
                "gc": gc,
                "Any": object,
                "_device_dram_tensor": lambda t: isinstance(t, FakeTensor),
                "_bytes_addr": lambda t: (t.nbytes, t.addr),
                "_walk": lambda root, path, seen, found, via, depth: found.extend((path, t, True) for t in root),
                "live_tracers": lambda: ["tr"],
                "tracer_io_tensors": lambda tr: [io1, shared, "host"],
                "StateTensor": SimpleNamespace,
            },
        )
        mgr = SimpleNamespace(_cached_buffer_entries=lambda: [c1])
        mesh = SimpleNamespace(num_program_cache_entries=lambda: 12)
        row = ns["dram_by_category"](
            mesh, weight_roots=[[w1, w2], [w1]], ccl_managers=[mgr, mgr], state_tensors=[state]
        )
        self.assertEqual(
            {k: row[k] for k in ("weights", "ccl", "tracer_io", "other", "allocated", "largest_free")},
            {"weights": 150, "ccl": 30, "tracer_io": 30, "other": 190, "allocated": 400, "largest_free": 500},
        )
        self.assertEqual((row["trace_used"], row["trace_size"], row["trace_largest_free"]), (40, 100, 50))
        self.assertEqual(row["program_cache_entries"], 12)
        self.assertIn("programs=12", ns["format_row"]("warm", row))


class PipelineResetForShape(unittest.TestCase):
    def test_releases_everything_before_freeing_ccl_and_warms_at_the_new_shape(self):
        log = []
        names = ["_ccl_managers", "_shape_state_tensors", "_release_ccl", "_clear_program_caches", "release_shape_buffers",
                 "reset_for_shape"]
        ttnn = SimpleNamespace(synchronize_device=lambda m: log.append("sync"))
        ns = _lift(
            "pipelines/ltx/pipeline_ltx.py",
            names,
            {
                "ttnn": ttnn,
                "time": __import__("time"),
                "CCLManager": object,
                "StateTensor": FakeState,
                "logger": SimpleNamespace(info=lambda *_: None),
                "format_dram_row": lambda label, row: label,
                "release_traced_functions": lambda c: log.append(("traced", c)) or 1,
                "live_tracers": lambda: [FakeTracer(False)],
            },
            cls="LTXPipeline",
        )

        class P:
            pass

        for k in names:
            setattr(P, k, ns[k])
        p = P()
        mesh = SimpleNamespace(
            num_program_cache_entries=lambda: 7,
            disable_and_clear_program_cache=lambda: log.append("clear_pc"),
            enable_program_cache=lambda: log.append("enable_pc"),
        )
        mgr = SimpleNamespace(release_shape_buffers=lambda: log.append("ccl") or (3, 300), mesh_device=mesh)
        p.ccl_manager, p.vae_ccl_manager, p.audio_ccl_manager = mgr, SimpleNamespace(
            release_shape_buffers=lambda: log.append("vae_ccl") or (1, 10), mesh_device=mesh
        ), mgr
        p.mesh_device = mesh
        p._trace_state = {"s1": SimpleNamespace(_tt_x=FakeState(log, "x"), _euler_tail=None)}
        p._prompt_v, p._prompt_a = FakeState(log, "pv"), FakeState(log, "pa")
        p._t12_io = {"noise": FakeState(log, "noise"), "weights": None}
        p._cached_trans_mat = "tm"
        p.transformer_states = [SimpleNamespace(model="dit")]
        p.gemma_encoder_pair, p.vae_decoder, p.vae_encoder, p.upsampler = "gemma", "vaed", None, "ups"
        p._vae_decoder_nf = p._upsampler_nf = 145
        p.release_traces = lambda: log.append("release_traces")
        p._ensure_vae_decoder_frames = lambda n: log.append(("vae_nf", p._vae_decoder_nf, n))
        p._ensure_upsampler_frames = lambda n: log.append(("ups_nf", p._upsampler_nf, n))
        p.dram_by_category = lambda: log.append("dram") or {"allocated": 1}
        p.warmup_buffers = lambda **kw: log.append(("warmup", kw))

        stats = p.reset_for_shape(241, 768, 1344)

        self.assertEqual(log[0], "sync")
        self.assertEqual(sorted(log[1:5]), ["free noise", "free pa", "free pv", "free x"])
        self.assertEqual(log[5], "release_traces")
        self.assertEqual({c for e in log if isinstance(e, tuple) and e[0] == "traced" for c in [e[1]]},
                         {p, "dit", "gemma", "vaed", "ups"})
        self.assertIn(("vae_nf", None, 241), log)  # rebuilt even though only H/W may change
        self.assertIn(("ups_nf", None, 241), log)
        order = [e if isinstance(e, str) else e[0] for e in log]
        self.assertLess(max(order.index("traced"), order.index("vae_nf")), order.index("ccl"))
        self.assertLess(order.index("ccl"), order.index("clear_pc"))  # stale CCL addresses dropped
        self.assertLess(order.index("clear_pc"), order.index("enable_pc"))
        self.assertLess(order.index("enable_pc"), order.index("warmup"))
        self.assertEqual(log.count("clear_pc"), 1)  # one mesh shared by every manager: cleared once
        self.assertEqual(stats["program_cache_cleared"], 7)
        self.assertEqual(log.count("ccl"), 1)  # a manager shared by two roles is released once
        self.assertEqual(log[order.index("warmup")][1],
                         {"num_frames": 241, "height": 768, "width": 1344, "capture_all": True})
        self.assertEqual((p._init_num_frames, p._init_height, p._init_width), (241, 768, 1344))
        self.assertIsNone(p._t12_io)
        self.assertIsNone(p._cached_trans_mat)
        self.assertEqual((stats["ccl_freed"], stats["ccl_freed_bytes"], stats["traces_released"]), (4, 310, 5))
        self.assertEqual(stats["traces_left"], 0)
        self.assertIn("warmup_s", stats)
        self.assertEqual(stats["dram_warm"], {"allocated": 1})
        log.clear()
        self.assertEqual(p.release_shape_buffers(), 4)  # an int: ltx-server reads it as ccl_freed


class FakeState:
    def __init__(self, log=None, name=""):
        self.log, self.name = log, name

    def release(self):
        self.log.append(f"free {self.name}")


if __name__ == "__main__":
    unittest.main()
