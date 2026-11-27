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
    dtype = "DataType.UINT8"

    def __init__(self, nbytes, addr):
        self.nbytes, self.addr = nbytes, addr

    def volume(self):
        return self.nbytes

    def element_size(self):
        return 1


class FakeBfpTensor:
    """A block-float tensor: element_size() raises as ttnn's does (tt_backend_api_types.hpp datum_size)."""

    def __init__(self, dtype, volume, addr):
        self.dtype, self.n, self.addr = f"DataType.{dtype}", volume, addr

    def volume(self):
        return self.n

    def element_size(self):
        raise ValueError("datum for bfp2, bfp4, bfp8 is invalid")


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


class FakeSemTTNN:
    def __init__(self):
        self.zeroed = []

    def reset_global_semaphore_value(self, sem, value):
        self.zeroed.append((sem, value))


class CCLManagerSemaphoreReset(unittest.TestCase):
    def test_all_pools_zeroes_every_pool_and_default_keeps_the_old_set(self):
        ttnn = FakeSemTTNN()
        ns = _lift("parallel/manager.py", ["reset_global_semaphores"], {"ttnn": ttnn}, cls="CCLManager")
        pool = lambda tag: {a: [f"{tag}{a}{i}" for i in range(2)] for a in (0, 1)}  # noqa: E731
        m = SimpleNamespace(
            **{
                f"{k}_semaphores": pool(k)
                for k in ("np_ping_pong", "sr_ping_pong", "rs_ping_pong", "ag_ping_pong", "exp_ring_ping_pong", "barrier")
            },
            rs_ping_pong_semaphores_fused=pool("rsf"),
            np_fused_ping_pong_semaphores=None,
            barrier_fused_semaphores=pool("bf"),
            np_region_progress_semaphores=None,
            _strided_ag_mm_sem_cache={(0, 2): [["s0a", "s0b"], ["s1a"]]},
        )
        reset = ns["reset_global_semaphores"].__get__(m)
        reset()
        base = {s for s, _ in ttnn.zeroed}
        self.assertFalse(any(s.startswith(("exp", "barrier", "rsf", "s0", "s1")) for s in base))
        ttnn.zeroed.clear()
        reset(all_pools=True)
        every = {s for s, v in ttnn.zeroed if v == 0}
        expect = {s for p in vars(m).values() if isinstance(p, dict) and 0 in p for a in (0, 1) for s in p[a]}
        expect |= {"s0a", "s0b", "s1a"}
        self.assertEqual(every, expect)


_METHODS = (
    "_cached_buffer_entries",
    "cached_buffer_bytes",
    "release_shape_buffers",
    "_reset_ping_pong_indices",
    "ping_pong_buffer_report",
)


def _manager_namespace(ttnn):
    ns = _lift("parallel/manager.py", ["_BFP_MANTISSA_BITS", "_tensor_nbytes"], {"ttnn": ttnn})
    return _lift("parallel/manager.py", _METHODS, ns, cls="CCLManager")


class CCLManagerRelease(unittest.TestCase):
    def manager(self, ttnn):
        ns = _manager_namespace(ttnn)
        resets = []
        m = SimpleNamespace(mesh_device="mesh", reset_global_semaphores=lambda **kw: resets.append(kw))
        for k in _METHODS:
            setattr(m, k, ns[k].__get__(m))
        rs = [[FakeTensor(10, 1), FakeTensor(5, 2)], [FakeTensor(10, 3), None]]
        ag = [FakeTensor(7, 4), FakeTensor(7, 5)]
        m._ping_pong_buffer_cache = {((1, 2), 3, 0): rs, ("ag", (4,), 0, 1, "bf16"): ag}
        m._ping_pong_buffer_indices = {((1, 2), 3, 0): 1, ("ag", (4,), 0, 1, "bf16"): 0}
        m._fused_norm_stats_buffer_cache = {"k": {"bufs": [FakeTensor(3, 6), FakeTensor(3, 7)], "idx": 1}}
        for name in self.IDX:
            setattr(m, name, [1, 1])
        m._strided_ag_mm_sem_idx = {(0, 2): 1, (1, 2): 1}
        return m, resets

    IDX = (
        "rs_ping_pong_idx",
        "rs_ping_pong_idx_fused",
        "ag_ping_pong_idx",
        "exp_ring_ping_pong_idx",
        "np_ping_pong_idx",
        "np_fused_ping_pong_idx",
        "sr_ping_pong_idx",
        "barrier_idx",
        "barrier_fused_idx",
    )

    def test_frees_every_cached_tensor_once_and_resets_semaphores(self):
        ttnn = FakeTTNN()
        m, resets = self.manager(ttnn)
        self.assertEqual(m.cached_buffer_bytes(), 45)
        self.assertEqual(m.release_shape_buffers(), (7, 45))
        freed = [t.addr for op, *a in ttnn.log if op == "free" for t in a]
        self.assertEqual(sorted(freed), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(ttnn.log[0], ("sync",))  # nothing freed while a collective may still read it
        self.assertEqual(ttnn.log[-1], ("sync",))
        self.assertEqual(resets, [{"all_pools": True}])
        # Host indices back where a new manager starts them, matching the zeroed semaphores.
        self.assertEqual({n: getattr(m, n) for n in self.IDX}, {n: [0, 0] for n in self.IDX})
        self.assertEqual(m._strided_ag_mm_sem_idx, {(0, 2): 0, (1, 2): 0})
        caches = (m._ping_pong_buffer_cache, m._ping_pong_buffer_indices, m._fused_norm_stats_buffer_cache)
        self.assertEqual(caches, ({}, {}, {}))

    def test_empty_caches_are_a_no_op(self):
        ttnn = FakeTTNN()
        m, resets = self.manager(ttnn)
        m._ping_pong_buffer_cache, m._fused_norm_stats_buffer_cache = {}, {}
        self.assertEqual(m.release_shape_buffers(), (0, 0))
        self.assertEqual((ttnn.log, resets), ([], []))

    def test_block_float_buffers_count_their_tile_bytes(self):
        # blx04 broker job 411 (11-15): release_shape_buffers raised on a bfp8 buffer, so reset_for_shape failed.
        nbytes = _manager_namespace(FakeTTNN())["_tensor_nbytes"]
        tile = 32 * 32
        self.assertEqual(
            [nbytes(FakeBfpTensor(d, tile, 0)) for d in ("BFLOAT8_B", "BFLOAT4_B", "BFLOAT2_B")], [1088, 576, 320]
        )
        self.assertEqual(nbytes(FakeBfpTensor("BFLOAT8_B", 3 * tile, 0)), 3 * 1088)
        self.assertEqual(nbytes(FakeTensor(5, 0)), 5)

    def test_releases_and_reports_block_float_buffers(self):
        ttnn = FakeTTNN()
        m, resets = self.manager(ttnn)
        tile = 32 * 32
        m._ping_pong_buffer_cache[("ag", (8,), 0, 1, "bfp8")] = [
            FakeBfpTensor("BFLOAT8_B", 2 * tile, 8),
            FakeBfpTensor("BFLOAT8_B", 2 * tile, 9),
        ]
        m._ping_pong_buffer_cache[((2, 4), 3, 0)] = [[FakeBfpTensor("BFLOAT4_B", tile, 10), None]]
        m._fused_norm_stats_buffer_cache["b"] = {"bufs": [FakeBfpTensor("BFLOAT2_B", tile, 11)], "idx": 0}
        total = 45 + 4 * 1088 + 576 + 320
        report = m.ping_pong_buffer_report()
        self.assertEqual(report[0], (4 * 1088, "ag x2 (8,)"))
        self.assertIn((576, "rs x1 (2, 4)"), report)
        self.assertEqual(m.cached_buffer_bytes(), total)
        self.assertEqual(m.release_shape_buffers(), (11, total))
        self.assertEqual(sorted(t.addr for op, *a in ttnn.log if op == "free" for t in a), list(range(1, 12)))
        self.assertEqual(resets, [{"all_pools": True}])


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


class DramProbeSkipsFreedTensors(unittest.TestCase):
    """A freed tensor is never read: its buffer address and pages are gone (blx06 job 367, SIGSEGV in warm_shapes)."""

    def test_a_deallocated_tensor_is_not_dram_and_reads_no_buffer(self):
        reads = []

        class Tensor:
            def __init__(self, live):
                self.live = live

            def is_allocated(self):
                return self.live

            def device(self):
                return "dev"

            def memory_config(self):
                return SimpleNamespace(buffer_type="dram")

            def buffer_aligned_page_size(self):
                reads.append(self)
                return 32

            def buffer_num_pages(self):
                return 4

            def buffer_address(self):
                return 7

        ttnn = SimpleNamespace(Tensor=Tensor, BufferType=SimpleNamespace(DRAM="dram"), get_device_tensors=lambda t: [t])
        ns = _lift("utils/dram_probe.py", ["_shard0", "_device_dram_tensor", "_bytes_addr"], {"ttnn": ttnn, "Any": object})
        live, freed = Tensor(True), Tensor(False)
        self.assertTrue(ns["_device_dram_tensor"](live))
        self.assertFalse(ns["_device_dram_tensor"](freed))
        self.assertEqual(ns["_bytes_addr"](live), (128, 7))
        self.assertEqual(ns["_bytes_addr"](freed), (0, None))
        self.assertEqual(reads, [live])


class PipelineResetForShape(unittest.TestCase):
    def test_releases_everything_before_freeing_ccl_and_warms_at_the_new_shape(self):
        log = []
        names = ["_ccl_managers", "_shape_state_tensors", "_release_ccl", "release_shape_buffers",
                 "_drop_shape_slots", "reset_for_shape"]
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
            _shape_slots = None
            _shape_suffix = ""

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
        gemma = type("Gemma", (), {"_trace_captured": True, "_trace_gate_open": True})()  # hashable
        gemma.forget_trace = lambda: setattr(gemma, "_trace_captured", False)
        gemma.defer_trace_capture = lambda: setattr(gemma, "_trace_gate_open", False)
        p._traced, p.dynamic_load, p.DEFERS_ENCODE_TRACE = True, False, True
        p.gemma_encoder_pair, p.vae_decoder, p.vae_encoder, p.upsampler = gemma, "vaed", None, "ups"
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
                         {p, "dit", gemma, "vaed", "ups"})
        # The warmup's early encode runs untraced and capture_trace() captures last, as on a fresh load.
        self.assertEqual((gemma._trace_captured, gemma._trace_gate_open), (False, False))
        self.assertIn(("vae_nf", None, 241), log)  # rebuilt even though only H/W may change
        self.assertIn(("ups_nf", None, 241), log)
        order = [e if isinstance(e, str) else e[0] for e in log]
        self.assertLess(max(order.index("traced"), order.index("vae_nf")), order.index("ccl"))
        self.assertNotIn("clear_pc", log)  # the program cache survives the swap
        self.assertNotIn("enable_pc", log)
        self.assertNotIn("program_cache_cleared", stats)
        self.assertLess(order.index("ccl"), order.index("warmup"))
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


class ShapeSlots(unittest.TestCase):
    """select_shape swaps the frame-baked modules, stage-transition I/O and trace-key suffix by pointer;
    _drop_shape_slots frees every parked slot and ends slot mode."""

    def _pipeline(self, log):
        names = ["_SHAPE_SLOT_ATTRS", "warm_shape", "resident_shapes", "_shape_key_suffix", "select_shape",
                 "_open_shape_slot", "_drop_shape_slots"]
        ns = _lift(
            "pipelines/ltx/pipeline_ltx.py",
            names,
            {
                "StateTensor": FakeState,
                "logger": SimpleNamespace(info=lambda *_: None),
                "release_traced_functions": lambda m: log.append(("traced", m.name)) or 1,
            },
            cls="LTXPipeline",
        )

        class P:
            _shape_slots = None
            _shape_suffix = ""

        for k in names:
            setattr(P, k, staticmethod(ns[k]) if k == "_shape_key_suffix" else ns[k])
        p = P()
        p._init_num_frames, p._init_height, p._init_width = 145, 704, 1280
        p.upsampler, p.vae_decoder = FakeModule(log, "ups145"), FakeModule(log, "vae145")
        p._upsampler_nf = p._vae_decoder_nf = 145
        p._t12_io = {"noise": FakeState(log, "noise145")}
        p._new_upsampler = lambda nf: FakeModule(log, f"ups{nf}")
        p._new_vae_decoder = lambda nf: FakeModule(log, f"vae{nf}")
        p._register_coresident_exclusions = lambda: log.append("excl")
        return p

    def test_open_select_and_drop(self):
        log = []
        p = self._pipeline(log)
        p._shape_slots, p._shape_suffix = {}, "@145x704x1280"
        p._open_shape_slot(241, 704, 1280)
        self.assertEqual(p.warm_shape(), (241, 704, 1280))
        self.assertEqual(p._shape_suffix, "@241x704x1280")
        self.assertEqual((p.upsampler.name, p.vae_decoder.name, p._t12_io), ("ups241", "vae241", None))
        self.assertEqual(p.resident_shapes(), [(241, 704, 1280), (145, 704, 1280)])
        self.assertNotIn("dealloc ups145", log)  # the parked shape keeps its weights

        self.assertTrue(p.select_shape(145, 704, 1280))
        self.assertEqual((p.upsampler.name, p._shape_suffix, p._t12_io["noise"].name),
                         ("ups145", "@145x704x1280", "noise145"))
        self.assertTrue(p.select_shape(145, 704, 1280))  # already active: no swap
        self.assertFalse(p.select_shape(121, 704, 1280))
        self.assertTrue(p.select_shape(241, 704, 1280))
        self.assertEqual(p.vae_decoder.name, "vae241")

        log.clear()
        self.assertEqual(p._drop_shape_slots(), 2)
        self.assertEqual(sorted(map(str, log)), sorted(map(str, ["free noise145", ("traced", "ups145"), ("traced", "vae145"),
                                              "dealloc ups145", "dealloc vae145"])))
        self.assertIsNone(p._shape_slots)
        self.assertEqual((p._shape_suffix, p.resident_shapes(), p.upsampler.name), ("", [], "ups241"))


class FakeModule:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def is_loaded(self):
        return True

    def deallocate_weights(self):
        self.log.append(f"dealloc {self.name}")


class FakeState:
    def __init__(self, log=None, name=""):
        self.log, self.name = log, name

    def release(self):
        self.log.append(f"free {self.name}")


if __name__ == "__main__":
    unittest.main()
