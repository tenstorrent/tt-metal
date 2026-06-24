# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, near-verbatim Pipeline for the standalone pi0.5 streamed-denoise port (SC5 core).

Distillation of ``tt_symbiote.core.d2d_pipeline.Pipeline`` (343 L -> ~190 L). The streamed
driver reads ``stages`` / ``meshes`` / ``_hop_in`` and calls ``_track_transport`` +
``capture_loop`` / ``replay_loop`` / ``release_loop`` (incl. the ``drain=="stage0"``
single-sync branch -- the SC5 single-drain win). KEPT verbatim: the topo-sort __init__,
the class-level live-trace/transport registries, ``_distinct_meshes``, ``capture_loop`` /
``replay_loop`` / ``release_loop``, ``_eager`` (for the reference / non-streamed builders),
``_as_l1`` / ``_src1`` helpers, ``stages`` / ``meshes`` properties + ``_hop_in``. DROPPED:
the per-call TracedRun dispatch in ``call()`` (-> native eager-default flag; the streamed
driver never calls ``__call__``), ``_warmup`` / ``_emit_stage`` / ``_capture`` / ``_replay``
(single-input trace path, unused by the streamed driver). ZERO tt_symbiote imports.
"""
from __future__ import annotations

import ttnn

from ._module import StatelessTTNNModule
from ._trace import trace_enabled, trace_running

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_L1 = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)


def _device_ttnn(x):
    """Raw on-device ttnn.Tensor backing ``x``, or None if not a device tensor."""
    if isinstance(x, ttnn.Tensor):
        return x
    return getattr(x, "ttnn_tensor", None)


def _rewrap_like(template, recv):
    if isinstance(template, ttnn.Tensor):
        return recv
    try:
        return type(template)(recv)
    except Exception:
        return recv


def _as_l1(t):
    if t.memory_config().buffer_type != ttnn.BufferType.L1:
        return ttnn.to_memory_config(t, _L1)
    return t


@trace_enabled
class Pipeline(StatelessTTNNModule):
    """Chain of stages whose dependencies are encoded as D2DBridge edges."""

    _live_traces: list = []
    _live_transports: list = []

    @classmethod
    def _track_traces(cls, tids):
        cls._live_traces.extend(tids or ())

    @classmethod
    def _track_transport(cls, transport):
        if transport is not None and all(transport is not t for t in cls._live_transports):
            cls._live_transports.append(transport)

    @classmethod
    def _untrack_traces(cls, tids):
        drop = {id(e) for e in (tids or ())}
        cls._live_traces = [e for e in cls._live_traces if id(e) not in drop]

    @classmethod
    def release_all(cls):
        for m, tid in cls._live_traces:
            try:
                ttnn.release_trace(m, tid)
            except Exception:
                pass
        cls._live_traces = []
        seen = set()
        for t in cls._live_transports:
            if id(t) in seen:
                continue
            seen.add(id(t))
            try:
                t.close()
            except Exception:
                pass
        cls._live_transports = []

    def __init__(self, bridges, *, sync_on_return=False):
        super().__init__()
        bridges = list(bridges)
        if not bridges:
            raise ValueError("Pipeline needs at least one D2DBridge edge")

        mods = {}
        out_edge = {}
        indeg = {}
        for b in bridges:
            p, c = b.producer, b.consumer
            mods[id(p)] = p
            mods[id(c)] = c
            if id(p) in out_edge:
                raise ValueError("Pipeline v1 supports a LINEAR chain; a stage fans out to >1 edge.")
            out_edge[id(p)] = (c, b)
            indeg[id(c)] = indeg.get(id(c), 0) + 1
            indeg.setdefault(id(p), indeg.get(id(p), 0))
        if any(v > 1 for v in indeg.values()):
            raise ValueError("Pipeline v1 supports a LINEAR chain; a stage has >1 incoming edge.")

        sources = [m for mid, m in mods.items() if indeg.get(mid, 0) == 0]
        if len(sources) != 1:
            raise ValueError(f"Pipeline: expected exactly one source stage, found {len(sources)}.")

        order, hop_in = [sources[0]], [None]
        cur = sources[0]
        while id(cur) in out_edge:
            nxt, b = out_edge[id(cur)]
            order.append(nxt)
            hop_in.append(b)
            cur = nxt
        if len(order) != len(mods):
            raise ValueError("Pipeline: the bridges do not form a single connected chain.")

        self._stages = order
        self._hop_in = hop_in
        self._meshes = [getattr(s, "device", None) for s in order]
        for i, d in enumerate(self._meshes):
            if d is None:
                raise ValueError(f"Pipeline: stage[{i}] is not bound; set_device(stage, mesh) first.")
        self._sync_on_return = sync_on_return
        self._bypass_tensor_wrapping = True
        self._device = self._meshes[-1]
        self._tids = None
        for b in self._hop_in:
            if b is not None:
                Pipeline._track_transport(b.transport)

    @property
    def stages(self):
        return list(self._stages)

    @property
    def meshes(self):
        return list(self._meshes)

    def to_device(self, device):
        return self

    def set_device_state(self, device_state=None):
        return self

    def forward(self, *a, **k):
        raise RuntimeError("Pipeline overrides call(); forward() should never be invoked.")

    def call(self, x):
        # Native eager default (the streamed driver never calls __call__; reference builders
        # use _eager via this path).
        return self._eager(x)

    __call__ = call

    def _as_l1(self, t):
        return _as_l1(t)

    def _src1(self, out, where):
        src = _device_ttnn(out)
        if src is None:
            raise TypeError(f"Pipeline ({where}): a stage returned a non/multi-tensor output.")
        return src

    def _eager(self, x):
        out = self._stages[0](x)
        for i in range(1, len(self._stages)):
            b = self._hop_in[i]
            recv = b.transport.send(self._as_l1(self._src1(out, "eager")), b.mesh_b, tag=b.tag)
            out = self._stages[i](_rewrap_like(out, recv))
        if self._sync_on_return:
            ttnn.synchronize_device(self._meshes[-1])
        return out

    @staticmethod
    def _distinct_meshes(submeshes):
        seen, order = set(), []
        for m in submeshes:
            if id(m) not in seen:
                seen.add(id(m))
                order.append(m)
        return order

    def capture_loop(self, submeshes, body_fn, n_steps):
        meshes = self._distinct_meshes(submeshes)
        tids = []
        with trace_running():
            for m in meshes:
                tids.append((m, ttnn.begin_trace_capture(m, cq_id=0)))
            for i in range(n_steps):
                body_fn(i)
            for m, tid in tids:
                ttnn.end_trace_capture(m, tid)
        Pipeline._track_traces(tids)
        return tids

    @staticmethod
    def replay_loop(loop_tids, *, drain="all", drain_mesh=None):
        for m, tid in loop_tids:
            ttnn.execute_trace(m, tid, cq_id=0, blocking=False)
        if drain == "stage0":
            ttnn.synchronize_device(drain_mesh if drain_mesh is not None else loop_tids[0][0])
        else:
            seen = set()
            for m, _ in loop_tids:
                if id(m) not in seen:
                    seen.add(id(m))
                    ttnn.synchronize_device(m)

    @classmethod
    def release_loop(cls, loop_tids):
        for m, tid in loop_tids or ():
            try:
                ttnn.release_trace(m, tid)
            except Exception:
                pass
        cls._untrack_traces(loop_tids)

    def release_traces(self):
        if self._tids:
            for mesh, tid in self._tids:
                try:
                    ttnn.release_trace(mesh, tid)
                except Exception:
                    pass
            Pipeline._untrack_traces(self._tids)
        self._tids = None

    def close(self):
        for m in self._meshes:
            try:
                ttnn.synchronize_device(m)
            except Exception:
                pass
        self.release_traces()
        seen = set()
        for b in self._hop_in:
            if b is not None and id(b.transport) not in seen:
                seen.add(id(b.transport))
                try:
                    b.transport.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
