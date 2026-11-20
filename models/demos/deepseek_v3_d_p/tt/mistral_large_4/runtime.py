# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill runner's runtime for mistral_large_4: the all-device model (model.py) over the served layers, on
the engine-owned caches (kv_table.py).

Input: the H2D stream's uint32 ROW_MAJOR [1, 1, chunk / SP] per chip (SP row r holds the chunk's rows in the engine's
ring_sdpa_reshuffle order; replicated over TP), PAD_ID past actual_end; TtModel.embed_device clamps it and embeds it.

Acks: one per served layer per chunk, in layer order, after that layer's cache writes (config 0 and 1, pad rows
zeroed): device-to-host (PREFILL_LAYER_ACK_D2H=1: a socket-service record on the same queue after the layer's ops),
else on the host after a device synchronize, through the completion sink.

Traced (PREFILL_USE_TRACE=1, params.use_trace; PREFILL_USE_TRACE=0 runs eagerly): the forward of every served layer
runs on the device-held chunk values (chunk_meta.py), captured ONCE as a ttnn trace and replayed per chunk after the
chunk's ids, record and metadata are copied into persistent buffers (as the GLM-5.3 TtPrefillRuntime). The capture runs
at startup, before the runner reports ready (startup_capture: a warm-up record through the H2D service first compiles
the socket program, which must happen before a capture); without a recorded service it falls back to the runner's
capture on the first record. Device-to-host acks are recorded inside the capture (one per layer, reading a persistent
copy of the chunk's metadata_msg); host-sink acks fire after the replay.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import sub_device

# the launcher records the H2D service the runner builds here, so the traced runtime can take a warm-up record through
# it before the request loop (startup_capture)
H2D_SERVICES = []
WARMUP_WORD = -2  # prefill_runner's WARMUP_METADATA_WORD: a record the runner's loop drops
# a client of the runner's H2D service (as prefill_producer) that pushes one warm-up record
_PUSH_WARMUP = """
import struct, sys
import numpy as np
import ttnn
service = ttnn.H2DStreamService.connect(sys.argv[1], timeout_ms=600000)
n = service.payload_size_bytes()
service.forward_to_tensor_bytes(np.zeros(n // 4, dtype=np.uint32), metadata=struct.pack("<3i", *([int(sys.argv[2])] * 3)))
service.barrier()
"""


class MistralLarge4PrefillRuntime:
    """model: model.py TtModel over geo.layers, built on the engine's caches; config: the runner's
    PrefillRunParams; geo: kv_table.Geometry."""

    def __init__(self, mesh_device, model, config, geo: KT.Geometry):
        self.mesh_device, self.model, self.config, self.geo = mesh_device, model, config, geo
        self.first = int(config.first_layer_idx)
        self.state = None
        self.sink = None
        self.d2h_service = None
        # traced serving (module docstring): persistent ids / record buffers, the capture and its output
        self.traced = bool(getattr(config, "use_trace", False))
        self._ids = self._md = self._out = None
        self._tid = None
        self._acks_warm = False  # the D2H ack programs compiled (set_d2h_ack_service's warm pass)
        self._kv = None  # the engine cache compile() bound (startup_capture records the trace on it)
        self.startup_capture_seconds = None

    def _bind(self, kv) -> None:
        """The engine's caches as the device model's, once, before any module is built (modules write the caches they
        are built on); then the state over them (State zeroes them: a fresh engine cache)."""
        m = self.model
        if self.state is not None:
            assert m.kvpe is kv.tensor(KT.KV) and m.index is kv.tensor(KT.INDEX), "one engine cache per runtime"
            return
        assert m.kvpe is None and m.index is None and not m.mods, "the device model was built on other caches"
        m.kvpe, m.index = kv.tensor(KT.KV), kv.tensor(KT.INDEX)
        self.state = m.new_state(self.geo.max_seq)

    def _zero_ids(self):
        import torch

        chunk, sp = self.geo.chunk, self.geo.sp
        return ttnn.from_torch(
            torch.zeros(sp, 1, chunk // sp, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(0, None)
            ),  # SP rows over mesh axis 0 (layout.SP_AXIS), replicated over TP, as the engine's H2D stream
        )

    def _forward_traced(self, after_layer=None):
        """The captured forward: embed the persistent ids, every served layer on the device-held chunk values."""
        m = self.model
        h = m.embed_device(self._ids)
        out = m.forward_meta(h, self.state, self.geo.layers, final=False, after_layer=after_layer)
        m.free(h)
        return out

    def _trace_ack(self):
        if self.d2h_service is None:
            return None
        d2h, md = self.d2h_service, self._md
        return lambda k, layer: ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=md)

    def _prepare_trace(self) -> None:
        """Persistent buffers, then one metadata-path chunk (compiles its programs and builds the exact fp32 table
        selections; no acks), then the caches zeroed again."""
        import torch

        m, chunk = self.model, self.geo.chunk
        meta = m.enable_meta()
        self._ids = self._zero_ids()
        self._md = ttnn.from_torch(
            torch.tensor([0, 0, chunk], dtype=torch.int64).reshape(1, 1, 1, 3),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # the staging copies compiled here too: a program first compiled after the capture allocates device buffers
        # the replays may overwrite
        ids = self._zero_ids()
        ttnn.copy(ids, self._ids)
        ttnn.deallocate(ids)
        md = ttnn.from_torch(
            torch.tensor([0, 0, chunk], dtype=torch.int64).reshape(1, 1, 1, 3),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy(md, self._md)
        ttnn.deallocate(md)
        meta.write(0, 0, chunk)
        self.state.slot = 0
        m.free(self._forward_traced())
        ttnn.synchronize_device(self.mesh_device)
        self.state = m.new_state(self.geo.max_seq)

    def warmup_ack_count(self) -> int:
        """D2H ack records capture_trace's warm pass emits (one per served layer); the runner drains them."""
        return len(self.geo.layers) if self.traced and self.d2h_service is not None else 0

    def capture_trace(self, kv_cache) -> None:
        """Record the traced forward once (the runner calls this on the first record, after set_d2h_ack_service:
        the socket programs must be compiled before a capture). With D2H acks, the warm pass
        that compiles the ack programs ran in set_d2h_ack_service (else here); its warmup_ack_count records go to the
        runner, which drains them. The caches are zeroed again before the capture."""
        if not self.traced or self._tid is not None:
            return
        self._bind(kv_cache)
        m, chunk = self.model, self.geo.chunk
        ack = self._trace_ack()
        if ack is not None and not self._acks_warm:
            self._warm_acks()
        m.meta.write(0, 0, chunk)
        self._tid = sub_device.begin_capture(self.mesh_device)
        try:
            self._out = self._forward_traced(ack)  # allocated inside the capture: the trace owns it
        finally:
            sub_device.end_capture(self.mesh_device, self._tid)

    def release_trace(self) -> None:
        if self._tid is not None:
            sub_device.release(self.mesh_device, self._tid)
            self._tid = None

    def compile(self, kv_cache) -> None:
        """Bind the state, then one warm-up chunk (slot 0, start 0, no acks) through every served layer: each layer's
        modules are built (weights read and uploaded) and its programs compiled on its first call, which would
        otherwise make the first request take as long as loading the model. Then the caches are zeroed again (a new
        state) before the runner admits any request. Traced: the metadata path's warm-up chunk instead."""
        self._bind(kv_cache)
        self._kv = kv_cache
        if self.traced:  # the metadata path's warm-up instead (its modules are the same; built on that pass)
            self._prepare_trace()
            return
        m, st, chunk = self.model, self.state, self.geo.chunk
        ids = self._zero_ids()
        st.slot = 0
        h = m.embed_device(ids)
        ttnn.deallocate(ids)
        for layer in self.geo.layers:
            out = m.layer(layer, h, 0, st, end=chunk)
            m.free(h)
            h = out
        m.free(h)
        ttnn.synchronize_device(self.mesh_device)
        self.state = m.new_state(self.geo.max_seq)

    def set_layer_completion_sink(self, sink) -> None:
        self.sink = sink
        if self.traced and self.state is not None:
            self.startup_capture()  # host-sink acks: none in the trace

    def startup_capture(self) -> None:
        """Capture before the request loop (the runner calls the ack hooks after opening the H2D service and before it
        reports ready): a client process pushes one warm-up record, taken here with the runner's socket op (its program
        compiles now, before the capture: a program first compiled after a capture can land on memory the replays
        write), then the capture. Falls back to the runner's capture on the first record when the
        H2D service was not recorded or the push fails (the record, if pushed, is then the runner's warm-up record)."""
        import time

        import torch
        from loguru import logger

        if self._tid is not None or not H2D_SERVICES or self._kv is None:
            return
        t0 = time.time()
        sid = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        r = subprocess.run([sys.executable, "-c", _PUSH_WARMUP, sid, str(WARMUP_WORD)], timeout=900, check=False)
        if r.returncode:
            logger.warning(f"[trace] warm-up record push failed (exit {r.returncode}); capturing on the first record")
            return
        ids, md = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            H2D_SERVICES[-1], metadata_size_bytes=12
        )
        words = ttnn.to_torch(ttnn.get_device_tensors(md)[0]).view(torch.int32).flatten().tolist()
        ttnn.deallocate(ids)
        ttnn.deallocate(md)
        assert words[:3] == [WARMUP_WORD] * 3, f"expected the warm-up record, got {words}"
        self.capture_trace(self._kv)
        self.startup_capture_seconds = round(time.time() - t0, 1)
        logger.info(f"[trace] captured at startup in {self.startup_capture_seconds} s (warm-up record + capture)")

    def set_d2h_ack_service(self, service) -> None:
        """Traced: the runner registers the service before the request loop, so the warm pass that compiles the ack
        programs runs here, not inside the first request (its records wait in the D2H FIFO, sized for them, until the
        runner drains warmup_ack_count of them after capture_trace)."""
        self.d2h_service = service
        if self.traced and self.state is not None and self._tid is None:
            self._warm_acks()
            self.startup_capture()

    def _warm_acks(self) -> None:
        m = self.model
        m.meta.write(0, 0, self.geo.chunk)
        m.free(self._forward_traced(self._trace_ack()))
        ttnn.synchronize_device(self.mesh_device)
        self.state = m.new_state(self.geo.max_seq)
        self._acks_warm = True

    def _ack(self, k: int, d2h, metadata_msg, slot_id: int, start: int, end: int, request_id: int) -> None:
        if d2h is not None:
            # a device op on the same queue after the layer's cache writes: the record reaches the host after them
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            return
        if self.sink is None:
            return
        ttnn.synchronize_device(self.mesh_device)  # the KV Manager reads DRAM out of band
        layer = self.first + k
        if hasattr(self.sink, "layers_completed"):
            self.sink.layers_completed(layer, layer + 1, slot_id, start, end)
        else:
            self.sink(layer, request_id)

    def _timing(self, **rec) -> None:
        """ML4_SERVE_TIMING=<path>: one JSON line per chunk (host times; a synchronize after each chunk's replay or
        eager forward, so the device's time per chunk is measured; for profiling only, off by default)."""
        path = os.environ.get("ML4_SERVE_TIMING")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        metadata_msg=None,
    ):
        """One chunk [actual_start, actual_start + chunk) of slot ``slot_id`` (rows at or past actual_end are pad) into
        the engine's caches; consumes input_tensor. Returns None (single rank: the caches are the output)."""
        start, end = int(actual_start), int(actual_end)
        if not 0 <= slot_id < self.geo.slots:
            raise ValueError(f"slot {slot_id} outside [0, {self.geo.slots})")
        if start % KT.RECORD_TOKENS or not start < end <= start + self.geo.chunk:
            raise ValueError(f"chunk [{start}, {end}) is not a 32-aligned start with 1..chunk tokens")
        if start + self.geo.chunk > self.geo.max_seq:
            raise ValueError(f"chunk at {start} runs past the slot's {self.geo.max_seq} positions")
        d2h = d2h_service if d2h_service is not None else self.d2h_service
        if d2h is not None and metadata_msg is None:
            raise ValueError("device-to-host acks need the chunk's metadata_msg")
        timing = bool(os.environ.get("ML4_SERVE_TIMING"))
        t_in = time.time()
        self._bind(kv_cache)
        if self.traced:
            self._replay(input_tensor, d2h, metadata_msg, int(slot_id), start, end, request_id)
            if timing:
                t_q = time.time()
                ttnn.synchronize_device(self.mesh_device)
                t_done = time.time()
                self._timing(
                    slot=int(slot_id),
                    start=start,
                    end=end,
                    rid=request_id,
                    acks="d2h" if d2h else "sink",
                    t_in=round(t_in, 4),
                    enqueue_ms=round((t_q - t_in) * 1e3, 2),
                    device_ms=round((t_done - t_in) * 1e3, 2),
                    t_done=round(t_done, 4),
                )
            return None
        m, st = self.model, self.state
        st.slot = int(slot_id)
        h = m.embed_device(input_tensor)
        ttnn.deallocate(input_tensor)
        for k, layer in enumerate(self.geo.layers):
            out = m.layer(layer, h, start, st, end=end)
            m.free(h)
            h = out
            self._ack(k, d2h, metadata_msg, slot_id, start, end, request_id)
        m.free(h)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)
        return None

    def _replay(self, ids, d2h, metadata_msg, slot: int, start: int, end: int, request_id: int) -> None:
        """Traced chunk: the ids, the D2H record and the chunk values into their persistent buffers, then the replay
        (non-blocking). Host-sink acks follow it, after a synchronize."""
        assert self._tid is not None, "use_trace: capture_trace() runs before the first chunk"
        assert d2h is None or d2h is self.d2h_service, "use_trace: the D2H service is bound by set_d2h_ack_service"
        ttnn.copy(ids, self._ids)
        ttnn.deallocate(ids)
        if metadata_msg is not None:
            if d2h is not None:
                ttnn.copy(metadata_msg, self._md)
            ttnn.deallocate(metadata_msg)
        self.model.meta.write(slot, start, end)
        self.state.slot = slot
        sub_device.execute(self.mesh_device, self._tid)
        if d2h is None and self.sink is not None:
            ttnn.synchronize_device(self.mesh_device)  # the KV Manager reads DRAM out of band
            for k in range(len(self.geo.layers)):
                layer = self.first + k
                if hasattr(self.sink, "layers_completed"):
                    self.sink.layers_completed(layer, layer + 1, slot, start, end)
                else:
                    self.sink(layer, request_id)

    # ------------------------------------------------------------------ the KV table
    def _banks(self) -> int:
        return int(self.mesh_device.dram_grid_size().x)

    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
        """One stage per table config, in config order (kv_table.CONFIGS)."""
        from models.demos.common.prefill.runners.migration import KvCacheStage

        first = self.first if first_layer_idx is None else int(first_layer_idx)
        n = len(self.geo.layers) if num_my_layers is None else int(num_my_layers)
        return [KvCacheStage(int(kv_cache.tensor(c).buffer_address()), first, n) for c in KT.CONFIGS]

    def kv_migration_base_address(self, kv_cache) -> int:
        return KT.base_address(self.geo, kv_cache.tensors(), self._banks())

    def build_kv_chunk_table(self, kv_cache, path: str, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        """The table of the engine's caches (kv_table.build_table), serialized to ``path``. Single rank: every stage
        layout (when given) names this mesh, its DRAM banks and its host."""
        from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table

        rows, cols = tuple(self.mesh_device.shape)
        banks, host = self._banks(), KT.host_name()
        if stage_layouts:
            assert all(len(s) == 1 for s in stage_layouts), "mistral_large_4 serves one rank"
            banks, host = int(stage_layouts[0][0]["num_banks"]), f"host-{stage_layouts[0][0]['host_tag']:08x}"
        row_fnids = [
            [self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)
        ]
        caches = kv_cache.tensors()
        table = KT.build_table(
            self.geo,
            caches,
            bases={n: int(t.buffer_address()) for n, t in caches.items()},
            banks=banks,
            row_fnids=row_fnids,
            host=host,
        )
        return serialize_prebuilt_kv_chunk_table(table=table, path=path)
