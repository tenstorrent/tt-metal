# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision-only PER-STAGE TRACE + FABRIC-SOCKET harness for the GLX SigLIP slice.

Combines the two wins that convert device-time into wall-clock:
  1. per-chip TRACE  -> kills per-op host dispatch (proven in bench_siglip_trace.py)
  2. SocketTransport -> kills the to_torch/from_torch host bounce between chips
     (proven bit-exact maxdiff=0.0 intra-island on this host)

The vision stage's 4 chips are all inside ONE 4x4 island, so intra-island
fabric is UP and sockets work here (cross-island is down, irrelevant for vision).

Structure (sockets stay BETWEEN per-chip traces, recv buf = next trace's input):
    exec(embed trace) -> socket send -> recv into layA_in -> exec(layA trace) -> ...

Run (optimized config):
    source models/experimental/pi0_5/local_env.sh
    PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
    PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_trace_socket.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchSigLIPVisionTower
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as TorchMMProjector
from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision
from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("BENCH_ITERS", "30"))


def _pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


class _Handles:
    def __init__(self, parent, vision_submesh, vision_per_chip):
        self.parent = parent
        self.vision_submesh = vision_submesh
        self.vision_per_chip = vision_per_chip


class VisionTraceSocketRunner:
    """4 per-chip traces stitched by fabric-socket transport (no host bounce)."""

    def __init__(self, stage, chips):
        self.stage = stage
        self.chips = chips
        self.tr = SocketTransport()
        self.in_buf = None  # chip0 persistent input (pixel values)
        self.recv = [None, None, None]  # socket recv buffers = inputs to layA/layB/tail
        self.tids = [None, None, None, None]
        self.outs = [None, None, None, None]

    def _bodies(self):
        return [
            lambda: self.stage.embed_slice.forward(self.in_buf),
            lambda: self.stage.layer_slice_a.forward(self.recv[0]),
            lambda: self.stage.layer_slice_b.forward(self.recv[1]),
            lambda: self.stage.tail_slice.forward(self.recv[2]),
        ]

    def capture(self, pixel_values):
        chips = self.chips
        self.in_buf = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=chips[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bodies = self._bodies()

        # --- WARMUP eager: compile kernels, allocate persistent recv buffers ---
        h0 = bodies[0]()
        self.recv[0] = self.tr.allocate_recv_buffer(h0, chips[1])
        self.tr.send(h0, chips[1], out_buf=self.recv[0])
        ttnn.synchronize_device(chips[1])
        h1 = bodies[1]()
        self.recv[1] = self.tr.allocate_recv_buffer(h1, chips[2])
        self.tr.send(h1, chips[2], out_buf=self.recv[1])
        ttnn.synchronize_device(chips[2])
        h2 = bodies[2]()
        self.recv[2] = self.tr.allocate_recv_buffer(h2, chips[3])
        self.tr.send(h2, chips[3], out_buf=self.recv[2])
        ttnn.synchronize_device(chips[3])
        _ = bodies[3]()

        # --- capture one compute trace per chip (sockets stay OUTSIDE) ---
        for i, (chip, body) in enumerate(zip(chips, bodies)):
            self.tids[i] = ttnn.begin_trace_capture(chip, cq_id=0)
            self.outs[i] = body()
            ttnn.end_trace_capture(chip, self.tids[i], cq_id=0)
        return self.outs[3]

    def replay_profiled(self, pixel_values, agg):
        chips = self.chips

        def ph(name, fn, sync_mesh):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(sync_mesh)
            agg.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)

        ph(
            "upload",
            lambda: ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.in_buf
            ),
            chips[0],
        )
        ph("embed", lambda: ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=True), chips[0])
        ph("send01", lambda: self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0]), chips[1])
        ph("layA", lambda: ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=True), chips[1])
        ph("send12", lambda: self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1]), chips[2])
        ph("layB", lambda: ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=True), chips[2])
        ph("send23", lambda: self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2]), chips[3])
        ph("tail", lambda: ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True), chips[3])
        return self.outs[3]

    def replay_fast(self, sync_only=True):
        chips = self.chips
        # NO upload (reuse whatever is in in_buf) - measures pure compute+socket pipelined
        ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=False)
        self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0])
        ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=False)
        self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1])
        ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=False)
        self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2])
        ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True)
        return self.outs[3]

    def replay_prestaged(self, host_tile):
        chips = self.chips
        ttnn.copy_host_to_device_tensor(host_tile, self.in_buf)  # reuse pre-tilized host tensor
        ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=False)
        self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0])
        ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=False)
        self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1])
        ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=False)
        self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2])
        ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True)
        return self.outs[3]

    def stream(self, n):
        chips = self.chips
        for _ in range(n):
            ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=False)
            self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0])
            ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=False)
            self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1])
            ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=False)
            self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2])
            ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=False)
        ttnn.synchronize_device(chips[3])

    def replay(self, pixel_values):
        chips = self.chips
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.in_buf
        )
        ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=False)
        self.tr.send(self.outs[0], chips[1], out_buf=self.recv[0])
        ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=False)
        self.tr.send(self.outs[1], chips[2], out_buf=self.recv[1])
        ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=False)
        self.tr.send(self.outs[2], chips[3], out_buf=self.recv[2])
        ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True)
        return self.outs[3]


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    bs = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    pixel_values = torch.randn(bs, 3, cfg.image_size, cfg.image_size)

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    ref_tower = TorchSigLIPVisionTower(cfg, loader.categorized_weights["vlm_vision"])
    ref_proj = TorchMMProjector(loader.categorized_weights["vlm_projector"])
    with torch.no_grad():
        ref_out = ref_proj.forward(ref_tower.forward(pixel_values))

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    try:
        vision_submesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        vision_per_chip = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c)) for c in range(4)]
        handles = _Handles(parent, vision_submesh, vision_per_chip)
        # build the stage but force OUR socket transport (not host)
        stage = StageVision(cfg, loader.categorized_weights, handles, transport=SocketTransport())

        runner = VisionTraceSocketRunner(stage, vision_per_chip)
        print(f"\n=== SigLIP vision PER-STAGE TRACE + FABRIC SOCKET  bs={bs} ===")
        print("capturing 4 per-chip traces + socket buffers...")
        runner.capture(pixel_values)
        out = ttnn.to_torch(runner.replay(pixel_values))
        pcc = _pcc(ref_out, out)
        print(f"  shape {tuple(out.shape)}  PCC={pcc:.6f}  ({'OK' if pcc>=0.997 else 'FAIL'})")

        times = []
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter()
            runner.replay(pixel_values)
            ttnn.synchronize_device(vision_per_chip[3])
            dt = (time.perf_counter() - t0) * 1e3
            if i >= WARMUP:
                times.append(dt)
        avg, mn = statistics.mean(times), min(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  traced+socket wall-clock ms: avg={avg:.3f} min={mn:.3f} stddev={sd:.3f}")
        print(f"METRIC siglip_trace_socket_ms={avg:.4f}")
        print(f"METRIC siglip_trace_socket_pcc={pcc:.6f}")
        # PRE-STAGED single-shot: host tensor tilized ONCE, only copy_host_to_device per shot
        host_tile = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tps = []
        for i in range(25):
            t0 = time.perf_counter()
            runner.replay_prestaged(host_tile)
            ttnn.synchronize_device(vision_per_chip[3])
            if i >= 5:
                tps.append((time.perf_counter() - t0) * 1e3)
        print(f"  PRESTAGED single-shot ms: avg={statistics.mean(tps):.3f} min={min(tps):.3f}")
        # THROUGHPUT: stream N frames back-to-back, sync once -> steady-state per-frame
        import time as _t

        runner.stream(5)  # warm
        for N in (20, 40):
            t0 = _t.perf_counter()
            runner.stream(N)
            dt = (_t.perf_counter() - t0) * 1e3
            print(f"  STREAM N={N}: total={dt:.2f}ms per-frame={dt/N:.3f}ms")
        # measure WITHOUT per-iter upload (upload overlaps / is pre-staged)
        tnoup = []
        for i in range(25):
            t0 = time.perf_counter()
            runner.replay_fast()
            ttnn.synchronize_device(vision_per_chip[3])
            if i >= 5:
                tnoup.append((time.perf_counter() - t0) * 1e3)
        print(f"  NO-UPLOAD pipelined wall-clock ms: avg={statistics.mean(tnoup):.3f} min={min(tnoup):.3f}")
        agg = {}
        for _ in range(20):
            runner.replay_profiled(pixel_values, agg)
        print("  --- per-phase avg ms (device-synced) ---")
        tot = 0.0
        for k, v in agg.items():
            m = statistics.mean(v)
            tot += m
            print(f"    {k:7s} {m:7.3f}")
        print(f"    {'TOTAL':7s} {tot:7.3f}")
    finally:
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
