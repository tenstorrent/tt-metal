# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision-only PER-STAGE TRACE harness for the GLX SigLIP slice.

WHY: device-kernel-time dropped -32% (8.06->5.47ms) but eager wall-clock got
WORSE (+22%) because the eager host-bounce harness runs host dispatch + device
serially. A TTNN trace replays the captured device program without per-op host
dispatch, so the device-time win should finally show up as wall-clock.

The vision stage spans 4 separate 1x1 submeshes (embed / layerA / layerB / tail)
glued by host-bounce transport. A single trace CANNOT span submeshes or contain
host I/O, so we capture ONE TRACE PER CHIP and stitch them with transport that
stays OUTSIDE the traces. This is the minimal proving ground for the per-stage
multi-trace pattern that prefill/denoise will reuse.

Design (the legal multi-trace structure):
    [embed trace]  send  [layerA trace]  send  [layerB trace]  send  [tail trace]
      chip0 CQ0   (host)    chip1 CQ0    (host)    chip2 CQ0    (host)   chip3 CQ0

Handoff buffers between chips are PERSISTENT (stable tensor-id) so the trace can
write/read them across replays. Transport refreshes them via
copy_host_to_device_tensor (host mode) - legal because it's between traces.

Run (with the optimized config):
    source models/experimental/pi0_5/local_env.sh
    PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
    PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_trace.py
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


class VisionTraceRunner:
    """Captures 4 per-chip traces and replays them stitched by host transport."""

    def __init__(self, stage, chips):
        self.stage = stage
        self.chips = chips
        # persistent handoff buffers (filled lazily on first capture)
        self.in_buf = None  # pixel values on chip0
        self.hand = [None, None, None]  # chip0->1, 1->2, 2->3 inputs (persistent)
        self.tids = [None, None, None, None]
        self.outs = [None, None, None, None]  # captured output handle per chip

    # --- the 4 pure-device trace bodies -----------------------------------
    def _body_embed(self):
        return self.stage.embed_slice.forward(self.in_buf)

    def _body_layA(self):
        return self.stage.layer_slice_a.forward(self.hand[0])

    def _body_layB(self):
        return self.stage.layer_slice_b.forward(self.hand[1])

    def _body_tail(self):
        return self.stage.tail_slice.forward(self.hand[2])

    @staticmethod
    def _persist(src, dst):
        """Copy src (device tensor on some mesh) into persistent dst on target
        mesh via host. Allocates dst on first call (stable id thereafter)."""
        host = ttnn.to_torch(src)
        if dst is None:
            return ttnn.from_torch(host, dtype=src.dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT), dst)
        return dst

    def _upload_to(self, src, mesh):
        host = ttnn.to_torch(src)
        return ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def capture(self, pixel_values):
        chips = self.chips
        # stage the persistent input on chip0
        self.in_buf = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=chips[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ---- WARMUP eager once: compiles every kernel + allocates handoffs --
        h0 = self._body_embed()
        self.hand[0] = self._upload_to(h0, chips[1])
        h1 = self._body_layA()
        self.hand[1] = self._upload_to(h1, chips[2])
        h2 = self._body_layB()
        self.hand[2] = self._upload_to(h2, chips[3])
        _ = self._body_tail()

        # ---- capture one trace per chip ------------------------------------
        bodies = [self._body_embed, self._body_layA, self._body_layB, self._body_tail]
        for i, (chip, body) in enumerate(zip(chips, bodies)):
            self.tids[i] = ttnn.begin_trace_capture(chip, cq_id=0)
            self.outs[i] = body()
            ttnn.end_trace_capture(chip, self.tids[i], cq_id=0)
        return self.outs[3]

    def replay(self, pixel_values):
        chips = self.chips
        # refresh chip0 input in place
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            self.in_buf,
        )
        ttnn.execute_trace(chips[0], self.tids[0], cq_id=0, blocking=True)
        # embed out -> layerA in
        self._persist_into(self.outs[0], self.hand[0])
        ttnn.execute_trace(chips[1], self.tids[1], cq_id=0, blocking=True)
        self._persist_into(self.outs[1], self.hand[1])
        ttnn.execute_trace(chips[2], self.tids[2], cq_id=0, blocking=True)
        self._persist_into(self.outs[2], self.hand[2])
        ttnn.execute_trace(chips[3], self.tids[3], cq_id=0, blocking=True)
        return self.outs[3]

    @staticmethod
    def _persist_into(src, dst):
        host = ttnn.to_torch(src)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT), dst)


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

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    try:
        vision_submesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        vision_per_chip = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c)) for c in range(4)]
        handles = _Handles(parent, vision_submesh, vision_per_chip)

        from models.experimental.pi0_5.tests.perf.bench_siglip_l1 import _HostTransport

        stage = StageVision(cfg, loader.categorized_weights, handles, transport=_HostTransport())

        runner = VisionTraceRunner(stage, vision_per_chip)
        print(f"\n=== SigLIP vision-only PER-STAGE TRACE  bs={bs} ===")
        print("capturing 4 per-chip traces...")
        out_handle = runner.capture(pixel_values)
        print("capture done. validating PCC...")
        out = ttnn.to_torch(runner.replay(pixel_values))
        pcc = _pcc(ref_out, out)
        print(f"  shape {tuple(out.shape)}  PCC={pcc:.6f}  ({'OK' if pcc>=0.997 else 'FAIL'})")

        # timing
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
        print(f"  traced wall-clock ms: avg={avg:.3f} min={mn:.3f} stddev={sd:.3f}")
        print(f"METRIC siglip_trace_ms={avg:.4f}")
        print(f"METRIC siglip_trace_pcc={pcc:.6f}")
    finally:
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
