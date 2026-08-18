"""Capture and replay a ttnn trace of the deterministic stages, and time the replay.

The upload stays outside the captured region (a trace bakes input addresses and refuses
host-to-device writes during capture), so the raw latent is uploaded once into a buffer the
capture reads and later replays read again.

DIFFVAE_STAGE_TIMING is deliberately left unset: its per-stage timers sync the mesh, and a
synchronize inside a captured region is both untraceable and a distorted measurement.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config
from models.tt_dit.parallel.manager import CCLManager

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)
ITERS = int(os.environ.get("ITERS", 3))
T_LAT = int(os.environ.get("DIFFVAE_LATENT_T", 4))
LH, LW = 34, 60


def heartbeat(stop: threading.Event) -> None:
    """Keep the broker's no-output watchdog fed; trace capture is silent for minutes."""
    n = 0
    while not stop.wait(20.0):
        n += 20
        print(f"[heartbeat] {n}s elapsed, still working", flush=True)


def main() -> None:
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), trace_region_size=200_000_000)
    try:
        config = decoder_config(CHECKPOINT)
        latent = torch.randn(1, config["in_channels"], T_LAT, LH, LW)
        ccl = CCLManager(mesh, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=ttnn.Topology.Linear)
        tp_axis = 0 if os.environ.get("DIFFVAE_TP_HEADS") == "1" else None
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh,
            ccl_manager=ccl,
            stage5_na3d_backend="op_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=tp_axis,
            stages_na3d_backend="op_sp_w_sharded",
            stages_sp_axis=1,
            stages_tp_axis=tp_axis,
        )
        print(f"[setup] loading checkpoint, latent T={T_LAT} ({8 * T_LAT - 7} frames)", flush=True)
        dec.load_checkpoint(CHECKPOINT)

        b, c, t, h, w = latent.shape
        raw = ttnn.from_torch(
            latent.reshape(1, c, t, h * w).contiguous(),
            device=mesh,
            dtype=dec.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        print("[setup] latent uploaded to a persistent buffer (outside the trace)", flush=True)

        run = lambda: dec.forward_context(latent, gather_output=False, latent_tt=raw)

        # Two eager passes: the first builds program cache, device plans, persistent CCL buffers and
        # the cached shard offsets, none of which may be created inside the capture.
        for i in range(2):
            t0 = time.perf_counter()
            ctx, dims = run()
            ttnn.synchronize_device(mesh)
            print(f"[eager {i}] {(time.perf_counter() - t0) * 1000:8.1f} ms  dims={dims}", flush=True)
            eager_ref = ttnn.to_torch(ttnn.get_device_tensors(ctx)[0]).float()
            ttnn.deallocate(ctx)

        print("[trace] begin_trace_capture", flush=True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ctx_t, dims_t = run()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        print(f"[trace] captured id={tid} dims={dims_t}", flush=True)

        for i in range(ITERS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"[replay {i}] {(time.perf_counter() - t0) * 1000:8.1f} ms", flush=True)

        got = ttnn.to_torch(ttnn.get_device_tensors(ctx_t)[0]).float()
        diff = float((eager_ref - got).abs().max())
        print(f"[check] max|eager-replay| = {diff:.6f}  identical={bool(torch.equal(eager_ref, got))}", flush=True)
        ttnn.release_trace(mesh, tid)
        print("[trace] released", flush=True)
    finally:
        stop.set()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
