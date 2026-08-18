"""Prove a captured trace actually re-executes, rather than leaving a stale buffer behind.

A bit-identical replay is not evidence on its own: the capture's output allocation can reuse the
address the eager output was just freed from, so a replay that did nothing would still read the
right numbers. Two checks close that gap -- poison the output buffer before replaying, and change
the input after capture and require the output to follow.
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
T_LAT = int(os.environ.get("DIFFVAE_LATENT_T", 4))
LH, LW = 34, 60


def heartbeat(stop: threading.Event) -> None:
    n = 0
    while not stop.wait(20.0):
        n += 20
        print(f"[heartbeat] {n}s", flush=True)


def main() -> None:
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), trace_region_size=200_000_000)
    try:
        config = decoder_config(CHECKPOINT)
        c = config["in_channels"]
        lat_a = torch.randn(1, c, T_LAT, LH, LW, generator=torch.Generator().manual_seed(1))
        lat_b = torch.randn(1, c, T_LAT, LH, LW, generator=torch.Generator().manual_seed(2))
        ccl = CCLManager(mesh, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=ttnn.Topology.Linear)
        tp = 0 if os.environ.get("DIFFVAE_TP_HEADS") == "1" else None
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh,
            ccl_manager=ccl,
            stage5_na3d_backend="op_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=tp,
            stages_na3d_backend="op_sp_w_sharded",
            stages_sp_axis=1,
            stages_tp_axis=tp,
        )
        dec.load_checkpoint(CHECKPOINT)

        def upload(t):
            return ttnn.from_torch(
                t.reshape(1, c, T_LAT, LH * LW).contiguous(),
                device=mesh,
                dtype=dec.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        raw = upload(lat_a)
        raw_b = upload(lat_b)
        read = lambda x: ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
        run = lambda lat: dec.forward_context(lat, gather_output=False, latent_tt=raw)

        out, _ = run(lat_a)
        ref_a = read(out)
        ttnn.deallocate(out)
        out, _ = run(lat_a)
        ref_a2 = read(out)
        ttnn.deallocate(out)
        print(f"[eager] A reproducible: {torch.equal(ref_a, ref_a2)}", flush=True)

        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out_t, _ = run(lat_a)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        print(f"[trace] captured {tid}", flush=True)

        # CHECK 1 -- poison the output, then replay. A no-op replay leaves the poison behind.
        ttnn.copy(ttnn.zeros_like(out_t), out_t)
        poisoned = read(out_t)
        print(f"[check1] output poisoned to zeros: max|.|={float(poisoned.abs().max()):.6f}", flush=True)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        got_a = read(out_t)
        print(
            f"[check1] after replay == eager A: {torch.equal(got_a, ref_a)}  "
            f"max|diff|={float((got_a - ref_a).abs().max()):.6f}",
            flush=True,
        )

        # CHECK 2 -- change the input in place. Replay must follow it.
        ttnn.copy(raw_b, raw)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        replay_ms = (time.perf_counter() - t0) * 1000
        got_b = read(out_t)
        out, _ = run(lat_b)
        ref_b = read(out)
        ttnn.deallocate(out)
        print(f"[check2] replay on new input differs from A: {not torch.equal(got_b, got_a)}", flush=True)
        print(
            f"[check2] replay on new input == eager B  : {torch.equal(got_b, ref_b)}  "
            f"max|diff|={float((got_b - ref_b).abs().max()):.6f}",
            flush=True,
        )
        print(f"[check2] replay {replay_ms:8.1f} ms", flush=True)
        ttnn.release_trace(mesh, tid)
    finally:
        stop.set()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
