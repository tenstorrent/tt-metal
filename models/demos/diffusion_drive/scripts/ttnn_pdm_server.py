#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN inference server for NavSim PDM evaluation (Stage-3.5 model).

Why a server?  The NavSim eval harness (`run_pdm_score.py`) runs in the
`navsim` conda env (Python 3.9); TTNN is only importable in the tt-metal venv
(Python 3.10).  The two cannot share the compiled `ttnn` wheel, so the agent
(conda side) delegates inference to this server (venv side) over a Unix-domain
socket.

This process:
  1. opens the Wormhole device (l1_small_size=32768),
  2. loads DiffusionDriveModel from the checkpoint and installs the full TTNN
     stack (build_stage2 → 3 → 3_4 → 3_5 → 3_6 → 3_7 → 4) — every weight-bearing op
     on-device, with the TransFuser backbone running as one device-native graph
     (consolidated, auto-enabled at build_stage3_6; DD_CONSOLIDATE=0 to opt out)
     and the perception block + DDIM decoder consolidated by build_stage4,
  3. serves one request at a time: recv {camera,lidar,status} numpy arrays →
     run forward → send back {trajectory} numpy array.

Protocol (both directions): 8-byte big-endian length prefix + pickle payload.

Run (tt-metal venv).  Paths come from env vars (see the demo README "Eval
environment" block); the flags below override them:
    source python_env/bin/activate && export PYTHONPATH="${TT_METAL_HOME:-$PWD}"
    export DD_CHECKPOINT_PATH="${DD_DATA_ROOT:-/mnt/diffusion-drive}/weights/diffusiondrive_navsim_88p1_PDMS.pth"
    export DD_ANCHOR_PATH="${DD_DATA_ROOT:-/mnt/diffusion-drive}/resnet34/kmeans_navsim_traj_20.npy"
    python models/demos/diffusion_drive/scripts/ttnn_pdm_server.py \
        --checkpoint "$DD_CHECKPOINT_PATH" \
        --anchors    "$DD_ANCHOR_PATH" \
        --socket     "${TTNN_DD_SOCKET:-/tmp/ttnn_dd.sock}"
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pickle
import socket
import struct
import sys
import traceback

import numpy as np
import torch


def _log(*a):
    print("[ttnn_pdm_server]", *a, file=sys.stderr, flush=True)


def _rss_gb() -> float:
    """Resident set size in GB (for watching memory)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return -1.0


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def _trim_heap() -> None:
    """Return freed glibc-arena pages to the OS.

    The per-request pickle/numpy alloc-free churn fragments the glibc heap,
    growing RSS ~2.6 MB/forward → host OOM over ~12k forwards. The model forward
    itself retains nothing (verified: live ttnn-tensor count stays 0 across
    forwards), so this is heap fragmentation, not a Python/ttnn reference leak —
    which is why gc.collect() did NOT help. malloc_trim keeps RSS flat.
    """
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _recv_exactly(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf += chunk
    return buf


def _recv_msg(conn):
    (length,) = struct.unpack(">Q", _recv_exactly(conn, 8))
    return pickle.loads(_recv_exactly(conn, length))


def _send_msg(conn, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


def _build_model(checkpoint: str, anchors: str, device):
    from models.demos.diffusion_drive.tt.config import ModelConfig
    from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    cfg = ModelConfig()
    cfg.plan_anchor_path = anchors
    # latent=False → use the real LiDAR BEV that the agent sends.
    model = TtnnDiffusionDriveModel.from_checkpoint(checkpoint, cfg, device, latent=False)
    # Full on-device stack: backbone (stems + BasicBlocks + FPN + GPT fusion),
    # perception head, DDIM denoiser, and agent head all run via TTNN ops.
    # Once build_stage3 (FPN) + build_stage3_6 (stems+fusion) are in, the backbone
    # runs as one device-native graph by default (consolidated; the 8 per-stage host
    # round-trips are gone). Stage 3.6 (fusion) requires the production resolution
    # the agent sends (camera 256×1024, LiDAR 256×256), where the pool/upsample
    # ratios are integer. Set DD_CONSOLIDATE=0 to fall back to the staged path.
    # build_stage4 then consolidates the perception block + the DDIM decoder loop
    # onto the device too (drops the per-drop-in round-trips) — measured ~1.20×
    # faster per request, and the prerequisite for whole-model trace capture.
    (
        model.build_stage2(device)
        .build_stage3(device)
        .build_stage3_4(device)
        .build_stage3_5(device)
        .build_stage3_6(device)
        .build_stage3_7(device)
        .build_stage4(device)
    )
    return model


def _to_features(req: dict) -> dict:
    """req has numpy arrays already carrying a batch dim (agent unsqueezed)."""
    return {
        "camera_feature": torch.from_numpy(np.asarray(req["camera_feature"])).float(),
        "lidar_feature": torch.from_numpy(np.asarray(req["lidar_feature"])).float(),
        "status_feature": torch.from_numpy(np.asarray(req["status_feature"])).float(),
    }


def serve(checkpoint: str, anchors: str, sock_path: str) -> None:
    import ttnn

    if os.path.exists(sock_path):
        os.unlink(sock_path)

    _log(f"opening device; checkpoint={checkpoint}")
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        model = _build_model(checkpoint, anchors, device)
        _log("model ready (full on-device stack: stage 3.7)")

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(8)
        _log(f"listening on {sock_path}")
        # touch a readiness file so the agent can wait for us
        open(sock_path + ".ready", "w").close()

        n = 0
        while True:
            conn, _ = srv.accept()
            try:
                while True:
                    try:
                        req = _recv_msg(conn)
                    except (ConnectionError, EOFError):
                        break
                    if req.get("cmd") == "shutdown":
                        _log("shutdown requested")
                        _send_msg(conn, {"ok": True})
                        return
                    features = _to_features(req)
                    with torch.no_grad():
                        out = model(features)
                    traj = out["trajectory"].squeeze(0).cpu().numpy().astype(np.float32)
                    _send_msg(conn, {"trajectory": traj})
                    del out, features, traj, req
                    _trim_heap()  # keep RSS flat — see _trim_heap (glibc fragmentation, not a ref leak)
                    n += 1
                    if n % 200 == 0:
                        _log(f"served {n} requests (rss={_rss_gb():.1f}GB)")
            except Exception:
                _log("request error:\n" + traceback.format_exc())
                try:
                    _send_msg(conn, {"error": traceback.format_exc()})
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        ttnn.close_device(device)
        if os.path.exists(sock_path + ".ready"):
            os.unlink(sock_path + ".ready")
        _log("device closed")


def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaults come from env vars so the harness is portable across machines;
    # the flags override them. See the demo README "Eval environment" block.
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get("DD_CHECKPOINT_PATH"),
        help="model checkpoint .pth (default: $DD_CHECKPOINT_PATH)",
    )
    ap.add_argument(
        "--anchors",
        default=os.environ.get("DD_ANCHOR_PATH"),
        help="K-means plan-anchor .npy (default: $DD_ANCHOR_PATH)",
    )
    ap.add_argument(
        "--socket",
        default=os.environ.get("TTNN_DD_SOCKET", "/tmp/ttnn_dd.sock"),
        help="Unix-domain socket path (default: $TTNN_DD_SOCKET or /tmp/ttnn_dd.sock)",
    )
    args = ap.parse_args()
    if not args.checkpoint or not args.anchors:
        ap.error("set --checkpoint/--anchors or export $DD_CHECKPOINT_PATH / $DD_ANCHOR_PATH")
    serve(args.checkpoint, args.anchors, args.socket)


if __name__ == "__main__":
    main()
