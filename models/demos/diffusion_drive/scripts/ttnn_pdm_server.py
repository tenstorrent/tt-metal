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
  2. loads DiffusionDriveModel from the checkpoint and installs the TTNN stack
     (build_stage2 → 3 → 3_4 → 3_5),
  3. serves one request at a time: recv {camera,lidar,status} numpy arrays →
     run forward → send back {trajectory} numpy array.

Protocol (both directions): 8-byte big-endian length prefix + pickle payload.

Run (tt-metal venv):
    source python_env/bin/activate && export PYTHONPATH=/root/tt/tt-metal
    python models/demos/diffusion_drive/scripts/ttnn_pdm_server.py \
        --checkpoint /root/02/weights/diffusiondrive_navsim_88p1_PDMS.pth \
        --anchors    /root/02/resnet34/kmeans_navsim_traj_20.npy \
        --socket     /tmp/ttnn_dd.sock
"""

from __future__ import annotations

import argparse
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
    model.build_stage2(device).build_stage3(device).build_stage3_4(device).build_stage3_5(device)
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
        _log("model ready (stage 3.5)")

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
                    n += 1
                    if n % 200 == 0:
                        _log(f"served {n} requests")
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--socket", default="/tmp/ttnn_dd.sock")
    args = ap.parse_args()
    serve(args.checkpoint, args.anchors, args.socket)


if __name__ == "__main__":
    main()
