#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remote pi0.5 policy server (TT kernels) — for a robot on a different machine.

The TT box runs this server; the robot runs a thin client that POSTs an
observation and gets back an action chunk. Reuses the SAME
`Pi0_5LiberoAdapter.predict_chunk` as the in-process demo, so behavior is
identical — only the transport differs.

Transport: a minimal stdlib HTTP + JSON endpoint (no extra deps). Images are
base64-encoded uint8 HWC arrays. This is intentionally simple; for an
openpi-native robot stack, swap this for openpi's websocket policy-server
protocol (msgpack-numpy over `websockets`) reusing the same `predict_chunk`.

Protocol:
    POST /predict   (Content-Type: application/json)
    request  = {
        "agent_image":  {"b64": <base64 uint8 bytes>, "shape": [H, W, 3]},
        "wrist_image":  {"b64": ..., "shape": [H, W, 3]},
        "state":        [8 floats],      # [eef_pos(3), axis_angle(3), gripper(2)]
        "task":         "pick up the cup",
        "num_denoise_steps": 5           # optional
    }
    response = { "actions": [[7 floats], ...] }   # raw robot space, denormalized

    GET /health  ->  {"ok": true, "backend": "...", "action_dim": 7}

Run (1×8 mesh):
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    python_env/bin/python models/experimental/pi0_5/demo/policy_server.py \
      --checkpoint $PI05_CHECKPOINT_DIR --backend ttnn_1x8 --port 8000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _decode_image(spec: dict) -> np.ndarray:
    """{"b64": <base64>, "shape": [H, W, 3]} -> (H, W, 3) uint8."""
    raw = base64.b64decode(spec["b64"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(spec["shape"])


def _make_handler(adapter, default_n: int):
    class _Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "backend": adapter.backend, "action_dim": adapter.real_action_dim})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                agent = _decode_image(req["agent_image"])
                wrist = _decode_image(req["wrist_image"])
                state = np.asarray(req["state"], dtype=np.float32)
                task = req["task"]
                nds = int(req.get("num_denoise_steps", default_n))
                chunk = adapter.predict_chunk(agent, wrist, state, task, num_denoising_steps=nds)
                self._send(200, {"actions": np.asarray(chunk, dtype=np.float32).tolist()})
            except Exception as e:  # noqa: BLE001 — surface any request error to the client
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):  # quieter than the default stderr spam
            pass

    return _Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="pi0.5 remote policy server (TT kernels).")
    ap.add_argument("--checkpoint", default=os.environ.get("PI05_CHECKPOINT_DIR"))
    ap.add_argument("--backend", default="ttnn_1x8", choices=["ttnn_1x8", "ttnn", "pytorch"])
    ap.add_argument("--action-horizon", type=int, default=None)
    ap.add_argument("--num-denoise-steps", type=int, default=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")))
    ap.add_argument("--state-in-prompt", default="false", choices=["true", "false"])
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("--checkpoint is required (or set PI05_CHECKPOINT_DIR)")

    from models.experimental.pi0_5.demo.policy import build_policy

    with build_policy(
        args.checkpoint,
        backend=args.backend,
        action_horizon=args.action_horizon,
        state_in_prompt=(args.state_in_prompt == "true"),
        device_id=args.device_id,
    ) as adapter:
        handler = _make_handler(adapter, args.num_denoise_steps)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(f"[server] pi0.5 policy ready on http://{args.host}:{args.port}  (backend={args.backend})", flush=True)
        print("[server] POST /predict  ·  GET /health  ·  Ctrl-C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] shutting down", flush=True)
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
