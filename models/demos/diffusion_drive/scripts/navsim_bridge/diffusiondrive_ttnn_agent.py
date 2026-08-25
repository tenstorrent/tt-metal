# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NavSim agent that delegates inference to the TTNN PDM server.

Runs in the `navsim` conda env (Python 3.9).  It builds features exactly like
the upstream TransfuserAgent, but instead of running the PyTorch model it sends
the features to ``ttnn_pdm_server.py`` (tt-metal venv, Python 3.10) over a
Unix-domain socket and wraps the returned trajectory.

It deliberately does NOT build or load the 60 M-param PyTorch model — the
weights live in the TTNN server — so each ray worker stays lightweight.

Wire-up (see scripts/navsim_bridge/README.md):
  1. start ttnn_pdm_server.py in the tt-metal venv,
  2. put this file on PYTHONPATH (or in navsim/agents/diffusiondrive/),
  3. put diffusiondrive_ttnn_agent.yaml in the navsim agent config dir,
  4. run run_pdm_score.py with agent=diffusiondrive_ttnn_agent.
"""

from __future__ import annotations

import io
import os
import socket
import struct
from typing import Any, Dict, List

import numpy as np
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import SensorConfig


def _recv_exactly(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ttnn server closed mid-message")
        buf += chunk
    return buf


def _encode_msg(obj: dict) -> bytes:
    """Serialise a ``{name: ndarray | scalar}`` mapping as an ``.npz`` payload.

    Deliberately not pickle. This frames a socket that the eval harness feeds, and
    the pickle loader executes code found in the payload, so a malformed or hostile
    frame would be arbitrary code execution in the process holding the device.
    ``.npz`` carries arrays only, and is read back with ``allow_pickle=False``.

    Duplicated in the two endpoints on purpose: the agent side runs in the navsim
    conda env (3.9) and the server side in the tt-metal venv (3.10), which is the
    reason this bridge exists at all — they cannot import a shared module.
    """
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in obj.items()})
    return buf.getvalue()


def _decode_msg(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        out = {}
        for key in npz.files:
            val = npz[key]
            # Scalars ("cmd", "ok", "error") were widened to 0-d arrays by
            # np.asarray on the way out — hand them back as Python scalars.
            out[key] = val.item() if val.ndim == 0 else val
        return out


class DiffusionDriveTtnnAgent(AbstractAgent):
    def __init__(self, config, checkpoint_path: str = None, socket_path: str = None, lr: float = 6e-4):
        super().__init__()
        self._config = config
        self._checkpoint_path = checkpoint_path  # accepted for CLI compat; weights live in the server
        self._lr = lr
        self._sock_path = socket_path or os.environ.get("TTNN_DD_SOCKET", "/tmp/ttnn_dd.sock")

    # -- required hooks --------------------------------------------------
    def name(self) -> str:
        return "diffusiondrive_ttnn_agent"

    def initialize(self) -> None:
        # No local model to load.  Fail fast if the server socket is absent.
        if not os.path.exists(self._sock_path):
            raise RuntimeError(
                f"TTNN server socket not found at {self._sock_path}. "
                f"Start scripts/ttnn_pdm_server.py in the tt-metal venv first."
            )

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_feature_builders(self) -> List[TransfuserFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    # -- inference via the TTNN server -----------------------------------
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        req = {
            "camera_feature": features["camera_feature"].detach().cpu().numpy().astype(np.float32),
            "lidar_feature": features["lidar_feature"].detach().cpu().numpy().astype(np.float32),
            "status_feature": features["status_feature"].detach().cpu().numpy().astype(np.float32),
        }
        payload = _encode_msg(req)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.connect(self._sock_path)
            c.sendall(struct.pack(">Q", len(payload)) + payload)
            (length,) = struct.unpack(">Q", _recv_exactly(c, 8))
            resp = _decode_msg(_recv_exactly(c, length))
        if "error" in resp:
            raise RuntimeError("ttnn server error:\n" + resp["error"])
        traj = torch.from_numpy(np.asarray(resp["trajectory"])).float().unsqueeze(0)  # (1, 8, 3)
        return {"trajectory": traj}
