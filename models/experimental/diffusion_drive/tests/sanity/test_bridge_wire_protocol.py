# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wire-protocol tests for the NavSim socket bridge.

The bridge frames messages as an 8-byte big-endian length prefix followed by a
numpy ``.npz`` payload.  It must never be pickle: the server holds the device and
deserialises whatever the harness sends it, and the pickle loader executes code
found in its input.

Covers the shapes actually exchanged — feature request, trajectory reply, and the
``cmd``/``ok``/``error`` scalars — plus the negative case that a pickle payload is
rejected rather than executed.  No device needed.
"""

from __future__ import annotations

import importlib.util
import socket
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

_SERVER = Path(__file__).resolve().parent.parent.parent / "scripts" / "ttnn_pdm_server.py"

# A real protocol-4 pickle frame of {"camera_feature": [1, 2, 3]}, embedded as a literal
# so this file never imports pickle to build one. Regenerate with:
#   python -c 'import pickle;print(pickle.dumps({"camera_feature":[1,2,3]},protocol=4))'
_PICKLE_FRAME = b"\x80\x04\x95\x1f\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x0ecamera_feature\x94]\x94(K\x01K\x02K\x03es."


def _load_server_module():
    """Import ttnn_pdm_server for its codec only (module import pulls in no ttnn)."""
    spec = importlib.util.spec_from_file_location("dd_pdm_server", _SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def srv():
    return _load_server_module()


def test_server_module_has_no_pickle_codec(srv) -> None:
    src = _SERVER.read_text()
    assert "import pickle" not in src
    assert "pickle.loads" not in src and "pickle.dumps" not in src


def test_roundtrip_feature_request(srv) -> None:
    req = {
        "camera_feature": np.random.rand(1, 3, 256, 1024).astype(np.float32),
        "lidar_feature": np.random.rand(1, 1, 256, 256).astype(np.float32),
        "status_feature": np.random.rand(1, 8).astype(np.float32),
    }
    out = srv._decode_msg(srv._encode_msg(req))
    assert set(out) == set(req)
    for k, v in req.items():
        assert out[k].dtype == v.dtype
        np.testing.assert_array_equal(out[k], v)


def test_roundtrip_trajectory_reply(srv) -> None:
    traj = np.random.rand(8, 3).astype(np.float32)
    out = srv._decode_msg(srv._encode_msg({"trajectory": traj}))
    np.testing.assert_array_equal(out["trajectory"], traj)


@pytest.mark.parametrize(
    "msg,key,expected",
    [
        ({"cmd": "shutdown"}, "cmd", "shutdown"),
        ({"ok": True}, "ok", True),
        ({"error": "Traceback...\nboom"}, "error", "Traceback...\nboom"),
    ],
)
def test_scalars_survive_as_python_types(srv, msg, key, expected) -> None:
    """The serve loop compares these with ==/+ , so they must not stay 0-d arrays."""
    out = srv._decode_msg(srv._encode_msg(msg))
    assert out[key] == expected
    assert type(out[key]) is type(expected)


def test_pickle_payload_is_rejected_not_executed(srv) -> None:
    """A pickle frame must be refused by the loader, not unpickled.

    The expect_error fixture is deliberately not used here: it matches against real
    device error text (the TT_FATAL line) for CI triage, and pulling it in would make
    this device-free test import ttnn via the root conftest.
    """
    with pytest.raises(ValueError, match="allow_pickle=False"):  # allow-pytest.raises: pure-Python codec check
        srv._decode_msg(_PICKLE_FRAME)


def test_roundtrip_over_a_real_socket(srv, tmp_path) -> None:
    sock_path = str(tmp_path / "wire.sock")
    reply = np.arange(24, dtype=np.float32).reshape(8, 3)
    seen = {}

    lsn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lsn.bind(sock_path)
    lsn.listen(1)

    def _serve():
        conn, _ = lsn.accept()
        with conn:
            seen["req"] = srv._recv_msg(conn)
            srv._send_msg(conn, {"trajectory": reply})

    t = threading.Thread(target=_serve)
    t.start()
    try:
        payload = srv._encode_msg({"status_feature": np.ones((1, 8), dtype=np.float32)})
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.connect(sock_path)
            c.sendall(struct.pack(">Q", len(payload)) + payload)
            (length,) = struct.unpack(">Q", srv._recv_exactly(c, 8))
            resp = srv._decode_msg(srv._recv_exactly(c, length))
    finally:
        t.join(timeout=10)
        lsn.close()

    np.testing.assert_array_equal(seen["req"]["status_feature"], np.ones((1, 8), dtype=np.float32))
    np.testing.assert_array_equal(resp["trajectory"], reply)
