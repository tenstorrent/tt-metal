# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read buffered host events from the owned serving process without device calls."""

import json
import socket


def query(path, command="status"):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(30)
        conn.connect(str(path))
        conn.sendall(command.encode() + b"\n")
        chunks = []
        while chunk := conn.recv(1048576):
            chunks.append(chunk)
    return json.loads(b"".join(chunks))
