# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Transport checks do not import the model or access devices."""

import json
import logging
import subprocess

from models.autoports.google_gemma_4_31b_it.tt.native_log_formatter import MARKER, NativeHexFormatter


def test_exact_ci_filter_roundtrip():
    message = "Received request INFO-id: arbitrary InFo\n雪 café 🐍\x00" + "".join(map(chr, range(256)))
    record = logging.LogRecord("vllm.INFO.metadata", logging.INFO, "private-path", 9, "%s", (message,), None)
    record.credential_not_logged = "EXTRA_SECRET_SENTINEL"
    encoded = NativeHexFormatter().format(record)
    assert "info" not in encoded.lower()
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", 'grep -v -i "INFO" | sed "s/^/[container] /"'],
        input=encoded + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(bytes.fromhex(result.stdout.split(MARKER, 1)[1].strip()).decode("utf-8"))
    assert data == {"version": 1, "name": record.name, "level": "INFO", "created": record.created, "message": message}
    assert "EXTRA_SECRET_SENTINEL" not in json.dumps(data)
    assert record.msg == "%s" and record.args == (message,)
