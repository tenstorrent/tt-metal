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


def test_framed_handler_concurrent_large_info_writes(tmp_path):
    import hashlib
    import os
    import sys
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "tt/native_log_formatter.py"
    writer = """
import importlib.util, logging, sys
spec=importlib.util.spec_from_file_location('native',sys.argv[1])
native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
logger=logging.getLogger('vllm.entrypoints.serve.middleware.log_response')
logger.addHandler(native.NativeFramedHandler(sys.stdout)); logger.setLevel(logging.INFO)
for i in range(12):
    logger.info('response_body={%s}', str(i)+' INFO 雪\\n'+'x'*150000)
"""
    noise = """
import sys
for i in range(150):
    sys.stdout.write('INFO '+('q'*80000)+'\\n'); sys.stdout.flush()
"""
    read_fd, write_fd = os.pipe()
    with (tmp_path / "filtered.log").open("wb") as output:
        filtering = subprocess.Popen(["grep", "-v", "-i", "INFO"], stdin=read_fd, stdout=output)
        os.close(read_fd)
        writers = [
            subprocess.Popen([sys.executable, "-u", "-c", writer, str(module_path)], stdout=write_fd),
            subprocess.Popen([sys.executable, "-u", "-c", noise], stdout=write_fd),
        ]
        os.close(write_fd)
        for process in writers:
            assert process.wait(timeout=30) == 0
        assert filtering.wait(timeout=30) == 0
    groups = {}
    for line in (tmp_path / "filtered.log").read_bytes().splitlines():
        if not line.startswith(b"TTNATIVEHEX2 "):
            continue
        assert len(line) + 2 <= 4096
        marker, checksum, index, count, size, chunk = line.decode("ascii").split()
        group = groups.setdefault(checksum, (int(count), int(size), {}))
        assert group[:2] == (int(count), int(size))
        assert int(index) not in group[2]
        group[2][int(index)] = bytes.fromhex(chunk)
    assert len(groups) == 12
    seen = set()
    for checksum, (count, size, chunks) in groups.items():
        assert set(chunks) == set(range(count))
        payload = b"".join(chunks[i] for i in range(count))
        assert len(payload) == size and hashlib.sha256(payload).hexdigest() == checksum
        envelope = json.loads(payload.decode("utf-8"))
        index = int(envelope["message"].split("{", 1)[1].split(" ", 1)[0])
        assert envelope["message"] == "response_body={" + str(index) + " INFO 雪\n" + "x" * 150000 + "}"
        seen.add(index)
    assert seen == set(range(12))


def test_framed_handler_reports_partial_write_without_retry(monkeypatch, tmp_path):
    import os

    from models.autoports.google_gemma_4_31b_it.tt.native_log_formatter import NativeFramedHandler

    calls = []
    errors = []
    with (tmp_path / "output").open("w") as stream:
        handler = NativeFramedHandler(stream)
        monkeypatch.setattr(os, "write", lambda fd, frame: calls.append(frame) or len(frame) - 1)
        monkeypatch.setattr(handler, "handleError", lambda record: errors.append(record))
        record = logging.LogRecord("native", logging.INFO, "", 1, "message", (), None)
        handler.emit(record)
    assert len(calls) == 1 and errors == [record]
