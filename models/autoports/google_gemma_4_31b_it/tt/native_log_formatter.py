# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Lossless native serving records for transports that filter text by severity."""

import hashlib
import json
import logging
import os

MARKER = "TTNATIVEHEX1 "


class NativeHexFormatter(logging.Formatter):
    """Encode only explicit log fields, never arbitrary LogRecord extras."""

    def format(self, record):
        payload = {
            "version": 1,
            "name": record.name,
            "level": record.levelname,
            "created": record.created,
            "message": record.getMessage(),
        }
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return MARKER + data.encode("utf-8").hex()


class NativeFramedHandler(logging.StreamHandler):
    """Keep each native frame atomic against other writers to the output pipe."""

    def emit(self, record):
        try:
            fd = self.stream.fileno()
            limit = min(4096, os.fpathconf(fd, "PC_PIPE_BUF"))
            # Reserve room for the marker, SHA256, decimal fields and delimiters.
            chunk_bytes = (limit - 160) // 2
            if chunk_bytes < 64:
                raise ValueError("Native log pipe atomic-write limit is too small")
            encoded = NativeHexFormatter().format(record)[len(MARKER) :]
            payload = bytes.fromhex(encoded)
            checksum = hashlib.sha256(payload).hexdigest()
            count = (len(payload) + chunk_bytes - 1) // chunk_bytes
            self.stream.flush()
            for index in range(count):
                chunk = payload[index * chunk_bytes : (index + 1) * chunk_bytes]
                # A preceding partial INFO line cannot absorb this frame: the
                # leading newline and the complete frame are one atomic write.
                frame = (f"\nTTNATIVEHEX2 {checksum} {index} {count} {len(payload)} {chunk.hex()}\n").encode("ascii")
                if len(frame) > limit:
                    raise ValueError("Native log frame exceeds pipe atomic-write limit")
                if os.write(fd, frame) != len(frame):
                    raise OSError("Incomplete native log frame write")
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)
