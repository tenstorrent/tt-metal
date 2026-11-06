# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Lossless native serving records for transports that filter text by severity."""

import json
import logging

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
