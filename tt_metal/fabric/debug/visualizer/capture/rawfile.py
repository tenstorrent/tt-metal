# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generates a file containing raw L1 / HAL blobs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


class RawBlobWriter:
    """Append blobs to a temporary file and rename into place on close."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        self._file = self._temporary_path.open("wb")
        self._digest = hashlib.sha256()
        self._offset = 0
        self._closed = False

    def add(self, payload: bytes) -> tuple[int, int, str]:
        if self._closed:
            raise ValueError("cannot add to a closed RawBlobWriter")
        offset = self._offset
        self._file.write(payload)
        self._digest.update(payload)
        self._offset += len(payload)
        return offset, len(payload), hashlib.sha256(payload).hexdigest()

    def close(self) -> tuple[int, str]:
        if self._closed:
            return self._offset, self._digest.hexdigest()
        self._file.close()
        os.replace(self._temporary_path, self.path)
        self._closed = True
        return self._offset, self._digest.hexdigest()

    def abort(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._temporary_path.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False
