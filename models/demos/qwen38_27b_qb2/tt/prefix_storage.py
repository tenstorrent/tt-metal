# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, atomic local-file reference backend for hybrid checkpoint tests.

Not an LMCache/vLLM connector or production eviction policy. A caller-owned
directory can live on local disk or a bounded RAM filesystem. No device, NFS,
raw-disk, or system configuration is touched. Admission/eviction are explicit.
"""

import fcntl
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

MARKER = b"qwen38-checkpoint-reference-store-v1\n"


class _ExactWriter:
    def __init__(self, stream, size):
        self.stream = stream
        self.remaining = size

    def write(self, data):
        if len(data) > self.remaining:
            raise ValueError("Checkpoint exceeds its reserved byte budget")
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("Short checkpoint write")
        self.remaining -= written
        return written


class AtomicDirectoryStore:
    def __init__(self, root, *, max_bytes, create=False):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("An explicit positive storage quota is required")
        self.root, self.max_bytes = Path(root), max_bytes
        if create:
            self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
            (self.root / ".owner").write_bytes(MARKER)
        if self.root.is_symlink() or (self.root / ".owner").read_bytes() != MARKER:
            raise ValueError("Require a task-owned checkpoint directory")
        # Keep one persisted quota for all cooperating instances/processes.
        with self._lock():
            quota = self.root / ".quota"
            if quota.exists():
                if quota.read_text() != str(max_bytes):
                    raise ValueError("Storage quota differs from the existing task directory")
            else:
                quota.write_text(str(max_bytes))
            if self.used_bytes > max_bytes:
                raise ValueError("Existing payloads exceed the storage quota")

    @contextmanager
    def _lock(self):
        with (self.root / ".lock").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _path(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Storage key must be a SHA256 digest")
        return self.root / (key + ".checkpoint")

    @property
    def used_bytes(self):
        # Count incomplete crash leftovers against capacity. Never silently
        # delete them or claim disk space is free; an operator may inspect them.
        return sum(p.stat().st_size for p in self.root.iterdir() if p.suffix in (".checkpoint", ".partial"))

    def contains(self, key):
        path = self._path(key)
        return path.is_file() and not path.is_symlink()

    @contextmanager
    def read(self, key):
        path = self._path(key)
        with self._lock():
            if path.is_symlink():
                raise ValueError("Checkpoint must not be a symlink")
            # Serialize this reference backend's reads and eviction: unlinking
            # a leased inode would hide its still-allocated bytes from quota
            # accounting. A production backend needs concurrent read leases.
            with path.open("rb") as stream:
                yield stream

    @contextmanager
    def write(self, key, size):
        path = self._path(key)
        if type(size) is not int or size < 1:
            raise ValueError("Checkpoint size must be positive")
        with self._lock():
            if path.exists() or path.is_symlink():
                raise FileExistsError("Published checkpoints are immutable")
            if self.used_bytes + size > self.max_bytes or size > shutil.disk_usage(self.root).free:
                raise OSError("Checkpoint exceeds available storage budget")
            fd, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".partial", dir=self.root)
            temporary = Path(temporary)
            try:
                with os.fdopen(fd, "wb") as stream:
                    writer = _ExactWriter(stream, size)
                    yield writer
                    if writer.remaining:
                        raise ValueError("Checkpoint did not fill its reserved byte budget")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                self._sync_directory()
            finally:
                temporary.unlink(missing_ok=True)

    def discard(self, key):
        with self._lock():
            self._path(key).unlink(missing_ok=True)
            self._sync_directory()

    def _sync_directory(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
