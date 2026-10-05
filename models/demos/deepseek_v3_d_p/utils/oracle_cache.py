# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side CPU oracle cache shared by every checkout on a host.

CPU oracles (references and the prepared inputs they consume) are keyed by their source identities and their
producer's explicit cache version, so their bytes do not depend on the checkout that computed them and one copy
serves every worktree. Device-format caches (TTNN ``.tensorbin`` weights) are not stored here: their bytes also
depend on the checkout's ttnn build (host tilization, mesh mapping, serialization format), which no key records,
so they stay in the checkout's ``ttnn.CONFIG.model_cache_path``.

Root: ``$TT_LINEAR_LAYERS_SHARED_CACHE`` when set; otherwise ``/localdev/<user>/.cache/tt-linear-layers-shared``
when ``/localdev/<user>`` exists (development hosts); otherwise ``ttnn.CONFIG.model_cache_path`` (CI and other
hosts, as before).

Concurrency contract: an entry appears only by atomic rename of a complete file, so lock-free readers see either
no file or a complete one. Producers of one entry serialize on a per-entry lock file and re-check under it, so
concurrent preparation from several checkouts computes each entry once. A shared root needs version bumps that
cannot collide across branches: a branch that changes what a producer stores bumps its version to a value no other
branch uses (see the producers' version comments).
"""

from __future__ import annotations

import fcntl
import getpass
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from loguru import logger

SHARED_ORACLE_CACHE_ENV = "TT_LINEAR_LAYERS_SHARED_CACHE"
_HOST_LOCAL_ROOT = Path("/localdev")
_DEFAULT_NAME = Path(".cache") / "tt-linear-layers-shared"

T = TypeVar("T")


def oracle_cache_root() -> Path:
    """Root of the shared CPU oracle cache (see the module docstring for the resolution order)."""
    configured = os.environ.get(SHARED_ORACLE_CACHE_ENV)
    if configured:
        return Path(configured)
    host_local = _HOST_LOCAL_ROOT / getpass.getuser()
    if host_local.is_dir():
        return host_local / _DEFAULT_NAME
    import ttnn

    return Path(ttnn.CONFIG.model_cache_path)


@contextmanager
def _producer_lock(path: Path) -> Iterator[None]:
    """Exclusive per-entry lock shared by every process on the host (released when the holder exits)."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(f"oracle cache: waiting for another producer of {path}")
            start = time.perf_counter()
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            logger.info(f"oracle cache: producer lock for {path} acquired after {time.perf_counter() - start:.1f} s")
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a unique temporary file in the same directory, flush it to disk, then rename it into place."""
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        write(temporary)
        with open(temporary, "rb") as written:
            os.fsync(written.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_once(
    path: Path, produce: Callable[[], T], write: Callable[[T, Path], None], load: Callable[[Path], T]
) -> tuple[T, bool]:
    """Return ``(value, produced)``: the published entry at ``path``, producing and publishing it if missing.

    ``produce`` runs at most once per entry across concurrent processes; a process that waited for another
    producer loads that producer's entry (``produced`` is False). ``write(value, file)`` serializes to ``file``.
    """
    if path.is_file():
        return load(path), False
    with _producer_lock(path):
        if path.is_file():
            return load(path), False
        value = produce()
        _write_atomically(path, lambda file: write(value, file))
    return value, True
