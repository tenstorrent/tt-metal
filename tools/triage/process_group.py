#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Process-group abstraction: identity + collectives, swappable per launcher.

`LocalProcessGroup` is the single-process default; `MPIProcessGroup` wraps a
real communicator under `tt-run`/`mpirun`."""

from __future__ import annotations

import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any


class ProcessGroup(ABC):
    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def gather(self, payload: Any) -> list[Any] | None:
        ...

    @abstractmethod
    def barrier(self) -> None:
        ...

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def is_multi(self) -> bool:
        return self.size > 1

    def report_fatal(self) -> None:
        """Handle the currently-being-handled exception. Default: re-raise
        (Python's default handler prints the traceback and exits non-zero)."""
        raise

    def shutdown(self) -> None:
        """Tear down launcher-specific state before process exit. Default no-op."""


class LocalProcessGroup(ProcessGroup):
    @property
    def rank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    def gather(self, payload: Any) -> list[Any] | None:
        return [payload]

    def barrier(self) -> None:
        pass


class MPIProcessGroup(ProcessGroup):
    """mpi4py-backed; `rank`/`size` from the wrapped communicator. Exposes
    the underlying `comm` so MPIScriptRunner can do async point-to-point."""

    def __init__(self, comm: Any) -> None:
        self._comm = comm
        self._rank = int(comm.Get_rank())
        self._size = int(comm.Get_size())

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def comm(self) -> Any:
        return self._comm

    def gather(self, payload: Any) -> list[Any] | None:
        parts = self._comm.gather(payload, root=0)
        return parts if self._rank == 0 else None

    def barrier(self) -> None:
        self._comm.Barrier()

    def report_fatal(self) -> None:
        # PRRTE cascade-kill firewall: log + return so caller can exit 0 cleanly.
        sys.stderr.write(f"[rank {self._rank}] tt-triage crashed:\n{traceback.format_exc()}\n")
        sys.stderr.flush()

    def shutdown(self) -> None:
        # `os._exit(0)` bypasses atexit, so mpi4py's auto-finalize never runs and
        # PRRTE flags us as improperly terminated. Finalize explicitly.
        from mpi4py import MPI

        if not MPI.Is_finalized():
            MPI.Finalize()


_active: ProcessGroup | None = None


def make_process_group() -> ProcessGroup:
    """Pick MPI when launched under `tt-run` / `mpirun` with size>1, else Local."""
    global _active
    if _active is not None:
        return _active
    try:
        from mpi4py import MPI

        if int(MPI.COMM_WORLD.Get_size()) > 1:
            _active = MPIProcessGroup(MPI.COMM_WORLD)
            return _active
    except Exception:
        pass
    _active = LocalProcessGroup()
    return _active


def get_process_group() -> ProcessGroup:
    return make_process_group()
