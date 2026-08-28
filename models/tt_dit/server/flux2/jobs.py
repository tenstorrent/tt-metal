# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory job store + single-thread device executor for the FLUX.2 server.

Trimmed from tt-media-server's ``utils/job_manager.py``: kept the Job lifecycle,
in-memory registry, retention/cleanup, and a single worker that serializes device
work; dropped the database, multiprocessing scheduler, telemetry, and auth.

The serialization point is a ``ThreadPoolExecutor(max_workers=1)``: every
device-touching call (startup + every generate) runs on that one thread, which is
both the ttnn single-thread requirement and how concurrent requests are serialized.
The asyncio ``worker_loop`` awaits the executor future, so the event loop stays
responsive (health checks, status polls, downloads) during a generation.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from .config import ServerConfig
from .pipeline_holder import PipelineHolder


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


_TERMINAL = {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}


class JobLimitError(Exception):
    """Raised by ``JobStore.create`` when ``MAX_JOBS`` non-terminal jobs exist."""


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result_path: str | None = None
    error: str | None = None

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING

    def mark_completed(self, result_path: str) -> None:
        self.status = JobStatus.DONE
        self.result_path = result_path
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.ERROR
        self.error = error
        self.completed_at = time.time()

    def mark_cancelled(self) -> None:
        self.status = JobStatus.CANCELLED
        self.completed_at = time.time()

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class JobStore:
    """Owns the job registry, the device executor, and the background tasks."""

    def __init__(self, cfg: ServerConfig, holder: PipelineHolder):
        self.cfg = cfg
        self.holder = holder
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux2-device")
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    # --- registry ------------------------------------------------------------
    def create(self, params: dict[str, Any]) -> Job:
        """Register a queued job and enqueue it for the worker. Raises
        :class:`JobLimitError` if too many non-terminal jobs are outstanding."""
        with self._lock:
            active = sum(1 for j in self._jobs.values() if not j.is_terminal())
            if active >= self.cfg.max_jobs:
                raise JobLimitError(f"Too many active jobs ({active} >= MAX_JOBS={self.cfg.max_jobs})")
            job = Job(id=uuid.uuid4().hex, params=params)
            self._jobs[job.id] = job
        self._queue.put_nowait(job.id)
        logger.info(f"job {job.id}: queued")
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.to_public_dict() for j in self._jobs.values()]

    def cancel(self, job_id: str) -> tuple[bool, str]:
        """Cancel a QUEUED job. Returns ``(ok, message)``.

        Only QUEUED jobs are cancellable: ttnn generate is not interruptible, so a
        RUNNING job cannot be cancelled (caller maps that to 409).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, "not found"
            if job.status == JobStatus.QUEUED:
                job.mark_cancelled()
                logger.info(f"job {job_id}: cancelled (was queued)")
                return True, "cancelled"
            if job.status == JobStatus.RUNNING:
                return False, "running jobs cannot be cancelled"
            return False, f"job is already {job.status.value}"

    # --- lifecycle -----------------------------------------------------------
    async def start(self, holder: PipelineHolder) -> None:
        """Warm the pipeline on the executor, then start worker + cleanup tasks."""
        loop = asyncio.get_running_loop()
        logger.info("JobStore.start: warming pipeline on device executor...")
        await loop.run_in_executor(self._executor, holder.startup)
        logger.info("JobStore.start: pipeline warm; starting worker + cleanup loops.")
        self._worker_task = asyncio.create_task(self._worker_loop(), name="flux2-worker")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="flux2-cleanup")

    async def shutdown(self) -> None:
        for task in (self._worker_task, self._cleanup_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Close device on the executor thread (ttnn single-thread requirement).
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self.holder.close)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error during holder.close: {exc}")
        self._executor.shutdown(wait=True)
        logger.info("JobStore.shutdown complete.")

    # --- background loops ----------------------------------------------------
    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self._queue.get()
            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None or job.status != JobStatus.QUEUED:
                        # Cancelled before it ran, or already gone.
                        continue
                    job.mark_running()
                logger.info(f"job {job_id}: running")
                try:
                    result_path = await loop.run_in_executor(self._executor, self.holder.generate, job_id, job.params)
                    with self._lock:
                        job.mark_completed(result_path)
                    logger.info(f"job {job_id}: done -> {result_path}")
                except Exception as exc:  # noqa: BLE001 — surface as job error, keep serving
                    logger.exception(f"job {job_id}: failed")
                    with self._lock:
                        job.mark_failed(str(exc))
            finally:
                self._queue.task_done()

    async def _cleanup_loop(self) -> None:
        retention = self.cfg.job_retention_seconds
        # Sweep at a fraction of the retention window (bounded to a sane range).
        interval = max(60, min(retention, 600))
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            removed = []
            with self._lock:
                for job_id, job in list(self._jobs.items()):
                    if job.is_terminal() and job.completed_at is not None and (now - job.completed_at) > retention:
                        removed.append(job)
                        del self._jobs[job_id]
            for job in removed:
                if job.result_path and os.path.exists(job.result_path):
                    try:
                        os.remove(job.result_path)
                    except OSError as exc:
                        logger.warning(f"job {job.id}: failed to remove {job.result_path}: {exc}")
                logger.info(f"job {job.id}: reaped (older than {retention}s retention)")
