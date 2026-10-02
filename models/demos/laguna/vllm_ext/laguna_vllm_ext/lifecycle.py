# SPDX-License-Identifier: Apache-2.0
"""Graceful Laguna adapter cleanup for the public TT worker lifecycle.

vLLM calls ``WorkerBase.shutdown`` during an orderly engine stop.  The pinned TT
worker inherits that no-op and closes its mesh later from ``__del__``.  Laguna
owns TT traces that must be released while the mesh is still open, so this
model-local plugin wraps both paths: shutdown performs the normal close, and the
destructor is a best-effort fallback before the TT worker closes the mesh.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

ADAPTER_MARKER = "_LAGUNA_VLLM_ADAPTER"
WORKER_PATCH_MARKER = "_laguna_adapter_lifecycle_patch"


def _laguna_adapter(worker: Any) -> Any | None:
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None)
    if not bool(getattr(model, ADAPTER_MARKER, False)):
        return None
    return model


def _close_laguna_adapter(worker: Any) -> bool:
    model = _laguna_adapter(worker)
    if model is None:
        return False
    close = getattr(model, "close", None)
    if not callable(close):
        raise RuntimeError("Laguna vLLM adapter does not expose close()")
    close()
    return True


def _patch_worker_lifecycle(worker_class: type) -> bool:
    """Install an idempotent close-before-mesh wrapper on one TT worker class."""

    if worker_class.__dict__.get(WORKER_PATCH_MARKER, False):
        return False

    original_shutdown = worker_class.shutdown
    original_del = getattr(worker_class, "__del__", None)

    @functools.wraps(original_shutdown)
    def shutdown(self: Any) -> Any:
        _close_laguna_adapter(self)
        return original_shutdown(self)

    worker_class.shutdown = shutdown

    if original_del is not None:

        @functools.wraps(original_del)
        def __del__(self: Any) -> Any:
            try:
                _close_laguna_adapter(self)
            except Exception:
                # Destructors cannot reliably propagate an exception, but still
                # run the TT worker's mesh close so the device is not stranded.
                logger.exception("Laguna adapter cleanup failed before TT mesh close")
            return original_del(self)

        worker_class.__del__ = __del__

    setattr(worker_class, WORKER_PATCH_MARKER, True)
    return True


def install_worker_lifecycle_patch() -> bool:
    from vllm_tt_plugin.worker import TTWorker

    installed = _patch_worker_lifecycle(TTWorker)
    if installed:
        logger.info("Installed Laguna close-before-mesh TT worker lifecycle")
    return installed


def worker_lifecycle_patch_is_installed() -> bool:
    from vllm_tt_plugin.worker import TTWorker

    return bool(TTWorker.__dict__.get(WORKER_PATCH_MARKER, False))


__all__ = [
    "install_worker_lifecycle_patch",
    "worker_lifecycle_patch_is_installed",
]
