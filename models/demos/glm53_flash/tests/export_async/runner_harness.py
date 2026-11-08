# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the async-export device check (tt-blaze G53F-0309): the REAL common prefill runner
(``prefill_runner.main``, H2D socket, layer-ack router, KV export to file) with the GLM runtime, plus TEST-ONLY patches
that switch the export mode per prompt, so one model build (~29 min) measures every mode on the same programs:

    EXPORT_CHECK_MODE_FILE holds the mode of the NEXT prompt (read when a chunk at start 0 arrives):
      sync [digest]       the default synchronous export
      async [digest]      GLM53F_EXPORT_ASYNC=pack with the socket guard
      unguarded           async with the guard bypassed: the pre-G53F-0309 code path (the hang reproducer)
    "digest" sets GLM53F_EXPORT_DIGEST=1 for that prompt (crc32 of every byte the export writes).

SIGUSR1 dumps every Python thread's stack to EXPORT_CHECK_STACKS (faulthandler: no GIL needed). Launch with
GLM53F_EXPORT_ASYNC=pack so the worker thread and the socket wrapper exist; driver.py drives it.
"""

from __future__ import annotations

import faulthandler
import os
import runpy
import signal

from loguru import logger

from models.demos.glm53_flash.tt.runners import runtime as R
from models.demos.glm53_flash.tt.runners.export_guard import ExportDrainGuard

_STACKS = open(os.environ.get("EXPORT_CHECK_STACKS", "/tmp/export_check_stacks.txt"), "a")
faulthandler.register(signal.SIGUSR1, file=_STACKS, all_threads=True, chain=False)
_MODE_FILE = os.environ["EXPORT_CHECK_MODE_FILE"]


def _mode() -> list:
    try:
        words = open(_MODE_FILE).read().split()
    except FileNotFoundError:
        words = []
    return words or ["sync"]


_prefill_chunk = R.GLM53FlashPrefillRuntime.prefill_chunk


def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **kw):
    if actual_start == 0 and not self._warming:
        words = _mode()
        want_async = words[0] in ("async", "unguarded")
        if want_async != self._async:
            self._xq.join()
            self._async = want_async
        if "digest" in words:
            os.environ["GLM53F_EXPORT_DIGEST"] = "1"
        else:
            os.environ.pop("GLM53F_EXPORT_DIGEST", None)
        logger.info(f"[export-check] chunk {request_id}: prompt mode {' '.join(words)} (async {self._async})")
    return _prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        slot_id=slot_id,
        actual_start=actual_start,
        actual_end=actual_end,
        request_id=request_id,
        **kw,
    )


_before_socket_wait = ExportDrainGuard.before_socket_wait


def before_socket_wait(self):
    if _mode()[0] == "unguarded":
        return "bypassed"
    return _before_socket_wait(self)


R.GLM53FlashPrefillRuntime.prefill_chunk = prefill_chunk
ExportDrainGuard.before_socket_wait = before_socket_wait
logger.info(f"[export-check] harness up: pid {os.getpid()}, mode file {_MODE_FILE}")
runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__", alter_sys=True)
