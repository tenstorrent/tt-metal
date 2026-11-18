# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The text a request that failed for good is answered with, and its test.

models/mimo_openrouter.py gives such a request ``[MIMO_REQUEST_FAILED <status>] <error>`` as its response text, so
the harness still gets a string for it. That text must never reach a task's answer extraction: the extractors look
for a letter anywhere (Artificial Analysis's last fallback takes the ``D`` of ``FAILED``; Seldon's takes the first
standalone letter of the error body), so about one failure in N options would be scored correct. The tasks of this
plugin call ``request_failed`` first and score such a response wrong; compare_runs.py does the same for the tasks
that come from lmms-eval itself, from the journal (failures are never journaled)."""

FAILED_PREFIX = "[MIMO_REQUEST_FAILED"


def failed_text(status, error):
    return f"{FAILED_PREFIX} {status}] {error}"


def request_failed(response):
    """Whether a response text is the marker of a request that failed for good."""
    return isinstance(response, str) and response.lstrip().startswith(FAILED_PREFIX)
