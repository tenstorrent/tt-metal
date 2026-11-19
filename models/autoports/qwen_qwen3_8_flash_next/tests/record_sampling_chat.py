# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Optional pytest observer for unchanged shared chat sampling requests.

Load with ``-p models.autoports.qwen_qwen3_8_flash_next.tests.record_sampling_chat``
and ``--sampling-chat-record-dir``. This records SDK request arguments and complete
nonstreamed responses without changing requests, response objects, or assertions.
"""

import functools
import itertools
import json
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--sampling-chat-record-dir", default=None)


def pytest_configure(config):
    destination = config.getoption("--sampling-chat-record-dir")
    if destination is None:
        return
    from openai.resources.chat.completions import AsyncCompletions

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    original = AsyncCompletions.create
    sequence = itertools.count()

    @functools.wraps(original)
    async def recorded(self, *args, **kwargs):
        index = next(sequence)
        path = output / f"request_{index:04d}.json"
        record = {"request": kwargs, "positional_arguments": args}
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            record["error"] = repr(exc)
            path.write_text(json.dumps(record, indent=2, default=str) + "\n")
            raise
        if hasattr(response, "model_dump"):
            record["response"] = response.model_dump(mode="json")
        else:
            record["response_kind"] = type(response).__name__
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        return response

    AsyncCompletions.create = recorded
    config._sampling_chat_original = original


def pytest_unconfigure(config):
    original = getattr(config, "_sampling_chat_original", None)
    if original is not None:
        from openai.resources.chat.completions import AsyncCompletions

        AsyncCompletions.create = original
