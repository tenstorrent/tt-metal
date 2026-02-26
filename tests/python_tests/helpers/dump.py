# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import difflib
import json
from dataclasses import asdict
from enum import IntEnum
from typing import Any, ClassVar

from helpers.logger import logger
from ttexalens.tt_exalens_lib import (
    get_tensix_state,
    read_words_from_device,
    write_words_to_device,
)


class TensixDump:

    TENSIX_DUMP_MAILBOX_ADDRESS: ClassVar[int] = 0x16AFE4

    class MailboxState(IntEnum):
        DONE = 0
        REQUESTED = 1

    @classmethod
    def initialize(cls, location: str) -> None:
        initial = [cls.MailboxState.DONE, cls.MailboxState.DONE, cls.MailboxState.DONE]
        write_words_to_device(location, cls.TENSIX_DUMP_MAILBOX_ADDRESS, initial)

    @classmethod
    def try_process_request(cls, dumps: list[Any], location: str):
        is_requested = cls._try_receive_request(location)

        if not is_requested:
            return False

        dumps.append(cls._fetch_state(location))
        cls._send_done(location)

        return True

    @classmethod
    def _try_receive_request(cls, location: str):
        all_requested = [
            cls.MailboxState.REQUESTED,
            cls.MailboxState.REQUESTED,
            cls.MailboxState.REQUESTED,
        ]
        mailbox = read_words_from_device(
            location, cls.TENSIX_DUMP_MAILBOX_ADDRESS, word_count=3
        )
        return mailbox == all_requested

    @classmethod
    def _fetch_state(cls, location: str) -> dict[str, Any]:
        return asdict(get_tensix_state(location, device_id=0))

    @classmethod
    def _send_done(cls, location: str):
        done = [cls.MailboxState.DONE, cls.MailboxState.DONE, cls.MailboxState.DONE]
        write_words_to_device(location, cls.TENSIX_DUMP_MAILBOX_ADDRESS, done)

    @classmethod
    def format_state(cls, state: dict) -> str:
        return json.dumps(state, indent=4)

    @classmethod
    def assert_equal(cls, left: dict, right: dict) -> None:
        if left == right:
            return

        left_lines = cls.format_state(left).splitlines(keepends=True)
        right_lines = cls.format_state(right).splitlines(keepends=True)
        diff = difflib.unified_diff(
            left_lines,
            right_lines,
            fromfile="left",
            tofile="right",
            n=max(
                len(left_lines), len(right_lines)
            ),  # sstanisic todo: better way to force full diff ?
        )
        msg = f"Assertion FAILED: Tensix dump mismatch:\n{''.join(diff)}"
        logger.opt(exception=True).error(msg)
        raise AssertionError(msg)
