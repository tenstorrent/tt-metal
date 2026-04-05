# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import difflib
import json
from dataclasses import asdict
from typing import Any

from helpers.logger import logger
from ttexalens.tt_exalens_lib import (
    get_tensix_state,
)


class TensixState:
    @classmethod
    def fetch(cls, location: str) -> dict[str, Any]:
        return asdict(get_tensix_state(location, device_id=0))

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
        msg = f"Assertion FAILED: Tensix state mismatch:\n{''.join(diff)}"
        logger.error(msg)
        raise AssertionError(msg)
