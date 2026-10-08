# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import difflib
import json
from dataclasses import asdict
from typing import Any

from helpers import device as device_module
from helpers.device import commit_brisc_command, get_register_store
from helpers.device_io import read_words_from_device
from helpers.llk_params import BriscCmd
from helpers.logger import logger
from helpers.test_config import TestConfig
from ttexalens.register_store import TensixGeneralPurposeRegisterDescription
from ttexalens.tt_exalens_lib import (
    get_tensix_state,
)

GPRS_PER_THREAD = 64
TENSIX_THREADS = 3
GPR_DUMP_WORDS = GPRS_PER_THREAD * TENSIX_THREADS

# Per thread (unpack, math, pack): GPRs ckernel_gpr_map.h designates as temporaries.
# They hold path-dependent intermediates, not configuration state.
SCRATCH_GPRS = (
    frozenset({12, 13, 18, 19}),  # TMP0, TMP1, TMP_LO, TMP_HI
    frozenset({60}),  # TMP0
    frozenset({20, 28, 29, 30, 31}),  # TEMP_TILE_OFFSET, TMP0, TMP1, TMP_LO, TMP_HI
)


class TensixState:
    @classmethod
    def fetch(cls, location: str) -> dict[str, Any]:
        state = asdict(get_tensix_state(location, device_id=0))
        names = cls._gpr_names(location)
        if any(state["gpr"]):
            values = {
                (thread, index): state["gpr"][thread][name]
                for (thread, index), name in names.items()
            }
        else:
            values = cls._dump_gprs(location)
        state["gpr"] = [
            {
                names[(thread, index)]: values[(thread, index)]
                for index in range(GPRS_PER_THREAD)
                if index not in SCRATCH_GPRS[thread]
            }
            for thread in range(TENSIX_THREADS)
        ]
        return state

    @classmethod
    def _gpr_names(cls, location: str) -> dict[tuple[int, int], str]:
        store = get_register_store(location, 0)
        names = {}
        for register in store.get_register_names():
            description = store.get_register_description(register)
            if isinstance(description, TensixGeneralPurposeRegisterDescription):
                # ttexalens names are "<NAME>_T<thread>"; its TensixState keys drop the suffix.
                names[(description.thread_id, description.index)] = register[
                    :-3
                ].lower()
        return names

    @classmethod
    def _dump_gprs(cls, location: str) -> dict[tuple[int, int], int]:
        # ttexalens reads GPRs by halting BRISC, which hangs it; BRISC copies them to L1 instead.
        commit_brisc_command(
            location,
            BriscCmd.DUMP_GPRS,
            timeout=(
                TestConfig.SIMULATOR_TIMEOUT
                if TestConfig.TEST_TARGET.run_simulator
                else 1
            ),
        )
        words = read_words_from_device(
            location,
            device_module.Mailboxes.Unpacker.value - GPR_DUMP_WORDS * 4,
            word_count=GPR_DUMP_WORDS,
        )
        return {
            (thread, index): words[thread * GPRS_PER_THREAD + index]
            for thread in range(TENSIX_THREADS)
            for index in range(GPRS_PER_THREAD)
        }

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

    @classmethod
    def assert_not_equal(cls, left: dict, right: dict) -> None:
        if left != right:
            return

        msg = "Assertion FAILED: Tensix states are equal but were expected to differ"
        logger.error(msg)
        raise AssertionError(msg)
