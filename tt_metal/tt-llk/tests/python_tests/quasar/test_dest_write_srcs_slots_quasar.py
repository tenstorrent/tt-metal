# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    TILE_COUNT,
    TemplateParameter,
)
from quasar.test_dest_write_path_quasar import (
    DEST_WRITE_PATH,
    DestWriteKernel,
    DestWritePath,
    _assert_bit_exact,
    _stamps_buffer,
    report_bank_stamps,
)


@dataclass
class SLOT_CONFIG(TemplateParameter):
    slot_count: int
    slot_two_pass: bool
    slot_per_instr: int = 1
    slot_stream_once: bool = False
    slot_sfpu_only: bool = False
    slot_copy_mode: int = 0

    def convert_to_cpp(self):
        return (
            f"constexpr unsigned SLOT_COUNT = {self.slot_count};\nconstexpr bool SLOT_TWO_PASS = {str(self.slot_two_pass).lower()};"
            f"\nconstexpr unsigned SLOT_PER_INSTR = {self.slot_per_instr};"
            f"\nconstexpr bool SLOT_STREAM_ONCE = {str(self.slot_stream_once).lower()};"
            f"\nconstexpr bool SLOT_SFPU_ONLY = {str(self.slot_sfpu_only).lower()};"
            f"\nconstexpr unsigned SLOT_COPY_MODE = {self.slot_copy_mode};"
        )


# SLOT_COPY_MODE values (dest_write_srcs_slots_quasar_test.cpp).
COPY_DRAIN, COPY_DONE, COPY_LOADMACRO = 0, 1, 2

# Two-pass capture calibration, clean division (48 slices), and remainder (64 slices).
CASES = [
    (n, tiles, two)
    for tiles, two in [(1, True), (3, False), (4, False)]
    for n in [1, 2, 3]
]
IDS = [f"slots{n}-tiles{t}-{'two_pass' if two else 'stream'}" for n, t, two in CASES]

# Configure-once streams (no Unpacker 2 drain per bank): slots per hand-off at 48 slices, and one UNPACR2 per
# hand-off through a 4 * slots-row descriptor. (slots, slots per UNPACR2, tiles per bank).
# One UNPACR2 across all three slots (3, 3, 3) is not a legal tile shape: U2 drops it and the SFPU waits forever.
CAUSE_CASES = [
    (1, 1, 3),
    (2, 1, 3),
    (3, 1, 3),
    (2, 2, 3),
    (1, 1, 4),
    (2, 1, 4),
    (2, 2, 4),
]
CAUSE_IDS = [f"slots{n}-instr{n // per}-tiles{t}" for n, per, t in CAUSE_CASES]


def run_slots(
    slots,
    tiles,
    two_pass,
    *,
    perf=False,
    perf_report=None,
    per_instr=1,
    stream_once=False,
    sfpu_only=False,
    writer_only=False,
    copy_mode=COPY_DRAIN,
):
    blocks = 2 if two_pass else 16
    loops = 1 if two_pass else 4
    count = blocks * tiles
    # Unique exact finite FP32 bit patterns expose skipped, repeated and stale slices. Each copy mode gets its own
    # base, so Dest rows a broken mode never writes cannot still hold a matching value from an earlier run.
    src = (
        torch.arange(count * 1024, dtype=torch.int32) + 0x3F000000 + (copy_mode << 20)
    ).view(torch.float32)
    stimuli = StimuliConfig(
        src,
        DataFormat.Float32,
        torch.zeros(1024),
        DataFormat.Float32,
        DataFormat.Float32,
        tile_count_A=count,
        tile_count_B=1,
        tile_count_res=count,
        buffer_C=_stamps_buffer(),
        stimuli_C_format=DataFormat.Int32,
        tile_count_C=1,
        num_faces=4,
    )
    config = create_test_or_perf_config(
        is_perf=perf,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs={
            "test_name": "sources/quasar/dest_write_srcs_slots_quasar_test.cpp",
            "formats": InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
            "templates": [
                DEST_SYNC(DestSync.Half),
                DEST_WRITE_PATH(
                    dwp_path=DestWritePath.B_DRAIN,
                    dwp_tiles_per_bank=tiles,
                    dwp_num_blocks=blocks,
                    dwp_loop_factor=loops,
                    dwp_srcs_addr_mod_step=True,
                    dwp_writer_only=writer_only,
                ),
                SLOT_CONFIG(
                    slots, two_pass, per_instr, stream_once, sfpu_only, copy_mode
                ),
            ],
            "runtimes": [LOOP_FACTOR(loops), TILE_COUNT(count)],
            "variant_stimuli": stimuli,
            "unpack_to_dest": False,
            "disable_format_inference": True,
            "dest_acc": DestAccumulation.Yes,
        },
    )
    if perf:
        config.run(perf_report)
    else:
        result = torch.tensor(config.run().result, dtype=torch.float32)
        assert result.numel() == src.numel()
        # SFPU-only copies input slice 0 (64 datums) into every Dest slice.
        golden = src[:64].repeat(count * 16) if sfpu_only else src
        if writer_only:
            golden = golden.view(blocks, tiles, 1024)[:, 0, :256].flatten()
            result = result[: golden.numel()]
        if copy_mode == COPY_LOADMACRO:
            import os

            torch.save(
                {"result": result, "golden": golden},
                os.path.join(os.environ["RUNNER_TEMP"], "macro-output.pt"),
            )
        _assert_bit_exact(
            golden,
            result,
            f"slots={slots} per_instr={per_instr} once={stream_once} "
            f"sfpu_only={sfpu_only} copy_mode={copy_mode} writer_only={writer_only} tiles={tiles} "
            f"two_pass={two_pass}",
        )
    srcs_opt = (
        "sfpu-only"
        if sfpu_only
        else ("multi-slot-once" if stream_once else "multi-slot")
    )
    report_bank_stamps(
        stimuli,
        DestWriteKernel.Copy,
        DestWritePath.B_DRAIN,
        tiles,
        DataFormat.Float32,
        perf,
        experimental=True,
        slots=slots,
        two_pass=two_pass,
        slot_per_instr=per_instr,
        slot_stream_once=stream_once,
        slot_sfpu_only=sfpu_only,
        slot_copy_mode=copy_mode,
        loop_factor=loops,
        num_blocks=blocks,
        stamp_sync=False,
        stamp_every=1,
        out_format="Float32",
        srcs_opt=srcs_opt,
        writer_only=writer_only,
    )


@pytest.mark.quasar
@pytest.mark.parametrize("slots,tiles,two_pass", CASES, ids=IDS)
def test_srcs_slots(slots, tiles, two_pass):
    run_slots(slots, tiles, two_pass)


@pytest.mark.quasar
@pytest.mark.parametrize("slots,per_instr,tiles", CAUSE_CASES, ids=CAUSE_IDS)
def test_srcs_slots_stream_once(slots, per_instr, tiles):
    run_slots(slots, tiles, False, per_instr=per_instr, stream_once=True)


# SFPU copy with and without U2 feeding it, both with the one-face pack: (sfpu_only, tiles per bank).
ISOLATE_CASES = [(sfpu_only, tiles) for tiles in (3, 4) for sfpu_only in (False, True)]
ISOLATE_IDS = [f"{'sfpu_only' if s else 'u2_fed'}-tiles{t}" for s, t in ISOLATE_CASES]


@pytest.mark.quasar
@pytest.mark.parametrize("sfpu_only,tiles", ISOLATE_CASES, ids=ISOLATE_IDS)
def test_srcs_isolate_sfpu(sfpu_only, tiles):
    run_slots(
        1,
        tiles,
        False,
        stream_once=not sfpu_only,
        sfpu_only=sfpu_only,
        writer_only=True,
    )


# Plain SFPLOAD/SFPSTORE copy vs SFPLOADMACRO copy, both handing each bank over with the done store, with and without
# U2 feeding the SFPU, at four tiles per bank: (sfpu_only, copy mode).
MACRO_CASES = [
    (sfpu_only, mode)
    for sfpu_only in (True, False)
    for mode in (COPY_DONE, COPY_LOADMACRO)
]
MACRO_IDS = [
    f"{'sfpu_only' if s else 'u2_fed'}-{'loadmacro' if m == COPY_LOADMACRO else 'plain'}"
    for s, m in MACRO_CASES
]


# The bit-exact check packs every face; a misplaced macro store can land anywhere in the bank.
@pytest.mark.quasar
@pytest.mark.parametrize("sfpu_only,copy_mode", MACRO_CASES, ids=MACRO_IDS)
def test_srcs_loadmacro(sfpu_only, copy_mode):
    run_slots(
        1, 4, False, stream_once=not sfpu_only, sfpu_only=sfpu_only, copy_mode=copy_mode
    )
