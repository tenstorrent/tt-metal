# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# lane FS  FP-3 architectural model: controlled cross-invocation replay-slot
# persistence experiments on silicon.  See laneFS-evidence-20260822/.
#
# EXP-1 (persistence across reset+reload): kernel A records a distinctive
# 2-word payload into replay slots 0..1 with a NO-EXEC record then exits
# (never launches).  Kernel B (separate ELF, separate run_elf_files => TRISC
# soft reset + reload between) launches slot 0 without recording, then copies
# DEST tile 0 back to L1.  If B's readback contains the sentinel 0x0000ABCD,
# the per-thread Replay-Expander buffer PERSISTED across the invocation
# boundary.
#
# EXP-2 (within-launch cross-function reassembly): one kernel launch, record
# in one basic block and launch in a later block separated by opaque control
# flow (models FP-3's pfj1 sibling-arm shape at the hardware-stream level).

import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig

SENTINEL = 0x0000ABCD


def _make_config(source, formats):
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    return TestConfig(
        source,
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )


def _skip_if_compile_only():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)


def _run_no_result(cfg):
    """Build + run a kernel that produces no readable result (kernel A)."""
    cfg.prepare()
    _skip_if_compile_only()
    cfg.write_runtimes_to_L1()
    if cfg.variant_stimuli:
        cfg.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished()


def _run_read_result(cfg):
    """Build + run a kernel and return its DEST->L1 readback (kernel B)."""
    cfg.prepare()
    _skip_if_compile_only()
    cfg.write_runtimes_to_L1()
    if cfg.variant_stimuli:
        cfg.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
        cfg.variant_stimuli.clear_result_buffer(TestConfig.TENSIX_LOCATION)
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished()
    return cfg.variant_stimuli.collect_results(TestConfig.TENSIX_LOCATION)


@parametrize(formats=input_output_formats([DataFormat.Int32], same=True))
def test_inline_ctrl_vehicle(formats):
    """Vehicle control: execute the payload inline (no replay), read DEST."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("RISC-DEST debug window is Blackhole-only.")
    formats = formats[0]
    cfg = _make_config("sources/replay_state_inline_ctrl.cpp", formats)
    res = _run_read_result(cfg)
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res, dtype=torch_format)
    n_sentinel = int((res_tensor.view(torch.int32) == SENTINEL).sum().item())
    print(
        f"\n[FS CTRL] sentinel 0x{SENTINEL:08X} count in INLINE DEST readback: "
        f"{n_sentinel} / {res_tensor.numel()}"
    )
    print(f"[FS CTRL] first 16 words: {[hex(int(x) & 0xffffffff) for x in res_tensor.view(torch.int32)[:16]]}")
    assert n_sentinel > 0, "Inline store/read vehicle is broken (no sentinel)."


@parametrize(formats=input_output_formats([DataFormat.Int32], same=True))
def test_replay_persist_across_reset(formats):
    """EXP-1: does a fresh kernel launch inherit the previous kernel's slots?"""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("RISC-DEST debug window is Blackhole-only.")
    formats = formats[0]

    cfg_a = _make_config("sources/replay_state_record_a.cpp", formats)
    cfg_b = _make_config("sources/replay_state_launch_b.cpp", formats)

    _run_no_result(cfg_a)  # record slots 0..1, no launch, exit
    res = _run_read_result(cfg_b)  # launch slot 0 (no record), read DEST

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res, dtype=torch_format)
    n_sentinel = int((res_tensor.view(torch.int32) == SENTINEL).sum().item())

    print(
        f"\n[FS EXP-1] sentinel 0x{SENTINEL:08X} count in kernel-B DEST readback: "
        f"{n_sentinel} / {res_tensor.numel()}"
    )
    print(f"[FS EXP-1] first 16 words: {[hex(int(x)) for x in res_tensor.view(torch.int32)[:16]]}")

    # This assert PROBES; either outcome is a recorded finding, not a bug.
    # PERSIST  => n_sentinel > 0 (buffer survived the invocation boundary)
    # RESET    => n_sentinel == 0 (buffer cleared)
    # We assert PERSIST so a persistence result is a green PASS; a RESET result
    # fails loudly and is recorded as the NOT-A-HAZARD verdict.
    assert n_sentinel > 0, (
        "Replay slots did NOT persist across the reset+reload boundary "
        "(kernel B saw no sentinel) => cross-invocation reset clears the buffer."
    )


@parametrize(formats=input_output_formats([DataFormat.Int32], same=True))
def test_replay_reassemble_within_launch(formats):
    """EXP-2: record in one BB, launch in a later BB, one kernel launch."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("RISC-DEST debug window is Blackhole-only.")
    formats = formats[0]

    cfg = _make_config("sources/replay_state_xfunc.cpp", formats)
    res = _run_read_result(cfg)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res, dtype=torch_format)
    n_sentinel = int((res_tensor.view(torch.int32) == SENTINEL).sum().item())

    print(
        f"\n[FS EXP-2] sentinel 0x{SENTINEL:08X} count in cross-function DEST readback: "
        f"{n_sentinel} / {res_tensor.numel()}"
    )
    assert n_sentinel > 0, (
        "Record in one BB + launch in a later BB did NOT reassemble "
        "(no sentinel) => within-launch buffer does not persist across BBs."
    )
