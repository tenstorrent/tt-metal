# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# lane FS  FP-3 architectural model, datacopy-based vehicle (valid acquired
# DEST + packer readback).  See laneFS-evidence-20260822/.
#
#   CTRL (inline)     : payload executes inline -> sentinel MUST appear.
#   EXP-2 (rec+launch): record in one fn, launch in another, one launch ->
#                       within-launch cross-function reassembly.
#   EXP-1 (persist)   : kernel A records only (no launch); kernel B launches
#                       only (no record).  Sentinel in B <=> replay buffer
#                       persisted across the reset+reload invocation boundary.

import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIMENSIONS, DataCopyGolden, get_golden_generator
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    Tilize,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)

SENTINEL_F32 = 123.5  # BF16 0x42F7 -> FP32 0x42F70000

_FMT = input_output_formats([DataFormat.Float32])[0]
_DIMS = [32, 32]
_NUM_FACES = 4


def _make_vehicle_config(source):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=_FMT.input_format,
        input_dimensions_A=_DIMS,
        stimuli_format_B=_FMT.input_format,
        input_dimensions_B=_DIMS,
    )
    dest_acc = DestAccumulation.Yes  # 32-bit DEST so FP32 store round-trips
    unpack_to_dest = _FMT.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        _FMT,
        _DIMS,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    cfg = TestConfig(
        source,
        _FMT,
        templates=[generate_input_dim(_DIMS, _DIMS), TILIZE(Tilize.No)],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(_NUM_FACES),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            _FMT.input_format,
            src_B,
            _FMT.input_format,
            _FMT.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=_NUM_FACES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )
    return cfg


def _skip_if_compile_only():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)


def _build_run_read(cfg):
    cfg.prepare()
    _skip_if_compile_only()
    cfg.write_runtimes_to_L1()
    cfg.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
    cfg.variant_stimuli.clear_result_buffer(TestConfig.TENSIX_LOCATION)
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished()
    return cfg.variant_stimuli.collect_results(TestConfig.TENSIX_LOCATION)


def _build_run_noread(cfg):
    cfg.prepare()
    _skip_if_compile_only()
    cfg.write_runtimes_to_L1()
    cfg.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
    cfg.run_elf_files()
    cfg.wait_for_tensix_operations_finished()


def _sentinel_count(res):
    t = torch.tensor(res, dtype=format_dict[_FMT.output_format])
    return int((t == SENTINEL_F32).sum().item()), t.numel()


def _skip_non_bh():
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole-only experiment.")


def test_vehicle_inline_ctrl():
    """CTRL: inline payload must overwrite DEST with the sentinel."""
    _skip_non_bh()
    res = _build_run_read(_make_vehicle_config("sources/replay_vehicle_inline.cpp"))
    n, total = _sentinel_count(res)
    print(f"\n[FS CTRL/vehicle] sentinel {SENTINEL_F32} count: {n} / {total}")
    assert n > 0, "Inline vehicle broken: replayable DEST store not visible."


def test_vehicle_reassemble_within_launch():
    """EXP-2: record in one fn, launch in another (one launch)."""
    _skip_non_bh()
    res = _build_run_read(_make_vehicle_config("sources/replay_vehicle_rl.cpp"))
    n, total = _sentinel_count(res)
    print(f"\n[FS EXP-2/vehicle] sentinel {SENTINEL_F32} count: {n} / {total}")
    assert n > 0, "Within-launch cross-function reassembly did not deliver."


def test_vehicle_launch_only_baseline():
    """Negative control: launch slots 0..1 with NO prior record anywhere.

    Run as the first REPLAY-issuing kernel after a full board reset.  If the
    reset cleared the Replay buffer, the launch replays zero words and the
    sentinel is absent (n == 0).  A non-zero count here would mean the buffer
    survived even a full board reset (or holds stale state)."""
    _skip_non_bh()
    res = _build_run_read(_make_vehicle_config("sources/replay_vehicle_launchonly.cpp"))
    n, total = _sentinel_count(res)
    print(f"\n[FS BASE/vehicle] sentinel {SENTINEL_F32} count (B alone): {n} / {total}")
    # Recorded either way; we EXPECT 0 (cleared) so assert it for a green pass.
    assert n == 0, (
        "Launch-only after full reset delivered the sentinel: the Replay "
        "buffer survived a full board reset (or holds stale recorded state)."
    )


def test_vehicle_persist_across_reset():
    """EXP-1: kernel A records only; kernel B (reset+reload) launches only."""
    _skip_non_bh()
    cfg_a = _make_vehicle_config("sources/replay_vehicle_reconly.cpp")
    cfg_b = _make_vehicle_config("sources/replay_vehicle_launchonly.cpp")
    _build_run_noread(cfg_a)  # record slots 0..1, never launch
    res = _build_run_read(cfg_b)  # datacopy + launch slots 0..1, read
    n, total = _sentinel_count(res)
    print(f"\n[FS EXP-1/vehicle] sentinel {SENTINEL_F32} count in kernel-B: {n} / {total}")
    assert n > 0, (
        "Replay buffer did NOT persist across the reset+reload invocation "
        "boundary (kernel B saw no sentinel)."
    )
