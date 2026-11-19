# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Experiment for tt-metal#64475, not meant for CI. PACK rewrites rows of one Dst half with their own values while
MATH writes other rows of the same half with the FPU. The rewrites cannot change data, so a wrong output value
is a lost FPU write. See sources/xthread_macro_probe_test.cpp for the mode word.

Nothing is asserted about lost writes: each variant appends one RESULT line to $XPROBE_RESULTS.
"""

import os

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, Tilize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from ttexalens.tt_exalens_lib import read_words_from_device

pytestmark = [blackhole_only]

# name -> (stream, delay)
STREAMS = {
    "none": (0, 0),
    "plain": (1, 0),
    **{f"cyc{d}": (2, d) for d in range(8)},
    "instr": (3, 0),
}
# name -> (writer, fp32 Dst)
WRITERS = {
    "elwadd32": (0, True),
    "elwadd16": (0, False),
    "mova2d16": (1, False),
    "movb2d16": (2, False),
}
CFGS = {"cfg0": 0, "remap": 1, "swizzle": 2, "cfg3": 3}
GAPS = [
    int(g) for g in os.environ.get("XPROBE_GAPS", "0,1,2,3,4,5,6,8,10,12,15").split(",")
]
KNOPS = [int(k) for k in os.environ.get("XPROBE_KNOPS", "0,1,3").split(",")]
LEADS = [int(m) for m in os.environ.get("XPROBE_LEADS", "0,3").split(",")]
SELECTED_CFGS = os.environ.get("XPROBE_CFGS", "cfg0").split(",")
SELECTED_WRITERS = os.environ.get("XPROBE_WRITERS", ",".join(WRITERS)).split(",")
SELECTED_STREAMS = os.environ.get("XPROBE_STREAMS", ",".join(STREAMS)).split(",")


@pytest.mark.parametrize("lead", LEADS)
@pytest.mark.parametrize("knop", KNOPS)
@pytest.mark.parametrize("gap", GAPS)
@pytest.mark.parametrize("stream", SELECTED_STREAMS)
@pytest.mark.parametrize("cfg", SELECTED_CFGS)
@pytest.mark.parametrize("writer", SELECTED_WRITERS)
def test_xthread_macro_probe(writer, cfg, stream, gap, knop, lead):
    writer_id, fp32 = WRITERS[writer]
    stream_id, delay = STREAMS[stream]
    if stream == "none" and gap != GAPS[0]:
        pytest.skip("no PACK stream: the gap does nothing")

    formats = InputOutputFormat(
        DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b
    )
    input_dimensions = [256, 256]
    # The plugin reseeds every test identically; fresh inputs keep stale Dst data from matching the expected values.
    seed = int.from_bytes(os.urandom(4), "little")
    torch.manual_seed(seed)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    mode = (
        CFGS[cfg]
        | (stream_id << 2)
        | (delay << 4)
        | (gap << 7)
        | (knop << 11)
        | (lead << 13)
        | (writer_id << 16)
    )
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=None,
        test_config_kwargs={
            "test_name": "sources/xthread_macro_probe_test.cpp",
            "formats": formats,
            "templates": [
                generate_input_dim(input_dimensions, input_dimensions),
                TILIZE(Tilize.No),
            ],
            "runtimes": [
                DEST_INDEX(mode),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(4),
                NUM_BLOCKS(tile_cnt_A),
                NUM_TILES_IN_BLOCK(1),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=tile_cnt_A,
                num_faces=4,
            ),
            "dest_acc": DestAccumulation.Yes if fp32 else DestAccumulation.No,
            "unpack_to_dest": False,
        },
    )
    result = configuration.run().result
    got = (
        torch.tensor(result, dtype=torch.float32 if fp32 else torch.bfloat16)
        .to(torch.float32)
        .flatten()
    )
    want = (
        (src_B if writer == "movb2d16" else src_A)
        .to(torch.bfloat16)
        .to(torch.float32)
        .flatten()[: got.numel()]
    )
    bad = (got != want).nonzero().flatten().tolist()
    bad_tiles = sorted({i // 1024 for i in bad})
    # Rows of the tile that hold a wrong value (the output is row-major within each 16x16 face: 4 faces of 16 rows).
    bad_rows = sorted({(i % 1024) // 16 for i in bad})

    blocks = tile_cnt_A
    math_done = read_words_from_device(
        addr=0x110000, word_count=blocks, location=TestConfig.TENSIX_LOCATION
    )
    pack_done = read_words_from_device(
        addr=0x111000, word_count=blocks, location=TestConfig.TENSIX_LOCATION
    )
    # MATH's block b + 1 is written under PACK's stream of block b if its writes had issued before the stream drained.
    under = sum(
        1
        for b in range(blocks - 1)
        if ((pack_done[b] - math_done[b + 1]) & 0xFFFFFFFF) < 0x80000000
    )

    line = (
        f"RESULT writer={writer} cfg={cfg} stream={stream} gap={gap} knop={knop} lead={lead} | "
        f"bad_values={len(bad)} bad_tiles={len(bad_tiles)}/{blocks} | under_stream={under}/{blocks - 1} | "
        f"first_tiles={bad_tiles[:6]} rows={bad_rows[:16]} | seed={seed}"
    )
    if bad and os.environ.get("XPROBE_DIAG"):
        got_t = got.reshape(-1, 1024)
        a_t = src_A.to(torch.bfloat16).to(torch.float32).flatten().reshape(-1, 1024)
        b_t = src_B.to(torch.bfloat16).to(torch.float32).flatten().reshape(-1, 1024)
        good = [t for t in range(got_t.shape[0]) if t not in bad_tiles]
        notes = []
        for t in bad_tiles[:4]:
            same_a = [u for u in range(a_t.shape[0]) if torch.equal(got_t[t], a_t[u])]
            same_b = [u for u in range(b_t.shape[0]) if torch.equal(got_t[t], b_t[u])]
            wrong = (got_t[t] != want.reshape(-1, 1024)[t]).nonzero().flatten().tolist()
            notes.append(
                f"t{t}: equals A{same_a} B{same_b} zeros={(got_t[t] == 0).sum().item()} wrong={len(wrong)} "
                f"first_wrong={wrong[:4]} got={got_t[t][wrong[:3]].tolist()} want={want.reshape(-1, 1024)[t][wrong[:3]].tolist()}"
            )
        line += f" | DIAG good={good} " + " ; ".join(notes)
    print("\n" + line)
    path = os.environ.get("XPROBE_RESULTS")
    if path:
        with open(path, "a") as out:
            out.write(line + "\n")
    if stream in ("none", "plain"):
        assert not bad, f"control lost data: {line}"
