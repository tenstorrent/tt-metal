# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Quick perf measurement: runs L1_TO_L1 perf config only, reads profiler buffer.
import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)


@pytest.mark.perf
@pytest.mark.parametrize("ct_dim", [2, 3, 4, 5, 6, 7, 8])
def test_perf_measure(perf_report, ct_dim, workers_tensix_coordinates):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    fmt = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    tile_count = ct_dim
    dims = (32, ct_dim * 32)

    cfg = PerfConfig(
        "sources/fast_tilize_bh_test.cpp",
        fmt,
        run_types=[PerfRunType.L1_TO_L1],  # L1_TO_L1 only
        templates=[],
        runtimes=[
            generate_input_dim(dims, dims),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(4),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            None,
            fmt.input_format,
            None,
            fmt.input_format,
            fmt.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        compile_time_formats=True,
    )

    cfg.run(perf_report, location=workers_tensix_coordinates)

    # Read profiler buffers right after L1_TO_L1 run
    from ttexalens.tt_exalens_lib import read_words_from_device

    total_tiles = tile_count * 4  # LOOP_FACTOR=4

    for name, addr in [("Unpack", 0x16B000), ("Math", 0x16C000), ("Pack", 0x16D000)]:
        data = read_words_from_device(
            addr=addr, word_count=0x400, location=workers_tensix_coordinates
        )
        starts = {}
        zones = []
        for i in range(0, len(data), 2):
            if data[i] == 0 and data[i + 1] == 0:
                continue
            etype = (data[i] >> 28) & 0xF
            eid = (data[i] >> 12) & 0xFFFF
            ts = data[i + 1]
            if etype == 0xA:
                starts[eid] = ts
            elif etype == 0xB and eid in starts:
                zones.append((eid, ts - starts[eid]))
                del starts[eid]

        zones.sort(key=lambda x: x[1])
        for eid, delta in zones:
            cyc = delta / total_tiles
            label = "TILE_LOOP" if delta == max(z[1] for z in zones) else ""
            print(
                f"  ct={ct_dim} {name:6s} zone=0x{eid:04x}: {delta:6d} cyc = {cyc:6.1f} cyc/tile {label}"
            )
