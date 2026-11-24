# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Micro-benchmark: when does the replay buffer pay off for an SFPU row loop on Quasar?

Quasar cannot execute a replay buffer while loading it (TEN-4690), so sfpi records the loop body
and then replays it, on every call. This benchmark times that against pushing the body inline.
It is a measurement, not a pass/fail test of an LLK: it runs
sources/quasar/replay_breakeven_quasar_test.cpp on the math thread, reads the wall-clock cycle
counts the kernel left in L1 and writes them out.

How to run (from tt_metal/tt-llk, with the Quasar emulator environment set up):

    export CHIP_ARCH=quasar
    export NNG_SOCKET_ADDR=tcp://<emulator host>:<port> NNG_SOCKET_LOCAL_PORT=5555
    export RUNNER_TEMP=$(mktemp -d)                 # fresh build cache
    export REPLAY_BENCH_OUT=$PWD/replay_breakeven.jsonl

    .claude/scripts/run_test.sh run --worktree $PWD --arch quasar \\
        --test test_replay_breakeven_quasar.py --sim-path <path to emu-quasar-1x3 build>

Add --k "bench:4-" (or any other set, see BENCH_COMBINATIONS) to run a subset. The 25 default
variants take about three minutes on the emulator once it is up.

Output:
    $REPLAY_BENCH_OUT              one JSON line per measurement (appended; delete it between runs)
    $REPLAY_BENCH_OUT.summary.txt  cycles per calculate_*() call, record+replay and replay-only
                                   compared with inline

Each config is measured twice in the kernel; the summary uses the faster pass (the first one pays
the instruction-cache misses of cold code). One "call" is one face: N row-pair iterations of an
L-instruction body, the unit a calculate_*() function covers. "push" is the time until the RISC-V
has issued everything, "total" the time until the Tensix backend is idle.

Variants:
    inline          N * L instructions pushed directly
    record_replay   REPLAY(load) + L instructions + N REPLAYs on every call (what sfpi emits)
    replay_only     recorded once before the timed region, each call is N REPLAYs
    empty           per-tile start / face advance / done overhead with no body
    real_kernel     a production sfpi kernel; auto_replay says whether the compiler rolled it up
    record_exec     execute-while-loading. Hangs while the erratum is present, so it only runs
                    with REPLAY_BENCH_EXEC_PROBE=1 and is expected to time out today.
"""

import json
import os
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from ttexalens.tt_exalens_lib import read_words_from_device

# Keep in sync with replay_breakeven_quasar_test.cpp
RESULT_ADDR = 0x160000
RESULT_MAGIC = 0x52504C59
RECORD_WORDS = 5
TILES = 8
FACES_PER_TILE = 4
CALLS = TILES * FACES_PER_TILE
VARIANT_NAMES = {
    0: "inline",
    1: "record_replay",
    2: "replay_only",
    3: "empty",
    4: "real_kernel",
    5: "record_exec",
}
MIX_NAMES = {
    0: "dependent SFPMAD chain",
    1: "dependent SFPABS chain",
    2: "SFPLOADI pairs",
    3: "independent SFPMAD",
    4: "independent SFPMOV",
}
REAL_KERNEL_NAMES = ["abs", "square", "sign", "softsign", "erf", "mish", "cube_root"]
# Body of each config of set 6, in measurement order (sequences lifted from the mish kernel)
SET6_BODIES = [
    "(SFPLOADI, SFPLOADI, SFPMAD) x4",
    "SFPMAD, SFPNOP, SFPIADD, SFPADDI, (SFPLOADI, SFPLOADI, SFPMAD) x2",
    "(SFPMAD, SFPMAD, SFPMAD, SFPEXEXP) x2",
    "SFPLOAD, SFPLOADI, SFPGT, SFPLOADI x4, SFPMAD",
    "the 32 instructions sfpi records for mish",
    "the 32 instructions sfpi records for mish",
    "the 32 instructions sfpi records for mish",
]

NO_AUTO_REPLAY_FLAG = " -mno-tt-tensix-optimize-replay "

# "<set>-<mix>-<auto_replay>":
#   sets 0-3  synthetic sweeps over body length L and iteration count N, one build per mix
#   set 4     production sfpi kernels, with and without the compiler's replay roll-up
#   set 5     execute-while-loading probe (opt-in, see the module docstring)
#   set 6     instruction sequences lifted from the sfpi-compiled mish kernel
#   sets 7-8  inline bodies of 320 to 896 instructions, around the 2 KB math TRISC icache
BENCH_COMBINATIONS = (
    [f"{s}-{m}-1" for s in (0, 1, 2, 3) for m in (0, 1, 2, 3, 4)]
    + ["4-0-1", "4-0-0"]
    + (["5-0-1"] if os.environ.get("REPLAY_BENCH_EXEC_PROBE") == "1" else [])
    + ["6-0-1"]
    + ["7-4-1", "8-4-1"]
)


@dataclass
class BENCH_SET(TemplateParameter):
    bench_set: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int BENCH_SET = {self.bench_set};"


@dataclass
class BENCH_MIX(TemplateParameter):
    bench_mix: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int BENCH_MIX = {self.bench_mix};"


@dataclass
class AUTO_REPLAY(TemplateParameter):
    """Not read by the kernel: keeps the two compiler-flag builds of set 4 apart in the build cache."""

    auto_replay: int = 1

    def convert_to_cpp(self) -> str:
        return f"constexpr int AUTO_REPLAY = {self.auto_replay};"


def _best_per_call(rows):
    """Fastest pass of each config, in cycles per call, in measurement order."""
    best = {}
    for index, row in enumerate(rows):
        # The kernel measures every config twice in a row.
        key = (index // 2, row["variant"], row["L"], row["N"])
        best[key] = min(best.get(key, float("inf")), row["total_cycles"] / CALLS)
    return best


def _summarize(bench_set, bench_mix, auto_replay, rows):
    best = _best_per_call(rows)
    lines = []
    if bench_set == 4:
        lines.append(
            f"set 4: production sfpi kernels, compiler replay roll-up "
            f"{'on' if auto_replay else 'off'} (cycles per call, 8 iterations)"
        )
    elif bench_set == 6:
        lines.append("set 6: sequences from the mish kernel (cycles per call)")
    else:
        lines.append(
            f"set {bench_set}: {MIX_NAMES[bench_mix]} (cycles per call; "
            "record+replay and replay-only against inline)"
        )

    inline = None
    config_index = -1
    for (_, variant, length, iters), cycles in best.items():
        if variant == "empty":
            lines.append(f"  empty call overhead      {cycles:8.2f}")
        elif variant == "real_kernel":
            lines.append(f"  {REAL_KERNEL_NAMES[length]:<24} {cycles:8.2f}")
        elif variant == "record_exec":
            lines.append(f"  L={length:<3} N={iters:<3} record_exec  {cycles:8.2f}")
        elif variant == "inline":
            inline = cycles
            config_index += 1
            body = f"  [{SET6_BODIES[config_index]}]" if bench_set == 6 else ""
            lines.append(
                f"  L={length:<3} N={iters:<3} inline       {cycles:8.2f}{body}"
            )
        else:
            delta = 100.0 * (cycles - inline) / inline
            lines.append(f"  {'':11} {variant:<13}{cycles:8.2f}  ({delta:+.1f}%)")
    return "\n".join(lines) + "\n\n"


@pytest.mark.quasar
@parametrize(bench=BENCH_COMBINATIONS)
def test_replay_breakeven_quasar(bench):
    bench = bench[0] if isinstance(bench, tuple) else bench
    bench_set, bench_mix, auto_replay = (int(x) for x in bench.split("-"))
    # The compiler flag is process-wide build state: set it for this variant only.
    TestConfig.OPTIONS_ALL = TestConfig.OPTIONS_ALL.replace(NO_AUTO_REPLAY_FLAG, " ")
    if not auto_replay:
        TestConfig.OPTIONS_ALL += NO_AUTO_REPLAY_FLAG

    # The kernel ignores the stimuli; the harness needs a format and one tile to launch.
    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
    )
    input_dimensions = [32, 32]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    configuration = TestConfig(
        "sources/quasar/replay_breakeven_quasar_test.cpp",
        formats,
        templates=[
            BENCH_SET(bench_set),
            BENCH_MIX(bench_mix),
            AUTO_REPLAY(auto_replay),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
    )

    configuration.run()

    header = read_words_from_device(
        TestConfig.TENSIX_LOCATION, RESULT_ADDR, word_count=2
    )
    assert header[0] == RESULT_MAGIC, f"benchmark did not finish: {header[0]:#x}"
    count = header[1]
    words = read_words_from_device(
        TestConfig.TENSIX_LOCATION, RESULT_ADDR + 8, word_count=count * RECORD_WORDS
    )

    rows = []
    for i in range(count):
        variant, length, iters, pushed, total = words[
            i * RECORD_WORDS : (i + 1) * RECORD_WORDS
        ]
        rows.append(
            {
                "set": bench_set,
                "mix": bench_mix,
                "auto_replay": auto_replay,
                "variant": VARIANT_NAMES[variant],
                "L": length,
                "N": iters,
                "push_cycles": pushed,
                "total_cycles": total,
                "push_per_call": pushed / CALLS,
                "total_per_call": total / CALLS,
            }
        )

    out_path = os.environ.get("REPLAY_BENCH_OUT", "/tmp/replay_breakeven.jsonl")
    with open(out_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(out_path + ".summary.txt", "a") as f:
        f.write(_summarize(bench_set, bench_mix, auto_replay, rows))
