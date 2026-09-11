# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for PerfConfig TILE_LOOP relevance projection and cache."""

from pathlib import Path

import pandas as pd
import pytest
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.perf.core import PerfConfig, PerfReport, postprocess_tile_loop
from helpers.perf.relevance import (
    MATH_MATMUL_RELEVANCE,
    MATMUL_RELEVANCE,
    PACK_RELEVANCE,
    PACK_UNTILIZE_RELEVANCE,
    UNPACK_TILIZE_RELEVANCE,
    execute_key,
    pin_template,
    project_runtimes,
    project_templates,
)
from helpers.perf.schema import MARKER, MEAN, stat_column
from helpers.profiler import Profiler, ProfilerData
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    RELU_CONFIG,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

try:
    import xdist  # noqa: F401
except ImportError:

    @pytest.fixture(scope="session")
    def worker_id():
        return "master"


_MARKERS = (("INIT", 0), ("TILE_LOOP", 1))
_THREADS = ("unpack", "math", "pack")

_MATMUL_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]


@pytest.fixture(autouse=True)
def _clear_relevance_cache():
    PerfConfig.clear_relevance_cache()
    yield
    PerfConfig.clear_relevance_cache()


def _one_run_events(seed: int) -> pd.DataFrame:
    rows = []
    ts = 100
    for marker, mid in _MARKERS:
        for thread in _THREADS:
            dur = 10 + mid * 5 + seed
            for etype, offset in (("ZONE_START", 0), ("ZONE_END", dur)):
                rows.append(
                    {
                        "thread": thread,
                        "type": etype,
                        MARKER: marker,
                        "timestamp": ts + offset,
                        "data": 0,
                        "marker_id": mid,
                        "file": "perf.cpp",
                        "line": 1,
                    }
                )
            ts += dur + 5
    return pd.DataFrame(rows)


def _stub_hw(monkeypatch, cfg, elf_calls, get_data_calls):
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False)
    monkeypatch.setattr(TestConfig, "ENABLE_PERF_COUNTERS", False)
    monkeypatch.setattr(TestConfig, "ARTEFACTS_DIR", Path("/tmp/hwfree"), raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", None, raising=False)
    monkeypatch.setattr(
        TestConfig, "get_elf_text_size", staticmethod(lambda path: 4096)
    )
    monkeypatch.setattr(PerfConfig, "TEST_COUNTER", PerfConfig.TEST_COUNTER)

    def fake_get_data(test_name, variant_id, location):
        df = _one_run_events(get_data_calls["n"])
        get_data_calls["n"] += 1
        return ProfilerData(df)

    monkeypatch.setattr(Profiler, "get_data", staticmethod(fake_get_data))

    def run_elf():
        elf_calls.append(cfg.current_run_type)

    cfg.write_runtimes_to_L1 = lambda *a, **k: None
    cfg.run_elf_files = run_elf
    cfg.wait_for_tensix_operations_finished = lambda *a, **k: None


def _matmul_params(fidelity, kt=1):
    templates = [
        MATH_FIDELITY(fidelity),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(),
    ]
    runtimes = [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(),
        LOOP_FACTOR(64),
        TILE_COUNT(2 * 2 * kt),
        CRK_TILE_DIMM(2, 2, kt),
    ]
    return templates, runtimes


def _run_matmul_cfg(
    monkeypatch,
    fidelity,
    elf_calls,
    get_data_calls,
    kt=1,
    relevance=MATMUL_RELEVANCE,
):
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False)
    monkeypatch.setattr(TestConfig, "ENABLE_PERF_COUNTERS", False)
    monkeypatch.setattr(TestConfig, "ARTEFACTS_DIR", Path("/tmp/hwfree"), raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", None, raising=False)
    monkeypatch.setattr(
        TestConfig, "get_elf_text_size", staticmethod(lambda path: 4096)
    )
    monkeypatch.setattr(PerfConfig, "TEST_COUNTER", PerfConfig.TEST_COUNTER)

    templates, runtimes = _matmul_params(fidelity, kt=kt)
    cfg = PerfConfig(
        test_name="perf_relevance",
        formats=None,
        run_types=_MATMUL_RUN_TYPES,
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.No,
        relevance=relevance,
    )
    _stub_hw(monkeypatch, cfg, elf_calls, get_data_calls)
    report = PerfReport()
    cfg.run(report, run_count=1)
    return report._frames[-1]


def test_project_templates_pins_unused_fidelity_and_dest_sync():
    templates = [
        MATH_FIDELITY(MathFidelity.HiFi4),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(0),
        PERF_RUN_TYPE(PerfRunType.UNPACK_ISOLATE),
    ]
    projected = project_templates(
        templates, MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    )
    fidelity = next(p for p in projected if isinstance(p, MATH_FIDELITY))
    dest_sync = next(p for p in projected if isinstance(p, DEST_SYNC))
    run_type = next(p for p in projected if isinstance(p, PERF_RUN_TYPE))
    assert fidelity.math_fidelity == MathFidelity.LoFi
    assert dest_sync.dest_sync == DestSync.Full
    assert run_type.perf_run_type == PerfRunType.UNPACK_ISOLATE


def test_pin_template_defaults():
    assert pin_template(MATH_FIDELITY(MathFidelity.HiFi2)).math_fidelity == (
        MathFidelity.LoFi
    )
    assert pin_template(DEST_SYNC(DestSync.Half)).dest_sync == DestSync.Full
    assert pin_template(THROTTLE_LEVEL(3)).throttle_level == 0


def test_execute_key_drops_fidelity_for_unpack_keeps_for_math():
    t_lo, runtimes = _matmul_params(MathFidelity.LoFi)
    t_hi, _ = _matmul_params(MathFidelity.HiFi4)
    unpack = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    math = MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE]
    kwargs = dict(
        test_name="perf_matmul",
        dest_acc=DestAccumulation.No,
        runtimes=runtimes,
        formats=None,
        speed_of_light=False,
    )
    assert execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        templates=t_lo,
        spec=unpack,
        **kwargs,
    ) == execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        templates=t_hi,
        spec=unpack,
        **kwargs,
    )
    assert execute_key(
        run_type=PerfRunType.MATH_ISOLATE,
        templates=t_lo,
        spec=math,
        **kwargs,
    ) != execute_key(
        run_type=PerfRunType.MATH_ISOLATE,
        templates=t_hi,
        spec=math,
        **kwargs,
    )


def test_execute_key_drops_kt_for_pack_only():
    t, rt_k1 = _matmul_params(MathFidelity.LoFi, kt=1)
    _, rt_k32 = _matmul_params(MathFidelity.LoFi, kt=32)
    pack = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    l1 = MATMUL_RELEVANCE[PerfRunType.L1_TO_L1]
    kwargs = dict(
        test_name="perf_matmul",
        dest_acc=DestAccumulation.No,
        templates=t,
        formats=None,
        speed_of_light=False,
    )
    assert execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        runtimes=rt_k1,
        spec=pack,
        **kwargs,
    ) == execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        runtimes=rt_k32,
        spec=pack,
        **kwargs,
    )
    assert execute_key(
        run_type=PerfRunType.L1_TO_L1,
        runtimes=rt_k1,
        spec=l1,
        **kwargs,
    ) != execute_key(
        run_type=PerfRunType.L1_TO_L1,
        runtimes=rt_k32,
        spec=l1,
        **kwargs,
    )


def test_default_relevance_does_not_skip_device(monkeypatch):
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, relevance=None)
    _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds, relevance=None)
    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 2
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2


def test_fidelity_change_reuses_unpack_and_pack(monkeypatch):
    elf_calls = []
    seeds = {"n": 0}
    frame_lo = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    frame_hi = _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)

    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 1
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2

    unpack_col = stat_column("UNPACK_ISOLATE", MEAN)
    assert frame_lo[unpack_col].tolist() == frame_hi[unpack_col].tolist()
    assert frame_lo["math_fidelity"].iloc[0] == MathFidelity.LoFi
    assert frame_hi["math_fidelity"].iloc[0] == MathFidelity.HiFi4


def test_kt_change_reuses_pack_only(monkeypatch):
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, kt=1)
    frame_k32 = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, kt=32)

    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 2
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2
    assert int(frame_k32["k_dimm"].iloc[0]) == 32


def test_postprocess_tile_loop_pack_uses_rt_ct_not_kt():
    raw = pd.DataFrame(
        {
            MARKER: ["INIT", "TILE_LOOP"],
            "loop_factor": [1, 2],
            "tile_cnt": [32, 32],
            "r_dimm": [2, 2],
            "c_dimm": [2, 2],
            "k_dimm": [8, 8],
            stat_column("MATH_ISOLATE", MEAN): [100.0, 128.0],
            stat_column("PACK_ISOLATE", MEAN): [80.0, 40.0],
            stat_column("L1_CONGESTION[PACK]", MEAN): [80.0, 24.0],
            "L1_TO_L1_mean(fpu_utilization_pct)": [50.0, 60.0],
        }
    )

    out = postprocess_tile_loop(raw.copy())
    tl = out[out[MARKER] == "TILE_LOOP"].iloc[0]
    init = out[out[MARKER] == "INIT"].iloc[0]

    assert tl[stat_column("MATH_ISOLATE", MEAN)] == 2.0
    assert tl[stat_column("PACK_ISOLATE", MEAN)] == 5.0
    assert tl[stat_column("L1_CONGESTION[PACK]", MEAN)] == 3.0
    assert init[stat_column("PACK_ISOLATE", MEAN)] == 80.0
    assert tl["L1_TO_L1_mean(fpu_utilization_pct)"] == 60.0


def _execute_kwargs(test_name, templates, runtimes, formats=None):
    return dict(
        test_name=test_name,
        dest_acc=DestAccumulation.No,
        templates=templates,
        runtimes=runtimes,
        formats=formats,
        speed_of_light=False,
    )


def _format(inp: DataFormat, out: DataFormat) -> FormatConfig:
    return FormatConfig(
        unpack_A_src=inp,
        unpack_A_dst=inp,
        pack_src=out,
        pack_dst=out,
        math=inp,
    )


def test_math_matmul_num_blocks_changes_isolate_keys():
    templates = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(0),
    ]
    shared = [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(),
        LOOP_FACTOR(64),
        CRK_TILE_DIMM(2, 2, 1),
    ]
    rt_1 = shared + [NUM_BLOCKS(1)]
    rt_4 = shared + [NUM_BLOCKS(4)]
    for run_type in (
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ):
        spec = MATH_MATMUL_RELEVANCE[run_type]
        assert execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_math_matmul", templates, rt_1),
        ) != execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_math_matmul", templates, rt_4),
        )


def test_math_matmul_fidelity_reuses_unpack_and_pack_keys():
    t_lo, runtimes = _matmul_params(MathFidelity.LoFi)
    t_hi, _ = _matmul_params(MathFidelity.HiFi4)
    runtimes = runtimes + [NUM_BLOCKS(1)]
    unpack = MATH_MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    pack = MATH_MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    math = MATH_MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE]
    base = dict(
        test_name="perf_math_matmul",
        dest_acc=DestAccumulation.No,
        runtimes=runtimes,
        formats=None,
        speed_of_light=False,
    )
    assert execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE, templates=t_lo, spec=unpack, **base
    ) == execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE, templates=t_hi, spec=unpack, **base
    )
    assert execute_key(
        run_type=PerfRunType.PACK_ISOLATE, templates=t_lo, spec=pack, **base
    ) == execute_key(
        run_type=PerfRunType.PACK_ISOLATE, templates=t_hi, spec=pack, **base
    )
    assert execute_key(
        run_type=PerfRunType.MATH_ISOLATE, templates=t_lo, spec=math, **base
    ) != execute_key(
        run_type=PerfRunType.MATH_ISOLATE, templates=t_hi, spec=math, **base
    )


def test_pack_relu_hits_unpack_math_misses_pack_cong():
    templates = [DEST_SYNC(DestSync.Half)]
    shared = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
    ]
    rt_off = shared + [RELU_CONFIG(0)]
    rt_on = shared + [RELU_CONFIG(1)]
    hits = (PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE)
    misses = (PerfRunType.PACK_ISOLATE, PerfRunType.L1_CONGESTION)
    for run_type in hits:
        spec = PACK_RELEVANCE[run_type]
        assert execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_pack", templates, rt_off),
        ) == execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_pack", templates, rt_on),
        )
    for run_type in misses:
        spec = PACK_RELEVANCE[run_type]
        assert execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_pack", templates, rt_off),
        ) != execute_key(
            run_type=run_type,
            spec=spec,
            **_execute_kwargs("perf_pack", templates, rt_on),
        )


def test_pack_untilize_input_format_reuses_pack_not_l1():
    templates = [INPUT_DIMENSIONS(2, 2, 2, 2)]
    runtimes = [TILE_COUNT(4), LOOP_FACTOR(32)]
    fmt_a = _format(DataFormat.Float16, DataFormat.Float32)
    fmt_b = _format(DataFormat.Float16_b, DataFormat.Float32)
    pack = PACK_UNTILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    l1 = PACK_UNTILIZE_RELEVANCE[PerfRunType.L1_TO_L1]
    assert execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        spec=pack,
        **_execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_a),
    ) == execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        spec=pack,
        **_execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_b),
    )
    assert execute_key(
        run_type=PerfRunType.L1_TO_L1,
        spec=l1,
        **_execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_a),
    ) != execute_key(
        run_type=PerfRunType.L1_TO_L1,
        spec=l1,
        **_execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_b),
    )


def test_unpack_tilize_output_format_reuses_unpack():
    runtimes = [INPUT_DIMENSIONS(2, 2, 2, 2), TILE_COUNT(4), LOOP_FACTOR(256)]
    fmt_a = _format(DataFormat.Float16, DataFormat.Float16)
    fmt_b = _format(DataFormat.Float16, DataFormat.Float32)
    unpack = UNPACK_TILIZE_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        spec=unpack,
        **_execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_a),
    ) == execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        spec=unpack,
        **_execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_b),
    )


def test_unpack_tilize_same_tile_cnt_different_dims_misses_pack():
    rt_2x4 = [INPUT_DIMENSIONS(2, 4, 4, 2), TILE_COUNT(8), LOOP_FACTOR(256)]
    rt_4x2 = [INPUT_DIMENSIONS(4, 2, 2, 4), TILE_COUNT(8), LOOP_FACTOR(256)]
    fmt = _format(DataFormat.Float16, DataFormat.Float16)
    pack = UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    unpack = UNPACK_TILIZE_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        spec=pack,
        **_execute_kwargs("perf_unpack_tilize", [], rt_2x4, fmt),
    ) != execute_key(
        run_type=PerfRunType.PACK_ISOLATE,
        spec=pack,
        **_execute_kwargs("perf_unpack_tilize", [], rt_4x2, fmt),
    )
    assert execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        spec=unpack,
        **_execute_kwargs("perf_unpack_tilize", [], rt_2x4, fmt),
    ) != execute_key(
        run_type=PerfRunType.UNPACK_ISOLATE,
        spec=unpack,
        **_execute_kwargs("perf_unpack_tilize", [], rt_4x2, fmt),
    )


def test_unpack_tilize_sol_pack_keeps_dim_tile_cnt_invariant():
    runtimes = [INPUT_DIMENSIONS(2, 3, 3, 2), TILE_COUNT(6), LOOP_FACTOR(256)]
    projected = project_runtimes(
        runtimes, UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    )
    dims = next(p for p in projected if isinstance(p, INPUT_DIMENSIONS))
    tiles = next(p for p in projected if isinstance(p, TILE_COUNT))
    assert dims.full_rt_dim * dims.full_ct_dim == tiles.tile_cnt
