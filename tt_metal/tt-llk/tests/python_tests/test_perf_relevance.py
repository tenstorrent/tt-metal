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
from helpers.perf.core import PerfConfig, PerfReport
from helpers.perf.relevance import (
    LLK_DISABLE_PERF_RELEVANCE,
    MATH_MATMUL_RELEVANCE,
    MATMUL_RELEVANCE,
    PACK_RELEVANCE,
    PACK_UNTILIZE_RELEVANCE,
    UNPACK_TILIZE_RELEVANCE,
    RunTypeRelevance,
    execute_key,
    maybe_relevance,
    pin_template,
    project_runtimes,
    project_templates,
)
from helpers.perf.schema import MARKER, MEAN, stat_column
from helpers.profiler import Profiler, ProfilerData
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
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
def _clear_relevance_cache(monkeypatch):
    monkeypatch.delenv(LLK_DISABLE_PERF_RELEVANCE, raising=False)
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
    assert_key_hit(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        {**kwargs, "templates": t_lo},
        {**kwargs, "templates": t_hi},
    )
    assert_key_miss(
        PerfRunType.MATH_ISOLATE,
        math,
        {**kwargs, "templates": t_lo},
        {**kwargs, "templates": t_hi},
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
    assert_key_hit(
        PerfRunType.PACK_ISOLATE,
        pack,
        {**kwargs, "runtimes": rt_k1},
        {**kwargs, "runtimes": rt_k32},
    )
    assert_key_miss(
        PerfRunType.L1_TO_L1,
        l1,
        {**kwargs, "runtimes": rt_k1},
        {**kwargs, "runtimes": rt_k32},
    )


def test_maybe_relevance_honors_disable_env(monkeypatch):
    monkeypatch.delenv(LLK_DISABLE_PERF_RELEVANCE, raising=False)
    assert maybe_relevance(MATMUL_RELEVANCE) is MATMUL_RELEVANCE
    monkeypatch.setenv(LLK_DISABLE_PERF_RELEVANCE, "1")
    assert maybe_relevance(MATMUL_RELEVANCE) is None
    assert maybe_relevance(None) is None


def test_disable_perf_relevance_env_does_not_skip_device(monkeypatch):
    monkeypatch.setenv(LLK_DISABLE_PERF_RELEVANCE, "1")
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)
    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 2
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2


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


def _execute_kwargs(test_name, templates, runtimes, formats=None):
    return dict(
        test_name=test_name,
        dest_acc=DestAccumulation.No,
        templates=templates,
        runtimes=runtimes,
        formats=formats,
        speed_of_light=False,
    )


def assert_key_hit(run_type, spec, left, right):
    assert execute_key(run_type=run_type, spec=spec, **left) == execute_key(
        run_type=run_type, spec=spec, **right
    )


def assert_key_miss(run_type, spec, left, right):
    assert execute_key(run_type=run_type, spec=spec, **left) != execute_key(
        run_type=run_type, spec=spec, **right
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
        assert_key_miss(
            run_type,
            spec,
            _execute_kwargs("perf_math_matmul", templates, rt_1),
            _execute_kwargs("perf_math_matmul", templates, rt_4),
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
    assert_key_hit(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        {**base, "templates": t_lo},
        {**base, "templates": t_hi},
    )
    assert_key_hit(
        PerfRunType.PACK_ISOLATE,
        pack,
        {**base, "templates": t_lo},
        {**base, "templates": t_hi},
    )
    assert_key_miss(
        PerfRunType.MATH_ISOLATE,
        math,
        {**base, "templates": t_lo},
        {**base, "templates": t_hi},
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
        assert_key_hit(
            run_type,
            spec,
            _execute_kwargs("perf_pack", templates, rt_off),
            _execute_kwargs("perf_pack", templates, rt_on),
        )
    for run_type in misses:
        spec = PACK_RELEVANCE[run_type]
        assert_key_miss(
            run_type,
            spec,
            _execute_kwargs("perf_pack", templates, rt_off),
            _execute_kwargs("perf_pack", templates, rt_on),
        )


def test_pack_untilize_input_format_reuses_pack_not_l1():
    templates = [INPUT_DIMENSIONS(2, 2, 2, 2)]
    runtimes = [TILE_COUNT(4), LOOP_FACTOR(32)]
    fmt_a = _format(DataFormat.Float16, DataFormat.Float32)
    fmt_b = _format(DataFormat.Float16_b, DataFormat.Float32)
    pack = PACK_UNTILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    l1 = PACK_UNTILIZE_RELEVANCE[PerfRunType.L1_TO_L1]
    assert_key_hit(
        PerfRunType.PACK_ISOLATE,
        pack,
        _execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_a),
        _execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_b),
    )
    assert_key_miss(
        PerfRunType.L1_TO_L1,
        l1,
        _execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_a),
        _execute_kwargs("perf_pack_untilize", templates, runtimes, fmt_b),
    )


def test_pack_untilize_same_tile_cnt_different_dims_misses_cong():
    rt_4x5 = [INPUT_DIMENSIONS(4, 5, 1, 4)]
    rt_5x4 = [INPUT_DIMENSIONS(5, 4, 4, 5)]
    runtimes = [TILE_COUNT(20), LOOP_FACTOR(32)]
    fmt = _format(DataFormat.Float16_b, DataFormat.Float16)
    cong = PACK_UNTILIZE_RELEVANCE[PerfRunType.L1_CONGESTION]
    pack = PACK_UNTILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    assert_key_miss(
        PerfRunType.L1_CONGESTION,
        cong,
        _execute_kwargs("perf_pack_untilize", rt_4x5, runtimes, fmt),
        _execute_kwargs("perf_pack_untilize", rt_5x4, runtimes, fmt),
    )
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        pack,
        _execute_kwargs("perf_pack_untilize", rt_4x5, runtimes, fmt),
        _execute_kwargs("perf_pack_untilize", rt_5x4, runtimes, fmt),
    )


def test_unpack_tilize_output_format_reuses_unpack():
    runtimes = [INPUT_DIMENSIONS(2, 2, 2, 2), TILE_COUNT(4), LOOP_FACTOR(256)]
    fmt_a = _format(DataFormat.Float16, DataFormat.Float16)
    fmt_b = _format(DataFormat.Float16, DataFormat.Float32)
    unpack = UNPACK_TILIZE_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert_key_hit(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        _execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_a),
        _execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_b),
    )


def test_unpack_tilize_pack_keeps_unpack_src_for_bh_tilize_workaround():
    # Same inferred pack_src/pack_dst as Float16→Fp8 vs Fp8→Fp8. PACK INIT
    # skip_bh_tilize_workaround uses unpack_A_src, so those must miss.
    runtimes = [INPUT_DIMENSIONS(8, 8, 8, 8), TILE_COUNT(64), LOOP_FACTOR(256)]
    fmt_f16 = FormatConfig(
        unpack_A_src=DataFormat.Float16,
        unpack_A_dst=DataFormat.Float16,
        pack_src=DataFormat.Float16,
        pack_dst=DataFormat.Fp8_e4m3,
        math=DataFormat.Float16,
    )
    fmt_fp8 = FormatConfig(
        unpack_A_src=DataFormat.Fp8_e4m3,
        unpack_A_dst=DataFormat.Fp8_e4m3,
        pack_src=DataFormat.Float16,
        pack_dst=DataFormat.Fp8_e4m3,
        math=DataFormat.Float16,
    )
    pack = UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        pack,
        _execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_f16),
        _execute_kwargs("perf_unpack_tilize", [], runtimes, fmt_fp8),
    )


def test_unpack_tilize_same_tile_cnt_different_dims_misses_pack():
    rt_2x4 = [INPUT_DIMENSIONS(2, 4, 4, 2), TILE_COUNT(8), LOOP_FACTOR(256)]
    rt_4x2 = [INPUT_DIMENSIONS(4, 2, 2, 4), TILE_COUNT(8), LOOP_FACTOR(256)]
    fmt = _format(DataFormat.Float16, DataFormat.Float16)
    pack = UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    unpack = UNPACK_TILIZE_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        pack,
        _execute_kwargs("perf_unpack_tilize", [], rt_2x4, fmt),
        _execute_kwargs("perf_unpack_tilize", [], rt_4x2, fmt),
    )
    assert_key_miss(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        _execute_kwargs("perf_unpack_tilize", [], rt_2x4, fmt),
        _execute_kwargs("perf_unpack_tilize", [], rt_4x2, fmt),
    )


def test_unpack_tilize_sol_pack_keeps_dim_tile_cnt_invariant():
    runtimes = [INPUT_DIMENSIONS(2, 3, 3, 2), TILE_COUNT(6), LOOP_FACTOR(256)]
    projected = project_runtimes(
        runtimes, UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    )
    dims = next(p for p in projected if isinstance(p, INPUT_DIMENSIONS))
    tiles = next(p for p in projected if isinstance(p, TILE_COUNT))
    assert dims.full_rt_dim * dims.full_ct_dim == tiles.tile_cnt


def test_execute_key_keeps_unpinnable_templates():
    spec = RunTypeRelevance(templates=frozenset({DEST_SYNC}))
    t_2x4 = [
        DEST_SYNC(DestSync.Half),
        INPUT_DIMENSIONS(2, 4, 4, 2),
        MATH_FIDELITY(MathFidelity.LoFi),
    ]
    t_4x2 = [
        DEST_SYNC(DestSync.Half),
        INPUT_DIMENSIONS(4, 2, 2, 4),
        MATH_FIDELITY(MathFidelity.LoFi),
    ]
    t_hifi = [
        DEST_SYNC(DestSync.Half),
        INPUT_DIMENSIONS(2, 4, 4, 2),
        MATH_FIDELITY(MathFidelity.HiFi4),
    ]
    runtimes = [LOOP_FACTOR(32)]
    assert_key_miss(
        PerfRunType.L1_CONGESTION,
        spec,
        _execute_kwargs("perf_unpinnable", t_2x4, runtimes),
        _execute_kwargs("perf_unpinnable", t_4x2, runtimes),
    )
    assert_key_hit(
        PerfRunType.L1_CONGESTION,
        spec,
        _execute_kwargs("perf_unpinnable", t_2x4, runtimes),
        _execute_kwargs("perf_unpinnable", t_hifi, runtimes),
    )


def _math_matmul_runtimes(
    *,
    num_faces=4,
    partial=False,
    in0_r=32,
    dest_index=0,
    num_blocks=1,
    kt=1,
):
    return [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(num_faces, num_faces, num_faces),
        LOOP_FACTOR(64),
        CRK_TILE_DIMM(2, 2, kt),
        NUM_BLOCKS(num_blocks),
        PARTIAL_FACE(
            partial_a=partial,
            partial_face_pack=partial,
            partial_b=partial,
            partial_face_math=partial,
        ),
        IN_TILE_DIMS(in0_r, 32, 32, 32),
        DEST_INDEX(dest_index),
    ]


def test_math_matmul_pack_keeps_faces_partial_tile_dims():
    templates = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(0),
    ]
    base = _math_matmul_runtimes()
    variants = (
        _math_matmul_runtimes(num_faces=2),
        _math_matmul_runtimes(partial=True),
        _math_matmul_runtimes(in0_r=16),
    )
    for run_type in (PerfRunType.PACK_ISOLATE, PerfRunType.L1_CONGESTION):
        spec = MATH_MATMUL_RELEVANCE[run_type]
        for runtimes in variants:
            assert_key_miss(
                run_type,
                spec,
                _execute_kwargs("perf_math_matmul", templates, base),
                _execute_kwargs("perf_math_matmul", templates, runtimes),
            )


def test_math_matmul_sol_pack_keeps_faces_partial_tile_dims():
    runtimes = _math_matmul_runtimes(num_faces=2, partial=True, in0_r=16)
    projected = project_runtimes(
        runtimes, MATH_MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    )
    faces = next(p for p in projected if isinstance(p, NUM_FACES))
    partial = next(p for p in projected if isinstance(p, PARTIAL_FACE))
    dims = next(p for p in projected if isinstance(p, IN_TILE_DIMS))
    dest = next(p for p in projected if isinstance(p, DEST_INDEX))
    assert faces.num_faces == 2
    assert partial.partial_face_pack is True
    assert dims.in0_r_dim == 16
    assert dest.dst_index == 0


def test_pack_dest_index_hits_unpack_misses_pack_math_cong():
    templates = [DEST_SYNC(DestSync.Half)]
    shared = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]
    rt_0 = shared + [DEST_INDEX(0)]
    rt_1 = shared + [DEST_INDEX(1)]
    hits = (PerfRunType.UNPACK_ISOLATE,)
    misses = (
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )
    for run_type in hits:
        spec = PACK_RELEVANCE[run_type]
        assert_key_hit(
            run_type,
            spec,
            _execute_kwargs("perf_pack", templates, rt_0),
            _execute_kwargs("perf_pack", templates, rt_1),
        )
    for run_type in misses:
        spec = PACK_RELEVANCE[run_type]
        assert_key_miss(
            run_type,
            spec,
            _execute_kwargs("perf_pack", templates, rt_0),
            _execute_kwargs("perf_pack", templates, rt_1),
        )


def test_pack_sol_pack_keeps_dest_index():
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
        DEST_INDEX(1),
    ]
    projected = project_runtimes(runtimes, PACK_RELEVANCE[PerfRunType.PACK_ISOLATE])
    dest = next(p for p in projected if isinstance(p, DEST_INDEX))
    assert dest.dst_index == 1
