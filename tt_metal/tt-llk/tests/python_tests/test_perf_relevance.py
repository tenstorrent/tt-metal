# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for PerfConfig TILE_LOOP relevance projection and cache."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    L1Accumulation,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.perf.core import PerfConfig, PerfReport
from helpers.perf.relevance import (
    ALL_PERF_RUN_TYPES,
    LLK_DISABLE_PERF_RELEVANCE,
    PerfRelevance,
    RunTypeRelevance,
    _hashable,
    execute_key,
    maybe_relevance,
    pin_template,
    project_formats,
    project_runtimes,
    project_stimuli,
    project_templates,
    spec_is_full_fidelity,
)
from helpers.perf.schema import MARKER, MEAN, stat_column
from helpers.profiler import Profiler, ProfilerData
from helpers.stimuli_config import StimuliConfig
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
    RuntimeParameter,
)
from perf_math_matmul import MATH_MATMUL_RELEVANCE
from perf_matmul import MATMUL_RELEVANCE
from perf_pack import PACK_RELEVANCE
from perf_pack_untilize import PACK_UNTILIZE_RELEVANCE
from perf_unpack_tilize import UNPACK_TILIZE_RELEVANCE

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


# ---------------------------------------------------------------------------
# 0. Shared constants and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_relevance_cache(monkeypatch):
    """Give every test a clean relevance cache and a clean opt-out env var.

    ``PerfConfig.EXECUTE_CACHE`` and the per-class ``as_map`` cache are both
    process-local class state, so without this fixture a cache entry written by
    one test would satisfy a lookup in the next and turn a real regression into
    a silent pass. The ``delenv`` also means a test that sets
    ``LLK_DISABLE_PERF_RELEVANCE`` cannot leak the opt-out into its neighbours.
    """
    monkeypatch.delenv(LLK_DISABLE_PERF_RELEVANCE, raising=False)
    PerfConfig.clear_relevance_cache()
    yield
    PerfConfig.clear_relevance_cache()


# ---------------------------------------------------------------------------
# 1. Test helpers
#
# Builders come first, then the assertion verbs, then the
# hardware-free device stubs, then the whole-run drivers that
# compose them. Nothing here touches a device.
# ---------------------------------------------------------------------------


def _format(inp: DataFormat, out: DataFormat) -> FormatConfig:
    """Build a ``FormatConfig`` with one input format and one output format.

    ``inp`` lands on everything the unpacker and math unit see, ``out`` on
    everything the packer writes, which is the shape almost every test here
    wants: vary one end of the pipeline and hold the other fixed.

    >>> f = _format(DataFormat.Float32, DataFormat.Bfp8_b)
    >>> f.unpack_A_src.name, f.unpack_A_dst.name, f.math.name
    ('Float32', 'Float32', 'Float32')
    >>> f.pack_src.name, f.pack_dst.name
    ('Bfp8_b', 'Bfp8_b')
    """
    return FormatConfig(
        unpack_A_src=inp,
        unpack_A_dst=inp,
        pack_src=out,
        pack_dst=out,
        math=inp,
    )


def _stimuli(tile_count, res_count=None, **kwargs):
    """Build a ``StimuliConfig`` carrying tile counts and nothing else.

    Both L1 buffers are ``None`` because no test in this file writes to a
    device; only the tile counts and formats matter, since those are what
    ``project_stimuli`` rewrites and what ``stimuli_key`` fingerprints. ``A``
    and ``B`` always get ``tile_count``, while the result operand can differ.

    >>> s = _stimuli(4)
    >>> s.tile_count_A, s.tile_count_B, s.tile_count_res
    (4, 4, 4)
    >>> s = _stimuli(16, res_count=4)
    >>> s.tile_count_A, s.tile_count_B, s.tile_count_res
    (16, 16, 4)

    ``kwargs`` reaches the rest of the constructor, which is how the narrow-tile
    tests shrink a face: ``_stimuli(4, face_r_dim=16)``.
    """
    res = tile_count if res_count is None else res_count
    return StimuliConfig(
        None,
        DataFormat.Float16,
        None,
        DataFormat.Float16,
        DataFormat.Float16,
        tile_count_A=tile_count,
        tile_count_B=tile_count,
        tile_count_res=res,
        **kwargs,
    )


def _matmul_params(fidelity, kt=1):
    """Build the ``(templates, runtimes)`` pair for a ``perf_matmul`` case.

    The two knobs are the ones the matmul policy cares about: ``fidelity`` is
    compile-time and invisible to the unpacker and packer, and ``kt`` is the
    inner dimension, invisible to the packer. ``TILE_COUNT`` is kept consistent
    with ``CRK_TILE_DIMM`` at ``c * r * k``, so callers cannot accidentally
    produce a case whose tile count contradicts its dimensions.

    >>> templates, runtimes = _matmul_params(MathFidelity.HiFi4, kt=32)
    >>> next(t for t in templates if isinstance(t, MATH_FIDELITY)).math_fidelity
    <MathFidelity.HiFi4: 4>
    >>> next(r for r in runtimes if isinstance(r, TILE_COUNT)).tile_cnt
    128
    >>> next(r for r in runtimes if isinstance(r, CRK_TILE_DIMM)).k_dimm
    32
    """
    _C_DIMM = 2
    _R_DIMM = 2
    templates = [
        MATH_FIDELITY(fidelity),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(),
    ]
    runtimes = [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(),
        LOOP_FACTOR(64),
        TILE_COUNT(_C_DIMM * _R_DIMM * kt),
        CRK_TILE_DIMM(c_dimm=_C_DIMM, r_dimm=_R_DIMM, k_dimm=kt),
    ]
    return templates, runtimes


def _math_matmul_runtimes(
    *,
    num_faces=4,
    partial=False,
    in0_r=32,
    dest_index=0,
    num_blocks=1,
    kt=1,
):
    """Build the runtime list for a ``perf_math_matmul`` case.

    Keyword-only so a test states just the one axis it sweeps and inherits a
    full-tile, non-partial, single-block default for the rest. ``num_faces``
    sets all three operand slots at once, and ``partial`` sets all four partial
    face flags together, because the pack thread reads them as a group.

    >>> rts = _math_matmul_runtimes(num_faces=2, partial=True, in0_r=16)
    >>> next(r for r in rts if isinstance(r, NUM_FACES)).num_faces
    2
    >>> next(r for r in rts if isinstance(r, PARTIAL_FACE)).partial_face_pack
    True
    >>> next(r for r in rts if isinstance(r, IN_TILE_DIMS)).in0_r_dim
    16
    """
    return [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(num_faces, num_faces, num_faces),
        LOOP_FACTOR(64),
        CRK_TILE_DIMM(c_dimm=2, r_dimm=2, k_dimm=kt),
        NUM_BLOCKS(num_blocks),
        PARTIAL_FACE(
            partial_a=partial,
            partial_face_pack=partial,
            partial_b=partial,
            partial_face_math=partial,
        ),
        IN_TILE_DIMS(in0_r_dim=in0_r),
        DEST_INDEX(dest_index),
    ]


def _execute_kwargs(test_name, templates, runtimes, formats=None):
    """Bundle the fixed half of an ``execute_key`` call.

    ``execute_key`` takes a wide keyword signature, and most of it is constant
    across a comparison: only the parameter under test differs between the two
    sides. Pairing this with ``{**base, "runtimes": other}`` keeps each test's
    diff down to the single axis it sweeps.

    >>> base = _execute_kwargs("perf_matmul", templates, runtimes)
    >>> sorted(base)
    ['dest_acc', 'formats', 'runtimes', 'speed_of_light', 'templates', 'test_name']
    """
    return dict(
        test_name=test_name,
        dest_acc=DestAccumulation.No,
        templates=templates,
        runtimes=runtimes,
        formats=formats,
        speed_of_light=False,
    )


def assert_key_hit(run_type, spec, left, right):
    """Assert two parameter sets collapse onto one ``execute_key``.

    A hit is the whole point of the feature: ``left`` and ``right`` differ in
    some parameter, ``run_type`` cannot observe that parameter under ``spec``,
    so the second case reuses the first one's cached TILE_LOOP result instead
    of compiling and running on the device again.

    >>> assert_key_hit(
    ...     PerfRunType.UNPACK_ISOLATE,
    ...     MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE],
    ...     _execute_kwargs("perf_matmul", t_lofi, runtimes),
    ...     _execute_kwargs("perf_matmul", t_hifi4, runtimes),
    ... )
    """
    assert execute_key(run_type=run_type, spec=spec, **left) == execute_key(
        run_type=run_type, spec=spec, **right
    )


def assert_key_miss(run_type, spec, left, right):
    """Assert two parameter sets keep distinct ``execute_key`` values.

    The safety direction, and the one worth more than the hits: ``run_type``
    can observe the parameter that differs, so collapsing these two cases would
    report one measurement under both sets of conditions. Every ``assert_key_hit``
    in this file should have a matching miss on a run type that does observe the
    parameter, otherwise the spec is not being pinned down from both sides.
    """
    assert execute_key(run_type=run_type, spec=spec, **left) != execute_key(
        run_type=run_type, spec=spec, **right
    )


def _one_run_events(seed: int) -> pd.DataFrame:
    """Synthesise the profiler event frame for one device measurement.

    This is the mechanism the whole cache half of this file rests on, so it is
    worth reading closely. The frame holds one ``ZONE_START``/``ZONE_END`` pair
    per marker per thread (2 markers x 3 threads = 6 zones, 12 rows), and
    ``seed`` shifts every duration in it by a constant: INIT zones last
    ``10 + seed`` cycles and TILE_LOOP zones ``15 + seed``.

    >>> d = _one_run_events(0)
    >>> len(d)
    12
    >>> [int(e - s) for s, e in zip(d[d.type == "ZONE_START"].timestamp,
    ...                             d[d.type == "ZONE_END"].timestamp)]
    [10, 10, 10, 15, 15, 15]

    The same expression on ``_one_run_events(1)`` gives ``[11, 11, 11, 16, 16,
    16]``: one seed later, every zone is one cycle longer.

    ``_stub_hw`` hands out a fresh, incrementing seed on every real device read.
    So two frames carrying identical durations cannot be coincidence: they are
    the same measurement, replayed from the cache. That is what lets a test
    write ``lo[col].tolist() == hi[col].tolist()`` and call it proof of reuse.
    """
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


def _patch_hwfree_classvars(monkeypatch, *, speed_of_light=False):
    """Point ``TestConfig`` at nothing, so ``PerfConfig`` can run without a device.

    ``BUILD_MODE.CONSUME`` skips compilation, ``TENSIX_LOCATION=None`` skips
    device discovery, and ``get_elf_text_size`` returns a fixed 4096 so code-size
    columns are predictable. Re-setting ``PerfConfig.TEST_COUNTER`` to its own
    value looks odd but is deliberate: it registers the attribute with
    ``monkeypatch``, so the counter a test increments is rolled back afterwards.

    ``speed_of_light`` is threaded through because under SoL the runtimes and
    formats are inlined into the compile header as ``constexpr``, which changes
    what projection is allowed to pin.
    """
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", speed_of_light)
    monkeypatch.setattr(TestConfig, "ENABLE_PERF_COUNTERS", False)
    monkeypatch.setattr(TestConfig, "ARTEFACTS_DIR", Path("/tmp/hwfree"), raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", None, raising=False)
    monkeypatch.setattr(
        TestConfig, "get_elf_text_size", staticmethod(lambda path: 4096)
    )
    monkeypatch.setattr(PerfConfig, "TEST_COUNTER", PerfConfig.TEST_COUNTER)


def _stub_hw(monkeypatch, cfg, elf_calls, get_data_calls, *, speed_of_light=False):
    """Replace every device interaction on ``cfg`` with an in-process stand-in.

    Three entry points are swapped: ``write_runtimes_to_L1`` and
    ``wait_for_tensix_operations_finished`` become no-ops, and ``run_elf_files``
    appends ``cfg.current_run_type`` to ``elf_calls``. That list is the skip
    counter for the whole file, because it grows only on a real device run and
    not on a cache hit:

    >>> elf_calls, seeds = [], {"n": 0}
    >>> _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    >>> _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)
    >>> elf_calls.count(PerfRunType.UNPACK_ISOLATE)   # fidelity is invisible here
    1
    >>> elf_calls.count(PerfRunType.MATH_ISOLATE)     # but not here
    2

    ``get_data_calls`` is a one-key dict used as a mutable counter: ``Profiler``
    returns ``_one_run_events(n)`` and then increments ``n``, so each real
    measurement gets its own durations. Callers share one counter across several
    runs to keep the seeds distinct across a whole test.
    """
    _patch_hwfree_classvars(monkeypatch, speed_of_light=speed_of_light)

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


def _run_config(cfg, run_type):
    """Recover the stored ``(templates, runtimes, run_type)`` triple for one run type.

    ``PerfConfig`` expands its constructor arguments into one triple per run type
    in ``cfg.run_configs``, appending ``PERF_RUN_TYPE(run_type)`` to the
    templates as it goes. Pulling a triple back out lets a test drive
    ``cfg._apply_run_config(*triple)`` directly and inspect the resulting
    ``variant_id`` or ``formats_config``, without a full ``cfg.run()`` and the
    stubbed device that would require.

    >>> pack = _run_config(cfg, PerfRunType.PACK_ISOLATE)
    >>> cfg._apply_run_config(*pack)
    >>> cfg.variant_id
    '...'

    Raises ``LookupError`` rather than returning ``None`` if the run type was
    never configured, so a typo fails at the call site.
    """
    for templates, runtimes, stored in cfg.run_configs:
        if stored == run_type:
            return templates, runtimes, stored
    raise LookupError(f"{run_type} not in run_configs")


def _run_matmul_cfg(
    monkeypatch,
    fidelity,
    elf_calls,
    get_data_calls,
    kt=1,
    relevance=MATMUL_RELEVANCE,
):
    """Run one full ``perf_matmul`` case through the stubbed device.

    Returns the report frame for the run (``report._frames[-1]``, reaching past
    the public API because nothing else exposes a single run's rows). Pass the
    same ``elf_calls`` and ``get_data_calls`` to two successive calls to compare
    them: ``elf_calls`` then shows how many device runs the pair actually cost,
    and matching durations in the two frames show the second was replayed.

    >>> elf_calls, seeds = [], {"n": 0}
    >>> lo = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    >>> hi = _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)

    ``relevance=None`` disables the feature for a baseline in which nothing may
    be skipped; ``kt`` sweeps the matmul inner dimension.
    """
    _patch_hwfree_classvars(monkeypatch)
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


def _run_pack_cfg(monkeypatch, elf_calls, get_data_calls, *, unpack_to_dest, relu=0):
    """Run one full ``perf_pack`` case through the stubbed device, returning the config.

    Returns ``cfg`` rather than a report frame because its callers inspect
    ``PerfConfig.EXECUTE_CACHE`` afterwards rather than the measurements.
    ``unpack_to_dest`` drives ``dest_acc`` along with it, since unpacking
    straight to dest is only meaningful with accumulation enabled, and that
    pairing is what makes MATH_ISOLATE's TILE_LOOP an empty no-op.
    """
    _patch_hwfree_classvars(monkeypatch)
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(relu),
        DEST_INDEX(0),
    ]
    cfg = PerfConfig(
        test_name="sources/pack_test.cpp",
        formats=_format(DataFormat.Float32, DataFormat.Float32),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.Yes if unpack_to_dest else DestAccumulation.No,
        unpack_to_dest=unpack_to_dest,
        disable_format_inference=True,
        relevance=PACK_RELEVANCE,
    )
    _stub_hw(monkeypatch, cfg, elf_calls, get_data_calls)
    cfg.run(PerfReport(), run_count=1)
    return cfg


# ---------------------------------------------------------------------------
# 2. Feature gate and spec validation
#
# Before any projection happens: is relevance even on, is the map
# complete, and does a spec actually buy anything.
# ---------------------------------------------------------------------------


def test_maybe_relevance_honors_disable_env(monkeypatch):
    """``LLK_DISABLE_PERF_RELEVANCE=1`` is the single kill switch for the feature.

    It works by collapsing the map to ``None``, which is also what a suite with
    no policy passes. Every downstream site already handles a ``None`` spec as
    "pin nothing, cache nothing", so one env var turns the whole optimisation
    off without a branch anywhere else.
    """
    monkeypatch.delenv(LLK_DISABLE_PERF_RELEVANCE, raising=False)
    assert maybe_relevance(MATMUL_RELEVANCE) is MATMUL_RELEVANCE
    monkeypatch.setenv(LLK_DISABLE_PERF_RELEVANCE, "1")
    assert maybe_relevance(MATMUL_RELEVANCE) is None
    assert maybe_relevance(None) is None


def test_relevance_map_requires_every_run_type():
    """A run type with no spec in the map is a construction-time error, not a default.

    ``PACK_RELEVANCE`` has no ``SFPU_ISOLATE`` entry, so asking for that run type
    must raise and name it. Falling back silently is the dangerous option: the
    fallback would have to guess, and guessing "observes nothing" would let an
    SFPU measurement be served from an unrelated case's cached result.
    """
    templates, runtimes = _matmul_params(MathFidelity.LoFi)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="SFPU_ISOLATE"
    ):
        PerfConfig(
            test_name="perf_relevance",
            formats=None,
            run_types=[PerfRunType.L1_TO_L1, PerfRunType.SFPU_ISOLATE],
            templates=templates,
            runtimes=runtimes,
            dest_acc=DestAccumulation.No,
            relevance=PACK_RELEVANCE,
        )


def test_widened_run_types_name_the_missing_spec():
    """A subclass that adds a run type this class cannot describe fails at map build.

    ``as_map`` used to index a fixed five-entry dict and raise a bare
    ``KeyError``. The error names the run type, matching ``PerfConfig``.
    """

    class _Wide(PerfRelevance):
        run_types = [*ALL_PERF_RUN_TYPES, PerfRunType.SFPU_ISOLATE]

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="SFPU_ISOLATE"
    ):
        _Wide().as_map


def test_l1_to_l1_spec_is_full_fidelity():
    """``spec_is_full_fidelity`` marks the specs that pin nothing.

    ``L1_TO_L1`` always measures the real pipeline, and ``PACK_RELEVANCE`` leaves
    ``MATH_ISOLATE`` at KEEP_ALL as well. Both are full fidelity, and section 11
    shows the consequence: neither ever enters the cache. ``MATMUL_RELEVANCE``
    does pin parameters for ``MATH_ISOLATE``, so the same run type is not full
    fidelity there; the answer depends on the policy, not the run type alone.
    """
    assert spec_is_full_fidelity(MATMUL_RELEVANCE[PerfRunType.L1_TO_L1])
    assert not spec_is_full_fidelity(MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    assert spec_is_full_fidelity(PACK_RELEVANCE[PerfRunType.MATH_ISOLATE])
    assert not spec_is_full_fidelity(MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE])


# ---------------------------------------------------------------------------
# 3. Projection: templates
#
# pin_template rewrites one compile-time parameter to its canonical
# default; project_templates applies that across a whole list for
# the parameters a run type cannot observe.
# ---------------------------------------------------------------------------


def test_pin_template_defaults():
    """The canonical stand-in for each pinnable template, one parameter at a time.

    ``HiFi2 -> LoFi``, ``Half -> Full``, ``3 -> 0``. The particular values are
    arbitrary; what matters is that they are stable, because that is what makes
    two cases differing only in an unobserved template hash to one ``variant_id``
    and share one compiled ELF.
    """
    assert pin_template(MATH_FIDELITY(MathFidelity.HiFi2)).math_fidelity == (
        MathFidelity.LoFi
    )
    assert pin_template(DEST_SYNC(DestSync.Half)).dest_sync == DestSync.Full
    assert pin_template(THROTTLE_LEVEL(3)).throttle_level == 0


def test_project_templates_pins_unused_fidelity_and_dest_sync():
    """Projection pins what the unpacker cannot see and leaves the rest alone.

    ``MATMUL_RELEVANCE[UNPACK_ISOLATE].templates`` is ``{DEST_SYNC}``, so from
    ``HiFi4``/``Half``/``0`` only the fidelity is rewritten (to ``LoFi``), while
    ``DEST_SYNC`` stays ``Half`` because the spec keeps it and ``PERF_RUN_TYPE``
    stays put because it is not pinnable at all.

    Note the name overstates this: ``dest_sync`` is kept here, not pinned.
    """
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
    assert dest_sync.dest_sync == DestSync.Half
    assert run_type.perf_run_type == PerfRunType.UNPACK_ISOLATE


def test_project_templates_math_keeps_dest_sync():
    """``MATH_ISOLATE`` observes every template, so projection is a no-op.

    Fidelity stays ``HiFi4``, dest sync stays ``Half``, throttle stays ``3``.
    The math unit is the one thread that reads all three, which is why the
    matmul policy buys nothing here and every math measurement costs a real run.
    """
    templates = [
        MATH_FIDELITY(MathFidelity.HiFi4),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(3),
        PERF_RUN_TYPE(PerfRunType.MATH_ISOLATE),
    ]
    projected = project_templates(templates, MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE])
    fidelity = next(p for p in projected if isinstance(p, MATH_FIDELITY))
    dest_sync = next(p for p in projected if isinstance(p, DEST_SYNC))
    throttle = next(p for p in projected if isinstance(p, THROTTLE_LEVEL))
    assert fidelity.math_fidelity == MathFidelity.HiFi4
    assert dest_sync.dest_sync == DestSync.Half
    assert throttle.throttle_level == 3


# ---------------------------------------------------------------------------
# 4. Projection: runtimes
#
# project_runtimes drops whole runtime types and individual fields.
# Under SPEED_OF_LIGHT these values are inlined into the compile
# header, so anything the kernel still reads must survive the drop.
# ---------------------------------------------------------------------------


def test_unpack_tilize_sol_pack_keeps_dim_tile_cnt_invariant():
    """Projection must not break the agreement between tile dims and tile count.

    ``INPUT_DIMENSIONS(2, 3, 3, 2)`` with ``TILE_COUNT(6)`` satisfies
    ``full_rt_dim * full_ct_dim == tile_cnt``, and it still does after
    projection. Pinning one of the two without the other would inline a compile
    header whose loop bound contradicts its geometry.
    """
    runtimes = [INPUT_DIMENSIONS(2, 3, 3, 2), TILE_COUNT(6), LOOP_FACTOR(256)]
    projected = project_runtimes(
        runtimes, UNPACK_TILIZE_RELEVANCE[PerfRunType.PACK_ISOLATE]
    )
    dims = next(p for p in projected if isinstance(p, INPUT_DIMENSIONS))
    tiles = next(p for p in projected if isinstance(p, TILE_COUNT))
    assert dims.full_rt_dim * dims.full_ct_dim == tiles.tile_cnt


def test_math_matmul_sol_isolates_keep_faces_partial_tile_dims():
    """Tiny-tile geometry survives projection for MATH and PACK isolates.

    Face count stays ``2``, the partial-face flag stays set, the narrow input
    row dimension stays ``16``, and the dest index stays ``0``. MATH_ISOLATE
    keeps the face count because its idle unpack/pack threads consume it during
    INIT before returning from TILE_LOOP.
    """
    runtimes = _math_matmul_runtimes(num_faces=2, partial=True, in0_r=16)
    for run_type in (PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE):
        projected = project_runtimes(runtimes, MATH_MATMUL_RELEVANCE[run_type])
        faces = next(p for p in projected if isinstance(p, NUM_FACES))
        partial = next(p for p in projected if isinstance(p, PARTIAL_FACE))
        dims = next(p for p in projected if isinstance(p, IN_TILE_DIMS))
        dest = next(p for p in projected if isinstance(p, DEST_INDEX))
        assert faces.num_faces == 2
        assert partial.partial_face_pack is True
        assert dims.in0_r_dim == 16
        assert dest.dst_index == 0


def test_pack_sol_pack_keeps_dest_index():
    """``DEST_INDEX`` reaches the pack thread, so ``PACK_ISOLATE`` keeps it.

    ``DEST_INDEX(1)`` is still ``1`` after projection. The packer reads its
    source register from this index, so canonicalising it to ``0`` would move
    what the measurement packs from.
    """
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


def test_project_runtimes_raises_on_enum_field_drop():
    """Dropping an individual field whose type is an ``Enum`` is refused, not guessed.

    The spec keeps ``UNPACK_TRANS_FACES`` as a type but lists none of its
    fields, so projection would have to invent a value for an enum member. The
    integer default it uses for plain fields is meaningless here, so it raises
    instead of compiling a kernel with an out-of-range enum.
    """
    spec = RunTypeRelevance(
        runtime_types=frozenset({UNPACK_TRANS_FACES}),
        runtime_fields=frozenset(),
    )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError, match="Enum"
    ):
        project_runtimes([UNPACK_TRANS_FACES(Transpose.No)], spec)


def test_project_runtimes_raises_without_default_constructor():
    """A runtime parameter that cannot be default-constructed is refused, not guessed.

    Dropping a whole runtime type means replacing it with a canonical instance,
    which requires a zero-argument constructor. ``_NoDefault`` has a required
    field, so projection raises rather than fabricating a value that would
    silently become part of the compile header.
    """

    @dataclass
    class _NoDefault(RuntimeParameter):
        x: int

        def convert_to_cpp(self) -> str:
            return ""

        def convert_to_struct_fields(self) -> tuple[str, str]:
            return "", ""

    spec = RunTypeRelevance(runtime_types=frozenset())
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError, match="no zero-argument"
    ):
        project_runtimes([_NoDefault(1)], spec)


# ---------------------------------------------------------------------------
# 5. Projection: formats
#
# project_formats pins the format fields a run type cannot observe.
# It must never pin unpack or math formats, because isolate INIT
# still runs those threads for real.
# ---------------------------------------------------------------------------


def test_project_formats_pins_pack_output_for_unpack():
    """For ``UNPACK_ISOLATE`` the output format is pinned and the input side is not.

    From ``Float32`` in and ``Bfp8_b`` out, the unpack and math formats stay
    ``Float32`` while ``pack_dst`` is rewritten to ``Float16``. The returned
    config is a copy: the caller's ``fmt`` still reads ``Bfp8_b``, which matters
    because the same object goes on to serve the full-fidelity measurement.
    """
    fmt = _format(DataFormat.Float32, DataFormat.Bfp8_b)
    projected = project_formats(fmt, PACK_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    assert projected is not fmt
    assert projected.unpack_A_src == DataFormat.Float32
    assert projected.unpack_A_dst == DataFormat.Float32
    assert projected.math == DataFormat.Float32
    assert projected.pack_dst == DataFormat.Float16
    assert fmt.pack_dst == DataFormat.Bfp8_b
    assert fmt.math == DataFormat.Float32


def test_project_formats_keeps_unpack_and_math_for_isolate_init():
    """Isolate INIT really runs unpack and math, so their formats are never pinned.

    Across all four run types the unpack and math formats stay ``Float32``;
    only ``UNPACK_ISOLATE`` has its ``pack_dst`` pinned to ``Float16``, while
    math, pack and congestion keep the caller's ``Bfp8_b``. This is a hardware
    constraint rather than a tidiness one: with ``dest_acc=Yes`` a pinned
    narrower input format makes isolate INIT hang on a real device.
    """
    fmt = _format(DataFormat.Float32, DataFormat.Bfp8_b)
    unpack = project_formats(fmt, PACK_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    math = project_formats(fmt, PACK_RELEVANCE[PerfRunType.MATH_ISOLATE])
    pack = project_formats(fmt, PACK_RELEVANCE[PerfRunType.PACK_ISOLATE])
    cong = project_formats(fmt, PACK_RELEVANCE[PerfRunType.L1_CONGESTION])
    for projected in (unpack, math, pack, cong):
        assert projected.unpack_A_src == DataFormat.Float32
        assert projected.unpack_A_dst == DataFormat.Float32
        assert projected.math == DataFormat.Float32
    assert unpack.pack_dst == DataFormat.Float16
    assert math.pack_dst == DataFormat.Bfp8_b
    assert pack.pack_dst == DataFormat.Bfp8_b
    assert cong.pack_dst == DataFormat.Bfp8_b


# ---------------------------------------------------------------------------
# 6. Projection: stimuli
#
# project_stimuli canonicalises L1 tile counts. It may only ever
# lower a count: raising one past what the caller allocated would
# overflow the L1 ring buffer on a real device.
# ---------------------------------------------------------------------------


def test_project_stimuli_never_raises_caller_clamp():
    """Projection may lower an L1 tile count but must never raise one.

    With ``c=16, r=1, k=32`` the per-operand arithmetic would suggest more tiles
    than the caller's 16, so every count is clamped back to 16. This is the
    guard against L1 overflow: the caller sized the ring buffer, and a projected
    count above that would have the kernel walk off the end of it on a real device.
    """
    spec = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    runtimes = [CRK_TILE_DIMM(c_dimm=16, r_dimm=1, k_dimm=32)]
    original = _stimuli(16, res_count=16)
    projected = project_stimuli(original, runtimes, spec)
    assert projected.tile_count_A == 16
    assert projected.tile_count_B == 16
    assert projected.tile_count_res == 16


def test_project_stimuli_pack_uses_per_operand_counts():
    """``PACK_ISOLATE`` cannot see the inner dimension, so counts collapse per operand.

    With ``k_dimm`` dropped, operand A needs only ``r * 1 = 1`` tile while B
    needs ``1 * c = 16`` and the result needs ``r * c = 16``. Each is then
    clamped to the caller's 16. Projecting all three to one number would either
    over-allocate A or under-allocate the result.
    """
    spec = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    runtimes = [CRK_TILE_DIMM(c_dimm=16, r_dimm=1, k_dimm=32)]
    original = _stimuli(16, res_count=16)
    projected = project_stimuli(original, runtimes, spec)
    # k_dimm is dropped: A=r×1, B=1×c, Res=r×c, then clamped to the original.
    assert projected.tile_count_A == 1
    assert projected.tile_count_B == 16
    assert projected.tile_count_res == 16


def test_project_stimuli_math_matmul_keeps_multiblock_result_count():
    """CRK inputs are reused across blocks while result storage is not.

    A 2x2x1 matmul needs two A tiles and two B tiles regardless of block count,
    but four destination-handoff blocks need 2x2x4 = 16 result tiles. Dropping
    that multiplier would move the projected result buffer into its inputs.
    """
    runtimes = _math_matmul_runtimes(num_blocks=4)
    original = _stimuli(2, res_count=16)
    projected = project_stimuli(
        original, runtimes, MATH_MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    )
    assert projected.tile_count_A == 2
    assert projected.tile_count_B == 2
    assert projected.tile_count_res == 16


def test_project_stimuli_block_fields_are_order_independent():
    """Raw and runtime-projected block shapes produce the same L1 counts.

    ``num_blocks`` is invisible in this synthetic policy, so it folds to one
    whether ``project_stimuli`` sees the original value four or the already
    projected runtime. ``num_tiles_in_block`` remains three.
    """
    spec = RunTypeRelevance(
        runtime_types=frozenset({NUM_BLOCKS, NUM_TILES_IN_BLOCK}),
        runtime_fields=frozenset({"num_tiles_in_block"}),
    )
    runtimes = [NUM_BLOCKS(4), NUM_TILES_IN_BLOCK(3)]
    projected_runtimes = project_runtimes(runtimes, spec)
    original = _stimuli(12)
    from_raw = project_stimuli(original, runtimes, spec)
    from_projected = project_stimuli(original, projected_runtimes, spec)
    assert from_raw.tile_count_A == 3
    assert from_raw.tile_count_B == 3
    assert from_raw.tile_count_res == 3
    assert from_projected.tile_count_A == from_raw.tile_count_A
    assert from_projected.tile_count_B == from_raw.tile_count_B
    assert from_projected.tile_count_res == from_raw.tile_count_res


def test_project_stimuli_pack_relevance_keeps_block_layout():
    """With no CRK dimensions in the spec, the block layout governs instead.

    ``PACK_RELEVANCE`` describes work as blocks and tiles-per-block rather than
    c/r/k, so there is no per-operand arithmetic to apply and all three counts
    stay at the caller's 4 for both ``UNPACK_ISOLATE`` and ``L1_CONGESTION``.
    """
    runtimes = [NUM_BLOCKS(2), NUM_TILES_IN_BLOCK(2)]
    original = _stimuli(4)
    for run_type in (PerfRunType.UNPACK_ISOLATE, PerfRunType.L1_CONGESTION):
        projected = project_stimuli(original, runtimes, PACK_RELEVANCE[run_type])
        assert projected.tile_count_A == 4
        assert projected.tile_count_B == 4
        assert projected.tile_count_res == 4


# ---------------------------------------------------------------------------
# 7. execute_key: identity fields
#
# Fields that land in the cache key for every spec. Two
# measurements that differ in any of these must never share a
# cached TILE_LOOP result.
# ---------------------------------------------------------------------------


def test_execute_key_keeps_unpinnable_templates():
    """The key can only drop a template that ``pin_template`` is able to pin.

    ``spec.templates`` lists what to keep, but that is not the whole story: with
    a spec keeping only ``DEST_SYNC``, differing ``INPUT_DIMENSIONS`` still miss
    because that type has no canonical value and stays in the compile header,
    while differing ``MATH_FIDELITY`` hits because it does. Reading the spec
    alone would predict a hit in both cases.
    """
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


def test_execute_key_includes_unpack_to_dest_and_l1_acc():
    """``unpack_to_dest`` and ``l1_acc`` are always part of the key, whatever the spec.

    Both reroute the dataflow itself rather than tuning it, so no run type can be
    said to be blind to them, and two cases differing in either must never share
    a measurement.
    """
    t, runtimes = _matmul_params(MathFidelity.LoFi)
    spec = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    base = _execute_kwargs("perf_matmul", t, runtimes)
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        spec,
        {**base, "unpack_to_dest": False},
        {**base, "unpack_to_dest": True},
    )
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        spec,
        {**base, "l1_acc": L1Accumulation.No},
        {**base, "l1_acc": L1Accumulation.Yes},
    )


def test_execute_key_includes_unpack_to_srcs():
    """``unpack_to_srcs`` changes tile size and buffer addresses, so it is always keyed.

    It is the SrcS sibling of ``unpack_to_dest``. A hit across the two modes would
    replay one L1 layout for the other.
    """
    t, runtimes = _matmul_params(MathFidelity.LoFi)
    spec = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    base = _execute_kwargs("perf_matmul", t, runtimes)
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        spec,
        {**base, "unpack_to_srcs": False},
        {**base, "unpack_to_srcs": True},
    )


def test_execute_key_includes_run_count_and_source():
    """Iteration count and owning module are part of the measurement's identity.

    ``run_count`` is how many iterations the reported statistics average over,
    so a 1-iteration result cannot stand in for a 2-iteration one. ``source``
    namespaces the cache per perf module, so two suites that happen to share a
    parameter shape but compile different kernels stay separate.
    """
    templates, runtimes = _matmul_params(MathFidelity.LoFi)
    spec = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    base = _execute_kwargs("perf_matmul", templates, runtimes)
    assert_key_miss(
        PerfRunType.UNPACK_ISOLATE,
        spec,
        {**base, "run_count": 1},
        {**base, "run_count": 2},
    )
    assert_key_miss(
        PerfRunType.UNPACK_ISOLATE,
        spec,
        {**base, "source": "perf_matmul.py"},
        {**base, "source": "perf_math_matmul.py"},
    )


def test_execute_key_includes_stimuli_fingerprint():
    """The L1 tile counts reach the key through ``stimuli_key``.

    4 tiles versus 16 miss even for ``PACK_ISOLATE``, because the counts survive
    projection here and they set how much data the run moves through L1.
    """
    templates, runtimes = _matmul_params(MathFidelity.LoFi)
    spec = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    base = _execute_kwargs("perf_matmul", templates, runtimes)
    assert_key_miss(
        PerfRunType.PACK_ISOLATE,
        spec,
        {**base, "stimuli": _stimuli(4)},
        {**base, "stimuli": _stimuli(16)},
    )


def test_relevance_source_uses_explicit_module(monkeypatch):
    """``relevance_source`` overrides the inferred module in the key's first slot.

    A suite that borrows another's policy passes the owning module explicitly, so
    the cache namespace follows the relevance map rather than the calling file.
    Here ``MATMUL_RELEVANCE`` is used under ``relevance_source="perf_pack"`` and
    ``key[0]`` reports ``perf_pack``.
    """
    _patch_hwfree_classvars(monkeypatch)
    templates, runtimes = _matmul_params(MathFidelity.LoFi)
    cfg = PerfConfig(
        test_name="perf_relevance",
        formats=None,
        run_types=_MATMUL_RUN_TYPES,
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.No,
        relevance=MATMUL_RELEVANCE,
        relevance_source="perf_pack",
    )
    assert cfg.relevance_source == "perf_pack"
    key = cfg._execute_cache_key(*_run_config(cfg, PerfRunType.PACK_ISOLATE))
    assert key[0] == "perf_pack"


def test_hashable_raises_instead_of_repr():
    """An unhashable key field raises rather than falling back to ``repr()``.

    Both fallbacks are wrong, which is why there is no fallback. A default
    ``repr`` carries a memory address, so equal values would look different and
    never reuse; a stable stand-in like the class name would make genuinely
    different values collide and share one measurement. Failing loudly turns
    either silent bug into a visible one.
    """

    class _Addr:
        pass

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError, match="Cannot hash"
    ):
        _hashable(_Addr())


# ---------------------------------------------------------------------------
# 8. execute_key: one subsection per relevance policy
#
# The payoff tests. For each perf suite, a parameter sweep either
# collapses onto one key (hit, one device run) or stays distinct
# (miss, one run each) depending on what the run type can observe.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8a. MATMUL_RELEVANCE (perf_matmul)
# ---------------------------------------------------------------------------


def test_execute_key_keeps_dest_sync_for_math():
    """Every isolate observes ``DEST_SYNC``, so ``Half`` and ``Full`` never share a key.

    Unpack, math and pack all miss. Dest sync sets the half/full double-buffer
    handshake on the dest register, so all three threads change their stalling
    behaviour with it and none may be collapsed.
    """
    t_half, runtimes = _matmul_params(MathFidelity.LoFi)
    t_full = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Full),
        THROTTLE_LEVEL(),
    ]
    unpack = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    math = MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE]
    pack = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    kwargs = dict(
        test_name="perf_matmul",
        dest_acc=DestAccumulation.No,
        runtimes=runtimes,
        formats=None,
        speed_of_light=False,
    )
    for run_type, spec in (
        (PerfRunType.UNPACK_ISOLATE, unpack),
        (PerfRunType.MATH_ISOLATE, math),
        (PerfRunType.PACK_ISOLATE, pack),
    ):
        assert_key_miss(
            run_type,
            spec,
            {**kwargs, "templates": t_half},
            {**kwargs, "templates": t_full},
        )


def test_execute_key_drops_fidelity_for_unpack_keeps_for_math():
    """The headline saving: fidelity is a math-unit setting, invisible to the unpacker.

    ``LoFi`` and ``HiFi4`` share one key for ``UNPACK_ISOLATE``, so a fidelity
    sweep costs one unpack measurement instead of one per point. ``MATH_ISOLATE``
    is exactly where the extra passes happen, so it must miss.
    """
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
    """The matmul inner dimension is invisible to the packer but not to the pipeline.

    ``kt=1`` and ``kt=32`` share one key for ``PACK_ISOLATE``, because the packer
    writes the same ``r x c`` result either way. ``L1_TO_L1`` measures the whole
    pipeline including the 32x longer accumulation, so it must miss.
    """
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


def test_matmul_congestion_misses_on_throttle():
    """Throttling is only observable where threads contend for L1.

    ``THROTTLE_LEVEL`` 0 versus 5 misses for ``L1_CONGESTION`` and hits for both
    ``UNPACK_ISOLATE`` and ``PACK_ISOLATE``. Throttle inserts math-unit stalls,
    which change the contention the congestion run is built to measure but not
    the work an isolated unpacker or packer does.
    """
    t_0, runtimes = _matmul_params(MathFidelity.LoFi)
    t_5 = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(5),
    ]
    cong = MATMUL_RELEVANCE[PerfRunType.L1_CONGESTION]
    unpack = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    pack = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    kwargs = dict(
        test_name="perf_matmul",
        dest_acc=DestAccumulation.No,
        runtimes=runtimes,
        formats=None,
        speed_of_light=False,
    )
    assert_key_miss(
        PerfRunType.L1_CONGESTION,
        cong,
        {**kwargs, "templates": t_0},
        {**kwargs, "templates": t_5},
    )
    assert_key_hit(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        {**kwargs, "templates": t_0},
        {**kwargs, "templates": t_5},
    )
    assert_key_hit(
        PerfRunType.PACK_ISOLATE,
        pack,
        {**kwargs, "templates": t_0},
        {**kwargs, "templates": t_5},
    )


# ---------------------------------------------------------------------------
# 8b. MATH_MATMUL_RELEVANCE (perf_math_matmul)
# ---------------------------------------------------------------------------


def test_math_matmul_num_blocks_changes_isolate_keys():
    """Block count is visible to every run type, so it buys no reuse at all.

    1 versus 4 blocks misses for unpack, math, pack and congestion alike: all
    three threads loop once per block, so the amount of work each does scales
    with it.
    """
    templates = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(0),
    ]
    shared = [
        UNPACK_TRANS_FACES(Transpose.No),
        NUM_FACES(),
        LOOP_FACTOR(64),
        CRK_TILE_DIMM(c_dimm=2, r_dimm=2, k_dimm=1),
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
    """Fidelity reuse for the data-movement threads; congestion pays full price.

    Unpack and pack both hit across ``LoFi`` and ``HiFi4``, while math misses as
    usual. Congestion also misses, unlike the plain matmul policy in 8a: this
    suite's congestion run includes the math unit, so the extra fidelity passes
    change the contention it measures.
    """
    t_lo, runtimes = _matmul_params(MathFidelity.LoFi)
    t_hi, _ = _matmul_params(MathFidelity.HiFi4)
    runtimes = runtimes + [NUM_BLOCKS(1)]
    unpack = MATH_MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    pack = MATH_MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    math = MATH_MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE]
    cong = MATH_MATMUL_RELEVANCE[PerfRunType.L1_CONGESTION]
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
    assert_key_miss(
        PerfRunType.L1_CONGESTION,
        cong,
        {**base, "templates": t_lo},
        {**base, "templates": t_hi},
    )


def test_math_matmul_pack_keeps_faces_partial_tile_dims():
    """Face count, partial faces and input tile dims all reach the pack thread.

    Each of the three, varied alone from the baseline, misses for both
    ``PACK_ISOLATE`` and ``L1_CONGESTION``. They set the geometry the packer
    walks, so none can be canonicalised away. Section 4 asserts the matching
    projection side: these fields also survive ``project_runtimes``.
    """
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


# ---------------------------------------------------------------------------
# 8c. PACK_RELEVANCE (perf_pack)
# ---------------------------------------------------------------------------


def test_pack_relu_hits_unpack_math_misses_pack_cong():
    """The relu setting is applied at pack time, so only the unpacker is blind to it.

    ``RELU_CONFIG`` 0 versus 1 hits for ``UNPACK_ISOLATE`` and misses for math,
    pack and congestion. Math misses because ``PACK_RELEVANCE`` leaves that run
    type at full fidelity rather than because the math unit reads the setting.

    Note the name overstates this: math is in the miss list, not the hit list.
    """
    templates = [DEST_SYNC(DestSync.Half)]
    shared = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
    ]
    rt_off = shared + [RELU_CONFIG(0)]
    rt_on = shared + [RELU_CONFIG(1)]
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


def test_pack_dest_index_hits_unpack_misses_pack_math_cong():
    """The dest register index is visible everywhere except the isolated unpacker.

    ``DEST_INDEX`` 0 versus 1 hits for ``UNPACK_ISOLATE`` and misses for math,
    pack and congestion, which all address dest by this index.
    """
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


def test_pack_unpack_misses_dest_sync():
    """``DEST_SYNC`` is observable even by the isolated unpacker.

    Worth stating separately because ``UNPACK_ISOLATE`` is the run type that hits
    on nearly everything else in this policy: relu and dest index both collapse
    for it, but the dest half/full handshake does not.
    """
    shared = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]
    unpack = PACK_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert_key_miss(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        _execute_kwargs("perf_pack", [DEST_SYNC(DestSync.Half)], shared),
        _execute_kwargs("perf_pack", [DEST_SYNC(DestSync.Full)], shared),
    )


def test_execute_key_l1_to_l1_keeps_s_format_fields():
    """The optional S operand feeds the unpacker, so only the packer can ignore it.

    Changing ``unpack_S_src`` from ``Float16`` to ``Float32`` misses for
    ``L1_TO_L1`` and ``UNPACK_ISOLATE`` and hits for ``PACK_ISOLATE``. The S
    buffer never reaches the pack thread, so its format cannot change what a
    pack measurement does.
    """
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [LOOP_FACTOR(32)]
    fmt_a = FormatConfig(
        unpack_A_src=DataFormat.Float16,
        unpack_A_dst=DataFormat.Float16,
        pack_src=DataFormat.Float16,
        pack_dst=DataFormat.Float16,
        math=DataFormat.Float16,
        unpack_S_src=DataFormat.Float16,
    )
    fmt_b = FormatConfig(
        unpack_A_src=DataFormat.Float16,
        unpack_A_dst=DataFormat.Float16,
        pack_src=DataFormat.Float16,
        pack_dst=DataFormat.Float16,
        math=DataFormat.Float16,
        unpack_S_src=DataFormat.Float32,
    )
    l1 = PACK_RELEVANCE[PerfRunType.L1_TO_L1]
    pack = PACK_RELEVANCE[PerfRunType.PACK_ISOLATE]
    unpack = PACK_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    assert_key_miss(
        PerfRunType.L1_TO_L1,
        l1,
        _execute_kwargs("perf_pack", templates, runtimes, fmt_a),
        _execute_kwargs("perf_pack", templates, runtimes, fmt_b),
    )
    assert_key_miss(
        PerfRunType.UNPACK_ISOLATE,
        unpack,
        _execute_kwargs("perf_pack", templates, runtimes, fmt_a),
        _execute_kwargs("perf_pack", templates, runtimes, fmt_b),
    )
    assert_key_hit(
        PerfRunType.PACK_ISOLATE,
        pack,
        _execute_kwargs("perf_pack", templates, runtimes, fmt_a),
        _execute_kwargs("perf_pack", templates, runtimes, fmt_b),
    )


# ---------------------------------------------------------------------------
# 8d. PACK_UNTILIZE_RELEVANCE (perf_pack_untilize)
# ---------------------------------------------------------------------------


def test_pack_untilize_input_format_reuses_pack_not_l1():
    """Two input formats of the same width collapse for pack but not for the pipeline.

    ``Float16`` and ``Float16_b`` differ only in exponent bias, so the packer's
    work is identical and the two share a key. ``L1_TO_L1`` still misses, because
    the unpacker and math unit do handle them differently.
    """
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
    """Equal tile counts are not equal work when the geometry differs.

    ``4x5`` and ``5x4`` both total 20 tiles, yet both ``L1_CONGESTION`` and
    ``PACK_ISOLATE`` miss. The row and column split sets the address stride the
    untilize walks, so collapsing on the product alone would report one shape's
    timing for the other.
    """
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


# ---------------------------------------------------------------------------
# 8e. UNPACK_TILIZE_RELEVANCE (perf_unpack_tilize)
# ---------------------------------------------------------------------------


def test_unpack_tilize_output_format_reuses_unpack():
    """The output format is downstream of the unpacker, so it collapses here.

    ``Float16`` and ``Float32`` outputs share one key with the input format held
    fixed, which is the mirror image of the pack-untilize case in 8d.
    """
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
    """The pack thread reads an unpack format here, so the key must keep it.

    Both configs present the same ``pack_src`` and ``pack_dst``, so a policy
    keyed on pack formats alone would call this a hit. It must miss: PACK INIT's
    ``skip_bh_tilize_workaround`` branches on ``unpack_A_src``, making that field
    genuinely observable from the pack side. The body comment records the same
    reasoning at the assertion.
    """
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
    """Equal tile counts with different geometry miss for both threads.

    ``2x4`` and ``4x2`` both total 8 tiles, and both ``PACK_ISOLATE`` and
    ``UNPACK_ISOLATE`` miss. The tilize walk order depends on the split, so the
    product alone is not a sufficient key.
    """
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


# ---------------------------------------------------------------------------
# 9. variant_id and ELF reuse under SPEED_OF_LIGHT
#
# A separate dedup axis from the execute cache: variant_id decides
# whether two cases share one compiled ELF. Under SoL the projected
# formats and stimuli feed the compile header, so they feed the hash.
# ---------------------------------------------------------------------------


def test_sol_unpack_variant_hash_reuses_irrelevant_pack_format(monkeypatch):
    """One ELF serves both output formats, and applying it leaves no residue.

    Two configs differing only in pack output reach the same ``variant_id``,
    because projection pins ``pack_dst`` to ``Float16`` in the config that feeds
    the compile header while ``passed_formats_config`` keeps each caller's real
    ``Float16``/``Float32`` for reporting.

    The middle assertion is the subtle one: the ``L1_TO_L1`` key computed after
    applying the unpack run config must equal the key computed on a fresh
    config. Projection state must not leak from one run type into the next.
    Applying the full-fidelity ``L1_TO_L1`` config then separates the two again.
    """
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]
    common = dict(
        test_name="perf_pack",
        run_types=[PerfRunType.UNPACK_ISOLATE, PerfRunType.L1_TO_L1],
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.No,
        disable_format_inference=True,
        relevance=PACK_RELEVANCE,
    )
    cfg_f16 = PerfConfig(
        formats=_format(DataFormat.Float16, DataFormat.Float16), **common
    )
    cfg_f32 = PerfConfig(
        formats=_format(DataFormat.Float16, DataFormat.Float32), **common
    )

    unpack_f16 = _run_config(cfg_f16, PerfRunType.UNPACK_ISOLATE)
    unpack_f32 = _run_config(cfg_f32, PerfRunType.UNPACK_ISOLATE)
    cfg_f16._apply_run_config(*unpack_f16)
    cfg_f32._apply_run_config(*unpack_f32)
    assert cfg_f16.variant_id == cfg_f32.variant_id
    assert cfg_f16.formats_config[0].pack_dst == DataFormat.Float16
    assert cfg_f16.passed_formats_config[0].pack_dst == DataFormat.Float16
    assert cfg_f32.passed_formats_config[0].pack_dst == DataFormat.Float32
    # Produce leftovers must not leak into consume keys.
    l1_f32 = _run_config(cfg_f32, PerfRunType.L1_TO_L1)
    key_after_unpack = cfg_f32._execute_cache_key(l1_f32[0], l1_f32[1], l1_f32[2])
    cfg_fresh = PerfConfig(
        formats=_format(DataFormat.Float16, DataFormat.Float32), **common
    )
    key_fresh = cfg_fresh._execute_cache_key(l1_f32[0], l1_f32[1], l1_f32[2])
    assert key_after_unpack == key_fresh

    l1_f16 = _run_config(cfg_f16, PerfRunType.L1_TO_L1)
    cfg_f16._apply_run_config(*l1_f16)
    cfg_f32._apply_run_config(*l1_f32)
    assert cfg_f16.variant_id != cfg_f32.variant_id


def test_sol_unpack_variant_hash_reuses_stimuli_res_format(monkeypatch):
    """Stimuli formats follow the same rule as format configs, and feed the same hash.

    ``UNPACK_ISOLATE`` pins ``stimuli_res_format`` to ``Float16`` for both
    configs, so they share one ELF. ``PACK_ISOLATE`` keeps each config's own
    result format, because the packer writes it, so those two hash apart.
    """
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]

    def _cfg(out_fmt):
        return PerfConfig(
            test_name="perf_pack",
            formats=_format(DataFormat.Float16, out_fmt),
            run_types=[PerfRunType.UNPACK_ISOLATE, PerfRunType.PACK_ISOLATE],
            templates=templates,
            runtimes=runtimes,
            variant_stimuli=StimuliConfig(
                None,
                DataFormat.Float16,
                None,
                DataFormat.Float16,
                out_fmt,
                tile_count_A=4,
                tile_count_B=4,
                tile_count_res=4,
            ),
            dest_acc=DestAccumulation.No,
            disable_format_inference=True,
            relevance=PACK_RELEVANCE,
        )

    cfg_f16 = _cfg(DataFormat.Float16)
    cfg_f32 = _cfg(DataFormat.Float32)
    unpack_f16 = _run_config(cfg_f16, PerfRunType.UNPACK_ISOLATE)
    unpack_f32 = _run_config(cfg_f32, PerfRunType.UNPACK_ISOLATE)
    cfg_f16._apply_run_config(*unpack_f16)
    cfg_f32._apply_run_config(*unpack_f32)
    assert cfg_f16.variant_stimuli.stimuli_res_format == DataFormat.Float16
    assert cfg_f32.variant_stimuli.stimuli_res_format == DataFormat.Float16
    assert cfg_f16.variant_id == cfg_f32.variant_id

    pack_f16 = _run_config(cfg_f16, PerfRunType.PACK_ISOLATE)
    pack_f32 = _run_config(cfg_f32, PerfRunType.PACK_ISOLATE)
    cfg_f16._apply_run_config(*pack_f16)
    cfg_f32._apply_run_config(*pack_f32)
    assert cfg_f16.variant_stimuli.stimuli_res_format == DataFormat.Float16
    assert cfg_f32.variant_stimuli.stimuli_res_format == DataFormat.Float32
    assert cfg_f16.variant_id != cfg_f32.variant_id


def test_sol_dest_acc_yes_keeps_unpack_and_math_formats(monkeypatch):
    """With ``dest_acc=Yes``, pinning the input formats would time out on real hardware.

    Across all three isolates the unpack and math formats stay ``Float32`` and
    ``unpack_size_a`` stays the ``Float32`` tile size, so isolate INIT keeps
    unpacking full-width data to dest. Only the output side is pinned, which is
    why ``UNPACK_ISOLATE`` shares one ``variant_id`` across ``Float16`` and
    ``Bfp8_b`` outputs while math and pack isolate each keep their own.
    """
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]
    common = dict(
        test_name="perf_pack",
        run_types=[
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        relevance=PACK_RELEVANCE,
    )
    cfg_f16 = PerfConfig(
        formats=_format(DataFormat.Float32, DataFormat.Float16), **common
    )
    cfg_bfp = PerfConfig(
        formats=_format(DataFormat.Float32, DataFormat.Bfp8_b), **common
    )
    for run_type in (
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
    ):
        cfg_f16._apply_run_config(*_run_config(cfg_f16, run_type))
        fmt = cfg_f16.formats_config[0]
        assert fmt.unpack_A_src == DataFormat.Float32
        assert fmt.math == DataFormat.Float32
        assert cfg_f16.unpack_size_a == TestConfig.TILE_SIZES[DataFormat.Float32]

    unpack_f16 = _run_config(cfg_f16, PerfRunType.UNPACK_ISOLATE)
    unpack_bfp = _run_config(cfg_bfp, PerfRunType.UNPACK_ISOLATE)
    cfg_f16._apply_run_config(*unpack_f16)
    cfg_bfp._apply_run_config(*unpack_bfp)
    assert cfg_f16.variant_id == cfg_bfp.variant_id
    assert cfg_f16.formats_config[0].math == DataFormat.Float32
    assert cfg_f16.formats_config[0].pack_dst == DataFormat.Float16

    math_f16 = _run_config(cfg_f16, PerfRunType.MATH_ISOLATE)
    math_bfp = _run_config(cfg_bfp, PerfRunType.MATH_ISOLATE)
    cfg_f16._apply_run_config(*math_f16)
    cfg_bfp._apply_run_config(*math_bfp)
    assert cfg_f16.variant_id != cfg_bfp.variant_id
    assert cfg_f16.formats_config[0].unpack_A_src == DataFormat.Float32
    assert cfg_f16.formats_config[0].pack_dst == DataFormat.Float16
    assert cfg_bfp.formats_config[0].pack_dst == DataFormat.Bfp8_b

    pack_f16 = _run_config(cfg_f16, PerfRunType.PACK_ISOLATE)
    pack_bfp = _run_config(cfg_bfp, PerfRunType.PACK_ISOLATE)
    cfg_f16._apply_run_config(*pack_f16)
    cfg_bfp._apply_run_config(*pack_bfp)
    assert cfg_f16.variant_id != cfg_bfp.variant_id
    assert cfg_f16.formats_config[0].unpack_A_src == DataFormat.Float32
    assert cfg_f16.formats_config[0].math == DataFormat.Float32
    assert cfg_f16.formats_config[0].pack_dst == DataFormat.Float16
    assert cfg_bfp.formats_config[0].pack_dst == DataFormat.Bfp8_b


def test_sol_pack_isolate_reuses_variant_id_across_kt_with_stimuli(monkeypatch):
    """ELF reuse across an inner-dimension sweep, with stimuli projected to match.

    ``kt=1`` with 4 tiles and ``kt=32`` with 16 reach one ``variant_id`` for
    ``PACK_ISOLATE`` and one projected ``tile_count_A``, so the sweep compiles
    once. ``UNPACK_ISOLATE`` sees ``kt``, so it hashes apart and compiles per
    point. This is the build-dedup counterpart to the key-level result in 8a.
    """
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    templates = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(),
    ]

    def _cfg(kt, tiles):
        runtimes = [
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(2 * 2 * kt),
            CRK_TILE_DIMM(c_dimm=2, r_dimm=2, k_dimm=kt),
        ]
        return PerfConfig(
            test_name="perf_matmul",
            formats=_format(DataFormat.Float16, DataFormat.Float16),
            run_types=[PerfRunType.PACK_ISOLATE, PerfRunType.UNPACK_ISOLATE],
            templates=templates,
            runtimes=runtimes,
            variant_stimuli=_stimuli(tiles, res_count=4),
            dest_acc=DestAccumulation.No,
            disable_format_inference=True,
            relevance=MATMUL_RELEVANCE,
        )

    cfg_k1 = _cfg(1, 4)
    cfg_k32 = _cfg(32, 16)
    pack_k1 = _run_config(cfg_k1, PerfRunType.PACK_ISOLATE)
    pack_k32 = _run_config(cfg_k32, PerfRunType.PACK_ISOLATE)
    cfg_k1._apply_run_config(*pack_k1)
    cfg_k32._apply_run_config(*pack_k32)
    assert cfg_k1.variant_id == cfg_k32.variant_id
    assert cfg_k1.variant_stimuli.tile_count_A == cfg_k32.variant_stimuli.tile_count_A

    unpack_k1 = _run_config(cfg_k1, PerfRunType.UNPACK_ISOLATE)
    unpack_k32 = _run_config(cfg_k32, PerfRunType.UNPACK_ISOLATE)
    cfg_k1._apply_run_config(*unpack_k1)
    cfg_k32._apply_run_config(*unpack_k32)
    assert cfg_k1.variant_id != cfg_k32.variant_id


def test_sol_run_keeps_caller_pack_src_patch(monkeypatch):
    """A format the caller patched by hand after construction survives projection.

    ``pack_src`` is set to ``Bfp8_b`` directly on the config, then a full run and
    a further ``_apply_run_config`` both leave it there. Projection has to copy
    and rewrite format configs, and rebuilding one from the original constructor
    arguments instead would quietly discard edits like this.
    """
    elf_calls = []
    seeds = {"n": 0}
    templates = [DEST_SYNC(DestSync.Half)]
    runtimes = [
        NUM_BLOCKS(1),
        NUM_TILES_IN_BLOCK(1),
        LOOP_FACTOR(32),
        NUM_FACES(),
        RELU_CONFIG(0),
    ]
    cfg = PerfConfig(
        test_name="perf_pack",
        formats=_format(DataFormat.Float16, DataFormat.Float32),
        run_types=[PerfRunType.L1_TO_L1],
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.No,
        disable_format_inference=True,
        relevance=PACK_RELEVANCE,
    )
    for fmt in cfg.formats_config:
        fmt.pack_src = DataFormat.Bfp8_b
    _stub_hw(monkeypatch, cfg, elf_calls, seeds)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    cfg.run(PerfReport(), run_count=1)
    assert cfg.formats_config[0].pack_src == DataFormat.Bfp8_b
    l1 = _run_config(cfg, PerfRunType.L1_TO_L1)
    cfg._apply_run_config(*l1)
    assert cfg.formats_config[0].pack_src == DataFormat.Bfp8_b


def test_sol_run_keeps_caller_stimuli_patch(monkeypatch):
    """A stimuli edit made after construction survives projection and restore.

    ``passed_stimuli`` used to be copied only in ``__init__``, so a later
    change to ``variant_stimuli`` was overwritten when ``run()`` restored the
    stale snapshot.
    """
    elf_calls = []
    seeds = {"n": 0}
    cfg = PerfConfig(
        test_name="perf_pack",
        formats=_format(DataFormat.Float16, DataFormat.Float16),
        run_types=[PerfRunType.L1_TO_L1],
        templates=[DEST_SYNC(DestSync.Half)],
        runtimes=[LOOP_FACTOR(32)],
        variant_stimuli=_stimuli(4),
        dest_acc=DestAccumulation.No,
        relevance=PACK_RELEVANCE,
    )
    cfg.variant_stimuli.tile_count_res = 8
    _stub_hw(monkeypatch, cfg, elf_calls, seeds)
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    cfg.run(PerfReport(), run_count=1)
    assert cfg.variant_stimuli.tile_count_res == 8
    assert cfg.passed_stimuli.tile_count_res == 8


def test_sol_refresh_tile_sizes_keeps_narrow_tile_rescale(monkeypatch):
    """Narrow tiles stay narrow through projection and through a manual size refresh.

    With 2 faces and a 16-row input, the tile size is half the full ``Float16``
    tile, and it holds after applying ``L1_TO_L1``, ``MATH_ISOLATE``,
    ``PACK_ISOLATE``, and a direct ``_refresh_tile_sizes`` on the passed
    parameters. MATH matters because its idle unpack/pack threads run INIT with
    this geometry before returning.
    """
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", True)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    templates = [
        MATH_FIDELITY(MathFidelity.LoFi),
        DEST_SYNC(DestSync.Half),
        THROTTLE_LEVEL(),
    ]
    runtimes = _math_matmul_runtimes(num_faces=2, in0_r=16)
    cfg = PerfConfig(
        test_name="perf_math_matmul",
        formats=_format(DataFormat.Float16, DataFormat.Float16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=templates,
        runtimes=runtimes,
        variant_stimuli=_stimuli(4, face_r_dim=16),
        dest_acc=DestAccumulation.No,
        disable_format_inference=True,
        relevance=MATH_MATMUL_RELEVANCE,
    )
    expected = (TestConfig.TILE_SIZES.get(DataFormat.Float16, 128) // 2) * (16 // 16)
    assert cfg.pack_size == expected
    assert cfg.unpack_size_a == expected
    l1 = _run_config(cfg, PerfRunType.L1_TO_L1)
    cfg._apply_run_config(*l1)
    assert cfg.pack_size == expected
    assert cfg.unpack_size_a == expected
    math = _run_config(cfg, PerfRunType.MATH_ISOLATE)
    cfg._apply_run_config(*math)
    assert cfg.pack_size == expected
    assert cfg.unpack_size_a == expected
    pack = _run_config(cfg, PerfRunType.PACK_ISOLATE)
    cfg._apply_run_config(*pack)
    assert cfg.pack_size == expected
    cfg.formats_config = cfg.passed_formats_config
    cfg.variant_stimuli = cfg.passed_stimuli
    cfg._refresh_tile_sizes(cfg.passed_templates + cfg.passed_runtimes)
    assert cfg.pack_size == expected
    assert cfg.unpack_size_a == expected


def test_refresh_tile_sizes_rescales_without_formats(monkeypatch):
    """The same narrow-tile rescale happens with no formats and no relevance map.

    The baseline for the test above: ``PerfConfig`` derives the size from the
    runtime geometry itself, so the behaviour belongs to the config rather than
    to projection.
    """
    _patch_hwfree_classvars(monkeypatch)
    cfg = PerfConfig(
        test_name="perf_relevance",
        formats=None,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=_math_matmul_runtimes(num_faces=2, in0_r=16),
        variant_stimuli=_stimuli(4, face_r_dim=16),
        dest_acc=DestAccumulation.No,
    )
    assert cfg.pack_size == (128 // 2) * (16 // 16)
    assert cfg.unpack_size_a == cfg.pack_size


# ---------------------------------------------------------------------------
# 10. Cache effect end-to-end
#
# Full cfg.run() through the stubbed device, counting real device
# runs in elf_calls. The two baselines come first: with relevance
# off, nothing may be skipped.
# ---------------------------------------------------------------------------


def test_default_relevance_does_not_skip_device(monkeypatch):
    """Baseline: with no relevance map, every run type runs for every case.

    Two cases differing in fidelity produce two device runs for all five run
    types. Nothing may be skipped by default, so a suite that never opts in
    cannot be affected by any of this.
    """
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, relevance=None)
    _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds, relevance=None)
    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 2
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2


def test_disable_perf_relevance_env_does_not_skip_device(monkeypatch):
    """Baseline: the env kill switch restores the same unoptimised behaviour.

    Identical expectations to the no-map case, but reached with a map present
    and ``LLK_DISABLE_PERF_RELEVANCE=1``. This is the escape hatch for
    bisecting a suspected relevance bug on real hardware.
    """
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


def test_fidelity_change_reuses_unpack_and_pack(monkeypatch):
    """End to end: a fidelity sweep costs one unpack and one pack run instead of two.

    Math, congestion and ``L1_TO_L1`` still run twice. The replayed frame carries
    identical unpack means for both INIT and TILE_LOOP, which is what proves
    reuse given each real measurement gets its own seed. The reported
    ``math_fidelity`` column still differs per row: a cached measurement is
    stitched into a row that reports its own parameters.
    """
    elf_calls = []
    seeds = {"n": 0}
    frame_lo = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    frame_hi = _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)

    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.L1_CONGESTION) == 2
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    assert elf_calls.count(PerfRunType.L1_TO_L1) == 2

    unpack_col = stat_column("UNPACK_ISOLATE", MEAN)
    for marker in ("INIT", "TILE_LOOP"):
        lo = frame_lo[frame_lo[MARKER] == marker]
        hi = frame_hi[frame_hi[MARKER] == marker]
        assert not lo.empty
        assert lo[unpack_col].tolist() == hi[unpack_col].tolist()
    assert frame_lo["math_fidelity"].iloc[0] == MathFidelity.LoFi
    assert frame_hi["math_fidelity"].iloc[0] == MathFidelity.HiFi4


def test_kt_change_reuses_pack_only(monkeypatch):
    """End to end: a ``kt`` sweep saves the pack run and nothing else.

    Pack runs once across ``kt=1`` and ``kt=32``; unpack, math, congestion and
    ``L1_TO_L1`` all run twice. The frame still reports ``k_dimm == 32`` for the
    second case, so the saving is invisible in the output.
    """
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


def test_sol_run_with_formats_reuses_unpack(monkeypatch):
    """The same reuse holds under SPEED_OF_LIGHT with explicit formats.

    Worth covering separately because SoL moves runtimes and formats into the
    compile header, so projection has more to get right. Unpack and pack still
    run once each, math twice, and the TILE_LOOP unpack means still match.
    """
    elf_calls = []
    seeds = {"n": 0}
    _patch_hwfree_classvars(monkeypatch, speed_of_light=True)

    def _cfg(fidelity):
        templates, runtimes = _matmul_params(fidelity)
        cfg = PerfConfig(
            test_name="perf_relevance",
            formats=_format(DataFormat.Float16, DataFormat.Float16),
            run_types=_MATMUL_RUN_TYPES,
            templates=templates,
            runtimes=runtimes,
            dest_acc=DestAccumulation.No,
            disable_format_inference=True,
            relevance=MATMUL_RELEVANCE,
        )
        _stub_hw(monkeypatch, cfg, elf_calls, seeds, speed_of_light=True)
        report = PerfReport()
        cfg.run(report, run_count=1)
        return report._frames[-1]

    frame_lo = _cfg(MathFidelity.LoFi)
    frame_hi = _cfg(MathFidelity.HiFi4)
    assert elf_calls.count(PerfRunType.UNPACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.PACK_ISOLATE) == 1
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2
    unpack_col = stat_column("UNPACK_ISOLATE", MEAN)
    tile_lo = frame_lo[frame_lo[MARKER] == "TILE_LOOP"]
    tile_hi = frame_hi[frame_hi[MARKER] == "TILE_LOOP"]
    assert tile_lo[unpack_col].tolist() == tile_hi[unpack_col].tolist()


def test_isolate_cache_replays_init_and_tile_loop(monkeypatch):
    """A cache hit replays the INIT marker as well as TILE_LOOP.

    Both markers are present in both frames and the pack means agree on both.
    INIT is a separate measurement from the loop body, so serving only TILE_LOOP
    from the cache would leave the second case with a missing or mismatched
    INIT row.
    """
    elf_calls = []
    seeds = {"n": 0}
    frame_lo = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    frame_hi = _run_matmul_cfg(monkeypatch, MathFidelity.HiFi4, elf_calls, seeds)
    pack_col = stat_column("PACK_ISOLATE", MEAN)
    assert {"INIT", "TILE_LOOP"} <= set(frame_lo[MARKER])
    assert {"INIT", "TILE_LOOP"} <= set(frame_hi[MARKER])
    for marker in ("INIT", "TILE_LOOP"):
        lo = frame_lo[frame_lo[MARKER] == marker]
        hi = frame_hi[frame_hi[MARKER] == marker]
        assert lo[pack_col].tolist() == hi[pack_col].tolist()


def test_cache_hit_recomputes_code_size(monkeypatch):
    """Code size is recomputed on a cache hit rather than replayed with the timings.

    The first case reports the stubbed 4096 bytes; the second, which hits the
    cache for its unpack measurement, reports a different size because
    ``get_elf_text_size`` is called again. Text size is a property of the ELF
    that was built for this case, not of the measurement being reused.
    """
    sizes = {"n": 0}

    def fake_size(path):
        sizes["n"] += 1
        return 1000 + sizes["n"]

    elf_calls = []
    seeds = {"n": 0}
    frame_lo = _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    templates, runtimes = _matmul_params(MathFidelity.HiFi4)
    cfg = PerfConfig(
        test_name="perf_relevance",
        formats=None,
        run_types=_MATMUL_RUN_TYPES,
        templates=templates,
        runtimes=runtimes,
        dest_acc=DestAccumulation.No,
        relevance=MATMUL_RELEVANCE,
    )
    _stub_hw(monkeypatch, cfg, elf_calls, seeds)
    monkeypatch.setattr(TestConfig, "get_elf_text_size", staticmethod(fake_size))
    report = PerfReport()
    cfg.run(report, run_count=1)
    frame_hi = report._frames[-1]
    unpack_col = "TEXT_SIZE(UNPACK_ISOLATE)"
    assert int(frame_lo[unpack_col].iloc[0]) == 4096
    assert int(frame_hi[unpack_col].iloc[0]) != 4096


# ---------------------------------------------------------------------------
# 11. Cache policy and eviction
#
# What must never enter the cache at all, and what happens when it
# fills up.
# ---------------------------------------------------------------------------


def test_l1_to_l1_is_not_cached(monkeypatch):
    """``L1_TO_L1`` never enters the cache, while the isolates do.

    It is the real end-to-end number the suite exists to report, and its spec is
    full fidelity, so there is nothing to gain and a correctness bug to lose:
    serving it from a cache would report one case's pipeline timing under
    another's parameters.
    """
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds)
    cached_types = {key[2] for key in PerfConfig.EXECUTE_CACHE}
    assert PerfRunType.L1_TO_L1 not in cached_types
    assert PerfRunType.UNPACK_ISOLATE in cached_types


def test_pack_math_isolate_is_never_cached(monkeypatch):
    """``PACK_RELEVANCE`` leaves ``MATH_ISOLATE`` at full fidelity, so it always runs.

    A relu sweep produces two math runs and no cache entry. The general rule from
    section 2 applied end to end: a spec that pins nothing is never cached.
    """
    elf_calls = []
    seeds = {"n": 0}
    _run_pack_cfg(monkeypatch, elf_calls, seeds, unpack_to_dest=False, relu=0)
    _run_pack_cfg(monkeypatch, elf_calls, seeds, unpack_to_dest=False, relu=1)
    cached_types = {key[2] for key in PerfConfig.EXECUTE_CACHE}
    assert PerfRunType.MATH_ISOLATE not in cached_types
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2


def test_pack_empty_math_isolate_is_not_cached(monkeypatch):
    """An empty measurement is still not cloned, even though cloning it looks free.

    With ``unpack_to_dest`` the ``MATH_ISOLATE`` TILE_LOOP is a no-op, so reusing
    it across a relu sweep would appear harmless. It stays uncached anyway,
    because the decision follows the spec rather than the measured content, and
    both math runs happen. Pack, whose spec does pin parameters, is cached.
    """
    elf_calls = []
    seeds = {"n": 0}
    _run_pack_cfg(monkeypatch, elf_calls, seeds, unpack_to_dest=True, relu=0)
    _run_pack_cfg(monkeypatch, elf_calls, seeds, unpack_to_dest=True, relu=1)
    cached_types = {key[2] for key in PerfConfig.EXECUTE_CACHE}
    assert spec_is_full_fidelity(PACK_RELEVANCE[PerfRunType.MATH_ISOLATE])
    assert PerfRunType.MATH_ISOLATE not in cached_types
    assert PerfRunType.PACK_ISOLATE in cached_types
    assert elf_calls.count(PerfRunType.MATH_ISOLATE) == 2


def test_execute_cache_evicts_at_max(monkeypatch):
    """The cache is bounded and evicts once full.

    With the limit set to 2 entries, three distinct ``kt`` values leave at most 2
    cached and at least one eviction recorded. The bound matters under
    ``pytest-xdist``, where every worker process holds its own cache.
    """
    monkeypatch.setenv("LLK_PERF_EXECUTE_CACHE_MAX", "2")
    elf_calls = []
    seeds = {"n": 0}
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, kt=1)
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, kt=2)
    _run_matmul_cfg(monkeypatch, MathFidelity.LoFi, elf_calls, seeds, kt=3)
    assert len(PerfConfig.EXECUTE_CACHE) <= 2
    assert PerfConfig.CACHE_EVICTIONS >= 1


def test_execute_cache_max_rejects_invalid(monkeypatch):
    """A malformed cache bound fails loudly instead of falling back to a default.

    ``"abc"`` and ``"0"`` both raise and name the variable; ``"4"`` is accepted.
    A silent fallback would leave a run that meant to cap memory using the
    default bound, and the operator would never find out.
    """
    monkeypatch.setenv("LLK_PERF_EXECUTE_CACHE_MAX", "abc")
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="LLK_PERF_EXECUTE_CACHE_MAX"
    ):
        PerfConfig.execute_cache_max()
    monkeypatch.setenv("LLK_PERF_EXECUTE_CACHE_MAX", "0")
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="LLK_PERF_EXECUTE_CACHE_MAX"
    ):
        PerfConfig.execute_cache_max()
    monkeypatch.setenv("LLK_PERF_EXECUTE_CACHE_MAX", "4")
    assert PerfConfig.execute_cache_max() == 4
