# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in TILE_LOOP relevance maps for PerfConfig.

Why this exists
---------------
One pytest perf case is one *L1 identity*: a fixed set of formats, templates and
runtimes. ``PerfConfig.run()`` then walks every requested ``PerfRunType``
(typically L1_TO_L1, UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE, L1_CONGESTION)
and compiles plus executes each one. Most isolate modes cannot observe most
sweep axes: ``MATH_FIDELITY`` does not change what UNPACK's TILE_LOOP does, and
``k_dimm`` does not change PACK's, which iterates RT x CT rather than KT.
Without a relevance map those isolates still build an ELF and run on device. On
``perf_math_matmul`` (tens of thousands of cases x 5 run types) that is the
dominant cost of the job.

A relevance map declares, per run type, which fields TILE_LOOP can actually see.
Everything else is either *pinned* to a canonical default (so the C++ header
still has the constexpr it needs) or *dropped* from the cache key (so a
measurement taken for one case can be replayed for another).

The three mechanisms
--------------------
``project_*``      Rewrite unobservable fields to canonical defaults. This is
                   what makes two different pytest cases hash to the same
                   ``variant_id`` and therefore share one compiled ELF.
``execute_key``    Hashable identity of one measurement. Equal keys mean
                   ``PerfConfig.EXECUTE_CACHE`` can replay the profiler frame
                   (INIT, KERNEL, TILE_LOOP) instead of touching the device.
``PerfRelevance``  The per-test policy. A class whose attributes name, for each
                   run type, the template types / runtime types / runtime field
                   names / FormatConfig fields that TILE_LOOP observes.

Flow for one pytest case, as ``PerfConfig.run()`` drives it::

    for templates, runtimes, run_type in run_configs:
        spec = relevance[run_type]                   # a RunTypeRelevance
        if execute_key(..., spec=spec) in EXECUTE_CACHE:
            replay the cached profiler frame, skip the device entirely
        else:
            templates = project_templates(templates, spec)   # pin invisibles
            if SPEED_OF_LIGHT:              # runtimes become constexpr, so
                runtimes = project_runtimes(runtimes, spec)   # pin those too
                formats = project_formats(formats, spec)
                stimuli = project_stimuli(stimuli, runtimes, spec)
            generate_variant_hash(); build ELF; run; store in EXECUTE_CACHE

Reading a spec
--------------
``None`` (spelled ``KEEP_ALL``) in a slot means "keep every value": nothing is
pinned and every value enters the key. An empty frozenset (``PIN_ALL``) means
the opposite. L1_TO_L1 is always ``KEEP_ALL`` in every slot, so its key carries
every field and it never reuses anything -- by design, since L1_TO_L1 is the
measurement the whole sweep exists to produce.

Two invariants that are easy to get wrong
-----------------------------------------
* Isolate modes still run *every* TRISC INIT, including the threads that return
  immediately afterwards. So SoL format projection never pins ``unpack_*`` or
  ``math``: those must still agree with the L1 stimuli and ``dest_acc`` (see
  ``_INIT_LIVE_FORMATS``, where pinning Float16 against a Float32 L1 buffer
  deadlocked Math and Unpacker). Pack and SFPU fields TILE_LOOP cannot see do
  pin.
* Under SPEED_OF_LIGHT a runtime dropped from the key is also *compiled as its
  default*. If any thread still reads it before the isolate returns -- including
  an ``LLK_ASSERT`` -- the kernel ebreaks. ``UnpackTilizeRelevance`` keeps
  ``INPUT_DIMENSIONS`` on PACK for exactly that reason.

Layout
------
Spec types and ``PerfRelevance`` first, then the ``project_*`` helpers
(templates, runtimes, formats, stimuli), then ``execute_key``. Constants sit
with the functions that use them.
"""

import copy
import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from functools import cached_property
from typing import Any, Iterable

from ..format_config import DataFormat
from ..llk_params import DestSync, MathFidelity, PerfRunType
from ..test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    RuntimeParameter,
    TemplateParameter,
)

KEEP_ALL = None  # keep every value in this slot
PIN_ALL = frozenset()  # keep none (pin templates / drop runtimes+formats)


@dataclass(frozen=True)
class RunTypeRelevance:
    """TILE_LOOP-observable fields for one PerfRunType.

    EXECUTE_CACHE keys off TILE_LOOP-observable fields, then stores and
    replays the full profiler frame (INIT, KERNEL, TILE_LOOP) as one unit.
    A hit therefore copies the donor's INIT/KERNEL as well. MATH and UNPACK
    keep DEST_SYNC so miss INIT matches the labeled dest_sync.

    ``KEEP_ALL`` (None) keeps every value. ``PIN_ALL`` (empty frozenset) keeps
    none of that slot (pin templates / drop runtimes and formats).

    ``runtime_fields`` is a flat set of field *names* applied to every runtime
    type, not per type. Maps must not share a field name across two runtime
    dataclasses; ``as_map`` asserts that when ``runtime_types`` is set.

    Slots:
        templates: Template types TILE_LOOP observes. Only the types in
            ``_PINNABLE_TEMPLATES`` can be pinned; anything else stays in the
            header and therefore stays in the key regardless of this set.
        runtime_types: Runtime dataclass types TILE_LOOP observes. Others are
            dropped from the key, and replaced by their zero-argument default
            under SPEED_OF_LIGHT.
        runtime_fields: Field *names* TILE_LOOP observes, across all kept
            types. ``KEEP_ALL`` means every field of every kept type.
        format_fields: ``FormatConfig`` attribute names TILE_LOOP observes.

    Example:
    A spec that says "UNPACK's TILE_LOOP sees the unpack formats and the CRK
    dims, and nothing else":
    >>> RunTypeRelevance(
    ...     templates=PIN_ALL,
    ...     runtime_types=frozenset({CRK_TILE_DIMM}),
    ...     runtime_fields=KEEP_ALL,
    ...     format_fields=_UNPACK_FORMATS,
    ... )

    The specs actually used are built by ``PerfRelevance.as_map``; reach for
    the constructor directly only in tests.
    """

    templates: frozenset[type] | None = None
    runtime_types: frozenset[type] | None = None
    runtime_fields: frozenset[str] | None = None
    format_fields: frozenset[str] | None = None


_UNPACK_FORMATS = frozenset(
    {
        "unpack_A_src",
        "unpack_B_src",
        "unpack_A_dst",
        "unpack_B_dst",
        "unpack_S_src",
        "unpack_S_dst",
    }
)
_PACK_FORMATS = frozenset({"pack_src", "pack_dst", "pack_S_src", "pack_S_dst"})
_MATH_FORMATS = frozenset({"math"})
_IO_FORMATS = _UNPACK_FORMATS | _PACK_FORMATS

# Canonical order shared by the default relevance map and the public driver
# list re-exported from core.py. Tests may still pass only the subset they can
# measure (for example, MATH_ISOLATE alone).
ALL_PERF_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]


def assert_unique_runtime_field_names(spec: RunTypeRelevance) -> None:
    """Reject a spec whose runtime dataclasses share a field name.

    ``runtime_fields`` is a flat name set, so dropping ``k_dimm`` would drop it
    on every type that used that name. Fail at map-build time instead.

    No two runtime dataclasses collide today, so in practice this is a guard
    against a future parameter being added with a name that is already taken.
    It only checks specs that pin ``runtime_types``: with ``KEEP_ALL`` there is
    no name filter to be ambiguous about.

    Raises:
        ValueError: naming the field and both owning types.

    Example:
    Passes silently for every shipped map, and is called for you by
    ``PerfRelevance.as_map``:
    >>> assert_unique_runtime_field_names(
    ...     MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    ... )

    Had ``NUM_BLOCKS`` and ``NUM_TILES_IN_BLOCK`` both declared a field called
    ``num_blocks``, the same call would raise ``runtime field 'num_blocks' is
    shared by NUM_BLOCKS and NUM_TILES_IN_BLOCK``.
    """
    types = spec.runtime_types
    if not types:
        return
    seen: dict[str, type] = {}
    for cls in types:
        try:
            cls_fields = fields(cls)
        except TypeError:
            continue
        for field in cls_fields:
            owner = seen.get(field.name)
            if owner is not None and owner is not cls:
                raise ValueError(
                    f"runtime field {field.name!r} is shared by {owner.__name__} "
                    f"and {cls.__name__}; runtime_fields is a flat set of names"
                )
            seen[field.name] = cls


def spec_is_full_fidelity(spec: RunTypeRelevance | None) -> bool:
    """True when the spec keeps every field, so an execute key can never hit.

    Such a key carries the whole L1 identity, which is unique per pytest case.
    ``PerfConfig`` uses this to skip caching entirely for those run types --
    storing an entry that can never be hit is pure memory cost. A spec is
    full fidelity when every slot is ``KEEP_ALL`` (``L1_TO_L1``, and pack
    ``MATH_ISOLATE``). ``None`` is full fidelity only when the caller passes
    ``spec=None`` directly. A run type the map does not contain is not that
    fallback: ``PerfConfig`` rejects it with ``ValueError``.

    Example:
    >>> spec_is_full_fidelity(MATMUL_RELEVANCE[PerfRunType.L1_TO_L1])
    True
    >>> spec_is_full_fidelity(MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    False
    >>> spec_is_full_fidelity(None)
    True
    """
    if spec is None:
        return True
    return (
        spec.templates is None
        and spec.runtime_types is None
        and spec.runtime_fields is None
        and spec.format_fields is None
    )


class PerfRelevance:
    """Isolate-mode TILE_LOOP defaults. Subclasses add test-specific runtimes.

    L1_TO_L1 always keeps every field. Runtime slots default to ``KEEP_ALL``.
    Subclasses set frozensets to drop unused types.

    Attribute naming is ``<mode>_<slot>``, where the four mode prefixes map to
    run types and the four slots map to ``RunTypeRelevance`` fields. Note that
    ``cong_`` is L1_CONGESTION -- the abbreviation is not obvious:

        unpack_ -> UNPACK_ISOLATE      _templates      -> templates
        math_   -> MATH_ISOLATE        _runtimes       -> runtime_types
        pack_   -> PACK_ISOLATE        _runtime_fields -> runtime_fields
        cong_   -> L1_CONGESTION       _formats        -> format_fields

    There is deliberately no ``l1_*`` group: L1_TO_L1 is hard-coded to a bare
    ``RunTypeRelevance()``, which keeps everything.

    Subclass only what your kernel differs on. The base class already encodes
    what is true of every LLK perf kernel: UNPACK keeps DEST_SYNC (INIT replay)
    and pins fidelity/throttle, MATH sees fidelity, throttle, and dest_sync,
    PACK and L1_CONGESTION see dest_sync, and each mode sees the formats on
    its own side of the pipe.

    Example:
    A kernel whose pack TILE_LOOP is driven only by the loop factor, and which
    has no math phase worth isolating:
    >>> class MyRelevance(PerfRelevance):
    ...     run_types = (
    ...         PerfRunType.L1_TO_L1,
    ...         PerfRunType.PACK_ISOLATE,
    ...     )
    ...     pack_runtimes = frozenset({LOOP_FACTOR})
    >>> MY_RELEVANCE = MyRelevance()   # module-level singleton, see below

    Then pass ``relevance=MY_RELEVANCE`` to ``PerfConfig``. Before adding a
    map, read the kernel's TILE_LOOP and every assert that runs before the
    isolate returns -- under SPEED_OF_LIGHT a dropped runtime is compiled as
    its default, and an ``LLK_ASSERT`` on it will ebreak.
    """

    run_types: list[PerfRunType] = ALL_PERF_RUN_TYPES

    unpack_templates: frozenset[type] | None = frozenset({DEST_SYNC})
    # INIT runs _llk_math_pack_sync_init_<dest_sync>; keep DEST_SYNC so that
    # miss INIT matches the labeled sweep. TILE_LOOP does not reuse across it.
    math_templates: frozenset[type] | None = frozenset(
        {MATH_FIDELITY, THROTTLE_LEVEL, DEST_SYNC}
    )
    pack_templates: frozenset[type] | None = frozenset({DEST_SYNC})
    cong_templates: frozenset[type] | None = frozenset({DEST_SYNC})

    unpack_runtimes: frozenset[type] | None = KEEP_ALL
    math_runtimes: frozenset[type] | None = KEEP_ALL
    pack_runtimes: frozenset[type] | None = KEEP_ALL
    cong_runtimes: frozenset[type] | None = KEEP_ALL

    unpack_runtime_fields: frozenset[str] | None = KEEP_ALL
    math_runtime_fields: frozenset[str] | None = KEEP_ALL
    pack_runtime_fields: frozenset[str] | None = KEEP_ALL
    cong_runtime_fields: frozenset[str] | None = KEEP_ALL

    unpack_formats: frozenset[str] | None = _UNPACK_FORMATS
    math_formats: frozenset[str] | None = _MATH_FORMATS
    pack_formats: frozenset[str] | None = _PACK_FORMATS
    cong_formats: frozenset[str] | None = _IO_FORMATS

    @cached_property
    def as_map(self) -> dict[PerfRunType, RunTypeRelevance]:
        """Collapse the class attributes into one spec per run type.

        Only the run types in ``self.run_types`` get an entry. A listed run
        type this class cannot describe (anything outside the five
        ``ALL_PERF_RUN_TYPES``) raises ``ValueError`` naming it, the same
        error ``PerfConfig`` raises when a caller asks for a run type the
        map omits. Cached because ``PerfConfig`` looks a spec up once per
        run type per pytest case, and the maps are module-level singletons.

        Example:
        >>> list(UNPACK_TILIZE_RELEVANCE.as_map)
        [<PerfRunType.L1_TO_L1: 1>, <PerfRunType.UNPACK_ISOLATE: 2>,
         <PerfRunType.PACK_ISOLATE: 4>, <PerfRunType.L1_CONGESTION: 5>]
        >>> MATMUL_RELEVANCE.as_map[PerfRunType.PACK_ISOLATE].runtime_fields
        frozenset({'loop_factor', 'c_dimm', 'r_dimm'})
        """
        specs = {
            PerfRunType.L1_TO_L1: RunTypeRelevance(),
            PerfRunType.UNPACK_ISOLATE: RunTypeRelevance(
                templates=self.unpack_templates,
                runtime_types=self.unpack_runtimes,
                runtime_fields=self.unpack_runtime_fields,
                format_fields=self.unpack_formats,
            ),
            PerfRunType.MATH_ISOLATE: RunTypeRelevance(
                templates=self.math_templates,
                runtime_types=self.math_runtimes,
                runtime_fields=self.math_runtime_fields,
                format_fields=self.math_formats,
            ),
            PerfRunType.PACK_ISOLATE: RunTypeRelevance(
                templates=self.pack_templates,
                runtime_types=self.pack_runtimes,
                runtime_fields=self.pack_runtime_fields,
                format_fields=self.pack_formats,
            ),
            PerfRunType.L1_CONGESTION: RunTypeRelevance(
                templates=self.cong_templates,
                runtime_types=self.cong_runtimes,
                runtime_fields=self.cong_runtime_fields,
                format_fields=self.cong_formats,
            ),
        }
        missing = [run_type for run_type in self.run_types if run_type not in specs]
        if missing:
            names = ", ".join(run_type.name for run_type in missing)
            raise ValueError(
                f"relevance map has no entry for {names}; "
                "every PerfConfig run_type must be in the map"
            )
        mapping = {run_type: specs[run_type] for run_type in self.run_types}
        for spec in mapping.values():
            assert_unique_runtime_field_names(spec)
        return mapping

    def __getitem__(self, run_type: PerfRunType) -> RunTypeRelevance:
        """Look up one run type's spec, so a map reads like the dict it replaces.

        Raises ``KeyError`` for a run type this map does not cover; callers
        that may see one (``PerfConfig``) should go through
        ``as_map.get(run_type)`` instead.

        Example:
        >>> MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE].format_fields
        frozenset({'unpack_A_src', 'unpack_A_dst', 'unpack_B_src',
                   'unpack_B_dst', 'unpack_S_src', 'unpack_S_dst'})
        """
        return self.as_map[run_type]


# Helpers for per-test PerfRelevance subclasses.


def _runtime_fields(*types: type, drop: Iterable[str] = ()) -> frozenset[str]:
    """Every field name declared by ``types``, minus ``drop``.

    Convenience for the ``*_runtime_fields`` slots: name the runtime types the
    run type keeps, then subtract the individual fields it cannot see. Writing
    the surviving names out by hand rots as soon as a dataclass gains a field.

    Example:
    PACK keeps LOOP_FACTOR and CRK_TILE_DIMM, but pack iterates RT x CT and so
    never observes the K dimension:
    >>> _runtime_fields(LOOP_FACTOR, CRK_TILE_DIMM, drop={"k_dimm"})
    frozenset({'loop_factor', 'c_dimm', 'r_dimm'})
    """
    names = {f.name for cls in types for f in fields(cls)}
    return frozenset(names - set(drop))


_PACK_BLOCK_RUNTIMES = frozenset(
    {NUM_BLOCKS, NUM_TILES_IN_BLOCK, LOOP_FACTOR, NUM_FACES}
)


LLK_DISABLE_PERF_RELEVANCE = "LLK_DISABLE_PERF_RELEVANCE"


def maybe_relevance(
    relevance: dict[PerfRunType, RunTypeRelevance] | PerfRelevance | None,
) -> dict[PerfRunType, RunTypeRelevance] | PerfRelevance | None:
    """Return ``None`` when ``LLK_DISABLE_PERF_RELEVANCE=1``, else ``relevance``.

    Honored in ``PerfConfig.__init__``. Set the env var to run the same sweep
    without isolate reuse.

    This is the A/B switch: the same pytest command with the var set to 1 runs
    every isolate mode for every case, which is the baseline a relevance map
    has to reproduce. The runner scripts and ``llk-perf-impl.yaml`` export it,
    so the var is read once per ``PerfConfig``, not once per process.

    Example:
    >>> maybe_relevance(MATMUL_RELEVANCE) is MATMUL_RELEVANCE
    True
    >>> os.environ[LLK_DISABLE_PERF_RELEVANCE] = "1"
    >>> maybe_relevance(MATMUL_RELEVANCE) is None
    True
    """
    if os.environ.get(LLK_DISABLE_PERF_RELEVANCE) == "1":
        return None
    return relevance


# -- project_templates -------------------------------------------------------

# pin_template only rewrites these. Any other template stays in the header, so
# execute_key must keep it even when spec.templates omits the type.
_PINNABLE_TEMPLATES = frozenset({MATH_FIDELITY, DEST_SYNC, THROTTLE_LEVEL})


def pin_template(param: TemplateParameter) -> TemplateParameter:
    """Canonical compile-time stand-in for a template that TILE_LOOP cannot see.

    Pinning rather than omitting matters because the C++ header still needs the
    constexpr. The choice of default is arbitrary but must be *stable*: it is
    what makes two cases that differ only in an unobserved template hash to one
    ``variant_id`` and share one ELF.

        MATH_FIDELITY  -> LoFi
        DEST_SYNC      -> Full
        THROTTLE_LEVEL -> 0

    Anything outside ``_PINNABLE_TEMPLATES`` is returned untouched, because no
    canonical value for it has been established.

    Example:
    >>> pin_template(MATH_FIDELITY(MathFidelity.HiFi4))
    MATH_FIDELITY(math_fidelity=<MathFidelity.LoFi: 0>)
    >>> pin_template(DEST_SYNC(DestSync.Half))
    DEST_SYNC(dest_sync=<DestSync.Full: 'SyncFull'>)
    >>> pin_template(THROTTLE_LEVEL(5))
    THROTTLE_LEVEL(throttle_level=0)
    """
    if type(param) not in _PINNABLE_TEMPLATES:
        return param
    if isinstance(param, MATH_FIDELITY):
        return MATH_FIDELITY(MathFidelity.LoFi)
    if isinstance(param, DEST_SYNC):
        return DEST_SYNC(DestSync.Full)
    if isinstance(param, THROTTLE_LEVEL):
        return THROTTLE_LEVEL(0)
    return param


def _template_visible(param: TemplateParameter, spec: RunTypeRelevance | None) -> bool:
    """Whether this template stays as-is, both in the header and in the key.

    The second clause is the subtle one: a template outside
    ``_PINNABLE_TEMPLATES`` has no canonical default, so ``pin_template``
    cannot rewrite it and it keeps its swept value in the compiled header. It
    must therefore stay in the execute key too, even when ``spec.templates``
    omits it -- otherwise two cases with genuinely different headers would
    share a measurement. ``PackUntilizeRelevance`` relies on this for
    ``INPUT_DIMENSIONS``.

    Example:
    >>> unpack = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]  # DEST_SYNC only
    >>> _template_visible(MATH_FIDELITY(MathFidelity.HiFi4), unpack)
    False
    >>> _template_visible(MATH_FIDELITY(MathFidelity.HiFi4),
    ...                   MATMUL_RELEVANCE[PerfRunType.MATH_ISOLATE])
    True
    >>> _template_visible(APPROX_MODE(), unpack)   # not pinnable -> always kept
    True
    """
    if spec is None or spec.templates is None:
        return True
    return type(param) in spec.templates or type(param) not in _PINNABLE_TEMPLATES


def project_templates(
    templates: Iterable[TemplateParameter],
    spec: RunTypeRelevance | None,
) -> list[TemplateParameter]:
    """Keep relevant templates; pin the rest. Always keep PERF_RUN_TYPE.

    Applied on every cache miss, in both SoL and non-SoL mode, because
    templates are compile-time either way. ``PERF_RUN_TYPE`` is exempt for the
    obvious reason: it is what selects the isolate mode, so pinning it would
    compile the wrong kernel.

    Returns a new list; the input parameters are never mutated.

    Example:
    UNPACK's spec pins both, and the run type survives:
    >>> project_templates(
    ...     [MATH_FIDELITY(MathFidelity.HiFi4), DEST_SYNC(DestSync.Half),
    ...      PERF_RUN_TYPE(PerfRunType.UNPACK_ISOLATE)],
    ...     MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE],
    ... )
    [MATH_FIDELITY(math_fidelity=<MathFidelity.LoFi: 0>),
     DEST_SYNC(dest_sync=<DestSync.Full: 'SyncFull'>),
     PERF_RUN_TYPE(perf_run_type=<PerfRunType.UNPACK_ISOLATE: 2>)]
    """
    projected = []
    for param in templates:
        if isinstance(param, PERF_RUN_TYPE) or _template_visible(param, spec):
            projected.append(param)
        else:
            projected.append(pin_template(param))
    return projected


# -- project_runtimes --------------------------------------------------------


def _default_runtime(param: RuntimeParameter) -> RuntimeParameter:
    """Replace a whole runtime parameter with its zero-argument default.

    Used when the spec drops the *type*, as opposed to some of its fields. The
    dataclass default is the canonical value by construction, which is why a
    runtime parameter must be constructible with no arguments to participate.

    Raises:
        TypeError: if the dataclass has a required field, naming the type. That
            is a map bug, not a user error: drop individual fields via
            ``runtime_fields`` instead of the type.

    Example:
    >>> _default_runtime(TILE_COUNT(256))
    TILE_COUNT(tile_cnt=0)
    >>> _default_runtime(NUM_FACES(2, 2, 2))
    NUM_FACES(num_faces=4, num_faces_A=4, num_faces_B=4)
    """
    cls = type(param)
    try:
        return cls()
    except TypeError as exc:
        raise TypeError(
            f"Cannot pin {cls.__name__}: no zero-argument constructor"
        ) from exc


def _default_field_value(value: Any) -> Any:
    """Canonical value for one dropped field of an otherwise-kept runtime type.

    Unlike ``_default_runtime`` there is no dataclass default to lean on here,
    so the rule is by type:

        bool            -> False
        int, nonzero    -> 1    (a dimension of 1 is the neutral one)
        int, zero       -> 0    (already canonical; TILE_COUNT defaults to 0)
        int-valued obj  -> type(value)(1), e.g. ctypes c_uint32

    ``1`` rather than ``0`` for nonzero ints matters: these values are often
    multiplied into a tile count, and zeroing them yields a degenerate layout
    that the kernel may assert on.

    Raises:
        TypeError: for an Enum, or anything with no int-valued ``.value``.
            Enums have no meaningful neutral member, so the map should drop the
            whole runtime type rather than one enum field.

    Example:
    >>> _default_field_value(True)
    False
    >>> _default_field_value(32)
    1
    >>> _default_field_value(0)
    0
    >>> _default_field_value(DataFormat.Float16)
    Traceback (most recent call last):
    TypeError: Cannot pin Enum field DataFormat=<DataFormat.Float16: ...>; ...
    """
    # Nonzero ints pin to 1. Dropped TILE_COUNT uses _default_runtime() (0).
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 1 if value else 0
    if isinstance(value, Enum):
        raise TypeError(
            f"Cannot pin Enum field {type(value).__name__}={value!r}; "
            "drop the runtime type instead of a single Enum field"
        )
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return type(value)(1)
    raise TypeError(
        f"Cannot pin {type(value).__name__}={value!r} to a canonical default"
    )


def _runtime_type_visible(cls: type, spec: RunTypeRelevance | None) -> bool:
    """Whether the spec keeps this runtime dataclass at all.

    ``KEEP_ALL`` (``runtime_types is None``) keeps every type, so an absent
    spec and a permissive spec answer the same.

    Example:
    >>> pack = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    >>> _runtime_type_visible(CRK_TILE_DIMM, pack)
    True
    >>> _runtime_type_visible(TILE_COUNT, pack)
    False
    """
    if spec is None or spec.runtime_types is None:
        return True
    return cls in spec.runtime_types


def _runtime_field_visible(name: str, spec: RunTypeRelevance | None) -> bool:
    """Whether the spec keeps this runtime field name.

    Flat across types by design -- see ``RunTypeRelevance.runtime_fields`` and
    ``assert_unique_runtime_field_names``.

    Example:
    >>> pack = MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE]
    >>> _runtime_field_visible("r_dimm", pack)
    True
    >>> _runtime_field_visible("k_dimm", pack)
    False
    """
    if spec is None or spec.runtime_fields is None:
        return True
    return name in spec.runtime_fields


def project_runtimes(
    runtimes: Iterable[RuntimeParameter],
    spec: RunTypeRelevance | None,
) -> list[RuntimeParameter]:
    """Pin unused runtime types/fields. Used when SPEED_OF_LIGHT inlines runtimes.

    Only called under SoL, where runtimes move onto ``self.templates`` and
    become constexpr, so pinning them is what actually collapses the
    ``variant_id``. In non-SoL mode runtimes are written to L1 at their
    original values and only the *key* drops them.

    Two levels, checked in that order: a dropped **type** becomes
    ``_default_runtime(param)``; a kept type with dropped **fields** is
    ``dataclasses.replace``d field by field. Returns a new list, and returns
    kept parameters by identity when nothing changed.

    Example:
    Matmul's PACK spec keeps LOOP_FACTOR and CRK_TILE_DIMM but drops the
    ``k_dimm`` field, and drops NUM_FACES / TILE_COUNT as types:
    >>> project_runtimes(
    ...     [CRK_TILE_DIMM(4, 2, 32), LOOP_FACTOR(64), NUM_FACES(2, 2, 2),
    ...      TILE_COUNT(256)],
    ...     MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE],
    ... )
    [CRK_TILE_DIMM(c_dimm=4, r_dimm=2, k_dimm=1),
     LOOP_FACTOR(loop_factor=64),
     NUM_FACES(num_faces=4, num_faces_A=4, num_faces_B=4),
     TILE_COUNT(tile_cnt=0)]
    """
    if spec is None:
        return list(runtimes)
    projected = []
    for param in runtimes:
        if spec.runtime_types is not None and type(param) not in spec.runtime_types:
            projected.append(_default_runtime(param))
            continue
        if spec.runtime_fields is None:
            projected.append(param)
            continue
        updates = {
            f.name: _default_field_value(getattr(param, f.name))
            for f in fields(param)
            if f.name not in spec.runtime_fields
        }
        projected.append(replace(param, **updates) if updates else param)
    return projected


# -- project_formats ---------------------------------------------------------

# SPEED_OF_LIGHT inlines every FormatConfig field; TILE_LOOP-unused pack/SFPU
# fields pin to this. Unpack and math stay original (see _INIT_LIVE_FORMATS).
_PINNED_FORMAT = DataFormat.Float16
_FORMAT_FIELDS = (
    "unpack_A_src",
    "unpack_B_src",
    "unpack_A_dst",
    "unpack_B_dst",
    "unpack_S_src",
    "unpack_S_dst",
    "pack_src",
    "pack_dst",
    "pack_S_src",
    "pack_S_dst",
    "math",
    "sfpu_src",
    "sfpu_dst",
)
_ALL_FORMATS = frozenset(_FORMAT_FIELDS)
# INIT of idle isolate threads still runs. Pinning these to Float16 while L1 is
# Float32 (unpack_to_dest + dest_acc=Yes) deadlocks Math and Unpacker.
_INIT_LIVE_FORMATS = frozenset(
    {
        "unpack_A_src",
        "unpack_B_src",
        "unpack_A_dst",
        "unpack_B_dst",
        "unpack_S_src",
        "unpack_S_dst",
        "math",
    }
)


def project_formats(formats: Any, spec: RunTypeRelevance | None) -> Any:
    """Pin TILE_LOOP-unused pack/SFPU fields when SPEED_OF_LIGHT inlines formats.

    Unpack and math are never pinned: isolate INIT still consumes them against
    L1 stimuli and dest_acc even when TILE_LOOP cannot see those fields.

    Accepts a single ``FormatConfig`` or a list of them, and copies rather than
    mutating so the report can still publish the original formats. A field the
    config does not declare is skipped, so this works across the several
    FormatConfig shapes in the suite.

    Note the asymmetry with ``execute_key``: the key drops every field outside
    ``spec.format_fields``, while this function leaves ``_INIT_LIVE_FORMATS``
    alone. So two cases can share a key while compiling different ELFs -- which
    is intended for the measurement, but is why ELF-derived values such as text
    size must not be replayed from the cache.

    Example:
    Under UNPACK's spec the pack side pins to Float16 while unpack and math
    keep the swept formats:
    >>> f = FormatConfig(DataFormat.Float32, DataFormat.Float32,
    ...                  DataFormat.Float32, DataFormat.Bfp8_b,
    ...                  DataFormat.Float32)
    >>> p = project_formats(f, MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    >>> p.unpack_A_src, p.math, p.pack_src, p.pack_dst
    (Float32, Float32, Float16, Float16)
    """
    if formats is None:
        return None
    if isinstance(formats, list):
        return [project_formats(fmt, spec) for fmt in formats]
    projected = copy.copy(formats)
    if spec is None or spec.format_fields is None:
        return projected
    for name in _FORMAT_FIELDS:
        if name in _INIT_LIVE_FORMATS:
            continue
        if not hasattr(projected, name):
            continue
        if name not in spec.format_fields:
            setattr(projected, name, _PINNED_FORMAT)
    return projected


def _format_field_pinned(name: str, spec: RunTypeRelevance) -> bool:
    """Whether ``project_formats`` would rewrite this FormatConfig field.

    The inverse of "visible", with the ``_INIT_LIVE_FORMATS`` carve-out
    applied, so ``_project_stimuli_formats`` can ask one question instead of
    re-deriving the rule.

    Example:
    >>> unpack = MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE]
    >>> _format_field_pinned("pack_dst", unpack)
    True
    >>> _format_field_pinned("unpack_A_src", unpack)   # in spec.format_fields
    False
    >>> _format_field_pinned("math", unpack)           # INIT-live, never pinned
    False
    """
    if name in _INIT_LIVE_FORMATS:
        return False
    if spec.format_fields is None:
        return False
    return name not in spec.format_fields


# -- project_stimuli ---------------------------------------------------------

# Stimuli format attrs hashed into variant_id under SoL. Pin them when every
# matching FormatConfig field would be pinned. The A/B/S rows are intentionally
# inert today because all of their unpack fields are _INIT_LIVE_FORMATS; keeping
# them here makes that rule explicit and central. T/result rows can pin, enabling
# pack-output sweeps to reuse UNPACK_ISOLATE compiles.
_STIMULI_FORMAT_FIELDS = (
    ("stimuli_A_format", ("unpack_A_src", "unpack_A_dst")),
    ("stimuli_B_format", ("unpack_B_src", "unpack_B_dst")),
    ("stimuli_S_format", ("unpack_S_src", "unpack_S_dst")),
    ("stimuli_T_format", ("pack_S_src", "pack_S_dst")),
    ("stimuli_res_format", ("pack_src", "pack_dst")),
)

_STIMULI_TILE_ATTRS = (
    "tile_count_A",
    "tile_count_B",
    "tile_count_res",
    "tile_count_S",
    "tile_count_T",
    "tile_count_C",
)


def _as_int(value: Any) -> int:
    """Read a dimension as a plain int, unwrapping ctypes / enum wrappers.

    Runtime dataclasses annotate some dims as ``c_uint32``, and callers pass
    either the wrapper or a bare int, so every arithmetic site would otherwise
    need the same two-line dance.

    Example:
    >>> _as_int(32)
    32
    >>> _as_int(ctypes.c_uint32(32))
    32
    """
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return raw
    return int(value)


def _project_stimuli_formats(stimuli: Any, spec: RunTypeRelevance) -> None:
    """Pin the stimuli's format attrs to match ``project_formats``. Mutates.

    ``StimuliConfig`` carries its own copy of the formats, and ``str(stimuli)``
    is hashed into ``variant_id`` under SoL. Pinning ``pack_dst`` in the
    FormatConfig while leaving ``stimuli_res_format`` at the swept value would
    leave the hash varying and defeat the whole projection.

    An attr is pinned only when *every* FormatConfig field behind it would be
    pinned, so a half-observed operand keeps its original format. Called on the
    copy that ``project_stimuli`` already made, so mutating in place is safe.

    Example:
    Under UNPACK's spec the result operand pins but operand A does not:
    >>> s = copy.copy(stimuli)   # tile formats all Float32
    >>> _project_stimuli_formats(s, MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    >>> s.stimuli_A_format, s.stimuli_res_format
    (Float32, Float16)
    """
    for attr, format_names in _STIMULI_FORMAT_FIELDS:
        if getattr(stimuli, attr, None) is None:
            continue
        if all(_format_field_pinned(name, spec) for name in format_names):
            setattr(stimuli, attr, _PINNED_FORMAT)


def _set_stimuli_tile_count(stimuli: Any, attr: str, count: int) -> None:
    """Clamp one operand tile count. ``attr`` is a closed set of field names.

    ``project_stimuli`` must not ``setattr`` a computed name: Cycode treats that
    as unsanitized input into code generation, and ``stimuli_config`` writes
    these counts into the kernel header as operand base addresses.
    """
    if attr not in _STIMULI_TILE_ATTRS:
        raise ValueError(f"unknown stimuli tile attr {attr!r}")
    original = getattr(stimuli, attr, None)
    if original is None:
        return
    clamped = min(count, _as_int(original))
    if attr == "tile_count_A":
        stimuli.tile_count_A = clamped
    elif attr == "tile_count_B":
        stimuli.tile_count_B = clamped
    elif attr == "tile_count_res":
        stimuli.tile_count_res = clamped
    elif attr == "tile_count_S":
        stimuli.tile_count_S = clamped
    elif attr == "tile_count_T":
        stimuli.tile_count_T = clamped
    else:
        stimuli.tile_count_C = clamped


def _operand_tile_counts(
    runtimes: Iterable[RuntimeParameter], spec: RunTypeRelevance
) -> dict[str, int] | None:
    """Projected L1 tiles per operand, or None if the spec cannot derive a layout.

    CRK: A is r×k, B is k×c, and result-like operands are
    r×c×visible-NUM_BLOCKS. Inputs are reused across destination handoff blocks.
    Invisible CRK dims and block fields pin to 1. Without CRK dimensions, pack
    block runtimes size every operand as NUM_BLOCKS × NUM_TILES_IN_BLOCK. The
    caller still clamps each count to the original. Never synthesize a 1-tile
    layout when neither source is present.

    Per-operand rather than one scalar because the operands genuinely differ in
    shape: collapsing them would both mis-size the buffers and, since
    ``stimuli_config`` lays operands out contiguously, move every later
    operand's base address.

    Returning ``None`` is meaningful: it says "this spec gives me no basis for
    a layout", and ``project_stimuli`` then leaves the counts alone. The
    alternative -- defaulting to one tile -- silently shrinks L1 for any test
    without CRK or block runtimes.

    Example:
    A 1x16 output with kt=32. UNPACK sees all three dims, PACK drops ``k_dimm``
    so its k folds to 1:
    >>> rts = [CRK_TILE_DIMM(16, 1, 32), LOOP_FACTOR(64)]
    >>> _operand_tile_counts(rts, MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    {'tile_count_A': 32, 'tile_count_B': 512, 'tile_count_res': 16, ...}
    >>> _operand_tile_counts(rts, MATMUL_RELEVANCE[PerfRunType.PACK_ISOLATE])
    {'tile_count_A': 1, 'tile_count_B': 16, 'tile_count_res': 16, ...}

    Those raw numbers can exceed what the caller asked for, which is why
    ``project_stimuli`` clamps rather than assigns.
    """
    if _runtime_type_visible(CRK_TILE_DIMM, spec):
        crk = next((p for p in runtimes if isinstance(p, CRK_TILE_DIMM)), None)
        if crk is not None:
            r = _as_int(crk.r_dimm) if _runtime_field_visible("r_dimm", spec) else 1
            c = _as_int(crk.c_dimm) if _runtime_field_visible("c_dimm", spec) else 1
            k = _as_int(crk.k_dimm) if _runtime_field_visible("k_dimm", spec) else 1
            block_count = 1
            if _runtime_type_visible(NUM_BLOCKS, spec):
                blocks = next((p for p in runtimes if isinstance(p, NUM_BLOCKS)), None)
                if blocks is not None and _runtime_field_visible("num_blocks", spec):
                    block_count = _as_int(blocks.num_blocks)
            a = max(r * k, 1)
            b = max(k * c, 1)
            res = max(r * c * block_count, 1)
            return {
                "tile_count_A": a,
                "tile_count_B": b,
                "tile_count_res": res,
                "tile_count_S": res,
                "tile_count_T": res,
                "tile_count_C": res,
            }
    if _runtime_type_visible(NUM_BLOCKS, spec) and _runtime_type_visible(
        NUM_TILES_IN_BLOCK, spec
    ):
        blocks = next((p for p in runtimes if isinstance(p, NUM_BLOCKS)), None)
        tiles = next((p for p in runtimes if isinstance(p, NUM_TILES_IN_BLOCK)), None)
        if blocks is not None and tiles is not None:
            block_count = (
                _as_int(blocks.num_blocks)
                if _runtime_field_visible("num_blocks", spec)
                else 1
            )
            tiles_per_block = (
                _as_int(tiles.num_tiles_in_block)
                if _runtime_field_visible("num_tiles_in_block", spec)
                else 1
            )
            count = max(block_count * tiles_per_block, 1)
            return {attr: count for attr in _STIMULI_TILE_ATTRS}
    return None


def project_stimuli(
    stimuli: Any,
    runtimes: Iterable[RuntimeParameter],
    spec: RunTypeRelevance | None,
) -> Any:
    """Canonicalize L1 tile counts and unused format metadata under SPEED_OF_LIGHT.

    ``str(variant_stimuli)`` is hashed into ``variant_id`` under SoL, and
    ``stimuli_key`` fingerprints tile counts into ``execute_key``. If TILE_COUNT
    is dropped from the header but stimuli still carry the original ``kt``-dependent
    counts, isolate compiles do not reuse. Recompute per-operand counts from the
    visible CRK dims (A=r×k, B=k×c, Res=r×c) or from NUM_BLOCKS ×
    NUM_TILES_IN_BLOCK; never raise a count the caller already clamped, and never
    collapse to one tile when neither source is visible. Pin stimulus formats
    that ``project_formats`` would pin so pack-output sweeps still share
    UNPACK_ISOLATE ELFs.

    The ``min`` against the original is load-bearing, not defensive. Callers
    clamp their own counts for hardware reasons -- ``perf_matmul`` caps at
    ``PERF_RING_TILES`` so a K=32 Float32 case does not overflow Tensix L1 --
    and ``stimuli_config`` turns tile counts directly into the operand base
    addresses it writes into the kernel header. A count raised above the
    caller's would put ``buf_res_addr`` outside L1.

    The count rewrite is skipped entirely while TILE_COUNT or INPUT_DIMENSIONS
    is still visible, because then the header already carries the real
    geometry and there is nothing to reconcile.

    Returns a copy; ``PerfConfig`` keeps the original for the report columns.

    Example:
    Matmul rt=1, ct=16, kt=32, with the caller's clamp at 16 tiles per
    operand. ``_operand_tile_counts`` would ask for 32 / 512 / 16, so the
    clamp holds every operand at 16 and only the formats change:
    >>> p = project_stimuli(stimuli, runtimes,
    ...                     MATMUL_RELEVANCE[PerfRunType.UNPACK_ISOLATE])
    >>> stimuli_key(p)
    (('tile_count_A', 16), ('tile_count_B', 16), ('tile_count_res', 16))

    Under PACK's spec ``k_dimm`` is invisible, so A really does shrink:
    >>> p = project_stimuli(stimuli, project_runtimes(runtimes, pack_spec),
    ...                     pack_spec)
    >>> stimuli_key(p)
    (('tile_count_A', 1), ('tile_count_B', 16), ('tile_count_res', 16))
    """
    if stimuli is None:
        return None
    projected = copy.copy(stimuli)
    if spec is None:
        return projected
    _project_stimuli_formats(projected, spec)
    if not (
        _runtime_type_visible(TILE_COUNT, spec)
        or _runtime_type_visible(INPUT_DIMENSIONS, spec)
    ):
        counts = _operand_tile_counts(runtimes, spec)
        if counts is not None:
            for attr in _STIMULI_TILE_ATTRS:
                _set_stimuli_tile_count(projected, attr, counts[attr])
    projected._calculate_tile_sizes()
    return projected


# -- execute_key -------------------------------------------------------------


def _hashable(value: Any) -> Any:
    """Reduce one parameter value to something stable enough to key a cache on.

    Ints, floats, strs, bools, None and Enums pass through; anything else with
    an int/float/str/bool ``.value`` (ctypes wrappers) is unwrapped.

    Deliberately raises instead of falling back to ``repr()``. A default
    ``__repr__`` embeds the object's address, which would make every key unique
    and silently disable reuse -- a map that quietly stops working is far worse
    than one that fails at the first case.

    Raises:
        TypeError: naming the type and value that needs an explicit conversion.

    Example:
    >>> _hashable(32), _hashable(None), _hashable(DataFormat.Float16)
    (32, None, <DataFormat.Float16: Float16/2B>)
    >>> _hashable(ctypes.c_uint32(4))
    4
    >>> _hashable(object())
    Traceback (most recent call last):
    TypeError: Cannot hash object=<object object at 0x...> into an execute key; ...
    """
    if isinstance(value, (int, float, str, bool, type(None), Enum)):
        return value
    raw = getattr(value, "value", None)
    if not isinstance(value, Enum) and isinstance(raw, (int, float, str, bool)):
        return raw
    raise TypeError(
        f"Cannot hash {type(value).__name__}={value!r} into an execute key; "
        "add an explicit conversion rather than falling back to repr()"
    )


def _dataclass_items(param: Any) -> list[tuple[str, Any]]:
    """One parameter as ``(field_name, hashable_value)`` pairs, in field order.

    Names travel with the values so the key stays readable when a cache miss
    has to be explained, and so two types with the same arity cannot produce
    interchangeable tuples.

    Example:
    >>> _dataclass_items(CRK_TILE_DIMM(4, 2, 32))
    [('c_dimm', 4), ('r_dimm', 2), ('k_dimm', 32)]
    """
    return [(f.name, _hashable(getattr(param, f.name))) for f in fields(param)]


def stimuli_key(stimuli: Any) -> tuple:
    """Hashable L1 tile-count fingerprint for ``execute_key``.

    The L1 layout is not otherwise reconstructible from the key's template and
    runtime items -- callers derive tile counts with their own clamps (matmul
    caps at ``PERF_RING_TILES``) -- so without this two cases with different L1
    footprints could share a measurement. Absent operands contribute nothing
    rather than a placeholder, so the tuple length varies by test.

    Example:
    >>> stimuli_key(None)
    ()
    >>> stimuli_key(stimuli)   # A, B and Res present, 16 tiles each
    (('tile_count_A', 16), ('tile_count_B', 16), ('tile_count_res', 16))
    """
    if stimuli is None:
        return ()
    items = []
    for attr in _STIMULI_TILE_ATTRS:
        value = getattr(stimuli, attr, None)
        if value is not None:
            items.append((attr, _as_int(value)))
    return tuple(items)


def execute_key(
    *,
    test_name: str,
    run_type: PerfRunType,
    dest_acc: Any,
    templates: Iterable[TemplateParameter],
    runtimes: Iterable[RuntimeParameter],
    formats: Any,
    speed_of_light: bool,
    spec: RunTypeRelevance | None,
    unpack_to_dest: Any = False,
    unpack_to_srcs: Any = False,
    l1_acc: Any = None,
    source: str = "",
    run_count: int = 1,
    stimuli: Any = None,
) -> tuple:
    """Hashable identity of one run-type measurement under ``spec``.

    Two pytest cases whose keys are equal are asserted to produce the same
    TILE_LOOP timings for this run type, so ``PerfConfig.EXECUTE_CACHE`` may
    replay the first one's rows for the second and skip the device. Everything
    the spec calls unobservable is left out; everything else goes in.

    The identity fields, and why each is separate:

    ``source``      Pytest module stem. ``test_name`` is the .cpp path, and
                    several modules share a driver source, so without this two
                    unrelated sweeps could collide.
    ``run_count``   Changes the shape of the stats frame (``std`` columns
                    appear), so a cached frame from a different count would
                    make sibling cases emit different columns.
    ``stimuli``     Via ``stimuli_key`` -- the L1 layout, which the template
                    and runtime items do not determine.
    ``dest_acc``, ``unpack_to_dest``, ``unpack_to_srcs``, ``l1_acc``
                    Always included; each changes dest geometry or the L1 read
                    path for every thread.
    ``speed_of_light``
                    SoL and non-SoL measurements are never interchangeable.

    Template items are included when ``_template_visible``, so a non-pinnable
    template stays in the key even under ``PIN_ALL``. Runtime items are
    filtered by type and then by field name. Format items are read from the
    first FormatConfig and sorted, so the tuple is order-stable across
    processes.

    Example:
    Matmul: UNPACK cannot see fidelity, MATH can, and PACK cannot see kt while
    L1_TO_L1 can:
    >>> def key(run_type, fidelity, kt):
    ...     return execute_key(
    ...         test_name="sources/matmul_test.cpp", run_type=run_type,
    ...         dest_acc="No", source="perf_matmul", speed_of_light=True,
    ...         templates=[MATH_FIDELITY(fidelity), DEST_SYNC(DestSync.Half),
    ...                    PERF_RUN_TYPE(run_type)],
    ...         runtimes=[CRK_TILE_DIMM(4, 2, kt), LOOP_FACTOR(64)],
    ...         formats=fmt, spec=MATMUL_RELEVANCE[run_type],
    ...     )
    >>> lo, hi = MathFidelity.LoFi, MathFidelity.HiFi4
    >>> key(PerfRunType.UNPACK_ISOLATE, lo, 4) == key(
    ...     PerfRunType.UNPACK_ISOLATE, hi, 4)      # reuse
    True
    >>> key(PerfRunType.MATH_ISOLATE, lo, 4) == key(
    ...     PerfRunType.MATH_ISOLATE, hi, 4)        # fidelity drives math
    False
    >>> key(PerfRunType.PACK_ISOLATE, lo, 4) == key(
    ...     PerfRunType.PACK_ISOLATE, lo, 32)       # pack iterates RT x CT
    True
    >>> key(PerfRunType.L1_TO_L1, lo, 4) == key(
    ...     PerfRunType.L1_TO_L1, lo, 32)           # never reuses
    False

    The PACK key above, with only the fields that survive its spec:
    >>> key(PerfRunType.PACK_ISOLATE, hi, 32)
    ('perf_matmul', 'sources/matmul_test.cpp', <PerfRunType.PACK_ISOLATE: 4>,
     1, 'No', False, False, None,
     (('dest_sync', <DestSync.Half: 'SyncHalf'>),),
     (('c_dimm', 4), ('r_dimm', 2), ('loop_factor', 64)),
     (('pack_S_dst', ...), ('pack_S_src', ...), ('pack_dst', ...),
      ('pack_src', ...)),
     (), True)
    """
    template_items: list[tuple[str, Any]] = []
    for param in templates:
        if isinstance(param, PERF_RUN_TYPE):
            continue
        if _template_visible(param, spec):
            template_items.extend(_dataclass_items(param))

    runtime_items: list[tuple[str, Any]] = []
    for param in runtimes:
        if spec is not None and spec.runtime_types is not None:
            if type(param) not in spec.runtime_types:
                continue
        for name, value in _dataclass_items(param):
            if spec is not None and spec.runtime_fields is not None:
                if name not in spec.runtime_fields:
                    continue
            runtime_items.append((name, value))

    format_items: list[tuple[str, Any]] = []
    fmt = None
    if formats:
        fmt = formats[0] if isinstance(formats, list) else formats
    names = (
        spec.format_fields
        if spec is not None and spec.format_fields is not None
        else _ALL_FORMATS
    )
    if fmt is not None:
        for name in sorted(names):
            format_items.append((name, _hashable(getattr(fmt, name, None))))

    return (
        source,
        test_name,
        run_type,
        run_count,
        _hashable(dest_acc),
        _hashable(unpack_to_dest),
        _hashable(unpack_to_srcs),
        _hashable(l1_acc),
        tuple(template_items),
        tuple(runtime_items),
        tuple(format_items),
        stimuli_key(stimuli),
        speed_of_light,
    )
