# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Eltwise SFPU kernel headers live in tt-metal's llk_sfpu, and only there.

Under #60919 the eltwise SFPU kernels moved from ``tt_llk_<arch>/common/inc/sfpu/`` into
``tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/``. tt-llk keeps the shared SFPU helpers
and a few non-eltwise kernels there (``SHARED``), plus ``experimental/``, which these
tests do not look at.

A ``ckernel_sfpu_<name>.h`` in both trees is two kernels behind one include, and which
one compiles depends on the build: tt-metal's JIT build does not search
``common/inc/sfpu``, while the LLK tests search it before ``llk_sfpu``. Once the two
copies drift, the LLK tests stop covering the kernel tt-metal runs, and nothing fails.

The same missing search path means an llk_sfpu header has to spell a header that
stayed in tt-llk as ``"sfpu/<name>"``. A bare ``"<name>"`` still compiles in the LLK
tests, and fails only when tt-metal JIT-builds the kernel on a device.

No hardware needed: the tests list the two directories and read the llk_sfpu headers.
Skip in standalone tt-llk checkouts, where llk_sfpu is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# tests/hardware_free/kernels -> hardware_free -> tests -> tt-llk -> tt_metal
_TT_LLK = Path(__file__).resolve().parents[3]
_TT_METAL = Path(__file__).resolve().parents[4]
ARCHES = ("wormhole_b0", "blackhole", "quasar")

# Not eltwise kernels, so they stay in tt-llk: the helpers that llk_lib, common/inc/ckernel_sfpu.h
# and the llk_sfpu kernels include (converter, load_config, welfords), and Blackhole's triangle_solve.
SHARED = frozenset(
    {
        "ckernel_sfpu_converter.h",
        "ckernel_sfpu_load_config.h",
        "ckernel_sfpu_welfords.h",
        "ckernel_sfpu_triangle_solve.h",
    }
)

# Eltwise kernels that #56276 moves into llk_sfpu on Wormhole and Blackhole. They stay in tt-llk
# until it lands; test_exceptions_are_current fails once a name here has moved.
AWAITING_MOVE = frozenset(
    {
        "ckernel_sfpu_add_int.h",
        "ckernel_sfpu_cdf.h",
        "ckernel_sfpu_ema.h",
        "ckernel_sfpu_isinf_isnan.h",
        "ckernel_sfpu_mul_int.h",
        "ckernel_sfpu_reduce_custom.h",
        "ckernel_sfpu_rounding_ops.h",
        "ckernel_sfpu_sub_int.h",
        "ckernel_sfpu_threshold.h",
        "ckernel_sfpu_where.h",
    }
)
# Of those, the ones llk_sfpu already has a same-named header for: a forward to "sfpu/<name>", so
# the Compute API includes one name on every arch. Both copies resolve to the tt-llk kernel.
FORWARDED = frozenset(
    {
        "ckernel_sfpu_isinf_isnan.h",
        "ckernel_sfpu_sub_int.h",
        "ckernel_sfpu_threshold.h",
    }
)
_AWAITING_ARCHES = ("wormhole_b0", "blackhole")

_BARE_INCLUDE = re.compile(r'\s*#\s*include\s*"(ckernel_sfpu_\w+\.h)"')


def _tt_llk_sfpu(arch: str) -> Path:
    return _TT_LLK / f"tt_llk_{arch}" / "common" / "inc" / "sfpu"


def _llk_sfpu(arch: str) -> Path:
    return _TT_METAL / "hw" / "ckernels" / arch / "metal" / "llk_api" / "llk_sfpu"


def _rel(path: Path) -> str:
    return str(path.relative_to(_TT_METAL.parent))


def _headers(directory: Path) -> set[str]:
    """Basenames of the ``*.h`` files directly in `directory`; none if it does not exist."""
    return {p.name for p in directory.glob("*.h") if p.is_file()}


def _trees(arch: str) -> tuple[set[str], set[str]]:
    """The tt-llk sfpu and llk_sfpu header basenames for `arch`, or skip outside tt-metal.

    Quasar's tt-llk sfpu directory is gone, so an empty set there is expected. An empty
    llk_sfpu, or a missing tt-llk arch tree, means a path here went stale -- and every test
    would then pass on nothing -- so those fail instead.
    """
    if not (_TT_METAL / "hw" / "ckernels").is_dir():
        pytest.skip(
            "standalone tt-llk checkout: no llk_sfpu to compare tt-llk's sfpu against"
        )
    llk_sfpu = _headers(_llk_sfpu(arch))
    assert (
        llk_sfpu
    ), f"no headers in {_rel(_llk_sfpu(arch))} -- moved? Update this test."
    inc = _tt_llk_sfpu(arch).parent
    assert inc.is_dir(), f"{_rel(inc)} does not exist -- moved? Update this test."
    return _headers(_tt_llk_sfpu(arch)), llk_sfpu


def _awaiting(arch: str) -> frozenset[str]:
    return AWAITING_MOVE if arch in _AWAITING_ARCHES else frozenset()


def _forwarded(arch: str) -> frozenset[str]:
    return FORWARDED if arch in _AWAITING_ARCHES else frozenset()


@pytest.mark.parametrize("arch", ARCHES)
def test_no_header_in_both_trees(arch: str):
    """A basename may be in tt-llk's sfpu or in llk_sfpu, not in both."""
    tt_llk, llk_sfpu = _trees(arch)
    both = sorted((tt_llk & llk_sfpu) - _forwarded(arch))
    assert not both, (
        f"{arch}: {both} exist in both {_rel(_tt_llk_sfpu(arch))} and "
        f"{_rel(_llk_sfpu(arch))}, so which one a bare include compiles depends on the "
        "build. Keep only the llk_sfpu one."
    )


@pytest.mark.parametrize("arch", ARCHES)
def test_tt_llk_sfpu_holds_only_shared_headers(arch: str):
    """tt-llk's sfpu directory keeps the SHARED headers, and no eltwise kernels."""
    tt_llk, _ = _trees(arch)
    extra = sorted(tt_llk - SHARED - _awaiting(arch))
    assert not extra, (
        f"{arch}: {extra} in {_rel(_tt_llk_sfpu(arch))}. Eltwise SFPU kernels belong in "
        f"{_rel(_llk_sfpu(arch))}. If llk_lib itself needs the header, add it to SHARED "
        "in this test instead."
    )


@pytest.mark.parametrize("arch", ARCHES)
def test_exceptions_are_current(arch: str):
    """AWAITING_MOVE and FORWARDED name only headers that are still where they say.

    A stale entry would let that name back into tt-llk unnoticed.
    """
    tt_llk, llk_sfpu = _trees(arch)
    moved = sorted(_awaiting(arch) - tt_llk)
    assert not moved, (
        f"{arch}: {moved} are no longer in {_rel(_tt_llk_sfpu(arch))}. Remove them "
        "from AWAITING_MOVE, and from FORWARDED, in this test."
    )
    unforwarded = sorted(_forwarded(arch) - llk_sfpu)
    assert not unforwarded, (
        f"{arch}: {unforwarded} are no longer in {_rel(_llk_sfpu(arch))}. Remove them "
        "from FORWARDED in this test."
    )


@pytest.mark.parametrize("arch", ARCHES)
def test_llk_sfpu_includes_tt_llk_headers_by_sfpu_path(arch: str):
    """An llk_sfpu header includes a header that stayed in tt-llk as "sfpu/<name>"."""
    tt_llk, llk_sfpu = _trees(arch)
    bare = []
    for path in sorted(_llk_sfpu(arch).glob("*.h")):
        for number, text in enumerate(path.read_text().splitlines(), 1):
            match = _BARE_INCLUDE.match(text)
            if match and match[1] in tt_llk and match[1] not in llk_sfpu:
                bare.append(f"{path.name}:{number}: {match[1]}")
    assert not bare, (
        f"{arch}: llk_sfpu headers include tt-llk sfpu headers by bare name: {bare}. "
        "tt-metal's JIT build does not search tt-llk's common/inc/sfpu, so these fail "
        'to compile there (the LLK tests do search it). Spell them "sfpu/<name>".'
    )
