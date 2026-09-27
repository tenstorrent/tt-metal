# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4b — the device-profiler zone-hash guard.

WHAT THIS PINS, AND WHY IT IS NOT COSMETIC
------------------------------------------
`tt_metal/impl/profiler/profiler.cpp` keys every device zone by a **16-bit**
hash of the string

    "<zone name>,<absolute kernel source path>,<line number>,KERNEL_PROFILER"

emitted as a `#pragma message` by `DeviceZoneScopedN`.  `populateZoneSrcLocations()`
does a hard `TT_THROW("Source location hashes are colliding, ...")` the instant two
DISTINCT such strings land on the same 16-bit value.  That throw fires on *every*
profiler read and eventually escapes as `terminate` at a device re-open — it takes
the whole pytest process down and with it the junit.xml, so a suite that was
otherwise green reports **zero tests run**.

The line number is part of the hashed string.  So every edit to a kernel file
re-rolls the dice for every zone in that file, and this op declares 35 distinct
stage names across ~41 sites: against a 65 536-slot space that is a ~2–3%
birthday collision *per kernel edit*.  Refinement 4's edits hit it —
`compute_scale`@`rms_norm_ttnn_compute.cpp:1680` and
`writer_tree_forward`@`rms_norm_ttnn_writer.cpp:662` both hash to `0x0773`.

Two things now stand between that hazard and a graded run:

  1. the zones are **opt-in** (`RMS_STAGE_ZONES` — see
     `kernels/perf_instrumentation.hpp` and `STAGE_ZONES` in the descriptor), so
     a run that merely sets `TT_METAL_DEVICE_PROFILER=1` registers **no** op zone
     locations at all; and
  2. **this test**, which re-implements the profiler's own hash over the CURRENT
     source lines and fails if the zones-ON build would collide.

If this ever goes red the fix is one line: move ONE of the two named zones by a
single line (a blank line above it is enough) and re-run.

No device is needed — this is pure source analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).resolve().parents[5] / "ttnn" / "ttnn" / "operations" / "rms_norm_ttnn" / "kernels"

#: Exactly the sites `perf_instrumentation.hpp`'s macro expands.
_ZONE_RE = re.compile(r'\bMaybeDeviceZoneScope\("([^"]+)"\)')

#: The other zone families the profiler already carries in the same 16-bit table
#: on any run of this op: the per-RISC firmware markers and the profiler's own
#: internal ones.  They are NOT ours to move, but they occupy slots, so the guard
#: has to price them.  Names + files + lines are read off
#: `generated/profiler/.logs/zone_src_locations.log` on a profiled run; the count
#: is what matters here, and it is small and stable (~16).
_FOREIGN_ZONE_SLOTS = 16


def _hash32(s: str) -> int:
    """FNV-1a, byte for byte with `hash32CT` in profiler.cpp."""
    basis = 2166136261
    for b in s.encode():
        basis = ((basis ^ b) * 16777619) & 0xFFFFFFFF
    return basis


def _hash16(s: str) -> int:
    """`hash16CT` in profiler.cpp: fold the 32-bit FNV-1a down to 16 bits."""
    res = _hash32(s)
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF


def _zone_src_locations() -> list[tuple[str, str]]:
    """Every `(hashed string, human label)` this op would register with the
    profiler when built with `RMS_STAGE_ZONES=1`.

    The hashed string is assembled exactly as the compiler's `#pragma message`
    emits it: zone name, ABSOLUTE source path, 1-based line, `KERNEL_PROFILER`.
    """
    out: list[tuple[str, str]] = []
    for src in sorted(KERNEL_DIR.glob("rms_norm_ttnn_*.cpp")):
        for lineno, line in enumerate(src.read_text().splitlines(), start=1):
            m = _ZONE_RE.search(line)
            if not m:
                continue
            name = m.group(1)
            out.append((f"{name},{src},{lineno},KERNEL_PROFILER", f"{name}@{src.name}:{lineno}"))
    return out


def test_zone_sites_are_discoverable():
    """Guard the guard: if the macro is ever renamed this test must not quietly
    pass by finding nothing."""
    zones = _zone_src_locations()
    assert len(zones) >= 30, f"expected the op's ~41 MaybeDeviceZoneScope sites, found {len(zones)}"


def test_zone_source_locations_do_not_collide_in_16_bits():
    """The invariant Refinement 4 broke, stated directly."""
    by_hash: dict[int, str] = {}
    collisions: list[str] = []
    for hashed, label in _zone_src_locations():
        h = _hash16(hashed)
        if h in by_hash and by_hash[h] != label:
            collisions.append(f"0x{h:04X}: {by_hash[h]}  <->  {label}")
        else:
            by_hash[h] = label
    assert not collisions, (
        "two device zones hash to the same 16-bit profiler slot; a profiled build would "
        "TT_THROW on every profiler read and `terminate` the process at the next device open.\n"
        "Fix: move ONE of the two by a single line.\n  " + "\n  ".join(collisions)
    )


def test_zone_population_stays_under_the_birthday_budget():
    """A soft ceiling on how many slots the op is allowed to occupy.

    Collision probability is ~1 - exp(-n(n-1)/2/65536); at n = 128 that is 11%,
    which is too high a per-edit failure rate for a permanent instrument.  This is
    a design budget, not a hard mechanism limit — if a refinement genuinely needs
    more stages, split them behind separate defines rather than raising this.
    """
    n = len(_zone_src_locations()) + _FOREIGN_ZONE_SLOTS
    assert n <= 128, f"{n} zone source locations is past the birthday budget for a 16-bit hash"


@pytest.mark.parametrize(
    "text, expected",
    [
        # Clone-path-independent fixtures, so the pin survives a move of the tree.
        ("KERNEL_PROFILER", 0x13E6),
        ("", 0x1CD9),
    ],
)
def test_hash16_matches_the_profiler_implementation(text, expected):
    """`hash16CT` is re-implemented here; pin it so a drift in profiler.cpp is
    visible as a failing expectation rather than a silently useless guard."""
    assert _hash16(text) == expected


# ---------------------------------------------------------------------------
#  Pre-flight: the CACHE can replay zone strings the source no longer has
# ---------------------------------------------------------------------------
# `extract_zone_src_locations()` in `tt_metal/jit_build/build.cpp` harvests zone
# `#pragma message`es out of the PREPROCESSED `.ii` (and `*.o.log`) sitting in each
# kernel's build directory — on the ELF-REUSE path as well as the compile path.  A
# build dir is keyed coarsely enough to be reused across source edits, so a `.ii`
# preprocessed from a SUPERSEDED version of a kernel keeps re-registering that
# version's zone strings, at line numbers the file no longer has, for as long as the
# cache lives.  That is how Refinement 4's collision survived the source fix: the
# hash table is populated from the cache, not from the tree.
#
# Two consequences, both handled here:
#   * with `RMS_STAGE_ZONES` off (the default) a fresh build contributes NOTHING, so
#     the population can only shrink — the hazard is closed going forward; but
#   * a profiling session run with `RMS_STAGE_ZONES=1` re-seeds the cache with zone-
#     carrying `.ii`, and a LATER graded run will harvest them.  After such a
#     session, purge:
#
#       find built -path '*/kernels/rms_norm_ttnn_*' \( -name '*.ii' -o -name '*.o.log' \) \
#            ! -newer ttnn/ttnn/operations/rms_norm_ttnn/kernels/perf_instrumentation.hpp -delete
#       # and drop the op's rows from generated/profiler/.logs/*zone_src_locations.log
#
# This test reads the ACCUMULATED log the profiler loads at startup and fails on the
# exact condition that crashes a run — a 16-bit collision among the strings actually
# registered.  It is a pre-flight check on the machine, not on the source, so it
# skips when the log is absent.
_ZONE_LOG = Path(__file__).resolve().parents[5] / "generated" / "profiler" / ".logs" / "zone_src_locations.log"
_ZONE_LOG_DELIM = "'#pragma message: "


def _registered_zone_strings(path: Path) -> list[str]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        i = line.find(_ZONE_LOG_DELIM)
        if i < 0:
            continue
        out.append(line[i + len(_ZONE_LOG_DELIM) : len(line) - 1])
    return out


def test_accumulated_zone_log_has_no_collision():
    """The condition that actually terminates a profiled run, checked on the file
    the profiler really loads."""
    if not _ZONE_LOG.exists():
        pytest.skip(f"no accumulated zone log at {_ZONE_LOG} — nothing registered yet")
    by_hash: dict[int, str] = {}
    seen: set[str] = set()
    collisions: list[str] = []
    for s in _registered_zone_strings(_ZONE_LOG):
        if s in seen:
            continue
        seen.add(s)
        h = _hash16(s)
        if h in by_hash:
            collisions.append(f"0x{h:04X}: {by_hash[h]}  <->  {s}")
        else:
            by_hash[h] = s
    assert not collisions, (
        f"{_ZONE_LOG} registers two zone locations on the same 16-bit slot; every profiler "
        "read will TT_THROW and the process will `terminate` at the next device open "
        "(this is what reported TOTAL=0 for the Refinement 4 golden run).\n"
        "Purge the stale cache artifacts and the op's rows from the log — see the comment "
        "above this test for the exact commands.\n  " + "\n  ".join(collisions)
    )
