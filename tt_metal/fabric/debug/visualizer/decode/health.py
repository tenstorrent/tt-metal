# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Turns one router's liveness samples and main pass into facts about whether it is running: whether each of its ERISCs
is held in reset, how its wall clock moves, and whether its heartbeat moves. It reads the manifest's names for these
(the soft_reset and wall clock core registers, the heartbeat area's magic, each ERISC's physical_erisc). Judging the
facts is left to whoever reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import get_router
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter

# The manifest names health reads.
SOFT_RESET = "soft_reset"
WALL_CLOCK_LOW = "wall_clock_low"
WALL_CLOCK_HIGH = "wall_clock_high"
HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class EriscReset:
    physical_erisc: int  # the ERISC on the Ethernet core that the router's processor runs on
    mask: int  # that ERISC's bit in soft_reset
    in_reset: tuple[bool | None, ...]  # per sample; None where soft_reset wasn't read


@dataclass(frozen=True)
class WallClock:
    cycles: tuple[int | None, ...]  # per sample, the two words joined; None where either wasn't read
    rises: bool | None  # whether each value read is above the one before; None with fewer than two values
    mhz: float | None  # cycles per microsecond from the first value read to the last; None with fewer than two


@dataclass(frozen=True)
class Heartbeat:
    words: tuple[int | None, ...]  # per sample; None where it wasn't read
    router_magic: tuple[bool | None, ...]  # per sample, whether the word carries the router's magic
    changed: bool | None  # whether the words with the router's magic differ; None with fewer than two of them


@dataclass(frozen=True)
class Health:
    """One router's health facts. Each tuple has one entry per sample: the liveness rounds, then the main pass."""

    t_ns: tuple[int, ...]  # each sample's host time, from time.monotonic_ns on the capture host
    eriscs: dict[str, EriscReset]  # by the router's erisc<n> key
    wall_clock: WallClock
    heartbeat: Heartbeat | None  # None when the manifest's heartbeat area is null


@dataclass(frozen=True)
class _Sample:
    t_ns: int
    core_registers: dict
    heartbeat: int | None


def _get_samples(entry: dict, router: DecodedRouter) -> list[_Sample]:
    """
    The liveness rounds, then the main pass. The main pass reads the core registers first and the heartbeat area with
    the other areas, a little later; its sample takes the first read's time.
    """
    samples = [_Sample(s["t_ns"], s.get("core_registers", {}), s.get("heartbeat")) for s in entry["liveness"]]
    if entry["started_ns"] is not None:
        samples.append(_Sample(entry["started_ns"], entry["core_registers"], router.areas.get(HEARTBEAT)))
    return samples


def _get_reset_masks(manifest: dict, where: str) -> list[int]:
    for index, register in enumerate(lookup(manifest, where, "array", of="object")):
        if register.get("name") == SOFT_RESET:
            return lookup(register, "reset_masks", "array", of="integer", where=f"{where}[{index}]")
    raise ManifestError(f"{where} has no {SOFT_RESET}")


def _get_eriscs(router_node: dict, where: str, masks: list[int], samples: list[_Sample]) -> dict[str, EriscReset]:
    eriscs = {}
    for key, erisc in lookup(router_node, "eriscs", "object", of="object", where=where).items():
        physical_erisc = lookup(erisc, "physical_erisc", "integer", where=f"{where}.eriscs.{key}")
        if not 0 <= physical_erisc < len(masks):
            raise ManifestError(
                f"{where}.eriscs.{key}.physical_erisc is {physical_erisc}; soft_reset has {len(masks)} reset masks"
            )
        mask = masks[physical_erisc]
        in_reset = tuple(
            None if sample.core_registers.get(SOFT_RESET) is None else bool(sample.core_registers[SOFT_RESET] & mask)
            for sample in samples
        )
        eriscs[key] = EriscReset(physical_erisc, mask, in_reset)
    return eriscs


def _get_wall_clock(samples: list[_Sample]) -> WallClock:
    cycles = []
    for sample in samples:
        low, high = sample.core_registers.get(WALL_CLOCK_LOW), sample.core_registers.get(WALL_CLOCK_HIGH)
        cycles.append(None if low is None or high is None else high << 32 | low)
    read = [(sample.t_ns, value) for sample, value in zip(samples, cycles) if value is not None]
    if len(read) < 2:
        return WallClock(tuple(cycles), None, None)
    rises = all(later > earlier for (_, earlier), (_, later) in zip(read, read[1:]))
    (first_ns, first), (last_ns, last) = read[0], read[-1]
    mhz = (last - first) * 1000 / (last_ns - first_ns) if last_ns > first_ns else None
    return WallClock(tuple(cycles), rises, mhz)


def _get_heartbeat(area: dict, where: str, samples: list[_Sample]) -> Heartbeat:
    magic = lookup(area, "magic", "integer", where=where)
    magic_mask = lookup(area, "magic_mask", "integer", where=where)
    words = tuple(sample.heartbeat for sample in samples)
    router_magic = tuple(None if word is None else (word & magic_mask) == magic for word in words)
    router_words = [word for word, is_router in zip(words, router_magic) if is_router]
    changed = len(set(router_words)) > 1 if len(router_words) >= 2 else None
    return Heartbeat(words, router_magic, changed)


def get_health(snapshot: Input, router: DecodedRouter) -> Health:
    """
    The health facts of a decoded router, from its snapshot entry's liveness samples and main pass. A router capture
    couldn't open has no samples. Raises ManifestError when the manifest lacks a name health reads.
    """
    manifest = snapshot.manifest
    arch_where = f"archs.{lookup(manifest, 'run.arch', 'string')}"
    entry = snapshot.json["routers"][router.path]
    router_node, router_where = get_router(manifest, router.path)
    samples = _get_samples(entry, router)

    masks = _get_reset_masks(manifest, f"{arch_where}.core_registers")
    area = lookup(manifest, f"{arch_where}.areas.{HEARTBEAT}", ("object", "null"))
    heartbeat = None if area is None else _get_heartbeat(area, f"{arch_where}.areas.{HEARTBEAT}", samples)
    return Health(
        t_ns=tuple(sample.t_ns for sample in samples),
        eriscs=_get_eriscs(router_node, router_where, masks, samples),
        wall_clock=_get_wall_clock(samples),
        heartbeat=heartbeat,
    )
