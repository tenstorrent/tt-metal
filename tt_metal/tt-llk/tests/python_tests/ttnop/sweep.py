# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What to perturb, and the loop that perturbs it.

Nothing here knows how a variant is written to a device; the caller hands in a
runtime that can run one variant and say how it went. That is the seam the Metal
backend would slot into.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from cave import DEFAULT_MAX_DELAY, filler_choices

HERE = Path(__file__).resolve().parent
# Every count up to the cave's capacity. A race often only opens inside a narrow
# band of delays, so sampling powers of two walks straight past it.
DEFAULT_DELAYS = f"1-{DEFAULT_MAX_DELAY}"
FILLER_NAMES = ("tti_nop", "sfpnop", "unpacr0", "unpacr1", "risc_nop")


@dataclass(frozen=True)
class Variant:
    thread: str
    site: object
    filler: str
    filler_word: int
    delay: int
    # Position in the full plan, kept so a record can be put back in sweep order
    # after several workers have appended to one log.
    seq: int = 0

    def label(self) -> str:
        return f"{self.thread} {self.site.label()} n={self.delay} {self.filler}"


def parse_delays(spec: str) -> tuple:
    """Delay counts from a spec: ints and lo-hi ranges, comma separated.

    "1-100" is the sweep, "1,5,10,20,40,60,80,100" a coarse probe, and the two mix
    freely ("1-8,16,32"). Order is kept and duplicates dropped, so the caller can
    put the interesting counts first.
    """
    delays = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, dash, high = part.partition("-")
        wanted = range(int(low, 0), int(high, 0) + 1) if dash else [int(low, 0)]
        delays.extend(delay for delay in wanted if delay not in delays)
    return tuple(delays)


def shard_from_env() -> tuple:
    """Which slice of the plan this process owns, as (shard, shards).

    Only a depth run splits a plan: it hands the same case to every xdist worker
    and sets TTNOP_SHARD_VARIANTS, so worker gwN takes every Nth variant and the
    eight cores share one sweep. A breadth run gives each worker a *different*
    case, and must keep the whole plan for the case it was given — hence the
    explicit opt-in rather than reading the xdist variables on their own.
    """
    if os.environ.get("TTNOP_SHARD_VARIANTS", "") in ("", "0"):
        return 0, 1
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not worker.startswith("gw"):
        return 0, 1
    shard = int(worker[2:])
    shards = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1") or 1)
    # A shard outside the count would silently drop the variants ahead of it.
    return (shard, shards) if shard < shards else (0, 1)


@dataclass
class Config:
    site_mode: str = "sync"
    threads: tuple = ("unpack", "math")  # pack skipped for now
    delays: tuple = field(default_factory=lambda: parse_delays(DEFAULT_DELAYS))
    max_delay: int = DEFAULT_MAX_DELAY
    filler: str = "auto"
    selector: str = ""
    repeats: int = 1
    # Which slice of the plan this process runs, when the runner has put the same
    # case on every xdist worker. (0, 1) is the whole thing.
    shard: int = 0
    shards: int = 1
    # Freeze the stimuli and compare each variant's output against the baseline's.
    # Off restores the rolling RNG stream, which samples different data per variant
    # but leaves nothing to compare against.
    drift: bool = True
    arch: str = "wormhole"
    report_dir: Path = field(default_factory=lambda: HERE / "reports")

    @classmethod
    def from_env(cls):
        shard, shards = shard_from_env()
        config = cls(
            site_mode=os.environ.get("TTNOP_SITE_MODE", "sync").strip().lower(),
            threads=tuple(
                part.strip()
                for part in os.environ.get("TTNOP_THREADS", "unpack,math").split(",")
                if part.strip()
            ),
            delays=parse_delays(
                os.environ.get("TTNOP_DELAYS", "").strip() or DEFAULT_DELAYS
            ),
            max_delay=int(os.environ.get("TTNOP_MAX_DELAY", DEFAULT_MAX_DELAY)),
            filler=os.environ.get("TTNOP_FILLER", "auto").strip().lower(),
            selector=os.environ.get("TTNOP_SITES", "").strip(),
            repeats=int(os.environ.get("TTNOP_REPEATS", "1")),
            shard=shard,
            shards=shards,
            drift=os.environ.get("TTNOP_DRIFT", "1").strip() not in ("", "0"),
            arch=os.environ.get("CHIP_ARCH", "wormhole").strip().lower(),
            report_dir=Path(
                os.environ.get("TTNOP_REPORT_DIR", str(HERE / "reports"))
            ).resolve(),
        )
        if config.site_mode not in ("sync", "all"):
            raise ValueError(
                f"TTNOP_SITE_MODE must be sync or all, got {config.site_mode!r}"
            )
        if config.filler != "auto" and config.filler not in FILLER_NAMES:
            try:
                int(config.filler, 0)
            except ValueError:
                raise ValueError(
                    f"TTNOP_FILLER must be auto, a raw word, or one of {FILLER_NAMES}"
                ) from None
        if not config.delays:
            raise ValueError("TTNOP_DELAYS selected no counts")
        out_of_range = [d for d in config.delays if not 0 <= d <= config.max_delay]
        if out_of_range:
            raise ValueError(
                f"delay(s) {out_of_range} outside 0..TTNOP_MAX_DELAY={config.max_delay}; "
                "raise TTNOP_MAX_DELAY to grow the cave"
            )
        return config

    @property
    def forced_filler(self):
        return None if self.filler == "auto" else self.filler

    def selected_indices(self, thread: str):
        """Parse TTNOP_SITES ("unpack:3,math:7"). None means every site on that thread."""
        if not self.selector:
            return None
        chosen = set()
        for part in self.selector.split(","):
            name, _, index = part.strip().partition(":")
            if name == thread and index:
                chosen.add(int(index, 0))
        return chosen


def plan(config: Config, scans: dict) -> list:
    """Expand the config into the ordered list of variants this process will try."""
    variants = []
    for thread in config.threads:
        scan = scans.get(thread)
        if scan is None or not scan.cave_start:
            continue
        wanted = config.selected_indices(thread)
        for site in scan.sites:
            if wanted is not None and site.index not in wanted:
                continue
            for name, word in filler_choices(thread, site, scan, config.forced_filler):
                # A depth run needs its own control: delay 0 still detours through the
                # cave, so it separates "the jump broke it" from "the fillers broke it".
                delays = (0,) + config.delays if config.repeats > 1 else config.delays
                for delay in delays:
                    variants.append(
                        Variant(thread, site, name, word, delay, len(variants))
                    )
    # Every Nth variant rather than an Nth of the list: a worker still walks one
    # site and filler at a time (so a delay step stays a single word write), and
    # the expensive high counts spread over the workers instead of landing on one.
    return variants[config.shard :: config.shards]


class DeviceWedged(RuntimeError):
    """A variant hung the core, and only a card reset will clear it.

    Carries the label of the variant that did it: that variant is the race the
    sweep exists to find, and the reset about to follow takes the evidence with
    it, so the label has to travel with the exception.
    """


def run(config: Config, variants: list, runtime, record_sink) -> list:
    """Run every variant `repeats` times. Returns (label, tags) per recorded variant.

    The tags come back with the label because not every tag is a failure: a caller
    that only wants the red ones has to be able to tell `drift` from `mismatch`.
    """
    failing = []
    for variant in variants:
        fails, tags, first_error = 0, set(), ""
        # Record in a finally so the hang that ends the case below is still
        # reported; losing the finding is worse than losing the rest of the sweep.
        try:
            for _ in range(config.repeats):
                tag, error = runtime.run(variant)
                if tag is None:
                    continue
                fails += 1
                tags.add(tag)
                first_error = first_error or error
                if tag == "hang":
                    raise DeviceWedged(variant.label())
        finally:
            if fails:
                failing.append((f"{variant.label()} {','.join(sorted(tags))}", tags))
                record_sink(variant, fails, tags, first_error)
    return failing


def same_elements(before, after):
    """Elementwise equality, with NaN counted as equal to itself.

    Two runs of the same kernel on the same stimuli put the same NaN in the same
    place; plain `==` calls that a difference, which would report drift against
    every test whose golden contains one.
    """
    if before.is_floating_point():
        return (before == after) | (before.isnan() & after.isnan())
    return before == after


def describe_drift(before: list, after: list):
    """How `after` differs from `before`.

    Returns (message, pcc). Message is "" when the two runs agree exactly
    """
    if len(before) != len(after):
        return f"{len(after)} device run(s) against the baseline's {len(before)}", None
    for base, now in zip(before, after):
        if base is None and now is None:
            continue
        if base is None or now is None or base.shape != now.shape:
            return "result buffer changed shape or vanished", None
        same = same_elements(base, now)
        if bool(same.all()):
            continue
        moved = f"{int((~same).sum())} of {base.numel()} element(s)"
        try:
            from helpers.utils import calculate_pcc

            pcc = float(calculate_pcc(now, base))
            gap = (base.double() - now.double()).abs().max().item()
            if not math.isfinite(pcc):
                # Constant tensors have no correlation to report
                raise ValueError("pcc is not finite")
        except Exception:
            # Scoring is a nicety. It must never turn a drift finding into an error.
            return f"output changed vs baseline: {moved}", None
        return (
            f"output changed vs baseline: pcc={pcc:.6f} (Δ {1.0 - pcc:.2g}), "
            f"{moved}, max |delta|={gap:g}",
            pcc,
        )
    return "", None
