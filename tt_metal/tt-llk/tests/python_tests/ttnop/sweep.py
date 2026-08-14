# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What to perturb, and the loop that perturbs it.

Nothing here knows how a variant is written to a device; the caller hands in a
runtime that can run one variant and recover from a hang. That is the seam the
Metal backend would slot into.
"""

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


@dataclass
class Config:
    site_mode: str = "sync"
    threads: tuple = ("unpack", "math")  # pack skipped for now
    delays: tuple = field(default_factory=lambda: parse_delays(DEFAULT_DELAYS))
    max_delay: int = DEFAULT_MAX_DELAY
    filler: str = "auto"
    selector: str = ""
    repeats: int = 1
    arch: str = "wormhole"
    report_dir: Path = field(default_factory=lambda: HERE / "reports")

    @classmethod
    def from_env(cls):
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
    """Expand the config into the ordered list of variants to try."""
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
                    variants.append(Variant(thread, site, name, word, delay))
    return variants


class DeviceWedged(RuntimeError):
    """Soft reset did not take. Every later variant would look like a failure."""


def run(config: Config, variants: list, runtime, record_sink) -> list:
    """Run every variant `repeats` times. Returns one label per failing variant."""
    failing = []
    for variant in variants:
        fails, tags, first_error = 0, set(), ""
        # Record in a finally so a hang we could not recover from is still reported;
        # losing the finding is worse than losing the rest of the sweep.
        try:
            for _ in range(config.repeats):
                tag, error = runtime.run(variant)
                if tag is None:
                    continue
                fails += 1
                tags.add(tag)
                first_error = first_error or error
                if tag == "hang" and not runtime.recover():
                    raise DeviceWedged(f"soft reset failed after {variant.label()}")
        finally:
            if fails:
                failing.append(f"{variant.label()} {','.join(sorted(tags))}")
                record_sink(variant, fails, tags, first_error)
    return failing
