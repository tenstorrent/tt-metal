# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vision tower's residency in a served process: row buckets, the DRAM admission, the warm-hook load and prewarm.

The served chain forbids program-cache misses after capture, so every image is padded to one of
``VISION_ROW_BUCKETS`` and each bucket is compiled in the warm hook; an image above the largest bucket is refused.
Padded patches form their own attention window, so padding costs time only. When the live DRAM admission does not
fit, the server starts text-only and image requests are refused with the shortfall's reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import vision_resident_layout
from models.demos.blackhole.qwen38_flash_next.vision_reference import VisionTowerConfig

# Rows per bucket: powers of two up to the stock maximum image (65,536 patches = 4096 x 4096 px). The eight-bucket
# ladder pads at most 2x the rows, the four-bucket ladder at most 4x.
EIGHT_BUCKET_LADDER: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
FOUR_BUCKET_LADDER: tuple[int, ...] = (1024, 4096, 16384, 65536)
# The stock ladder, picked by ``ladder_comparison`` from the measured READY costs.
VISION_ROW_BUCKETS: tuple[int, ...] = EIGHT_BUCKET_LADDER
READY_COST_DIFFERENCE_SECONDS_FOR_EIGHT = 30.0
# READY prewarm totals in seconds (both window forms per bucket); from device measurements.
PREWARM_READY_SECONDS_MEASURED = {"eight": 43.65, "four": 33.47}
PREWARM_READY_SECONDS_COLD_MEASURED = {"eight": 103.11}
STOCK_MAXIMUM_PATCHES = 65536
# Upper bound on the tower's transient DRAM per padded row per die; from a device measurement.
PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE = 18_432
# The largest single activation buffer: the fused qkv output [rows, 3 x 16 x 96] BF16 (its allocation must fit the
# largest contiguous block per bank).
QKV_ACTIVATION_BYTES_PER_ROW_PER_DIE = 3 * 16 * 96 * 2
RESIDENT_DRAM_BANKS = 8
# The margin the MTP and lanes admissions carry on their modeled terms.
VISION_GROWTH_ESTIMATE_MARGIN_PERCENT = 10
# Modeled compile plus input-tilize seconds of one bucket's shape set at READY; the forward law below is
# t(rows) = A rows + B rows^2. Both are fitted to device measurements.
PREWARM_COMPILE_SECONDS_PER_BUCKET_MODELED = 12.0
PREWARM_FIRST_BUCKET_EXTRA_SECONDS_MODELED = 3.0  # the tower's shape-independent programs compile with the first bucket
WARM_FORWARD_SECONDS_PER_ROW = 1.91e-5
WARM_FORWARD_SECONDS_PER_ROW_SQUARED = 2.6e-9
PREWARM_SECONDS_MEASURED = {1024: 14.6, 4096: 13.1, 16384: 13.1, 65536: 22.3}  # cold, one form
# Steady state per bucket (both forms, warm JIT cache).
PREWARM_SECONDS_STEADY_MEASURED = {
    512: 0.27,
    1024: 0.25,
    2048: 0.32,
    4096: 0.51,
    8192: 1.03,
    16384: 2.66,
    32768: 8.52,
    65536: 30.1,
}


def bucket_rows(patches: int, buckets: Sequence[int] = VISION_ROW_BUCKETS) -> int:
    """The bucket an image of ``patches`` patches runs in: the smallest bucket with at least that many rows.

    Refuses an image above the largest bucket (the stock maximum): its programs would not exist in the served process.
    """

    if isinstance(patches, bool) or type(patches) is not int or patches <= 0:
        raise ValueError(f"patches must be a positive int, got {patches!r}")
    ordered = sorted(int(b) for b in buckets)
    for rows in ordered:
        if patches <= rows:
            return rows
    raise ValueError(
        f"an image of {patches} patches exceeds the largest row bucket {ordered[-1]} (the stock maximum "
        f"image, {STOCK_MAXIMUM_PATCHES} patches = 4096 x 4096 px); no program for it exists in a served process"
    )


def padding_cost(patches: int, buckets: Sequence[int] = VISION_ROW_BUCKETS) -> dict[str, float | int]:
    """What the bucket costs an image in work: the padded rows, and the linear and attention work relative to the
    image's own (the padded patches attend only among themselves, so attention grows by the pad window's square)."""

    rows = bucket_rows(patches, buckets)
    pad = rows - patches
    return {
        "patches": patches,
        "rows": rows,
        "pad_rows": pad,
        "linear_work_factor": rows / patches,
        "attention_work_factor": (patches**2 + pad**2) / patches**2,
    }


def warm_forward_seconds_modeled(rows: int) -> float:
    return WARM_FORWARD_SECONDS_PER_ROW * rows + WARM_FORWARD_SECONDS_PER_ROW_SQUARED * rows * rows


def prewarm_plan(buckets: Sequence[int] = VISION_ROW_BUCKETS) -> dict[str, Any]:
    """The buckets the open compiles, ascending, each with its modeled READY cost and the measured one if known."""

    ordered = tuple(sorted(set(int(b) for b in buckets)))
    if not ordered or any(b % 32 or b <= 0 for b in ordered):
        raise ValueError(f"buckets must be positive tile multiples, got {buckets!r}")
    rows_list = []
    total = 0.0
    for index, rows in enumerate(ordered):
        modeled = (
            PREWARM_COMPILE_SECONDS_PER_BUCKET_MODELED
            + (PREWARM_FIRST_BUCKET_EXTRA_SECONDS_MODELED if index == 0 else 0.0)
            + warm_forward_seconds_modeled(rows)
        )
        total += modeled
        rows_list.append(
            {
                "rows": rows,
                "ready_seconds_modeled": round(modeled, 1),
                "ready_seconds_measured": PREWARM_SECONDS_MEASURED.get(rows),
                "warm_forward_seconds_modeled": round(warm_forward_seconds_modeled(rows), 3),
            }
        )
    return {"buckets": rows_list, "ready_seconds_modeled_total": round(total, 1), "count": len(ordered)}


def ladder_comparison(
    eight_ready_seconds: float | None = None, four_ready_seconds: float | None = None
) -> dict[str, Any]:
    """The two ladders side by side: steady-state READY costs (measured, or the caller's), modeled cold costs, worst
    padding, and the pick (four buckets unless the eight cost under 30 s more at READY)."""

    eight, four = prewarm_plan(EIGHT_BUCKET_LADDER), prewarm_plan(FOUR_BUCKET_LADDER)
    eight_cost = PREWARM_READY_SECONDS_MEASURED["eight"] if eight_ready_seconds is None else eight_ready_seconds
    four_cost = PREWARM_READY_SECONDS_MEASURED["four"] if four_ready_seconds is None else four_ready_seconds
    difference = eight_cost - four_cost
    return {
        "eight": {
            "buckets": list(EIGHT_BUCKET_LADDER),
            "ready_seconds": eight_cost,
            "ready_seconds_cold_modeled": eight["ready_seconds_modeled_total"],
            "worst_linear_work_factor": 2.0,
        },
        "four": {
            "buckets": list(FOUR_BUCKET_LADDER),
            "ready_seconds": four_cost,
            "ready_seconds_cold_modeled": four["ready_seconds_modeled_total"],
            "worst_linear_work_factor": 4.0,
        },
        "ready_seconds_difference": difference,
        "measured": True,
        "rule": f"four buckets unless the eight cost under {READY_COST_DIFFERENCE_SECONDS_FOR_EIGHT:.0f} s more at READY",
        "pick": "eight" if difference < READY_COST_DIFFERENCE_SECONDS_FOR_EIGHT else "four",
    }


def _per_bank(bytes_per_die: int, banks: int) -> int:
    return -(-bytes_per_die // banks)


def vision_capacity_admission(
    *,
    dies: int,
    live: Mapping[str, Any],
    reserved_bytes_per_bank: int = 0,
    buckets: Sequence[int] = VISION_ROW_BUCKETS,
) -> dict[str, Any]:
    """Whether the resident tower and its largest bucket's transient peak fit the live allocator.

    ``live`` is the mesh DRAM view read in the warm hook; ``reserved_bytes_per_bank`` is what the chain still allocates
    after that reading (its traces).
    """

    if isinstance(dies, bool) or type(dies) is not int or dies <= 0:
        raise ValueError(f"dies must be a positive int, got {dies!r}")
    if (
        isinstance(reserved_bytes_per_bank, bool)
        or type(reserved_bytes_per_bank) is not int
        or reserved_bytes_per_bank < 0
    ):
        raise ValueError(f"reserved_bytes_per_bank must be a non-negative int, got {reserved_bytes_per_bank!r}")
    banks = int(live.get("num_banks", RESIDENT_DRAM_BANKS))
    if banks != RESIDENT_DRAM_BANKS:
        raise ValueError(f"the live DRAM view has {banks} banks, the admission is written for {RESIDENT_DRAM_BANKS}")
    live_free = int(live["free_bytes_per_bank"])
    live_largest = int(live["largest_contiguous_bytes_free_per_bank"])
    if live_free < 0 or live_largest < 0 or live_largest > live_free:
        raise ValueError(f"inconsistent live DRAM view: free {live_free}, largest contiguous {live_largest}")
    free = live_free - reserved_bytes_per_bank
    largest = max(live_largest - reserved_bytes_per_bank, 0)
    largest_bucket = max(int(b) for b in buckets)
    layout = vision_resident_layout(mesh_size=dies)
    weights = _per_bank(layout["device_total"], banks)
    peak = _per_bank(largest_bucket * PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE, banks)
    peak_with_margin = -(-peak * (100 + VISION_GROWTH_ESTIMATE_MARGIN_PERCENT) // 100)
    required = weights + peak_with_margin
    # The largest single buffers: the merger's fc1 weight (resident) and the largest bucket's fused qkv activation.
    largest_weight = 4608 * 4608 * 2
    largest_activation = largest_bucket * QKV_ACTIVATION_BYTES_PER_ROW_PER_DIE
    contiguous = _per_bank(max(largest_weight, largest_activation), banks)
    shortfalls = [
        name
        for name, short in (
            ("free_bytes_below_estimate", free < required),
            ("largest_contiguous_below_largest_buffer", largest < contiguous),
        )
        if short
    ]
    return {
        "dies": dies,
        "num_banks": banks,
        "buckets": [int(b) for b in sorted(set(buckets))],
        "largest_bucket_rows": largest_bucket,
        "free_bytes_source": "measured_live",
        "free_bytes_per_bank": live_free,
        "largest_contiguous_bytes_free_per_bank": live_largest,
        "reserved_bytes_per_bank": reserved_bytes_per_bank,
        "resident_weights_bytes_per_bank": weights,
        "resident_weights_bytes_per_die_modeled": layout["device_total"],
        "peak_activation_bytes_per_row_per_die": PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE,
        "peak_activation_bytes_per_bank_largest_bucket": peak,
        "growth_estimate_margin_percent": VISION_GROWTH_ESTIMATE_MARGIN_PERCENT,
        "required_free_bytes_per_bank": required,
        "required_largest_contiguous_bytes_per_bank": contiguous,
        "headroom_bytes_per_bank": free - required,
        "decided_by": {
            "free_side": "measured_live",
            "required_side": f"resident weights ({dies} dies) + peak of {largest_bucket} rows with margin",
            "shortfalls": shortfalls,
        },
        "fits": not shortfalls,
    }


def compose_warm_hooks(*hooks: Callable[[Any], None] | None) -> Callable[[Any], None] | None:
    """One warm hook that runs the given hooks in order (the lanes' then the tower's); None when none is given."""

    live = [hook for hook in hooks if hook is not None]
    if not live:
        return None

    def hook(chain) -> None:
        for one in live:
            one(chain)

    return hook


class Qwen38VisionResidencyError(RuntimeError):
    """An image cannot be served by this process (the tower is not resident, or the image exceeds the largest bucket);
    the message is the reason a request is refused with."""


def shortfall_reason(admission: Mapping[str, Any]) -> str | None:
    """The one-line reason an image request is refused with when the admission did not fit; None when it did."""

    if admission.get("fits", False):
        return None
    shortfalls = list(admission["decided_by"]["shortfalls"])
    parts = []
    if "free_bytes_below_estimate" in shortfalls:
        parts.append(
            f"{admission['required_free_bytes_per_bank'] - (admission['free_bytes_per_bank'] - admission['reserved_bytes_per_bank'])} "
            f"B per bank short of the {admission['required_free_bytes_per_bank']} the resident tower and a "
            f"{admission['largest_bucket_rows']}-row image need"
        )
    if "largest_contiguous_below_largest_buffer" in shortfalls:
        parts.append(
            f"the largest contiguous block {admission['largest_contiguous_bytes_free_per_bank']} B per bank is under the "
            f"{admission['required_largest_contiguous_bytes_per_bank']} the largest image buffer needs"
        )
    return "the vision tower is not resident in this process: " + "; ".join(parts)


@dataclass
class Qwen38VisionResidency:
    """The tower's life in one served process: admitted on the live allocator, loaded and prewarmed in the warm hook
    when it fits (text-only otherwise), resident until ``release_vision``; the record for READY and ``/health``.

    ``reserved_bytes_per_bank`` is what the chain allocates after the hook (its traces).
    """

    mesh_device: Any
    state_dict: Mapping[str, Any]
    dram_view: Callable[[], Mapping[str, Any]]
    reserved_bytes_per_bank: int = 0
    buckets: tuple[int, ...] = VISION_ROW_BUCKETS
    config: VisionTowerConfig = field(default_factory=VisionTowerConfig)
    admission: dict[str, Any] = field(default_factory=dict)
    tower: Any = None
    prewarm_records: list[dict[str, Any]] = field(default_factory=list)
    load_seconds: float | None = None
    dram_after_load: dict[str, Any] | None = None
    dram_after_prewarm: dict[str, Any] | None = None
    shortfall: str | None = None  # the image refusal's reason after a warm hook that did not fit; None otherwise

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.mesh_device.shape)
        if len(shape) != 2:
            raise ValueError(f"expected a two-dimensional mesh, got {shape}")
        self.dies = shape[1]
        self.buckets = tuple(sorted(set(int(b) for b in self.buckets)))
        if not self.buckets or any(b % 32 for b in self.buckets):
            raise ValueError(f"buckets must be tile multiples, got {self.buckets}")

    @property
    def tower_resident(self) -> bool:
        return self.tower is not None

    def rows_for(self, patches: int) -> int:
        """The bucket an image runs in (``bucket_rows`` over this residency's buckets)."""

        try:
            return bucket_rows(patches, self.buckets)
        except ValueError as error:
            raise Qwen38VisionResidencyError(str(error)) from error

    def refusal_reason(self, patches: int | None = None) -> str | None:
        """Why an image request would be refused now (None = it would run): the tower not resident (the admission's
        shortfall, or the hook not yet run) or the image above the largest bucket.  The server's HTTP 400 text."""

        if self.tower is None:
            return self.shortfall or "the vision tower is not resident in this process"
        if patches is not None:
            try:
                bucket_rows(patches, self.buckets)
            except ValueError as error:
                return str(error)
        return None

    def vision_admission(self) -> dict[str, Any]:
        """The admission on the live allocator now (the warm hook's first act)."""

        self.admission = vision_capacity_admission(
            dies=self.dies,
            live=self.dram_view(),
            reserved_bytes_per_bank=self.reserved_bytes_per_bank,
            buckets=self.buckets,
        )
        return self.admission

    def vision_warm_hook(self, chain) -> None:
        """For ``construct_chain(warm_hook=...)``: admit; on a shortfall keep the process text-only (the reason is kept
        for the image refusals); on a fit load the weights and run one forward per bucket so every served shape is
        compiled before the captures."""

        admission = self.vision_admission()
        if not admission["fits"]:
            self.shortfall = shortfall_reason(admission)
            return
        self.shortfall = None
        from models.demos.blackhole.qwen38_flash_next.ttnn.vision import VisionTower

        started = time.perf_counter()
        self.tower = VisionTower(self.mesh_device, self.state_dict, self.config)
        self.tower.load()
        self.load_seconds = time.perf_counter() - started
        self.dram_after_load = dict(self.dram_view())
        self.vision_prewarm()
        self.dram_after_prewarm = dict(self.dram_view())

    def vision_prewarm(self) -> list[dict[str, Any]]:
        """One forward per bucket, ascending: the compiled shape sets of the served process, with the measured cost."""

        if self.tower is None:
            raise Qwen38VisionResidencyError("prewarm needs the resident tower")
        self.prewarm_records = []
        for rows in self.buckets:
            started = time.perf_counter()
            timing = self.tower.prewarm_rows(rows)
            self.prewarm_records.append(
                {"rows": rows, "ready_seconds_measured": time.perf_counter() - started, "forward": timing}
            )
        return self.prewarm_records

    def run_image_in_bucket(self, pixel_patches, grid_thw, **kwargs):
        """One image in its bucket (the only entry the serving path takes)."""

        reason = self.refusal_reason(int(pixel_patches.shape[0]))
        if reason is not None:
            raise Qwen38VisionResidencyError(reason)
        rows = self.rows_for(int(pixel_patches.shape[0]))
        return self.tower.run_image(pixel_patches, grid_thw, rows=rows, **kwargs)

    def vision_summary(self) -> dict[str, Any]:
        """The READY / ``/health`` record."""

        return {
            "resident": self.tower_resident,
            "shortfall": self.shortfall,
            "shortfall_bytes_per_bank": (
                None
                if self.tower_resident or not self.admission
                else max(
                    0,
                    self.admission["required_free_bytes_per_bank"]
                    - (self.admission["free_bytes_per_bank"] - self.admission["reserved_bytes_per_bank"]),
                )
            ),
            "dies": self.dies,
            "buckets": list(self.buckets),
            "largest_bucket_rows": self.buckets[-1],
            "stock_maximum_patches": STOCK_MAXIMUM_PATCHES,
            "admission": dict(self.admission),
            "load_seconds": self.load_seconds,
            "prewarm": list(self.prewarm_records),
            "prewarm_plan_modeled": prewarm_plan(self.buckets),
            "ladders": ladder_comparison(),
            "dram_after_load": self.dram_after_load,
            "dram_after_prewarm": self.dram_after_prewarm,
            "resident_bytes_per_bank": None if self.tower is None else self.tower.resident_bytes_per_bank(),
        }

    def release_vision(self) -> None:
        if self.tower is not None:
            self.tower.free()
            self.tower = None
