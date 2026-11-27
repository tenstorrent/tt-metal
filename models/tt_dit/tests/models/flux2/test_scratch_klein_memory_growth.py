# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: what FLUX.2 klein compiles and keeps per image size, on the WH 2x4 or BH 2x2 preset.

The pipeline warms up at its default size only (no trace warmup). Then each case runs twice, eager:
text to image at KLEIN_GROWTH_SIZES sizes (default 20, half with a side below 1024) drawn with a fixed
seed from the target range (sides 832 to 2048, H a multiple of 64 and W of 32 so the VAE takes whole
token rows per chip, H*W a multiple of 32768 so ring SDPA does), edits with 1 to 4 reference images of
at most 1 MP, and two prompts at once. After each call it records the kernel binaries compiled (run
with an empty TT_METAL_CACHE), the program-cache entries, the host RSS and the DRAM allocated on chip 0,
also at the start and end of each pipeline section. The repeat of a case keeps nothing new, so the
first run minus the repeat is what the case kept. KLEIN_OUT_DIR gets calls.csv and sections.csv.
"""

import csv
import gc
import os
import random
from pathlib import Path

import pytest
from PIL import Image

import ttnn
from models.tt_dit.pipelines.events import SectionEnd, SectionStart

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ..scratch_op_hook import kernel_count
from .device_params import line_params_flux2_klein


def _rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no VmRSS")


def _dram(mesh_device: ttnn.MeshDevice) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _extreme_sizes(n: int) -> list[tuple[int, int]]:
    """(height, width) in the range (4,056 to 16,384 tokens, aspect 1:8 to 8:1) with a side below 832 or above
    2048, that the 2x4 VAE and ring SDPA take. Picks the most extreme ratio of each token count, both ways."""
    sides = range(64, 8193, 16)
    valid = [
        (h, w)
        for h in sides
        for w in sides
        if 4056 * 256 <= h * w <= 16384 * 256
        and max(h, w) <= 8 * min(h, w)
        and (min(h, w) < 832 or max(h, w) > 2048)
        and (h * w) % 32768 == 0
        and h % 64 == 0
        and w % 32 == 0
    ]
    rng = random.Random(0)
    extreme = sorted(valid, key=lambda s: -max(s) / min(s))[: n // 2]
    rest = [s for s in valid if s not in extreme]
    return extreme + rng.sample(rest, n - len(extreme))


def _sizes(n: int) -> list[tuple[int, int]]:
    """(height, width): n // 2 with a side below 1024, the rest with both sides 1024 or more."""
    sides = range(832, 2049, 16)
    valid = [
        (h, w)
        for h in sides
        for w in sides
        if (h * w) % 32768 == 0 and h % 64 == 0 and w % 32 == 0 and (h, w) != (1024, 1024)
    ]
    small = [s for s in valid if min(s) < 1024]
    large = [s for s in valid if min(s) >= 1024]
    rng = random.Random(0)
    return rng.sample(small, n // 2) + rng.sample(large, n - n // 2)


# Reference image sizes (width, height), each at most 1 MP.
_REFS = [(1024, 1024), (768, 1024), (1024, 768), (640, 960), (832, 1216)]


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_memory_growth(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_root = Path(os.environ["TT_METAL_CACHE"])
    n = int(os.environ.get("KLEIN_GROWTH_SIZES", "20"))
    sizes = _extreme_sizes(n) if os.environ.get("KLEIN_GROWTH_SET") == "extreme" else _sizes(n)

    cases: list[tuple[str, int, int, list[str], list[tuple[int, int]]]] = []
    for h, w in sizes:
        cases.append((f"t2i {h}x{w}", h, w, ["a lighthouse on a cliff, golden hour"], []))
    for i, (h, w) in enumerate([(1024, 1024), (1024, 1184), (1536, 1536), (1024, 1024)]):
        refs = [_REFS[(i + j) % len(_REFS)] for j in range(i + 1)]  # 1, 2, 3, 4 references
        cases.append((f"edit {h}x{w} refs={len(refs)}", h, w, ["make it a winter scene"], refs))
    for h, w in [(1024, 1024), sizes[-1]]:
        cases.append((f"2 prompts {h}x{w}", h, w, ["a lighthouse on a cliff", "a red fox in the snow"], []))

    resolutions = sorted({(1024, 1024), (2048, 2048), *((w, h) for _, h, w, _, _ in cases)})
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        resolutions=resolutions,
        trace_warmup=False,
    )

    calls: list[dict] = []
    sections: list[dict] = []
    current: dict = {}

    def on_event(event: object) -> None:
        if isinstance(event, (SectionStart, SectionEnd)):
            ttnn.synchronize_device(mesh_device)
            sections.append(
                {
                    "call": current["call"],
                    "case": current["case"],
                    "run": current["run"],
                    "section": event.name,
                    "edge": "start" if isinstance(event, SectionStart) else "end",
                    "dram_mb": round(_dram(mesh_device) / 2**20, 2),
                    "entries": mesh_device.num_program_cache_entries(),
                }
            )

    def snapshot() -> dict:
        ttnn.synchronize_device(mesh_device)
        gc.collect()
        return {
            "rss_mb": round(_rss_bytes() / 2**20, 1),
            "dram_mb": round(_dram(mesh_device) / 2**20, 2),
            "entries": mesh_device.num_program_cache_entries(),
            "kernels": kernel_count(kernel_root),
        }

    def write() -> None:
        for name, rows in (("calls.csv", calls), ("sections.csv", sections)):
            if rows:
                with open(out_dir / name, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    before = snapshot()
    plan = [("warm 1024x1024", 1024, 1024, ["warm"], [])] + cases
    for case, h, w, prompts, refs in plan:
        for run in ("first", "repeat"):
            current.update(call=len(calls), case=case, run=run)
            images = [Image.new("RGB", size, (90 + 20 * i, 120, 150)) for i, size in enumerate(refs)] or None
            status = "ok"
            try:
                pipeline(
                    prompts=prompts,
                    num_inference_steps=2,
                    seed=0,
                    width=w,
                    height=h,
                    images=images,
                    traced=False,
                    on_event=on_event,
                )
            except Exception as e:  # noqa: BLE001 - record and go on to the next case
                status = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
            after = snapshot()
            calls.append(
                {
                    "call": current["call"],
                    "case": case,
                    "run": run,
                    "status": status,
                    **{f"d_{k}": round(after[k] - before[k], 2) for k in after},
                    **after,
                }
            )
            print(f"\nGROWTH {calls[-1]}", flush=True)
            before = after
            write()
