# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: what FLUX.2 klein still compiles and builds at a new image size.

The pipeline warms up at its default size only (no trace warmup). Each size in KLEIN_SIZES (HxW,
comma-separated, default 1024x1184) then runs twice, eager (with KLEIN_EDIT=1 as an edit of a reference image), and every ttnn op is recorded with the
program-cache entries and kernel binaries it adds and its wall time. Per op, the first run minus the
second is what the new size costs. Run with
TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' (so the op hooks fire) and an empty
TT_METAL_CACHE (so new kernel binaries can be counted). KLEIN_OUT_DIR gets ops.csv and a summary.
"""

import os
import time
from collections import defaultdict
from pathlib import Path

import pytest

import ttnn

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ..scratch_op_hook import OpRecorder
from .device_params import line_params_flux2_klein


def _sizes() -> list[tuple[int, int]]:
    spec = os.environ.get("KLEIN_SIZES", "1024x1184")
    return [tuple(int(v) for v in s.split("x")) for s in spec.split(",")]  # (height, width)


def _summary(rows: list[dict], first: str, second: str) -> list[str]:
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["depth"] != 0 or r["run"] not in (first, second):
            continue
        a = acc[r["op"]]
        if r["run"] == first:
            a["calls"] += 1
            a["kernels"] += r["kernels"]
            a["entries"] += r["entries"]
            a["first"] += r["seconds"]
        else:
            a["second"] += r["seconds"]
    ops = sorted(acc.items(), key=lambda kv: (kv[1]["kernels"] > 0, kv[1]["first"] - kv[1]["second"]), reverse=True)
    total_k = sum(a["kernels"] for _, a in ops)
    total_e = sum(a["entries"] for _, a in ops)
    total_s = sum(a["first"] - a["second"] for _, a in ops)
    lines = [
        f"{first}: {total_k:.0f} kernels, {total_e:.0f} new cache entries, {total_s:.2f} s over the second run",
        f"{'op':60} {'calls':>5} {'kernels':>7} {'entries':>7} {'extra s':>8}",
    ]
    # The ops whose first run took the most longer than the second, whatever they built.
    slowest = sorted(ops, key=lambda kv: kv[1]["first"] - kv[1]["second"], reverse=True)[:12]
    lines.append(
        "slowest first runs: "
        + "; ".join(f"{op} {a['calls']:.0f}x +{a['first'] - a['second']:.3f} s" for op, a in slowest)
    )
    for op, a in ops:
        if a["kernels"] or a["entries"]:
            lines.append(
                f"{op:60} {a['calls']:5.0f} {a['kernels']:7.0f} {a['entries']:7.0f} {a['first'] - a['second']:8.3f}"
            )
    return lines


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_new_size(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_root = Path(os.environ["TT_METAL_CACHE"])
    sizes = _sizes()

    # The target range's largest size and the default, so the row capacity is the one the server uses.
    resolutions = sorted({(1024, 1024), (2048, 2048), *((w, h) for h, w in sizes)})
    if os.environ.get("KLEIN_ANY") == "1":  # sizes outside the resolutions
        resolutions = [(1024, 1024), (2048, 2048)]
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        resolutions=resolutions,
        trace_warmup=False,
    )

    recorder = OpRecorder([mesh_device], out_dir / "ops.csv", kernel_root)
    conv_log = _log_convs(mesh_device, recorder, kernel_root, out_dir / "convs.csv")
    summary: list[str] = []
    # KLEIN_EDIT=1: every request edits a reference image, whose encoded size follows the output's.
    reference = None
    if os.environ.get("KLEIN_EDIT") == "1":
        from PIL import Image

        reference = Image.effect_noise((1536, 1152), 64).convert("RGB")
    for h, w in sizes:
        runs = [f"{h}x{w} first", f"{h}x{w} second"]
        walls = []
        for run in runs:
            t0 = time.perf_counter()
            with recorder.recording(run):
                pipeline(
                    prompts=["a lighthouse on a cliff, golden hour"],
                    images=[reference] if reference is not None else None,
                    num_inference_steps=2,
                    seed=0,
                    width=w,
                    height=h,
                    traced=False,
                    on_event=recorder.on_event,
                )
            walls.append(time.perf_counter() - t0)
        summary.append(f"{h}x{w} wall: first {walls[0]:.2f} s, second {walls[1]:.2f} s")
        for sec in sorted({r["section"] for r in recorder.rows if r["run"] in runs}):
            t = [
                sum(r["seconds"] for r in recorder.rows if r["run"] == run and r["section"] == sec and r["depth"] == 0)
                for run in runs
            ]
            summary.append(f"  ops in {sec}: first {t[0]:.3f} s, second {t[1]:.3f} s")
        summary += _summary(recorder.rows, *runs) + [""]
        (out_dir / "summary.txt").write_text("\n".join(summary))
        conv_log.flush()
    print("\n".join(summary))


def _log_convs(mesh_device: ttnn.MeshDevice, recorder: OpRecorder, kernel_root: Path, path: Path):
    """Patch Conv2d.forward and ttnn.prepare_conv_weights to log, per conv call, its input, whether the
    prepared weights were reused, and the time, cache entries and kernels of the preparation and of
    the whole call."""
    import csv
    import time

    from ....layers import conv2d as conv_mod
    from ..scratch_op_hook import kernel_count

    rows: list[dict] = []
    state = {"prep_s": 0.0, "prep_entries": 0, "prepared": False}
    entries = lambda: mesh_device.num_program_cache_entries()  # noqa: E731

    orig_prepare = ttnn.prepare_conv_weights
    orig_forward = conv_mod.Conv2d.forward

    def prepare(*args, **kwargs):
        ttnn.synchronize_device(mesh_device)
        e, t = entries(), time.perf_counter()
        out = orig_prepare(*args, **kwargs)
        ttnn.synchronize_device(mesh_device)
        state["prep_s"] += time.perf_counter() - t
        state["prep_entries"] += entries() - e
        state["prepared"] = True
        return out

    def forward(self, x, /, **kwargs):
        ttnn.synchronize_device(mesh_device)
        state.update(prep_s=0.0, prep_entries=0, prepared=False)
        e, k, t = entries(), kernel_count(kernel_root), time.perf_counter()
        out = orig_forward(self, x, **kwargs)
        ttnn.synchronize_device(mesh_device)
        rows.append(
            {
                "run": recorder.run,
                "in": "x".join(str(v) for v in x.shape),
                "out_channels": self.out_channels,
                "kernel": self.kernel_size[0],
                "input": f"{x.layout} {x.memory_config().memory_layout}",
                "shard_layout": self.shard_layout,
                "prepared_now": state["prepared"],
                "prep_s": round(state["prep_s"], 4),
                "prep_entries": state["prep_entries"],
                "seconds": round(time.perf_counter() - t, 4),
                "entries": entries() - e,
                "kernels": kernel_count(kernel_root) - k,
            }
        )
        return out

    ttnn.prepare_conv_weights = prepare
    conv_mod.ttnn.prepare_conv_weights = prepare
    conv_mod.Conv2d.forward = forward

    class _Log:
        def flush(self) -> None:
            if rows:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)

    return _Log()
