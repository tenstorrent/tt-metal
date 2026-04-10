#!/usr/bin/env python3
"""
Per-chip YOLO on **1×1 submeshes** of one parent mesh (single process, no subprocesses).

Opens a full mesh (e.g. Galaxy 8×4), calls ``mesh_device.create_submeshes(MeshShape(1, 1))`` to
obtain one ``MeshDevice`` view per physical chip, then constructs a
``YOLOv8sPerformantRunner`` / ``YOLOv8xPerformantRunner`` with ``device_batch_size=1`` on each
submesh, runs the input image, and times setup + device + post per slot.

Modes:

  **Sequential** (default)
    Slots run one after another. Safest; useful for per-chip bring-up.

  **--parallel**
    All slots run concurrently via ``ThreadPoolExecutor``. Each 1×1 submesh is a separate physical
    chip with its own CQs, so runner construction (compile/trace) and ``runner.run`` happen in
    parallel. Host thread work (PyTorch weight prep, ``to_torch``, postprocess) contends on GIL, but
    device ops release it so device work overlaps. This is a middle ground between ``--mesh-dp``
    (one sharded forward, fastest) and ``yolo_dp_32_independent_infer.py`` (32 processes, slowest).

  **--mesh-dp**
    True data parallel: one runner on the full parent mesh, ``batch_size = device count``,
    ``ShardTensorToMesh``. Delegates to ``yolo_dp_mesh_infer.run_tt``.

Input:

  ``--input FILE``  — Single image; same image on every submesh slot.
  ``--input-dir DIR`` — Directory of images; each slot gets a different image (sorted by name).
                        Requires at least as many images as slots. Overrides ``--input``.

**Lifecycle:** Submeshes share the parent's command queues; do **not** close submesh handles
explicitly — only ``runner.release()`` per slot, then ``ttnn.close_mesh_device(parent)`` at the end.

Run from the tt-metal repo root:

  python models/demos/yolo_eval/yolo_dp_submesh_infer.py \\
    --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \\
    --tt-mesh-shape 8 4 --tt-model yolov8s

Parallel (all 32 submeshes at once):

  python models/demos/yolo_eval/yolo_dp_submesh_infer.py --parallel \\
    --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \\
    --tt-mesh-shape 8 4 --tt-model yolov8s

Parallel with 32 different images:

  python models/demos/yolo_eval/yolo_dp_submesh_infer.py --parallel \\
    --input-dir models/demos/yolo_eval/sample_images/batch32/ \\
    --tt-mesh-shape 8 4 --tt-model yolov8s

Full-mesh data parallel (same as ``yolo_dp_mesh_infer.py``; no submesh loop):

  python models/demos/yolo_eval/yolo_dp_submesh_infer.py --mesh-dp \\
    --input models/demos/yolo_eval/sample_images/crowded_freeway.jpg \\
    --tt-mesh-shape 8 4 --tt-model yolov8s
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __name__ == "__main__" and "--tt-eth-dispatch" in sys.argv:
    os.environ.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_TT_INPUT_RES = (640, 640)
_DEFAULT_GALAXY_MESH_ROWS = 8
_DEFAULT_GALAXY_MESH_COLS = 4
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _fmt_s(sec: float) -> str:
    return f"{float(sec):.4f}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "YOLO on 1×1 submeshes of a parent mesh (sequential or --parallel), "
            "or --mesh-dp for one sharded DP forward."
        ),
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", type=Path, help="Single image file (replicated to all slots).")
    inp.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of images. Sorted by name; slot i gets image i. Need >= K images.",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Run all submesh slots concurrently (ThreadPoolExecutor). Each 1×1 submesh is a "
            "different physical chip; device work overlaps, host work contends on GIL."
        ),
    )
    p.add_argument(
        "--mesh-dp",
        action="store_true",
        help=(
            "Data parallel on the full parent mesh: one runner, global batch = device count, "
            "ShardTensorToMesh. Delegates to yolo_dp_mesh_infer.run_tt."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Annotated JPEGs when --save-images. Default: {_SCRIPT_DIR}/sample_images_output/submesh_{{stem}}.",
    )
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--tt-model", choices=["yolov8s", "yolov8x"], default="yolov8s")
    p.add_argument(
        "--tt-mesh-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=None,
        help=(
            f"Parent mesh shape. Default: {_DEFAULT_GALAXY_MESH_ROWS} {_DEFAULT_GALAXY_MESH_COLS} "
            "unless --tt-use-system-mesh."
        ),
    )
    p.add_argument(
        "--tt-use-system-mesh",
        action="store_true",
        help="Parent mesh shape from SystemMeshDescriptor(); ignores --tt-mesh-shape.",
    )
    p.add_argument(
        "--max-slots",
        type=int,
        default=None,
        metavar="K",
        help="Only use the first K submesh slots. Default: all chips in parent mesh.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        metavar="W",
        help="Thread pool size for --parallel. Default: K (all slots at once).",
    )
    p.add_argument(
        "--tt-warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help="Untimed runner.run()+sync per slot before measured iters. Default: 0.",
    )
    p.add_argument(
        "--tt-measured-iters",
        type=int,
        default=1,
        metavar="N",
        help="Timed runner.run()+sync intervals per slot; min/mean/max when N>1. Default: 1.",
    )
    p.add_argument("--tt-l1-small-size", type=int, default=24576)
    p.add_argument("--tt-trace-region-size", type=int, default=6434816)
    p.add_argument("--tt-row-dispatch", action="store_true", help="DispatchCoreConfig(axis=ROW) for parent mesh open.")
    p.add_argument(
        "--tt-eth-dispatch", action="store_true", help="Set TT_METAL_GTEST_ETH_DISPATCH=1 before device init."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_parent_mesh_shape(ttnn, args: argparse.Namespace) -> tuple[int, int, str]:
    if args.tt_use_system_mesh:
        sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
        if len(sys_shape) != 2:
            raise RuntimeError(f"Unexpected system mesh shape tuple: {sys_shape}")
        rows, cols = int(sys_shape[0]), int(sys_shape[1])
        if rows * cols <= 1:
            raise RuntimeError("--tt-use-system-mesh needs a multi-device system (descriptor reports <=1 device).")
        return rows, cols, f"SystemMeshDescriptor() {rows}x{cols}"
    if args.tt_mesh_shape is None:
        args.tt_mesh_shape = [_DEFAULT_GALAXY_MESH_ROWS, _DEFAULT_GALAXY_MESH_COLS]
    rows, cols = int(args.tt_mesh_shape[0]), int(args.tt_mesh_shape[1])
    n = rows * cols
    if n <= 1:
        raise ValueError("Parent mesh must have more than one device for submesh sweep.")
    return rows, cols, f"--tt-mesh-shape {rows}x{cols}"


def _namespace_for_run_tt(src: argparse.Namespace, batch_size: int) -> argparse.Namespace:
    """Minimal Namespace for yolo_dp_mesh_infer.run_tt (TT YOLO only)."""
    mesh_shape = src.tt_mesh_shape
    if mesh_shape is not None:
        mesh_shape = [int(mesh_shape[0]), int(mesh_shape[1])]
    return argparse.Namespace(
        tt_use_system_mesh=bool(src.tt_use_system_mesh),
        tt_mesh_shape=mesh_shape,
        tt_force_single_device=False,
        tt_device_id=0,
        batch_size=int(batch_size),
        tt_model=src.tt_model,
        tt_warmup_iters=int(src.tt_warmup_iters),
        tt_measured_iters=int(src.tt_measured_iters),
        save_images=bool(src.save_images),
        tt_l1_small_size=int(src.tt_l1_small_size),
        tt_trace_region_size=int(src.tt_trace_region_size),
        tt_row_dispatch=bool(src.tt_row_dispatch),
    )


def _collect_images(args: argparse.Namespace, k: int):
    """Return (images_bgr: list[ndarray], stems: list[str], read_sec: float)."""
    import cv2

    t0 = time.perf_counter()
    if args.input_dir is not None:
        d = args.input_dir.resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"--input-dir not a directory: {d}")
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if len(paths) < k:
            raise ValueError(
                f"--input-dir has {len(paths)} images, but need >= {k} (one per slot). "
                f"Provide more images or reduce --max-slots."
            )
        paths = paths[:k]
        images = []
        stems = []
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"Failed to read image: {p}")
            images.append(bgr)
            stems.append(p.stem)
        return images, stems, time.perf_counter() - t0

    inp = args.input.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")
    bgr = cv2.imread(str(inp))
    if bgr is None:
        raise ValueError(f"Failed to read image: {inp}")
    return [bgr] * k, [inp.stem] * k, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-slot runner
# ---------------------------------------------------------------------------


def _run_one_slot(
    *,
    slot: int,
    submesh,
    im,
    image_bgr,
    stem: str,
    out_dir: Path | None,
    save_images: bool,
    args: argparse.Namespace,
) -> dict:
    """Build runner on submesh, warmup/measured runs, postprocess; returns timing dict."""
    import torch

    import ttnn
    from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess
    from models.demos.yolo_eval.yolo_dp_mesh_infer import save_yolo_result_bgr
    from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
    from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner

    wall0 = time.perf_counter()
    t_setup0 = time.perf_counter()
    if args.tt_model == "yolov8s":
        runner = YOLOv8sPerformantRunner(
            submesh,
            device_batch_size=1,
            mesh_mapper=None,
            mesh_composer=None,
            weights_mesh_mapper=None,
        )
    else:
        runner = YOLOv8xPerformantRunner(
            submesh,
            device_batch_size=1,
            inputs_mesh_mapper=None,
            weights_mesh_mapper=None,
            outputs_mesh_composer=None,
        )
    setup_runner_sec = time.perf_counter() - t_setup0

    names = load_coco_class_names()
    orig_imgs = [image_bgr]
    paths = ([f"{stem}#submesh{slot}"],)

    # Split timing: prepare_host_input once (host tensor prep), then separate
    # H2D (push_host_input_to_device_dram) and compute (execute_reshard_and_trace)
    # in the measured loop — same approach as test_e2e_performant split_host_device_timing.
    use_split = args.tt_model == "yolov8s"
    t_prep0 = time.perf_counter()
    if use_split:
        tt_host = runner.prepare_host_input(im)
    t_prep1 = time.perf_counter()
    host_prep_sec = t_prep1 - t_prep0

    if use_split:
        for _ in range(int(args.tt_warmup_iters)):
            runner.push_host_input_to_device_dram(tt_host)
            _wp = runner.execute_reshard_and_trace()
            del _wp
    else:
        for _ in range(int(args.tt_warmup_iters)):
            preds_w = runner.run(im)
            ttnn.synchronize_device(submesh)
            del preds_w

    n_meas = int(args.tt_measured_iters)
    device_secs: list[float] = []
    h2d_secs: list[float] = []
    compute_secs: list[float] = []
    preds = None
    if use_split:
        for _ in range(n_meas):
            t_h0 = time.perf_counter()
            runner.push_host_input_to_device_dram(tt_host)
            t_h1 = time.perf_counter()
            preds = runner.execute_reshard_and_trace()
            t_c1 = time.perf_counter()
            h2d_secs.append(t_h1 - t_h0)
            compute_secs.append(t_c1 - t_h1)
            device_secs.append(t_c1 - t_h0)
    else:
        for _ in range(n_meas):
            t0 = time.perf_counter()
            preds = runner.run(im)
            ttnn.synchronize_device(submesh)
            device_secs.append(time.perf_counter() - t0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    t_torch0 = time.perf_counter()
    preds_t = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=None)
    t_torch1 = time.perf_counter()
    t_pp0 = time.perf_counter()
    results = postprocess(preds_t, im, orig_imgs, paths, names)
    t_pp1 = time.perf_counter()
    t_sv0 = time.perf_counter()
    if save_images and out_dir is not None:
        save_yolo_result_bgr(
            results[0],
            out_dir / f"{stem}_submesh_{slot:02d}.jpg",
            tt_style=True,
        )
    t_sv1 = time.perf_counter()

    try:
        runner.release()
    except Exception:
        pass

    device_sec = sum(device_secs) / len(device_secs)
    payload = {
        "slot": slot,
        "ok": True,
        "setup_runner_sec": setup_runner_sec,
        "host_prep_sec": host_prep_sec,
        "device_sec": device_sec,
        "device_sec_min": min(device_secs) if len(device_secs) > 1 else device_sec,
        "device_sec_max": max(device_secs) if len(device_secs) > 1 else device_sec,
        "to_torch_sec": t_torch1 - t_torch0,
        "postprocess_sec": t_pp1 - t_pp0,
        "save_sec": (t_sv1 - t_sv0) if save_images else 0.0,
        "wall_slot_sec": time.perf_counter() - wall0,
        "device_ids": list(submesh.get_device_ids()) if hasattr(submesh, "get_device_ids") else [],
    }
    if h2d_secs:
        payload["h2d_sec"] = sum(h2d_secs) / len(h2d_secs)
        payload["h2d_sec_min"] = min(h2d_secs)
        payload["h2d_sec_max"] = max(h2d_secs)
    if compute_secs:
        payload["compute_sec"] = sum(compute_secs) / len(compute_secs)
        payload["compute_sec_min"] = min(compute_secs)
        payload["compute_sec_max"] = max(compute_secs)
    return payload


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_submesh_summary(
    oks: list[dict],
    bads: list[dict],
    k: int,
    parent_open_sec: float,
    submesh_create_sec: float,
    pre_once_sec: float,
    orchestration_wall: float,
    args: argparse.Namespace,
    out_dir: Path,
    parallel: bool,
) -> None:
    from models.demos.yolo_eval.yolo_dp_mesh_infer import print_run_summary

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in oks) / len(oks)

    post_wall_mean = _mean("to_torch_sec") + _mean("postprocess_sec") + _mean("save_sec")
    mode_label = "parallel" if parallel else "sequential"

    has_split = any("h2d_sec" in r for r in oks)
    timing = {
        "setup_sec": _mean("setup_runner_sec"),
        "pre_sec": pre_once_sec,
        "device_sec": _mean("device_sec"),
        "post_sec": post_wall_mean,
        "total_sec": orchestration_wall,
        "to_torch_sec": _mean("to_torch_sec"),
        "postprocess_sec": _mean("postprocess_sec"),
        "save_sec": _mean("save_sec"),
        "tt_device_label": f"device(push_dram+reshard+trace) mean@slot ({mode_label})"
        if has_split
        else f"device(run+sync) mean@slot ({mode_label})",
        "tt_pre_label": "pre(read+letterbox, shared)" if not args.input_dir else "pre(read+letterbox, per-slot)",
        "tt_setup_label": f"setup(runner on 1x1 submesh) mean@slot ({mode_label})",
        "device_measured_iters": int(args.tt_measured_iters),
    }
    if has_split:
        timing["host_prep_sec"] = _mean("host_prep_sec")
        timing["h2d_sec"] = _mean("h2d_sec")
        timing["compute_sec"] = _mean("compute_sec")
    if int(args.tt_measured_iters) > 1:
        timing["device_sec_min"] = min(float(r["device_sec_min"]) for r in oks)
        timing["device_sec_max"] = max(float(r["device_sec_max"]) for r in oks)
        if has_split:
            timing["h2d_sec_min"] = min(float(r["h2d_sec_min"]) for r in oks)
            timing["h2d_sec_max"] = max(float(r["h2d_sec_max"]) for r in oks)
            timing["compute_sec_min"] = min(float(r["compute_sec_min"]) for r in oks)
            timing["compute_sec_max"] = max(float(r["compute_sec_max"]) for r in oks)

    max_par = int(args.max_parallel) if args.max_parallel else k
    print(
        f"\nOrchestration ({mode_label}): wall(parent open → done, {k} slots"
        f"{f', max_parallel={max_par}' if parallel else ''})={orchestration_wall:.4f}s  "
        f"pre(once)={pre_once_sec:.4f}s\n"
        f"  parent_open+create_submeshes={parent_open_sec:.4f}s  "
        f"(create_submeshes only={submesh_create_sec:.4f}s)\n"
        f"  mean runner setup per slot={_mean('setup_runner_sec'):.4f}s  "
        f"mean slot wall={_mean('wall_slot_sec'):.4f}s\n"
        f"  per-slot device spread (mean@slot):  "
        f"min={min(r['device_sec'] for r in oks):.4f}s  mean={timing['device_sec']:.4f}s  "
        f"max={max(r['device_sec'] for r in oks):.4f}s\n"
    )
    if has_split:
        print(
            f"  Split timing (mean@slot):  "
            f"host_prep(once)={timing['host_prep_sec']:.6f}s  "
            f"h2d(push_dram)={timing['h2d_sec']:.6f}s  "
            f"compute(reshard+trace+sync)={timing['compute_sec']:.6f}s\n"
        )
    if int(args.tt_measured_iters) > 1:
        print(
            f"  over {args.tt_measured_iters} timed iters/slot:  "
            f"min-of-mins={timing['device_sec_min']:.4f}s  "
            f"max-of-maxes={timing['device_sec_max']:.4f}s"
        )
        if has_split:
            print(
                f"  h2d: min={timing['h2d_sec_min']:.6f}s  max={timing['h2d_sec_max']:.6f}s  "
                f"compute: min={timing['compute_sec_min']:.6f}s  max={timing['compute_sec_max']:.6f}s"
            )
        print()

    if bads:
        print(f"Failed slots ({len(bads)}):")
        for r in bads:
            print(f"  slot {r.get('slot')}: {r.get('error', r)}")
        print()

    print_run_summary(out_dir, "tt", timing, args.save_images)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if int(args.tt_warmup_iters) < 0:
        raise ValueError("--tt-warmup-iters must be >= 0")
    if int(args.tt_measured_iters) < 1:
        raise ValueError("--tt-measured-iters must be >= 1")

    import cv2

    import ttnn
    from models.demos.utils.common_demo_utils import preprocess
    from models.demos.yolo_eval.yolo_dp_mesh_infer import open_tt_mesh, print_run_summary

    rows, cols, mesh_src = _resolve_parent_mesh_shape(ttnn, args)
    num_slots = rows * cols

    # ---- --mesh-dp: delegate to full-mesh DP (no submeshes) ----
    if args.mesh_dp:
        if args.input_dir is not None:
            raise ValueError("--mesh-dp requires --input (single file), not --input-dir.")
        inp = args.input.resolve()
        if not inp.is_file():
            raise FileNotFoundError(f"Input not found: {inp}")
        if args.max_slots is not None and int(args.max_slots) != num_slots:
            raise ValueError(
                f"--mesh-dp runs the full parent mesh ({num_slots} devices). "
                f"Unset --max-slots or set --max-slots {num_slots}."
            )
        image_bgr = cv2.imread(str(inp))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {inp}")
        mesh_args = _namespace_for_run_tt(args, num_slots)
        from models.demos.yolo_eval.yolo_dp_mesh_infer import run_tt

        out_dir = args.output_dir
        if out_dir is None:
            out_dir = _SCRIPT_DIR / "sample_images_output" / f"dp{num_slots}_glx"
        out_dir = out_dir.resolve()
        if args.save_images:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Mode: --mesh-dp  ({mesh_src})  devices={num_slots}  → yolo_dp_mesh_infer.run_tt\n")
        timing = run_tt(mesh_args, image_bgr, out_dir, inp.stem)
        print_run_summary(out_dir, "tt", timing, args.save_images)
        print("Done.")
        return

    # ---- Submesh path (sequential or --parallel) ----
    k = int(args.max_slots) if args.max_slots is not None else num_slots
    if k < 1 or k > num_slots:
        raise ValueError(f"--max-slots must be in [1, {num_slots}] for this mesh.")
    max_par = int(args.max_parallel) if args.max_parallel else k
    if max_par < 1:
        raise ValueError("--max-parallel must be >= 1")

    images_bgr, stems, read_sec = _collect_images(args, k)
    using_dir = args.input_dir is not None
    input_desc = str(args.input_dir.resolve()) if using_dir else str(args.input.resolve())

    t_pre0 = time.perf_counter()
    ims = [preprocess([img], res=_TT_INPUT_RES) for img in images_bgr]
    t_pre1 = time.perf_counter()
    letterbox_sec = t_pre1 - t_pre0
    pre_once_sec = read_sec + letterbox_sec

    args.tt_force_single_device = False
    args.tt_device_id = 0

    out_dir = args.output_dir
    if out_dir is None:
        tag = stems[0] if not using_dir else "dir"
        out_dir = _SCRIPT_DIR / "sample_images_output" / f"submesh_{tag}"
    out_dir = out_dir.resolve()
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    wall0 = time.perf_counter()
    t_open0 = time.perf_counter()
    parent, is_mesh = open_tt_mesh(ttnn, args, rows, cols)
    if not is_mesh:
        raise RuntimeError("internal: expected mesh device from open_tt_mesh")
    try:
        if parent.get_num_devices() != num_slots:
            raise RuntimeError(
                f"Parent mesh reports {parent.get_num_devices()} devices; "
                f"expected {num_slots} from shape {rows}x{cols}."
            )

        t_sub0 = time.perf_counter()
        submeshes = parent.create_submeshes(ttnn.MeshShape(1, 1))
        t_sub1 = time.perf_counter()
        if len(submeshes) != num_slots:
            raise RuntimeError(f"create_submeshes(1x1) returned {len(submeshes)} views; expected {num_slots}.")

        parent_open_sec = t_sub1 - t_open0
        submesh_create_sec = t_sub1 - t_sub0

        mesh_shape_list = None
        try:
            mesh_shape_list = list(parent.shape)
        except Exception:
            pass

        mode_str = "PARALLEL" if args.parallel else "sequential"
        print(
            f"Submesh sweep ({mode_str}): parent {mesh_src}  shape={mesh_shape_list!r}  "
            f"slots={num_slots}  running_first={k}"
            f"{'  max_parallel=' + str(max_par) if args.parallel else ''}\n"
            f"  input={'dir ' + input_desc if using_dir else input_desc}  "
            f"{'(' + str(k) + ' images)' if using_dir else '(same image, all slots)'}  "
            f"tt_model={args.tt_model}\n"
            f"  parent_open+submeshes={_fmt_s(parent_open_sec)}  "
            f"(create_submeshes={_fmt_s(submesh_create_sec)})\n"
            f"  read+letterbox={_fmt_s(pre_once_sec)}  "
            f"tt_warmup_iters={args.tt_warmup_iters}  tt_measured_iters={args.tt_measured_iters}  "
            f"save_images={args.save_images}\n"
            f"  output_dir={out_dir}\n"
        )

        oks: list[dict] = []
        bads: list[dict] = []

        def _slot_fn(slot_idx: int) -> dict:
            sub = submeshes[slot_idx]
            ids = list(sub.get_device_ids()) if hasattr(sub, "get_device_ids") else []
            print(f"  slot {slot_idx:02d}/{k}  device_ids={ids}  start", flush=True)
            try:
                result = _run_one_slot(
                    slot=slot_idx,
                    submesh=sub,
                    im=ims[slot_idx],
                    image_bgr=images_bgr[slot_idx],
                    stem=stems[slot_idx],
                    out_dir=out_dir,
                    save_images=args.save_images,
                    args=args,
                )
                print(
                    f"  slot {slot_idx:02d}/{k}  device_ids={ids}  "
                    f"done  setup={_fmt_s(result['setup_runner_sec'])}  "
                    f"device={_fmt_s(result['device_sec'])}  "
                    f"wall={_fmt_s(result['wall_slot_sec'])}",
                    flush=True,
                )
                return result
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"  slot {slot_idx:02d}/{k}  device_ids={ids}  FAILED: {exc}", flush=True)
                return {"slot": slot_idx, "ok": False, "error": f"{exc}\n{tb}"}

        if args.parallel:
            with ThreadPoolExecutor(max_workers=max_par) as pool:
                futures = {pool.submit(_slot_fn, i): i for i in range(k)}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r.get("ok"):
                        oks.append(r)
                    else:
                        bads.append(r)
        else:
            for slot in range(k):
                r = _slot_fn(slot)
                if r.get("ok"):
                    oks.append(r)
                else:
                    bads.append(r)

        oks.sort(key=lambda r: r.get("slot", 0))
        bads.sort(key=lambda r: r.get("slot", 0))

    finally:
        # quiesce_devices recursively quiesces all child submeshes and resets
        # the in_use_ flag via finish_and_reset_in_use(), allowing
        # close_mesh_device to proceed without TT_THROW.
        try:
            ttnn.quiesce_devices(parent)
        except Exception:
            pass
        try:
            ttnn.close_mesh_device(parent)
        except Exception:
            pass

    wall1 = time.perf_counter()
    orchestration_wall = wall1 - wall0

    if not oks:
        print(f"\nAll {k} slots failed!")
        for r in bads:
            print(f"  slot {r.get('slot')}: {r.get('error', r)}")
        # os._exit instead of sys.exit — prevents Py_FinalizeEx from destroying
        # stale submesh MeshDevice wrappers (CQ double-close abort).
        os._exit(1)

    _print_submesh_summary(
        oks=oks,
        bads=bads,
        k=k,
        parent_open_sec=parent_open_sec,
        submesh_create_sec=submesh_create_sec,
        pre_once_sec=pre_once_sec,
        orchestration_wall=orchestration_wall,
        args=args,
        out_dir=out_dir,
        parallel=args.parallel,
    )
    print("Done.", flush=True)

    # Submesh MeshDevice C++ destructors crash during Py_FinalizeEx if any stale
    # Python wrapper survives past close_mesh_device (the parent already freed shared
    # CQs).  os._exit skips interpreter cleanup and avoids the double-close abort.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        os._exit(1)
