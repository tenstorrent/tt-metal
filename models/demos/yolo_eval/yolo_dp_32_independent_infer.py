#!/usr/bin/env python3
"""
32× independent single-chip YOLO inference (same image on each chip).

Unlike ``yolo_dp_mesh_infer.py`` (one **mesh** with batch sharded across devices,
e.g. 8×4 mesh + global batch 32), this script runs **N separate processes**, each
with ``TT_VISIBLE_DEVICES=<i>`` and ``ttnn.open_device(0, ...)`` — one
``YOLOv8sPerformantRunner`` / ``YOLOv8xPerformantRunner`` per chip with
``device_batch_size=1`` and the **same** letterboxed input.

This matches the UMD isolation pattern used elsewhere (subprocess per chip) and
avoids opening a single multi-device mesh.

Example (32 chips in parallel, same image):

  python models/demos/yolo_eval/yolo_dp_32_independent_infer.py \\
    --input path/to/image.jpg --num-devices 32 --tt-model yolov8s --save-images

Limit concurrent spawns (e.g. host RAM) with ``--max-parallel``:

  python models/demos/yolo_eval/yolo_dp_32_independent_infer.py \\
    --input path/to/image.jpg --num-devices 32 --max-parallel 8

Warmup + multiple timed forwards (same flags as ``yolo_dp_mesh_infer.py``):

  python models/demos/yolo_eval/yolo_dp_32_independent_infer.py \\
    --input path/to/image.jpg --num-devices 32 \\
    --tt-warmup-iters 2 --tt-measured-iters 5

Ethernet dispatch (set before ttnn init in workers):

  python models/demos/yolo_eval/yolo_dp_32_independent_infer.py \\
    --input path/to/image.jpg --num-devices 32 --tt-eth-dispatch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ethernet dispatch must be selected before ttnn import when requested.
if __name__ == "__main__" and "--tt-eth-dispatch" in sys.argv and "--_worker" not in sys.argv:
    os.environ.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_TT_INPUT_RES = (640, 640)
_YDP_RESULT_PREFIX = "YDP_RESULT "


def _fmt_s(sec: float) -> str:
    return f"{float(sec):.4f}s"


def print_run_summary(out_dir: Path, backend: str, timing: dict, save_images: bool) -> None:
    """Print output directory and pre / device / post wall times (same layout as yolo_dp_mesh_infer)."""
    print("")
    print(f"Output directory: {out_dir}")
    if not save_images:
        print("  (--save-images not set: no files written; inference still ran.)")
    pre = timing.get("pre_sec", 0.0)
    dev = timing.get("device_sec", 0.0)
    post = timing.get("post_sec", 0.0)
    total = timing.get("total_sec", pre + dev + post)
    setup = timing.get("setup_sec")
    dev_label = timing.get("tt_device_label", "device(run+sync)")
    if backend == "tt" and setup is not None:
        pre_label = timing.get("tt_pre_label", "pre(letterbox+tensor)")
        setup_label = timing.get("tt_setup_label", "setup(open_mesh+runner)")
        print(
            f"Timing ({backend}):  {setup_label}={_fmt_s(setup)}  "
            f"{pre_label}={_fmt_s(pre)}  "
            f"{dev_label}={_fmt_s(dev)}  "
            f"post(host)={_fmt_s(post)}  total={_fmt_s(total)}"
        )
    else:
        print(
            f"Timing ({backend}):  pre={_fmt_s(pre)}  device={_fmt_s(dev)}  post={_fmt_s(post)}  total={_fmt_s(total)}"
        )
    pt = timing.get("pre_transforms_sec")
    pg = timing.get("pre_golden_cpu_sec")
    if backend == "tt" and pt is not None and pg is not None:
        print(f"  pre detail:  transforms+repeat={_fmt_s(pt)}  cpu_golden_forward={_fmt_s(pg)}")
    plb = timing.get("pre_letterbox_tensor_sec")
    phost = timing.get("pre_prepare_host_input_sec")
    if backend == "tt" and plb is not None and phost is not None:
        print(
            f"  pre detail:  letterbox+repeat={_fmt_s(plb)}  "
            f"prepare_host_input(_setup_l1_sharded_input)={_fmt_s(phost)}"
        )
    if backend == "tt" and any(k in timing for k in ("to_torch_sec", "postprocess_sec", "save_sec")):
        tt = timing.get("to_torch_sec")
        pp = timing.get("postprocess_sec")
        sv = timing.get("save_sec")
        if tt is not None and pp is not None and sv is not None:
            print(
                f"  post detail:  to_torch={_fmt_s(tt)}  postprocess(NMS+scale)={_fmt_s(pp)}  save_images={_fmt_s(sv)}"
            )
    n_dev_iters = timing.get("device_measured_iters", 1) if backend == "tt" else 1
    dmin, dmax = timing.get("device_sec_min"), timing.get("device_sec_max")
    if backend == "tt" and n_dev_iters > 1 and dmin is not None and dmax is not None:
        print(
            f"  {dev_label} over {n_dev_iters} timed iters:  min={_fmt_s(dmin)}  "
            f"mean={_fmt_s(dev)}  max={_fmt_s(dmax)}"
        )
    if backend == "cpu":
        print(
            f"  note: device=predict() only (letterbox+infer inside Ultralytics); "
            f"pre=batch copy only; post={'plot+save' if save_images else 'skipped (no --save-images)'}."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run N independent single-device YOLO runners (same image per chip).")
    p.add_argument("--input", required=True, type=Path, help="Input image path.")
    p.add_argument(
        "--num-devices",
        type=int,
        default=32,
        metavar="N",
        help="Number of physical chips (0..N-1) to run in parallel subprocesses. Default: 32.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=32,
        metavar="K",
        help="Max concurrent worker subprocesses. Default: 32 (all at once).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Annotated JPEGs when --save-images. Default: {_SCRIPT_DIR}/sample_images_output/ind32_{{stem}}.",
    )
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--tt-model", choices=["yolov8s", "yolov8x"], default="yolov8s")
    p.add_argument(
        "--tt-l1-small-size",
        type=int,
        default=24576,
        help="Matches YOLOV8S_L1_SMALL_SIZE in yolov8s perf tests (same default as yolo_dp_mesh_infer).",
    )
    p.add_argument(
        "--tt-trace-region-size",
        type=int,
        default=6434816,
        help="Matches test_e2e_performant trace_region_size for YOLOv8s trace+2CQ (same as yolo_dp_mesh_infer).",
    )
    p.add_argument(
        "--tt-device-id",
        type=int,
        default=0,
        metavar="ID",
        help=(
            "Logical device id passed to ttnn.open_device in each worker. "
            "With TT_VISIBLE_DEVICES=<chip>, this is almost always 0 (default). Same flag name as yolo_dp_mesh_infer."
        ),
    )
    p.add_argument(
        "--tt-row-dispatch",
        action="store_true",
        help=(
            "Open device with DispatchCoreConfig(axis=ROW). Dispatch core type stays the cluster default "
            "(same as yolo_dp_mesh_infer)."
        ),
    )
    p.add_argument(
        "--tt-eth-dispatch",
        action="store_true",
        help="Set TT_METAL_GTEST_ETH_DISPATCH=1 before device init in workers (same as yolo_dp_mesh_infer).",
    )
    p.add_argument(
        "--tt-warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Before timing, each worker runs N extra runner.run()+sync calls (not included in device(run+sync) stats). "
            "Default: 0."
        ),
    )
    p.add_argument(
        "--tt-measured-iters",
        type=int,
        default=1,
        metavar="N",
        help=(
            "After warmup, each worker times N consecutive runner.run()+sync intervals; "
            "aggregate summary prints per-chip min/mean/max when N>1. Default: 1."
        ),
    )
    p.add_argument(
        "--_worker",
        type=int,
        metavar="CHIP_ID",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def _worker_run(chip_id: int, args: argparse.Namespace) -> None:
    """Runs inside subprocess; TT_VISIBLE_DEVICES must map this chip to logical device 0."""
    import cv2
    import torch

    import ttnn
    from models.demos.utils.common_demo_utils import load_coco_class_names, postprocess, preprocess
    from models.demos.yolo_eval.yolo_dp_mesh_infer import save_yolo_result_bgr
    from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
    from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner

    inp = args.input.resolve()
    stem = inp.stem
    out_dir = args.output_dir.resolve() if args.output_dir else None

    t_read0 = time.perf_counter()
    image_bgr = cv2.imread(str(inp))
    t_read1 = time.perf_counter()
    if image_bgr is None:
        print(f"YDP_RESULT {json.dumps({'chip': chip_id, 'ok': False, 'error': 'imread failed'})}", flush=True)
        sys.exit(1)

    t_pre0 = time.perf_counter()
    im = preprocess([image_bgr], res=_TT_INPUT_RES)
    t_pre1 = time.perf_counter()
    read_sec = t_read1 - t_read0
    letterbox_sec = t_pre1 - t_pre0
    pre_sec = read_sec + letterbox_sec

    kwargs = {
        "l1_small_size": args.tt_l1_small_size,
        "trace_region_size": args.tt_trace_region_size,
        "num_command_queues": 2,
    }
    if args.tt_row_dispatch:
        kwargs["dispatch_core_config"] = ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)

    wall0 = time.perf_counter()
    t_setup0 = time.perf_counter()
    device = ttnn.open_device(device_id=args.tt_device_id, **kwargs)
    try:
        if args.tt_model == "yolov8s":
            runner = YOLOv8sPerformantRunner(
                device,
                device_batch_size=1,
                mesh_mapper=None,
                mesh_composer=None,
                weights_mesh_mapper=None,
            )
        else:
            runner = YOLOv8xPerformantRunner(
                device,
                device_batch_size=1,
                inputs_mesh_mapper=None,
                weights_mesh_mapper=None,
                outputs_mesh_composer=None,
            )
        setup_sec = time.perf_counter() - t_setup0

        names = load_coco_class_names()
        orig_imgs = [image_bgr]
        paths = ([f"{stem}#chip{chip_id}"],)

        use_split = args.tt_model == "yolov8s"
        t_prep0 = time.perf_counter()
        if use_split:
            tt_host = runner.prepare_host_input(im)
        host_prep_sec = time.perf_counter() - t_prep0

        if use_split:
            for _ in range(args.tt_warmup_iters):
                runner.push_host_input_to_device_dram(tt_host)
                _wp = runner.execute_reshard_and_trace()
                del _wp
        else:
            for _ in range(args.tt_warmup_iters):
                preds = runner.run(im)
                ttnn.synchronize_device(device)
                del preds

        device_secs: list[float] = []
        h2d_secs: list[float] = []
        compute_secs: list[float] = []
        preds = None
        n_meas = int(args.tt_measured_iters)
        if n_meas < 1:
            raise ValueError("--tt-measured-iters must be >= 1")
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
                ttnn.synchronize_device(device)
                device_secs.append(time.perf_counter() - t0)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        t_torch0 = time.perf_counter()
        preds_t = ttnn.to_torch(preds, dtype=torch.float32)
        t_torch1 = time.perf_counter()
        t_pp0 = time.perf_counter()
        results = postprocess(preds_t, im, orig_imgs, paths, names)
        t_pp1 = time.perf_counter()
        t_sv0 = time.perf_counter()
        if args.save_images and out_dir is not None:
            save_yolo_result_bgr(
                results[0],
                out_dir / f"{stem}_chip_{chip_id:02d}.jpg",
                tt_style=True,
            )
        t_sv1 = time.perf_counter()

        try:
            runner.release()
        except Exception:
            pass

        to_torch_sec = t_torch1 - t_torch0
        postprocess_sec = t_pp1 - t_pp0
        save_sec = (t_sv1 - t_sv0) if args.save_images else 0.0
        device_sec = sum(device_secs) / len(device_secs)
        payload = {
            "chip": chip_id,
            "ok": True,
            "setup_sec": setup_sec,
            "pre_sec": pre_sec,
            "read_sec": read_sec,
            "letterbox_sec": letterbox_sec,
            "host_prep_sec": host_prep_sec,
            "tt_warmup_iters": int(args.tt_warmup_iters),
            "tt_measured_iters": n_meas,
            "device_sec": device_sec,
            "device_sec_min": min(device_secs) if len(device_secs) > 1 else device_sec,
            "device_sec_max": max(device_secs) if len(device_secs) > 1 else device_sec,
            "to_torch_sec": to_torch_sec,
            "postprocess_sec": postprocess_sec,
            "save_sec": save_sec,
            "total_sec": time.perf_counter() - wall0,
        }
        if h2d_secs:
            payload["h2d_sec"] = sum(h2d_secs) / len(h2d_secs)
            payload["h2d_sec_min"] = min(h2d_secs)
            payload["h2d_sec_max"] = max(h2d_secs)
        if compute_secs:
            payload["compute_sec"] = sum(compute_secs) / len(compute_secs)
            payload["compute_sec_min"] = min(compute_secs)
            payload["compute_sec_max"] = max(compute_secs)
        print(f"{_YDP_RESULT_PREFIX}{json.dumps(payload)}", flush=True)
    finally:
        try:
            ttnn.synchronize_device(device)
        except Exception:
            pass
        try:
            ttnn.close_device(device)
        except Exception:
            pass


def _build_worker_command(chip_id: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--_worker",
        str(chip_id),
        "--input",
        str(args.input.resolve()),
        "--tt-model",
        args.tt_model,
        "--tt-l1-small-size",
        str(args.tt_l1_small_size),
        "--tt-trace-region-size",
        str(args.tt_trace_region_size),
        "--tt-device-id",
        str(args.tt_device_id),
        "--tt-warmup-iters",
        str(args.tt_warmup_iters),
        "--tt-measured-iters",
        str(args.tt_measured_iters),
    ]
    if args.output_dir is not None:
        cmd += ["--output-dir", str(args.output_dir.resolve())]
    if args.save_images:
        cmd.append("--save-images")
    if args.tt_row_dispatch:
        cmd.append("--tt-row-dispatch")
    if args.tt_eth_dispatch:
        cmd.append("--tt-eth-dispatch")
    return cmd


def _launch_worker(chip_id: int, args: argparse.Namespace) -> dict:
    env = {**os.environ, "TT_VISIBLE_DEVICES": str(chip_id)}
    if args.tt_eth_dispatch:
        env.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")
    cmd = _build_worker_command(chip_id, args)
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    result_line = None
    for line in out.splitlines():
        if line.strip().startswith(_YDP_RESULT_PREFIX):
            result_line = line.strip()[len(_YDP_RESULT_PREFIX) :]
    if proc.returncode != 0 or not result_line:
        return {
            "chip": chip_id,
            "ok": False,
            "error": f"exit={proc.returncode} stdout/stderr tail: {out[-2000:]}",
        }
    try:
        return json.loads(result_line)
    except json.JSONDecodeError:
        return {"chip": chip_id, "ok": False, "error": f"bad json: {result_line!r}"}


def _parent_main(args: argparse.Namespace) -> None:
    inp = args.input.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    n = int(args.num_devices)
    if n < 1:
        raise ValueError("--num-devices must be >= 1")
    max_par = int(args.max_parallel)
    if max_par < 1:
        raise ValueError("--max-parallel must be >= 1")
    if int(args.tt_measured_iters) < 1:
        raise ValueError("--tt-measured-iters must be >= 1")
    if int(args.tt_warmup_iters) < 0:
        raise ValueError("--tt-warmup-iters must be >= 0")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _SCRIPT_DIR / "sample_images_output" / f"ind32_{inp.stem}"
    args.output_dir = out_dir
    out_dir = out_dir.resolve()
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Independent {n}×1-chip YOLO ({args.tt_model}), same image per chip.\n"
        f"  input={inp}\n"
        f"  workers: TT_VISIBLE_DEVICES=i, open_device({args.tt_device_id}), batch=1\n"
        f"  max_parallel={max_par}  tt_warmup_iters={args.tt_warmup_iters}  "
        f"tt_measured_iters={args.tt_measured_iters}  save_images={args.save_images}\n"
        f"  output_dir={out_dir}\n"
    )

    wall0 = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_par) as pool:
        futures = {pool.submit(_launch_worker, i, args): i for i in range(n)}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r.get("chip", 0))
    wall1 = time.perf_counter()

    oks = [r for r in results if r.get("ok")]
    bads = [r for r in results if not r.get("ok")]

    print(
        f"\nOrchestration: wall(all {n} workers, max_parallel={max_par})="
        f"{(wall1 - wall0):.4f}s  successful_chips={len(oks)}/{n}\n"
    )

    if oks:

        def _mean(key: str) -> float:
            return sum(float(r[key]) for r in oks) / len(oks)

        post_wall = _mean("to_torch_sec") + _mean("postprocess_sec") + _mean("save_sec")
        read_mean = _mean("read_sec")
        lb_mean = _mean("letterbox_sec")
        has_split = any("h2d_sec" in r for r in oks)
        timing = {
            "setup_sec": _mean("setup_sec"),
            "pre_sec": _mean("pre_sec"),
            "device_sec": _mean("device_sec"),
            "post_sec": post_wall,
            "total_sec": wall1 - wall0,
            "to_torch_sec": _mean("to_torch_sec"),
            "postprocess_sec": _mean("postprocess_sec"),
            "save_sec": _mean("save_sec"),
            "tt_device_label": "device(push_dram+reshard+trace) mean@chip"
            if has_split
            else "device(run+sync) mean@chip",
            "tt_pre_label": "pre(read+letterbox+tensor) mean@chip",
            "tt_setup_label": "setup(open_dev+runner) mean@chip",
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
        print_run_summary(out_dir, "tt", timing, args.save_images)
        print(
            f"  pre detail:  read+imdecode={read_mean:.4f}s  "
            f"letterbox+tensor={lb_mean:.4f}s  "
            f"(means over {len(oks)} successful workers)\n"
            f"  per-chip device spread:  min={min(r['device_sec'] for r in oks):.4f}s  "
            f"mean={timing['device_sec']:.4f}s  max={max(r['device_sec'] for r in oks):.4f}s"
        )
        if has_split:
            print(
                f"  split timing (mean@chip):  "
                f"host_prep(once)={timing['host_prep_sec']:.6f}s  "
                f"h2d(push_dram)={timing['h2d_sec']:.6f}s  "
                f"compute(reshard+trace+sync)={timing['compute_sec']:.6f}s"
            )
        if args.tt_measured_iters > 1:
            print(
                f"  per-chip over {args.tt_measured_iters} timed iters (each worker):  "
                f"min-of-mins={min(r['device_sec_min'] for r in oks):.4f}s  "
                f"max-of-maxes={max(r['device_sec_max'] for r in oks):.4f}s"
            )
            if has_split:
                print(
                    f"  h2d: min={timing['h2d_sec_min']:.6f}s  max={timing['h2d_sec_max']:.6f}s  "
                    f"compute: min={timing['compute_sec_min']:.6f}s  max={timing['compute_sec_max']:.6f}s"
                )
    if bads:
        print(f"\nFailed workers ({len(bads)}):")
        for r in bads:
            print(f"  chip {r.get('chip')}: {r.get('error', r)}")
    print("Done.")


def main() -> None:
    args = parse_args()
    if args._worker is not None:
        if args.tt_eth_dispatch:
            os.environ.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")
        _worker_run(args._worker, args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
