#!/usr/bin/env python3
"""
Data-parallel YOLO inference: one letterboxed 640×640 input replicated across every mesh device
(same as batch dim sharded with identical images). No SAHI.

Example (32-way on 8×4 Galaxy; add --save-images to write annotated JPEGs under sample_images_output/dp32_glx):

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --batch-size 32 --tt-mesh-shape 8 4 --tt-model yolov8s --save-images

CPU (Ultralytics) with batch of 32 — same, pass --save-images to export slot PNGs/JPEGs:

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py --backend cpu --input path/to/image.jpg --batch-size 32 --save-images
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ethernet dispatch must be selected before ttnn import when requested.
if __name__ == "__main__" and "--tt-eth-dispatch" in sys.argv:
    os.environ.setdefault("TT_METAL_GTEST_ETH_DISPATCH", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_TT_INPUT_RES = (640, 640)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate one image across N devices / batch (YOLO DP smoke test).")
    p.add_argument("--input", required=True, type=Path, help="Input image path.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Number of identical inputs (must match mesh device count for --backend tt). Default: 32.",
    )
    p.add_argument("--backend", choices=["tt", "cpu"], default="tt", help="tt: Tenstorrent mesh; cpu: Ultralytics.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Where to save annotated outputs when --save-images is set. Default: {_SCRIPT_DIR}/sample_images_output/dp{{N}}_glx.",
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help="Write annotated per-slot JPEGs to --output-dir. If omitted, inference and timing only (no image files).",
    )
    p.add_argument("--tt-model", choices=["yolov8s", "yolov8x"], default="yolov8s")
    p.add_argument(
        "--tt-mesh-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=[8, 4],
        help="Mesh shape for --backend tt. ROWS*COLS must equal --batch-size. Default: 8 4.",
    )
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--tt-l1-small-size", type=int, default=24576)
    p.add_argument("--tt-trace-region-size", type=int, default=6434816)
    p.add_argument("--tt-force-single-device", action="store_true")
    p.add_argument(
        "--tt-eth-dispatch",
        action="store_true",
        help="Set TT_METAL_GTEST_ETH_DISPATCH=1 before device init (multi-chip).",
    )
    p.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics weights for --backend cpu (e.g. yolov8s.pt, yolov8x.pt).",
    )
    return p.parse_args()


def _tt_open_kwargs(args) -> dict:
    return {
        "l1_small_size": args.tt_l1_small_size,
        "trace_region_size": args.tt_trace_region_size,
        "num_command_queues": 2,
    }


def open_tt_mesh(ttnn, args):
    kwargs = _tt_open_kwargs(args)
    if args.tt_force_single_device:
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False
    rows, cols = int(args.tt_mesh_shape[0]), int(args.tt_mesh_shape[1])
    mesh_shape = ttnn.MeshShape(rows, cols)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **kwargs), True


def _fmt_s(sec: float) -> str:
    return f"{float(sec):.4f}s"


def print_run_summary(out_dir: Path, backend: str, timing: dict[str, float], save_images: bool) -> None:
    """Print output directory and pre / device / post wall times."""
    print("")
    print(f"Output directory: {out_dir}")
    if not save_images:
        print("  (--save-images not set: no files written; inference still ran.)")
    pre = timing.get("pre_sec", 0.0)
    dev = timing.get("device_sec", 0.0)
    post = timing.get("post_sec", 0.0)
    total = timing.get("total_sec", pre + dev + post)
    setup = timing.get("setup_sec")
    if backend == "tt" and setup is not None:
        print(
            f"Timing ({backend}):  setup(open_mesh+runner)={_fmt_s(setup)}  "
            f"pre(letterbox+tensor)={_fmt_s(pre)}  device(run)={_fmt_s(dev)}  "
            f"post(host)={_fmt_s(post)}  total={_fmt_s(total)}"
        )
    else:
        print(
            f"Timing ({backend}):  pre={_fmt_s(pre)}  device={_fmt_s(dev)}  post={_fmt_s(post)}  total={_fmt_s(total)}"
        )
    if backend == "tt" and any(k in timing for k in ("to_torch_sec", "postprocess_sec", "save_sec")):
        tt = timing.get("to_torch_sec")
        pp = timing.get("postprocess_sec")
        sv = timing.get("save_sec")
        if tt is not None and pp is not None and sv is not None:
            print(
                f"  post detail:  to_torch={_fmt_s(tt)}  postprocess(NMS+scale)={_fmt_s(pp)}  save_images={_fmt_s(sv)}"
            )
    if backend == "cpu":
        print(
            f"  note: device=predict() only (letterbox+infer inside Ultralytics); "
            f"pre=batch copy only; post={'plot+save' if save_images else 'skipped (no --save-images)'}."
        )


def save_yolo_result_bgr(result: dict, out_path: Path, tt_style: bool) -> None:
    """Draw boxes on result['orig_img'] (BGR uint8) and write JPEG."""
    image = np.ascontiguousarray(result["orig_img"].copy())
    if tt_style:
        box_color, label_color = (255, 0, 0), (255, 255, 0)
    else:
        box_color, label_color = (0, 255, 0), (0, 255, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.detach().cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {float(score):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def run_tt(args, image_bgr: np.ndarray, out_dir: Path, stem: str) -> dict[str, float]:
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers, load_coco_class_names, postprocess, preprocess
    from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
    from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner

    n = args.batch_size
    rows, cols = int(args.tt_mesh_shape[0]), int(args.tt_mesh_shape[1])
    if rows * cols != n:
        raise ValueError(f"--tt-mesh-shape {rows} {cols} is {rows * cols} devices; must equal --batch-size {n}.")

    wall0 = time.perf_counter()
    device, is_mesh = open_tt_mesh(ttnn, args)
    try:
        num_dev = device.get_num_devices()
        if num_dev != n:
            raise RuntimeError(
                f"Opened mesh has {num_dev} devices; expected {n} (match --batch-size and --tt-mesh-shape)."
            )

        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        if num_dev > 1 and inputs_mesh_mapper is None:
            raise RuntimeError("Expected ShardTensorToMesh for multi-device mesh; get_mesh_mappers returned None.")

        mesh_shape_list = None
        try:
            mesh_shape_list = list(device.shape)
        except Exception:
            pass

        if args.tt_model == "yolov8s":
            runner = YOLOv8sPerformantRunner(
                device,
                device_batch_size=n,
                mesh_mapper=inputs_mesh_mapper,
                mesh_composer=output_mesh_composer,
                weights_mesh_mapper=weights_mesh_mapper,
            )
        else:
            runner = YOLOv8xPerformantRunner(
                device,
                device_batch_size=n,
                inputs_mesh_mapper=inputs_mesh_mapper,
                weights_mesh_mapper=weights_mesh_mapper,
                outputs_mesh_composer=output_mesh_composer,
            )

        names = load_coco_class_names()
        t_setup1 = time.perf_counter()
        setup_sec = t_setup1 - wall0

        t_pre0 = time.perf_counter()
        im_one = preprocess([image_bgr], res=_TT_INPUT_RES)
        im = im_one.repeat(n, 1, 1, 1)
        orig_imgs = [image_bgr] * n
        paths = ([f"{stem}#slot{i}" for i in range(n)],)
        t_pre1 = time.perf_counter()

        print(
            "\nTT data-parallel verify (one runner.run, batch sharded on mesh — not 32 sequential single-device runs):\n"
            f"  mesh_device_opened={is_mesh}  get_num_devices()={num_dev}  device.shape={mesh_shape_list!r}\n"
            f"  --tt-mesh-shape={rows}x{cols}  device_batch_size(runner)={n}\n"
            f"  host input tensor shape={tuple(im.shape)}  (dim0 must equal {num_dev})\n"
            f"  inputs: ShardTensorToMesh(dim=0)  weights: ReplicateTensorToMesh  outputs: ConcatMeshToTensor(dim=0)\n"
            "  Sanity: device time should be ~one multi-chip forward, not ~32× a single-chip forward.\n"
        )

        t_dev0 = time.perf_counter()
        preds = runner.run(im)
        t_dev1 = time.perf_counter()
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        t_torch0 = time.perf_counter()
        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=output_mesh_composer)
        t_torch1 = time.perf_counter()
        t_pp0 = time.perf_counter()
        results = postprocess(preds, im, orig_imgs, paths, names)
        t_pp1 = time.perf_counter()
        t_sv0 = time.perf_counter()
        if args.save_images:
            for i, res in enumerate(results):
                save_yolo_result_bgr(res, out_dir / f"{stem}_slot_{i:02d}.jpg", tt_style=True)
        t_sv1 = time.perf_counter()

        runner.release()

        pre_sec = t_pre1 - t_pre0
        device_sec = t_dev1 - t_dev0
        to_torch_sec = t_torch1 - t_torch0
        postprocess_sec = t_pp1 - t_pp0
        save_sec = (t_sv1 - t_sv0) if args.save_images else 0.0
        post_sec = to_torch_sec + postprocess_sec + save_sec
        timing = {
            "setup_sec": setup_sec,
            "pre_sec": pre_sec,
            "device_sec": device_sec,
            "post_sec": post_sec,
            "to_torch_sec": to_torch_sec,
            "postprocess_sec": postprocess_sec,
            "save_sec": save_sec,
            "total_sec": time.perf_counter() - wall0,
        }
        return timing
    finally:
        try:
            ttnn.synchronize_device(device)
        except Exception:
            pass
        if is_mesh:
            ttnn.close_mesh_device(device)
        else:
            ttnn.close_device(device)


def run_cpu(args, image_bgr: np.ndarray, out_dir: Path, stem: str) -> dict[str, float]:
    from ultralytics import YOLO

    n = args.batch_size
    model = YOLO(args.model)
    wall0 = time.perf_counter()
    t_pre0 = time.perf_counter()
    batch = [image_bgr.copy() for _ in range(n)]
    t_pre1 = time.perf_counter()
    t_dev0 = time.perf_counter()
    results = model.predict(batch, imgsz=_TT_INPUT_RES[0], verbose=False)
    t_dev1 = time.perf_counter()
    t_sv0 = time.perf_counter()
    if args.save_images:
        for i, r in enumerate(results):
            out_path = out_dir / f"{stem}_slot_{i:02d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = r.plot()
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    t_sv1 = time.perf_counter()
    pre_sec = t_pre1 - t_pre0
    device_sec = t_dev1 - t_dev0
    save_sec = (t_sv1 - t_sv0) if args.save_images else 0.0
    return {
        "pre_sec": pre_sec,
        "device_sec": device_sec,
        "post_sec": save_sec,
        "save_sec": save_sec,
        "total_sec": time.perf_counter() - wall0,
    }


def main() -> None:
    args = parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _SCRIPT_DIR / "sample_images_output" / f"dp{args.batch_size}_glx"
    out_dir = out_dir.resolve()
    if args.save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    t_read0 = time.perf_counter()
    image_bgr = cv2.imread(str(inp))
    t_read1 = time.perf_counter()
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {inp}")

    stem = inp.stem
    print(f"Input: {inp} shape={image_bgr.shape}  (read={_fmt_s(t_read1 - t_read0)})")
    print(f"Backend: {args.backend} batch_size={args.batch_size}  save_images={args.save_images}")

    if args.backend == "tt":
        timing = run_tt(args, image_bgr, out_dir, stem)
    else:
        timing = run_cpu(args, image_bgr, out_dir, stem)

    print_run_summary(out_dir, args.backend, timing, args.save_images)
    print("Done.")


if __name__ == "__main__":
    main()
