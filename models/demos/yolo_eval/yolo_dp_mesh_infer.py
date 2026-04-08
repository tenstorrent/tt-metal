#!/usr/bin/env python3
"""
Data-parallel YOLO inference: letterboxed 640×640 inputs sharded across the mesh (global batch =
sum of per-device batches). Use the same image repeated global_batch times for a ResNet-style DP
smoke test (e.g. 16 images/chip × 32 chips = 512).

Default Wormhole Galaxy **mesh** is 8×4 (32 devices) with default global batch 32. Use
``--tt-row-dispatch`` to open the mesh with ``DispatchCoreConfig(axis=ROW)`` (dispatch core type stays
the machine default). The YOLOv8s performant runner still L1 height-shards the input on a fixed **8×8**
tensix block so it matches the model’s hardcoded cores; ROW dispatch changes device routing, not that
grid (using the full 8×9 / fragmented grid caused matmul shard mismatch and very long first runs).
``--tt-eth-dispatch`` only sets ``TT_METAL_GTEST_ETH_DISPATCH`` when needed for your cluster.

**YOLOv8s on a multi-device mesh** uses the same device settings and runner phases as pytest
``models/demos/yolov8s/tests/perf/test_e2e_performant.py::test_run_yolov8s_trace_2cqs_dp_galaxy_8x4_fps_without_h2d``:
``l1_small_size`` / ``trace_region_size`` / 2 CQs (see ``--tt-l1-small-size``, ``--tt-trace-region-size``),
and timed forward uses ``prepare_host_input`` once then each iteration
``push_host_input_to_device_dram`` + ``execute_reshard_and_trace`` (not ``runner.run`` per iter).
Single-device YOLOv8s still uses ``runner.run`` + sync.

Example (32-way on 8×4 Galaxy; add --save-images for JPEGs):

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --batch-size 32 --tt-mesh-shape 8 4 --tt-model yolov8s --save-images

Galaxy with **ROW** dispatch axis:

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --batch-size 32 --tt-mesh-shape 8 4 --tt-model yolov8s --tt-row-dispatch

Per-device batch 16 (global 512 on 32 devices), system mesh from the driver:

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --batch-size 512 --tt-use-system-mesh --tt-model yolov8s

Single chip, global batch 16:

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --batch-size 16 --tt-force-single-device --tt-model yolov8s

ResNet50 (same mesh/batch rules; reuses ``create_test_infra`` + tt_cnn 2CQ traced pipeline as in
``perf_e2e_resnet50``). Per-device batch must be 16, 20 (Blackhole only), or 32. Weights use host
preprocess with ``weights_mesh_mapper=None`` (no ``ReplicateTensorToMesh``); inputs use
``ShardTensorToMesh(dim=0)``. Timed interval is ``pipeline.enqueue([host])+pop_all()`` (includes
output D2H + sync), unlike YOLO's ``runner.run`` + ``synchronize_device`` before host readback.

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/cat.jpg --batch-size 512 --tt-mesh-shape 8 4 --tt-model resnet50

(512 = 32 devices × 16 samples/chip; YOLO’s default --batch-size 32 would be 1/chip here and is invalid for ResNet.)

Optional steady-state device timing:

  python models/demos/yolo_eval/yolo_dp_mesh_infer.py \\
    --backend tt --input path/to/image.jpg --tt-use-system-mesh --tt-warmup-iters 2 --tt-measured-iters 5

CPU (Ultralytics) with batch of 32:

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
# Default explicit mesh for Wormhole Galaxy when not using --tt-use-system-mesh (32 devices).
_DEFAULT_GALAXY_MESH_ROWS = 8
_DEFAULT_GALAXY_MESH_COLS = 4
_DEFAULT_TT_BATCH_SIZE = _DEFAULT_GALAXY_MESH_ROWS * _DEFAULT_GALAXY_MESH_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate one image across N devices / batch (YOLO DP smoke test).")
    p.add_argument("--input", required=True, type=Path, help="Input image path.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_TT_BATCH_SIZE,
        metavar="N",
        help=(
            "Global batch: N identical letterboxed inputs. For --backend tt, N must be divisible by "
            f"mesh device count (default mesh {_DEFAULT_GALAXY_MESH_ROWS}×{_DEFAULT_GALAXY_MESH_COLS} "
            f"→ default N={_DEFAULT_TT_BATCH_SIZE})."
        ),
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
    p.add_argument(
        "--tt-model",
        choices=["yolov8s", "yolov8x", "resnet50"],
        default="yolov8s",
        help=(
            "tt: yolov8s/yolov8x use YOLO performant runners; resnet50 uses ResNet50TestInfra + tt_cnn "
            "2CQ trace pipeline (see module docstring). CPU backend ignores this."
        ),
    )
    p.add_argument(
        "--tt-mesh-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=None,
        help=(
            f"Mesh shape for --backend tt; global batch must be divisible by ROWS*COLS. "
            f"Default: {_DEFAULT_GALAXY_MESH_ROWS} {_DEFAULT_GALAXY_MESH_COLS} unless --tt-use-system-mesh."
        ),
    )
    p.add_argument(
        "--tt-use-system-mesh",
        action="store_true",
        help=(
            "Open the full system mesh from SystemMeshDescriptor() (same idea as multi-chip ResNet / Galaxy DP bring-up). "
            "Mesh shape follows the machine; --batch-size is the global batch and must be divisible by the device count. "
            "Ignores --tt-mesh-shape."
        ),
    )
    p.add_argument(
        "--tt-warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help="Before timing, run N extra runner.run()+sync calls (not included in device(run+sync) stats). Default: 0.",
    )
    p.add_argument(
        "--tt-measured-iters",
        type=int,
        default=1,
        metavar="N",
        help=(
            "After warmup, time N consecutive runner.run()+sync intervals; print device(run+sync) min/mean/max when N>1. "
            "Default: 1 (single timed forward)."
        ),
    )
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument(
        "--tt-l1-small-size",
        type=int,
        default=24576,
        help="Matches YOLOV8S_L1_SMALL_SIZE in yolov8s perf tests (e.g. galaxy 8×4 e2e performant).",
    )
    p.add_argument(
        "--tt-trace-region-size",
        type=int,
        default=6434816,
        help="Matches test_e2e_performant trace_region_size for YOLOv8s trace+2CQ.",
    )
    p.add_argument("--tt-force-single-device", action="store_true")
    p.add_argument(
        "--tt-row-dispatch",
        action="store_true",
        help=(
            "Open device/mesh with DispatchCoreConfig(axis=ROW). Dispatch core type stays the cluster default "
            "(e.g. ETH on T3K/Galaxy). YOLOv8s input L1 sharding stays on an 8×8 tensix block to match the model."
        ),
    )
    p.add_argument(
        "--tt-eth-dispatch",
        action="store_true",
        help="Set TT_METAL_GTEST_ETH_DISPATCH=1 before device init (Ethernet cores for dispatch on supported clusters).",
    )
    p.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics weights for --backend cpu (e.g. yolov8s.pt, yolov8x.pt).",
    )
    return p.parse_args()


def _tt_open_kwargs(ttnn, args) -> dict:
    # ResNet50 e2e perf tests use these defaults. YOLOv8s defaults (24576 / 6434816 / 2 CQs) match
    # test_e2e_performant::test_run_yolov8s_trace_2cqs_dp_galaxy_8x4_fps_without_h2d when left at defaults.
    if args.tt_model == "resnet50":
        kwargs = {
            "l1_small_size": 32768,
            "trace_region_size": 1332224,
            "num_command_queues": 2,
        }
    else:
        kwargs = {
            "l1_small_size": args.tt_l1_small_size,
            "trace_region_size": args.tt_trace_region_size,
            "num_command_queues": 2,
        }
    if args.tt_row_dispatch:
        kwargs["dispatch_core_config"] = ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)
    return kwargs


def resolve_tt_mesh_and_batch(ttnn, args) -> tuple[int, int, int]:
    """
    Return (global_batch, mesh_rows, mesh_cols). Validates global_batch % (rows*cols) == 0.
    For --tt-use-system-mesh, mesh shape comes from SystemMeshDescriptor().
    """
    global_batch = int(args.batch_size)
    if args.tt_force_single_device:
        if global_batch < 1:
            raise ValueError("--batch-size must be >= 1")
        return global_batch, 1, 1

    if args.tt_use_system_mesh:
        sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
        if len(sys_shape) != 2:
            raise RuntimeError(f"Unexpected system mesh shape tuple: {sys_shape}")
        rows, cols = int(sys_shape[0]), int(sys_shape[1])
        n_sys = rows * cols
        if n_sys <= 1:
            raise RuntimeError(
                "--tt-use-system-mesh needs a multi-device system (descriptor reports <=1 device). "
                "Use --tt-force-single-device without --tt-use-system-mesh for a single chip."
            )
        if global_batch % n_sys != 0:
            raise ValueError(
                f"--batch-size {global_batch} must be divisible by system mesh device count {n_sys} "
                f"(MeshShape({rows}, {cols})) when using --tt-use-system-mesh."
            )
        return global_batch, rows, cols

    assert args.tt_mesh_shape is not None
    rows, cols = int(args.tt_mesh_shape[0]), int(args.tt_mesh_shape[1])
    n_mesh = rows * cols
    if global_batch % n_mesh != 0:
        raise ValueError(
            f"--batch-size {global_batch} must be divisible by mesh device count {n_mesh} ({rows}x{cols})."
        )
    return global_batch, rows, cols


def open_tt_mesh(ttnn, args, mesh_rows: int, mesh_cols: int):
    kwargs = _tt_open_kwargs(ttnn, args)
    if args.tt_force_single_device:
        return ttnn.open_device(device_id=args.tt_device_id, **kwargs), False
    mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
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
    if backend == "tt" and n_dev_iters > 1:
        print(
            f"  {dev_label} over {n_dev_iters} timed iters:  min={_fmt_s(timing['device_sec_min'])}  "
            f"mean={_fmt_s(timing['device_sec'])}  max={_fmt_s(timing['device_sec_max'])}"
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

    if args.tt_model == "resnet50":
        raise RuntimeError("internal: use run_tt_resnet for --tt-model resnet50")
    if args.tt_use_system_mesh and args.tt_force_single_device:
        raise ValueError("--tt-use-system-mesh cannot be used with --tt-force-single-device.")
    if args.tt_warmup_iters < 0:
        raise ValueError("--tt-warmup-iters must be >= 0.")
    if args.tt_measured_iters < 1:
        raise ValueError("--tt-measured-iters must be >= 1.")

    if args.tt_mesh_shape is None and not args.tt_use_system_mesh:
        args.tt_mesh_shape = [_DEFAULT_GALAXY_MESH_ROWS, _DEFAULT_GALAXY_MESH_COLS]

    global_batch, rows, cols = resolve_tt_mesh_and_batch(ttnn, args)
    num_mesh_devices = rows * cols

    wall0 = time.perf_counter()
    device, is_mesh = open_tt_mesh(ttnn, args, rows, cols)
    try:
        num_dev = device.get_num_devices()
        if num_dev != num_mesh_devices:
            raise RuntimeError(
                f"Opened mesh has {num_dev} devices; expected {num_mesh_devices} (from mesh shape {rows}x{cols})."
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
                device_batch_size=global_batch,
                mesh_mapper=inputs_mesh_mapper,
                mesh_composer=output_mesh_composer,
                weights_mesh_mapper=weights_mesh_mapper,
            )
        else:
            runner = YOLOv8xPerformantRunner(
                device,
                device_batch_size=global_batch,
                inputs_mesh_mapper=inputs_mesh_mapper,
                weights_mesh_mapper=weights_mesh_mapper,
                outputs_mesh_composer=output_mesh_composer,
            )

        names = load_coco_class_names()
        t_setup1 = time.perf_counter()
        setup_sec = t_setup1 - wall0

        t_pre0 = time.perf_counter()
        im_one = preprocess([image_bgr], res=_TT_INPUT_RES)
        im = im_one.repeat(global_batch, 1, 1, 1)
        orig_imgs = [image_bgr] * global_batch
        paths = ([f"{stem}#slot{i}" for i in range(global_batch)],)
        t_pre1 = time.perf_counter()

        mesh_src = "SystemMeshDescriptor()" if args.tt_use_system_mesh else f"--tt-mesh-shape {rows}x{cols}"
        per_dev = global_batch // num_dev
        yolov8s_mesh_e2e_performant = args.tt_model == "yolov8s" and is_mesh and num_dev > 1
        if yolov8s_mesh_e2e_performant:
            device_phase = (
                "prepare_host_input once, then per timed iter: push_host_input_to_device_dram + "
                "execute_reshard_and_trace (sync inside; same as test_e2e_performant "
                "test_run_yolov8s_trace_2cqs_dp_galaxy_8x4_fps_without_h2d)"
            )
        else:
            device_phase = "runner.run + ttnn.synchronize_device (execute_reshard_and_trace already syncs; extra sync is redundant)"
        print(
            "\nTT data-parallel verify (batch sharded on mesh — not N sequential single-device runs):\n"
            f"  mesh_device_opened={is_mesh}  get_num_devices()={num_dev}  device.shape={mesh_shape_list!r}\n"
            f"  mesh={mesh_src}  global_batch(runner)={global_batch}  per_device_batch={per_dev}\n"
            f"  host input tensor shape={tuple(im.shape)}  (dim0 = global batch, divisible by {num_dev})\n"
            f"  inputs: host shards / ShardTensorToMesh(dim=0)  weights: ReplicateTensorToMesh  outputs: ConcatMeshToTensor(dim=0)\n"
            f"  warmup_iters={args.tt_warmup_iters}  measured_iters={args.tt_measured_iters}\n"
            f"  device phase: {device_phase}\n"
            "  Single-chip example: --tt-force-single-device --batch-size 16. "
            "For steady-state device time, use --tt-warmup-iters and --tt-measured-iters.\n"
        )

        device_secs: list[float] = []
        preds = None
        host_shards_prep_sec = 0.0
        if yolov8s_mesh_e2e_performant:
            # Same as test_e2e_performant prep_once: _setup_l1_sharded_input (not part of per-iter device_sec).
            t_hprep0 = time.perf_counter()
            tt_host = runner.prepare_host_input(im)
            host_shards_prep_sec = time.perf_counter() - t_hprep0
            for _ in range(args.tt_warmup_iters):
                runner.push_host_input_to_device_dram(tt_host)
                _wp = runner.execute_reshard_and_trace()
                del _wp
            for _ in range(args.tt_measured_iters):
                t_dev0 = time.perf_counter()
                runner.push_host_input_to_device_dram(tt_host)
                preds = runner.execute_reshard_and_trace()
                t_dev1 = time.perf_counter()
                device_secs.append(t_dev1 - t_dev0)
        else:
            for _ in range(args.tt_warmup_iters):
                _wp = runner.run(im)
                ttnn.synchronize_device(device)
                del _wp

            for _ in range(args.tt_measured_iters):
                t_dev0 = time.perf_counter()
                preds = runner.run(im)
                ttnn.synchronize_device(device)
                t_dev1 = time.perf_counter()
                device_secs.append(t_dev1 - t_dev0)
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

        pre_letterbox_tensor_sec = t_pre1 - t_pre0
        pre_sec = pre_letterbox_tensor_sec + host_shards_prep_sec
        device_sec = sum(device_secs) / len(device_secs)
        device_sec_min = min(device_secs) if len(device_secs) > 1 else None
        device_sec_max = max(device_secs) if len(device_secs) > 1 else None
        to_torch_sec = t_torch1 - t_torch0
        postprocess_sec = t_pp1 - t_pp0
        save_sec = (t_sv1 - t_sv0) if args.save_images else 0.0
        post_sec = to_torch_sec + postprocess_sec + save_sec
        timing = {
            "setup_sec": setup_sec,
            "pre_sec": pre_sec,
            "device_sec": device_sec,
            "device_sec_min": device_sec_min,
            "device_sec_max": device_sec_max,
            "device_measured_iters": args.tt_measured_iters,
            "post_sec": post_sec,
            "to_torch_sec": to_torch_sec,
            "postprocess_sec": postprocess_sec,
            "save_sec": save_sec,
            "total_sec": time.perf_counter() - wall0,
        }
        if yolov8s_mesh_e2e_performant:
            timing["tt_device_label"] = "device(push_dram+reshard+trace+sync)"
            timing["tt_pre_label"] = "pre(letterbox+tensor+prepare_host_input)"
            timing["pre_letterbox_tensor_sec"] = pre_letterbox_tensor_sec
            timing["pre_prepare_host_input_sec"] = host_shards_prep_sec
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


def _assert_resnet_per_device_batch(per_dev: int, global_batch: int, num_dev: int) -> None:
    from models.common.utility_functions import is_wormhole_b0

    ok = per_dev in (16, 20, 32)
    if ok and is_wormhole_b0() and per_dev == 20:
        ok = False
    if ok:
        return

    wh = is_wormhole_b0()
    extra = ""
    if wh:
        extra = " On Wormhole B0, per-device batch 20 is also unsupported."
    raise ValueError(
        "ResNet50 needs per-device batch 16, 20 (Blackhole only), or 32 (ResNet50TestInfra L1 input grid). "
        f"Got per_device_batch={per_dev} from --batch-size={global_batch} // num_devices={num_dev}.{extra}\n"
        f"  Examples for this mesh: --batch-size {16 * num_dev} (16/chip), --batch-size {32 * num_dev} (32/chip)."
    )


def run_tt_resnet(args, image_bgr: np.ndarray, out_dir: Path, stem: str) -> dict[str, float]:
    """
    ResNet50 data-parallel path: ``create_test_infra`` + ``tt_cnn`` traced 2CQ pipeline (same stack as
    ``perf_e2e_resnet50.run_trace_2cq_model_pipeline``). Weights avoid ``ReplicateTensorToMesh``
    (``weights_mesh_mapper=None``); inputs shard on dim0; outputs concatenate on dim0.

    Timed wall interval per iteration: ``pipeline.enqueue([host_input]).pop_all()`` (includes output
    device-to-host and ``synchronize_device`` inside ``pop_all``), not the same split as YOLO's
    ``runner.run`` + sync with readback timed separately under ``post``.
    """
    import torchvision.transforms as T
    from PIL import Image

    import ttnn
    from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.perf_e2e_resnet50 import model_config
    from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
        create_test_infra,
        load_resnet50_model,
    )
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

    if args.tt_use_system_mesh and args.tt_force_single_device:
        raise ValueError("--tt-use-system-mesh cannot be used with --tt-force-single-device.")
    if args.tt_warmup_iters < 0:
        raise ValueError("--tt-warmup-iters must be >= 0.")
    if args.tt_measured_iters < 1:
        raise ValueError("--tt-measured-iters must be >= 1.")

    if args.tt_mesh_shape is None and not args.tt_use_system_mesh:
        args.tt_mesh_shape = [_DEFAULT_GALAXY_MESH_ROWS, _DEFAULT_GALAXY_MESH_COLS]

    global_batch, rows, cols = resolve_tt_mesh_and_batch(ttnn, args)
    num_mesh_devices = rows * cols

    wall0 = time.perf_counter()
    device, is_mesh = open_tt_mesh(ttnn, args, rows, cols)
    pipeline = None
    try:
        num_dev = device.get_num_devices()
        if num_dev != num_mesh_devices:
            raise RuntimeError(
                f"Opened mesh has {num_dev} devices; expected {num_mesh_devices} (from mesh shape {rows}x{cols})."
            )
        per_dev = global_batch // num_dev
        _assert_resnet_per_device_batch(per_dev, global_batch, num_dev)

        test_infra = create_test_infra(
            device,
            per_dev,
            model_config["ACTIVATIONS_DTYPE"],
            model_config["WEIGHTS_DTYPE"],
            model_config["MATH_FIDELITY"],
            use_pretrained_weight=True,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
            model_location_generator=None,
        )

        t_pre0 = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tfm = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        one = tfm(pil).unsqueeze(0).to(dtype=torch.bfloat16)
        torch_host = one.repeat(global_batch, 1, 1, 1).contiguous()
        test_infra.torch_input_tensor = torch_host
        t_golden0 = time.perf_counter()
        with torch.no_grad():
            golden_m = load_resnet50_model(None).eval().to(torch.bfloat16)
            test_infra.torch_output_tensor = golden_m(test_infra.torch_input_tensor)
        t_pre1 = time.perf_counter()

        def model_wrapper(l1_input_tensor):
            test_infra.input_tensor = l1_input_tensor
            return test_infra.run()

        tt_inputs_host, sharded_mem_config_dram, input_mem_config = test_infra.setup_dram_sharded_input(device)
        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=True,
                num_command_queues=2,
                all_transfers_on_separate_command_queue=False,
            ),
            model=model_wrapper,
            device=device,
            dram_input_memory_config=sharded_mem_config_dram,
            l1_input_memory_config=input_mem_config,
        )

        t_compile0 = time.perf_counter()
        pipeline.compile(tt_inputs_host)
        # One host output slot per forward; each timed iter calls enqueue([one]).pop_all() sequentially.
        pipeline.preallocate_output_tensors_on_host(1)
        t_compile1 = time.perf_counter()

        mesh_src = "SystemMeshDescriptor()" if args.tt_use_system_mesh else f"--tt-mesh-shape {rows}x{cols}"
        pre_transforms_sec = t_golden0 - t_pre0
        pre_golden_sec = t_pre1 - t_golden0
        print(
            "\nTT ResNet50 data-parallel (tt_cnn 2CQ traced pipeline; same infra as ttnn_resnet perf e2e):\n"
            f"  mesh_device_opened={is_mesh}  get_num_devices()={num_dev}  device.shape={getattr(device, 'shape', None)!r}\n"
            f"  mesh={mesh_src}  global_batch={global_batch}  per_device_batch={per_dev}\n"
            f"  torch input shape={tuple(torch_host.shape)}  ttnn host tensor shape={list(tt_inputs_host.shape)} "
            f"(with ShardTensorToMesh, dim0 is often per-device logical N, not global batch)\n"
            f"  pre breakdown (see Timing 'pre' below):  transforms+repeat={_fmt_s(pre_transforms_sec)}  "
            f"cpu_PyTorch_golden_forward(batch={global_batch})={_fmt_s(pre_golden_sec)}\n"
            f"  inputs: ShardTensorToMesh(dim=0)  weights: host preprocess, weights_mesh_mapper=None "
            f"(no ReplicateTensorToMesh)  outputs: ConcatMeshToTensor(dim=0)\n"
            f"  timed interval per iter: pipeline.enqueue([host])+pop_all() (D2H + sync inside pop_all)\n"
            f"  compile+prealloc={_fmt_s(t_compile1 - t_compile0)}  warmup_iters={args.tt_warmup_iters}  "
            f"measured_iters={args.tt_measured_iters}\n"
        )
        if args.save_images:
            print("  note: --save-images is ignored for ResNet50 (classification); no box drawing.\n")

        for _ in range(args.tt_warmup_iters):
            pipeline.enqueue([tt_inputs_host]).pop_all()

        device_secs: list[float] = []
        last_host_output = None
        for _ in range(args.tt_measured_iters):
            t0 = time.perf_counter()
            pipeline.enqueue([tt_inputs_host]).pop_all()
            t1 = time.perf_counter()
            device_secs.append(t1 - t0)
            out = pipeline.output_tensors
            if out:
                last_host_output = out[0]

        # Release trace before validate(); ttnn.to_torch during an active trace can trigger device
        # allocations and Metal warnings ("Allocating device buffers is unsafe...").
        if pipeline is not None:
            pipeline.cleanup()
            pipeline = None

        t_val0 = time.perf_counter()
        if last_host_output is not None:
            test_infra.validate(last_host_output)
            print(f"  validate(last iter): {test_infra.pcc_message}")
        t_val1 = time.perf_counter()

        pre_sec = t_pre1 - t_pre0
        # Open mesh + infra + pipeline build + compile + prealloc (excludes image tensor prep).
        setup_sec = (t_compile1 - wall0) - pre_sec
        validate_sec = t_val1 - t_val0
        device_sec = sum(device_secs) / len(device_secs)
        device_sec_min = min(device_secs) if len(device_secs) > 1 else None
        device_sec_max = max(device_secs) if len(device_secs) > 1 else None

        return {
            "setup_sec": setup_sec,
            "pre_sec": pre_sec,
            "pre_transforms_sec": pre_transforms_sec,
            "pre_golden_cpu_sec": pre_golden_sec,
            "device_sec": device_sec,
            "device_sec_min": device_sec_min,
            "device_sec_max": device_sec_max,
            "device_measured_iters": args.tt_measured_iters,
            "post_sec": validate_sec,
            "to_torch_sec": None,
            "postprocess_sec": validate_sec,
            "save_sec": 0.0,
            "total_sec": time.perf_counter() - wall0,
            "tt_device_label": "device(enqueue+pop_all)",
            "tt_pre_label": "pre(224+ImageNet+tensor)",
            "tt_setup_label": "setup(mesh+infra+compile)",
        }
    finally:
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:
                pass
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

    if args.backend == "tt":
        if args.tt_use_system_mesh and args.tt_force_single_device:
            raise ValueError("--tt-use-system-mesh cannot be used with --tt-force-single-device.")
        if args.tt_mesh_shape is None and not args.tt_use_system_mesh:
            args.tt_mesh_shape = [_DEFAULT_GALAXY_MESH_ROWS, _DEFAULT_GALAXY_MESH_COLS]
        if args.tt_use_system_mesh:
            import ttnn

            resolve_tt_mesh_and_batch(ttnn, args)

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
        if args.tt_model == "resnet50":
            timing = run_tt_resnet(args, image_bgr, out_dir, stem)
        else:
            timing = run_tt(args, image_bgr, out_dir, stem)
    else:
        timing = run_cpu(args, image_bgr, out_dir, stem)

    print_run_summary(out_dir, args.backend, timing, args.save_images)
    print("Done.")


if __name__ == "__main__":
    main()
