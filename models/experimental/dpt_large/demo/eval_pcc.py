# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..tt.config import DPTLargeConfig
from ..tt.fallback import DPTFallbackPipeline
from ..tt.perf_counters import PERF_COUNTERS, reset_perf_counters


def _to_numpy(x):
    # Avoid importing models.common.utility_functions (pulls in pytest).
    # Also avoid hard-depending on torch for PCC computation.
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def comp_pcc(golden, calculated, pcc: float = 0.99):
    """
    Minimal Pearson correlation checker for demo scripts.
    Returns: (passed, pcc_value)
    """
    a = _to_numpy(golden).astype(np.float64).ravel()
    b = _to_numpy(calculated).astype(np.float64).ravel()

    if a.size == 0 or b.size == 0:
        return False, float("nan")
    if a.shape != b.shape:
        return False, float("nan")

    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        same = bool(np.allclose(a, b, atol=0.0, rtol=0.0))
        return same, 1.0 if same else 0.0

    val = float(np.dot(a, b) / denom)
    return val >= float(pcc), val


def _collect_images(args) -> list[str]:
    if args.image:
        return [args.image]
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = sorted(str(p) for p in Path(args.images_dir).iterdir() if p.suffix.lower() in exts)
        if args.limit is not None:
            imgs = imgs[: int(args.limit)]
        return imgs
    raise ValueError("Provide either --image or --images-dir.")


def _fps_from_ms(latency_ms: float) -> float:
    return 1000.0 / latency_ms if latency_ms > 0 else 0.0


def _resolve_dp_and_batch_size(args, use_tt: bool) -> tuple[int, int]:
    dp = int(args.dp)
    if dp < 1:
        raise SystemExit("--dp must be >= 1")
    if args.batch_size is not None and int(args.batch_size) < 1:
        raise SystemExit("--batch-size must be >= 1")

    if dp not in (1, 2):
        raise SystemExit("Only dp=1 or dp=2 is currently supported")
    if dp > 1 and not use_tt:
        raise SystemExit("Data parallel mode requires --tt-run")
    if dp > 1 and str(args.device).lower() != "wormhole_n300":
        raise SystemExit("Data parallel mode (dp=2) is currently supported only on --device wormhole_n300")

    batch_size = int(args.batch_size) if args.batch_size is not None else (dp if (use_tt and dp > 1) else 1)
    if use_tt and dp > 1 and batch_size != dp:
        raise SystemExit("--batch-size must equal --dp when --dp > 1 in TT mode")

    return dp, batch_size


def _open_dp_mesh_device(effective_dp: int, num_cq: int):
    import ttnn  # type: ignore

    mesh_shape = ttnn.MeshShape(1, int(effective_dp))
    common_kwargs = dict(
        l1_small_size=24576,
        trace_region_size=8 * 1024 * 1024,
        num_command_queues=int(num_cq),
    )
    # Keep N300 mesh dispatch aligned with standard wormhole demo fixtures.
    try:
        dispatch_core_type = getattr(getattr(ttnn, "device", None), "DispatchCoreType", None)
        if dispatch_core_type is not None and hasattr(dispatch_core_type, "WORKER"):
            common_kwargs["dispatch_core_type"] = dispatch_core_type.WORKER
    except Exception:
        pass

    def _call_open(open_fn):
        kwargs = dict(common_kwargs)
        try:
            return open_fn(kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "dispatch_core_type" in msg and "dispatch_core_type" in kwargs:
                kwargs.pop("dispatch_core_type", None)
                return open_fn(kwargs)
            raise

    return _call_open(lambda kwargs: ttnn.open_mesh_device(mesh_shape=mesh_shape, **kwargs))


def _select_iteration_inputs(pixel_values_list: list, tt_inputs_host_list: list | None, batch_size: int):
    indices = [i % len(pixel_values_list) for i in range(int(batch_size))]
    iter_pixel_values = [pixel_values_list[i] for i in indices]
    iter_tt_inputs = None
    if tt_inputs_host_list is not None:
        iter_tt_inputs = [tt_inputs_host_list[i] for i in indices]
    return iter_pixel_values, iter_tt_inputs


def _run_single_tt_call(pipeline, pixel_values, tt_input_host):
    if tt_input_host is not None:
        return pipeline.forward_tt_host_tensor(tt_input_host, normalize=True)
    return pipeline.forward_pixel_values(pixel_values, normalize=True)


def _run_dp_batch(tt_pipelines: list, pixel_values_batch: list, tt_inputs_host_batch: list | None, executor) -> list:
    outputs = []
    worker_count = len(tt_pipelines)
    for start in range(0, len(pixel_values_batch), worker_count):
        end = min(start + worker_count, len(pixel_values_batch))
        futures = []
        for worker_idx, batch_idx in enumerate(range(start, end)):
            pipeline = tt_pipelines[worker_idx]
            tt_input_host = None if tt_inputs_host_batch is None else tt_inputs_host_batch[batch_idx]
            futures.append(executor.submit(_run_single_tt_call, pipeline, pixel_values_batch[batch_idx], tt_input_host))
        for future in futures:
            outputs.append(future.result())
    return outputs


def time_pipeline_single(pipeline, images: list[str], warmup: int, repeat: int, batch_size: int):
    prepare = None
    if hasattr(pipeline, "fallback") and hasattr(pipeline.fallback, "_prepare"):
        prepare = pipeline.fallback._prepare
    elif hasattr(pipeline, "_prepare"):
        prepare = pipeline._prepare

    pixel_values_list = None
    if prepare is not None and hasattr(pipeline, "forward_pixel_values"):
        pixel_values_list = [prepare(img) for img in images]
    if pixel_values_list is None or len(pixel_values_list) == 0:
        raise RuntimeError("Failed to create preprocessed inputs for timing")

    tt_inputs_host_list = None
    exec_mode = str(getattr(getattr(pipeline, "config", None), "tt_execution_mode", "eager")).lower()
    if hasattr(pipeline, "forward_tt_host_tensor") and exec_mode in ("trace", "trace_2cq"):
        import ttnn  # type: ignore

        tt_inputs_host_list = [
            ttnn.from_torch(pv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for pv in pixel_values_list
        ]

    iter_pixel_values, iter_tt_inputs = _select_iteration_inputs(pixel_values_list, tt_inputs_host_list, batch_size=batch_size)

    for _ in range(max(0, int(warmup))):
        if iter_tt_inputs is not None:
            _ = pipeline.forward_tt_host_tensor(iter_tt_inputs[0], normalize=True)
        else:
            _ = pipeline.forward_pixel_values(iter_pixel_values[0], normalize=True)

    # Steady-state guard: after warmup, do not allow any silent host fallbacks.
    if hasattr(pipeline, "config") and not bool(getattr(pipeline.config, "allow_cpu_fallback", True)):
        reset_perf_counters()

    timings_ms: list[float] = []
    last = None
    for _ in range(max(1, int(repeat))):
        start = time.perf_counter()
        if iter_tt_inputs is not None:
            for tth in iter_tt_inputs:
                last = pipeline.forward_tt_host_tensor(tth, normalize=True)
        else:
            for pv in iter_pixel_values:
                last = pipeline.forward_pixel_values(pv, normalize=True)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    total_ms_mean = float(np.mean(timings_ms))
    per_image_ms = total_ms_mean / max(1, len(iter_pixel_values))
    stats = {
        "repeat_total_ms": timings_ms,
        "total_ms_mean": total_ms_mean,
        "total_ms_std": float(np.std(timings_ms)),
        "num_images_per_iter": int(len(iter_pixel_values)),
        "per_image_ms": per_image_ms,
        "fps": _fps_from_ms(per_image_ms),
        "last_output": last,
        "fallback_counts": PERF_COUNTERS.snapshot(),
        "data_parallel_degree": 1,
    }
    if hasattr(pipeline, "config") and not bool(getattr(pipeline.config, "allow_cpu_fallback", True)):
        counts = stats["fallback_counts"]
        if int(counts.get("vit_backbone_fallback_count", 0)) != 0 or int(
            counts.get("reassembly_readout_fallback_count", 0)
        ) != 0 or int(
            counts.get("upsample_host_fallback_count", 0)
        ) != 0:
            raise RuntimeError(f"Unexpected TT host fallbacks in perf run: {counts}")
    return stats


def time_pipeline_dp(*, mesh_device, tt_pipelines: list, images: list[str], warmup: int, repeat: int, batch_size: int):
    if len(tt_pipelines) < 2:
        raise RuntimeError("time_pipeline_dp requires at least 2 TT pipelines")

    prepare = tt_pipelines[0].fallback._prepare
    pixel_values_list = [prepare(img) for img in images]
    if len(pixel_values_list) == 0:
        raise RuntimeError("Failed to create preprocessed inputs for timing")

    tt_inputs_host_list = None
    exec_mode = str(getattr(getattr(tt_pipelines[0], "config", None), "tt_execution_mode", "eager")).lower()
    if exec_mode not in ("trace", "trace_2cq"):
        raise RuntimeError("DP timing requires trace/trace_2cq execution mode")
    import ttnn  # type: ignore

    tt_inputs_host_list = [ttnn.from_torch(pv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for pv in pixel_values_list]

    iter_pixel_values, iter_tt_inputs = _select_iteration_inputs(pixel_values_list, tt_inputs_host_list, batch_size=batch_size)

    timings_ms: list[float] = []
    last = None
    assert iter_tt_inputs is not None

    from .submesh_trace_dp import SubmeshTraceDPExecutor

    dp_exec = SubmeshTraceDPExecutor(tt_pipelines=tt_pipelines, execution_mode=exec_mode)
    warmup_tt_inputs = [iter_tt_inputs[i % len(iter_tt_inputs)] for i in range(len(tt_pipelines))]
    dp_exec.prepare(warmup_tt_inputs)

    for _ in range(max(0, int(warmup))):
        _ = dp_exec.run(iter_tt_inputs, normalize=True)

    reset_perf_counters()

    for _ in range(max(1, int(repeat))):
        start = time.perf_counter()
        outs = dp_exec.run(iter_tt_inputs, normalize=True)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)
        if len(outs) > 0:
            last = outs[-1]
    # Traces are released by pipeline.close() in the caller.

    total_ms_mean = float(np.mean(timings_ms))
    per_image_ms = total_ms_mean / max(1, len(iter_pixel_values))
    stats = {
        "repeat_total_ms": timings_ms,
        "total_ms_mean": total_ms_mean,
        "total_ms_std": float(np.std(timings_ms)),
        "num_images_per_iter": int(len(iter_pixel_values)),
        "per_image_ms": per_image_ms,
        "fps": _fps_from_ms(per_image_ms),
        "last_output": last,
        "fallback_counts": PERF_COUNTERS.snapshot(),
        "data_parallel_degree": len(tt_pipelines),
        "stage_breakdown_ms_per_worker": [
            dict(p.last_perf) for p in tt_pipelines if getattr(p, "last_perf", None) is not None
        ],
    }
    counts = stats["fallback_counts"]
    if int(counts.get("vit_backbone_fallback_count", 0)) != 0 or int(
        counts.get("reassembly_readout_fallback_count", 0)
    ) != 0 or int(
        counts.get("upsample_host_fallback_count", 0)
    ) != 0:
        raise RuntimeError(f"Unexpected TT host fallbacks in perf run: {counts}")
    return stats


def main():
    parser = argparse.ArgumentParser("DPT-Large TTNN PCC + FPS evaluator")
    parser.add_argument("--image", type=str, default=None, help="Single image path.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of images (jpg/jpeg/png/bmp).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images from --images-dir.")

    parser.add_argument("--device", type=str, default="cpu", help="cpu|wormhole_n300|wormhole_n150|blackhole")
    parser.add_argument("--tt-run", action="store_true", help="Run TT pipeline and compare to CPU reference.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained HF weights (downloads).")
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false", help="Use random init (no download)."
    )
    parser.set_defaults(pretrained=True)

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images processed per timed iteration. Default is 1 for dp=1 and defaults to --dp for dp>1.",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data-parallel degree. Use dp=2 on wormhole_n300 to process two images concurrently.",
    )
    parser.add_argument(
        "--tt-execution-mode",
        type=str,
        default="eager",
        choices=("eager", "trace", "trace_2cq"),
        help="Execution mode for TT path. trace/trace_2cq execute a captured full-model trace (backbone+neck+head).",
    )
    parser.add_argument("--dump-json", type=str, default=None, help="Write a JSON summary.")
    args = parser.parse_args()

    images = _collect_images(args)
    if len(images) == 0:
        raise SystemExit(
            "No input images found. Supported extensions: .jpg/.jpeg/.png/.bmp "
            "(provide --image or a non-empty --images-dir)."
        )

    if args.tt_run and args.device == "cpu":
        raise SystemExit("--tt-run requires --device != cpu")

    use_tt = bool(args.tt_run)
    effective_dp, effective_batch_size = _resolve_dp_and_batch_size(args, use_tt=use_tt)

    # Always build a CPU reference pipeline (the thing we compare to for PCC).
    cfg_cpu = DPTLargeConfig(
        image_size=int(args.image_size),
        patch_size=16,
        device="cpu",
        allow_cpu_fallback=True,
        enable_tt_device=False,
        tt_device_reassembly=False,
        tt_device_fusion=False,
        tt_perf_encoder=False,
        tt_perf_neck=False,
    )
    cpu = DPTFallbackPipeline(config=cfg_cpu, pretrained=bool(args.pretrained), device="cpu")
    cpu_stats = time_pipeline_single(cpu, images, warmup=args.warmup, repeat=args.repeat, batch_size=effective_batch_size)

    result: dict[str, Any] = {
        "num_images": len(images),
        "image_size": int(args.image_size),
        "batch_size": int(effective_batch_size),
        "data_parallel_degree": int(effective_dp if use_tt else 1),
        "pretrained": bool(args.pretrained),
        "tt_execution_mode": str(args.tt_execution_mode) if bool(args.tt_run) else None,
        "cpu": {k: v for k, v in cpu_stats.items() if k != "last_output"},
    }

    if not args.tt_run:
        if args.dump_json:
            out = Path(args.dump_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    from ..tt.pipeline import DPTTTPipeline

    # PCC path: keep TT encoder active but route neck/head through HF modules.
    # This compares TT-encoded features against the strict HF reference without
    # introducing traced/full-TT neck/head approximation differences.
    cfg_tt_pcc = DPTLargeConfig(
        image_size=int(args.image_size),
        patch_size=16,
        device=str(args.device),
        allow_cpu_fallback=False,
        enable_tt_device=True,
        tt_device_reassembly=False,
        tt_device_fusion=False,
        tt_perf_encoder=True,
        tt_perf_neck=False,
        tt_approx_align_corners=False,
        tt_execution_mode="eager",
    )
    with DPTTTPipeline(config=cfg_tt_pcc, pretrained=bool(args.pretrained), device="cpu") as tt_pcc:
        pccs: list[float] = []
        pcc_pass_flags: list[bool] = []
        for img in images:
            depth_cpu = tt_pcc.fallback.forward(img, normalize=True)
            depth_tt = tt_pcc.forward(img, normalize=True)
            passed, pcc = comp_pcc(depth_cpu, depth_tt, pcc=0.99)
            pccs.append(float(pcc))
            pcc_pass_flags.append(bool(passed))

    # Perf path: full TT backbone+neck+head and traced execution mode.
    cfg_tt_perf = DPTLargeConfig(
        image_size=int(args.image_size),
        patch_size=16,
        device=str(args.device),
        allow_cpu_fallback=False,
        enable_tt_device=True,
        # Keep neck/head fully on TT device for practical hot path.
        tt_device_reassembly=True,
        tt_device_fusion=True,
        tt_perf_encoder=True,
        tt_perf_neck=True,
        tt_approx_align_corners=True,
        tt_execution_mode=str(args.tt_execution_mode),
    )

    tt_stats = None
    if effective_dp > 1:
        import ttnn  # type: ignore

        mesh_device = None
        tt_pipelines = []
        try:
            num_cq = 2 if str(args.tt_execution_mode).lower() == "trace_2cq" else 1
            mesh_device = _open_dp_mesh_device(effective_dp=effective_dp, num_cq=num_cq)
            try:
                submeshes = list(mesh_device.create_submeshes(ttnn.MeshShape(1, 1)))
            except TypeError:
                submeshes = list(mesh_device.create_submeshes(ttnn.MeshShape((1, 1))))
            if len(submeshes) < effective_dp:
                raise RuntimeError(f"Expected at least {effective_dp} submeshes but got {len(submeshes)}")

            for i in range(effective_dp):
                tt_pipelines.append(
                    DPTTTPipeline(
                        config=cfg_tt_perf,
                        pretrained=bool(args.pretrained),
                        device="cpu",
                        tt_device_override=submeshes[i],
                    )
                )
            tt_stats = time_pipeline_dp(
                mesh_device=mesh_device,
                tt_pipelines=tt_pipelines,
                images=images,
                warmup=args.warmup,
                repeat=args.repeat,
                batch_size=effective_batch_size,
            )
        finally:
            for pipeline in tt_pipelines:
                pipeline.close()
            if mesh_device is not None:
                try:
                    ttnn.close_mesh_device(mesh_device)
                except Exception:
                    pass
    else:
        with DPTTTPipeline(config=cfg_tt_perf, pretrained=bool(args.pretrained), device="cpu") as tt_perf:
            tt_stats = time_pipeline_single(
                tt_perf,
                images,
                warmup=args.warmup,
                repeat=args.repeat,
                batch_size=effective_batch_size,
            )
    assert tt_stats is not None

    result["tt"] = {k: v for k, v in tt_stats.items() if k != "last_output"}
    result["pcc_mode"] = "tt_encoder_with_hf_neck_head"
    result["pcc"] = {
        "mean": float(np.nanmean(pccs)) if len(pccs) else float("nan"),
        "min": float(np.nanmin(pccs)) if len(pccs) else float("nan"),
        "per_image": pccs,
        "threshold": 0.99,
        "all_pass": all(pcc_pass_flags) if len(pcc_pass_flags) else False,
    }

    if args.dump_json:
        out = Path(args.dump_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
