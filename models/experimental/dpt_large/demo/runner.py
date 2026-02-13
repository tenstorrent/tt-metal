# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..tt.config import DPTLargeConfig
from ..tt.fallback import DPTFallbackPipeline
from ..tt.perf_counters import PERF_COUNTERS, reset_perf_counters


def _collect_images(args) -> list[str]:
    if args.image:
        return [args.image]
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(str(p) for p in Path(args.images_dir).iterdir() if p.suffix.lower() in exts)
    raise ValueError("Provide either --image or --images-dir.")


def _save_depth_color(depth: np.ndarray, path: str):
    depth_min = depth.min()
    depth_max = depth.max()
    norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    img = Image.fromarray((norm.squeeze() * 255).astype(np.uint8))
    img.save(path)


def _stage_breakdown_to_seconds(stage_breakdown_ms: dict | None) -> dict:
    if not isinstance(stage_breakdown_ms, dict):
        return {}
    converted = {}
    for key, value in stage_breakdown_ms.items():
        if isinstance(value, (int, float)) and key.endswith("_ms"):
            converted[key[: -len("_ms")] + "_s"] = float(value) / 1000.0
        else:
            converted[key] = value
    return converted


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
    if len(pixel_values_list) == 0:
        raise RuntimeError("No preprocessed inputs available")
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


def _aggregate_stage_breakdown(stage_breakdowns: list[dict]) -> dict:
    if len(stage_breakdowns) == 0:
        return {}

    aggregated: dict[str, float | str] = {}
    numeric_keys: set[str] = set()
    for breakdown in stage_breakdowns:
        for key, value in breakdown.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    for key in sorted(numeric_keys):
        values = [float(b[key]) for b in stage_breakdowns if isinstance(b.get(key), (int, float))]
        if len(values) > 0:
            aggregated[key] = float(np.mean(values))

    for key in ("mode", "execution_mode", "effective_execution_mode", "requested_execution_mode"):
        values = [str(b.get(key)) for b in stage_breakdowns if b.get(key) is not None]
        if len(values) > 0 and all(v == values[0] for v in values):
            aggregated[key] = values[0]

    return aggregated


def main():
    parser = argparse.ArgumentParser("DPT-Large TTNN runner")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of images.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|wormhole_n300|wormhole_n150|blackhole")
    parser.add_argument("--tt-run", action="store_true", help="Run TT pipeline (requires --device != cpu).")
    parser.add_argument("--image-size", type=int, default=384, help="Square model input size.")
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
    parser.add_argument("--dump-depth", type=str, default=None)
    parser.add_argument("--dump-depth-color", type=str, default=None)
    parser.add_argument("--dump-perf", type=str, default=None)
    parser.add_argument(
        "--dump-perf-header",
        type=str,
        default=None,
        help="Optional explicit path for the perf header JSON (defaults to <dump-perf>_header.json).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--tt-execution-mode",
        type=str,
        default="eager",
        choices=("eager", "trace", "trace_2cq"),
        help="Execution mode for TT path. trace/trace_2cq execute a captured full-model trace (backbone+neck+head).",
    )
    args = parser.parse_args()

    images = _collect_images(args)
    if len(images) == 0:
        raise SystemExit(
            "No input images found. Supported extensions: .jpg/.jpeg/.png/.bmp "
            "(provide --image or a non-empty --images-dir)."
        )

    if args.tt_run and args.device == "cpu":
        raise SystemExit("--tt-run requires --device != cpu")
    if not args.tt_run and args.device != "cpu":
        raise SystemExit("--device must be 'cpu' unless --tt-run is set")

    use_tt = bool(args.tt_run)
    effective_dp, effective_batch_size = _resolve_dp_and_batch_size(args, use_tt=use_tt)

    config = DPTLargeConfig(
        image_size=args.image_size,
        patch_size=16,
        device=args.device,
        allow_cpu_fallback=not use_tt,
        enable_tt_device=use_tt,
        # Keep the full neck/head hot path on device in TT mode.
        tt_device_reassembly=use_tt,
        tt_device_fusion=use_tt,
        tt_perf_encoder=use_tt,
        tt_perf_neck=use_tt,
        tt_approx_align_corners=use_tt,
        tt_execution_mode=str(args.tt_execution_mode),
    )

    tt_pipelines = []
    cpu_pipeline = None
    mesh_device = None
    dp_executor = None
    try:
        if use_tt:
            from ..tt.pipeline import DPTTTPipeline

            if effective_dp > 1:
                import ttnn  # type: ignore

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
                            config=config,
                            device="cpu",
                            tt_device_override=submeshes[i],
                        )
                    )
            else:
                tt_pipelines.append(DPTTTPipeline(config=config, device="cpu"))

            cpu_pipeline = tt_pipelines[0].fallback
        else:
            cpu_pipeline = DPTFallbackPipeline(config=config, device="cpu")

        assert cpu_pipeline is not None
        pixel_values_list = [cpu_pipeline._prepare(img) for img in images]
        tt_inputs_host_list = None
        if use_tt and str(args.tt_execution_mode).lower() in ("trace", "trace_2cq"):
            import ttnn  # type: ignore

            # Host-side TT tensors; copied into a persistent device input buffer for trace execution.
            tt_inputs_host_list = [
                ttnn.from_torch(pv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for pv in pixel_values_list
            ]

        iter_pixel_values, iter_tt_inputs = _select_iteration_inputs(
            pixel_values_list=pixel_values_list,
            tt_inputs_host_list=tt_inputs_host_list,
            batch_size=effective_batch_size,
        )

        stage_breakdowns = []
        timings = []
        depth = None

        if use_tt and effective_dp > 1:
            requested_exec_mode = str(args.tt_execution_mode).lower()
            if requested_exec_mode not in {"trace", "trace_2cq"}:
                raise SystemExit("--dp > 1 currently requires --tt-execution-mode trace or trace_2cq")
            if iter_tt_inputs is None:
                raise RuntimeError("DP trace mode requires host TT inputs but none were prepared")

            from .submesh_trace_dp import SubmeshTraceDPExecutor

            # Capture one full trace per submesh device, then enqueue both traces
            # non-blocking to allow device-side overlap without Python threading.
            dp_executor = SubmeshTraceDPExecutor(tt_pipelines=tt_pipelines, execution_mode=requested_exec_mode)
            warmup_tt_inputs = [iter_tt_inputs[i % len(iter_tt_inputs)] for i in range(effective_dp)]
            dp_executor.prepare(warmup_tt_inputs)

            for _ in range(max(0, int(args.warmup))):
                _ = dp_executor.run(iter_tt_inputs, normalize=True)

            reset_perf_counters()

            for _ in range(max(1, int(args.repeat))):
                start = time.perf_counter()
                outs = dp_executor.run(iter_tt_inputs, normalize=True)
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)
                if len(outs) > 0:
                    depth = outs[-1]
        else:
            # Warmup around the selected pipeline only.
            for _ in range(max(0, int(args.warmup))):
                if use_tt:
                    assert len(tt_pipelines) == 1
                    if iter_tt_inputs is not None:
                        _ = tt_pipelines[0].forward_tt_host_tensor(iter_tt_inputs[0], normalize=True)
                    else:
                        _ = tt_pipelines[0].forward_pixel_values(iter_pixel_values[0], normalize=True)
                else:
                    assert cpu_pipeline is not None
                    _ = cpu_pipeline.forward_pixel_values(iter_pixel_values[0], normalize=True)

            if use_tt:
                # Steady-state guard: after warmup, do not allow any silent host fallbacks.
                reset_perf_counters()

            for _ in range(max(1, int(args.repeat))):
                start = time.perf_counter()
                for i in range(len(iter_pixel_values)):
                    if use_tt:
                        assert len(tt_pipelines) == 1
                        if iter_tt_inputs is not None:
                            depth = tt_pipelines[0].forward_tt_host_tensor(iter_tt_inputs[i], normalize=True)
                        else:
                            depth = tt_pipelines[0].forward_pixel_values(iter_pixel_values[i], normalize=True)
                    else:
                        assert cpu_pipeline is not None
                        depth = cpu_pipeline.forward_pixel_values(iter_pixel_values[i], normalize=True)
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)

        for pipeline in tt_pipelines:
            if getattr(pipeline, "last_perf", None) is not None:
                stage_breakdowns.append(dict(pipeline.last_perf))

        if depth is None:
            raise RuntimeError("No images were processed (check --image/--images-dir input).")

        latency_ms_mean = float(np.mean(timings))
        latency_ms_std = float(np.std(timings))
        num_images_per_iter = max(1, len(iter_pixel_values))
        per_image_ms_mean = latency_ms_mean / float(num_images_per_iter)
        per_image_ms_std = latency_ms_std / float(num_images_per_iter)
        fps = 1000.0 / per_image_ms_mean if per_image_ms_mean > 0 else 0.0
        inference_time_s = latency_ms_mean / 1000.0
        inference_time_std_s = latency_ms_std / 1000.0
        throughput_iter_per_s = (1.0 / inference_time_s) if inference_time_s > 0 else 0.0
        first_run_s = (float(timings[0]) / 1000.0) if len(timings) > 0 else inference_time_s
        compile_time_s = max(first_run_s - inference_time_s, 0.0)

        perf = dict(
            mode="tt" if use_tt else "cpu",
            tt_execution_mode=str(args.tt_execution_mode) if use_tt else "cpu",
            latency_ms_mean=latency_ms_mean,
            latency_ms_std=latency_ms_std,
            total_ms=latency_ms_mean,
            num_images_per_iter=num_images_per_iter,
            per_image_ms_mean=per_image_ms_mean,
            per_image_ms_std=per_image_ms_std,
            fps=fps,
            inference_time_s=inference_time_s,
            inference_time_std_s=inference_time_std_s,
            throughput_iter_per_s=throughput_iter_per_s,
            first_run_s=first_run_s,
            compile_time_s=compile_time_s,
            device=args.device,
            dtype="bfloat16",
            input_h=args.image_size,
            input_w=args.image_size,
            batch_size=num_images_per_iter,
            per_chip_batch_size=1,
            data_parallel_degree=effective_dp if use_tt else 1,
            model_name="dpt-large",
            setting=f"{'tt' if use_tt else 'cpu'}-{args.image_size}x{args.image_size}-b{num_images_per_iter}-dp{(effective_dp if use_tt else 1)}",
            modules=["backbone", "reassembly", "fusion_head"] if use_tt else ["cpu_fallback"],
        )

        if len(stage_breakdowns) > 0:
            perf["stage_breakdown_ms"] = _aggregate_stage_breakdown(stage_breakdowns)
            perf["stage_breakdown_ms_per_worker"] = stage_breakdowns
            perf["stage_breakdown_s"] = _stage_breakdown_to_seconds(perf["stage_breakdown_ms"])
        perf["fallback_counts"] = PERF_COUNTERS.snapshot()

        if use_tt:
            counts = perf["fallback_counts"]
            if int(counts.get("vit_backbone_fallback_count", 0)) != 0 or int(
                counts.get("reassembly_readout_fallback_count", 0)
            ) != 0 or int(
                counts.get("upsample_host_fallback_count", 0)
            ) != 0:
                raise RuntimeError(f"Unexpected TT host fallbacks in perf run: {counts}")

        if args.dump_depth:
            Path(args.dump_depth).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.dump_depth, depth)
        if args.dump_depth_color:
            Path(args.dump_depth_color).parent.mkdir(parents=True, exist_ok=True)
            _save_depth_color(depth, args.dump_depth_color)

        header = dict(
            model=dict(
                name="dpt-large",
                type="depth_estimation",
                backbone="vit-large",
                num_layers=24,
                hidden_size=1024,
                num_heads=16,
                intermediate_size=4096,
            ),
            input_shape=[num_images_per_iter, 3, args.image_size, args.image_size],
            patch_size=16,
            num_tokens=(args.image_size // 16) * (args.image_size // 16) + 1,
            device=args.device,
            dtype="bfloat16",
            tt_execution_mode=perf.get("tt_execution_mode", "unknown"),
            mode=perf.get("mode", "unknown"),
            data_parallel_degree=perf.get("data_parallel_degree", 1),
            latency_ms=perf.get("total_ms", 0),
            inference_time_s=perf.get("inference_time_s", 0.0),
            throughput_iter_per_s=perf.get("throughput_iter_per_s", 0.0),
            first_run_s=perf.get("first_run_s", 0.0),
            compile_time_s=perf.get("compile_time_s", 0.0),
            fps=perf.get("fps", 0),
            stage_breakdown_ms=perf.get("stage_breakdown_ms", {}),
            stage_breakdown_s=perf.get("stage_breakdown_s", {}),
            fallback_counts=perf.get("fallback_counts", {}),
        )

        if args.dump_perf:
            Path(args.dump_perf).parent.mkdir(parents=True, exist_ok=True)
            Path(args.dump_perf).write_text(json.dumps(perf, indent=2))

        header_path = None
        if args.dump_perf_header:
            header_path = Path(args.dump_perf_header)
        elif args.dump_perf:
            header_path = Path(args.dump_perf).with_name(Path(args.dump_perf).stem + "_header.json")

        if header_path is not None:
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(json.dumps(header, indent=2))
    finally:
        if dp_executor is not None and hasattr(dp_executor, "close"):
            try:
                dp_executor.close()
            except Exception:
                pass
        for pipeline in tt_pipelines:
            if pipeline is not None and hasattr(pipeline, "close"):
                pipeline.close()
        if mesh_device is not None:
            try:
                import ttnn  # type: ignore

                ttnn.close_mesh_device(mesh_device)
            except Exception:
                pass


if __name__ == "__main__":
    main()
