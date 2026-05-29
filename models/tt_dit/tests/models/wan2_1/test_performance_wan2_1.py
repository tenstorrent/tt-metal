# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan2_1 import WanPipeline21

from ....utils.test import line_params


def _build_pipeline(
    mesh_device,
    *,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    height,
    width,
    num_frames,
    is_fsdp,
    encoder_tp_axis=None,
):
    """Construct WanPipeline21 directly, bypassing the create_pipeline config lookup."""
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    encoder_tp_axis = tp_axis if encoder_tp_axis is None else encoder_tp_axis
    encoder_tp_factor = tuple(mesh_device.shape)[encoder_tp_axis]

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        width_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    encoder_parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis),
    )
    full_latent_T = (num_frames - 1) // 4 + 1

    return WanPipeline21(
        mesh_device,
        parallel_config,
        vae_parallel_config,
        encoder_parallel_config,
        num_links,
        checkpoint_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        boundary_ratio=None,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        vae_t_chunk_size=full_latent_T,
        num_frames=num_frames,
        run_warmup=False,
    )


# (batch_size, height, width) — B x H x W as provided.
# All batch=1 resolutions enabled. batch>1 is not yet supported by the model
# (see transformer/pipeline batch=1 assumptions), so those remain disabled.
SHAPES = [
    (1, 1536, 2048),
    (1, 1152, 2048),
    (1, 2048, 1152),
    (1, 2048, 2048),
    (1, 2048, 1536),
    (1, 1344, 2016),
    (1, 1152, 1536),
    (1, 960, 1696),
    (1, 2016, 1344),
    (1, 1536, 1536),
    (1, 1696, 960),
    (1, 1024, 1280),
    (1, 1536, 1152),
    (1, 768, 1024),
    (1, 1088, 1632),
    (1, 2048, 880),
    (1, 1632, 1088),
    (1, 1024, 1024),
    (1, 1680, 720),
    (1, 1280, 720),
    (1, 1080, 1920),
    (1, 576, 1024),
    (1, 832, 1216),
    (1, 1344, 768),
    (1, 768, 1344),
    (1, 1152, 896),
    (1, 1344, 2048),
    (1, 2048, 1344),
]

NUM_FRAMES = 1
NUM_INFERENCE_STEPS = 20
PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

MAX_HEIGHT = max(height for _, height, _ in SHAPES)
MAX_WIDTH = max(width for _, _, width in SHAPES)

# Reserve a DRAM trace region so denoising steps replay from a captured trace
# instead of being dispatched op-by-op from host. The trace records one
# combined_step; its size scales with op count (constant across resolutions),
# not activation size, so a single fixed region serves every shape.
# Per-config trace region. A single captured trace is ~50-58 MB regardless of
# resolution, so the region must clear that to capture. But the region is carved
# from DRAM, shrinking the activation budget — which matters on the smaller 2x4
# mesh, where the largest shapes are already near the DRAM limit. So:
#   - 4x4: large bank, plenty of headroom -> 120 MB (captures everything).
#   - 2x4: tight -> 72 MB, enough to capture (~58 MB) while leaving activation
#     room for all but the very largest shapes (which OOM'd non-traced anyway).
# NOTE: a *failed* capture leaks its trace buffer into the next shape (cumulative
# get_trace_buffers_size), so the region must stay above the per-shape trace size.
TRACE_REGION_SIZE_4X4 = int(os.environ.get("WAN_TRACE_REGION_SIZE_4X4", 120_000_000))
TRACE_REGION_SIZE_2X4 = int(os.environ.get("WAN_TRACE_REGION_SIZE_2X4", 72_000_000))
traced_params_4x4 = {**line_params, "trace_region_size": TRACE_REGION_SIZE_4X4}
traced_params_2x4 = {**line_params, "trace_region_size": TRACE_REGION_SIZE_2X4}


def _save_output_images(
    *,
    frames,
    run_id: str,
    batch_size: int,
    height: int,
    width: int,
    output_dir: Path = Path("."),
) -> None:
    for batch_idx in range(batch_size):
        frame = frames[batch_idx][0]
        filename = f"wan2_1_{run_id}_batch{batch_size}_h{height}_w{width}_idx{batch_idx}.png"
        Image.fromarray(frame).save(output_dir / filename)
        logger.info(f"  Saved {filename}")


def _print_table(results, mesh_shape, sp_factor, tp_factor):
    col_widths = [14, 8, 12, 14, 14, 10, 12]
    headers = ["Shape(BxHxW)", "Tokens", "Encoder(s)", "Denoising(s)", "Step(s)", "VAE(s)", "Total(s)"]
    divider = "-+-".join("-" * w for w in col_widths)
    total_w = sum(col_widths) + 3 * (len(col_widths) - 1)

    title = (
        f"WAN2.1 SHAPE SWEEP | mesh={mesh_shape} | sp={sp_factor} | tp={tp_factor} "
        f"| steps={NUM_INFERENCE_STEPS} | frames={NUM_FRAMES}"
    )
    print()
    print(title)
    print("=" * total_w)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print(divider)

    for batch_size, height, width in SHAPES:
        label = f"{batch_size}x{height}x{width}"
        tokens = height * width // 256
        r = results.get(label)
        if r is None:
            cells = [label, str(tokens), "N/A", "N/A", "N/A", "N/A", "N/A"]
        elif "error" in r:
            err_w = total_w - col_widths[0] - col_widths[1] - 6
            err_msg = r["error"][:err_w]
            print(f"{label.ljust(col_widths[0])} | {str(tokens).ljust(col_widths[1])} | FAILED: {err_msg}")
            continue
        else:
            cells = [
                label,
                str(tokens),
                f"{r['encoder']:.3f}",
                f"{r['denoising']:.3f}",
                f"{r['denoising'] / NUM_INFERENCE_STEPS:.3f}",
                f"{r['vae']:.3f}",
                f"{r['total']:.3f}",
            ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths)))

    print("=" * total_w)


def _write_csv(results, csv_path: Path) -> None:
    fieldnames = [
        "batch",
        "height",
        "width",
        "shape",
        "tokens",
        "encoder_s",
        "denoising_s",
        "denoising_step_s",
        "vae_s",
        "total_s",
        "error",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for batch_size, height, width in SHAPES:
            label = f"{batch_size}x{height}x{width}"
            tokens = height * width // 256
            row = {
                "batch": batch_size,
                "height": height,
                "width": width,
                "shape": label,
                "tokens": tokens,
            }
            result = results.get(label)
            if result is None:
                row.update(
                    {
                        "encoder_s": "",
                        "denoising_s": "",
                        "denoising_step_s": "",
                        "vae_s": "",
                        "total_s": "",
                        "error": "missing",
                    }
                )
            elif "error" in result:
                row.update(
                    {
                        "encoder_s": "",
                        "denoising_s": "",
                        "denoising_step_s": "",
                        "vae_s": "",
                        "total_s": "",
                        "error": result["error"],
                    }
                )
            else:
                row.update(
                    {
                        "encoder_s": f"{result['encoder']:.6f}",
                        "denoising_s": f"{result['denoising']:.6f}",
                        "denoising_step_s": f"{result['denoising'] / NUM_INFERENCE_STEPS:.6f}",
                        "vae_s": f"{result['vae']:.6f}",
                        "total_s": f"{result['total']:.6f}",
                        "error": "",
                    }
                )
            writer.writerow(row)

    logger.info(f"Wrote performance CSV to {csv_path}")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, encoder_tp_axis",
    [
        [(4, 8), (4, 4), 1, 0, 4, False, traced_params_4x4, ttnn.Topology.Linear, True, None],
        [(4, 8), (2, 4), 0, 1, 4, False, traced_params_2x4, ttnn.Topology.Linear, True, None],
        [(4, 8), (2, 4), 1, 0, 4, False, traced_params_2x4, ttnn.Topology.Linear, True, 1],
    ],
    ids=[
        "wh_4x4_sp1tp0",
        "wh_2x4_sp0tp1",
        "wh_2x4_sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
def test_resolution_sweep(
    *,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
    encoder_tp_axis: int | None,
    is_ci_env: bool,
    galaxy_type: str,
) -> None:
    """
    Sweep all shapes in SHAPES at num_frames=1 (image-generation mode). Opens the
    parent mesh from the fixture and carves out the requested submesh. Creates one
    pipeline sized to the maximum height/width in SHAPES and reuses it across all
    resolutions.
    """
    if galaxy_type == "4U":
        pytest.skip("4U is not supported for this test")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    run_id = f"{'x'.join(str(d) for d in mesh_shape)}_sp{sp_factor}tp{tp_factor}"
    perf_csv_path = Path(f"wan2_1_{run_id}_shape_sweep_perf.csv")

    pipeline = _build_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        height=MAX_HEIGHT,
        width=MAX_WIDTH,
        num_frames=NUM_FRAMES,
        is_fsdp=is_fsdp,
        encoder_tp_axis=encoder_tp_axis,
    )

    results = {}

    # Optional debug subset: WAN_SWEEP_SHAPES="1x1152x2048,1x768x1024"
    sweep_shapes = SHAPES
    shape_filter = os.environ.get("WAN_SWEEP_SHAPES")
    if shape_filter:
        wanted = {s.strip() for s in shape_filter.split(",") if s.strip()}
        sweep_shapes = [s for s in SHAPES if f"{s[0]}x{s[1]}x{s[2]}" in wanted]
        logger.info(f"WAN_SWEEP_SHAPES set; running {len(sweep_shapes)} shape(s): {sorted(wanted)}")

    for batch_size, height, width in sweep_shapes:
        label = f"{batch_size}x{height}x{width}"
        prompts = [PROMPT] * batch_size
        logger.info(f"=== {label} ===")
        try:
            # Eager (non-traced) warmup first. This allocates all persistent
            # buffers — notably the CCL ping-pong buffers, which are created
            # lazily via host->device writes inside combined_step. Those writes
            # are illegal during trace capture, so they must happen here, before
            # the traced warmup captures the trace.
            logger.info(f"  Eager warmup {label}...")
            with torch.no_grad():
                pipeline(
                    prompt=prompts,
                    height=height,
                    width=width,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    traced=False,
                    output_type="uint8",
                )
            ttnn.synchronize_device(mesh_device)

            # Traced warmup: captures the trace (first combined_step) and replays
            # it (second). The measured run below then replays instead of
            # dispatching denoising steps op-by-op from host.
            logger.info(f"  Traced warmup {label}...")
            with torch.no_grad():
                pipeline(
                    prompt=prompts,
                    height=height,
                    width=width,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    traced=True,
                    output_type="uint8",
                )
            ttnn.synchronize_device(mesh_device)

            logger.info(f"  Measuring {label}...")
            profiler = BenchmarkProfiler()
            with profiler("run", iteration=0):
                with torch.no_grad():
                    result = pipeline(
                        prompt=prompts,
                        height=height,
                        width=width,
                        num_frames=NUM_FRAMES,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=5.0,
                        profiler=profiler,
                        profiler_iteration=0,
                        seed=42,
                        traced=True,
                        output_type="uint8",
                    )
                ttnn.synchronize_device(mesh_device)

            results[label] = {
                "encoder": profiler.get_duration("encoder", 0),
                "denoising": profiler.get_duration("denoising", 0),
                "vae": profiler.get_duration("vae", 0),
                "total": profiler.get_duration("run", 0),
            }
            logger.info(
                f"  {label}: total={results[label]['total']:.2f}s  "
                f"denoise={results[label]['denoising']:.2f}s  "
                f"vae={results[label]['vae']:.2f}s"
            )

            if not is_ci_env and int(ttnn.distributed_context_get_rank()) == 0:
                frames = result.frames if hasattr(result, "frames") else result[0]
                _save_output_images(
                    frames=frames,
                    run_id=run_id,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )

        except Exception as e:
            logger.warning(f"  {label} FAILED: {e}")
            results[label] = {"error": str(e)}
        finally:
            # Release this resolution's trace before moving to the next shape, so
            # the next warmup recaptures and the freed shape-dependent buffers
            # (CCL / latent / solver) aren't still referenced by a live trace.
            pipeline.release_traces()

    _print_table(results, mesh_shape, sp_factor, tp_factor)
    if int(ttnn.distributed_context_get_rank()) == 0:
        _write_csv(results, perf_csv_path)


def _build_cfg_pipeline(parent_mesh, *, cfg_axis, sp_axis, tp_axis, num_links, topology, height, width, num_frames):
    """Build a CFG-parallel WanPipeline21 on the full parent mesh.

    The parent (e.g. 4x8) is split into two submeshes along cfg_axis (-> two 4x4).
    sp/tp factors are taken from the SUBMESH shape; cfg_parallel splits the parent.
    """
    parent_shape = tuple(parent_mesh.shape)
    submesh_shape = list(parent_shape)
    submesh_shape[cfg_axis] //= 2
    sp_factor = submesh_shape[sp_axis]
    tp_factor = submesh_shape[tp_axis]

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=2, mesh_axis=cfg_axis),
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        width_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    encoder_parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
    )
    full_latent_T = (num_frames - 1) // 4 + 1

    return WanPipeline21(
        parent_mesh,
        parallel_config,
        vae_parallel_config,
        encoder_parallel_config,
        num_links,
        checkpoint_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        boundary_ratio=None,
        dynamic_load=False,
        topology=topology,
        is_fsdp=True,
        height=height,
        width=width,
        vae_t_chunk_size=full_latent_T,
        num_frames=num_frames,
        run_warmup=False,
    )


CFG_SHAPES = [(1, 768, 1024), (1, 1152, 2048)]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), line_params]],
    ids=["wh_4x8_cfg2"],
    indirect=["mesh_device", "device_params"],
)
def test_cfg_parallel(*, mesh_device: ttnn.MeshDevice, is_ci_env: bool, galaxy_type: str) -> None:
    """CFG parallelism: split the 4x8 parent into two 4x4 submeshes (uncond/cond)."""
    if galaxy_type == "4U":
        pytest.skip("4U is not supported for this test")

    only = os.environ.get("WAN_SWEEP_SHAPES")
    shapes = CFG_SHAPES
    if only:
        wanted = {s.strip() for s in only.split(",")}
        shapes = [s for s in CFG_SHAPES if f"{s[0]}x{s[1]}x{s[2]}" in wanted] or CFG_SHAPES

    pipeline = _build_cfg_pipeline(
        mesh_device,
        cfg_axis=1,
        sp_axis=1,
        tp_axis=0,
        num_links=4,
        topology=ttnn.Topology.Linear,
        height=max(h for _, h, _ in shapes),
        width=max(w for _, _, w in shapes),
        num_frames=NUM_FRAMES,
    )

    # Construction smoke check: two submeshes, two transformers on distinct devices.
    assert pipeline.cfg_factor == 2
    assert len(pipeline.cfg_dit) == 2
    assert len(pipeline.cfg_submeshes) == 2
    logger.info("CFG-parallel pipeline constructed: 2 submeshes, 2 transformers.")

    for batch_size, height, width in shapes:
        label = f"{batch_size}x{height}x{width}"
        cfg_steps = int(os.environ.get("WAN_CFG_STEPS", NUM_INFERENCE_STEPS))
        logger.info(f"=== CFG {label} ({cfg_steps} steps) ===")
        with torch.no_grad():
            result = pipeline(
                prompt=[PROMPT] * batch_size,
                height=height,
                width=width,
                num_frames=NUM_FRAMES,
                num_inference_steps=cfg_steps,
                guidance_scale=5.0,
                seed=42,
                output_type="uint8",
            )
        # NOTE: do NOT synchronize the PARENT mesh here. Once it is split into
        # submeshes (create_submeshes), the parent and child share per-physical-device
        # cq0 completion/event state; a synchronize/record-event on the parent waits on
        # a device completion that never arrives -> hang. Synchronize the submeshes.
        for sm in pipeline.cfg_submeshes or [mesh_device]:
            ttnn.synchronize_device(sm)
        logger.info(f"  CFG {label} done")
        if not is_ci_env and int(ttnn.distributed_context_get_rank()) == 0:
            frames = result.frames if hasattr(result, "frames") else result[0]
            _save_output_images(frames=frames, run_id="4x8_cfg2", batch_size=batch_size, height=height, width=width)
