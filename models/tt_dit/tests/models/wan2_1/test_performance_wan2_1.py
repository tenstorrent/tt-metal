# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.wan.pipeline_wan2_1 import WanPipeline21

from ....utils.test import line_params, ring_params, ring_params_8k

# (label, width, height) — ordered small to large
RESOLUTIONS = [
    ("480p", 832, 480),
    ("720p", 1280, 720),
    ("1k", 1024, 1024),
    ("2k", 2048, 2048),
    # ("4k", 4096, 4096),
]

NUM_FRAMES = 1
NUM_INFERENCE_STEPS = 40
PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
        [(2, 4), (2, 4), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True],
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params_8k, ttnn.Topology.Ring, False],
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x4_sp0tp1",
        "bh_2x4_sp1tp0",
        "bh_2x4_sp0tp1",
        "wh_4x8_sp1tp0",
        "ring_bh_4x8_sp1tp0",
        "line_bh_4x8_sp1tp0",
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
    dynamic_load: dict,
    topology: ttnn.Topology,
    is_ci_env: bool,
    galaxy_type: str,
    is_fsdp: bool,
) -> None:
    """
    Sweep 480p / 720p / 1k / 2k / 4k at num_frames=1 (image-generation mode).
    Creates a fresh pipeline per resolution so latent buffers never collide.
    Uses run_warmup=False to skip the default 81-frame warmup (would OOM at 2k/4k).
    Prints a summary table at the end.
    """

    if galaxy_type == "4U":
        pytest.skip("4U is not supported for this test")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    results = {}  # label -> {"encoder": float, ...} | {"error": str}

    for label, width, height in RESOLUTIONS:
        logger.info(f"=== {label} ({width}x{height}) ===")
        pipeline = None
        try:
            pipeline = WanPipeline21.create_pipeline(
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                num_links=num_links,
                dynamic_load=dynamic_load,
                topology=topology,
                is_fsdp=is_fsdp,
                target_height=height,
                target_width=width,
                num_frames=NUM_FRAMES,
                run_warmup=False,  # We warmup manually with num_frames=1 below
            )

            logger.info(f"  Warmup {label}...")
            with torch.no_grad():
                pipeline(
                    prompt=PROMPT,
                    height=height,
                    width=width,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    output_type="uint8",
                )
            ttnn.synchronize_device(mesh_device)

            logger.info(f"  Measuring {label}...")
            profiler = BenchmarkProfiler()
            with profiler("run", iteration=0):
                with torch.no_grad():
                    result = pipeline(
                        prompt=PROMPT,
                        height=height,
                        width=width,
                        num_frames=NUM_FRAMES,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=5.0,
                        profiler=profiler,
                        profiler_iteration=0,
                        seed=42,
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

            # Save image (first frame of the single-frame output)
            if not is_ci_env and int(ttnn.distributed_context_get_rank()) == 0:
                frames = result.frames if hasattr(result, "frames") else result[0]
                # frames shape: (B, T, H, W, C) -> take batch 0, frame 0
                frame = frames[0][0]
                Image.fromarray(frame).save(f"wan2_1_{label}_{width}x{height}.png")
                logger.info(f"  Saved wan2_1_{label}_{width}x{height}.png")

        except Exception as e:
            logger.warning(f"  {label} FAILED: {e}")
            results[label] = {"error": str(e)}

        finally:
            del pipeline
            gc.collect()

    _print_table(results, mesh_shape, sp_factor, tp_factor)


def _print_table(results, mesh_shape, sp_factor, tp_factor):
    col_widths = [12, 12, 14, 10, 12]
    headers = ["Resolution", "Encoder(s)", "Denoising(s)", "VAE(s)", "Total(s)"]
    divider = "-+-".join("-" * w for w in col_widths)
    total_w = sum(col_widths) + 3 * (len(col_widths) - 1)

    title = (
        f"WAN2.1 RESOLUTION SWEEP | mesh={mesh_shape} | sp={sp_factor} | tp={tp_factor} "
        f"| steps={NUM_INFERENCE_STEPS} | frames={NUM_FRAMES}"
    )
    print()
    print(title)
    print("=" * total_w)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print(divider)

    for label, _, _ in RESOLUTIONS:
        r = results.get(label)
        if r is None:
            cells = [label, "N/A", "N/A", "N/A", "N/A"]
        elif "error" in r:
            err_w = total_w - col_widths[0] - 3
            err_msg = r["error"][:err_w]
            print(f"{label.ljust(col_widths[0])} | FAILED: {err_msg}")
            continue
        else:
            cells = [
                label,
                f"{r['encoder']:.3f}",
                f"{r['denoising']:.3f}",
                f"{r['vae']:.3f}",
                f"{r['total']:.3f}",
            ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths)))

    print("=" * total_w)
