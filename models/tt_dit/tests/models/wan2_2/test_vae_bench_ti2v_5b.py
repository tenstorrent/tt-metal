# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VAE-decode-only benchmark and host op profile for Wan2.2 TI2V-5B.

The pipeline perf test measures the decode only as one section of a ~17 s run. This
isolates it so a change can be evaluated in ~1 min per repeat instead of a full
generation, and adds the two measurements the project has never had:

1.  ``t_chunk_size`` A/B on one fixed latent (production 7 vs full-T), which decides
    whether the 5B override of the mesh preset is still earning its place at 81 f.
2.  A host-side op profile (see ``_ttnn_host_profiler``) answering whether the decode
    is host-dispatch-bound -- i.e. whether tracing it is worth the work.

    WAN5B_BENCH_FRAMES=81 WAN5B_BENCH_REPS=3 WAN5B_BENCH_PROFILE=1 \
      pytest models/tt_dit/tests/models/wan2_2/test_vae_bench_ti2v_5b.py -sv
"""

import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests dir is not a package
from _ttnn_host_profiler import profile_ttnn_ops  # noqa: E402


def _parse_tchunks(raw, latent_t):
    out = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("none", "full", "fullt", "full-t"):
            out.append(None)
        else:
            out.append(int(tok))
    return [t for t in out if t is None or t < latent_t]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_vae_decode_bench_ti2v_5b(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_BENCH_HEIGHT", 704))
    width = int(os.environ.get("WAN5B_BENCH_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_BENCH_FRAMES", 81))
    reps = int(os.environ.get("WAN5B_BENCH_REPS", 3))
    output_type = os.environ.get("WAN5B_BENCH_OUTPUT", "uint8")
    do_profile = bool(int(os.environ.get("WAN5B_BENCH_PROFILE", 0)))
    # The warmup runs a full generation (denoise + trace capture) which this bench does
    # not measure; skipping it cuts ~2 min per iteration. The decode below does its own
    # warmup pass, so the VAE program cache is hot either way.
    run_warmup = bool(int(os.environ.get("WAN5B_BENCH_WARMUP", 1)))

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=height,
        width=width,
        num_frames=num_frames,
        run_warmup=run_warmup,
    )

    z_dim = pipeline._vae.config.z_dim
    torch.manual_seed(0)
    latents, _ = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        device="cpu",
    )
    latent_t = latents.shape[2]
    default_tchunk = pipeline._vae._t_chunk_size
    tchunks = _parse_tchunks(os.environ.get("WAN5B_BENCH_TCHUNKS", "7,none"), latent_t)

    logger.info(
        f"VAE bench: {height}x{width}, {num_frames}f (latent T={latent_t}), "
        f"output_type={output_type}, pipeline default t_chunk={default_tchunk}, "
        f"sweeping {tchunks}, reps={reps}"
    )

    def _decode(tchunk):
        orig = pipeline._vae._t_chunk_size
        pipeline._vae._t_chunk_size = tchunk
        try:
            t0 = time.perf_counter()
            out = pipeline._vae.decode(latents, output_type=output_type)
            dt = time.perf_counter() - t0
        finally:
            pipeline._vae._t_chunk_size = orig
        return dt, out

    if do_profile:
        # Profile first: a monkeypatch failure must not cost the t_chunk A/B below.
        prof_tchunk = tchunks[0]
        label = "fullT" if prof_tchunk is None else f"t{prof_tchunk}"
        orig = pipeline._vae._t_chunk_size
        pipeline._vae._t_chunk_size = prof_tchunk
        try:
            warm_dt, out = _decode(prof_tchunk)  # compile programs before measuring
            logger.info(f"VAE_PROFILE {label}: warmup {warm_dt:.4f}s, out shape {tuple(out.shape)}")
            del out
            with profile_ttnn_ops(mode="dispatch") as p:
                pipeline._vae.decode(latents, output_type=output_type)
            logger.info("\n" + p.result.report(f"VAE decode {label} {height}x{width} {num_frames}f"))
            with profile_ttnn_ops(mode="sync", device=mesh_device) as p2:
                pipeline._vae.decode(latents, output_type=output_type)
            logger.info("\n" + p2.result.report(f"VAE decode {label} {height}x{width} {num_frames}f"))
        except Exception as e:
            logger.error(f"VAE_PROFILE failed ({e!r}); continuing to the t_chunk sweep")
        finally:
            pipeline._vae._t_chunk_size = orig

    results = {}
    for tchunk in tchunks:
        label = "fullT" if tchunk is None else f"t{tchunk}"
        warm_dt, out = _decode(tchunk)  # first call compiles programs for these shapes
        logger.info(f"VAE_BENCH {label}: warmup {warm_dt:.4f}s, out shape {tuple(out.shape)}")
        del out
        times = []
        for i in range(reps):
            dt, out = _decode(tchunk)
            del out
            times.append(dt)
            logger.info(f"VAE_BENCH {label}: run {i} {dt:.4f}s")
        mean = sum(times) / len(times)
        spread = (max(times) - min(times)) / mean if mean else 0.0
        results[label] = (mean, min(times), max(times))
        logger.info(
            f"VAE_BENCH {label}: mean={mean:.4f}s min={min(times):.4f}s max={max(times):.4f}s spread={100*spread:.2f}%"
        )

    logger.info("=== VAE_BENCH SUMMARY ===")
    for label, (mean, lo, hi) in results.items():
        logger.info(f"VAE_BENCH_SUMMARY {label}: mean={mean:.4f}s min={lo:.4f}s max={hi:.4f}s")
    if len(results) > 1:
        base_label = "t7" if "t7" in results else list(results)[0]
        base = results[base_label][0]
        for label, (mean, _, _) in results.items():
            if label != base_label:
                logger.info(f"VAE_BENCH_DELTA {label} vs {base_label}: {100.0 * (mean - base) / base:+.2f}%")

    assert results, "no VAE decode configurations were measured"
