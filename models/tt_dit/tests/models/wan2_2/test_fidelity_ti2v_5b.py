# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Same-seed fidelity of a Wan2.2 TI2V-5B configuration against a saved reference (sprint 7).

Two entry points:

1. ``test_fidelity_ti2v_5b[bh_4x8]`` (device): generate once at a fixed seed with the configuration
   selected by the environment (``WAN5B_QUANT_CONFIG`` preset, pipeline defaults), save the denoised
   latents and the decoded frames under ``$WAN5B_FID_DIR/<tag>/`` and, when ``WAN5B_FID_REF`` names a
   previously saved directory, print and save the metrics against it::

       WAN5B_FID_TAG=bf16_s7tip pytest .../test_fidelity_ti2v_5b.py -k bh_4x8 -sv --timeout=0
       WAN5B_QUANT_CONFIG=all_bf8_lofi WAN5B_FID_REF=/mnt/tt-data/nkira/fidelity/bf16_s7tip \\
           pytest .../test_fidelity_ti2v_5b.py -k bh_4x8 -sv --timeout=0

2. ``python test_fidelity_ti2v_5b.py <ref_dir> <cand_dir>`` (host only): recompute the metrics for two
   saved runs.

Knobs (all env): ``WAN5B_FID_DIR`` (default ``/mnt/tt-data/nkira/fidelity``), ``WAN5B_FID_TAG`` (default:
the quant preset name or ``bf16``), ``WAN5B_FID_REF``, ``WAN5B_FID_HEIGHT`` 704, ``WAN5B_FID_WIDTH`` 1280,
``WAN5B_FID_FRAMES`` 81, ``WAN5B_FID_STEPS`` 40, ``WAN5B_FID_SEED`` 42, ``WAN5B_FID_PROMPT``,
``WAN5B_FID_TRACED`` 0 (eager by default: the trace modes are bit-identical to eager,
``test_trace_modes_ti2v_5b.py``, and eager skips the capture warmup).

Metrics: latent PCC (Pearson over every element of the fp32 denoised latents), latent max |diff| and relative
L2, and per-frame PSNR of the decoded uint8 frames (peak 255; mean, min, first / mid / last frame). The latent
numbers isolate the transformer; PSNR includes the VAE decode. Reading them: a bit-exact change gives PCC 1.0 and
PSNR inf; bf16 vs ``all_bf8_lofi`` at 720p measured 17.9 dB / SSIM 0.78 in sprint 6 (a different sample after 80
forwards, not a quality verdict); a PCC-class lever measured against its own track should sit well above that.
CLIP is not used anywhere here on purpose (it measures prompt similarity, notes section 8).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_DIR = "/mnt/tt-data/nkira/fidelity"
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
)


def _to_uint8_frames(video) -> np.ndarray:
    """Any (B,)T/C,H,W layout from the pipeline's ``pt`` output -> (T, H, W, 3) uint8."""
    v = video
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    v = v.detach().float().cpu()
    while v.dim() > 4:
        assert v.shape[0] == 1, f"batch > 1 not supported, got {tuple(v.shape)}"
        v = v[0]
    assert v.dim() == 4, tuple(v.shape)
    # Move the channel axis (size 3) last; the remaining axes keep their order (T, H, W).
    chan = [i for i, s in enumerate(v.shape) if s == 3]
    assert chan, f"no channel axis of size 3 in {tuple(v.shape)}"
    c = chan[-1] if len(chan) > 1 and v.shape[0] == 3 else chan[0]
    v = v.movedim(c, -1)
    if v.max() <= 1.0 + 1e-3:
        v = v * 255.0
    return v.clamp(0, 255).round().to(torch.uint8).numpy()


def compute_metrics(
    ref_latents: torch.Tensor, cand_latents: torch.Tensor, ref_frames: np.ndarray, cand_frames: np.ndarray
):
    a = ref_latents.detach().double().flatten()
    b = cand_latents.detach().double().flatten()
    assert a.numel() == b.numel(), (ref_latents.shape, cand_latents.shape)
    a0, b0 = a - a.mean(), b - b.mean()
    denom = a0.norm() * b0.norm()
    pcc = float((a0 @ b0) / denom) if denom > 0 else float("nan")
    diff = a - b
    out = {
        "latent_pcc": pcc,
        "latent_max_abs_diff": float(diff.abs().max()),
        "latent_rel_l2": float(diff.norm() / a.norm()) if a.norm() > 0 else float("nan"),
        "latent_shape": list(ref_latents.shape),
    }
    assert ref_frames.shape == cand_frames.shape, (ref_frames.shape, cand_frames.shape)
    rf = ref_frames.astype(np.float64)
    cf = cand_frames.astype(np.float64)
    mse = ((rf - cf) ** 2).reshape(rf.shape[0], -1).mean(axis=1)
    psnr = np.where(mse > 0, 10.0 * np.log10(255.0**2 / np.maximum(mse, 1e-12)), np.inf)
    t = len(psnr)
    out.update(
        {
            "frames": int(t),
            "psnr_mean_db": float(np.mean(psnr)) if np.all(np.isfinite(psnr)) else float("inf"),
            "psnr_min_db": float(np.min(psnr)),
            "psnr_first_db": float(psnr[0]),
            "psnr_mid_db": float(psnr[t // 2]),
            "psnr_last_db": float(psnr[-1]),
            "frames_identical": bool(np.array_equal(ref_frames, cand_frames)),
            "psnr_per_frame_db": [None if not math.isfinite(x) else round(float(x), 3) for x in psnr],
        }
    )
    return out


def load_run(d: str | Path):
    d = Path(d)
    latents = torch.load(d / "latents.pt", map_location="cpu")
    frames = np.load(d / "frames.npy")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return latents, frames, meta


def compare_dirs(ref_dir: str | Path, cand_dir: str | Path, *, save: bool = True) -> dict:
    ref_lat, ref_frames, ref_meta = load_run(ref_dir)
    cand_lat, cand_frames, cand_meta = load_run(cand_dir)
    m = compute_metrics(ref_lat, cand_lat, ref_frames, cand_frames)
    m["ref"] = str(ref_dir)
    m["cand"] = str(cand_dir)
    m["ref_meta"] = ref_meta
    m["cand_meta"] = cand_meta
    line = (
        f"FIDELITY cand={Path(cand_dir).name} ref={Path(ref_dir).name} "
        f"latent_pcc={m['latent_pcc']:.6f} latent_max_abs={m['latent_max_abs_diff']:.4g} "
        f"latent_rel_l2={m['latent_rel_l2']:.4g} psnr_mean={m['psnr_mean_db']:.2f}dB "
        f"psnr_min={m['psnr_min_db']:.2f}dB first/mid/last={m['psnr_first_db']:.2f}/{m['psnr_mid_db']:.2f}/{m['psnr_last_db']:.2f}dB "
        f"identical={m['frames_identical']}"
    )
    print(line)
    if save:
        out = Path(cand_dir) / f"fidelity_vs_{Path(ref_dir).name}.json"
        out.write_text(json.dumps(m, indent=1))
        print(f"saved {out}")
    return m


def _device_test_body(mesh_device, mesh_shape):
    import pytest
    from loguru import logger

    import ttnn
    from models.tt_dit.pipelines.events import log_event_section
    from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
    from models.tt_dit.pipelines.wan.quant_config import set_quant_config_from_env
    from models.tt_dit.utils.test import skip_if_unsupported_num_links

    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_FID_HEIGHT", 704))
    width = int(os.environ.get("WAN5B_FID_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_FID_FRAMES", 81))
    steps = int(os.environ.get("WAN5B_FID_STEPS", 40))
    seed = int(os.environ.get("WAN5B_FID_SEED", 42))
    prompt = os.environ.get("WAN5B_FID_PROMPT", DEFAULT_PROMPT)
    traced = os.environ.get("WAN5B_FID_TRACED", "0") == "1"
    ref_dir = os.environ.get("WAN5B_FID_REF")

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=num_frames, run_warmup=True
    )
    quant_name = set_quant_config_from_env(pipeline, rewarm=traced)
    tag = os.environ.get("WAN5B_FID_TAG") or (quant_name or "bf16")
    out_dir = Path(os.environ.get("WAN5B_FID_DIR", DEFAULT_DIR)) / tag
    flow_shift = pipeline._solver._scheduler.config.flow_shift
    logger.info(
        f"fidelity run tag={tag} quant={quant_name or 'bf16'} {width}x{height} {num_frames}f {steps} steps seed={seed} "
        f"flow_shift={flow_shift} traced={traced} -> {out_dir}"
    )

    runs = 2 if traced else 1  # traced: cold capture, then the warm run is the one we keep
    video = latents = None
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            video, latents = pipeline(
                prompts=[prompt],
                num_inference_steps=steps,
                seed=seed,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                output_type="pt_with_last_latent",
                traced=traced,
                on_event=log_event_section,
            )
        logger.info(f"generate took {time.perf_counter() - t0:.2f} s")

    if int(ttnn.distributed_context_get_rank()) != 0:
        return
    frames = _to_uint8_frames(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents.detach().float().cpu(), out_dir / "latents.pt")
    np.save(out_dir / "frames.npy", frames)
    meta = {
        "tag": tag,
        "quant_config": quant_name or "bf16",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "steps": steps,
        "seed": seed,
        "prompt": prompt,
        "flow_shift": flow_shift,
        "traced": traced,
        "guidance_scale": 4.0,
        "latent_shape": list(latents.shape),
        "frames_shape": list(frames.shape),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    try:
        from PIL import Image

        t = frames.shape[0]
        for name, idx in (("first", 0), ("mid", t // 2), ("last", t - 1)):
            Image.fromarray(frames[idx]).save(out_dir / f"{name}.png")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"no PNG previews: {e!r}")
    logger.info(f"saved latents {tuple(latents.shape)} and frames {frames.shape} to {out_dir}")
    if ref_dir:
        m = compare_dirs(ref_dir, out_dir)
        logger.info(f"latent PCC {m['latent_pcc']:.6f}, PSNR mean {m['psnr_mean_db']:.2f} dB vs {ref_dir}")


try:
    import pytest

    import ttnn
    from models.tt_dit.utils.test import ring_params_req_exact_devices

    @pytest.mark.parametrize(
        "mesh_device, mesh_shape, device_params, topology",
        [[(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring]],
        ids=["bh_4x8"],
        indirect=["mesh_device", "device_params"],
    )
    def test_fidelity_ti2v_5b(mesh_device, mesh_shape, topology):
        _device_test_body(mesh_device, mesh_shape)

except ImportError:  # host-only use of compare_dirs
    pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    compare_dirs(sys.argv[1], sys.argv[2])
