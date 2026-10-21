# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A/B test for DBCache (cache-dit style step skipping) on the Wan 2.2 T2V pipeline.

One pipeline instance, same prompt and seed, three runs:

1. ``baseline``     - the regular ``combined_step`` path.
2. ``split_nocache`` - the head/body/tail split with caching disabled (threshold 0). Must
                       reproduce the baseline; validates the split itself.
3. ``dbcache``       - cache-dit's Wan 2.2 preset (or ``WAN_DBCACHE_RDT`` threshold override).

Reports denoising time, cached-step counts and PSNR against the baseline video.

Environment overrides: ``WAN_DBCACHE_STEPS`` (default 40), ``WAN_DBCACHE_RDT`` (default: preset),
``WAN_DBCACHE_RUNS`` (comma separated subset of the named configs below, or ``custom:...`` entries,
see ``_parse_custom_config``). The ``*_traced`` parametrizations run the traced path.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.events import DenoiseStep, SectionEnd, SectionStart
from models.tt_dit.pipelines.wan.dbcache import WanDBCacheConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils.dbcache import DBCacheConfig

from ....utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import check_output_sanity


class _Timer:
    """`on_event` callback that records section durations and per-step denoising times."""

    def __init__(self) -> None:
        self._start: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.step_times: list[float] = []
        self._last_step_t: float | None = None

    def __call__(self, event) -> None:
        now = time.perf_counter()
        if isinstance(event, SectionStart):
            self._start[event.name] = now
            if event.name == "denoising":
                self._last_step_t = now
        elif isinstance(event, SectionEnd):
            if event.name in self._start:
                self.durations[event.name] = now - self._start.pop(event.name)
        elif isinstance(event, DenoiseStep):
            if self._last_step_t is not None:
                self.step_times.append(now - self._last_step_t)
            self._last_step_t = now


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _parse_custom_config(name: str) -> WanDBCacheConfig:
    """``custom:rdt=0.06&Fn=2&Bn=0&ts=1&mc=2&mcs=12&warm=4&cfgsep=0`` -> WanDBCacheConfig.

    ``mcs`` sets ``max_cached_steps`` on both experts (the preset caps high/low at 8/20 otherwise).
    """
    shared: dict[str, object] = {}
    per_expert: dict[str, object] = {}
    high: dict[str, object] = {}
    low: dict[str, object] = {}
    # Items are separated by "&" (the run list itself is comma separated).
    for item in name[len("custom:") :].split("&"):
        if not item:
            continue
        k, v = item.split("=")
        if k == "rdt":
            shared["residual_diff_threshold"] = float(v)
        elif k == "Fn":
            shared["Fn_compute_blocks"] = int(v)
        elif k == "Bn":
            shared["Bn_compute_blocks"] = int(v)
        elif k == "ts":
            shared["taylorseer_order"] = int(v)
        elif k == "mc":
            shared["max_continuous_cached_steps"] = int(v)
        elif k == "cfgsep":
            shared["cfg_diff_compute_separate"] = bool(int(v))
        elif k == "mcs":
            per_expert["max_cached_steps"] = int(v)
        elif k == "warm":
            per_expert["max_warmup_steps"] = int(v)
        elif k == "hi_rdt":
            high["residual_diff_threshold"] = float(v)
        elif k == "lo_rdt":
            low["residual_diff_threshold"] = float(v)
        else:
            msg = f"unknown custom config key {k!r}"
            raise ValueError(msg)
    cfg = WanDBCacheConfig.default(**shared)
    return WanDBCacheConfig(
        high_noise=cfg.high_noise.replace(**per_expert, **high),
        low_noise=cfg.low_noise.replace(**per_expert, **low),
    )


def _save_frame_strip(frames: np.ndarray, path: str, *, count: int = 5, scale: int = 2) -> None:
    """Save `count` evenly spaced frames side by side (downscaled) for a quick visual check."""
    from PIL import Image

    idx = np.linspace(0, frames.shape[0] - 1, count, dtype=int)
    tiles = [Image.fromarray(frames[i]).reduce(scale) for i in idx]
    strip = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width
    strip.save(path)
    logger.info(f"Saved frame strip to: {path}")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


# Baseline needs 2 traces (one per expert), DBCache 3 per expert; other tt_dit pipelines use 26-60 MB.
_TRACE_REGION_SIZE = 256 * 1024 * 1024


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, traced",
    [
        [(4, 8), 1, 0, 2, False, ring_params_req_exact_devices, ttnn.Topology.Ring, False, False],
        [
            (4, 8),
            1,
            0,
            2,
            False,
            {**ring_params_req_exact_devices, "trace_region_size": _TRACE_REGION_SIZE},
            ttnn.Topology.Ring,
            False,
            True,
        ],
        [(4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False, False],
        [(2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False, False],
    ],
    ids=["bh_4x8sp1tp0nl2_ring", "bh_4x8sp1tp0nl2_traced_ring", "bh_4x8sp1tp0nl2_linear", "bh_2x4sp1tp0nl2_linear"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(832, 480), (1280, 720)], ids=["480p", "720p"])
@pytest.mark.timeout(6 * 3600)  # first run converts the weight cache; then 3 full videos
def test_wan_dbcache_ab(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
    traced: bool,
    width: int,
    height: int,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    if not ttnn.device.is_blackhole():
        pytest.skip("DBCache A/B test is parametrized for Blackhole meshes")

    num_frames = 81
    num_inference_steps = int(os.environ.get("WAN_DBCACHE_STEPS", "40"))
    rdt_env = os.environ.get("WAN_DBCACHE_RDT")
    shared = {"residual_diff_threshold": float(rdt_env)} if rdt_env else {}  # else: preset default
    runs = os.environ.get("WAN_DBCACHE_RUNS", "baseline,split_nocache,dbcache").split(",")

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = WanPipeline(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            checkpoint_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    prompt = os.environ.get(
        "WAN_DBCACHE_PROMPT",
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    )
    seed = int(os.environ.get("WAN_DBCACHE_SEED", "0"))
    flow_shift_env = os.environ.get("WAN_DBCACHE_FLOW_SHIFT")
    flow_shift = float(flow_shift_env) if flow_shift_env else None  # None -> pipeline default (5.0)

    configs = {
        "baseline": None,
        # Same head/body/tail path as dbcache but never allowed to cache.
        "split_nocache": DBCacheConfig(residual_diff_threshold=0.0, max_warmup_steps=0),
        # The shipped default preset (asserted on below); also what a call with no cache_config gets.
        "dbcache": WanDBCacheConfig.default(**shared),
        # Matrix: plain residual reuse vs. TaylorSeer forecasting orders, and cache-dit's 0.08 threshold.
        "dbcache_ts0": WanDBCacheConfig.default(**shared, taylorseer_order=0),
        "dbcache_ts1": WanDBCacheConfig.default(**shared, taylorseer_order=1),
        "dbcache_ts2": WanDBCacheConfig.default(**shared, taylorseer_order=2),
        "dbcache_r08": WanDBCacheConfig.default(residual_diff_threshold=0.08),
    }

    results: dict[str, dict] = {}
    for name in runs:
        cache_config = _parse_custom_config(name) if name.startswith("custom:") else configs[name]
        timer = _Timer()
        logger.info(
            f"=== run '{name}' (steps={num_inference_steps}, flow_shift={flow_shift}, traced={traced}, cache={cache_config}) ==="
        )
        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                flow_shift=flow_shift,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                output_type="uint8",
                traced=traced,
                on_event=timer,
                cache_config=cache_config,
            )
        frames = np.asarray(frames[0])
        check_output_sanity(frames, num_frames=num_frames, height=height, width=width)

        summary = pipeline.cache_summary() if cache_config is not None else []
        cached_steps = sum(len(b) for s in summary for b in s["cached_steps"])
        results[name] = {
            "frames": frames,
            "denoising_s": timer.durations.get("denoising", float("nan")),
            "vae_s": timer.durations.get("vae", float("nan")),
            "step_times": timer.step_times,
            "cached_branch_steps": cached_steps,
            "summary": summary,
        }
        logger.info(
            f"run '{name}': denoising {results[name]['denoising_s']:.1f}s, "
            f"vae {results[name]['vae_s']:.1f}s, cached branch-steps {cached_steps}"
        )
        offset = 0
        cached_global: set[int] = set()
        for s in summary:
            logger.info(f"  [{s['name']}] {s['config']}: cached_steps={s['cached_steps']}")
            for branch, diffs in enumerate(s["residual_diffs"]):
                logger.info(f"    branch {branch} diffs: " + ", ".join(f"{k}:{v:.3f}" for k, v in diffs.items()))
            if s["profile_ms"]:
                logger.info(
                    "    profile (ms, mean per branch-call): "
                    + ", ".join(f"{k}={v:.1f}" for k, v in s["profile_ms"].items())
                )
            # Steps where both branches cached, in global step numbering (experts run back to back).
            branch_sets = [set(b) for b in s["cached_steps"]]
            both = set.intersection(*branch_sets) if branch_sets else set()
            cached_global |= {offset + st for st in both}
            offset += s["executed_steps"]
        if timer.step_times and cached_global:
            cached_t = [t for i, t in enumerate(timer.step_times) if i in cached_global]
            computed_t = [t for i, t in enumerate(timer.step_times) if i not in cached_global]
            logger.info(
                f"  step time: cached {np.mean(cached_t):.3f}s (n={len(cached_t)}), "
                f"computed {np.mean(computed_t):.3f}s (n={len(computed_t)})"
            )

        if int(ttnn.distributed_context_get_rank()) == 0:
            safe = name.replace(":", "_").replace("&", "_").replace("=", "")
            shift_tag = f"_fs{flow_shift:g}" if flow_shift is not None else ""
            stem = f"wan_dbcache_{safe}_{width}x{height}_s{num_inference_steps}{shift_tag}{'_traced' if traced else ''}"
            _save_frame_strip(frames, f"{stem}_strip.png")
            try:
                from models.tt_dit.utils.video import export_to_video

                export_to_video(frames, f"{stem}.mp4", fps=16)
                logger.info(f"Saved video to: {stem}.mp4")
            except ImportError:
                logger.info("Could not export video - imageio_ffmpeg not available")

    # ---------------------------------------------------------------- report
    base = results.get("baseline")
    lines = ["", f"{'run':<15}{'denoise(s)':>12}{'step(s)':>10}{'cached':>8}{'PSNR vs base':>14}{'PCC vs base':>13}"]
    for name, r in results.items():
        compare = base is not None and name != "baseline"
        psnr = _psnr(r["frames"], base["frames"]) if compare else float("nan")
        pcc = _pcc(r["frames"], base["frames"]) if compare else float("nan")
        r["psnr"] = psnr
        r["pcc"] = pcc
        mean_step = float(np.mean(r["step_times"])) if r["step_times"] else float("nan")
        lines.append(
            f"{name:<15}{r['denoising_s']:>12.1f}{mean_step:>10.2f}{r['cached_branch_steps']:>8}"
            f"{psnr:>14.2f}{pcc:>13.4f}"
        )
    if base is not None:
        for name, r in results.items():
            if name != "baseline":
                lines.append(f"speedup {name}: {base['denoising_s'] / r['denoising_s']:.2f}x on denoising")
    logger.info("\n".join(lines))

    # ---------------------------------------------------------------- checks
    if base is not None and "split_nocache" in results:
        assert results["split_nocache"]["cached_branch_steps"] == 0
        assert results["split_nocache"]["psnr"] > 35.0, "split path without caching diverged from baseline"
        assert results["split_nocache"]["pcc"] > 0.99, "split path without caching diverged from baseline"
    if "dbcache" in results:
        assert results["dbcache"]["cached_branch_steps"] > 0, "DBCache never cached a step"
        if base is not None:
            assert results["dbcache"]["psnr"] > 18.0, "DBCache output diverged too far from baseline"
            assert results["dbcache"]["pcc"] > 0.90, "DBCache output diverged too far from baseline"
            assert results["dbcache"]["denoising_s"] < base["denoising_s"], "DBCache did not reduce denoising time"
