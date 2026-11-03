# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DBCache A/B for the Wan 2.2 **I2V** pipeline (``Wan-AI/Wan2.2-I2V-A14B-Diffusers``).

I2V counterpart of ``test_pipeline_wan_dbcache.py`` (same helpers, same run names, same
``custom:...`` syntax). One model load serves every ``WAN_DBCACHE_FLOW_SHIFTS`` value: for each
flow shift the configured runs are generated from the same seed image / prompt / seed and compared
(PSNR, PCC) against that flow shift's uncached baseline.

Environment:
  WAN_DBCACHE_RUNS         comma separated run names (default: baseline,split_nocache,dbcache)
  WAN_DBCACHE_FLOW_SHIFTS  comma separated flow shifts (default: the pipeline default only)
  WAN_DBCACHE_STEPS        denoising steps (default 40)
  WAN_DBCACHE_RDT          residual_diff_threshold override for the ``dbcache*`` presets
  WAN_DBCACHE_IMAGE        seed image path (default: models/sample_data/house_in_field_1080p.jpg)
  WAN_DBCACHE_PROMPT, WAN_DBCACHE_SEED
  WAN_DBCACHE_OFFICIAL_SIZE  1 = size the output like Wan's image2video.py: the seed image's aspect
                           ratio at the parametrized pixel area, in whole latent patches
  WAN_DBCACHE_BOUNDARY_RATIO expert switch boundary override (default: the checkpoint's 0.9)
  WAN_DBCACHE_OUT_DIR      where videos, frame strips and results.json go (default: cwd)
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.dbcache import WanDBCacheConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.utils import dram_probe
from models.tt_dit.utils.dbcache import DBCacheConfig

from ....utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import check_first_frame_matches_seed, check_output_sanity
from .test_pipeline_wan_dbcache import _TRACE_REGION_SIZE, _parse_custom_config, _pcc, _psnr, _save_frame_strip, _Timer

_DEFAULT_IMAGE = Path(__file__).resolve().parents[4] / "sample_data" / "house_in_field_1080p.jpg"
_DEFAULT_PROMPT = (
    "A gentle breeze moves through the tall grass in front of the house while clouds drift slowly "
    "across the sky; the camera pushes in slowly and steadily."
)


def _official_size(max_area: int, image: PIL.Image.Image) -> tuple[int, int]:
    """Wan's image2video.py sizing: the image's aspect ratio at max_area, in whole latent patches."""
    aspect = image.height / image.width
    lat_h = round(math.sqrt(max_area * aspect) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / aspect) // 8 // 2 * 2)
    return lat_w * 8, lat_h * 8


def _dram(mesh_device: ttnn.MeshDevice) -> str:
    """Per-chip DRAM occupancy (all banks), for the Loudbox 2x4 memory budget."""
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        mb = 1 << 20
        return (
            f"DRAM/chip {v.total_bytes_allocated_per_bank * v.num_banks / mb:.0f} MB allocated of "
            f"{v.total_bytes_per_bank * v.num_banks / mb:.0f} MB, largest free block/bank "
            f"{v.largest_contiguous_bytes_free_per_bank / mb:.0f} MB"
        )
    except Exception as err:  # noqa: BLE001 - diagnostics only
        return f"DRAM view unavailable: {err}"


def _arm_dram_probe(pipeline) -> None:
    """Point main's ``dram_probe`` (written for MiniMax-H3) at a Wan pipeline when
    ``WAN_DBCACHE_DRAM_PROBE=1``: arm it, skip the Wan CCL managers and device handles in the
    owner walk, and list the Wan managers' ping-pong caches as ``ccl_ping_pong``. The experts
    (``transformer_states[i].model``) and the VAE encoder are not among the probe's owners, so they
    show under ``pipeline_state`` with their full paths."""
    import types

    dram_probe.ENABLED = os.environ.get("WAN_DBCACHE_DRAM_PROBE") == "1"
    dram_probe._SKIP_ATTRS |= {"dit_ccl_manager", "vae_ccl_manager", "_device", "_ccl_manager"}  # noqa: SLF001
    orig_entries = dram_probe._ccl_entries  # noqa: SLF001

    def _wan_ccl_entries(p):
        shim = types.SimpleNamespace(
            encoder_ccl_manager=getattr(p, "encoder_ccl_manager", None),
            ccl_manager=getattr(p, "dit_ccl_manager", None),
            audio_ccl_manager=getattr(p, "vae_ccl_manager", None),
        )
        renames = [("audio_ccl_manager", "vae_ccl_manager"), ("ccl_manager", "dit_ccl_manager")]
        out = []
        for path, t in orig_entries(shim):
            for old, new in renames:
                if path.startswith(old):
                    path = new + path[len(old) :]
                    break
            out.append((path, t))
        return out

    dram_probe._ccl_entries = _wan_ccl_entries  # noqa: SLF001
    dram_probe.register(pipeline)


def _free_ccl_buffers(mgr) -> int:
    """Drop a CCLManager's shape-keyed persistent all-gather / reduce-scatter pairs (untraced only).

    The cache never evicts, so after a denoise at 832x1104 the DiT manager keeps ~20 GB/chip of
    full-sequence gather buffers resident through the VAE decode and the next T5 load. They are
    re-allocated on first use by the next denoise.
    """

    def dealloc(entry) -> int:
        if entry is None:
            return 0
        if isinstance(entry, (list, tuple)):
            return sum(dealloc(e) for e in entry)
        if isinstance(entry, ttnn.Tensor):
            try:
                ttnn.deallocate(entry)
            except Exception:  # noqa: BLE001 - already freed
                return 0
            return 1
        return 0

    n = 0
    for name in ("_ping_pong_buffer_cache", "_fused_norm_stats_buffer_cache"):
        cache = getattr(mgr, name, None)
        if cache:
            for key in list(cache):
                n += dealloc(cache.pop(key))
    idx = getattr(mgr, "_ping_pong_buffer_indices", None)
    if idx is not None:
        idx.clear()
    return n


def _flow_shifts() -> list[float | None]:
    raw = os.environ.get("WAN_DBCACHE_FLOW_SHIFTS", "").strip()
    if not raw:
        return [None]  # pipeline default
    return [float(x) for x in raw.split(",") if x.strip()]


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, traced",
    [
        # The only 4x8 Blackhole config test_pipeline_wan.py runs I2V on (it skips the 4x8 ring for I2V).
        [(4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False, False],
        [
            (4, 8),
            1,
            0,
            2,
            False,
            {**line_params_req_exact_devices, "trace_region_size": _TRACE_REGION_SIZE},
            ttnn.Topology.Linear,
            False,
            True,
        ],
        [(4, 8), 1, 0, 2, False, ring_params_req_exact_devices, ttnn.Topology.Ring, False, False],
        [(2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False, False],
        # FSDP over all 8 chips (3.5 GB/chip per expert instead of 14 GB): 832x1104 on the Loudbox
        # OOMs in the transformer with the TP-2 layout even with the T5 encoder swapped out.
        [(2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, False],
    ],
    ids=[
        "bh_4x8sp1tp0nl2_linear",
        "bh_4x8sp1tp0nl2_traced_linear",
        "bh_4x8sp1tp0nl2_ring",
        "bh_2x4sp1tp0nl2_linear",
        "bh_2x4sp0tp1nl1_linear_fsdp",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(832, 480), (1280, 720)], ids=["480p", "720p"])
@pytest.mark.timeout(8 * 3600)  # first run converts the weight cache; then one video per run per flow shift
def test_wan_i2v_dbcache_ab(
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
    shared = {"residual_diff_threshold": float(rdt_env)} if rdt_env else {}
    runs = os.environ.get("WAN_DBCACHE_RUNS", "baseline,split_nocache,dbcache").split(",")
    flow_shifts = _flow_shifts()
    out_dir = Path(os.environ.get("WAN_DBCACHE_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(os.environ.get("WAN_DBCACHE_IMAGE", str(_DEFAULT_IMAGE)))
    seed_image = PIL.Image.open(image_path).convert("RGB")
    if os.environ.get("WAN_DBCACHE_OFFICIAL_SIZE") == "1":
        width, height = _official_size(width * height, seed_image)
    logger.info(f"seed image: {image_path} ({seed_image.width}x{seed_image.height}) -> {width}x{height}")
    boundary_env = os.environ.get("WAN_DBCACHE_BOUNDARY_RATIO")
    boundary_ratio = float(boundary_env) if boundary_env else None
    image_prompt = [ImagePrompt(image=seed_image, frame_pos=0)]
    prompt = os.environ.get("WAN_DBCACHE_PROMPT", _DEFAULT_PROMPT)
    seed = int(os.environ.get("WAN_DBCACHE_SEED", "0"))

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    # WAN_DBCACHE_EVICT_T5=1: on a mesh whose DRAM cannot hold the T5 encoder next to one
    # transformer (BH 2x4: 14 GB/chip transformer + 11 GB T5, job 139804 OOM'd in warmup), add
    # the text encoder to the dynamic-load swap chain; __call__ already re-prepares the encoder
    # and the experts before use, so each call pays one extra T5 cache load.
    evict_t5 = os.environ.get("WAN_DBCACHE_EVICT_T5") == "1"
    mem_probe = os.environ.get("WAN_DBCACHE_MEM_PROBE") == "1"
    pipeline = WanPipelineI2V(
        run_warmup=not (evict_t5 or mem_probe),
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
            checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            model_type="i2v",
            height=height,
            width=width,
            num_frames=num_frames,
            # WAN_DBCACHE_VAE_T_CHUNK=N decodes the video N frames at a time (the Wan VAE's causal
            # feat_cache path, the BH 2x4 preset); the 4x8 preset's full-T decode OOMs at 832x1104.
            **(
                {"vae_t_chunk_size": int(vae_chunk_env)}
                if (vae_chunk_env := os.environ.get("WAN_DBCACHE_VAE_T_CHUNK"))
                else {}
            ),
        ),
    )
    logger.info(
        f"VAE decode T chunk: {pipeline.config.vae_t_chunk_size if hasattr(pipeline, 'config') else vae_chunk_env or 'preset'}"
    )
    logger.info(f"after pipeline init: {_dram(mesh_device)}")
    if mem_probe:
        t5 = pipeline._text_encoder._tt_encoder  # noqa: SLF001
        t5.deallocate_weights()
        logger.info(f"mem probe: T5 evicted -> {_dram(mesh_device)}")
        for idx, ts in enumerate(pipeline.transformer_states):
            if ts.model.is_loaded():
                ts.model.deallocate_weights()
                logger.info(f"mem probe: transformer[{idx}] evicted -> {_dram(mesh_device)}")
        pipeline._vae.deallocate_weights()  # noqa: SLF001
        logger.info(f"mem probe: VAE decoder evicted -> {_dram(mesh_device)}")
        pipeline._vae.reload_weights()  # noqa: SLF001 - __call__ does not reload the decoder
    if evict_t5:
        t5 = pipeline._text_encoder._tt_encoder  # noqa: SLF001
        pipeline.transformer.register_coresident_exclusions(t5)
        pipeline.transformer_2.register_coresident_exclusions(t5)
        t5.register_coresident_exclusions(pipeline.transformer, pipeline.transformer_2)
        # The chain only acts on a load, and init left T5 + transformer[0] resident: drop the
        # experts now so the warmup's first expert load evicts T5 before the VAE encode runs.
        for ts in pipeline.transformer_states:
            ts.model.deallocate_weights()
        logger.info(f"T5 text encoder added to the dynamic-load swap chain; experts dropped -> {_dram(mesh_device)}")
    if os.environ.get("WAN_DBCACHE_EVICT_FOR_VAE") == "1":
        # Also keep the experts out of DRAM during the VAE decode (the WH T3K chain); the decoder
        # adapter reloads its own weights in decode(), and the next call reloads the experts anyway.
        dec = pipeline._vae.decoder  # noqa: SLF001
        pipeline.transformer.register_coresident_exclusions(dec)
        pipeline.transformer_2.register_coresident_exclusions(dec)
        dec.register_coresident_exclusions(pipeline.transformer, pipeline.transformer_2)
        logger.info("VAE decoder added to the dynamic-load swap chain")
    # WAN_DBCACHE_FREE_CCL=1: release the DiT CCL manager's persistent gather buffers before the
    # VAE decode and before the T5 load (see _free_ccl_buffers). WAN_DBCACHE_DRAM_PROBE=1: per-owner
    # DRAM attribution at the same checkpoints.
    free_ccl = os.environ.get("WAN_DBCACHE_FREE_CCL") == "1"
    if free_ccl:
        assert not traced, "WAN_DBCACHE_FREE_CCL frees ping-pong buffers a trace would reference"
    _arm_dram_probe(pipeline)
    dram_probe.report("after pipeline init")
    orig_decode = pipeline._decode_latents  # noqa: SLF001
    orig_prepare_t5 = pipeline._prepare_text_encoder  # noqa: SLF001
    orig_prepare_latents = pipeline.prepare_latents
    # The VAE encoder's full-resolution neighbour-pad gathers (vae_ccl_manager, ~15 GB/chip at
    # 832x1104, a new pair per frame count) and the DiT's full-sequence gathers (~6 GB/chip) each
    # only matter inside their own stage; the T5 encoder shares the VAE manager.
    dit_mgrs = {id(pipeline.dit_ccl_manager): pipeline.dit_ccl_manager}
    vae_mgrs = {id(m): m for m in (pipeline.vae_ccl_manager, pipeline.encoder_ccl_manager)}

    def _free(mgrs: dict, where: str) -> None:
        if free_ccl:
            n = sum(_free_ccl_buffers(m) for m in mgrs.values())
            logger.info(f"freed {n} CCL buffers {where} -> {_dram(mesh_device)}")
        elif mem_probe:
            logger.info(f"mem probe: {where} -> {_dram(mesh_device)}")

    def _prepare_text_encoder():
        _free({**dit_mgrs, **vae_mgrs}, "before T5 load")
        orig_prepare_t5()

    def _prepare_latents(*args, **kwargs):
        out = orig_prepare_latents(*args, **kwargs)
        _free(vae_mgrs, "after VAE encode")
        return out

    def _decode_latents(*args, **kwargs):
        dram_probe.report("before VAE decode")
        _free(dit_mgrs, "before VAE decode")
        out = orig_decode(*args, **kwargs)
        if mem_probe:
            logger.info(f"mem probe: after VAE decode -> {_dram(mesh_device)}")
        return out

    pipeline._prepare_text_encoder = _prepare_text_encoder  # noqa: SLF001
    pipeline.prepare_latents = _prepare_latents
    pipeline._decode_latents = _decode_latents  # noqa: SLF001

    if evict_t5 or mem_probe:
        pipeline._warmup()  # noqa: SLF001 - skipped in the constructor above
        logger.info(f"after warmup: {_dram(mesh_device)}")
        dram_probe.report("after warmup")

    configs = {
        "baseline": None,
        "split_nocache": DBCacheConfig(residual_diff_threshold=0.0, max_warmup_steps=0),
        "dbcache": WanDBCacheConfig.default(**shared),
        "dbcache_ts0": WanDBCacheConfig.default(**shared, taylorseer_order=0),
        "dbcache_ts1": WanDBCacheConfig.default(**shared, taylorseer_order=1),
        "dbcache_ts2": WanDBCacheConfig.default(**shared, taylorseer_order=2),
        "dbcache_r08": WanDBCacheConfig.default(residual_diff_threshold=0.08),
    }

    report: dict[str, dict[str, dict]] = {}
    is_rank0 = int(ttnn.distributed_context_get_rank()) == 0

    for flow_shift in flow_shifts:
        fs_key = "default" if flow_shift is None else f"{flow_shift:g}"
        results: dict[str, dict] = {}
        for name in runs:
            cache_config = _parse_custom_config(name) if name.startswith("custom:") else configs[name]
            timer = _Timer()
            logger.info(
                f"=== I2V run '{name}' (flow_shift={fs_key}, steps={num_inference_steps}, traced={traced}, "
                f"cache={cache_config}) ==="
            )
            t_call = time.perf_counter()
            with torch.no_grad():
                frames = pipeline(
                    prompts=[prompt],
                    image_prompt=image_prompt,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    flow_shift=flow_shift,
                    boundary_ratio=boundary_ratio,
                    guidance_scale=3.5,
                    guidance_scale_2=3.5,
                    output_type="uint8",
                    traced=traced,
                    on_event=timer,
                    cache_config=cache_config,
                )
            e2e_s = time.perf_counter() - t_call  # whole pipeline call: T5 + VAE encode + denoising + VAE decode
            frames = np.asarray(frames[0])
            check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
            check_first_frame_matches_seed(frames, seed_image=seed_image, width=width, height=height)

            summary = pipeline.cache_summary() if cache_config is not None else []
            cached_steps = sum(len(b) for s in summary for b in s["cached_steps"])
            results[name] = {
                "frames": frames,
                "e2e_s": e2e_s,
                "encoder_s": timer.durations.get("encoder", float("nan")),
                "prepare_latents_s": timer.durations.get("prepare_latents", float("nan")),
                "denoising_s": timer.durations.get("denoising", float("nan")),
                "vae_s": timer.durations.get("vae", float("nan")),
                "step_times": timer.step_times,
                "cached_branch_steps": cached_steps,
                "summary": summary,
            }
            logger.info(
                f"run '{name}' fs={fs_key}: end-to-end {e2e_s:.1f}s = encoder {results[name]['encoder_s']:.1f}s + "
                f"prepare_latents {results[name]['prepare_latents_s']:.1f}s + denoising {results[name]['denoising_s']:.1f}s + "
                f"vae {results[name]['vae_s']:.1f}s (+ host); cached branch-steps {cached_steps}"
            )
            for s in summary:
                logger.info(f"  [{s['name']}] {s['config']}: cached_steps={s['cached_steps']}")
                for branch, diffs in enumerate(s["residual_diffs"]):
                    logger.info(f"    branch {branch} diffs: " + ", ".join(f"{k}:{v:.3f}" for k, v in diffs.items()))

            if is_rank0:
                safe = name.replace(":", "_").replace("&", "_").replace("=", "")
                stem = f"wan_i2v_dbcache_{safe}_{width}x{height}_s{num_inference_steps}_fs{fs_key}{'_traced' if traced else ''}"
                _save_frame_strip(frames, str(out_dir / f"{stem}_strip.png"))
                try:
                    from models.tt_dit.utils.video import export_to_video

                    export_to_video(frames, str(out_dir / f"{stem}.mp4"), fps=16)
                    logger.info(f"Saved video to: {out_dir / (stem + '.mp4')}")
                except ImportError:
                    logger.info("Could not export video - imageio_ffmpeg not available")

        # ------------------------------------------------------------ report (per flow shift)
        base = results.get("baseline")
        lines = [
            "",
            f"I2V {width}x{height} flow_shift={fs_key} steps={num_inference_steps}",
            f"{'run':<36}{'e2e(s)':>9}{'denoise(s)':>12}{'step(s)':>10}{'cached':>8}{'PSNR vs base':>14}{'PCC vs base':>13}",
        ]
        for name, r in results.items():
            compare = base is not None and name != "baseline"
            r["psnr"] = _psnr(r["frames"], base["frames"]) if compare else float("nan")
            r["pcc"] = _pcc(r["frames"], base["frames"]) if compare else float("nan")
            mean_step = float(np.mean(r["step_times"])) if r["step_times"] else float("nan")
            lines.append(
                f"{name:<36}{r['e2e_s']:>9.1f}{r['denoising_s']:>12.1f}{mean_step:>10.2f}{r['cached_branch_steps']:>8}"
                f"{r['psnr']:>14.2f}{r['pcc']:>13.4f}"
            )
        if base is not None:
            for name, r in results.items():
                if name != "baseline":
                    lines.append(
                        f"speedup {name}: {base['denoising_s'] / r['denoising_s']:.2f}x on denoising, "
                        f"{base['e2e_s'] / r['e2e_s']:.2f}x end-to-end"
                    )
        logger.info("\n".join(lines))

        report[fs_key] = {name: {k: v for k, v in r.items() if k not in ("frames",)} for name, r in results.items()}

        if base is not None and "split_nocache" in results:
            assert results["split_nocache"]["cached_branch_steps"] == 0
            assert results["split_nocache"]["psnr"] > 35.0, "split path without caching diverged from baseline"
            assert results["split_nocache"]["pcc"] > 0.99, "split path without caching diverged from baseline"
        for name, r in results.items():
            if name not in ("baseline", "split_nocache") and r["cached_branch_steps"] == 0:
                # Informational, not a failure: e.g. 720p with flow_shift=12 at threshold 0.05 never
                # gets a residual diff under the threshold. The run still counts as a data point.
                logger.warning(f"{name} fs={fs_key}: DBCache never cached a step")

    if is_rank0:
        path = out_dir / f"wan_i2v_dbcache_results_{width}x{height}{'_traced' if traced else ''}.json"
        path.write_text(json.dumps(report, indent=2, default=float))
        logger.info(f"Saved results to: {path}")
