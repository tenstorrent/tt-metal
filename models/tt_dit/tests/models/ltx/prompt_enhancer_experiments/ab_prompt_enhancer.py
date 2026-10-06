# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A/B: LTX-2.3 distilled on the Galaxy 4x8 ring, one raw prompt, without vs with prompt enhancement.

One pipeline, one warmup, one seed. Arm A sends the raw prompt straight to the Gemma-3 encoder; arm B
routes it through the Gemma-4-E2B-it enhancer first, on the host CPU or on the pipeline's own mesh
handle. Videos, the exact text each arm encoded, and per-stage timings land under OUT_DIR ($LTX_ENHANCER_EXP_DIR, default ~/ltx_enhancer_experiments).

With the device backend the run also collects host-vs-device parity material: a greedy rewrite of the
raw prompt from both backends, and (optionally) the host's seeded sampled rewrite next to the device's.

    AB_PROMPT          raw prompt (default "beekeeper")
    AB_TAG             filename prefix (default: the prompt with spaces replaced)
    AB_ENHANCER        host (default) | device
    AB_MAX_NEW_TOKENS  rewrite token budget (default: the enhancer's own, 512)
    AB_HOST_SAMPLED    1 -> also run the host's sampled rewrite for the parity record (device backend only)
    AB_ENHANCER_FIRST  1 -> load and warm the enhancer before the pipeline's own warmup, i.e. before the
                       Gemma-3 encoder captures its encode trace (the resident 4x8 encoder traces even when
                       the DiT does not), so the rewriter's buffers are allocated under no active trace
    SEED, NUM_FRAMES, FPS, LTX_TRACED  as in test_pipeline_ltx_distilled.py

With the device backend a greedy probe rewrite runs after warmup, after arm A and after arm B, so a rewriter
that was coherent when it loaded but is not after the encoder's trace replays shows up as such.
"""

import json
import os
import time

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.pipelines.ltx.prompt_enhancer import HostPromptEnhancer, build_prompt_enhancer, default_ltx_enhancer
from models.tt_dit.tests.models.ltx.ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL
from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma, print_ltx_timing_table, traced_default
from models.tt_dit.utils.test import skip_if_unsupported_num_links

# Outputs (videos, logs, JSON) stay out of the repo tree; LTX_ENHANCER_EXP_DIR relocates them.
OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "ab_prompt_enhancer"
)
os.makedirs(OUT_DIR, exist_ok=True)
GALAXY_RING = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]
# Greedy parity rewrites are bounded so the CPU side stays well under a minute.
PARITY_GREEDY_NEW_TOKENS = 128


def _instrument_lifecycle(enhancer, stats: dict) -> None:
    """Time the enhancer's load and warmup from outside: the pipeline drives both from its own warmup,
    so the instance methods are wrapped rather than called here."""
    load = enhancer.ensure_loaded
    warm = enhancer.warmup

    def timed_load():
        was_loaded = enhancer.is_loaded()
        t0 = time.time()
        load()
        if not was_loaded:
            stats["load_s"] = round(time.time() - t0, 1)

    def timed_warmup():
        t0 = time.time()
        warm()
        # The pipeline's own warmup calls this again once warm; that no-op must not overwrite the first timing.
        if stats.get("warmup_total_s") is not None:
            return
        stats["warmup_total_s"] = round(time.time() - t0, 1)
        stats["warmup_s"] = round(stats["warmup_total_s"] - (stats.get("load_s") or 0.0), 1)

    enhancer.ensure_loaded = timed_load
    enhancer.warmup = timed_warmup

    # Every device rewrite (warmup ones included) is recorded with its text, so a rewriter that is
    # coherent right after load but not later can be told apart from one that never was.
    generate = getattr(enhancer, "_generate", None)
    if generate is not None:
        stats["calls"] = []

        def logged_generate(messages, **kw):
            text = generate(messages, **kw)
            call = {"max_new_tokens": kw.get("max_new_tokens"), "temperature": kw.get("temperature")}
            call.update(getattr(enhancer, "last_stats", None) or {})
            call["text"] = text
            stats["calls"].append(call)
            logger.info(f"device rewrite #{len(stats['calls'])}: {call}")
            return text

        enhancer._generate = logged_generate


def _probe_video(path: str) -> dict:
    """Decode the mp4 with PyAV: frame count, size, fps, durations, audio stream presence."""
    import av

    info = {"path": path, "bytes": os.path.getsize(path)}
    with av.open(path) as c:
        vs = c.streams.video[0]
        info["fps"] = float(vs.average_rate) if vs.average_rate else None
        info["video_duration_s"] = round(float(vs.duration * vs.time_base), 3) if vs.duration else None
        n, size = 0, None
        for fr in c.decode(vs):
            n += 1
            size = (fr.width, fr.height)
        info["frames"] = n
        info["frame_size"] = size
    with av.open(path) as c:
        if c.streams.audio:
            a = c.streams.audio[0]
            samples = sum(fr.samples for fr in c.decode(a))
            info["audio"] = {"rate": a.rate, "channels": a.channels, "duration_s": round(samples / a.rate, 3)}
        else:
            info["audio"] = None
    return info


def _first_divergence(tokenizer, a: str, b: str) -> dict:
    """Token-level first difference between two rewrites (re-tokenized text, not the generated ids)."""
    ia = tokenizer.encode(a, add_special_tokens=False)
    ib = tokenizer.encode(b, add_special_tokens=False)
    idx = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
    if idx is None and len(ia) != len(ib):
        idx = min(len(ia), len(ib))
    return {
        "first_differing_token_index": idx,
        "tokens_a": len(ia),
        "tokens_b": len(ib),
        "identical": idx is None,
        "token_a": tokenizer.decode([ia[idx]]) if idx is not None and idx < len(ia) else None,
        "token_b": tokenizer.decode([ib[idx]]) if idx is not None and idx < len(ib) else None,
    }


def _greedy_rewrite(enhancer, prompt: str) -> tuple[str, dict | None]:
    """A temperature-0 rewrite through ``enhance`` with the parity budget, restoring the enhancer after."""
    saved = (enhancer.temperature, enhancer.max_new_tokens)
    enhancer.temperature, enhancer.max_new_tokens = 0.0, PARITY_GREEDY_NEW_TOKENS
    try:
        t0 = time.time()
        text = enhancer.enhance(prompt, mode="t2v", seed=0)
        stats = dict(getattr(enhancer, "last_stats", None) or {})
        stats["wall_s"] = round(time.time() - t0, 1)
        return text, stats
    finally:
        enhancer.temperature, enhancer.max_new_tokens = saved


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_ab_prompt_enhancer(mesh_device, device_params, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    skip_if_unsupported_num_links(mesh_device, num_links)

    raw_prompt = os.environ.get("AB_PROMPT", "beekeeper")
    tag = os.environ.get("AB_TAG", raw_prompt.strip().replace(" ", "_"))
    backend = os.environ.get("AB_ENHANCER", "host").strip().lower()
    assert backend in ("host", "device"), f"AB_ENHANCER={backend!r}: expected host or device"
    max_new_tokens = os.environ.get("AB_MAX_NEW_TOKENS")
    seed = int(os.environ.get("SEED", "10"))
    num_frames = int(os.environ.get("NUM_FRAMES", "153"))
    fps = float(os.environ.get("FPS", "25"))
    height, width = 1088, 1920

    parent_mesh = mesh_device
    mesh_shape = tuple(parent_mesh.shape)
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    traced = traced_default(device_params, os.environ.get("LTX_TRACED"))

    # mesh_device stays None: the pipeline binds its own submesh handle (and CCL topology) to a device
    # enhancer, so the rewriter and the DiT share one allocator.
    enhancer = build_prompt_enhancer(backend, model_path=default_ltx_enhancer())
    if max_new_tokens:
        enhancer.max_new_tokens = int(max_new_tokens)
    lifecycle = {"load_s": None, "warmup_s": None, "warmup_total_s": None}
    _instrument_lifecycle(enhancer, lifecycle)
    logger.info(
        f"A/B enhancer backend={backend} name={enhancer.name} max_new_tokens={enhancer.max_new_tokens} "
        f"temperature={enhancer.temperature} cache_dir={getattr(enhancer, 'cache_dir', None)}"
    )

    # Untraced runs skip warmup_buffers unless asked, and warmup_buffers is where the pipeline loads and
    # warms a device enhancer (after the encoder's first encode). AB_ENHANCER_FIRST loads it before that
    # encode instead, through the same pipeline hook, and runs the pipeline warmup afterwards.
    enhancer_first = os.environ.get("AB_ENHANCER_FIRST", "0") == "1"
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        traced=traced,
        run_warmup=not enhancer_first,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        image_conditioning=False,
        prompt_enhancer=enhancer,
    )
    if enhancer_first:
        logger.info("AB_ENHANCER_FIRST=1: warming the enhancer before the pipeline warmup (no trace captured yet)")
        pipeline._warmup_prompt_enhancer()
        pipeline.warmup_buffers(num_frames=num_frames, height=height, width=width)
    logger.info(f"enhancer lifecycle after pipeline warmup: {lifecycle} loaded={enhancer.is_loaded()}")

    probes = {}

    def probe(stage: str) -> None:
        if backend != "device":
            return
        logger.info(f"=== probe: greedy rewrite on device, {stage} ===")
        text, stats = _greedy_rewrite(enhancer, raw_prompt)
        probes[stage] = {"text": text, "stats": stats}
        logger.info(f"device greedy {stage}: {text!r} {stats}")

    probe("after_warmup")

    arms = [("A_raw_prompt", None), (f"B_{backend}_enhanced_prompt", enhancer)]
    results = {}
    for arm, arm_enhancer in arms:
        pipeline.prompt_enhancer = arm_enhancer
        video = os.path.join(OUT_DIR, f"{tag}_{arm}_{width}x{height}_{num_frames}f_seed{seed}.mp4")
        logger.info(f"=== arm {arm}: enhancer={'none' if arm_enhancer is None else arm_enhancer.name} ===")
        t0 = time.time()
        pipeline.generate(
            raw_prompt, output_path=video, num_frames=num_frames, height=height, width=width, seed=seed, fps=fps
        )
        wall = time.time() - t0
        print_ltx_timing_table(
            pipeline,
            label=f"AB {arm}",
            num_frames=num_frames,
            height=height,
            width=width,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            output_path=video,
            prompt=pipeline.last_enhanced_prompt or raw_prompt,
        )
        results[arm] = {
            "video": video,
            "prompt_encoded": pipeline.last_enhanced_prompt or raw_prompt,
            "timings_s": {name: round(secs, 2) for name, secs in pipeline.last_timings},
            "wall_s": round(wall, 1),
            "enhancer_stats": dict(getattr(arm_enhancer, "last_stats", None) or {}) if arm_enhancer else None,
        }
        try:
            results[arm]["video_probe"] = _probe_video(video)
        except Exception as e:  # noqa: BLE001 — the probe is diagnostics; the generate already succeeded
            results[arm]["video_probe"] = {"error": repr(e)}
        logger.info(f"arm {arm} -> {video} ({wall:.1f}s wall) probe={results[arm]['video_probe']}")
        probe(f"after_arm_{arm[0]}")

    summary = {
        "raw_prompt": raw_prompt,
        "seed": seed,
        "enhancer_seed": pipeline.enhancer_seed,
        "shape": {"frames": num_frames, "height": height, "width": width, "fps": fps},
        "mesh": {"shape": mesh_shape, "sp_axis": sp_axis, "tp_axis": tp_axis, "topology": str(topology)},
        "traced": traced,
        "enhancer": {
            "backend": backend,
            "name": enhancer.name,
            "max_new_tokens": enhancer.max_new_tokens,
            "temperature": enhancer.temperature,
            "cache_dir": getattr(enhancer, "cache_dir", None),
            "model_dir": getattr(enhancer, "_model_dir", None),
            "stop_tokens": getattr(enhancer, "_stop_tokens", None),
            **{k: v for k, v in lifecycle.items() if k != "calls"},
        },
        "arms": results,
    }

    summary["enhancer_first"] = enhancer_first
    summary["device_rewrite_calls"] = lifecycle.get("calls")

    if backend == "device":
        parity = {"greedy_new_tokens": PARITY_GREEDY_NEW_TOKENS, "device_greedy_probes": probes}
        last = probes["after_arm_B"]
        parity["device_greedy"], parity["device_greedy_stats"] = last["text"], last["stats"]

        host = HostPromptEnhancer(default_ltx_enhancer(), max_new_tokens=enhancer.max_new_tokens)
        logger.info("=== parity: greedy rewrite on host ===")
        parity["host_greedy"], parity["host_greedy_stats"] = _greedy_rewrite(host, raw_prompt)
        logger.info(f"host greedy: {parity['host_greedy']!r}")
        parity["greedy_divergence"] = {
            stage: _first_divergence(host._tokenizer, parity["host_greedy"], p["text"]) for stage, p in probes.items()
        }

        parity["device_sampled"] = results[f"B_{backend}_enhanced_prompt"]["prompt_encoded"]
        if os.environ.get("AB_HOST_SAMPLED", "0") == "1":
            logger.info("=== parity: sampled rewrite on host (same seed/budget as arm B) ===")
            t0 = time.time()
            parity["host_sampled"] = host.enhance(raw_prompt, mode="t2v", seed=pipeline.enhancer_seed)
            parity["host_sampled_wall_s"] = round(time.time() - t0, 1)
            parity["sampled_divergence"] = _first_divergence(
                host._tokenizer, parity["host_sampled"], parity["device_sampled"]
            )
            logger.info(f"host sampled: {parity['host_sampled']!r}")
        host.unload()
        summary["parity"] = parity

    suffix = "" if backend == "host" else f"_{backend}"
    summary_path = os.path.join(OUT_DIR, f"{tag}_ab_results{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"A/B summary -> {summary_path}")

    if traced:
        pipeline.release_traces()
