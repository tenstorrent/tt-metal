# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One process, frozen cached weights, two matched prompts across Wan recipes."""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest
import torch
import ttnn

from attention import WanAttentionAdapter, kernel
from diagnostics import Diagnostics
from models.tt_dit.pipelines.events import DenoiseStep, SectionEnd, SectionStart
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils import cache
from models.tt_dit.utils.video import export_to_video

PROMPTS = [
    "A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.",
    "A woman sits at an outdoor café in soft daylight, turns toward the camera, smiles, and lifts a ceramic cup with both hands. Natural facial expressions, clearly visible hands, realistic skin texture, a steady camera.",
]


@pytest.fixture
def device_params():
    return dict(fabric_config=ttnn.FabricConfig.FABRIC_1D, l1_small_size=65536, trace_region_size=64 * 1024**2)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.timeout(14400)
def test_wan_frontier(mesh_device, monkeypatch):
    torch.set_num_threads(16)
    destination = Path(os.environ["WAN_SUITE_OUTPUT"])
    destination.mkdir(parents=True, exist_ok=False)
    variants = os.environ.get("WAN_VARIANTS", "D C B E F G stock").split()
    checkpoint = os.environ["WAN_CHECKPOINT"]
    records = []
    original_load = cache.load_model

    def cached_load(model=None, **kwargs):
        tt_model = kwargs.pop("tt_model", model)
        was_loaded = tt_model.is_loaded()

        def forbid_conversion():
            raise RuntimeError(f"Unexpected converted-weight cache miss: {kwargs['subfolder']}")

        kwargs["get_torch_state_dict"] = forbid_conversion
        start = time.perf_counter()
        result = original_load(tt_model, **kwargs)
        records.append(
            dict(
                component=kwargs["subfolder"],
                already_loaded=was_loaded,
                cache_hit=not was_loaded,
                seconds=time.perf_counter() - start,
            )
        )
        return result

    monkeypatch.setattr(cache, "load_model", cached_load)
    start = time.perf_counter()
    pipeline = WanPipeline(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=checkpoint, height=480, width=832, num_frames=81
        ),
        run_warmup=False,
    )
    setup = time.perf_counter() - start
    diagnostics = Diagnostics(pipeline)
    for variant in variants:
        assert variant in ("stock", "D", "C", "B", "E", "F", "G")
        output = destination / variant
        output.mkdir()
        for state in pipeline.transformer_states:
            for block in state.model.blocks:
                if hasattr(block.attn1, "_attention_override"):
                    del block.attn1._attention_override
        adapter = None if variant == "stock" else WanAttentionAdapter(variant)
        if adapter is not None:
            adapter.install(pipeline)
        numeric = None
        if adapter is not None:
            fp32, _, defines, fidelity = kernel.recipe(variant)
            numeric = dict(
                fp32_dst=fp32,
                defines=defines,
                fidelity=str(fidelity),
                tail_mask="native partial-K palette",
                explicit_mask_format_reconfig=variant in "DCF",
            )
        manifest = dict(
            status="running",
            variant=variant,
            checkpoint=checkpoint,
            checkpoint_revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
            width=832,
            height=480,
            frames=81,
            steps=40,
            seed=42,
            guidance=[4.0, 3.0],
            prompts=PROMPTS,
            mesh=[2, 4],
            sp=4,
            tp=2,
            logical_tokens=32760,
            padded_tokens=32768,
            recipe=numeric,
            pipeline_setup_seconds=setup,
            attention_schedule=(
                "stock ring" if variant == "stock" else "prepared-format KV all-gather then local Q attention"
            ),
            results=[],
            block_bench={},
            captures={},
            converted_weight_cache=os.environ["TT_DIT_CACHE_DIR"],
        )

        def save():
            manifest["weight_loads"] = records
            manifest["transport"] = {} if adapter is None else adapter.transport
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        save()
        print("START_VARIANT", variant, flush=True)
        diagnostics.reset(output, capture=variant == "D")
        diagnostics.enabled = True
        start = time.perf_counter()
        with torch.no_grad():
            pilot = pipeline(prompts=[PROMPTS[0]], num_inference_steps=2, seed=42, output_type="uint8", traced=False)
        ttnn.synchronize_device(mesh_device)
        assert tuple(pilot.shape) == (1, 81, 480, 832, 3)
        diagnostics.enabled = False
        manifest["pilot_seconds"] = time.perf_counter() - start
        manifest["block_bench"] = diagnostics.records
        manifest["captures"] = diagnostics.captures
        assert len(diagnostics.records) == 4
        save()
        print("PILOT_COMPLETE", variant, manifest["pilot_seconds"], flush=True)
        del pilot
        for prompt_id, prompt in enumerate(PROMPTS):
            phases, phase_start, steps = {}, {}, []
            run_start = time.perf_counter()

            def on_event(event):
                now = time.perf_counter()
                if isinstance(event, SectionStart):
                    phase_start[event.name] = now
                elif isinstance(event, SectionEnd):
                    phases[event.name] = now - phase_start[event.name]
                elif isinstance(event, DenoiseStep):
                    steps.append(dict(step=event.step, elapsed_seconds=now - run_start))
                    if event.step % 5 == 0:
                        print("WAN_PROGRESS", variant, prompt_id, event.step, round(now - run_start, 3), flush=True)

            with torch.no_grad():
                frames = pipeline(
                    prompts=[prompt],
                    num_inference_steps=40,
                    seed=42,
                    output_type="uint8",
                    guidance_scale=4.0,
                    guidance_scale_2=3.0,
                    traced=False,
                    on_event=on_event,
                )
            ttnn.synchronize_device(mesh_device)
            seconds = time.perf_counter() - run_start
            frames = np.asarray(frames)
            assert frames.shape == (1, 81, 480, 832, 3) and frames.dtype == np.uint8
            video = output / f"prompt{prompt_id}-seed42.mp4"
            export_to_video(frames[0], str(video), fps=16)
            sampled = []
            for index in np.linspace(0, 80, 8, dtype=int):
                path = output / f"prompt{prompt_id}-frame{index:03d}.png"
                Image.fromarray(frames[0, index]).save(path)
                sampled.append(
                    dict(frame=int(index), file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                )
            manifest["results"].append(
                dict(
                    prompt_id=prompt_id,
                    seed=42,
                    pipeline_seconds=seconds,
                    phases=phases,
                    step_progress=steps,
                    video=video.name,
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    sampled_frames=sampled,
                    decoded_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                    pixel_min=int(frames.min()),
                    pixel_max=int(frames.max()),
                )
            )
            save()
            print("VIDEO_COMPLETE", variant, prompt_id, seconds, flush=True)
            del frames
        if adapter is not None:
            assert len(adapter.transport) == 80
        manifest["status"] = "completed"
        save()
        print("COMPLETED_VARIANT", variant, flush=True)
    pipeline.release_traces()
