# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed-recipe FLUX.2 evaluation; unsupported inputs fail explicitly.

Run from the repository root with pytest. Output directories must be new.
Host profiler durations are diagnostics, NOT transformer device benchmarks.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROMPTS = (
    "A photo of a cat sitting on a windowsill at sunset",
    "A lifelike portrait of a woman in her late 30s, golden-hour lighting, "
    "85mm lens, shallow depth of field, sharp focus on the eyes, detailed skin texture.",
    "An alien world with floating rock islands, massive glowing tropical plants, "
    "and twin moons in the night sky. Wide-angle view, cinematic lighting, vivid colors.",
)


def positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@pytest.fixture
def device_params():
    import ttnn

    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 256 * 1024**2}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(3600)
def test_stock_pipeline(mesh_device, model_location_generator, monkeypatch):
    import torch
    import ttnn

    from models.perf.benchmarking_utils import BenchmarkProfiler
    from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from models.tt_dit.models.vae.vae import VaeAttention
    from models.tt_dit.utils.tensor import to_torch
    from block_bench import BlockBench
    from model_attention import FrontierAttention
    import device_attention as kernel

    # D512 VAE SDPA exceeds BH L1 at Q128/K128 with the pinned research build.
    # Apply this scheduling-only bring-up setting identically to every variant.
    monkeypatch.setattr(VaeAttention, "sdpa_chunk_size_map", {(True, 2, 1, 4): (64, 64)})

    variant = os.environ.get("FLUX2_VARIANT", "stock")
    if variant != "stock" and variant not in kernel.VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; stock fallback is forbidden")
    capture_enabled = False
    capture_description = "first denoising step of two-step prompt0 seed0 qualification"
    captures = {}

    def capture(name, call, attn, q, k, v):
        if not capture_enabled or name not in {"dual.0", "dual.7", "single.23", "single.47"} or name in captures:
            return
        values = []
        for index, value in enumerate((q, k, v)):
            # One local head from each TP rank, all SP tokens: four global heads.
            end = list(value.shape)
            end[1] = 1
            selected = ttnn.slice(value, [0, 0, 0, 0], end)
            host = to_torch(selected, mesh_axes=[None, 1, 0, None]).clone()
            assert bool(torch.isfinite(host).all()), f"Non-finite real QKV at {name}"
            if variant == "G" and index:
                kernel.B4.validate_input(host)
            values.append(host)
        filename = f"qkv-{name}.pt"
        torch.save(dict(q=values[0], k=values[1], v=values[2]), output / filename)
        captures[name] = dict(
            file=filename,
            sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
            local_head_indices=[0],
            tp_ranks=[0, 1, 2, 3],
            shape=list(values[0].shape),
            source=capture_description,
            rms=[float(x.float().square().mean().sqrt()) for x in values],
            max_abs=[float(x.float().abs().max()) for x in values],
        )

    adapter = None if variant == "stock" else FrontierAttention(variant, capture=capture)
    bench = BlockBench() if os.environ.get("FLUX2_BLOCK_BENCH", "0") == "1" else None
    conditioning_mode = os.environ.get("FLUX2_CONDITIONING", "stock")
    if conditioning_mode not in ("stock", "corrected"):
        raise ValueError("Unknown conditioning mode")
    model_repair = os.environ.get("FLUX2_MODEL_REPAIR", "none")
    if model_repair not in ("none", "both", "main_fused"):
        raise ValueError("Unknown model repair")

    def setup(transformer):
        if model_repair != "none":
            from model_fixes import install as install_model_fixes

            install_model_fixes(transformer, residual=True, per_head_norm=True, fused=model_repair == "main_fused")
        if conditioning_mode == "corrected":
            from conditioning import install

            install(transformer)
        if adapter is not None:
            adapter.install(transformer)
        if bench is not None:
            bench.install(transformer)

    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        raise ValueError("This evaluation requires Blackhole")
    steps = positive_env("FLUX2_STEPS", 2)
    prompt_count = positive_env("FLUX2_PROMPTS", 1)
    seed_count = positive_env("FLUX2_SEEDS", 1)
    if prompt_count > len(PROMPTS) or seed_count > 2:
        raise ValueError("The fixed suite supports three prompts and two seeds")
    traced = os.environ.get("FLUX2_TRACED", "0")
    if traced not in ("0", "1"):
        raise ValueError("FLUX2_TRACED must be 0 or 1")
    traced = traced == "1"
    exploratory = os.environ.get("FLUX2_EXPLORATORY", "0") == "1"
    output = Path(os.environ["FLUX2_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = os.environ.get("FLUX2_CHECKPOINT") or str(model_location_generator("black-forest-labs/FLUX.2-dev"))
    sources = [
        Path(__file__).resolve(),
        ROOT / "models/tt_dit/pipelines/flux2/pipeline_flux2.py",
        ROOT / "models/tt_dit/models/transformers/transformer_flux2.py",
        ROOT / "models/tt_dit/blocks/attention_opt.py",
        HERE / "device_attention.py",
        HERE / "model_attention.py",
        HERE / "block_bench.py",
        HERE / "conditioning.py",
        HERE / "model_fixes.py",
    ]
    manifest = {
        "schema": "flux2-frontier-eval-v1",
        "status": "started",
        "evaluation_mode": "exploratory_repeatability_not_required" if exploratory else "strict",
        "variant": variant,
        "conditioning": conditioning_mode,
        "model_repair": model_repair,
        "vae_sdpa_chunks": [64, 64],
        "trace_region_bytes": 256 * 1024**2,
        "checkpoint": checkpoint,
        "converted_weight_cache": os.environ.get("TT_DIT_CACHE_DIR"),
        "pinned_memory_cache_limit_bytes": os.environ.get("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "entry_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers")},
        "command": sys.argv,
        "mesh": list(mesh_device.shape),
        "width": 1024,
        "height": 1024,
        "steps": steps,
        "guidance": 4.0,
        "prompt_upsampling": False,
        "traced": traced,
        "prompts": list(PROMPTS[:prompt_count]),
        "seeds": [0, 42][:seed_count],
        "timing_scope": "Host stage durations only; not device/block performance",
        "attention_schedule": (
            "stock ring" if variant == "stock" else "local Q, prepared-format KV all-gather, rectangular attention"
        ),
        "attention_recipe": None if adapter is None else kernel.recipe(variant)[2],
        "attention_qk_chunks": None if adapter is None else [256, 512],
        "results": [],
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    from models.tt_dit.utils import cache as weight_cache

    original_load_model = weight_cache.load_model
    manifest["weight_loads"] = []

    def measured_load_model(tt_model, **kwargs):
        record = {
            "model_name": kwargs["model_name"],
            "subfolder": kwargs["subfolder"],
            "already_loaded": tt_model.is_loaded(),
            "torch_state_dict_requested": False,
        }
        getter = kwargs.get("get_torch_state_dict")
        if getter is not None:

            def measured_getter():
                record["torch_state_dict_requested"] = True
                if os.environ.get("FLUX2_REQUIRE_WEIGHT_CACHE", "0") == "1":
                    raise AssertionError(f"Unexpected weight-cache miss: {record['subfolder']}")
                return getter()

            kwargs["get_torch_state_dict"] = measured_getter
        begin = time.perf_counter()
        try:
            return original_load_model(tt_model, **kwargs)
        finally:
            record["seconds_including_cache_write"] = time.perf_counter() - begin
            manifest["weight_loads"].append(record)
            save_manifest()

    monkeypatch.setattr(weight_cache, "load_model", measured_load_model)
    started = time.perf_counter()
    try:
        pipeline = Flux2Pipeline.create_pipeline(
            mesh_device=mesh_device,
            checkpoint_name=checkpoint,
            sp_axis=0,
            tp_axis=1,
            encoder_tp_axis=1,
            vae_tp_axis=1,
            vae_h_axis=0,
            vae_w_axis=None,
            num_links=2,
            topology=ttnn.Topology.Linear,
            width=1024,
            height=1024,
            is_fsdp=False,
            dynamic_load=False,
            trace_warmup=False,
            shard_prompt=True,
            transformer_setup=setup,
        )
        manifest["pipeline_setup_seconds"] = time.perf_counter() - started
        if os.environ.get("FLUX2_VERIFY_CACHED_WEIGHT", "0") == "1":
            # This replicated weight was the original cache-load stall site.
            weight = pipeline.transformer.time_guidance_embed.timestep_embedder.linear_2.weight.data
            manifest["cache_validation_weight"] = {
                "name": "time_guidance_embed.timestep_embedder.linear_2.weight",
                "device_shard_sha256": [
                    hashlib.sha256(ttnn.to_torch(shard).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for shard in ttnn.get_device_tensors(weight)
                ],
            }
            save_manifest()
        cache_dir = output.parent / "prompt-embeddings"
        cache_dir.mkdir(exist_ok=True)
        original_encode = pipeline._prompt_encoder.encode

        def encode(prompts, *, num_images_per_prompt, sequence_length, traced):
            key = dict(checkpoint=checkpoint, prompts=prompts, num_images=num_images_per_prompt, length=sequence_length)
            digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            path = cache_dir / f"{digest}.pt"
            if path.exists():
                result = torch.load(path, weights_only=True)
            else:
                result = original_encode(
                    prompts, num_images_per_prompt=num_images_per_prompt, sequence_length=sequence_length, traced=traced
                )
                torch.save(result, path)
            assert bool(torch.isfinite(result[0]).all())
            manifest.setdefault("embedding_cache", {})[digest] = hashlib.sha256(path.read_bytes()).hexdigest()
            return result

        pipeline._prompt_encoder.encode = encode
        original_decode = pipeline._vae_decoder.forward

        def decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            host = ttnn.to_torch(ttnn.get_device_tensors(result)[0])
            assert bool(torch.isfinite(host).all()), "Non-finite VAE output before uint8 conversion"
            return result

        pipeline._vae_decoder.forward = decode

        def latents():
            value = to_torch(pipeline.ts.tt_latents_step, mesh_axes=[None, 0, None]).clone()
            assert bool(torch.isfinite(value).all()), "Non-finite final latents"
            return value

        # Capture benchmark inputs before allocating the persistent model trace.
        # Verify exact replay with the same prompt, seed and two-step schedule.
        if traced or bench is not None:
            capture_enabled = True
            if bench is not None:
                bench.enabled = True
            pipeline(
                prompts=[PROMPTS[0]],
                num_inference_steps=2,
                seed=0,
                guidance_scale=4.0,
                prompt_upsample_temperature=None,
                traced=False,
            )
            reference = latents()
            capture_enabled = False
            manifest["real_input_captures"] = captures
            if bench is not None:
                bench.enabled = False
                # Benchmark before the full-model trace can reuse scratch buffers.
                manifest["block_bench"] = bench.benchmark(mesh_device, strict=not exploratory)
            if traced:
                # The stock _step tracer uses prep_run=True with
                # clone_prep_inputs=False, while _step updates latents in place.
                # Its first capture call therefore performs an extra scheduler
                # update. Discard it, like the pipeline's standard trace warmup.
                pipeline(
                    prompts=[PROMPTS[0]],
                    num_inference_steps=2,
                    seed=0,
                    guidance_scale=4.0,
                    prompt_upsample_temperature=None,
                    traced=True,
                )
                capture_latents = latents()
                manifest["discarded_capture_run_l2_vs_untraced_pct"] = 100 * float(
                    torch.linalg.vector_norm(capture_latents.float() - reference.float())
                    / torch.linalg.vector_norm(reference.float())
                )
                replay_checks = []
                for _ in range(2):
                    pipeline(
                        prompts=[PROMPTS[0]],
                        num_inference_steps=2,
                        seed=0,
                        guidance_scale=4.0,
                        prompt_upsample_temperature=None,
                        traced=True,
                    )
                    actual = latents()
                    exact = torch.equal(actual, reference)
                    replay_checks.append(
                        dict(
                            exact=exact,
                            l2_pct=100
                            * float(
                                torch.linalg.vector_norm(actual.float() - reference.float())
                                / torch.linalg.vector_norm(reference.float())
                            ),
                        )
                    )
                    manifest["two_step_trace_checks"] = replay_checks
                    save_manifest()
                    if not exploratory:
                        assert exact, "Steady model trace differs from untraced two-step latents"
                manifest["two_step_trace_latents_bitwise_equal"] = all(r["exact"] for r in replay_checks)
            save_manifest()
        if not traced and bench is None:
            capture_enabled = True
            capture_description = f"first denoising step of {steps}-step prompt0 seed0 generation"
        for prompt_id, prompt in enumerate(manifest["prompts"]):
            for seed in manifest["seeds"]:
                profiler = BenchmarkProfiler()
                ttnn.synchronize_device(mesh_device)
                with torch.no_grad(), profiler("run", iteration=0):
                    images = pipeline(
                        prompts=[prompt],
                        num_inference_steps=steps,
                        seed=seed,
                        guidance_scale=4.0,
                        prompt_upsample_temperature=None,
                        traced=traced,
                        profiler=profiler,
                        profiler_iteration=0,
                    )
                    ttnn.synchronize_device(mesh_device)
                assert len(images) == 1 and images[0].size == (1024, 1024)
                final_latents = latents()
                filename = f"{variant}-prompt{prompt_id}-seed{seed}.png"
                images[0].save(output / filename)
                latent_file = filename.replace(".png", "-latents.pt")
                torch.save(final_latents, output / latent_file)
                manifest["results"].append(
                    {
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "image": filename,
                        "image_sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                        "latents": latent_file,
                        "latents_sha256": hashlib.sha256((output / latent_file).read_bytes()).hexdigest(),
                        "latents_finite": True,
                        "denoising_step_seconds": [
                            profiler.get_duration(f"denoising_step_{i}", 0) for i in range(steps)
                        ],
                        "host_seconds": {
                            stage: profiler.get_duration(stage, 0) for stage in ("run", "encoder", "denoising", "vae")
                        },
                    }
                )
                if adapter is not None:
                    assert len(adapter.transport) == 56, "Not every transformer block used the selected attention"
                    manifest["observed_attention_transport"] = adapter.transport
                    manifest["real_input_captures"] = captures
                save_manifest()
                print("FLUX2_IMAGE_DONE", variant, prompt_id, seed, flush=True)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_manifest()
