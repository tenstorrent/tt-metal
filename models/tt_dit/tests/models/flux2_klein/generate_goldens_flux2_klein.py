#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU goldens of FLUX.2 Klein through diffusers' ``Flux2KleinPipeline`` (``pipeline_flux2_klein.py``, cited as ``P``):
the reference render of one (model, size, prompt, canvas, seed, steps) with the per-step latents, for
``test_pipeline_flux2_klein.py``. The format is the FLUX 3 Klein ``gpu/generate_goldens.py`` layout reduced to what this
port consumes; ``tests/models/flux2_klein/common.py`` documents every file and key.

The pipeline runs exactly as a user runs it (``pipe(prompt, width, height, num_inference_steps, guidance_scale,
generator)``) and is observed, never altered: the noise through ``randn_tensor`` as ``prepare_latents`` draws it
(``P:478-510``), the state entering each step and the velocity ``scheduler.step`` consumes (``P:877``), the per-sample
DiT outputs of the CFG pair through a forward hook (``P:847-873``), the decoded image through ``output_type="pt"``
(``image_processor.postprocess`` before the uint8 rounding) and the text encodings through ``encode_prompt``
(``P:427-460``). The port's own ``packing.make_noise`` / ``Flux2KleinScheduler`` are cross-checked against the
observed noise, schedule and Euler chain and the verdicts recorded (``noise_matches_port`` / ``schedule_matches_port``
/ ``euler_matches_port``); the port's host modules are loaded by path so this script needs no ``ttnn``.

  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 python models/tt_dit/tests/models/flux2_klein/generate_goldens_flux2_klein.py \\
      --model distilled --size 4b --prompt man_dog --canvas 1024 1024 --seed 7 --threads 40 \\
      --out /data/<user>/flux2_klein/outputs/goldens_cpu_1024
  ... --model base --size 4b --steps 4 --guidance 4.0 ...      (the base 4-step CFG smoke golden)

The snapshot comes from ``--weights``, ``$FLUX2_KLEIN_MODEL_PATH`` or the HuggingFace cache (gated weights: nothing is
downloaded here). A distilled 4-step 1024 render takes minutes on a 40-thread node CPU; base at 4 steps twice that.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

TT_DIT = Path(__file__).resolve().parents[3]
FLUX2_KLEIN_PIPELINE_DIR = TT_DIT / "pipelines" / "flux2_klein"
TEXT_LENGTH = 512

PROMPTS = {
    "man_dog": "A man and his dog on a hill at sunset, photorealistic, 35mm.",
    "iceberg": (
        "A chunky knitted diorama: three white cable-knit icebergs with tiny pale blue houses, a cream banner "
        'embroidered "KNITTED ICEBERG VILLAGE", a giant magenta knitted whale tail bursting from a yarn sea'
    ),
}


def load_by_path(name: str, path: Path):
    """Import one of the port's host-only modules without its package ``__init__`` (which pulls in ``ttnn``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules[cls.__module__]
    spec.loader.exec_module(module)
    return module


packing = load_by_path("flux2_klein_packing", FLUX2_KLEIN_PIPELINE_DIR / "packing.py")
scheduler_module = load_by_path("flux2_klein_scheduler", FLUX2_KLEIN_PIPELINE_DIR / "scheduler.py")
weights = load_by_path("flux2_klein_weights", FLUX2_KLEIN_PIPELINE_DIR / "weights_flux2_klein.py")


def sha256_of(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_copy(t: torch.Tensor) -> torch.Tensor:
    """A fresh, contiguous host copy: safetensors refuses tensors that share storage (views, slices)."""
    return t.detach().cpu().clone().contiguous()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=weights.FLUX2_KLEIN_VARIANTS, default="distilled")
    parser.add_argument("--size", choices=weights.FLUX2_KLEIN_MODEL_SIZES, default="4b")
    parser.add_argument("--prompt", default="man_dog", help="a key of PROMPTS or literal text")
    parser.add_argument("--canvas", type=int, nargs=2, default=[1024, 1024], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=None, help="default: 4 (distilled) / 50 (base)")
    parser.add_argument("--guidance", type=float, default=None, help="default: none (distilled) / 4.0 (base)")
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--weights", type=Path, default=None, help="the diffusers snapshot (default: env / HF cache)")
    parser.add_argument("--out", type=Path, required=True, help="golden root (cases go under <out>/<model>/<case>)")
    parser.add_argument("--case", default=None, help="case name (default: <prompt key>_<W>x<H>_s<steps>_seed<seed>)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    width, height = args.canvas
    if width % packing.FLUX2_KLEIN_DOWNSAMPLING_RATIO or height % packing.FLUX2_KLEIN_DOWNSAMPLING_RATIO:
        print(f"error: the canvas must be a multiple of {packing.FLUX2_KLEIN_DOWNSAMPLING_RATIO}", file=sys.stderr)
        return 2
    prompt = PROMPTS.get(args.prompt, args.prompt)
    prompt_key = args.prompt if args.prompt in PROMPTS else "custom"
    steps = args.steps if args.steps is not None else scheduler_module.FLUX2_KLEIN_STEPS[args.model]
    cfg = args.model == "base"
    if not cfg and args.guidance not in (None, 1.0):
        print(f"error: the distilled model takes no guidance (got {args.guidance})", file=sys.stderr)
        return 2
    guidance = (
        (args.guidance if args.guidance is not None else scheduler_module.FLUX2_KLEIN_GUIDANCE["base"]) if cfg else None
    )
    try:
        snapshot = weights.resolve_weights_dir(args.model, args.size, weights_dir=args.weights, allow_download=False)
    except weights.WeightsNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    case_name = args.case or f"{prompt_key}_{width}x{height}_s{steps}_seed{args.seed}"
    case_dir = args.out / args.model / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    h, w = packing.latent_shape(height, width)
    print(
        f"RECORD {args.model} {args.size} snapshot {snapshot} -> {case_dir}; {steps} steps, guidance {guidance}",
        flush=True,
    )

    import diffusers
    import transformers
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2 import pipeline_flux2_klein as klein_module
    from safetensors.torch import save_file

    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    assert bool(pipe.config.is_distilled) == (not cfg), f"model_index.json is_distilled {pipe.config.is_distilled}"
    print(f"RECORD pipeline loaded in {time.time() - t0:.0f}s (is_distilled={pipe.config.is_distilled})", flush=True)

    # Observation 1: the noise exactly as prepare_latents draws it (P:478-510).
    noises: list[torch.Tensor] = []
    real_randn = klein_module.randn_tensor

    def recording_randn(*a, **k):
        drawn = real_randn(*a, **k)
        noises.append(host_copy(drawn))
        return drawn

    klein_module.randn_tensor = recording_randn

    # Observation 2: every DiT output in call order (cond, then uncond per step under CFG; P:847-873).
    dit_outputs: list[torch.Tensor] = []

    def dit_hook(module, inputs, output):
        sample = output[0] if isinstance(output, tuple) else output.sample
        dit_outputs.append(host_copy(sample[0, : h * w]))

    hook = pipe.transformer.register_forward_hook(dit_hook)

    # Observation 3: the state entering each step and the velocity the Euler update consumes (P:877).
    records: list[dict] = []
    real_step = pipe.scheduler.step

    def recording_step(model_output, timestep, sample, *a, **k):
        records.append(
            {"x_t": host_copy(sample[0]).float(), "velocity": host_copy(model_output[0]), "t": float(timestep)}
        )
        return real_step(model_output, timestep, sample, *a, **k)

    pipe.scheduler.step = recording_step

    # Observation 4: the latents after each step (the callback's ``latents``, P:884-889); the last is the final state.
    after_step: list[torch.Tensor] = []

    def on_step_end(pipeline, i, t, kwargs):
        after_step.append(host_copy(kwargs["latents"][0]).float())
        print(f"step {i} t={float(t):.4f} done at {time.time() - t0:.0f}s", flush=True)
        return {}

    # Observation 5: the text encodings (P:427-460), the same call the pipeline makes.
    t1 = time.time()
    text_tokens, text_ids = pipe.encode_prompt(
        prompt=prompt, device=torch.device("cpu"), max_sequence_length=TEXT_LENGTH
    )
    null_tokens, null_ids = (
        pipe.encode_prompt(prompt="", device=torch.device("cpu"), max_sequence_length=TEXT_LENGTH)
        if cfg
        else (None, None)
    )
    print(f"RECORD text encoded in {time.time() - t1:.0f}s: {tuple(text_tokens.shape)} {text_tokens.dtype}", flush=True)

    t1 = time.time()
    out = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance if cfg else 1.0,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        output_type="pt",
        max_sequence_length=TEXT_LENGTH,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    render_s = time.time() - t1
    hook.remove()
    klein_module.randn_tensor = real_randn
    pipe.scheduler.step = real_step
    print(f"RECORD rendered in {render_s:.0f}s: {len(records)} steps, {len(dit_outputs)} DiT calls", flush=True)

    assert len(noises) == 1, f"expected one randn_tensor call, saw {len(noises)}"
    assert len(records) == steps == len(after_step) and len(dit_outputs) == steps * (2 if cfg else 1)
    assert all(
        torch.equal(after_step[i], records[i + 1]["x_t"]) for i in range(steps - 1)
    ), "callback latents != next x_t"
    noise = noises[0].reshape(1, packing.FLUX2_KLEIN_IN_CHANNELS, h * w).permute(0, 2, 1)[0].float().contiguous()
    assert torch.equal(noise, records[0]["x_t"]), "the recorded step-0 state is not the packed noise"
    noise_matches_port = torch.equal(noise, packing.make_noise(args.seed, h, w))
    own_schedule = scheduler_module.Flux2KleinScheduler(steps, h * w)
    sigmas = pipe.scheduler.sigmas.detach().float().cpu()
    schedule_matches_port = torch.equal(own_schedule.sigmas, sigmas) and all(
        float(own_schedule.timesteps[i]) == records[i]["t"] for i in range(steps)
    )
    print(f"RECORD noise_matches_port {noise_matches_port} schedule_matches_port {schedule_matches_port}", flush=True)

    # The decoded image: postprocess(output_type="pt") is the denormalized [0, 1] image; golden.png rounds it as
    # numpy_to_pil(pt_to_numpy(.)) does.
    image = out.images[0]
    image_f32 = host_copy(image).float()
    png = (image_f32.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    from PIL import Image

    Image.fromarray(png).save(case_dir / "golden.png")

    # Final state: the latents after the last step = what the VAE decoded (P:877, P:905); the port's Euler update of
    # the recorded velocity must reproduce every recorded step bit for bit (the chain is exact).
    final = after_step[-1]
    euler_matches_port = all(
        torch.equal(
            scheduler_module.Flux2KleinScheduler.update(
                records[i]["x_t"], records[i]["velocity"], own_schedule.step(i)
            ).float(),
            after_step[i],
        )
        for i in range(steps)
    )
    print(f"RECORD euler_matches_port {euler_matches_port}", flush=True)
    inputs = {
        "noise": noise,
        "image_ids": host_copy(packing.latent_ids(h, w)),
        "text_tokens": host_copy(text_tokens[0]).to(torch.bfloat16),
        "text_ids": host_copy(text_ids[0] if text_ids.ndim == 3 else text_ids).to(torch.int64),
        "schedule": host_copy(sigmas),
    }
    if cfg:
        inputs["text_null_tokens"] = host_copy(null_tokens[0]).to(torch.bfloat16)
        inputs["text_null_ids"] = host_copy(null_ids[0] if null_ids.ndim == 3 else null_ids).to(torch.int64)
    step_tensors: dict[str, torch.Tensor] = {}
    for i, record in enumerate(records):
        tag = f"step{i:02d}"
        step_tensors[f"{tag}.x_t"] = record["x_t"]
        step_tensors[f"{tag}.velocity"] = record["velocity"].to(torch.bfloat16)
        if cfg:
            step_tensors[f"{tag}.velocity_cond"] = dit_outputs[2 * i].to(torch.bfloat16)
            step_tensors[f"{tag}.velocity_uncond"] = dit_outputs[2 * i + 1].to(torch.bfloat16)
    step_tensors["t"] = torch.tensor([r["t"] for r in records], dtype=torch.float32)
    step_tensors["t_next"] = host_copy(sigmas[1 : steps + 1] * 1000).float()
    step_tensors["final_tokens"] = host_copy(final).float()
    step_tensors["image_f32"] = image_f32
    save_file({k: v.contiguous() for k, v in inputs.items()}, str(case_dir / "inputs.safetensors"))
    save_file({k: v.contiguous() for k, v in step_tensors.items()}, str(case_dir / "steps.safetensors"))

    replay = {
        "model": args.model,
        "size": args.size,
        "repo": weights.repo_id(args.model, args.size),
        "snapshot": snapshot.name,
        "prompt": prompt,
        "prompt_key": prompt_key,
        "seed": args.seed,
        "canvas": [width, height],
        "latent_hw": [h, w],
        "steps": steps,
        "guidance": guidance,
        "timesteps": [r["t"] for r in records],
        "sigmas": [float(v) for v in sigmas.tolist()],
        "noise_sha256": sha256_of(noise),
        "noise_matches_port": noise_matches_port,
        "schedule_matches_port": schedule_matches_port,
        "euler_matches_port": euler_matches_port,
        "text_shape": list(text_tokens.shape),
        "device": "cpu",
        "threads": args.threads,
        "render_s": render_s,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    files = {p.name: sha256_file(p) for p in sorted(case_dir.iterdir()) if p.suffix in (".safetensors", ".png")}
    (case_dir / "record.json").write_text(
        json.dumps({"replay": replay, "files": files}, indent=2, ensure_ascii=False) + "\n"
    )

    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"device": "cpu", "cases": {}}
    index["cases"][f"{args.model}/{case_name}"] = {"replay": replay, "files": files}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT golden {case_dir}: {steps} steps in {render_s:.0f}s; keys {sorted(step_tensors)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
