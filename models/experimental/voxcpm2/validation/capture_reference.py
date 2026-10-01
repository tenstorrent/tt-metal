# SPDX-License-Identifier: Apache-2.0
"""Capture native, uncompiled VoxCPM2 generation on CUDA; no device fallback."""
import argparse
from contextlib import ExitStack, contextmanager
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import random
import re
import numpy as np
import torch
from .manifest import SCHEMA_VERSION, TensorRecorder, sha256_file

UPSTREAM_REVISION = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"
FORWARD_COMPONENTS = (
    "base_lm", "residual_lm", "feat_encoder", "fsq_layer", "enc_to_lm_proj",
    "fusion_concat_proj", "lm_to_dit_proj", "res_to_dit_proj", "stop_proj",
    "stop_actn", "stop_head", "feat_decoder", "feat_decoder.estimator",
)
METHOD_COMPONENTS = (("base_lm", "forward_step"), ("residual_lm", "forward_step"),
                     ("audio_vae", "encode"), ("audio_vae", "decode"))


def _resolve(model, name):
    value = model
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _wrap_method(recorder, component, method, module=None):
    def call(*args, **kwargs):
        event = recorder.begin(component, args, kwargs)
        position = None
        if component.endswith(".forward_step") and module is not None:
            position_tensor = kwargs.get("position_id", args[1] if len(args) > 1 else None)
            position = int(position_tensor.item())
            caches = [module.kv_cache.get_layer_cache(i) for i in range(len(module.layers))]
            event["kv_cache_capacity"] = int(caches[0][0].shape[2])
            event["state_before"] = recorder.snapshot(
                [(key[:, :, :position, :], val[:, :, :position, :]) for key, val in caches],
                event["key"] + "/state_before")
        output = method(*args, **kwargs)
        recorder.end(event, output)
        if position is not None:
            event["state_after"] = recorder.snapshot(
                [(key[:, :, :position + 1, :], val[:, :, :position + 1, :]) for key, val in caches],
                event["key"] + "/state_after")
        return output
    return call


@contextmanager
def _replace_method(module, name, replacement):
    had_instance_attribute = name in module.__dict__
    previous = module.__dict__.get(name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        if had_instance_attribute:
            setattr(module, name, previous)
        else:
            delattr(module, name)


def install_capture(stack, model, recorder):
    # Methods are wrapped, not reconstructed. This includes forward_step calls
    # which bypass torch Module.forward hooks in the official autoregressive loop.
    codec_components = ["audio_vae.decoder", "audio_vae.encoder", "audio_vae.encoder.fc_mu"]
    codec_components.extend("audio_vae.decoder.model." + name for name, _ in model.audio_vae.decoder.model.named_children())
    for name in (*FORWARD_COMPONENTS, *codec_components):
        module = _resolve(model, name)
        stack.enter_context(_replace_method(module, "forward", _wrap_method(recorder, name + ".forward", module.forward)))
    for name, method_name in METHOD_COMPONENTS:
        module = _resolve(model, name)
        method = getattr(module, method_name)
        stack.enter_context(_replace_method(module, method_name, _wrap_method(recorder, name + "." + method_name, method, module)))
    original = model._inference

    def inference(*args, **kwargs):
        event = recorder.begin("inference", args, kwargs)
        # Upstream reseeds immediately before this call. Save the state that
        # produced diffusion noise, rather than merely the process initial seed.
        recorder.snapshot(torch.get_rng_state(), event["key"] + "/rng/cpu")
        recorder.snapshot(torch.cuda.get_rng_state(model.device), event["key"] + "/rng/cuda")
        generator = original(*args, **kwargs)
        try:
            for index, output in enumerate(generator):
                event.setdefault("yields", []).append(recorder.snapshot(output, event["key"] + f"/yield/{index}"))
                yield output
        finally:
            generator.close()
    stack.enter_context(_replace_method(model, "_inference", inference))


def _official_source():
    distribution = importlib.metadata.distribution("voxcpm")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct_url.get("vcs_info", {}).get("commit_id")
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Install official voxcpm from pinned git revision {UPSTREAM_REVISION}; installed revision is {revision!r}")
    url = direct_url.get("url", "").removesuffix(".git")
    if url not in ("https://github.com/OpenBMB/VoxCPM", "https://github.com/openbmb/VoxCPM"):
        raise RuntimeError(f"Unexpected voxcpm source URL: {url}")
    return {"revision": revision, "url": url, "version": distribution.version}


def capture_reference(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.checkpoint_revision):
        raise ValueError("--checkpoint-revision must be an exact 40-character commit SHA")
    if not args.text.strip():
        raise ValueError("--text must be nonempty")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official reference; CPU/MPS fallback is disabled")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    source = _official_source()
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    checkpoint = args.checkpoint.resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("architecture", "").lower() != "voxcpm2":
        raise ValueError("Checkpoint config must declare architecture='voxcpm2'")
    if bool(args.prompt_wav) != bool(args.prompt_text):
        raise ValueError("--prompt-wav and --prompt-text must be supplied together")
    if args.max_len < 1 or args.min_len < 0 or args.inference_timesteps < 1:
        raise ValueError("Invalid generation length or inference timesteps")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recorder = TensorRecorder(output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = VoxCPM2Model.from_local(str(checkpoint), optimize=False, device=str(device))
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cuda"}:
        raise RuntimeError(f"Native model parameters must all be CUDA, got {devices}")
    generation = dict(target_text=args.text, prompt_text=args.prompt_text or "",
                      prompt_wav_path=str(args.prompt_wav.resolve()) if args.prompt_wav else "",
                      reference_wav_path=str(args.reference_wav.resolve()) if args.reference_wav else "",
                      min_len=args.min_len, max_len=args.max_len,
                      inference_timesteps=args.inference_timesteps, cfg_value=args.cfg_value,
                      retry_badcase=False, trim_silence_vad=False, seed=args.seed)
    with ExitStack() as stack, torch.inference_mode():
        install_capture(stack, model, recorder)
        waveform = model.generate(**generation)
    torch.cuda.synchronize(device)
    recorder.snapshot(waveform, "waveform")
    import soundfile as sf
    sf.write(output / "reference.wav", waveform.detach().cpu().float().squeeze().numpy(), model.sample_rate, subtype="FLOAT")
    files = {}
    for path in sorted(checkpoint.iterdir()):
        if path.is_file():
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    prompt_files = {}
    for name, path in (("prompt_wav", args.prompt_wav), ("reference_wav", args.reference_wav)):
        if path:
            prompt_files[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    source["model_file_sha256"] = sha256_file(inspect.getfile(VoxCPM2Model))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "soundfile", "safetensors")}
    manifest = {
        "schema_version": SCHEMA_VERSION, "complete": True, "oracle": "official_voxcpm2_cuda", "backend": "cuda",
        "source": source,
        "checkpoint": {"repository": args.checkpoint_repo, "revision": args.checkpoint_revision, "files": files},
        "cuda": {"device": str(device), "device_name": torch.cuda.get_device_name(device),
                 "capability": list(torch.cuda.get_device_capability(device)), "runtime": torch.version.cuda,
                 "uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
                 "cudnn": torch.backends.cudnn.version()},
        "versions": versions, "python": platform.python_version(), "generation": generation,
        "prompt_files": prompt_files, "last_successful_seed": model.last_successful_seed,
        "rng": {"seed": args.seed, "torch_states": "inference/000000/rng/{cpu,cuda}",
                "tf32": False, "compile": False, "retries": False},
        "sample_rate": model.sample_rate, "wav": {"file": "reference.wav", "sha256": sha256_file(output / "reference.wav")},
        "events": recorder.events, "tensors": recorder.tensors,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-repo", default="openbmb/VoxCPM2")
    parser.add_argument("--checkpoint-revision", required=True, help="Exact Hugging Face checkpoint commit")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite an oracle")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--reference-wav", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    return parser


def main(argv=None):
    capture_reference(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
