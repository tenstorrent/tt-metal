# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import gc
import importlib
import importlib.metadata
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request

VOXTRAL_DEPLOY_CONFIG = "vllm_omni/model_executor/stage_configs/voxtral_tts.yaml"


def _installed_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _dependency_versions() -> str:
    return (
        "installed versions: "
        f"vllm=={_installed_version('vllm')}, "
        f"vllm-omni=={_installed_version('vllm-omni')}, "
        f"torch=={_installed_version('torch')}"
    )


def _import_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import `{module_name}` with {sys.executable}: {exc}. "
            f"{_dependency_versions()}. Install or activate the dependency in this same Python environment. "
            f"{install_hint}"
        ) from exc


def _default_voxtral_deploy_config() -> str:
    try:
        dist = importlib.metadata.distribution("vllm-omni")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("vllm-omni is required to locate the Voxtral deploy config.") from exc

    config_path = dist.locate_file(VOXTRAL_DEPLOY_CONFIG)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find packaged Voxtral deploy config: {config_path}")
    return str(config_path)


def _require_vllm_omni_gpu_runtime() -> None:
    vllm_platforms = _import_dependency("vllm.platforms", "Expected: vLLM with a supported runtime platform.")
    current_platform = vllm_platforms.current_platform
    device_type = getattr(current_platform, "device_type", "")
    if device_type == "cuda":
        return

    raise RuntimeError(
        "Voxtral's packaged vLLM-Omni reference config uses CUDA GPU workers "
        "(`GPUARWorker` and `GPUGenerationWorker`) and cannot run on the current vLLM platform. "
        f"Detected platform={current_platform.__class__.__name__}, device_type={device_type!r}. "
        "The earlier `libcudart.so.12` warning means this Python environment cannot load the CUDA runtime. "
        "Run this reference on an NVIDIA/CUDA environment with a compatible vLLM build, or use these "
        "reference helpers only for config/tokenization while implementing the TTNN runtime separately."
    )


def _sampling_params(max_tokens: int, cfg_alpha: float | None = None) -> list[Any]:
    vllm = _import_dependency("vllm", "Expected: vllm >= 0.18.0.")
    SamplingParams = vllm.SamplingParams

    extra_args = {"cfg_alpha": cfg_alpha} if cfg_alpha is not None else None
    sampling_params = SamplingParams(max_tokens=max_tokens, extra_args=extra_args)
    return [sampling_params, sampling_params]


def _write_audio(path: Path, audio_array, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Writing wav output requires soundfile.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio_array, sample_rate)


def run_cpu(args: argparse.Namespace) -> None:
    if args.streaming:
        raise ValueError("--streaming is only supported with --backend vllm.")
    if args.concurrency is not None:
        raise ValueError("--concurrency is only supported with --backend vllm --streaming.")
    if args.num_prompts != 1:
        raise ValueError("CPU reference currently supports --num-prompts 1.")

    start = time.time()
    reference = VoxtralCPUReference(model_name_or_path=args.model, dtype=args.dtype, device=args.device)
    audio_array = reference.generate(
        args.text,
        voice=args.voice or None,
        ref_audio=args.ref_audio,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    elapsed = time.time() - start
    output_audio_dur = len(audio_array) / args.sample_rate

    if args.write_audio:
        output_path = Path(args.output_dir) / "tts_output_0.wav"
        _write_audio(output_path, audio_array.numpy(), args.sample_rate)
        print(f"Audio saved to {output_path}")

    print(
        f"CPU Reference: Generation={elapsed:.4f}s | Audio={output_audio_dur:.2f}s | RTF={output_audio_dur / elapsed:.4f}"
    )


def run_non_streaming(inputs: dict[str, Any] | list[dict[str, Any]], args: argparse.Namespace) -> None:
    torch = _import_dependency("torch", "Expected: torch.")
    omni_module = _import_dependency("vllm_omni.entrypoints.omni", "Expected: vllm-omni >= 0.18.0 and vllm.")
    Omni = omni_module.Omni
    _require_vllm_omni_gpu_runtime()

    llm = Omni(model=args.model, log_stats=args.log_stats, stage_configs_path=args.deploy_config)

    start = time.time()
    outputs = llm.generate(inputs, _sampling_params(args.max_tokens, args.cfg_alpha))
    elapsed = time.time() - start

    output_audio_dur = 0.0
    for idx, output in enumerate(outputs):
        audio_tensor = torch.cat(output.multimodal_output["audio"])
        audio_array = audio_tensor.float().detach().cpu().numpy()
        output_audio_dur += len(audio_array) / args.sample_rate

        if args.write_audio:
            output_path = Path(args.output_dir) / f"tts_output_{idx}.wav"
            _write_audio(output_path, audio_array, args.sample_rate)
            print(f"Audio saved to {output_path}")

    print(f"Generation={elapsed:.4f}s | TotalAudio={output_audio_dur:.2f}s | RTF={output_audio_dur / elapsed:.4f}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()


async def run_streaming(inputs: dict[str, Any] | list[dict[str, Any]], args: argparse.Namespace) -> None:
    np = _import_dependency("numpy", "Expected: numpy.")
    torch = _import_dependency("torch", "Expected: torch.")
    vllm_omni = _import_dependency("vllm_omni", "Expected: vllm-omni >= 0.18.0 and vllm.")
    AsyncOmni = vllm_omni.AsyncOmni
    _require_vllm_omni_gpu_runtime()

    async_omni = AsyncOmni(model=args.model, stage_configs_path=args.deploy_config, log_stats=args.log_stats)
    inputs_list = inputs if isinstance(inputs, list) else [inputs]
    sampling_params_list = _sampling_params(args.max_tokens, args.cfg_alpha)
    concurrency = args.concurrency or len(inputs_list)

    total_audio_dur = 0.0
    total_gen_time = 0.0
    total_ttfa = 0.0
    ttfa_count = 0
    results_lock = asyncio.Lock()

    async def generate_one(batch_idx: int, single_input: dict[str, Any]) -> None:
        nonlocal total_audio_dur, total_gen_time, total_ttfa, ttfa_count

        request_id = str(uuid.uuid4())
        all_audio_chunks = []
        accumulated_sample = 0
        chunk_idx = 0
        gen_start = time.time()
        ttfa = None

        async for stage_output in async_omni.generate(
            single_input,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            mm_output = stage_output.multimodal_output
            if not mm_output or "audio" not in mm_output:
                continue

            now = time.time()
            if ttfa is None:
                ttfa = now - gen_start

            audio_chunk = mm_output["audio"]
            if isinstance(audio_chunk, torch.Tensor):
                audio_numpy = audio_chunk[accumulated_sample:].float().detach().cpu().numpy()
            elif isinstance(audio_chunk, list):
                audio_numpy = audio_chunk[chunk_idx].float().detach().cpu().numpy()
            else:
                audio_numpy = audio_chunk

            accumulated_sample += len(audio_numpy)
            all_audio_chunks.append(audio_numpy)
            chunk_idx += 1

        gen_elapsed = time.time() - gen_start
        if not all_audio_chunks:
            return

        full_audio = np.concatenate(all_audio_chunks)
        output_audio_dur = len(full_audio) / args.sample_rate
        if args.write_audio:
            output_path = Path(args.output_dir) / f"tts_output_{batch_idx}.wav"
            _write_audio(output_path, full_audio, args.sample_rate)
            print(f"Request {batch_idx}: audio saved to {output_path}")

        print(
            f"Request {batch_idx}: TTFA={ttfa:.4f}s | Generation={gen_elapsed:.4f}s | "
            f"Audio={output_audio_dur:.2f}s | RTF={output_audio_dur / gen_elapsed:.4f}"
        )

        async with results_lock:
            total_audio_dur += output_audio_dur
            total_gen_time += gen_elapsed
            if ttfa is not None:
                total_ttfa += ttfa
                ttfa_count += 1

    start_all = time.time()
    for wave_start in range(0, len(inputs_list), concurrency):
        wave = inputs_list[wave_start : wave_start + concurrency]
        await asyncio.gather(*(generate_one(wave_start + idx, inp) for idx, inp in enumerate(wave)))

    elapsed_all = time.time() - start_all
    avg_ttfa = total_ttfa / ttfa_count if ttfa_count else float("nan")
    print(
        f"All requests: Generation={elapsed_all:.4f}s | TotalAudio={total_audio_dur:.2f}s | "
        f"AvgTTFA={avg_ttfa:.4f}s | RTF(total)={total_audio_dur / elapsed_all:.4f} | "
        f"RTF(per-request)={total_audio_dur / total_gen_time:.4f}"
    )

    async_omni.shutdown()
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference Voxtral TTS runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (CPU backend, WAV under --output-dir, default model):\n"
            "  python -m models.experimental.voxtraltts.reference.demo_reference \\\n"
            "    --backend cpu --write-audio --text 'Hello world.' --voice casual_male\n"
            "\n"
            "Flags like --write-audio are boolean switches; do not append a literal '...' "
            "(the shell passes it through and argparse reports an unknown argument)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "vllm"),
        default="cpu",
        help="Reference backend. Use cpu for host/TT bring-up; vllm requires NVIDIA CUDA.",
    )
    parser.add_argument("--model", default=DEFAULT_VOXTRAL_MODEL, help="HF repo id or local model directory.")
    parser.add_argument("--text", default="This is a test message.", help="Text to synthesize.")
    parser.add_argument("--voice", default="casual_male", help="Preset voice name. Set to empty with --ref-audio.")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path for voice cloning.")
    parser.add_argument("--output-dir", default="output_audio", help="Directory for wav outputs.")
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Optional vLLM-Omni deploy config override. Defaults to the packaged Voxtral TTS config.",
    )
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of repeated prompts.")
    parser.add_argument("--max-tokens", type=int, default=2500, help="Maximum generated tokens.")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output audio sample rate.")
    parser.add_argument("--cfg-alpha", type=float, default=None, help="Flow-matching CFG alpha override.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16", help="CPU reference dtype.")
    parser.add_argument("--device", default="cpu", help="Torch device for CPU reference.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for flow-matching acoustic sampling.")
    parser.add_argument("--log-stats", action="store_true", help="Enable vLLM stats logging.")
    parser.add_argument("--write-audio", action="store_true", help="Write wav output files.")
    parser.add_argument("--streaming", action="store_true", help="Use AsyncOmni streaming generation.")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent streaming requests per wave.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voice = args.voice or None
    if voice is None and args.ref_audio is None:
        raise ValueError("Either --voice or --ref-audio must be provided.")
    if args.concurrency is not None and not args.streaming:
        raise ValueError("--concurrency requires --streaming.")
    if args.concurrency is not None and args.num_prompts % args.concurrency != 0:
        raise ValueError("--num-prompts must be divisible by --concurrency.")

    config = load_voxtral_config(args.model)
    args.sample_rate = config.audio_model_args.audio_encoding_args.sampling_rate
    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == "cpu":
        run_cpu(args)
        return

    if args.deploy_config is None:
        args.deploy_config = _default_voxtral_deploy_config()

    inputs = compose_speech_request(args.text, args.model, voice=voice, ref_audio=args.ref_audio)
    if args.num_prompts > 1:
        inputs = [inputs] * args.num_prompts

    if args.streaming:
        asyncio.run(run_streaming(inputs, args))
    else:
        run_non_streaming(inputs, args)


if __name__ == "__main__":
    main()
