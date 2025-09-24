# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.perf.benchmarking_utils import BenchmarkProfiler


def _default_mesh_shape() -> ttnn.MeshShape:
    device_ids = ttnn.get_device_ids()
    if len(device_ids) == 32:
        return ttnn.MeshShape(4, 8)
    return ttnn.MeshShape(1, max(1, len(device_ids)))


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DeepSeek-V3 Demo on TT-NN")
    # Prompt is required for full-model mode, optional/ignored for --random-weights
    p.add_argument(
        "prompt", type=str, nargs="?", help="Prompt text (required for full-model mode; ignored with --random-weights)"
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"),
        help="Path to local HF DeepSeek-V3 model (safetensors)",
    )
    p.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    p.add_argument("--cache-dir", type=str, default=os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3"))
    # Random-weights mode options (reuse Model1D pipeline; single dense layer only)
    p.add_argument(
        "--random-weights", action="store_true", help="Use randomly initialized weights instead of loading safetensors"
    )
    p.add_argument(
        "--single-layer",
        choices=["mlp", "moe"],
        help="When using --random-weights, request a single layer (mlp supported)",
    )
    # Teacher-forcing / accuracy verification options
    p.add_argument("--token-accuracy", action="store_true", help="Enable teacher-forced decode and report accuracy")
    p.add_argument(
        "--reference-file",
        type=str,
        help="Path to reference .pt/.refpt file containing 'reference_tokens' and optional 'top5_tokens'",
    )
    p.add_argument(
        "--tf-prompt-len",
        type=int,
        help="Teacher-forcing prompt length in tokens (from reference file). Defaults to half+1 if omitted.",
    )
    # Tracing / performance options
    p.add_argument(
        "--trace",
        action="store_true",
        help=("Enable tracing for decode. No separate code path; tracing is passed to the generator."),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=("Batch size for traced decode benchmark (defaults to DeepSeek MAX_BATCH_SIZE)."),
    )
    return p


def validate_model_path(model_path_str: str, require_safetensors: bool, require_tokenizer: bool) -> None:
    """Validate model path for presence of config, tokenizer (optional), and safetensors (optional)."""
    mp = Path(model_path_str)
    env_hint = (
        "Set DEEPSEEK_V3_HF_MODEL to a directory containing the model files,\n"
        "or pass --model-path /path/to/local/hf/model.\n"
        "Example: export DEEPSEEK_V3_HF_MODEL=/abs/path/to/deepseek-v3"
    )

    if not mp.exists():
        raise SystemExit(f"Model path does not exist: '{mp}'.\n{env_hint}")
    if not mp.is_dir():
        raise SystemExit(f"Model path is not a directory: '{mp}'.\n{env_hint}")

    # Config: always required so AutoConfig can load
    has_config = (mp / "config.json").exists()
    if not has_config:
        raise SystemExit("config.json not found in the model directory.\n" f"Checked: '{mp}'.\n{env_hint}")

    # Tokenizer files: common possibilities (optional in random-weights mode)
    if require_tokenizer:
        has_tokenizer = any(
            (mp / name).exists()
            for name in ("tokenizer.model", "tokenizer.json", "spiece.model", "tokenizer_config.json")
        )
        if not has_tokenizer:
            raise SystemExit(
                "Tokenizer files not found in the model directory. Expected one of: "
                "tokenizer.model, tokenizer.json, spiece.model, tokenizer_config.json.\n"
                f"Checked: '{mp}'.\n{env_hint}"
            )

    if require_safetensors:
        # Weights: require at least one safetensors shard
        has_safetensors = len(glob(str(mp / "*.safetensors"))) > 0
        if not has_safetensors:
            raise SystemExit("No .safetensors files found in the model directory.\n" f"Checked: '{mp}'.\n{env_hint}")


def run_demo(
    prompt: str | None = None,
    *,
    model_path: str | Path | None = None,
    max_new_tokens: int = 32,
    cache_dir: str | Path | None = None,
    random_weights: bool = False,
    single_layer: str | None = None,
    token_accuracy: bool = False,
    trace: bool = False,
    batch_size: int | None = None,
    reference_file: str | Path | None = None,
    tf_prompt_len: int | None = None,
) -> dict:
    """Programmatic entrypoint for the DeepSeek-V3 demo.

    Returns a dict with keys:
        - tokens: List[int] of generated token IDs
        - text: Optional[str] decoded text (only when a tokenizer is present)
    """
    model_path = str(model_path or os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))
    cache_dir = str(cache_dir or os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3"))
    logger.info(f"trace={trace}, model_path='{model_path}', cache_dir='{cache_dir}'")
    # Validate model directory per mode
    validate_model_path(
        model_path,
        require_safetensors=not random_weights,
        require_tokenizer=not random_weights,
    )

    # Open mesh device (reusing test fixture defaults) and set fabric to 1D
    mesh_shape = _default_mesh_shape()
    logger.info("Setting fabric config to FABRIC_1D for demo run")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    logger.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    # Load tokenizer only for full-model mode; in random-weights mode we synthesize token ids
    tokenizer = None
    if not random_weights:
        try:
            tokenizer = load_tokenizer(model_path)
        except Exception:
            logger.error(
                "Failed to load tokenizer from model path. Ensure the directory contains tokenizer files (e.g., tokenizer.model or tokenizer.json)."
            )
            raise

    try:
        # If random single-layer requested with 'moe', fail fast (Model1D demo is MLP-only)
        if random_weights and single_layer and single_layer.lower() == "moe":
            raise SystemExit(
                "--single-layer=moe not supported by Model1D-based demo. Use --single-layer=mlp or drop --random-weights."
            )

        token_acc = None
        if token_accuracy:
            if random_weights:
                raise SystemExit("--token-accuracy requires full-model mode (disable --random-weights)")
            if tokenizer is None:
                raise SystemExit("--token-accuracy requires a tokenizer. Ensure model path has tokenizer files.")
            if not reference_file:
                raise SystemExit("--token-accuracy requires --reference-file pointing to a .pt/.refpt file")

            # Lazy import to avoid overhead when not used
            from models.demos.deepseek_v3.demo.token_accuracy import TokenAccuracy

            token_acc = TokenAccuracy(str(reference_file), prompt_len=tf_prompt_len)

        gen = DeepseekGenerator(
            mesh_device,
            Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
            random_weights=bool(random_weights),
            dense_layers=(1 if random_weights and single_layer else None),
            override_num_layers=(1 if random_weights else None),
            single_layer=(single_layer if random_weights else None),
        )
        # Warn if user passed a different batch_size; generator is initialized with MAX_BATCH_SIZE
        if batch_size is not None and batch_size > 0 and batch_size != gen.batch_size:
            logger.warning(
                f"Requested batch_size={batch_size} but generator is initialized for batch_size={gen.batch_size}. "
                f"Continuing with generator batch_size={gen.batch_size}."
            )

        # Build the prompt list
        if random_weights:
            prompts = [""]
        else:
            if token_acc is not None:
                # Prepare prompt text from reference tokens to align with teacher forcing
                prompts = [token_acc.prepare_ref_tokens(gen.tokenizer)]
                # If not overridden, ensure we don’t decode past the available ground truth
                max_new_tokens = min(max_new_tokens, token_acc.num_gt_tokens())
            else:
                if not prompt:
                    raise SystemExit("A prompt is required unless --random-weights is used.")
                prompts = [prompt]

        # Single-prompt generation
        profiler = BenchmarkProfiler()
        profiler.start("run")
        generations = gen.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            teacher_forcing=token_acc,
            enable_trace=trace,
            profiler=profiler,
        )
        profiler.end("run")
        result = {"tokens": generations[0], "text": None}
        if gen.tokenizer is not None:
            result["text"] = gen.tokenizer.decode(generations[0], skip_special_tokens=True)
        if token_acc is not None:
            acc = token_acc.compute_accuracy()
            result.update({"accuracy_top1": acc.get("top1"), "accuracy_top5": acc.get("top5")})
        # Log performance in normal mode too
        measurements = _summarize_and_log_perf(profiler, gen.batch_size, max_new_tokens)
        result["measurements"] = measurements
        return result
    finally:
        # Clean up mesh device(s)
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        # Reset fabric config back to disabled after the run
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    args = create_parser().parse_args()

    if not args.trace and (not args.random_weights and not args.prompt):
        raise SystemExit("A prompt is required unless --random-weights is used.")

    result = run_demo(
        args.prompt,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        random_weights=bool(args.random_weights),
        single_layer=args.single_layer,
        token_accuracy=bool(args.token_accuracy),
        trace=bool(args.trace),
        batch_size=args.batch_size,
        reference_file=args.reference_file,
        tf_prompt_len=args.tf_prompt_len,
    )

    print("\n===== Generated =====\n")
    if result.get("text") is not None:
        print(result["text"])  # type: ignore
    else:
        print("[random-weights mode] token IDs:")
        print(result["tokens"])  # type: ignore
    print("\n=====================\n")


def _summarize_and_log_perf(profiler: BenchmarkProfiler, B: int, max_new_tokens: int) -> dict:
    """Summarize decode performance from a BenchmarkProfiler and log nicely.

    Returns a dict with compile/inference timing summaries and throughput.
    """
    # Aggregate decode iterations that exist
    decode_durations = []
    for i in range(1, max_new_tokens + 1):
        step = f"inference_decode_time_{i}"
        if profiler.contains_step(step):
            decode_durations.append(profiler.get_duration(step))

    total_decode_time = sum(decode_durations)
    num_measured = len(decode_durations)
    avg_decode_it = (total_decode_time / num_measured) if num_measured > 0 else 0.0
    decode_tok_s_u = (num_measured / total_decode_time) if total_decode_time > 0 else 0.0
    decode_tok_s = decode_tok_s_u * B

    compile_prefill_time = (
        profiler.get_duration("compile_prefill") if profiler.contains_step("compile_prefill") else 0.0
    )
    compile_decode_time = profiler.get_duration("compile_decode") if profiler.contains_step("compile_decode") else 0.0
    total_inference_prefill_time = (
        profiler.get_duration("inference_prefill") if profiler.contains_step("inference_prefill") else 0.0
    )
    total_inference_decode_time = (
        profiler.get_duration("inference_decode") if profiler.contains_step("inference_decode") else total_decode_time
    )

    tok_1 = profiler.get_duration("inference_decode_time_1") if profiler.contains_step("inference_decode_time_1") else 0
    tok_128 = (
        profiler.get_duration("inference_decode_time_127") if profiler.contains_step("inference_decode_time_127") else 0
    )
    tok_1024 = (
        profiler.get_duration("inference_decode_time_1023")
        if profiler.contains_step("inference_decode_time_1023")
        else 0
    )

    logger.info("")
    logger.info("=== Performance metrics (decode) ===")
    if tok_1 > 0:
        logger.info(
            f"1st token decode time: {tok_1*1000:.2f}ms [{round(1/tok_1, 2)} t/s/u, {round((1/tok_1)*B, 2)} t/s]"
        )
    if tok_128 > 0:
        logger.info(
            f"128th token decode time: {tok_128*1000:.2f}ms [{round(1/tok_128, 2)} t/s/u, {round((1/tok_128)*B, 2)} t/s]"
        )
    if tok_1024 > 0:
        logger.info(
            f"1024th token decode time: {tok_1024*1000:.2f}ms [{round(1/tok_1024, 2)} t/s/u, {round((1/tok_1024)*B, 2)} t/s]"
        )
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(
        f"Average speed: {round(avg_decode_it * 1000, 2)}ms @ {round(decode_tok_s_u, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )

    return {
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "decode_t/s/u": decode_tok_s_u,
        "decode_t/s": decode_tok_s,
    }


if __name__ == "__main__":
    main()
