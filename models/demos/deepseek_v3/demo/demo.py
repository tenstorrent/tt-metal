# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator as DeepseekGeneratorDP
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape


def _print_performance_metrics(results: dict) -> None:
    """Print performance metrics from results if available."""
    if "statistics" in results and results["statistics"]:
        statistics = results["statistics"]
        logger.info("=== Performance Metrics ===")
        logger.info(f"Config preparation - Prefill: {statistics.get('preparing_prefill_config', 0)*1000:.2f}ms")
        logger.info(f"Config preparation - Decode: {statistics.get('preparing_decode_config', 0)*1000:.2f}ms")
        logger.info(f"Prefill time: {statistics['inference_prefill']*1000:.2f}ms")
        logger.info(f"Average time to first token: {statistics['prefill_time_to_token']*1000:.2f}ms")
        logger.info(f"Prefill tokens/sec: {statistics['prefill_t/s']:.2f}")
        logger.info(f"Decode tokens/sec/user: {statistics['decode_t/s/u']:.2f}")
        logger.info(f"Decode tokens/sec (total): {statistics['decode_t/s']:.2f}")
        trace_metric = statistics["trace_execution_t/s/u @128th token"]
        trace_str = f"{trace_metric:.2f}" if trace_metric is not None else "N/A (requires --max-new-tokens >= 128)"
        logger.info(f"Trace execution tokens/sec/user @128th token: {trace_str}")
        logger.info(f"Full demo runtime: {statistics['Full demo runtime']:.2f}s")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DeepSeek-V3 Demo on TT-NN")
    # Prompt is required for full-model mode, optional/ignored for --random-weights
    p.add_argument(
        "prompts",
        type=str,
        nargs="*",
        help="Prompt text(s) (required for full-model mode; ignored with --random-weights). Can pass multiple prompts.",
    )
    p.add_argument(
        "--prompts-file",
        type=str,
        help="Path to JSON file containing prompts. The JSON should have a 'prompts' array with objects containing a 'prompt' field. If provided, all prompts from the file will be used.",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        help="Maximum number of prompts to load from the JSON file. If not specified, all prompts will be used.",
    )
    p.add_argument(
        "--output-path",
        type=str,
        help="Path to output JSON file. If --prompts-file is provided and --output-path is not specified, output will be saved to <prompts-file-stem>_output.json in the same directory as the prompts file.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local HF DeepSeek-V3 model (safetensors)",
    )
    p.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    p.add_argument("--cache-dir", type=str, required=True)
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
    p.add_argument(
        "--early_print_first_user",
        action="store_true",
        default=False,
        help="Print generated tokens for the first user token as they are produced, instead of waiting until the end.",
    )
    p.add_argument(
        "--generator",
        choices=["bp"],
        default="bp",
        help="Select generator implementation: default = bp (batch parallel).",
    )
    p.add_argument(
        "--enable-trace",
        action="store_true",
        default=False,
        help="Enable trace for decode forward pass",
    )
    p.add_argument(
        "--repeat-batches",
        type=int,
        default=1,
        help="Number of times to repeat the generation process.",
    )
    return p


def load_prompts_from_json(json_file_path: str, max_prompts: int | None = None) -> list[str]:
    """Load prompts from a JSON file.

    Supports two JSON formats:
    1. Array format: [{"prompt": "..."}, {"prompt": "..."}, ...]
    2. Object format: {"prompts": [{"prompt": "..."}, ...]}

    Args:
        json_file_path: Path to the JSON file containing prompts
        max_prompts: Maximum number of prompts to load. If None, loads all prompts.

    Returns a list of prompt strings.
    """
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise SystemExit(f"Prompts file does not exist: '{json_path}'")
    if not json_path.is_file():
        raise SystemExit(f"Prompts path is not a file: '{json_path}'")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON file '{json_path}': {e}")
    except Exception as e:
        raise SystemExit(f"Failed to read prompts file '{json_path}': {e}")

    # Handle both array format and object format with "prompts" key
    if isinstance(data, list):
        prompt_items = data
    elif isinstance(data, dict) and "prompts" in data:
        prompt_items = data["prompts"]
    else:
        raise SystemExit(
            f"JSON file '{json_path}' must be either an array of prompt objects "
            f"or an object with a 'prompts' key containing an array"
        )

    prompts = []
    for item in prompt_items:
        if max_prompts is not None and len(prompts) >= max_prompts:
            break
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid prompt item (not a dict): {item}")
            continue
        if "prompt" not in item:
            logger.warning(f"Skipping prompt item missing 'prompt' field: {item}")
            continue
        prompts.append(str(item["prompt"]))

    if not prompts:
        raise SystemExit(f"No valid prompts found in '{json_path}'")

    logger.info(
        f"Loaded {len(prompts)} prompts from '{json_path}'"
        + (f" (limited to {max_prompts})" if max_prompts is not None else "")
    )
    return prompts


def validate_model_path(model_path_str: str, require_safetensors: bool, require_tokenizer: bool) -> None:
    """Validate model path for presence of config, tokenizer (optional), and safetensors (optional)."""
    mp = Path(model_path_str)

    if not mp.exists():
        raise SystemExit(f"Model path does not exist: '{mp}'.")
    if not mp.is_dir():
        raise SystemExit(f"Model path is not a directory: '{mp}'.")

    # Config: always required so AutoConfig can load
    has_config = (mp / "config.json").exists()
    if not has_config:
        raise SystemExit(f"config.json not found in the model directory. Checked: '{mp}'.")

    # Tokenizer files: common possibilities (optional in random-weights mode)
    if require_tokenizer:
        has_tokenizer = any(
            (mp / name).exists()
            for name in ("tokenizer.model", "tokenizer.json", "spiece.model", "tokenizer_config.json")
        )
        if not has_tokenizer:
            raise SystemExit(
                "Tokenizer files not found in the model directory. Expected one of: "
                "tokenizer.model, tokenizer.json, spiece.model, tokenizer_config.json. "
                f"Checked: '{mp}'."
            )

    if require_safetensors:
        # Weights: require at least one safetensors shard
        has_safetensors = len(glob(str(mp / "*.safetensors"))) > 0
        if not has_safetensors:
            raise SystemExit("No .safetensors files found in the model directory. " f"Checked: '{mp}'.")


def run_demo(
    prompts: list[str] | None = None,
    *,
    model_path: str | Path | None = None,
    max_new_tokens: int = 32,
    cache_dir: str | Path | None = None,
    random_weights: bool = False,
    single_layer: str | None = None,
    token_accuracy: bool = False,
    reference_file: str | Path | None = None,
    tf_prompt_len: int | None = None,
    early_print_first_user: bool = True,
    generator: str = "bp",
    enable_trace: bool = False,
    override_num_layers: int | None = None,
    repeat_batches: int = 1,
) -> dict:
    """Programmatic entrypoint for the DeepSeek-V3 demo.

    Returns a dict with keys:
        - tokens: List[int] of generated token IDs
        - text: Optional[str] decoded text (only when a tokenizer is present)
    """
    if model_path is None:
        raise SystemExit("Missing model path. Provide --model-path.")
    model_path = Path(model_path)

    if cache_dir is None:
        raise SystemExit("Missing cache directory. Provide --cache-dir.")
    cache_dir = Path(cache_dir)

    # Validate model directory per mode
    validate_model_path(
        str(model_path),
        require_safetensors=not random_weights,
        require_tokenizer=not random_weights,
    )

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    logger.info(f"Selected MESH_DEVICE: '{requested_system_name}' - mesh shape will be set to: {mesh_shape}")

    fabric_config = ttnn.FabricConfig.FABRIC_1D
    logger.info(f"Setting fabric config to {fabric_config} for demo run")
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)

    logger.info(f"Opening mesh device with shape {mesh_shape}")
    if enable_trace:
        logger.info("Enabling trace for decode forward pass")
        trace_region_size = 4880384 + int(0.20 * 4880384)  # 20% additional
        logger.info(f"Trace region size set to {trace_region_size}")
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, trace_region_size=trace_region_size)
    else:
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
        if generator == "bp":
            gen = DeepseekGeneratorDP(
                mesh_device=mesh_device,
                model_path=model_path,
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                random_weights=bool(random_weights),
                dense_layers=(1 if random_weights and single_layer else None),
                override_num_layers=(
                    override_num_layers if override_num_layers is not None else (1 if random_weights else None)
                ),
                single_layer=(single_layer if random_weights else None),
                enable_trace=enable_trace,
            )
        # Build the prompt list
        pre_tokenized_prompts = None
        if random_weights:
            prompt_list = [""]
        else:
            if token_acc is not None:
                # Use pre-tokenized tokens directly to avoid re-encoding with chat template.
                # This ensures the TT model uses the exact same token sequence as the reference.
                pre_tokenized_prompts = [token_acc.get_prompt_token_ids()]
                # Still need a placeholder prompt for the generator API
                prompt_list = [""]
                # If not overridden, ensure we don't decode past the available ground truth
                max_new_tokens = min(max_new_tokens, token_acc.num_gt_tokens())
            else:
                if not prompts:
                    raise SystemExit("A prompt is required unless --random-weights is used.")
                prompt_list = prompts

        # Multi-prompt generation
        generations, statistics = gen.generate(
            prompt_list,
            max_new_tokens=max_new_tokens,
            teacher_forcing=token_acc,
            early_print_first_user=early_print_first_user,
            repeat_batches=repeat_batches,
            pre_tokenized=pre_tokenized_prompts,
        )

        # Process all generations
        results = []
        for i, generation_tokens in enumerate(generations):
            result = {"tokens": generation_tokens, "text": None}
            if gen.tokenizer is not None:
                result["text"] = gen.tokenizer.decode(generation_tokens, skip_special_tokens=True)
            if token_acc is not None and i == 0:  # Only compute accuracy for first generation
                acc = token_acc.compute_accuracy()
                result.update(
                    {
                        "accuracy_top1": acc.get("top1"),
                        "accuracy_top5": acc.get("top5"),
                        "predicted_tokens": token_acc._pred_tokens,
                    }
                )
            results.append(result)

        return {"generations": results, "statistics": statistics}
    finally:
        # Clean up generator resources
        try:
            gen.cleanup_all()
        except Exception as e:
            logger.warning(f"Failed to cleanup generator: {e}")
        # Clean up mesh device(s)
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        # Reset fabric config back to disabled after the run
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    args = create_parser().parse_args()

    # Load prompts from JSON file if provided
    prompts_file_path = None
    if args.prompts_file:
        prompts_file_path = Path(args.prompts_file)
        json_prompts = load_prompts_from_json(args.prompts_file, max_prompts=args.num_prompts)
        # Merge with command-line prompts if any, JSON prompts take precedence if both provided
        if args.prompts:
            logger.info(
                f"Both --prompts-file and command-line prompts provided. Using {len(json_prompts)} prompts from JSON file (command-line prompts ignored)."
            )
        args.prompts = json_prompts

    if not args.random_weights and not args.prompts:
        raise SystemExit("A prompt is required unless --random-weights is used.")

    results = run_demo(
        args.prompts,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        random_weights=bool(args.random_weights),
        single_layer=args.single_layer,
        token_accuracy=bool(args.token_accuracy),
        reference_file=args.reference_file,
        tf_prompt_len=args.tf_prompt_len,
        early_print_first_user=args.early_print_first_user,
        generator=args.generator,
        enable_trace=args.enable_trace,
    )

    # If prompts were loaded from a JSON file, save output to JSON file instead of printing
    if prompts_file_path:
        # Use provided output path, or generate default: input_name + "_output.json"
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = prompts_file_path.parent / f"{prompts_file_path.stem}_output.json"

        # Prepare output data structure
        output_data = {
            "prompts": args.prompts if args.prompts else [],
            "generations": [],
            "statistics": results.get("statistics", {}),
        }

        # Add generation results
        for i, gen_result in enumerate(results["generations"]):
            prompt_text = ""
            if args.prompts is not None and i < len(args.prompts):
                prompt_text = args.prompts[i]
            elif args.random_weights:
                prompt_text = "[random-weights default prompt]"

            output_data["generations"].append(
                {
                    "index": i + 1,
                    "prompt": prompt_text if prompt_text else "[empty prompt]",
                    "text": gen_result.get("text"),
                }
            )

        # Write to JSON file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to '{output_path}'")
            print(f"\nResults saved to '{output_path}'\n")
        except Exception as e:
            raise SystemExit(f"Failed to write output file '{output_path}': {e}")
    else:
        # Print to terminal as before
        print("\n===== Generated =====\n")

        for i, gen_result in enumerate(results["generations"]):
            prompt_text = ""
            if args.prompts is not None and i < len(args.prompts):
                prompt_text = args.prompts[i]
            elif args.random_weights:
                prompt_text = "[random-weights default prompt]"

            print("-" * 30)
            if prompt_text:
                print(f"Prompt[{i+1}]: {prompt_text}")
            else:
                print(f"Prompt[{i+1}]: [empty prompt]")
            print(f"Generation[{i+1}]:")
            if gen_result.get("text") is not None:
                print(gen_result["text"])  # type: ignore
            else:
                print("[random-weights mode] token IDs:")
                print(gen_result["tokens"])  # type: ignore
            print("-" * 30)

        print("=====================\n")

        # Print performance metrics if available
        _print_performance_metrics(results)

    # Print performance metrics if available
    _print_performance_metrics(results)


if __name__ == "__main__":
    main()
