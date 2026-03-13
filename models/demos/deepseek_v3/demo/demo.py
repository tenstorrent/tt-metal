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
from models.common.sampling.sampling_params import SamplingParams
from models.demos.deepseek_v3.tt.generator import DEFAULT_MAX_SEQ_LEN
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator as DeepseekGeneratorDP
from models.demos.deepseek_v3.utils.config_helpers import (
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_K,
    DEFAULT_SAMPLING_TOP_P,
    USERS_PER_ROW,
    get_fabric_config,
)
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape


def _prompt_text_for_index(prompts: list[str] | None, random_weights: bool, index: int) -> str:
    if prompts is not None and index < len(prompts):
        return prompts[index]
    if random_weights:
        return "[random-weights default prompt]"
    return "[empty prompt]"


def _build_output_data(
    prompts: list[str] | None,
    generations: list[dict],
    statistics: dict,
    model_params: dict,
    random_weights: bool,
) -> dict:
    output_data = {
        "prompts": prompts if prompts else [],
        "generations": [],
        "statistics": statistics,
        "model_params": model_params,
    }
    for i, gen_result in enumerate(generations):
        output_data["generations"].append(
            {
                "index": i + 1,
                "prompt": _prompt_text_for_index(prompts, random_weights, i),
                "text": gen_result.get("text"),
            }
        )
    return output_data


def _resolve_saved_output_path(prompts_file_path: Path | None, output_path_arg: str | None) -> Path | None:
    if output_path_arg:
        return Path(output_path_arg)
    if prompts_file_path is not None:
        return prompts_file_path.parent / f"{prompts_file_path.stem}_output.json"
    return None


def _write_json_output(path: Path, payload: dict, label: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"{label} saved to '{path}'")
        print(f"\n{label} saved to '{path}'\n")
    except Exception as e:
        raise SystemExit(f"Failed to write {label.lower()} '{path}': {e}")


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


def _format_model_params_for_reporting(model_params: dict, summarize_sampling: bool = True) -> dict:
    """Summarize sampling arrays for concise reporting."""
    if not summarize_sampling:
        return model_params

    sampling = model_params.get("sampling")
    if not isinstance(sampling, dict):
        return model_params

    formatted_sampling = {}
    for key, value in sampling.items():
        if isinstance(value, (list, tuple)):
            if value and all(v == value[0] for v in value):
                formatted_sampling[key] = {"same_value_all_users": value[0], "count": len(value)}
            else:
                formatted_sampling[key] = {
                    "same_value_all_users": False,
                    "note": "all values are not same",
                    "first_3_values": list(value[:3]),
                    "count": len(value),
                }
        else:
            formatted_sampling[key] = value

    return {**model_params, "sampling": formatted_sampling}


def _print_model_params(results: dict, summarize_sampling: bool = True) -> None:
    """Print model parameters from model_params."""
    if "model_params" in results and results["model_params"]:
        logger.info("=== Model Parameters ===")
        model_params = _format_model_params_for_reporting(
            results["model_params"], summarize_sampling=summarize_sampling
        )
        for key in sorted(model_params):
            logger.info(f"{key}: {model_params[key]}")
        logger.info("=====================")


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
        "--checkpoint-jsonl",
        type=str,
        help="Optional JSONL path for appending per-user results during generation.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local HF DeepSeek-V3 model (safetensors)",
    )
    p.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Maximum configured sequence length for the demo runtime (default: {DEFAULT_MAX_SEQ_LEN}).",
    )
    p.add_argument(
        "--sampling-temperature",
        type=float,
        default=DEFAULT_SAMPLING_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_SAMPLING_TEMPERATURE}).",
    )
    p.add_argument(
        "--sampling-top-k",
        type=int,
        default=DEFAULT_SAMPLING_TOP_K,
        help=f"Top-k value for sampling (default: {DEFAULT_SAMPLING_TOP_K}).",
    )
    p.add_argument(
        "--sampling-top-p",
        type=float,
        default=DEFAULT_SAMPLING_TOP_P,
        help=f"Top-p value for sampling (default: {DEFAULT_SAMPLING_TOP_P}).",
    )
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
    p.add_argument(
        "--override-num-layers",
        type=int,
        help="Override the number of layers in the model. Defaults to None.",
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
        "--disable-trace",
        action="store_false",
        dest="enable_trace",
        default=True,
        help="Disable trace for decode forward pass.",
    )
    p.add_argument(
        "--enable-mem-profile",
        action="store_true",
        default=False,
        help="Enable TTNN memory profiling dumps during setup",
    )
    p.add_argument(
        "--mtp",
        choices=["on", "off"],
        default="off",
        help="Control MTP usage: on (requires MTP weights), off (default).",
    )
    p.add_argument(
        "--repeat-batches",
        type=int,
        default=1,
        help="Number of times to repeat the generation process.",
    )
    p.add_argument(
        "--signpost",
        action="store_true",
        help="Enable signpost for tracing.",
    )
    p.add_argument(
        "--prefill-max-tokens",
        type=int,
        help="Maximum number of tokens to prefill.",
    )
    p.add_argument(
        "--profile-decode",
        action="store_true",
        default=False,
        help="Profile decode performance: skip prefill (use random tokens), and run only first dense layer + first MoE layer during decode.",
    )
    p.add_argument(
        "--sample-on-host",
        action="store_false",
        dest="sample_on_device",
        default=True,
        help="Disable on-device sampling and use host-side sampling.",
    )
    p.add_argument(
        "--force-recalculate",
        "--recalculate-weights",
        dest="force_recalculate",
        action="store_true",
        default=False,
        help="Force regeneration of cached TTNN weight files and config.",
    )
    p.add_argument(
        "--stop-at-eos",
        dest="stop_at_eos",
        action="store_true",
        help="Stop recording output tokens for a user after EOS (default).",
    )
    p.add_argument(
        "--no-stop-at-eos",
        dest="stop_at_eos",
        action="store_false",
        help="Always record max-new-tokens even after EOS.",
    )
    p.set_defaults(stop_at_eos=True)
    p.add_argument(
        "--max-users-per-row",
        type=int,
        default=USERS_PER_ROW,
        help=f"Maximum number of active users per row for demo decode (default: {USERS_PER_ROW}).",
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
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_users_per_row: int = USERS_PER_ROW,
    cache_dir: str | Path | None = None,
    random_weights: bool = False,
    single_layer: str | None = None,
    override_num_layers: int | None = None,
    token_accuracy: bool = False,
    reference_file: str | Path | None = None,
    tf_prompt_len: int | None = None,
    early_print_first_user: bool = True,
    generator: str = "bp",
    enable_trace: bool = True,
    enable_mem_profile: bool = False,
    repeat_batches: int = 1,
    signpost: bool = False,
    prefill_max_tokens: int = None,
    profile_decode: bool = False,
    sample_on_device: bool = True,
    force_recalculate: bool = False,
    stop_at_eos: bool = True,
    checkpoint_jsonl: str | Path | None = None,
    enable_mtp: bool = False,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    sampling_top_k: int = DEFAULT_SAMPLING_TOP_K,
    sampling_top_p: float = DEFAULT_SAMPLING_TOP_P,
) -> dict:
    """Programmatic entrypoint for the DeepSeek-V3 demo.

    Returns a dict with keys:
        - generations: List[dict] with per-prompt tokens/text
        - statistics: Performance counters from the generator
    """
    if model_path is None:
        raise SystemExit("Missing model path. Provide --model-path.")
    model_path = Path(model_path)

    if cache_dir is None:
        raise SystemExit("Missing cache directory. Provide --cache-dir.")
    cache_dir = Path(cache_dir)

    if sampling_temperature < 0:
        raise SystemExit("--sampling-temperature must be >= 0 (use 0 for greedy decoding).")
    if not (0.0 < sampling_top_p <= 1.0):
        raise SystemExit("--sampling-top-p must be in the interval (0, 1].")
    if sampling_top_k < 0:
        raise SystemExit(
            "--sampling-top-k must be >= 0. For top-k=0, use --sample-on-host. See https://github.com/tenstorrent/tt-metal/issues/40236"
        )
    if sampling_top_k == 0 and sample_on_device:
        raise SystemExit(
            "--sampling-top-k=0 is not supported when sampling on device. Use --sample-on-host. See https://github.com/tenstorrent/tt-metal/issues/40236"
        )

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
    fabric_config = get_fabric_config()
    logger.info(f"Setting fabric config to {fabric_config} for demo run")
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)

    logger.info(f"Opening mesh device with shape {mesh_shape}")
    if enable_trace:
        logger.info("Enabling trace for decode forward pass")
        # NOTE:
        # The base trace region size below (~36.3 MiB) was empirically determined from
        # vLLM decode workloads to be sufficient to keep the trace buffer from
        # overflowing under typical DeepSeek-V3 demo settings (batch size, sequence
        # length, and mesh configuration). We add 20% headroom as a conservative
        # safety margin to accommodate variability across models / prompts without
        # repeatedly re-tuning this value.
        #
        # If you are optimizing memory usage, this can be reduced after verifying
        # that tracing completes without buffer exhaustion for your target workload.
        BASE_TRACE_REGION_BYTES = 38_070_272
        trace_region_size = BASE_TRACE_REGION_BYTES + int(0.20 * BASE_TRACE_REGION_BYTES)
        if enable_mtp:
            trace_region_size = max(trace_region_size, 134_217_728)
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

    batch_size_per_row = USERS_PER_ROW
    batch_size = batch_size_per_row * mesh_device.shape[0]

    # Configure sampling
    # sampling values of all users are assumed to be the same when initialized with run_demo function.
    sampling_params = SamplingParams(
        temperature=[sampling_temperature] * batch_size,
        top_p=[sampling_top_p] * batch_size,
        top_k=[sampling_top_k] * batch_size,
    )

    gen = None
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
                enable_mem_profile=enable_mem_profile,
                signpost=signpost,
                max_seq_len=max_seq_len,
                prefill_max_tokens=prefill_max_tokens,
                force_recalculate=force_recalculate,
                profile_decode=profile_decode,
                sample_on_device=sample_on_device,
                enable_mtp=enable_mtp,
                batch_size_per_row=max_users_per_row,
                sampling_params=sampling_params,
            )
        else:
            raise ValueError(f"Unsupported generator: {generator}")
        # Build the prompt list
        pre_tokenized_prompts = None
        if random_weights:
            prompt_list = [""]
        else:
            if token_acc is not None:
                # Use pre-tokenized tokens directly to avoid re-encoding with chat template.
                # This ensures the TT model uses the exact same token sequence as the reference.
                if prompts:
                    prompt_list = prompts
                else:
                    # Still need a placeholder prompt for the generator API
                    prompt_list = [""]
                pre_tokenized_prompts = [token_acc.get_prompt_token_ids() for _ in range(len(prompt_list))]
                # If not overridden, ensure we don't decode past the available ground truth
                max_new_tokens = min(max_new_tokens, token_acc.num_gt_tokens())
            else:
                if not prompts:
                    raise SystemExit("A prompt is required unless --random-weights is used.")
                prompt_list = prompts

        checkpoint_fh = None
        checkpoint_written: set[int] = set()
        checkpoint_path = Path(checkpoint_jsonl) if checkpoint_jsonl is not None else None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_fh = open(checkpoint_path, "w", encoding="utf-8")
            if not stop_at_eos:
                logger.info(
                    "checkpoint-jsonl is enabled without stop-at-eos; records will be written after generation."
                )

        def checkpoint_user(user_idx: int, token_ids: list[int]) -> None:
            if checkpoint_fh is None or user_idx in checkpoint_written:
                return
            text = None
            if gen.tokenizer is not None:
                text = gen.tokenizer.decode(token_ids, skip_special_tokens=True)
            record = {
                "index": user_idx + 1,
                "prompt": prompt_list[user_idx] if user_idx < len(prompt_list) else "",
                "text": text,
            }
            checkpoint_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_fh.flush()
            os.fsync(checkpoint_fh.fileno())
            checkpoint_written.add(user_idx)

        def make_checkpoint_callback(index_offset: int):
            if checkpoint_fh is None:
                return None

            def checkpoint_user_with_offset(user_idx: int, token_ids: list[int]) -> None:
                checkpoint_user(index_offset + user_idx, token_ids)

            return checkpoint_user_with_offset

        try:
            # Multi-prompt generation
            use_mtp_path = gen.enable_mtp and token_acc is None and max_new_tokens > 1
            max_prompts_per_batch = gen.batch_size
            if use_mtp_path:
                max_prompts_per_batch = max(1, gen.batch_size // 2)

            if use_mtp_path and len(prompt_list) > max_prompts_per_batch:
                logger.info(
                    f"MTP enabled with {len(prompt_list)} prompts; running in batches of up to {max_prompts_per_batch} "
                    "to reserve lanes for verify batching."
                )
                all_generations = []
                all_stats = []
                for start in range(0, len(prompt_list), max_prompts_per_batch):
                    batch_prompts = prompt_list[start : start + max_prompts_per_batch]
                    batch_pre_tokenized = (
                        pre_tokenized_prompts[start : start + max_prompts_per_batch]
                        if pre_tokenized_prompts is not None
                        else None
                    )
                    batch_generations, batch_stats, model_params = gen.generate(
                        batch_prompts,
                        max_new_tokens=max_new_tokens,
                        teacher_forcing=token_acc,
                        early_print_first_user=early_print_first_user,
                        repeat_batches=repeat_batches,
                        pre_tokenized=batch_pre_tokenized,
                        stop_at_eos=stop_at_eos,
                        on_user_finished=make_checkpoint_callback(start),
                    )
                    all_generations.extend(batch_generations)
                    all_stats.append(batch_stats)

                generations = all_generations
                statistics = all_stats[-1] if all_stats else {}
                if all_stats:
                    statistics["batch_count"] = len(all_stats)
                    mtp_accepts = [s.get("mtp_accepts") for s in all_stats if s.get("mtp_accepts") is not None]
                    mtp_verifies = [s.get("mtp_verifies") for s in all_stats if s.get("mtp_verifies") is not None]
                    if mtp_accepts and mtp_verifies:
                        total_accepts = sum(int(x) for x in mtp_accepts)
                        total_verifies = sum(int(x) for x in mtp_verifies)
                        statistics["mtp_accepts"] = total_accepts
                        statistics["mtp_accept_rate"] = total_accepts / total_verifies if total_verifies > 0 else 0.0
                    else:
                        mtp_rates = [
                            s.get("mtp_accept_rate") for s in all_stats if s.get("mtp_accept_rate") is not None
                        ]
                        if mtp_rates:
                            statistics["mtp_accept_rate"] = sum(mtp_rates) / len(mtp_rates)
                    for key in (
                        "preparing_prefill_config",
                        "preparing_decode_config",
                        "inference_prefill",
                        "inference_decode",
                        "decode_forward_passes",
                        "Full demo runtime",
                    ):
                        if any(key in s for s in all_stats):
                            statistics[key] = sum(float(s.get(key, 0) or 0) for s in all_stats)
            else:
                generations, statistics, model_params = gen.generate(
                    prompt_list,
                    max_new_tokens=max_new_tokens,
                    teacher_forcing=token_acc,
                    early_print_first_user=early_print_first_user,
                    repeat_batches=repeat_batches,
                    pre_tokenized=pre_tokenized_prompts,
                    stop_at_eos=stop_at_eos,
                    on_user_finished=make_checkpoint_callback(0),
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

            if checkpoint_fh is not None:
                for i, result in enumerate(results):
                    if i in checkpoint_written:
                        continue
                    record = {
                        "index": i + 1,
                        "prompt": prompt_list[i] if i < len(prompt_list) else "",
                        "text": result["text"],
                    }
                    checkpoint_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint_fh.flush()
                os.fsync(checkpoint_fh.fileno())

            return {"generations": results, "statistics": statistics, "model_params": model_params}
        finally:
            if checkpoint_fh is not None:
                checkpoint_fh.close()
    finally:
        # Clean up generator resources
        try:
            if gen is not None:
                gen.cleanup_all()
        except Exception as e:
            logger.warning(f"Failed to cleanup generator: {e}")
        # Synchronize device before closing to flush pending ops (e.g. profiler data)
        ttnn.synchronize_device(mesh_device)
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
        max_seq_len=args.max_seq_len,
        max_users_per_row=args.max_users_per_row,
        cache_dir=args.cache_dir,
        random_weights=bool(args.random_weights),
        single_layer=args.single_layer,
        override_num_layers=args.override_num_layers,
        token_accuracy=bool(args.token_accuracy),
        reference_file=args.reference_file,
        tf_prompt_len=args.tf_prompt_len,
        early_print_first_user=args.early_print_first_user,
        generator=args.generator,
        enable_trace=args.enable_trace,
        enable_mem_profile=args.enable_mem_profile,
        signpost=args.signpost,
        prefill_max_tokens=args.prefill_max_tokens,
        profile_decode=args.profile_decode,
        sample_on_device=args.sample_on_device,
        force_recalculate=bool(args.force_recalculate),
        sampling_temperature=args.sampling_temperature,
        sampling_top_k=args.sampling_top_k,
        sampling_top_p=args.sampling_top_p,
        stop_at_eos=bool(args.stop_at_eos),
        checkpoint_jsonl=args.checkpoint_jsonl,
        enable_mtp=(args.mtp == "on"),
    )

    saved_output_path = _resolve_saved_output_path(prompts_file_path, args.output_path)

    # If prompts were loaded from a JSON file, save output to JSON file instead of printing.
    # Only host rank 0 writes the shared file to avoid rank races corrupting the JSON.
    if prompts_file_path and saved_output_path is not None:
        output_data = _build_output_data(
            prompts=args.prompts,
            generations=results["generations"],
            statistics=results.get("statistics", {}),
            model_params=_format_model_params_for_reporting(
                results.get("model_params", {}),
            ),
            random_weights=bool(args.random_weights),
        )
        if int(os.getenv("TT_MESH_HOST_RANK", "0")) == 0:
            _write_json_output(saved_output_path, output_data, "Results")
    else:
        # Print to terminal as before
        print("\n===== Generated =====\n")

        for i, gen_result in enumerate(results["generations"]):
            print("-" * 30)
            print(f"Prompt[{i+1}]: {_prompt_text_for_index(args.prompts, bool(args.random_weights), i)}")
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

    # Print model parameters if available
    _print_model_params(results)


if __name__ == "__main__":
    main()
