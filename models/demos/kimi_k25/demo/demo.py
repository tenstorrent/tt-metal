# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""demo.py — Kimi K2.5 inference demo on Tenstorrent hardware.

Runs Kimi K2.5 generation on a Tenstorrent Galaxy (TG), Quad, or Dual mesh
using the tt-metal TTNN software stack.

Usage examples::

    # Full-model inference with real weights
    MESH_DEVICE=TG python demo.py "What is 2+2?" \\
        --model-path /workspace/extra/Kimi-K2.5 \\
        --cache-dir  /workspace/extra/kimi_cache

    # Multiple prompts
    MESH_DEVICE=TG python demo.py \\
        "Explain quantum entanglement in simple terms." \\
        "Write a haiku about silicon chips." \\
        --model-path /workspace/extra/Kimi-K2.5 \\
        --cache-dir  /workspace/extra/kimi_cache \\
        --max-new-tokens 128

    # Prompts from JSON file
    MESH_DEVICE=TG python demo.py \\
        --prompts-file test_prompts.json \\
        --output-path results.json \\
        --model-path /workspace/extra/Kimi-K2.5 \\
        --cache-dir  /workspace/extra/kimi_cache

    # Random-weights smoke test (no safetensors needed — validates hardware pipeline)
    MESH_DEVICE=TG python demo.py \\
        --random-weights \\
        --model-path /workspace/extra/Kimi-K2.5 \\
        --cache-dir  /workspace/extra/kimi_cache \\
        --override-num-layers 2

Requirements:
    - ``MESH_DEVICE`` env var: ``TG`` (32-chip Galaxy), ``QUAD`` (8-chip), or
      ``DUAL`` (4-chip Wormhole pair).
    - ``--model-path`` pointing to the Kimi K2.5 checkpoint directory (must
      contain ``model.safetensors.index.json`` and the 64 shard ``.safetensors``
      files, plus ``tokenizer.json`` and ``config.json``).
    - ``--cache-dir`` for converted TTNN tensors.  First run converts (slow);
      subsequent runs reload from cache (fast).  Pass ``--force-recalculate``
      to regenerate.

Kimi K2.5 architecture facts (for debugging)::

    layers      = 61  (1 dense MLP + 60 MoE)
    experts     = 384 (12 per chip on TG)
    attn_heads  = 64
    hidden_dim  = 7168
    vocab_size  = 163840
    min_target  = Galaxy TG (32×WH B0, 384 GB DRAM)
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.model.row_batched_model import get_fabric_config
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape
from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

# ---------------------------------------------------------------------------
# Environment variable conventions
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = os.getenv("KIMI_HF_MODEL", "/workspace/extra/Kimi-K2.5")
_DEFAULT_CACHE_DIR = os.getenv("KIMI_CACHE", "/workspace/extra/kimi_cache")

# Trace region size: empirically determined from DSV3 vLLM workloads + 20% headroom.
# Adjust if trace buffer overflow occurs with larger batch sizes or sequence lengths.
_BASE_TRACE_REGION_BYTES = 38_070_272
_TRACE_REGION_SIZE = _BASE_TRACE_REGION_BYTES + int(0.20 * _BASE_TRACE_REGION_BYTES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_performance_metrics(statistics: dict | None) -> None:
    """Log performance metrics from generator statistics dict."""
    if not statistics:
        return
    logger.info("=== Performance Metrics ===")
    if "inference_prefill" in statistics:
        logger.info(f"Prefill time:           {statistics['inference_prefill']*1000:.2f} ms")
    if "prefill_t/s" in statistics:
        logger.info(f"Prefill throughput:     {statistics['prefill_t/s']:.2f} t/s")
    if "decode_t/s/u" in statistics:
        logger.info(f"Decode throughput:      {statistics['decode_t/s/u']:.2f} t/s/user")
    if "decode_t/s" in statistics:
        logger.info(f"Decode throughput:      {statistics['decode_t/s']:.2f} t/s (total)")
    if "Full demo runtime" in statistics:
        logger.info(f"Full demo runtime:      {statistics['Full demo runtime']:.2f} s")


def load_prompts_from_json(json_file_path: str, max_prompts: int | None = None) -> list[str]:
    """Load prompts from a JSON file.

    Supported formats::

        # Array of objects
        [{"prompt": "..."}, {"prompt": "..."}, ...]

        # Object with prompts key
        {"prompts": [{"prompt": "..."}, ...]}

        # Array of strings
        ["prompt 1", "prompt 2", ...]

    Args:
        json_file_path: Path to JSON file.
        max_prompts:    Maximum prompts to load (``None`` = all).

    Returns:
        List of prompt strings.

    Raises:
        SystemExit: On file read error, bad format, or empty result.
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
    except OSError as e:
        raise SystemExit(f"Failed to read prompts file '{json_path}': {e}")

    if isinstance(data, list):
        prompt_items = data
    elif isinstance(data, dict) and "prompts" in data:
        prompt_items = data["prompts"]
    else:
        raise SystemExit(
            f"JSON file '{json_path}' must be an array of prompts "
            f"or an object with a 'prompts' key."
        )

    prompts: list[str] = []
    for item in prompt_items:
        if max_prompts is not None and len(prompts) >= max_prompts:
            break
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]))
        else:
            logger.warning(f"Skipping unrecognised prompt item: {item!r}")

    if not prompts:
        raise SystemExit(f"No valid prompts found in '{json_path}'")

    logger.info(
        f"Loaded {len(prompts)} prompt(s) from '{json_path}'"
        + (f" (capped at {max_prompts})" if max_prompts is not None else "")
    )
    return prompts


def validate_model_path(model_path_str: str, *, require_safetensors: bool, require_tokenizer: bool) -> None:
    """Validate the Kimi K2.5 model directory before opening a mesh device.

    Args:
        model_path_str:      Path to the checkpoint directory.
        require_safetensors: If ``True``, verify shard files exist.
        require_tokenizer:   If ``True``, verify tokenizer files exist.

    Raises:
        SystemExit: If a required file is missing.
    """
    mp = Path(model_path_str)
    if not mp.exists():
        raise SystemExit(f"Model path does not exist: '{mp}'")
    if not mp.is_dir():
        raise SystemExit(f"Model path is not a directory: '{mp}'")
    if not (mp / "config.json").exists():
        raise SystemExit(f"config.json not found in '{mp}'")

    if require_tokenizer:
        tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json")
        if not any((mp / f).exists() for f in tokenizer_files):
            raise SystemExit(
                f"No tokenizer files found in '{mp}'. "
                f"Expected one of: {', '.join(tokenizer_files)}"
            )

    if require_safetensors:
        if not (mp / "model.safetensors.index.json").exists():
            raise SystemExit(f"model.safetensors.index.json not found in '{mp}'")
        shards = glob(str(mp / "model-*.safetensors"))
        if not shards:
            raise SystemExit(f"No model shard files (model-*.safetensors) in '{mp}'")
        logger.info(f"Found {len(shards)} safetensors shard(s) in '{mp}'")


# ---------------------------------------------------------------------------
# Core demo runner
# ---------------------------------------------------------------------------


def run_demo(
    prompts: list[str] | None = None,
    *,
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    max_new_tokens: int = 32,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    random_weights: bool = False,
    override_num_layers: int | None = None,
    enable_trace: bool = True,
    repeat_batches: int = 1,
    prefill_max_tokens: int | None = None,
    sample_on_device: bool = True,
    force_recalculate: bool = False,
) -> dict:
    """Run the Kimi K2.5 inference demo.

    This is the programmatic entry point — useful for importing from tests or
    scripts without going through ``argparse``.

    Args:
        prompts:             List of prompt strings.  Pass ``None`` or ``[]``
                             together with ``random_weights=True`` for a
                             hardware smoke test.
        model_path:          Path to Kimi K2.5 checkpoint directory.
        max_new_tokens:      Tokens to generate per prompt.
        cache_dir:           TTNN weight cache directory.
        random_weights:      Use random weights instead of loading safetensors.
        override_num_layers: Override number of model layers (useful for fast
                             tests; ``2`` runs 1 dense + 1 MoE).
        enable_trace:        Enable TTNN trace capture for decode (faster after
                             first decode, at the cost of graph compilation).
        repeat_batches:      Repeat generation N times (perf measurement).
        prefill_max_tokens:  Cap prefill token count.
        sample_on_device:    Run top-p sampling on device (faster than host).
        force_recalculate:   Regenerate TTNN weight cache even if it exists.

    Returns:
        ``dict`` with keys:

        * ``generations`` — list of ``{"tokens": list[int], "text": str|None}``
        * ``statistics`` — dict of latency/throughput metrics (may be ``None``)

    Raises:
        ValueError:  If ``MESH_DEVICE`` env var is not set.
        SystemExit:  If model path validation fails.
    """
    model_path = Path(model_path)
    cache_dir = Path(cache_dir)

    validate_model_path(
        str(model_path),
        require_safetensors=not random_weights,
        require_tokenizer=not random_weights,
    )

    requested_system = os.getenv("MESH_DEVICE")
    if requested_system is None:
        raise ValueError(
            "MESH_DEVICE environment variable is not set. "
            "Set it to TG (32-chip Galaxy), QUAD (8-chip), or DUAL (4-chip)."
        )

    mesh_shape = system_name_to_mesh_shape(requested_system.upper())
    logger.info(f"MESH_DEVICE={requested_system!r} → mesh shape {mesh_shape}")

    fabric_config = get_fabric_config()
    logger.info(f"Fabric config: {fabric_config}")
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)

    # Keep mesh_device initialised to None so the finally block can guard
    # cleanup correctly even if open_mesh_device or load_tokenizer raises.
    mesh_device = None
    gen: KimiGenerator | None = None
    try:
        if enable_trace:
            logger.info(f"Trace region size: {_TRACE_REGION_SIZE} bytes")
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, trace_region_size=_TRACE_REGION_SIZE)
        else:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        tokenizer = None
        if not random_weights:
            logger.info(f"Loading tokenizer from '{model_path}'")
            tokenizer = load_tokenizer(model_path)

        gen = KimiGenerator(
            mesh_device=mesh_device,
            model_path=model_path,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            random_weights=random_weights,
            # Default to 2 layers for random-weights smoke test (1 dense + 1 MoE)
            override_num_layers=(
                override_num_layers if override_num_layers is not None else (2 if random_weights else None)
            ),
            enable_trace=enable_trace,
            prefill_max_tokens=prefill_max_tokens,
            force_recalculate=force_recalculate,
            sample_on_device=sample_on_device,
        )

        if random_weights:
            prompt_list = [""]  # dummy prompt — tokens are random anyway
        else:
            if not prompts:
                raise SystemExit("At least one prompt is required in full-model mode.")
            prompt_list = prompts

        logger.info(f"Generating up to {max_new_tokens} token(s) for {len(prompt_list)} prompt(s)")
        generations, statistics = gen.generate(
            prompt_list,
            max_new_tokens=max_new_tokens,
            repeat_batches=repeat_batches,
        )

        results = []
        for tokens in generations:
            entry: dict = {"tokens": tokens, "text": None}
            if gen.tokenizer is not None:
                entry["text"] = gen.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(entry)

        return {"generations": results, "statistics": statistics}

    finally:
        if gen is not None:
            try:
                gen.cleanup_all()
            except Exception as exc:
                logger.warning(f"Generator cleanup failed: {exc}")

        if mesh_device is not None:
            ttnn.synchronize_device(mesh_device)
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Kimi K2.5 demo."""
    p = argparse.ArgumentParser(
        prog="demo.py",
        description="Kimi K2.5 inference demo on Tenstorrent hardware.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "prompts",
        nargs="*",
        type=str,
        metavar="PROMPT",
        help="Prompt text(s).  Required in full-model mode; ignored with --random-weights.",
    )
    p.add_argument(
        "--prompts-file",
        type=str,
        metavar="FILE",
        help=(
            "JSON file containing prompts.  Supported formats: "
            "array of strings, array of {\"prompt\":\"...\"} objects, "
            "or {\"prompts\":[...]} object.  Takes precedence over positional prompts."
        ),
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        metavar="N",
        help="Maximum prompts to load from --prompts-file.",
    )
    p.add_argument(
        "--output-path",
        type=str,
        metavar="FILE",
        help="Save results to this JSON file instead of printing to stdout.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        metavar="DIR",
        help=(
            f"Path to Kimi K2.5 checkpoint directory.  "
            f"Defaults to $KIMI_HF_MODEL or '{_DEFAULT_MODEL_PATH}'."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=_DEFAULT_CACHE_DIR,
        metavar="DIR",
        help=(
            f"TTNN weight cache directory.  Created on first run.  "
            f"Defaults to $KIMI_CACHE or '{_DEFAULT_CACHE_DIR}'."
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        metavar="N",
        help="Number of new tokens to generate per prompt.  Default: 32.",
    )
    p.add_argument(
        "--random-weights",
        action="store_true",
        help="Use randomly initialised weights — hardware smoke test without real checkpoint.",
    )
    p.add_argument(
        "--override-num-layers",
        type=int,
        metavar="N",
        help=(
            "Override model layer count (default: all 61 layers).  "
            "Use 2 with --random-weights for a fast 2-layer smoke test."
        ),
    )
    p.add_argument(
        "--force-recalculate",
        "--recalculate-weights",
        dest="force_recalculate",
        action="store_true",
        default=False,
        help="Discard TTNN weight cache and reconvert from safetensors.",
    )
    p.add_argument(
        "--disable-trace",
        action="store_false",
        dest="enable_trace",
        default=True,
        help="Disable TTNN trace capture for the decode forward pass.",
    )
    p.add_argument(
        "--repeat-batches",
        type=int,
        default=1,
        metavar="N",
        help="Repeat generation N times (useful for latency benchmarking).",
    )
    p.add_argument(
        "--prefill-max-tokens",
        type=int,
        metavar="N",
        help="Cap the number of tokens processed during prefill.",
    )
    p.add_argument(
        "--sample-on-host",
        action="store_false",
        dest="sample_on_device",
        default=True,
        help="Run top-p sampling on the host CPU instead of on-device.",
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = create_parser().parse_args()

    # --prompts-file overrides positional prompts
    prompts_file_path: Path | None = None
    if args.prompts_file:
        prompts_file_path = Path(args.prompts_file)
        loaded_prompts = load_prompts_from_json(str(prompts_file_path), max_prompts=args.num_prompts)
        if args.prompts:
            logger.info(
                f"--prompts-file provided — ignoring {len(args.prompts)} positional prompt(s) "
                f"in favour of {len(loaded_prompts)} from file."
            )
        args.prompts = loaded_prompts

    if not args.random_weights and not args.prompts:
        raise SystemExit(
            "A prompt is required in full-model mode.  "
            "Either pass prompt text(s) as positional arguments, "
            "use --prompts-file, or add --random-weights for a smoke test."
        )

    results = run_demo(
        args.prompts,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        random_weights=args.random_weights,
        override_num_layers=args.override_num_layers,
        enable_trace=args.enable_trace,
        repeat_batches=args.repeat_batches,
        prefill_max_tokens=args.prefill_max_tokens,
        sample_on_device=args.sample_on_device,
        force_recalculate=args.force_recalculate,
    )

    # Output
    if prompts_file_path is not None and args.output_path:
        output_path = Path(args.output_path)
        output_data = {
            "model": str(args.model_path),
            "prompts": args.prompts,
            "generations": results["generations"],
            "statistics": results.get("statistics") or {},
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to '{output_path}'")
            print(f"\nResults saved to '{output_path}'\n")
        except OSError as exc:
            raise SystemExit(f"Failed to write output file '{output_path}': {exc}")
    else:
        print("\n===== Kimi K2.5 Output =====\n")
        for i, gen_result in enumerate(results["generations"]):
            print(f"--- [{i + 1}] ---")
            if args.prompts and i < len(args.prompts):
                print(f"Prompt:  {args.prompts[i]}")
            elif args.random_weights:
                print("Prompt:  [random-weights smoke test]")
            print("Output:")
            if gen_result.get("text") is not None:
                print(gen_result["text"])
            else:
                token_ids = gen_result.get("tokens", [])
                print(f"[token IDs ({len(token_ids)})] {token_ids[:16]}{'...' if len(token_ids) > 16 else ''}")
            print()
        print("============================\n")

    _print_performance_metrics(results.get("statistics"))


if __name__ == "__main__":
    main()
