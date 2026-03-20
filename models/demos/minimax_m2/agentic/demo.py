# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
demo.py — CLI entry point for the General Agentic Mode on N300.

Usage:
    # Interactive mode (default)
    python models/demos/minimax_m2/agentic/demo.py

    # Non-interactive: single query
    python models/demos/minimax_m2/agentic/demo.py --query "What is 2+2?"

    # Non-interactive: query with attachment
    python models/demos/minimax_m2/agentic/demo.py \\
        --query "What did I say?" \\
        --attachments /tmp/voice.wav

    # Load only specific tools (faster startup for debugging)
    python models/demos/minimax_m2/agentic/demo.py \\
        --tools llm,bert \\
        --query "What is the capital of France?"

Environment variables:
    HF_MODEL        HuggingFace model ID for Llama 3B
                    (default: meta-llama/Llama-3.2-3B-Instruct)
    TT_CACHE_PATH   Path for TTNN weight cache
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add repo root to PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

import ttnn
from models.demos.minimax_m2.agentic.loader import load_all_models, open_n300_device
from models.demos.minimax_m2.agentic.orchestrator import process_single_query, run_agentic_loop

_ALL_TOOLS = {"llm", "whisper", "speecht5", "owlvit", "bert"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tenstorrent N300 General Agentic Mode demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process (non-interactive mode)",
    )
    parser.add_argument(
        "--attachments",
        nargs="*",
        default=None,
        help="File paths to attach to the query",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="all",
        help=("Comma-separated list of tools to load, or 'all'. " f"Available: {', '.join(sorted(_ALL_TOOLS))}"),
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of Wormhole chips (1=N150, 2=N300)",
    )
    return parser.parse_args()


def _build_load_flags(tools_arg: str) -> dict:
    """Convert tools argument to load_all_models keyword flags."""
    if tools_arg.strip().lower() == "all":
        selected = _ALL_TOOLS
    else:
        selected = {t.strip().lower() for t in tools_arg.split(",")}

    unknown = selected - _ALL_TOOLS
    if unknown:
        logger.warning(f"Unknown tools ignored: {unknown}")

    return {
        "load_llm": "llm" in selected,
        "load_whisper": "whisper" in selected,
        "load_speecht5": "speecht5" in selected,
        "load_owlvit": "owlvit" in selected,
        "load_bert": "bert" in selected,
    }


def main():
    args = _parse_args()

    # Open device
    if args.num_devices == 2:
        mesh_device = open_n300_device()
    else:
        mesh_device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=10_000_000)
        mesh_device.enable_program_cache()

    try:
        load_flags = _build_load_flags(args.tools)
        models = load_all_models(mesh_device, **load_flags)

        if args.query is not None:
            # Non-interactive single query
            response, _ = process_single_query(
                query=args.query,
                models=models,
                attachments=args.attachments or [],
            )
            print(f"\nResponse: {response}\n")
        else:
            # Interactive mode
            run_agentic_loop(models=models, device=mesh_device)

    finally:
        if hasattr(mesh_device, "get_num_devices"):
            ttnn.close_mesh_device(mesh_device)
        else:
            ttnn.close_device(mesh_device)
        logger.info("Device closed.")


if __name__ == "__main__":
    main()
