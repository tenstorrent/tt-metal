# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Example:
#   timeout --signal=INT --kill-after=30s 45m \
#     python models/demos/deepseek_v3/demo/compare_mtp_demo.py \
#     --model-path "$DEEPSEEK_V3_HF_MODEL" \
#     --cache-dir "$DEEPSEEK_V3_CACHE" \
#     --prompts-file models/demos/deepseek_v3/demo/test_prompts.json \
#     --num-prompts 128 \
#     --max-new-tokens 128 \
#     --output-dir logs

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_SCRIPT = REPO_ROOT / "models/demos/deepseek_v3/demo/demo.py"
DS_RUN = Path("/home/shared/scripts/ds-run")


def _write_json_output(path: Path, payload: dict, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"{label} saved to '{path}'")


def _build_compare_log(
    baseline: dict,
    baseline_path: Path,
    current: dict,
    current_path: Path,
) -> dict:
    baseline_prompts = baseline.get("prompts", [])
    baseline_generations = baseline.get("generations", [])
    current_prompts = current.get("prompts", [])
    current_generations = current.get("generations", [])

    mismatch_reason = None
    if baseline_prompts and baseline_prompts != current_prompts:
        mismatch_reason = "Output mismatch: baseline and current prompts differ."
    if mismatch_reason is None and (
        len(baseline_generations) != len(current_generations) or len(current_prompts) != len(current_generations)
    ):
        mismatch_reason = (
            "Baseline/current generation counts do not match prompt count "
            f"({len(baseline_generations)} baseline vs {len(current_generations)} current vs {len(current_prompts)} prompts)."
        )

    max_len = max(len(baseline_generations), len(current_generations), len(current_prompts))
    compare_entries = []
    for i in range(max_len):
        base_gen = baseline_generations[i] if i < len(baseline_generations) else {}
        cur_gen = current_generations[i] if i < len(current_generations) else {}
        base_prompt = base_gen.get("prompt")
        if base_prompt is None and i < len(baseline_prompts):
            base_prompt = baseline_prompts[i]
        cur_prompt = current_prompts[i] if i < len(current_prompts) else None
        base_text = base_gen.get("text") if base_gen else None
        cur_text = cur_gen.get("text") if cur_gen else None
        prompt_match = None
        text_match = None
        if base_prompt is not None and cur_prompt is not None:
            prompt_match = base_prompt == cur_prompt
        if base_text is not None and cur_text is not None:
            text_match = base_text == cur_text
        if mismatch_reason is None:
            if prompt_match is False:
                mismatch_reason = f"Output mismatch at generation {i}: baseline and current prompts differ."
            elif text_match is False:
                mismatch_reason = f"Output mismatch at generation {i}: baseline and current text differ."
        compare_entries.append(
            {
                "index": i + 1,
                "baseline_prompt": base_prompt,
                "current_prompt": cur_prompt,
                "baseline_text": base_text,
                "current_text": cur_text,
                "prompt_match": prompt_match,
                "text_match": text_match,
            }
        )

    return {
        "baseline_path": str(baseline_path),
        "current_output_path": str(current_path),
        "baseline_count": len(baseline_generations),
        "current_count": len(current_generations),
        "prompt_count": len(current_prompts),
        "mismatch_reason": mismatch_reason,
        "entries": compare_entries,
    }


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Compare DeepSeek demo outputs with MTP off vs on")
    p.add_argument("prompts", type=str, nargs="*", help="Prompt text(s). Ignored if --prompts-file is provided.")
    p.add_argument("--prompts-file", type=str, help="JSON file containing prompts.")
    p.add_argument("--num-prompts", type=int, help="Maximum number of prompts to load from the JSON file.")
    p.add_argument("--model-path", type=str, required=True, help="Path to local HF DeepSeek-V3 model.")
    p.add_argument("--cache-dir", type=str, required=True, help="Path to the TT weight cache.")
    p.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate.")
    p.add_argument("--override-num-layers", type=int, help="Optional layer-count override.")
    p.add_argument("--early-print-first-user", action="store_true", default=False)
    p.add_argument("--enable-mem-profile", action="store_true", default=False)
    p.add_argument("--signpost", action="store_true", default=False)
    p.add_argument("--prefill-max-tokens", type=int, help="Maximum number of tokens to prefill.")
    p.add_argument("--profile-decode", action="store_true", default=False)
    p.add_argument("--min-mtp-accept-rate", type=float, default=None)
    p.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Directory for baseline/current output JSONs and the compare log.",
    )
    p.add_argument("--baseline-output", type=str, help="Optional explicit path for the MTP-off output JSON.")
    p.add_argument("--mtp-output", type=str, help="Optional explicit path for the MTP-on output JSON.")
    p.add_argument("--compare-log-output", type=str, help="Optional explicit path for the compare log JSON.")
    return p


def _load_prompts_from_json(json_file_path: str, max_prompts: int | None = None) -> list[str]:
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise SystemExit(f"Prompts file does not exist: '{json_path}'")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON file '{json_path}': {e}")
    except Exception as e:
        raise SystemExit(f"Failed to read prompts file '{json_path}': {e}")

    if isinstance(data, list):
        prompt_items = data
    elif isinstance(data, dict) and "prompts" in data:
        prompt_items = data["prompts"]
    else:
        raise SystemExit(
            f"JSON file '{json_path}' must be either an array of prompt objects or an object with a 'prompts' key"
        )

    prompts = []
    for item in prompt_items:
        if max_prompts is not None and len(prompts) >= max_prompts:
            break
        if isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]))
    if not prompts:
        raise SystemExit(f"No valid prompts found in '{json_path}'")
    return prompts


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        prompts = _load_prompts_from_json(args.prompts_file, max_prompts=args.num_prompts)
        if args.prompts:
            logger.info(
                f"Both --prompts-file and command-line prompts provided. Using {len(prompts)} prompts from JSON file."
            )
        return prompts
    if args.prompts:
        return args.prompts
    raise SystemExit("A prompt or --prompts-file is required.")


def _resolve_paths(args: argparse.Namespace, prompts: list[str]) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompts_file:
        base_stem = Path(args.prompts_file).stem
    elif prompts:
        base_stem = "deepseek_demo_prompt_compare"
    else:
        base_stem = "deepseek_demo_compare"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    baseline_output = (
        Path(args.baseline_output) if args.baseline_output else output_dir / f"{base_stem}_{timestamp}_mtp_off.json"
    )
    mtp_output = Path(args.mtp_output) if args.mtp_output else output_dir / f"{base_stem}_{timestamp}_mtp_on.json"
    compare_log_output = (
        Path(args.compare_log_output)
        if args.compare_log_output
        else output_dir / f"{base_stem}_{timestamp}_compare.json"
    )
    return baseline_output, mtp_output, compare_log_output


def _build_demo_command(args: argparse.Namespace, output_path: Path, enable_mtp: bool) -> list[str]:
    cmd = [
        str(DS_RUN),
        "python",
        str(DEMO_SCRIPT.relative_to(REPO_ROOT)),
        "--model-path",
        args.model_path,
        "--cache-dir",
        args.cache_dir,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--mtp",
        "on" if enable_mtp else "off",
        "--output-path",
        str(output_path),
    ]
    if args.prompts_file:
        cmd.extend(["--prompts-file", args.prompts_file])
        if args.num_prompts is not None:
            cmd.extend(["--num-prompts", str(args.num_prompts)])
    else:
        cmd.extend(args.prompts)
    if args.override_num_layers is not None:
        cmd.extend(["--override-num-layers", str(args.override_num_layers)])
    if args.early_print_first_user:
        cmd.append("--early_print_first_user")
    if args.enable_mem_profile:
        cmd.append("--enable-mem-profile")
    if args.signpost:
        cmd.append("--signpost")
    if args.prefill_max_tokens is not None:
        cmd.extend(["--prefill-max-tokens", str(args.prefill_max_tokens)])
    if args.profile_decode:
        cmd.append("--profile-decode")
    if enable_mtp and args.min_mtp_accept_rate is not None:
        cmd.extend(["--min-mtp-accept-rate", str(args.min_mtp_accept_rate)])
    return cmd


def _run_demo_command(cmd: list[str], label: str) -> None:
    logger.info(f"Running {label}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"{label} failed with exit code {e.returncode}") from e


def _load_output_json(path: Path, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to read {label.lower()} '{path}': {e}") from e


def main() -> None:
    args = create_parser().parse_args()
    prompts = _load_prompts(args)
    baseline_output_path, mtp_output_path, compare_log_path = _resolve_paths(args, prompts)

    _run_demo_command(_build_demo_command(args, baseline_output_path, enable_mtp=False), "baseline demo")
    baseline_output = _load_output_json(baseline_output_path, "Baseline output")

    _run_demo_command(_build_demo_command(args, mtp_output_path, enable_mtp=True), "MTP demo")
    mtp_output = _load_output_json(mtp_output_path, "MTP output")

    compare_log = _build_compare_log(
        baseline=baseline_output,
        baseline_path=baseline_output_path,
        current=mtp_output,
        current_path=mtp_output_path,
    )
    _write_json_output(compare_log_path, compare_log, "Comparison log")

    if compare_log["mismatch_reason"] is not None:
        raise SystemExit(compare_log["mismatch_reason"])
    logger.info("Output comparison passed: prompt+generated text content matches exactly.")


if __name__ == "__main__":
    main()
