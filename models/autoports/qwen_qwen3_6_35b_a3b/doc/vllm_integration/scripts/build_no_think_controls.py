# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build no-thinking qualitative controls for the Qwen3.6 vLLM stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_INPUT = MODEL_DIR / "readiness_vllm/vllm_chat_no_think_qualitative_outputs.json"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "readiness_vllm"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _record_text(record: dict[str, Any]) -> str:
    if "text" in record:
        return str(record["text"])
    return str(record["response"]["choices"][0]["message"]["content"])


def _load_vllm_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    if not records:
        raise ValueError(f"{path} has no qualitative records")
    return str(data.get("model") or MODEL_ID), records


def _render_no_think_prompts(tokenizer: Any, prompts: list[str]) -> list[dict[str, Any]]:
    rendered = []
    for idx, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized = tokenizer(rendered_prompt, add_special_tokens=False).input_ids
        rendered.append(
            {
                "prompt_index": idx,
                "prompt": prompt,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "rendered_prompt": str(rendered_prompt),
                "prompt_token_ids": [int(token_id) for token_id in tokenized],
                "prompt_len": len(tokenized),
                "template_contains_empty_think_block": "<think>\n\n</think>" in str(rendered_prompt),
            }
        )
    return rendered


def _checker_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for record in records:
        prompt_index = int(record["prompt_index"])
        item = grouped.setdefault(prompt_index, {"prompt": record["prompt"]})
        profile = str(record["profile"])
        if profile == "greedy":
            item["greedy_completion"] = _record_text(record)
        elif profile == "sampled":
            item["sampled_completion"] = _record_text(record)
        else:
            item[f"{profile}_completion"] = _record_text(record)
    return [grouped[idx] for idx in sorted(grouped)]


def _run_tt_greedy_controls(
    *,
    model_dir: Path,
    mesh_device_label: str,
    fabric_config: str | None,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    mesh_device = open_readiness_mesh_device(mesh_device_label, fabric_config)
    try:
        generator = build_generator(
            model_dir=model_dir.resolve(),
            mesh_device=mesh_device,
            model_id=MODEL_ID,
            local_files_only=local_files_only,
        )
        outputs: list[dict[str, Any]] = []
        try:
            for prompt in prompts:
                token_ids = generator.generate(
                    list(prompt["prompt_token_ids"]),
                    max_new_tokens=max_new_tokens,
                    enable_trace=True,
                )
                outputs.append(
                    {
                        "prompt_index": prompt["prompt_index"],
                        "prompt": prompt["prompt"],
                        "profile": "greedy",
                        "token_ids": [int(token_id) for token_id in token_ids],
                        "num_tokens": len(token_ids),
                        "completion": _decode(tokenizer, token_ids),
                        "timings": dict(generator.last_timings),
                        "trace_present": generator._trace is not None,
                        "trace_generated_steps": generator._trace.generated if generator._trace is not None else 0,
                    }
                )
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
        return outputs
    finally:
        close_readiness_mesh_device(mesh_device, fabric_config)


def _write_control_verdict(
    *,
    output_dir: Path,
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    tt_controls: list[dict[str, Any]],
) -> None:
    vllm_greedy = {
        int(record["prompt_index"]): _record_text(record) for record in records if record["profile"] == "greedy"
    }
    vllm_sampled = {
        int(record["prompt_index"]): _record_text(record) for record in records if record["profile"] == "sampled"
    }
    tt_by_prompt = {int(record["prompt_index"]): record for record in tt_controls}
    comparisons = []
    for prompt in prompts:
        idx = int(prompt["prompt_index"])
        tt_text = tt_by_prompt.get(idx, {}).get("completion")
        greedy_text = vllm_greedy.get(idx, "")
        comparisons.append(
            {
                "prompt_index": idx,
                "prompt": prompt["prompt"],
                "vllm_greedy": greedy_text,
                "vllm_sampled": vllm_sampled.get(idx, ""),
                "full_model_greedy_control": tt_text,
                "vllm_greedy_matches_full_model_control": bool(tt_text is not None and greedy_text == tt_text),
                "vllm_greedy_word_count": len(greedy_text.split()),
                "full_model_greedy_word_count": len(str(tt_text or "").split()),
            }
        )
    _write_json(
        output_dir / "qualitative_no_think_control_verdict.json",
        {
            "scope": "no-thinking chat qualitative control",
            "accepted_serving_artifact": str(DEFAULT_INPUT),
            "prompt_format_artifact": str(output_dir / "qualitative_no_think_prompt_format.json"),
            "prompts_artifact": str(output_dir / "qualitative_no_think_prompts.json"),
            "full_model_control_artifact": str(output_dir / "full_model_no_think_greedy_controls.json"),
            "vllm_checker_artifact": str(output_dir / "vllm_chat_no_think_checker_outputs.json"),
            "comparison": comparisons,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-tt", action="store_true")
    add_mesh_device_args(parser)
    args = parser.parse_args()

    model_id, records = _load_vllm_records(args.input)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise RuntimeError(f"{args.hf_model} tokenizer does not expose a chat template")

    prompts = []
    seen = set()
    for record in records:
        prompt = str(record["prompt"])
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    rendered_prompts = _render_no_think_prompts(tokenizer, prompts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "qualitative_no_think_prompt_format.json",
        {
            "hf_model_id": args.hf_model,
            "served_model_id": model_id,
            "tokenizer_class": type(tokenizer).__name__,
            "prompt_source": str(args.input),
            "format": "hf_tokenizer_chat_template",
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(rendered_prompts),
            "num_vllm_records": len(records),
        },
    )
    _write_json(args.output_dir / "qualitative_no_think_prompts.json", rendered_prompts)
    _write_json(args.output_dir / "vllm_chat_no_think_checker_outputs.json", _checker_items(records))

    tt_controls: list[dict[str, Any]] = []
    if not args.skip_tt:
        tt_controls = _run_tt_greedy_controls(
            model_dir=args.model_dir,
            mesh_device_label=args.mesh_device,
            fabric_config=args.fabric_config,
            tokenizer=tokenizer,
            prompts=rendered_prompts,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )
        _write_json(args.output_dir / "full_model_no_think_greedy_controls.json", tt_controls)

    _write_control_verdict(
        output_dir=args.output_dir,
        prompts=rendered_prompts,
        records=records,
        tt_controls=tt_controls,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "prompts": len(rendered_prompts)}, indent=2))


if __name__ == "__main__":
    main()
