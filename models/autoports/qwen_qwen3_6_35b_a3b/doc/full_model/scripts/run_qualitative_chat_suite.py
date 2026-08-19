# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the full-model qualitative chat-template suite for Qwen3.6-35B-A3B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator
from models.common.readiness_check.generate import _generation_stop_ids, _safe_pad_id
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_PROMPTS = Path("models/common/readiness_check/vllm_prompts.txt")
DEFAULT_OUTPUT_DIR = MODEL_DIR / "doc/full_model/artifacts/qualitative_chat_suite_64"


def _read_prompts(path: Path) -> list[str]:
    chunks = [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n\n")]
    prompts = [chunk for chunk in chunks if chunk]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _chat_prompt(tokenizer: Any, prompt: str) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(tokenized, torch.Tensor):
        token_ids = tokenized.reshape(-1).tolist()
    elif isinstance(tokenized, Mapping):
        token_ids = tokenized["input_ids"]
    else:
        token_ids = tokenized
    return str(rendered), [int(token_id) for token_id in token_ids]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _load_hf(model_id: str, local_files_only: bool):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map="auto",
    ).eval()


def _hf_generate(
    *,
    model_id: str,
    tokenizer: Any,
    prompt_token_ids: list[list[int]],
    max_new_tokens: int,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    model = _load_hf(model_id, local_files_only)
    input_device = next(model.parameters()).device
    stop_ids = _generation_stop_ids(tokenizer, model)
    eos_token_id: int | list[int] = stop_ids[0] if len(stop_ids) == 1 else stop_ids
    outputs: list[dict[str, Any]] = []
    try:
        for ids in prompt_token_ids:
            input_ids = torch.tensor([ids], dtype=torch.long, device=input_device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=_safe_pad_id(tokenizer, stop_ids),
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
            sequence = generated.sequences if hasattr(generated, "sequences") else generated
            token_ids = [int(token_id) for token_id in sequence[0, input_ids.shape[1] :].detach().cpu().tolist()]
            outputs.append(
                {
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                    "completion": _decode(tokenizer, token_ids),
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def _tt_generate(
    *,
    model_dir: Path,
    mesh_device_label: str,
    fabric_config: str | None,
    tokenizer: Any,
    prompt_token_ids: list[list[int]],
    max_new_tokens: int,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    mesh_device = open_readiness_mesh_device(mesh_device_label, fabric_config)
    try:
        generator = build_generator(
            model_dir=model_dir,
            mesh_device=mesh_device,
            model_id=MODEL_ID,
            local_files_only=local_files_only,
        )
        outputs: list[dict[str, Any]] = []
        try:
            for ids in prompt_token_ids:
                token_ids = generator.generate(ids, max_new_tokens=max_new_tokens, enable_trace=True)
                outputs.append(
                    {
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-tt", action="store_true")
    add_mesh_device_args(parser)
    args = parser.parse_args()

    prompts = _read_prompts(args.prompts_file)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise RuntimeError(f"{args.hf_model} tokenizer does not expose a chat template")

    rendered_prompts: list[dict[str, Any]] = []
    prompt_token_ids: list[list[int]] = []
    for idx, prompt in enumerate(prompts):
        rendered, ids = _chat_prompt(tokenizer, prompt)
        prompt_token_ids.append(ids)
        rendered_prompts.append(
            {
                "prompt_id": idx,
                "prompt": prompt,
                "rendered_prompt": rendered,
                "prompt_token_ids": ids,
                "prompt_len": len(ids),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "qualitative_prompt_format.json",
        {
            "hf_model_id": args.hf_model,
            "tokenizer_class": type(tokenizer).__name__,
            "prompt_source": str(args.prompts_file),
            "format": "hf_tokenizer_chat_template",
            "add_generation_prompt": True,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompts),
        },
    )
    _write_json(args.output_dir / "qualitative_prompts.json", rendered_prompts)

    hf_outputs = (
        []
        if args.skip_hf
        else _hf_generate(
            model_id=args.hf_model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )
    )
    if hf_outputs:
        _write_json(args.output_dir / "hf_qualitative_outputs.json", hf_outputs)

    tt_outputs = (
        []
        if args.skip_tt
        else _tt_generate(
            model_dir=args.model_dir.resolve(),
            mesh_device_label=args.mesh_device,
            fabric_config=args.fabric_config,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )
    )
    if tt_outputs:
        _write_json(args.output_dir / "tt_qualitative_outputs.json", tt_outputs)

    checker_items = []
    for idx, prompt in enumerate(prompts):
        item = {"prompt": prompt}
        if idx < len(tt_outputs):
            item["greedy_completion"] = tt_outputs[idx]["completion"]
        checker_items.append(item)
    _write_json(args.output_dir / "vllm_qualitative_outputs.json", checker_items)

    verdict_lines = [
        "# Qualitative Chat Suite",
        "",
        f"- Model: `{args.hf_model}`",
        f"- Prompt format: HF tokenizer chat template, `add_generation_prompt=True`",
        f"- Prompts: {len(prompts)}",
        f"- Max new tokens: {args.max_new_tokens}",
        "",
        "| Prompt | Prompt tokens | HF tokens | TT tokens | TT trace |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, ids in enumerate(prompt_token_ids):
        hf_count = hf_outputs[idx]["num_tokens"] if idx < len(hf_outputs) else "skipped"
        if idx < len(tt_outputs):
            tt_count = tt_outputs[idx]["num_tokens"]
            tt_trace = f"{tt_outputs[idx]['trace_present']} ({tt_outputs[idx]['trace_generated_steps']} replays)"
        else:
            tt_count = "skipped"
            tt_trace = "skipped"
        verdict_lines.append(f"| {idx} | {len(ids)} | {hf_count} | {tt_count} | {tt_trace} |")
    verdict_lines.extend(
        [
            "",
            "Automatic summary only: record the manual verdict after reading "
            "`hf_qualitative_outputs.json`, `tt_qualitative_outputs.json`, "
            "and the degenerate-output checker report.",
        ]
    )
    (args.output_dir / "qualitative_verdict.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "prompts": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
