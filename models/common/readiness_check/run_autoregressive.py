# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run the autoregressive (free-running) readiness check.

Given a prompt file and a model directory containing `tt/generator.py`, this
runner generates a completion to the same prompt from two places:

  1. The HuggingFace reference model, via `model.generate(do_sample=False)`.
  2. The TT generator under test, via `generator.generate(next_input=None)`.

Both completions are saved to disk for side-by-side inspection. There is **no
programmatic comparison** — the caller (typically a `/decoder-to-productized`
agent run) reads both outputs and judges whether they're consistent.

CLI:
    python -m models.common.readiness_check.run_autoregressive \\
        --model-dir models/autoports/<model_name> \\
        --hf-model meta-llama/Llama-3.1-8B-Instruct \\
        --mesh-device N150
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.common.readiness_check.contract import (
    BUILD_GENERATOR_FUNCTION_NAME,
    GENERATOR_MODULE_RELPATH,
    BuildGeneratorFn,
    Generator,
)
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

DEFAULT_PROMPT_FILE = Path(__file__).parent / "autoregressive_prompt.txt"
DEFAULT_MAX_NEW_TOKENS = 256


def _import_build_generator(model_dir: Path) -> BuildGeneratorFn:
    generator_path = model_dir / GENERATOR_MODULE_RELPATH
    if not generator_path.exists():
        raise FileNotFoundError(
            f"Expected generator at {generator_path}. The readiness check requires "
            f"<model_dir>/{GENERATOR_MODULE_RELPATH} to exist and expose `{BUILD_GENERATOR_FUNCTION_NAME}`."
        )

    module_name = f"_readiness_generator_{model_dir.resolve().name}"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    fn = getattr(module, BUILD_GENERATOR_FUNCTION_NAME, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{generator_path} does not export a callable `{BUILD_GENERATOR_FUNCTION_NAME}`. "
            f"See models/common/readiness_check/contract.py."
        )
    return fn  # type: ignore[return-value]


def _hf_generate_greedy(
    *,
    hf_model_id: str,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> List[int]:
    """
    Greedy autoregressive decode via HF. Returns only the generated tokens
    (prompt stripped). Stops early on EOS; HF handles multi-EOS configs
    (e.g. Llama 3.1's eos_token_id list) automatically.
    """
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos = tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    generated = out[0, input_ids.shape[1] :].cpu().tolist()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return generated


def run_autoregressive(
    *,
    model_dir: Path,
    hf_model_id: str,
    prompt_file: Path,
    mesh_device,
    output_dir: Path,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    build_kwargs: Optional[Dict[str, Any]] = None,
    chat_template: bool = False,
) -> Dict[str, Path]:
    """
    Programmatic entry point. Generates a completion from HF and from the TT
    generator, writes both to `output_dir`, returns the paths written.
    """
    build_kwargs = build_kwargs or {}

    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file {prompt_file} is empty")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    if chat_template:
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    else:
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prompt ({len(prompt_token_ids)} tokens):\n{prompt_text}\n")

    # 1. HF reference. Load, generate, free before touching the TT device.
    hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HF reference on {hf_device}...")
    hf_tokens = _hf_generate_greedy(
        hf_model_id=hf_model_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        device=hf_device,
    )
    hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=False)
    print(f"HF produced {len(hf_tokens)} tokens.")

    # 2. TT under test.
    print("Building TT generator...")
    build_generator = _import_build_generator(model_dir)
    generator: Generator = build_generator(model_dir=model_dir, mesh_device=mesh_device, **build_kwargs)
    try:
        print("Running TT autoregressive generation...")
        tt_tokens = generator.generate(
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            next_input=None,
        )
    finally:
        teardown = getattr(generator, "teardown", None)
        if callable(teardown):
            teardown()
    tt_text = tokenizer.decode(tt_tokens, skip_special_tokens=False)
    print(f"TT produced {len(tt_tokens)} tokens.")

    hf_path = output_dir / "hf_completion.txt"
    tt_path = output_dir / "tt_completion.txt"
    meta_path = output_dir / "autoregressive_meta.json"

    hf_path.write_text(hf_text, encoding="utf-8")
    tt_path.write_text(tt_text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "hf_model_id": hf_model_id,
                "prompt_file": str(prompt_file),
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "chat_template": chat_template,
                "max_new_tokens": max_new_tokens,
                "hf": {"token_ids": list(hf_tokens), "num_tokens": len(hf_tokens)},
                "tt": {"token_ids": list(tt_tokens), "num_tokens": len(tt_tokens)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved HF completion → {hf_path}")
    print(f"Saved TT completion → {tt_path}")
    print(f"Saved metadata     → {meta_path}")

    return {"hf_completion": hf_path, "tt_completion": tt_path, "meta": meta_path}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the autoregressive (free-running) readiness comparison.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the model directory.")
    parser.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model id or local path used as the reference.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to the prompt file. Default: {DEFAULT_PROMPT_FILE}",
    )
    add_mesh_device_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write hf_completion.txt / tt_completion.txt / autoregressive_meta.json. "
        "Defaults to <model_dir>/readiness_autoregressive/.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Decode-step budget per side; both stop early on EOS (default {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render the prompt as one user turn with the tokenizer's chat template.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.model_dir / "readiness_autoregressive")

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        run_autoregressive(
            model_dir=args.model_dir.resolve(),
            hf_model_id=args.hf_model,
            prompt_file=args.prompt_file.resolve(),
            mesh_device=mesh_device,
            output_dir=output_dir.resolve(),
            max_new_tokens=args.max_new_tokens,
            chat_template=args.chat_template,
        )
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    _main()
