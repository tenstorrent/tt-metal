# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a teacher-forcing reference file from a HuggingFace model.

Runs the HF model greedily for `max_new_tokens` on each prompt and captures
the top-K next-token candidates at every generated position. Saves the
result in the `readiness_v1` schema (see schema.py).

CLI:
    python -m models.common.readiness_check.generate \\
        --hf-model meta-llama/Llama-3.1-8B-Instruct \\
        --prompt "Hello, who are you?" \\
        --max-new-tokens 256 \\
        --output llama31_8b.refpt

Python:
    from models.common.readiness_check.generate import generate_reference
    generate_reference(
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        prompts=["Hello, who are you?"],
        max_new_tokens=256,
        output_path="llama31_8b.refpt",
    )
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from models.common.readiness_check.schema import Reference, ReferenceEntry, save_reference

DEFAULT_K = 100


HFModelLoader = Callable[[str], "torch.nn.Module"]


def _default_model_loader(hf_model_id: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True)


class _TopKSnapshotter(StoppingCriteria):
    """Captures top-K predictions for each newly generated token."""

    def __init__(self, prompt_len: int, max_new_tokens: int, k: int, pbar=None) -> None:
        self.prompt_len = prompt_len
        self.k = k
        self.pbar = pbar
        self.last_len = prompt_len
        self.topk_rows: List[torch.Tensor] = []  # appended per generated step

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        current_len = input_ids.shape[-1]
        if current_len > self.last_len and scores is not None:
            step_scores = scores[-1] if isinstance(scores, (list, tuple)) else scores
            if step_scores is not None:
                if step_scores.dim() == 2:
                    step_scores = step_scores[0]
                k = min(self.k, int(step_scores.numel()))
                topk_ids = torch.topk(step_scores, k=k, dim=-1).indices.to(torch.int32).cpu()
                self.topk_rows.append(topk_ids)
            self.last_len = current_len
            if self.pbar is not None:
                generated = current_len - self.prompt_len
                if generated > self.pbar.n:
                    self.pbar.update(generated - self.pbar.n)
        return False


def _generate_one_entry(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k: int,
    device: torch.device,
    eos_id: int,
    pad_id: int,
) -> ReferenceEntry:
    raw_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_len = len(raw_prompt_tokens)
    prompt_tokens_tensor = torch.tensor([raw_prompt_tokens], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(prompt_tokens_tensor, dtype=torch.long, device=device)

    pbar = None
    if tqdm is not None and max_new_tokens > 0:
        pbar = tqdm(total=max_new_tokens, desc=f"gen ({prompt[:24]!r}…)", unit="tok", mininterval=1)
    snapshotter = _TopKSnapshotter(prompt_len, max_new_tokens, k, pbar)
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_tokens_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([snapshotter]),
            )
    finally:
        if pbar is not None:
            pbar.close()

    full_sequence = outputs.sequences[0].detach().cpu()  # [prompt_len + gen_len]
    generated_tokens = full_sequence[prompt_len:]
    gen_len = int(generated_tokens.numel())

    if gen_len != len(snapshotter.topk_rows):
        raise RuntimeError(f"Captured {len(snapshotter.topk_rows)} top-K rows but generated {gen_len} tokens")

    if gen_len == 0:
        topk_tensor = torch.zeros((0, k), dtype=torch.int32)
    else:
        topk_tensor = torch.stack(snapshotter.topk_rows, dim=0).contiguous()
        if topk_tensor.shape[1] < k:
            # Vocab smaller than K — pad with -1 so downstream comparisons can't accidentally match.
            pad = torch.full((topk_tensor.shape[0], k - topk_tensor.shape[1]), -1, dtype=torch.int32)
            topk_tensor = torch.cat([topk_tensor, pad], dim=1)

    return ReferenceEntry(
        prompt_text=prompt,
        prompt_tokens=torch.tensor([raw_prompt_tokens], dtype=torch.int64),
        generated_tokens=generated_tokens.unsqueeze(0).to(torch.int64),
        topk_tokens=topk_tensor,
        tf_prompt_len=prompt_len,
    )


def generate_reference(
    hf_model_id: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    output_path: Path | str,
    top_k: int = DEFAULT_K,
    model_loader: Optional[HFModelLoader] = None,
    device: Optional[torch.device] = None,
) -> Path:
    """
    Run HF teacher greedily on each prompt for `max_new_tokens`, capture
    top-`top_k` next-token candidates at every generated position, and save
    a `readiness_v1` reference file at `output_path`.
    """
    if not prompts:
        raise ValueError("`prompts` must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError(f"`max_new_tokens` must be > 0, got {max_new_tokens}")
    if top_k <= 0:
        raise ValueError(f"`top_k` must be > 0, got {top_k}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    loader = model_loader or _default_model_loader
    model = loader(hf_model_id).eval().to(device)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            cfg_eos = cfg_eos[0]
        eos_id = cfg_eos
    if eos_id is None:
        raise RuntimeError("Could not determine eos_token_id from tokenizer or model config")

    bos_id = tokenizer.bos_token_id
    # Use a safe pad id: prefer tokenizer.pad_token_id, but fall back to eos when pad collides with bos.
    pad_id = tokenizer.pad_token_id
    safe_pad_id = pad_id if (pad_id is not None and pad_id != bos_id) else int(eos_id)

    entries = [
        _generate_one_entry(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            k=top_k,
            device=device,
            eos_id=int(eos_id),
            pad_id=int(safe_pad_id),
        )
        for prompt in prompts
    ]

    reference = Reference(
        k=top_k,
        hf_model_id=hf_model_id,
        entries=entries,
        token_ids_meta={
            "bos_id": int(bos_id) if bos_id is not None else None,
            "eos_id": int(eos_id),
            "pad_id": int(pad_id) if pad_id is not None else None,
        },
    )
    return save_reference(reference, output_path)


def _load_prompts_file(path: Path) -> List[str]:
    """
    Load prompts from a JSON file. Accepts either:
      - a flat list of strings: ["prompt1", "prompt2"]
      - a list of objects with a "prompt" key: [{"prompt": "...", "name": "factual"}, ...]
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"prompts file {path} must contain a non-empty JSON list")
    prompts: List[str] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]))
        else:
            raise ValueError(f"prompts[{i}] must be a string or an object with a 'prompt' key")
    return prompts


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a readiness-check reference file from a HF model.")
    parser.add_argument("--hf-model", required=True, help="HuggingFace model id or local path.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Single prompt text to generate from.")
    src.add_argument("--prompts-file", type=Path, help="Path to a JSON file containing a list of prompts.")
    parser.add_argument("--max-new-tokens", type=int, required=True, help="Number of new tokens to generate.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output .refpt file.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_K, help=f"Top-K to capture per position (default {DEFAULT_K})."
    )
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt is not None else _load_prompts_file(args.prompts_file)

    path = generate_reference(
        hf_model_id=args.hf_model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        top_k=args.top_k,
    )
    print(f"Reference saved to: {path}")


if __name__ == "__main__":
    _main()
