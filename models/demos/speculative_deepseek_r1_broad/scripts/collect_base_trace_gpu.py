#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.distributed_utils import barrier, init_distributed, is_main_process
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter


BASE_MODEL_PRESETS = {
    "r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1": "deepseek-ai/DeepSeek-R1",
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
}


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Collect base decode trace (GPU) for CPU replay")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--base-model-id", type=str, default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument("--base-model-preset", type=str, default=None, choices=sorted(BASE_MODEL_PRESETS))
    p.add_argument("--base-impl", type=str, default="reference", choices=["adapter", "reference", "accelerate"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=8, help="Store base top-k logits/tokens per step for diagnostics.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel world size.")
    p.add_argument("--tp-backend", type=str, default="deepspeed", choices=["deepspeed"], help="TP backend.")
    p.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed launch (torchrun).")
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--out", type=Path, required=True)
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.base_model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    dist_ctx = init_distributed(args.tp_size, args.local_rank)

    base = DeepSeekBaseAdapter(
        args.base_model_id,
        device=args.device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        base_impl=args.base_impl,
        tp_size=args.tp_size,
        tp_backend=args.tp_backend,
        local_rank=dist_ctx.local_rank,
    )
    prompt_token_ids = base.encode_prompt(args.prompt)
    state = base.forward_prefill(prompt_token_ids)

    step_hidden: list[torch.Tensor] = []
    step_multi_hidden: list[torch.Tensor] = []
    step_next_tokens: list[int] = []
    step_topk_tokens: list[torch.Tensor] = []
    step_topk_scores: list[torch.Tensor] = []

    for step_idx in range(max(0, args.max_new_tokens)):
        logits = state.next_token_logits.detach().cpu().float()
        topk_scores, topk_tokens = torch.topk(logits, k=max(1, args.topk), dim=-1)
        next_token = int(torch.argmax(logits, dim=-1).item())

        step_hidden.append(state.last_hidden_state.detach().cpu().float().squeeze(0))
        if state.multi_layer_hidden is not None:
            step_multi_hidden.append(state.multi_layer_hidden.detach().cpu().float().squeeze(0))
        step_next_tokens.append(next_token)
        step_topk_tokens.append(topk_tokens.squeeze(0).to(torch.long))
        step_topk_scores.append(topk_scores.squeeze(0).float())

        if step_idx + 1 < args.max_new_tokens:
            state = base.forward_decode(state, next_token)

    payload = {
        "model_id": args.base_model_id,
        "base_impl": args.base_impl,
        "prompt": args.prompt,
        "prompt_token_ids": [int(x) for x in prompt_token_ids],
        "step_next_tokens": [int(x) for x in step_next_tokens],
        "step_last_hidden": torch.stack(step_hidden, dim=0) if step_hidden else torch.empty((0, 0), dtype=torch.float32),
        "step_multi_layer_hidden": torch.stack(step_multi_hidden, dim=0) if step_multi_hidden else None,
        "topk_tokens": torch.stack(step_topk_tokens, dim=0) if step_topk_tokens else torch.empty((0, 0), dtype=torch.long),
        "topk_scores": torch.stack(step_topk_scores, dim=0) if step_topk_scores else torch.empty((0, 0), dtype=torch.float32),
    }

    barrier(dist_ctx)
    if is_main_process(dist_ctx):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(args.out))
        print("=" * 60)
        print("Base trace collected")
        print(f"Model id: {args.base_model_id}")
        print(f"Base impl: {args.base_impl}")
        print(f"TP size: {args.tp_size}, TP backend: {args.tp_backend}")
        print(f"Prompt tokens: {len(prompt_token_ids)}")
        print(f"Recorded steps: {len(step_next_tokens)}")
        print(f"Hidden shape: {tuple(payload['step_last_hidden'].shape)}")
        print(f"Output: {args.out}")
        print("=" * 60)
    barrier(dist_ctx)


if __name__ == "__main__":
    main()
