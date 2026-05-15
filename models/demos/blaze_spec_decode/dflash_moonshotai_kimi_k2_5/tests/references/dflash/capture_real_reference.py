#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


def _find_repo_root(path: Path) -> Path:
    for parent in (path.resolve().parent, *path.resolve().parents):
        if (parent / "models").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    return path.resolve().parent


REPO_ROOT = _find_repo_root(Path(__file__))
DEFAULT_GOLDEN_PATH = (
    Path("/Users/jrock/repos/flows/spec-decode-paper-reproduction")
    / "dflash_block_diffusion_agent_23"
    / "golden_dflash_block_diffusion_agent_23"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("reference_kimi_k2_5_dflash.pt")


def _add_import_paths(golden_path: Path) -> None:
    for path in (REPO_ROOT, golden_path):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _resolve_snapshot(model_or_path: str, *, cache_dir: str) -> Path:
    path = Path(model_or_path)
    if path.exists():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_or_path, cache_dir=cache_dir))


def _format_humaneval(row: dict[str, Any]) -> str:
    return (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{row['prompt']}\n```"
    )


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def _host_writes_for_block(block_token_ids: list[int], *, anchor_pos: int) -> list[dict[str, int | str]]:
    return [
        {
            "token_id": int(token_id),
            "token_type": "BASE" if offset == 0 else "SPEC",
            "position_id": int(anchor_pos + offset),
            "user_id": 0,
            "prefill_token_id": -1,
        }
        for offset, token_id in enumerate(block_token_ids)
    ]


def _greedy_sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab = logits.shape
    probs = torch.softmax(logits.reshape(-1, vocab) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


class CapturingDFlashRuntime:
    """DFlash runtime that captures full-shape drafter stage I/O while proposing tokens."""

    def __init__(self, drafter, *, block_size: int | None = None, mask_token_id: int | None = None) -> None:
        self.drafter = drafter
        self.block_size = int(block_size or drafter.block_size)
        self.mask_token_id = int(mask_token_id if mask_token_id is not None else drafter.mask_token_id)
        self.target_layer_ids = list(drafter.target_layer_ids)
        self.records: list[dict[str, Any]] = []

    @torch.inference_mode()
    def propose(
        self,
        *,
        pass_index: int,
        target,
        target_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
        block_position_ids: torch.LongTensor,
        draft_position_ids: torch.LongTensor,
        temperature: float,
        anchor_position: int,
    ) -> torch.LongTensor:
        noise_embedding = target.embed(block_output_ids)
        target_context = self.drafter.hidden_norm(self.drafter.fc(target_hidden))
        position_cos, position_sin = self.drafter.rotary_emb(
            draft_position_ids,
            noise_embedding.dtype,
            noise_embedding.device,
        )

        hidden_states = noise_embedding
        decoder_records: list[dict[str, Any]] = []
        for layer_idx, layer in enumerate(self.drafter.layers):
            layer_input = hidden_states
            hidden_states = layer(layer_input, target_context, (position_cos, position_sin))
            decoder_records.append(
                {
                    "layer_idx": layer_idx,
                    "inputs": {
                        "hidden_states": _cpu_tensor(layer_input),
                        "target_context": _cpu_tensor(target_context),
                        "position_cos": _cpu_tensor(position_cos),
                        "position_sin": _cpu_tensor(position_sin),
                    },
                    "expected": {
                        "hidden_states": _cpu_tensor(hidden_states),
                    },
                }
            )

        final_hidden = self.drafter.norm(hidden_states)
        draft_logits = target.lm_head(final_hidden[:, 1 - self.block_size :, :])
        draft_token_ids = _greedy_sample(draft_logits, temperature)
        proposal = draft_token_ids[:, : self.block_size - 1]
        host_packet = {
            "type": "DRAFT_BLOCK_PROPOSAL",
            "anchor_position": int(anchor_position),
            "token_ids": [int(x) for x in proposal[0].detach().cpu().tolist()],
            "positions": list(range(int(anchor_position) + 1, int(anchor_position) + self.block_size)),
        }

        self.records.append(
            {
                "pass_index": int(pass_index),
                "anchor_position": int(anchor_position),
                "pre_decoder_fused": {
                    "inputs": {
                        "target_hidden": _cpu_tensor(target_hidden),
                        "noise_embedding": _cpu_tensor(noise_embedding),
                        "block_output_ids": _cpu_tensor(block_output_ids),
                        "block_position_ids": _cpu_tensor(block_position_ids),
                        "draft_position_ids": _cpu_tensor(draft_position_ids),
                    },
                    "expected": {
                        "target_context": _cpu_tensor(target_context),
                        "position_cos": _cpu_tensor(position_cos),
                        "position_sin": _cpu_tensor(position_sin),
                        "decoder_input": _cpu_tensor(noise_embedding),
                    },
                },
                "decoder_layers": decoder_records,
                "post_decoder_fused": {
                    "inputs": {
                        "hidden_states": _cpu_tensor(hidden_states),
                        "anchor_position": int(anchor_position),
                    },
                    "expected": {
                        "final_hidden": _cpu_tensor(final_hidden),
                        "draft_logits": _cpu_tensor(draft_logits),
                        "draft_token_ids": _cpu_tensor(draft_token_ids),
                        "host_packet": host_packet,
                    },
                },
                "combined_drafter": {
                    "expected": {
                        "final_hidden": _cpu_tensor(final_hidden),
                        "draft_logits": _cpu_tensor(draft_logits),
                        "draft_token_ids": _cpu_tensor(draft_token_ids),
                        "host_packet": host_packet,
                    },
                },
                "tensor_shapes": {
                    "target_hidden": _tensor_meta(target_hidden),
                    "noise_embedding": _tensor_meta(noise_embedding),
                    "target_context": _tensor_meta(target_context),
                    "position_cos": _tensor_meta(position_cos),
                    "position_sin": _tensor_meta(position_sin),
                    "final_hidden": _tensor_meta(final_hidden),
                    "draft_logits": _tensor_meta(draft_logits),
                    "draft_token_ids": _tensor_meta(draft_token_ids),
                },
            }
        )
        return proposal


class CapturingSpecDecodeManager:
    """Lossless DFlash manager that mirrors the golden loop and records host/device packets."""

    def __init__(self, target, draft_runtime: CapturingDFlashRuntime, *, temperature: float = 0.0) -> None:
        self.target = target
        self.draft = draft_runtime
        self.temperature = float(temperature)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        stop_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        from golden_dflash.spec_decode import extract_context_feature

        if input_ids.shape[0] != 1:
            raise ValueError("Reference capture supports batch size 1.")

        device = self.target.device
        input_ids = input_ids.to(device)
        num_input_tokens = int(input_ids.shape[1])
        max_length = num_input_tokens + int(max_new_tokens)
        block_size = int(self.draft.block_size)
        stop_token_ids = stop_token_ids if stop_token_ids is not None else []

        output_ids = torch.full(
            (1, max_length + block_size),
            fill_value=self.draft.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        prefill = self.target.prefill(
            input_ids,
            position_ids[:, :num_input_tokens],
            output_hidden_states=block_size > 1,
        )
        output_ids[:, :num_input_tokens] = input_ids
        first = _greedy_sample(prefill.logits[:, -1:, :], self.temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = first

        target_hidden_context = None
        if block_size > 1:
            target_hidden_context = extract_context_feature(prefill.hidden_states, self.draft.target_layer_ids)

        start = num_input_tokens
        acceptance_lengths: list[int] = []
        host_writes: list[dict[str, int | str]] = []
        draft_blocks: list[dict[str, Any]] = []
        verification_packets: list[dict[str, Any]] = []
        pass_index = 0

        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]

            if block_size > 1:
                draft_position_ids = position_ids[:, : start + block_size]
                proposal = self.draft.propose(
                    pass_index=pass_index,
                    target=self.target,
                    target_hidden=target_hidden_context,
                    block_output_ids=block_output_ids,
                    block_position_ids=block_position_ids,
                    draft_position_ids=draft_position_ids,
                    temperature=self.temperature,
                    anchor_position=start,
                )
                block_output_ids[:, 1:] = proposal[:, : block_size - 1]

            block_token_ids = [int(x) for x in block_output_ids[0].detach().cpu().tolist()]
            host_writes.extend(_host_writes_for_block(block_token_ids, anchor_pos=start))
            draft_blocks.append(
                {
                    "pass_index": pass_index,
                    "anchor_pos": int(start),
                    "token_ids": block_token_ids,
                    "position_ids": [int(x) for x in block_position_ids[0].detach().cpu().tolist()],
                }
            )

            verified = self.target.verify(
                block_output_ids,
                block_position_ids,
                output_hidden_states=block_size > 1,
            )
            posterior = _greedy_sample(verified.logits, self.temperature)
            matches = block_output_ids[:, 1:] == posterior[:, :-1]
            accepted_after_anchor = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
            committed = accepted_after_anchor + 1

            posterior_token_ids = [int(x) for x in posterior[0].detach().cpu().tolist()]
            verification_packets.append(
                {
                    "pass_index": pass_index,
                    "anchor_pos": int(start),
                    "target_posterior_token_ids": posterior_token_ids,
                    "accepted_after_anchor": int(accepted_after_anchor),
                    "committed_tokens": int(committed),
                }
            )

            output_ids[:, start : start + committed] = block_output_ids[:, :committed]
            if start + committed < output_ids.shape[1]:
                output_ids[:, start + committed] = posterior[:, accepted_after_anchor]
            acceptance_lengths.append(committed)
            start += committed
            if hasattr(self.target, "crop_cache"):
                self.target.crop_cache(start)

            if block_size > 1:
                new_hidden = extract_context_feature(verified.hidden_states, self.draft.target_layer_ids)[
                    :, :committed, :
                ]
                target_hidden_context = torch.cat([target_hidden_context, new_hidden], dim=1)

            pass_index += 1
            if stop_token_ids:
                generated = output_ids[0, num_input_tokens:start]
                if any(int(x) in stop_token_ids for x in generated.tolist()):
                    break

        output_ids = output_ids[:, : min(start, max_length)]
        if stop_token_ids:
            generated = output_ids[0, num_input_tokens:]
            for idx, token_id in enumerate(generated.tolist()):
                if int(token_id) in stop_token_ids:
                    output_ids = output_ids[:, : num_input_tokens + idx + 1]
                    break

        generated_token_ids = [int(x) for x in output_ids[0, num_input_tokens:].detach().cpu().tolist()]
        output_token_ids = [int(x) for x in output_ids[0].detach().cpu().tolist()]
        num_accepts = sum(int(packet["accepted_after_anchor"]) for packet in verification_packets)
        num_rejects = sum(
            int(packet["accepted_after_anchor"]) < block_size - 1 for packet in verification_packets
        )
        eos_token_id = stop_token_ids[0] if len(stop_token_ids) == 1 else None
        return {
            "params": {
                "prompt_token_ids": [int(x) for x in input_ids[0].detach().cpu().tolist()],
                "max_new_tokens": int(max_new_tokens),
                "eos_token_id": eos_token_id,
                "block_size": int(block_size),
                "temperature": float(self.temperature),
                "top_k": 1,
                "top_p": 1.0,
            },
            "num_input_tokens": int(num_input_tokens),
            "num_output_tokens": int(len(generated_token_ids)),
            "generated_token_ids": generated_token_ids,
            "output_token_ids": output_token_ids,
            "acceptance_lengths": acceptance_lengths,
            "verification_passes": int(len(acceptance_lengths)),
            "committed_tokens": int(sum(acceptance_lengths)),
            "num_accepts": int(num_accepts),
            "num_rejects": int(num_rejects),
            "average_committed_tokens": float(sum(acceptance_lengths) / len(acceptance_lengths))
            if acceptance_lengths
            else 0.0,
            "host_writes": host_writes,
            "draft_blocks": draft_blocks,
            "target_verification_packets": verification_packets,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real Kimi-K2.5 DFlash reference I/O for tt-metal tests.")
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--target-model", default="moonshotai/Kimi-K2.5")
    parser.add_argument("--draft-model", default="z-lab/Kimi-K2.5-DFlash")
    parser.add_argument("--cache-dir", default="/workspace/models")
    parser.add_argument("--vllm-hidden-dir", default="/workspace/jrock/dflash_reference/vllm_hidden")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--humaneval-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=None, help="Alias for --max-new-tokens for small captures.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="0 means use the drafter checkpoint config block_size.",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--enable-precompute-kv-ctx", action="store_true")
    args = parser.parse_args()
    max_new_tokens = int(args.num_tokens if args.num_tokens is not None else args.max_new_tokens)

    _add_import_paths(args.golden_path.resolve())

    from datasets import load_dataset
    from golden_dflash.config import DFlashConfig
    from golden_dflash.draft_model import KimiDFlashDraftModel
    from golden_dflash.vllm_kimi_target import VllmKimiTargetAdapter

    target_dir = _resolve_snapshot(args.target_model, cache_dir=args.cache_dir)
    draft_dir = _resolve_snapshot(args.draft_model, cache_dir=args.cache_dir)
    config = DFlashConfig.from_json(draft_dir / "config.json")
    block_size = int(args.block_size or config.block_size)

    print(f"Loading drafter checkpoint from {draft_dir}", flush=True)
    drafter_kwargs: dict[str, Any] = {}
    if "enable_precompute_kv_ctx" in inspect.signature(KimiDFlashDraftModel).parameters:
        drafter_kwargs["enable_precompute_kv_ctx"] = args.enable_precompute_kv_ctx
    elif args.enable_precompute_kv_ctx:
        raise RuntimeError("The selected golden KimiDFlashDraftModel does not support --enable-precompute-kv-ctx.")
    drafter = KimiDFlashDraftModel(config, **drafter_kwargs).to(dtype=torch.bfloat16, device="cuda:0").eval()
    drafter.load_safetensors(draft_dir)

    print(f"Loading target through vLLM from {args.target_model} (snapshot {target_dir})", flush=True)
    target = VllmKimiTargetAdapter.from_pretrained(
        args.target_model,
        snapshot_dir=target_dir,
        layer_ids=config.target_layer_ids,
        storage_dir=args.vllm_hidden_dir,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    dataset = load_dataset("openai/openai_humaneval", split="test")
    row = dataset[int(args.humaneval_index)]
    prompt_text = _format_humaneval(row)
    messages = [{"role": "user", "content": prompt_text}]
    rendered_prompt = target.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    input_ids = target.tokenizer.encode(rendered_prompt, return_tensors="pt").to(target.device)

    runtime = CapturingDFlashRuntime(drafter, block_size=block_size)
    manager = CapturingSpecDecodeManager(target, runtime)
    target.reset_cache()

    started = time.perf_counter()
    host_trace = manager.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=[target.tokenizer.eos_token_id],
    )
    elapsed_s = time.perf_counter() - started

    reference = {
        "schema_version": 1,
        "metadata": {
            "approach": "dflash",
            "target_model": "moonshotai/Kimi-K2.5",
            "draft_model": "z-lab/Kimi-K2.5-DFlash",
            "target_model_arg": args.target_model,
            "draft_model_arg": args.draft_model,
            "target_snapshot_dir": str(target_dir),
            "draft_snapshot_dir": str(draft_dir),
            "golden_path": str(args.golden_path.resolve()),
            "dataset": "openai/openai_humaneval",
            "humaneval_index": int(args.humaneval_index),
            "task_id": row.get("task_id"),
            "max_new_tokens": int(max_new_tokens),
            "block_size": int(block_size),
            "temperature": 0.0,
            "top_k": 1,
            "use_chat_template": True,
            "enable_thinking": True,
            "enable_precompute_kv_ctx": bool(args.enable_precompute_kv_ctx),
            "tp_size": int(args.tp_size),
            "max_model_len": int(args.max_model_len),
            "max_num_batched_tokens": int(args.max_num_batched_tokens),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "elapsed_s": float(elapsed_s),
            "weights_policy": (
                "Drafter and target weights are loaded from checkpoint/state dict; "
                "weights are not stored in this reference file."
            ),
        },
        "config": {
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "block_size": int(config.block_size),
            "runtime_block_size": int(block_size),
            "max_position_embeddings": int(config.max_position_embeddings),
            "rope_theta": float(config.rope_theta),
            "rms_norm_eps": float(config.rms_norm_eps),
            "target_layer_ids": list(config.target_layer_ids),
            "mask_token_id": int(config.mask_token_id),
            "rope_scaling": dict(config.rope_scaling),
        },
        "prompt": {
            "text": prompt_text,
            "rendered_chat_prompt": rendered_prompt,
            "input_ids": _cpu_tensor(input_ids),
        },
        "host_trace": host_trace,
        "stages": {
            "passes": runtime.records,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference, args.output)
    print(f"Wrote {args.output}", flush=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": host_trace["num_output_tokens"],
                "requested_max_new_tokens": int(max_new_tokens),
                "verification_passes": host_trace["verification_passes"],
                "average_committed_tokens": host_trace["average_committed_tokens"],
                "captured_stage_passes": len(runtime.records),
                "elapsed_s": elapsed_s,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
