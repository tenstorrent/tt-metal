# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run prompt-correct HF/TT qualitative controls for the full-model generator."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.autoports.google_gemma_4_31b.tt.model import (
    Gemma4FullModelConfig,
    _load_checkpoint_state,
    _resolve_checkpoint,
)
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.schema import load_reference


def _load_prompts(path: Path) -> list[str]:
    prompts = [entry.strip() for entry in path.read_text(encoding="utf-8").split("\n\n") if entry.strip()]
    if not prompts:
        raise ValueError(f"prompt source is empty: {path}")
    return prompts


def _generate_hf(model, tokenizer, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    source = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        generated = model.generate(
            source,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    return generated[0, source.shape[1] :].cpu().tolist()


def _stable_logits_summary(logits: torch.Tensor, *, top_k: int = 10) -> dict:
    """Summarize one BF16 vocabulary row with deterministic lower-ID tie order."""
    logits = logits.reshape(-1).to(torch.bfloat16).contiguous().cpu()
    scores = logits.float()
    order = torch.argsort(scores, descending=True, stable=True)
    top_ids = order[:top_k]
    bits = logits.view(torch.int16)
    maximum = scores[top_ids[0]]
    max_ids = torch.nonzero(scores == maximum, as_tuple=False).reshape(-1)
    return {
        "argmax": int(top_ids[0]),
        "top1_top2_margin": float(scores[top_ids[0]] - scores[top_ids[1]]),
        "exact_max_count": int(max_ids.numel()),
        "exact_max_ids": [int(token) for token in max_ids[:64]],
        "top": [
            {
                "token_id": int(token),
                "score": float(scores[token]),
                "bf16_bits": int(bits[token]) & 0xFFFF,
            }
            for token in top_ids
        ],
    }


def _aligned_logits_comparison(legacy: torch.Tensor, optimized: torch.Tensor) -> dict:
    legacy_bf16 = legacy.reshape(-1).to(torch.bfloat16).contiguous().cpu()
    optimized_bf16 = optimized.reshape(-1).to(torch.bfloat16).contiguous().cpu()
    if legacy_bf16.shape != optimized_bf16.shape:
        raise ValueError(f"aligned logits must have the same shape: {legacy_bf16.shape} != {optimized_bf16.shape}")
    legacy_float = legacy_bf16.float()
    optimized_float = optimized_bf16.float()
    delta = optimized_float - legacy_float
    legacy_centered = legacy_float - legacy_float.mean()
    optimized_centered = optimized_float - optimized_float.mean()
    denominator = torch.linalg.vector_norm(legacy_centered) * torch.linalg.vector_norm(optimized_centered)
    pcc = float(torch.dot(legacy_centered, optimized_centered) / denominator) if denominator > 0 else 1.0
    pcc = max(-1.0, min(1.0, pcc))
    overlap_k = min(10, legacy_bf16.numel())
    legacy_top = _stable_logits_summary(legacy_bf16, top_k=overlap_k)["top"]
    optimized_top = _stable_logits_summary(optimized_bf16, top_k=overlap_k)["top"]
    return {
        "pcc": pcc,
        "exact_bf16_fraction": float(
            (legacy_bf16.view(torch.int16) == optimized_bf16.view(torch.int16)).float().mean()
        ),
        "max_abs_delta": float(delta.abs().max()),
        "mean_abs_delta": float(delta.abs().mean()),
        "rms_delta": float(torch.sqrt(torch.mean(delta.square()))),
        "top10_overlap": len(
            {entry["token_id"] for entry in legacy_top} & {entry["token_id"] for entry in optimized_top}
        ),
    }


def _block_match_summary(legacy: torch.Tensor, optimized: torch.Tensor, *, block_size: int = 8192) -> list[dict]:
    legacy = legacy.reshape(-1).float().cpu()
    optimized = optimized.reshape(-1).float().cpu()
    if legacy.shape != optimized.shape or legacy.numel() % block_size:
        raise ValueError("block comparison requires equal rows divisible by block_size")
    legacy_blocks = legacy.reshape(-1, block_size)
    optimized_blocks = optimized.reshape(-1, block_size)
    result = []
    for index, block in enumerate(optimized_blocks):
        losses = torch.mean((legacy_blocks - block) ** 2, dim=1)
        best = int(torch.argmin(losses))
        result.append(
            {
                "optimized_block": index,
                "best_legacy_block": best,
                "best_mse": float(losses[best]),
                "identity_mse": float(losses[index]),
            }
        )
    return result


def _boundary_deltas(legacy: torch.Tensor, optimized: torch.Tensor) -> list[dict]:
    legacy = legacy.reshape(-1).float().cpu()
    optimized = optimized.reshape(-1).float().cpu()
    vocab_size = legacy.numel()
    boundaries = set(range(8192, vocab_size, 8192)) | set(range(65_536, vocab_size, 65_536))
    token_ids = sorted(
        token for boundary in boundaries for token in range(max(0, boundary - 2), min(vocab_size, boundary + 3))
    )
    return [
        {
            "token_id": token,
            "legacy": float(legacy[token]),
            "optimized": float(optimized[token]),
            "delta": float(optimized[token] - legacy[token]),
        }
        for token in token_ids
    ]


def _tensor_descriptor(tensor: ttnn.Tensor) -> dict:
    def shape(value) -> list[int]:
        return [int(dimension) for dimension in value]

    return {
        "shape": shape(tensor.shape),
        "padded_shape": shape(tensor.padded_shape),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "memory_config": str(tensor.memory_config()),
        "device_shapes": [shape(device_tensor.shape) for device_tensor in ttnn.get_device_tensors(tensor)],
    }


def _compose_bf16_row(model, logits: ttnn.Tensor, row: int) -> torch.Tensor:
    host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=-1))
    return host.reshape(-1, model.vocab_size)[row, : model.vocab_size].to(torch.bfloat16).contiguous().cpu()


def _apply_logit_softcap(model, logits: ttnn.Tensor) -> ttnn.Tensor:
    if model.final_logit_softcapping <= 0:
        return logits
    scaled = ttnn.mul(logits, 1.0 / model.final_logit_softcapping)
    logits.deallocate(True)
    capped = ttnn.tanh(scaled)
    scaled.deallocate(True)
    result = ttnn.mul(capped, model.final_logit_softcapping)
    capped.deallocate(True)
    return result


def _sample_device_row(generator, logits: ttnn.Tensor, row: int) -> int:
    row_view = ttnn.slice(
        logits,
        [0, 0, row, 0],
        [1, 1, row + 1, logits.shape[-1]],
        memory_config=logits.memory_config(),
    )
    owned_row = ttnn.clone(row_view, memory_config=logits.memory_config())
    del row_view
    output = generator._new_token_buffer(1)
    try:
        sampled, _ = generator._sample_eager(owned_row, tt_out_tok=output)
        ttnn.synchronize_device(generator.mesh_device)
        return generator.read_sampled_token(sampled)
    finally:
        if output.is_allocated():
            output.deallocate(True)
        if owned_row.is_allocated():
            owned_row.deallocate(True)


def _deallocate_if_live(tensor) -> None:
    if tensor is not None and tensor.is_allocated():
        tensor.deallocate(True)


def _run_lm_head_aligned_ab(generator, prompt_token_ids: list[int], checkpoint: Path) -> dict:
    """Compare legacy and optimized LM heads from one normalized real-model hidden tile."""
    model = generator.model
    legacy_weight = hidden = normed = legacy_normed = None
    legacy_pre = optimized_pre = legacy_post = optimized_post = None
    cache_touched = False
    try:
        state = _load_checkpoint_state(_resolve_checkpoint(checkpoint), layer_indices=())
        embedding = state["model.language_model.embed_tokens.weight"].to(torch.bfloat16)
        legacy_source = embedding.transpose(0, 1).contiguous()
        legacy_weight = ttnn.from_torch(
            legacy_source,
            device=model.mesh_device,
            dtype=model.config.lm_head_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(model.mesh_device, dim=-1),
        )
        del state, embedding, legacy_source
        gc.collect()

        prompt_len = len(prompt_token_ids)
        if not 1 <= prompt_len <= ttnn.TILE_SIZE:
            raise ValueError("aligned LM-head diagnostic currently requires a one-tile prompt")
        cache_touched = True
        generator._cache_dirty = True
        hidden = model.prefill_hidden(
            torch.tensor([prompt_token_ids], dtype=torch.long),
            page_tables=model._normalize_page_tables(generator.page_tables),
            kv_cache=generator.kv_cache,
            user_id=0,
            prompt_len=prompt_len,
        )
        normed = model.final_norm.forward(hidden)
        hidden.deallocate(True)
        hidden = None
        legacy_normed = ttnn.clone(normed, memory_config=normed.memory_config())

        optimized_pre = model._project_sharded_lm_head_tile(normed)
        normed = None  # _project_sharded_lm_head_tile owns and deallocates its input.
        legacy_pre = ttnn.linear(
            legacy_normed,
            legacy_weight,
            dtype=model.config.logits_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=model.lm_head_compute,
        )
        legacy_normed.deallocate(True)
        legacy_normed = None

        row = prompt_len - 1
        legacy_pre_host = _compose_bf16_row(model, legacy_pre, row)
        optimized_pre_host = _compose_bf16_row(model, optimized_pre, row)
        pre_descriptors = {
            "legacy": _tensor_descriptor(legacy_pre),
            "optimized": _tensor_descriptor(optimized_pre),
        }
        legacy_post = _apply_logit_softcap(model, legacy_pre)
        legacy_pre = None
        optimized_post = _apply_logit_softcap(model, optimized_pre)
        optimized_pre = None
        legacy_post_host = _compose_bf16_row(model, legacy_post, row)
        optimized_post_host = _compose_bf16_row(model, optimized_post, row)
        post_descriptors = {
            "legacy": _tensor_descriptor(legacy_post),
            "optimized": _tensor_descriptor(optimized_post),
        }
        legacy_sample = _sample_device_row(generator, legacy_post, row)
        optimized_sample = _sample_device_row(generator, optimized_post, row)

        def evidence(legacy: torch.Tensor, optimized: torch.Tensor) -> dict:
            legacy_summary = _stable_logits_summary(legacy)
            optimized_summary = _stable_logits_summary(optimized)
            candidates = {
                str(token): {
                    "legacy": float(legacy[token].float()),
                    "optimized": float(optimized[token].float()),
                    "delta": float(optimized[token].float() - legacy[token].float()),
                }
                for token in (108, 669)
            }
            return {
                "legacy": legacy_summary,
                "optimized": optimized_summary,
                "aligned": _aligned_logits_comparison(legacy, optimized),
                "block_matches_8192": _block_match_summary(legacy, optimized),
                "boundary_deltas": _boundary_deltas(legacy, optimized),
                "known_prompt5_candidates": candidates,
            }

        return {
            "prompt_token_ids": prompt_token_ids,
            "prompt_len": prompt_len,
            "logical_row": row,
            "config": {
                "legacy": {
                    "weight_layout": "interleaved DRAM",
                    "program_config": "auto (no explicit program_config)",
                    "weight_dtype": str(model.config.lm_head_weight_dtype),
                    "logits_dtype": str(model.config.logits_dtype),
                    "math_fidelity": str(model.config.lm_head_math_fidelity),
                },
                "optimized": {
                    "weight_layout": "width-sharded DRAM",
                    "num_cores": model.config.lm_head_num_cores,
                    "in0_block_w": model.config.lm_head_in0_block_w,
                    "split_size": model.config.lm_head_split_size,
                    "program_config": str(model.lm_head_program_config),
                },
                "softcap": model.final_logit_softcapping,
            },
            "pre_softcap": {"tensor": pre_descriptors, **evidence(legacy_pre_host, optimized_pre_host)},
            "post_softcap": {"tensor": post_descriptors, **evidence(legacy_post_host, optimized_post_host)},
            "sampler": {
                "legacy_device_token": legacy_sample,
                "legacy_host_argmax": int(torch.argmax(legacy_post_host.float())),
                "optimized_device_token": optimized_sample,
                "optimized_host_argmax": int(torch.argmax(optimized_post_host.float())),
            },
        }
    finally:
        for tensor in (
            legacy_pre,
            optimized_pre,
            legacy_post,
            optimized_post,
            legacy_normed,
            normed,
            hidden,
            legacy_weight,
        ):
            _deallocate_if_live(tensor)
        if cache_touched:
            generator._cache_dirty = True
            generator.reset()


def _model_config_from_environment() -> Gemma4FullModelConfig:
    return Gemma4FullModelConfig(
        lm_head_dram_sharded=os.environ.get("GEMMA4_31B_LM_HEAD_DRAM_SHARDED", "1") == "1",
        lm_head_num_cores=int(os.environ.get("GEMMA4_31B_LM_HEAD_NUM_CORES", "4")),
        lm_head_in0_block_w=int(os.environ.get("GEMMA4_31B_LM_HEAD_IN0_BLOCK_W", "2")),
        lm_head_split_size=int(os.environ.get("GEMMA4_31B_LM_HEAD_SPLIT_SIZE", "8192")),
    )


def _run_aligned_ab_only(args: argparse.Namespace) -> None:
    if args.aligned_ab_output is None:
        raise ValueError("--aligned-ab-only requires --aligned-ab-output")
    prompts = _load_prompts(args.prompt_source)
    if len(prompts) <= 5:
        raise ValueError("LM-head aligned A/B requires prompt ID 5")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.chat_template:
        raise RuntimeError("google/gemma-4-31B unexpectedly acquired a chat template; update prompt rendering")
    prompt_token_ids = tokenizer.encode(prompts[5], add_special_tokens=True)
    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            model_dir=args.model_dir,
            mesh_device=mesh,
            model_config=_model_config_from_environment(),
        )
        result = _run_lm_head_aligned_ab(generator, prompt_token_ids, args.hf_model)
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")
    args.aligned_ab_output.parent.mkdir(parents=True, exist_ok=True)
    args.aligned_ab_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _run_benchmark_only(args: argparse.Namespace) -> None:
    if args.benchmark_reference is None or args.benchmark_output is None:
        raise ValueError("--benchmark-only requires --benchmark-reference and --benchmark-output")
    if args.benchmark_warmups < 0 or args.benchmark_repeats < 1:
        raise ValueError("benchmark warmups must be nonnegative and repeats must be positive")
    reference = load_reference(args.benchmark_reference)
    prompt_token_ids = reference.entries[0].prompt_tokens.reshape(-1).tolist()
    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        config = _model_config_from_environment()
        generator = build_generator(model_dir=args.model_dir, mesh_device=mesh, model_config=config)
        for _ in range(args.benchmark_warmups):
            generator.benchmark_token_out_no_readback(prompt_token_ids, max_new_tokens=args.benchmark_tokens)
        samples = [
            generator.benchmark_token_out_no_readback(prompt_token_ids, max_new_tokens=args.benchmark_tokens)
            for _ in range(args.benchmark_repeats)
        ]
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")

    metrics = ("ttft_ms", "decode_t/s/u", "steady_decode_t/s/u")
    summary = {}
    for metric in metrics:
        values = [float(sample[metric]) for sample in samples]
        summary[metric] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "samples": values,
        }
    result = {
        "configuration": {
            "lm_head_dram_sharded": config.lm_head_dram_sharded,
            "lm_head_num_cores": config.lm_head_num_cores,
            "lm_head_in0_block_w": config.lm_head_in0_block_w,
            "lm_head_split_size": config.lm_head_split_size,
        },
        "workload": {"prompt_len": len(prompt_token_ids), "gen_len": args.benchmark_tokens, "batch": 1},
        "warmups": args.benchmark_warmups,
        "repeats": args.benchmark_repeats,
        "summary": summary,
        "samples": samples,
    }
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.aligned_ab_only:
        _run_aligned_ab_only(args)
        return
    if args.benchmark_only:
        _run_benchmark_only(args)
        return
    prompts = _load_prompts(args.prompt_source)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.chat_template:
        raise RuntimeError("google/gemma-4-31B unexpectedly acquired a chat template; update prompt rendering")

    rendered = []
    for prompt_id, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        rendered.append({"id": prompt_id, "prompt": prompt, "prompt_token_ids": token_ids})

    hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    hf_model.to(hf_device)
    outputs = []
    for entry in rendered:
        tokens = _generate_hf(hf_model, tokenizer, entry["prompt_token_ids"], args.max_new_tokens)
        outputs.append(
            {
                "id": entry["id"],
                "prompt": entry["prompt"],
                "hf_token_ids": tokens,
                "hf_greedy_completion": tokenizer.decode(tokens, skip_special_tokens=False),
            }
        )
    del hf_model
    gc.collect()
    if hf_device.type == "cuda":
        torch.cuda.empty_cache()

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    benchmark = None
    aligned_ab = None
    try:
        generator = build_generator(
            model_dir=args.model_dir,
            mesh_device=mesh,
            model_config=_model_config_from_environment(),
        )
        if args.lm_head_aligned_ab:
            if len(rendered) <= 5:
                raise ValueError("LM-head aligned A/B requires prompt ID 5")
            aligned_ab = _run_lm_head_aligned_ab(
                generator,
                rendered[5]["prompt_token_ids"],
                args.hf_model,
            )
        for output, entry in zip(outputs, rendered):
            tokens = generator.generate(
                prompt_token_ids=entry["prompt_token_ids"],
                max_new_tokens=args.max_new_tokens,
                enable_trace=True,
                stop_on_eos=True,
            )
            output["tt_token_ids"] = tokens
            output["tt_greedy_completion"] = tokenizer.decode(tokens, skip_special_tokens=False)
            output["greedy_completion"] = output["tt_greedy_completion"]
        if args.benchmark_reference is not None:
            reference = load_reference(args.benchmark_reference)
            benchmark_prompt = reference.entries[0].prompt_tokens.reshape(-1).tolist()
            benchmark = generator.benchmark_token_out_no_readback(
                benchmark_prompt, max_new_tokens=args.benchmark_tokens
            )
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hf_model": str(args.hf_model),
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": False,
        "prompt_mode": "completion",
        "rendering_method": "tokenizer.encode(prompt, add_special_tokens=True)",
        "prompt_source_path": str(args.prompt_source),
        "max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False, "num_beams": 1},
    }
    (args.output_dir / "qualitative_prompt_format.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "qualitative_rendered_prompts.json").write_text(
        json.dumps(rendered, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "vllm_qualitative_outputs.json").write_text(
        json.dumps(outputs, indent=2) + "\n", encoding="utf-8"
    )
    if aligned_ab is not None:
        (args.output_dir / "lm_head_aligned_ab.json").write_text(
            json.dumps(aligned_ab, indent=2) + "\n", encoding="utf-8"
        )
    if benchmark is not None:
        (args.output_dir.parent / "token_out_no_readback.json").write_text(
            json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"prompt_count": len(outputs), "output_dir": str(args.output_dir)}))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--benchmark-reference", type=Path)
    parser.add_argument("--benchmark-tokens", type=int, default=100)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-warmups", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="skip text generation and collect repeated full-model token-out timing samples",
    )
    parser.add_argument("--aligned-ab-output", type=Path)
    parser.add_argument(
        "--aligned-ab-only",
        action="store_true",
        help="skip HF/text generation and run only the same-hidden legacy/optimized LM-head comparison",
    )
    parser.add_argument(
        "--lm-head-aligned-ab",
        action="store_true",
        help="compare legacy and optimized LM-head logits from one normalized prompt-5 hidden tile",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
