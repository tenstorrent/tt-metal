# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay one DiffusionGemma denoise block through HF and TT.

This is a focused bring-up harness for R0.5/#48291 fidelity debugging. It drives
both implementations with the same prompt, initial canvas, Gumbel noise, and
renoise tokens, then saves the decision-level trajectory comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from models.experimental.diffusion_gemma.checkpoint import (
    build_tt_model_from_checkpoint_inputs,
    generate_text_from_checkpoint_model_inputs,
    load_checkpoint_inputs,
    text_generation_prefixes_for_layers,
)
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _log_mesh_dram, _open_mesh_device
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.generate import (
    make_host_canvas_init_fn,
    make_host_gumbel_noise_fn,
    make_host_noise_tokens_fn,
    tokenize_prompt,
)

DEFAULT_PROMPT = "Complete the sentence: Once upon a time"


def _pad_prompt_tokens_for_hf_prefill(prompt_tokens: torch.Tensor, *, multiple: int = 32) -> torch.Tensor:
    pad = (-prompt_tokens.shape[1]) % multiple
    if pad == 0:
        return prompt_tokens
    padding = torch.zeros((prompt_tokens.shape[0], pad), dtype=prompt_tokens.dtype, device=prompt_tokens.device)
    return torch.cat([prompt_tokens, padding], dim=1)


def _make_config(args) -> DiffusionConfig:
    return DiffusionConfig(
        canvas_length=args.canvas_length,
        max_denoise_steps=args.max_denoising_steps,
        entropy_stop_threshold=args.entropy_stop_threshold,
        stable_steps_to_halt=args.stable_steps_to_halt,
    )


def _top_counts(tokens: torch.Tensor, *, limit: int = 8) -> list[list[int]]:
    values, counts = torch.unique(tokens.cpu(), return_counts=True)
    pairs = sorted([(int(tok), int(count)) for tok, count in zip(values, counts)], key=lambda x: -x[1])
    return [[tok, count] for tok, count in pairs[:limit]]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _make_replay_noise(
    *,
    seed: int,
    steps: int,
    canvas_length: int,
    vocab_size: int,
    mode: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if mode == "zero":
        return (
            [torch.zeros((1, canvas_length, vocab_size), dtype=torch.float32) for _ in range(steps)],
            [torch.zeros((1, canvas_length), dtype=torch.long) for _ in range(steps)],
        )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    gumbel_noise = []
    renoise_tokens = []
    for _step in range(steps):
        uniform = torch.rand((1, canvas_length, vocab_size), dtype=torch.float32, generator=generator)
        uniform = uniform.clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - torch.finfo(torch.float32).eps)
        gumbel_noise.append(-torch.log(-torch.log(uniform)))
        renoise_tokens.append(torch.randint(0, vocab_size, (1, canvas_length), dtype=torch.long, generator=generator))
    return gumbel_noise, renoise_tokens


def _logits_topk_summary(logits: torch.Tensor, positions: list[int], *, k: int, step: int) -> dict:
    logits = logits.float().cpu()
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    rows = []
    for pos in positions:
        values, token_ids = torch.topk(logits[0, pos], k=k)
        rows.append(
            {
                "pos": pos,
                "token_ids": [int(token_id) for token_id in token_ids],
                "values": [float(value) for value in values],
            }
        )
    return {"step": step, "rows": rows}


def _pcc(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.float().flatten()
    rhs = rhs.float().flatten()
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    denominator = lhs.norm() * rhs.norm()
    if denominator == 0:
        return 1.0 if torch.equal(lhs, rhs) else 0.0
    return float(torch.dot(lhs, rhs) / denominator)


def _layer_hidden_summary(hf_layers, tt_layers, positions: list[int]) -> list[dict]:
    summaries = []
    for layer_idx, (hf_hidden, tt_hidden) in enumerate(zip(hf_layers, tt_layers)):
        per_position = []
        for row_idx, pos in enumerate(positions):
            hf_row = hf_hidden[0, row_idx]
            tt_row = tt_hidden[0, row_idx]
            per_position.append(
                {
                    "pos": pos,
                    "pcc": _pcc(hf_row, tt_row),
                    "max_abs": float((hf_row - tt_row).abs().max()),
                }
            )
        summaries.append(
            {
                "layer": layer_idx,
                "pcc": _pcc(hf_hidden, tt_hidden),
                "max_abs": float((hf_hidden - tt_hidden).abs().max()),
                "per_position": per_position,
            }
        )
    return summaries


def _trajectory_summary(prefix: str, trajectory, *, eos_token_id: int | None) -> dict:
    eos_count = None
    non_eos_count = None
    if eos_token_id is not None:
        eos_count = int((trajectory.committed == eos_token_id).sum())
        non_eos_count = int((trajectory.committed != eos_token_id).sum())
    return {
        f"{prefix}_num_steps": trajectory.num_steps,
        f"{prefix}_halted": trajectory.halted,
        f"{prefix}_committed_eos": eos_count,
        f"{prefix}_committed_non_eos": non_eos_count,
        f"{prefix}_committed_top": _top_counts(trajectory.committed),
        f"{prefix}_accept_counts": [int(step.accept_mask.sum()) for step in trajectory.per_step],
    }


def _decision_diff_summary(hf_traj, tt_traj, *, limit: int = 32) -> dict:
    per_step = []
    for hf_step, tt_step in zip(hf_traj.per_step, tt_traj.per_step):
        argmax_diff = (hf_step.argmax != tt_step.argmax).nonzero(as_tuple=False)
        accept_diff = (hf_step.accept_mask != tt_step.accept_mask).nonzero(as_tuple=False)
        canvas_diff = (hf_step.canvas != tt_step.canvas).nonzero(as_tuple=False)
        step_summary = {
            "step": hf_step.step,
            "argmax_diff_count": int(argmax_diff.shape[0]),
            "accept_diff_count": int(accept_diff.shape[0]),
            "canvas_diff_count": int(canvas_diff.shape[0]),
            "argmax_diff": [],
            "accept_diff": [],
        }
        for b, pos in argmax_diff[:limit].tolist():
            step_summary["argmax_diff"].append(
                {
                    "batch": int(b),
                    "pos": int(pos),
                    "hf_argmax": int(hf_step.argmax[b, pos]),
                    "tt_argmax": int(tt_step.argmax[b, pos]),
                    "hf_sampled": int(hf_step.sampled[b, pos]),
                    "tt_sampled": int(tt_step.sampled[b, pos]),
                    "hf_accept": bool(hf_step.accept_mask[b, pos]),
                    "tt_accept": bool(tt_step.accept_mask[b, pos]),
                    "hf_entropy": float(hf_step.entropy[b, pos]),
                    "tt_entropy": float(tt_step.entropy[b, pos]),
                }
            )
        for b, pos in accept_diff[:limit].tolist():
            step_summary["accept_diff"].append(
                {
                    "batch": int(b),
                    "pos": int(pos),
                    "hf_accept": bool(hf_step.accept_mask[b, pos]),
                    "tt_accept": bool(tt_step.accept_mask[b, pos]),
                    "hf_argmax": int(hf_step.argmax[b, pos]),
                    "tt_argmax": int(tt_step.argmax[b, pos]),
                    "hf_entropy": float(hf_step.entropy[b, pos]),
                    "tt_entropy": float(tt_step.entropy[b, pos]),
                }
            )
        per_step.append(step_summary)
    return {"per_step_diffs": per_step}


def _compare_summary(prompt: str, seed: int, hf_traj, tt_traj, *, eos_token_id: int | None) -> tuple[object, dict]:
    comparison = compare_trajectories(
        hf_traj,
        tt_traj,
        min_argmax_agreement=0.0,
        min_sampled_agreement=0.0,
        min_accept_iou=0.0,
        min_canvas_agreement=0.0,
        min_per_step_entropy_pcc=0.0,
        max_entropy_abs_err_threshold=10.0,
        committed_match_threshold=0.0,
        entropy_pcc_threshold=0.0,
    )
    summary = {
        "prompt": prompt,
        "seed": seed,
        **_trajectory_summary("hf", hf_traj, eos_token_id=eos_token_id),
        **_trajectory_summary("tt", tt_traj, eos_token_id=eos_token_id),
        "committed_match": comparison.committed_match,
        "per_step_argmax_agreement": comparison.per_step_argmax_agreement,
        "per_step_sampled_agreement": comparison.per_step_sampled_agreement,
        "per_step_accept_iou": comparison.per_step_accept_iou,
        "per_step_canvas_agreement": comparison.per_step_canvas_agreement,
        "per_step_entropy_pcc": comparison.per_step_entropy_pcc,
        "per_step_entropy_max_abs": comparison.per_step_entropy_max_abs,
        **_decision_diff_summary(hf_traj, tt_traj),
    }
    return comparison, summary


def _load_hf_model(checkpoint: str | Path, *, local_files_only: bool, dtype=torch.bfloat16):
    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).eval()
    return tokenizer, model


def _hf_text_vocab_size(model, tokenizer) -> int:
    text_config = getattr(model.config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    return int(len(tokenizer))


def _run_hf_reference(
    model,
    tokenizer,
    prompt: str,
    host_canvas: torch.Tensor,
    config: DiffusionConfig,
    *,
    capture_positions: list[int] | None = None,
    capture_top_k: int = 8,
    capture_layer_hidden: bool = False,
    capture_prompt_kv: bool = False,
    capture_routing: bool = False,
    capture_hidden_injection_layer: int | None = None,
    capture_attention_injection_layer: int | None = None,
    capture_all_layer_inputs: bool = False,
    capture_all_encoder_layer_inputs: bool = False,
    capture_residual_branch_layer: int | None = None,
    capture_self_conditioning_signals: bool = False,
    gumbel_noise: list[torch.Tensor],
    renoise_tokens: list[torch.Tensor],
):
    prompt_tokens = tokenize_prompt(tokenizer, prompt)
    prompt_tokens = _pad_prompt_tokens_for_hf_prefill(prompt_tokens)
    cache_len = prompt_tokens.shape[1]
    canvas_len = host_canvas.shape[1]
    position_ids = torch.arange(cache_len, dtype=torch.int64).unsqueeze(0)
    decoder_position_ids = torch.arange(cache_len, cache_len + canvas_len, dtype=torch.int64).unsqueeze(0)
    vocab_size = _hf_text_vocab_size(model, tokenizer)
    layer_hidden = []
    attention_hidden = []
    routing = []
    hidden_injection_inputs = []
    attention_injection_outputs = []
    all_layer_inputs = []
    all_encoder_layer_inputs = []
    residual_branch_outputs = {"post_attn": [], "shared_ff": [], "expert_ff": [], "post_ff": []}
    self_conditioning_signals = []
    layer_hooks = []
    if capture_self_conditioning_signals:

        def capture_self_conditioning_signal(_module, inputs):
            self_conditioning_signals.append(inputs[1].detach().to(torch.bfloat16).cpu())

        layer_hooks.append(
            model.model.decoder.self_conditioning.register_forward_pre_hook(capture_self_conditioning_signal)
        )
    if capture_layer_hidden:
        for layer in model.model.decoder.layers:

            def capture_layer(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                layer_hidden.append(hidden[:, capture_positions, :].detach().float().cpu())

            layer_hooks.append(layer.register_forward_hook(capture_layer))

            def capture_attention(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                attention_hidden.append(hidden[:, capture_positions, :].detach().float().cpu())

            layer_hooks.append(layer.self_attn.register_forward_hook(capture_attention))
    if capture_routing:
        for layer in model.model.decoder.layers:

            def capture_router(module, _inputs, output):
                _, top_k_weights, top_k_indices = output
                dense = torch.zeros(
                    (top_k_indices.shape[0], module.config.num_experts),
                    dtype=top_k_weights.dtype,
                    device=top_k_weights.device,
                )
                dense.scatter_(dim=-1, index=top_k_indices, src=top_k_weights)
                routing.append(dense.reshape(1, 1, canvas_len, module.config.num_experts).detach().cpu())

            layer_hooks.append(layer.router.register_forward_hook(capture_router))
    if capture_hidden_injection_layer is not None:
        if capture_hidden_injection_layer < 0:
            raise ValueError("hidden injection layer must be non-negative")
        if capture_hidden_injection_layer == 0:

            def capture_layer_zero_input(_module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None:
                    hidden = args[0]
                hidden_injection_inputs.append(hidden.detach().to(torch.bfloat16).cpu())

            layer_hooks.append(
                model.model.decoder.layers[0].register_forward_pre_hook(capture_layer_zero_input, with_kwargs=True)
            )
        else:

            def capture_injection_input(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                hidden_injection_inputs.append(hidden.detach().to(torch.bfloat16).cpu())

            layer_hooks.append(
                model.model.decoder.layers[capture_hidden_injection_layer - 1].register_forward_hook(
                    capture_injection_input
                )
            )
    if capture_attention_injection_layer is not None:

        def capture_attention_injection(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            attention_injection_outputs.append(hidden.detach().to(torch.bfloat16).cpu())

        layer_hooks.append(
            model.model.decoder.layers[capture_attention_injection_layer].self_attn.register_forward_hook(
                capture_attention_injection
            )
        )
    if capture_all_layer_inputs:
        for layer in model.model.decoder.layers:

            def capture_layer_input(_module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None:
                    hidden = args[0]
                all_layer_inputs.append(hidden.detach().to(torch.bfloat16).cpu())

            layer_hooks.append(layer.register_forward_pre_hook(capture_layer_input, with_kwargs=True))
    if capture_all_encoder_layer_inputs:
        for layer in model.model.encoder.language_model.layers:

            def capture_encoder_layer_input(_module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None:
                    hidden = args[0]
                all_encoder_layer_inputs.append(hidden.detach().to(torch.bfloat16).cpu())

            layer_hooks.append(layer.register_forward_pre_hook(capture_encoder_layer_input, with_kwargs=True))
    if capture_residual_branch_layer is not None:
        layer = model.model.decoder.layers[capture_residual_branch_layer]

        def capture_post_attn(_module, _inputs, output):
            residual_branch_outputs["post_attn"].append(output.detach().to(torch.bfloat16).cpu())

        def capture_post_ff(_module, _inputs, output):
            residual_branch_outputs["post_ff"].append(output.detach().to(torch.bfloat16).cpu())

        layer_hooks.append(layer.post_attention_layernorm.register_forward_hook(capture_post_attn))
        layer_hooks.append(
            layer.post_feedforward_layernorm_1.register_forward_hook(
                lambda _module, _inputs, output: residual_branch_outputs["shared_ff"].append(
                    output.detach().to(torch.bfloat16).cpu()
                )
            )
        )
        layer_hooks.append(
            layer.post_feedforward_layernorm_2.register_forward_hook(
                lambda _module, _inputs, output: residual_branch_outputs["expert_ff"].append(
                    output.detach().to(torch.bfloat16).cpu()
                )
            )
        )
        layer_hooks.append(layer.post_feedforward_layernorm.register_forward_hook(capture_post_ff))

    class HfLogits:
        def __init__(self) -> None:
            self.prev_raw = None
            self.prev_step = None
            self.captures = []
            self.prompt_kv = None

        def __call__(self, canvas, step):
            prev_sc = None
            if self.prev_raw is not None and os.environ.get("DG_DISABLE_SELF_CONDITIONING", "0") != "1":
                temperature = S.temperature_at_step(
                    self.prev_step,
                    config.max_denoise_steps,
                    config.temperature_start,
                    config.temperature_end,
                )
                prev_sc = (self.prev_raw / temperature).to(torch.bfloat16)
            with torch.no_grad():
                out = model(
                    input_ids=prompt_tokens,
                    position_ids=position_ids,
                    decoder_input_ids=canvas,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=prev_sc,
                )
            logits = out.logits.float().cpu()
            if capture_prompt_kv and self.prompt_kv is None:
                self.prompt_kv = [
                    (layer.keys.detach().cpu(), layer.values.detach().cpu()) for layer in out.past_key_values.layers
                ]
            if capture_positions:
                self.captures.append(_logits_topk_summary(logits, capture_positions, k=capture_top_k, step=step))
            self.prev_raw = logits
            self.prev_step = step
            return logits

    logits_fn = HfLogits()
    try:
        trajectory = denoise_block(
            logits_fn,
            host_canvas,
            config,
            vocab_size,
            sampler=S.SAMPLER_GUMBEL,
            gumbel_noise_fn=lambda step: gumbel_noise[step],
            noise_tokens_fn=lambda step: renoise_tokens[step],
        )
    finally:
        for hook in layer_hooks:
            hook.remove()
    return (
        prompt_tokens,
        trajectory,
        vocab_size,
        logits_fn.captures,
        layer_hidden,
        attention_hidden,
        routing,
        hidden_injection_inputs,
        attention_injection_outputs,
        all_layer_inputs,
        all_encoder_layer_inputs,
        residual_branch_outputs,
        self_conditioning_signals,
        logits_fn.prompt_kv,
    )


def _run_tt_replay(
    args,
    prompt: str,
    host_canvas: torch.Tensor,
    config: DiffusionConfig,
    vocab_size: int,
    *,
    capture_positions: list[int] | None = None,
    capture_top_k: int = 8,
    capture_layer_hidden: bool = False,
    hf_prompt_kv=None,
    hf_routing=None,
    hf_router_modules=None,
    hf_router_use_tt_norm: bool = False,
    hf_router_norm_only: bool = False,
    hf_router_live_layers: set[int] | None = None,
    hf_router_live_steps: set[int] | None = None,
    routing_layers: set[int] | None = None,
    hidden_injection_layer: int | None = None,
    hf_hidden_injection_inputs=None,
    hidden_injection_steps: set[int] | None = None,
    attention_injection_layer: int | None = None,
    hf_attention_injection_outputs=None,
    hf_all_layer_inputs=None,
    hf_all_encoder_layer_inputs=None,
    residual_branch_layer: int | None = None,
    residual_branches: set[str] | None = None,
    hf_residual_branch_outputs=None,
    capture_residual_branch_outputs: bool = False,
    hf_self_conditioning_signals=None,
    hf_self_conditioning_embedding_weight=None,
    hf_self_conditioning_embed_scale=None,
    hf_live_post_attn_norm=None,
    hf_live_post_attn_branch=None,
    capture_tp_partials_layer: int | None = None,
    gumbel_noise: list[torch.Tensor],
    renoise_tokens: list[torch.Tensor],
):
    import ttnn

    from models.experimental.diffusion_gemma.tt import denoise_forward as denoise_forward_module
    from models.experimental.diffusion_gemma.tt import diffusion_attention as diffusion_attention_module

    DenoiseLogitsAdapter = denoise_forward_module.DenoiseLogitsAdapter

    replay_gumbel = [gumbel_noise]
    replay_renoise = [renoise_tokens]
    captures = []
    layer_hidden = []
    attention_hidden = []
    final_hidden = []

    original_call = DenoiseLogitsAdapter.__call__
    original_layer_forward = denoise_forward_module._denoise_layer_forward
    original_denoise_attention = denoise_forward_module.denoise_attention
    original_prefix_reader_call = denoise_forward_module.MutablePrefixKVReader.__call__
    original_router_forward = denoise_forward_module._denoise_router_forward
    original_chunked_norm_forward = denoise_forward_module._chunked_norm_forward
    original_attention_allreduce = diffusion_attention_module.apply_allreduce
    injected_prompt_kv = []
    prompt_kv_summary = []
    router_call_index = 0
    hidden_injection_index = 0
    attention_call_index = 0
    attention_injection_index = 0
    all_layer_input_index = 0
    residual_branch_indices = {"post_attn": 0, "post_ff": 0}
    tt_residual_branch_outputs = {"post_attn": [], "shared_ff": [], "expert_ff": [], "post_ff": []}
    residual_norm_map = {}
    tp_allreduce_call_index = 0
    tp_partial_captures = []
    live_residual_layer_inputs = []
    current_denoise_step = -1

    def capturing_call(adapter, canvas_tokens, step):
        nonlocal current_denoise_step
        current_denoise_step = step
        original_condition = None
        injected_signal_host = None
        if hf_self_conditioning_signals is not None and step > 0:
            injected_signal_host = hf_self_conditioning_signals[step]
        elif hf_self_conditioning_embedding_weight is not None and step > 0:
            previous_logits_shard = ttnn.get_device_tensors(adapter.prev_logits)[0]
            previous_logits_host = ttnn.to_torch(previous_logits_shard).squeeze(1)
            temperature = adapter._temperature_at_step(step - 1)
            processed_logits = (previous_logits_host / temperature).to(torch.bfloat16)
            probabilities = processed_logits.softmax(dim=-1, dtype=torch.float32).to(
                hf_self_conditioning_embedding_weight.dtype
            )
            injected_signal_host = (
                torch.matmul(probabilities, hf_self_conditioning_embedding_weight) * hf_self_conditioning_embed_scale
            ).to(torch.bfloat16)
        if injected_signal_host is not None:
            original_condition = adapter.self_conditioning.condition

            def injected_condition(module, inputs_embeds, *_args, **_kwargs):
                signal = ttnn.from_torch(
                    injected_signal_host.unsqueeze(1),
                    device=inputs_embeds.device(),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(inputs_embeds.device()),
                )
                conditioned = module.forward(inputs_embeds, signal)
                signal.deallocate(True)
                return conditioned

            adapter.self_conditioning.condition = MethodType(injected_condition, adapter.self_conditioning)
        try:
            logits = original_call(adapter, canvas_tokens, step)
        finally:
            if original_condition is not None:
                adapter.self_conditioning.condition = original_condition
        if capture_positions:
            tile_start = min(capture_positions) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            tile_end = (max(capture_positions) + 1 + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            logits_slice = (
                logits
                if tile_start == 0 and tile_end == logits.shape[2]
                else ttnn.slice(
                    logits,
                    [0, 0, tile_start, 0],
                    [1, 1, tile_end, logits.shape[-1]],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            logits_shard = ttnn.get_device_tensors(logits_slice)[0]
            host_logits = ttnn.to_torch(logits_shard).float().cpu()
            if logits_slice is not logits:
                logits_slice.deallocate(True)
            local_positions = [pos - tile_start for pos in capture_positions]
            capture = _logits_topk_summary(host_logits, local_positions, k=capture_top_k, step=step)
            for row, absolute_pos in zip(capture["rows"], capture_positions):
                row["pos"] = absolute_pos
            captures.append(capture)
        return logits

    def capturing_layer_forward(*layer_args, **layer_kwargs):
        nonlocal hidden_injection_index, all_layer_input_index
        mutable_args = list(layer_args)
        layer_idx = mutable_args[1]
        if hf_live_post_attn_branch is not None and layer_idx == residual_branch_layer:
            hidden_shard = ttnn.get_device_tensors(mutable_args[2])[0]
            live_residual_layer_inputs.append(ttnn.to_torch(hidden_shard).squeeze(1).to(torch.bfloat16).cpu())
        if hf_all_layer_inputs is not None:
            original_hidden = mutable_args[2]
            mutable_args[2] = ttnn.from_torch(
                hf_all_layer_inputs[all_layer_input_index].unsqueeze(1),
                device=original_hidden.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(original_hidden.device()),
            )
            original_hidden.deallocate(True)
            all_layer_input_index += 1
        elif (
            hidden_injection_layer is not None
            and layer_idx == hidden_injection_layer
            and (hidden_injection_steps is None or current_denoise_step in hidden_injection_steps)
        ):
            original_hidden = mutable_args[2]
            mutable_args[2] = ttnn.from_torch(
                hf_hidden_injection_inputs[current_denoise_step].unsqueeze(1),
                device=original_hidden.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(original_hidden.device()),
            )
            original_hidden.deallocate(True)
            hidden_injection_index += 1
        output = original_layer_forward(*mutable_args, **layer_kwargs)
        if capture_layer_hidden:
            tile_start = min(capture_positions) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            tile_end = (max(capture_positions) + 1 + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            hidden_slice = (
                output
                if tile_start == 0 and tile_end == output.shape[2]
                else ttnn.slice(
                    output,
                    [0, 0, tile_start, 0],
                    [1, 1, tile_end, output.shape[-1]],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            hidden_shard = ttnn.get_device_tensors(hidden_slice)[0]
            host_hidden = ttnn.to_torch(hidden_shard).float().cpu().squeeze(1)
            if hidden_slice is not output:
                hidden_slice.deallocate(True)
            local_positions = [pos - tile_start for pos in capture_positions]
            layer_hidden.append(host_hidden[:, local_positions, :])
        return output

    def capturing_denoise_attention(*attention_args, **attention_kwargs):
        nonlocal attention_call_index, attention_injection_index
        output = original_denoise_attention(*attention_args, **attention_kwargs)
        layer_idx = attention_call_index % 30
        if attention_injection_layer is not None and layer_idx == attention_injection_layer:
            injected = ttnn.from_torch(
                hf_attention_injection_outputs[attention_injection_index].unsqueeze(1),
                device=output.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(output.device()),
            )
            output.deallocate(True)
            output = injected
            attention_injection_index += 1
        attention_call_index += 1
        if capture_layer_hidden:
            tile_start = min(capture_positions) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            tile_end = (max(capture_positions) + 1 + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            hidden_slice = (
                output
                if tile_start == 0 and tile_end == output.shape[2]
                else ttnn.slice(
                    output,
                    [0, 0, tile_start, 0],
                    [1, 1, tile_end, output.shape[-1]],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            hidden_shard = ttnn.get_device_tensors(hidden_slice)[0]
            host_hidden = ttnn.to_torch(hidden_shard).float().cpu().squeeze(1)
            if hidden_slice is not output:
                hidden_slice.deallocate(True)
            local_positions = [pos - tile_start for pos in capture_positions]
            attention_hidden.append(host_hidden[:, local_positions, :])
        return output

    def injected_router_forward(router, hidden_states):
        nonlocal router_call_index
        if hf_router_modules is not None:
            layer_idx = router_call_index % 30
            step_idx = router_call_index // 30
            if (hf_router_live_layers is not None and layer_idx not in hf_router_live_layers) or (
                hf_router_live_steps is not None and step_idx not in hf_router_live_steps
            ):
                router_call_index += 1
                return original_router_forward(router, hidden_states)
            hf_router = hf_router_modules[layer_idx]
            if hf_router_norm_only:
                hidden_shard = ttnn.get_device_tensors(hidden_states)[0]
                hidden_host = ttnn.to_torch(hidden_shard).squeeze(1).to(torch.bfloat16)
                with torch.no_grad():
                    normed_host = hf_router.norm(hidden_host)
                normed = ttnn.from_torch(
                    normed_host.unsqueeze(1),
                    device=hidden_states.device(),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(hidden_states.device()),
                )
                scaled = ttnn.mul(normed, router.scale)
                normed.deallocate(True)
                scaled_root = ttnn.mul(scaled, router.scalar_root_size)
                scaled.deallocate(True)
                scores = ttnn.linear(scaled_root, router.proj_weight)
                scaled_root.deallocate(True)
                probabilities = ttnn.softmax(scores, dim=-1)
                scores.deallocate(True)
                top_weights, top_indices = ttnn.topk(probabilities, k=router.top_k, dim=-1)
                top_sum = ttnn.sum(top_weights, dim=-1, keepdim=True)
                normalized = ttnn.div(top_weights, top_sum)
                top_weights.deallocate(True)
                top_sum.deallocate(True)
                dense = ttnn.scatter(
                    ttnn.zeros_like(probabilities),
                    dim=-1,
                    index=top_indices,
                    src=normalized,
                )
                probabilities.deallocate(True)
                normalized.deallocate(True)
                top_indices.deallocate(True)
                if router.per_expert_scale is not None:
                    scaled_dense = ttnn.mul(dense, router.per_expert_scale)
                    dense.deallocate(True)
                    dense = scaled_dense
                router_call_index += 1
                return dense
            if hf_router_use_tt_norm:
                normed = denoise_forward_module._chunked_norm_forward(router.norm, hidden_states)
                hidden_shard = ttnn.get_device_tensors(normed)[0]
                hidden_host = ttnn.to_torch(hidden_shard).squeeze(1).to(torch.bfloat16)
                normed.deallocate(True)
            else:
                hidden_shard = ttnn.get_device_tensors(hidden_states)[0]
                hidden_host = ttnn.to_torch(hidden_shard).squeeze(1).to(torch.bfloat16)
            with torch.no_grad():
                if hf_router_use_tt_norm:
                    scaled = hidden_host * hf_router.scale * hf_router.scalar_root_size
                    scores = hf_router.proj(scaled)
                    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
                    top_weights, top_indices = torch.topk(probabilities, k=hf_router.config.top_k_experts, dim=-1)
                    top_weights /= top_weights.sum(dim=-1, keepdim=True)
                    top_weights *= hf_router.per_expert_scale[top_indices]
                else:
                    _probabilities, top_weights, top_indices = hf_router(hidden_host)
            dense_host = torch.zeros(
                (*top_weights.shape[:-1], hf_router.config.num_experts),
                dtype=torch.float32,
            )
            dense_host.scatter_(-1, top_indices.cpu(), top_weights.float().cpu())
            dense = ttnn.from_torch(
                dense_host.to(torch.bfloat16).unsqueeze(1),
                device=hidden_states.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(hidden_states.device()),
            )
            router_call_index += 1
            return dense
        if router_call_index >= len(hf_routing):
            return original_router_forward(router, hidden_states)
        layer_idx = router_call_index % 30
        use_oracle = routing_layers is not None and layer_idx in routing_layers
        if use_oracle:
            dense = ttnn.from_torch(
                hf_routing[router_call_index],
                device=hidden_states.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(hidden_states.device()),
            )
        else:
            dense = original_router_forward(router, hidden_states)
        router_call_index += 1
        return dense

    def capturing_chunked_norm_forward(norm, hidden_states, *, chunk_size=32):
        output = original_chunked_norm_forward(norm, hidden_states, chunk_size=chunk_size)
        branch = residual_norm_map.get(id(norm))
        if branch is not None and capture_residual_branch_outputs:
            output_shard = ttnn.get_device_tensors(output)[0]
            tt_residual_branch_outputs[branch].append(ttnn.to_torch(output_shard).squeeze(1).to(torch.bfloat16).cpu())
        if branch == "post_attn" and hf_live_post_attn_branch is not None and live_residual_layer_inputs:
            with torch.no_grad():
                normalized_host = hf_live_post_attn_branch(live_residual_layer_inputs.pop(0))
            injected = ttnn.from_torch(
                normalized_host.unsqueeze(1),
                device=output.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(output.device()),
            )
            output.deallocate(True)
            return injected
        if branch == "post_attn" and hf_live_post_attn_norm is not None:
            hidden_shard = ttnn.get_device_tensors(hidden_states)[0]
            hidden_host = ttnn.to_torch(hidden_shard).squeeze(1).to(torch.bfloat16)
            with torch.no_grad():
                normalized_host = hf_live_post_attn_norm(hidden_host)
            injected = ttnn.from_torch(
                normalized_host.unsqueeze(1),
                device=output.device(),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(output.device()),
            )
            output.deallocate(True)
            return injected
        if branch is None or residual_branches is None or branch not in residual_branches:
            return output
        index = residual_branch_indices[branch]
        if index >= len(hf_residual_branch_outputs[branch]):
            return output
        injected = ttnn.from_torch(
            hf_residual_branch_outputs[branch][index].unsqueeze(1),
            device=output.device(),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(output.device()),
        )
        output.deallocate(True)
        residual_branch_indices[branch] += 1
        return injected

    def capturing_attention_allreduce(tensor, mesh_config, ccl_manager, hidden_size):
        nonlocal tp_allreduce_call_index
        layer_idx = tp_allreduce_call_index % 30
        partials = None
        if capture_tp_partials_layer is not None and layer_idx == capture_tp_partials_layer:
            partials = [ttnn.to_torch(shard).to(torch.bfloat16).cpu() for shard in ttnn.get_device_tensors(tensor)]
        output = original_attention_allreduce(tensor, mesh_config, ccl_manager, hidden_size)
        if partials is not None:
            output_shard = ttnn.get_device_tensors(output)[0]
            tp_partial_captures.append(
                {
                    "layer": layer_idx,
                    "partials": partials,
                    "output": ttnn.to_torch(output_shard).to(torch.bfloat16).cpu(),
                }
            )
        tp_allreduce_call_index += 1
        return output

    checkpoint_inputs = load_checkpoint_inputs(
        args.checkpoint,
        tokenizer_kwargs={"local_files_only": args.local_files_only, "trust_remote_code": True},
        state_prefixes=text_generation_prefixes_for_layers(args.num_layers),
        device="cpu",
    )
    mesh_device = _open_mesh_device(args.mesh)
    prefill_layer_type = None
    original_prefill_layer_call = None
    try:
        DenoiseLogitsAdapter.__call__ = capturing_call
        denoise_forward_module._denoise_layer_forward = capturing_layer_forward
        denoise_forward_module.denoise_attention = capturing_denoise_attention
        denoise_forward_module._chunked_norm_forward = capturing_chunked_norm_forward
        diffusion_attention_module.apply_allreduce = capturing_attention_allreduce
        if hf_routing is not None or hf_router_modules is not None:
            denoise_forward_module._denoise_router_forward = injected_router_forward
        _log_mesh_dram(mesh_device, "replay-baseline")
        checkpoint_model_inputs = build_tt_model_from_checkpoint_inputs(
            mesh_device,
            checkpoint_inputs,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            bounded_sliding_kv_cache=args.bounded_sliding_kv_cache,
        )
        tt_model = checkpoint_model_inputs.tt_model
        if residual_branch_layer is not None:
            target_layer = tt_model.layers[residual_branch_layer]
            residual_norm_map[id(target_layer.post_attention_layernorm)] = "post_attn"
            residual_norm_map[id(target_layer.post_feedforward_layernorm_1)] = "shared_ff"
            residual_norm_map[id(target_layer.post_feedforward_layernorm_2)] = "expert_ff"
            residual_norm_map[id(target_layer.post_feedforward_layernorm)] = "post_ff"
        if hf_all_encoder_layer_inputs is not None:
            prefill_layer_type = type(tt_model.layers[0])
            original_prefill_layer_call = prefill_layer_type.__call__
            prefill_layer_indices = {id(layer): idx for idx, layer in enumerate(tt_model.layers)}

            def teacher_forced_prefill_layer_call(layer, hidden_states, *layer_args, **layer_kwargs):
                layer_idx = prefill_layer_indices[id(layer)]
                if not layer_kwargs.get("is_decode", False):
                    injected = ttnn.from_torch(
                        hf_all_encoder_layer_inputs[layer_idx].unsqueeze(1),
                        device=hidden_states.device(),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(hidden_states.device()),
                    )
                    hidden_states.deallocate(True)
                    hidden_states = injected
                return original_prefill_layer_call(layer, hidden_states, *layer_args, **layer_kwargs)

            prefill_layer_type.__call__ = teacher_forced_prefill_layer_call
        if hf_prompt_kv is not None:
            tp = int(mesh_device.shape[1])
            mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(1, tp), dims=(None, 1))
            for key, value in hf_prompt_kv:
                if key.shape[1] < tp:
                    repeats = tp // key.shape[1]
                    key = key.repeat_interleave(repeats, dim=1)
                    value = value.repeat_interleave(repeats, dim=1)
                injected_prompt_kv.append(
                    (
                        ttnn.from_torch(
                            key,
                            device=mesh_device,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            mesh_mapper=mapper,
                        ),
                        ttnn.from_torch(
                            value,
                            device=mesh_device,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            mesh_mapper=mapper,
                        ),
                    )
                )

            def injected_prefix_reader(_reader, layer_idx):
                key, value = injected_prompt_kv[layer_idx]
                if layer_idx >= len(prompt_kv_summary):
                    tt_key, tt_value = original_prefix_reader_call(_reader, layer_idx)
                    layer_summary = {"layer": layer_idx}
                    for name, tt_tensor, hf_tensor in (
                        ("key", tt_key, key),
                        ("value", tt_value, value),
                    ):
                        per_device = []
                        for device_idx, (tt_shard, hf_shard) in enumerate(
                            zip(ttnn.get_device_tensors(tt_tensor), ttnn.get_device_tensors(hf_tensor))
                        ):
                            tt_host = ttnn.to_torch(tt_shard).float().cpu()
                            hf_host = ttnn.to_torch(hf_shard).float().cpu()
                            per_device.append(
                                {
                                    "device": device_idx,
                                    "pcc": _pcc(tt_host, hf_host),
                                    "max_abs": float((tt_host - hf_host).abs().max()),
                                }
                            )
                        layer_summary[name] = per_device
                    prompt_kv_summary.append(layer_summary)
                    tt_key.deallocate(True)
                    tt_value.deallocate(True)
                return (
                    ttnn.clone(key, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                )

            denoise_forward_module.MutablePrefixKVReader.__call__ = injected_prefix_reader
        original_lm_head = tt_model._apply_lm_head

        def capturing_lm_head(_model, hidden_states, *lm_args, **lm_kwargs):
            if capture_positions and hidden_states.shape[-2] == config.canvas_length:
                tile_start = min(capture_positions) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
                tile_end = (max(capture_positions) + 1 + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
                hidden_slice = (
                    hidden_states
                    if tile_start == 0 and tile_end == hidden_states.shape[2]
                    else ttnn.slice(
                        hidden_states,
                        [0, 0, tile_start, 0],
                        [1, 1, tile_end, hidden_states.shape[-1]],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                hidden_shard = ttnn.get_device_tensors(hidden_slice)[0]
                host_hidden = ttnn.to_torch(hidden_shard).float().cpu().squeeze(1)
                if hidden_slice is not hidden_states:
                    hidden_slice.deallocate(True)
                local_positions = [pos - tile_start for pos in capture_positions]
                final_hidden.append(host_hidden[:, local_positions, :])
            return original_lm_head(hidden_states, *lm_args, **lm_kwargs)

        tt_model._apply_lm_head = MethodType(capturing_lm_head, tt_model)
        _log_mesh_dram(mesh_device, "replay-post-build")
        try:
            generation = generate_text_from_checkpoint_model_inputs(
                checkpoint_model_inputs,
                prompt,
                num_blocks=1,
                config=config,
                init_canvas_fn=make_host_canvas_init_fn(mesh_device, [host_canvas]),
                gumbel_noise_fn=make_host_gumbel_noise_fn(mesh_device, replay_gumbel),
                noise_tokens_fn=make_host_noise_tokens_fn(mesh_device, replay_renoise),
                max_new_tokens=config.canvas_length,
                eos_token_id=None,
                stop_token_ids=None,
            )
        finally:
            tt_model._apply_lm_head = original_lm_head
        return (
            generation.generation.trajectories[0],
            captures,
            layer_hidden,
            attention_hidden,
            final_hidden,
            prompt_kv_summary,
            tp_partial_captures,
            tt_residual_branch_outputs,
        )
    finally:
        DenoiseLogitsAdapter.__call__ = original_call
        denoise_forward_module._denoise_layer_forward = original_layer_forward
        denoise_forward_module.denoise_attention = original_denoise_attention
        denoise_forward_module.MutablePrefixKVReader.__call__ = original_prefix_reader_call
        denoise_forward_module._denoise_router_forward = original_router_forward
        denoise_forward_module._chunked_norm_forward = original_chunked_norm_forward
        diffusion_attention_module.apply_allreduce = original_attention_allreduce
        if prefill_layer_type is not None:
            prefill_layer_type.__call__ = original_prefill_layer_call
        for key, value in injected_prompt_kv:
            key.deallocate(True)
            value.deallocate(True)
        _close_mesh_device(mesh_device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one DiffusionGemma block through HF and TT.")
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("DG_CKPT", "google/diffusiongemma-26B-A4B-it"),
        help="DiffusionGemma checkpoint directory or model id for TT weights/tokenizer.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=os.getenv("DG_HF_CKPT"),
        help="Optional separate HF checkpoint path/model id; defaults to --checkpoint.",
    )
    parser.add_argument("--prompt", default=os.getenv("DG_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--seed", type=int, default=1, help="Seed for the initial host canvas.")
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--max-denoising-steps", type=int, default=1)
    parser.add_argument("--entropy-stop-threshold", type=float, default=-1.0)
    parser.add_argument("--stable-steps-to-halt", type=int, default=1)
    parser.add_argument("--mesh", default=os.getenv("MESH_DEVICE", "P150x4"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bounded-sliding-kv-cache", action="store_true")
    parser.add_argument("--hf-only", action="store_true", help="Save only the HF reference trajectory.")
    parser.add_argument(
        "--hf-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="HF backbone compute dtype. bfloat16 is the production reference; float32 gives the "
        "quantization-ideal trajectory for the #48291 bf16-floor self-consistency control "
        "(see doc/decision_fidelity/measure_bf16_floor.py). Forbidden under --stage-gate.",
    )
    parser.add_argument(
        "--capture-logits-topk",
        action="store_true",
        help="Capture top logits at --capture-positions for #48291 localization.",
    )
    parser.add_argument("--capture-positions", default="2,3,4")
    parser.add_argument("--capture-top-k", type=int, default=8)
    parser.add_argument("--noise-mode", choices=("zero", "seeded"), default="zero")
    parser.add_argument(
        "--stage-gate",
        action="store_true",
        help="Enforce the seeded #48291 decision-fidelity gate and return nonzero on failure.",
    )
    parser.add_argument(
        "--capture-layer-hidden",
        action="store_true",
        help="Capture failing-position hidden states after every decoder layer.",
    )
    parser.add_argument(
        "--inject-hf-prompt-kv",
        action="store_true",
        help="Diagnostic control: replace TT frozen prompt KV reads with the exact HF encoder cache.",
    )
    parser.add_argument(
        "--inject-hf-routing-layers",
        help="Diagnostic control: comma-separated decoder layers (or 'all') using exact HF dense routing.",
    )
    parser.add_argument(
        "--inject-hf-router-on-tt-hidden",
        action="store_true",
        help="Diagnostic control: run each HF router on its live TT hidden state and upload the routing.",
    )
    parser.add_argument(
        "--inject-hf-router-on-tt-hidden-layers",
        help="Diagnostic control: comma-separated layers whose HF router runs on the live TT hidden state.",
    )
    parser.add_argument(
        "--inject-hf-router-on-tt-hidden-steps",
        help="Diagnostic control: comma-separated denoise steps on which the HF router handles live TT hidden states.",
    )
    parser.add_argument(
        "--inject-hf-router-tail-on-tt-norm",
        action="store_true",
        help="Diagnostic control: use TT router RMSNorm, then run the HF projection/softmax/top-k tail.",
    )
    parser.add_argument(
        "--inject-hf-router-norm-on-tt-hidden",
        action="store_true",
        help="Diagnostic control: run only HF router RMSNorm, then use the normal TT router tail.",
    )
    parser.add_argument(
        "--inject-hf-hidden-layer",
        type=int,
        help="Diagnostic control: replace this TT layer's input with the exact HF hidden state.",
    )
    parser.add_argument(
        "--inject-hf-hidden-steps",
        help="Optional comma-separated denoise steps for --inject-hf-hidden-layer.",
    )
    parser.add_argument(
        "--inject-hf-attention-layer",
        type=int,
        help="Diagnostic control: replace this TT layer's raw attention output with the exact HF output.",
    )
    parser.add_argument(
        "--teacher-force-all-layer-inputs",
        action="store_true",
        help="Diagnostic control: execute every TT denoise layer from its exact HF input hidden state.",
    )
    parser.add_argument(
        "--teacher-force-prefill-layer-inputs",
        action="store_true",
        help="Diagnostic control: execute every TT causal-prefill layer from its exact HF encoder input.",
    )
    parser.add_argument(
        "--inject-hf-residual-branch-layer",
        type=int,
        help="Diagnostic control: decoder layer whose normalized residual branch is replaced from HF.",
    )
    parser.add_argument(
        "--capture-residual-branch-layer",
        type=int,
        help="Capture HF/TT post-attention and post-feedforward normalized branches for one layer.",
    )
    parser.add_argument(
        "--inject-hf-residual-branches",
        choices=("post_attn", "post_ff", "both"),
        help="Residual branch to replace at --inject-hf-residual-branch-layer.",
    )
    parser.add_argument(
        "--inject-hf-post-attn-norm-on-tt-layer",
        type=int,
        help="Diagnostic control: run this layer's HF post-attention RMSNorm on its live TT attention output.",
    )
    parser.add_argument(
        "--inject-hf-post-attn-branch-on-tt-layer",
        type=int,
        help="Diagnostic control: run HF attention plus post-attention norm on this layer's live TT input.",
    )
    parser.add_argument(
        "--capture-tp-partials-layer",
        type=int,
        help="Capture this denoise layer's per-device attention o_proj partials before TP all-reduce.",
    )
    parser.add_argument(
        "--inject-hf-self-conditioning-signal",
        action="store_true",
        help="Diagnostic control: replace each TT previous-logit soft embedding with the exact HF signal.",
    )
    parser.add_argument(
        "--inject-hf-self-conditioning-on-tt-logits",
        action="store_true",
        help="Diagnostic control: compute the exact HF soft embedding from each live TT previous-logit tensor.",
    )
    parser.add_argument("--output", default="/tmp/dg_replay_hf_tt_compare.pt")
    return parser


def _validate_stage_gate_args(args) -> None:
    if not args.stage_gate:
        return
    if args.hf_only:
        raise ValueError("--stage-gate requires both HF and TT trajectories")
    if args.noise_mode != "seeded":
        raise ValueError("--stage-gate requires --noise-mode seeded")
    if args.canvas_length != 256:
        raise ValueError("--stage-gate requires --canvas-length 256")
    if args.max_denoising_steps != 8:
        raise ValueError("--stage-gate requires --max-denoising-steps 8")
    if args.mesh != "P150x4":
        raise ValueError("--stage-gate requires --mesh P150x4")
    if args.num_layers is not None:
        raise ValueError("--stage-gate requires the full model (omit --num-layers)")
    if os.environ.get("DG_SPARSE_MOE", "0") != "1":
        raise ValueError("--stage-gate requires the production sparse path (DG_SPARSE_MOE=1)")
    if getattr(args, "hf_dtype", "bfloat16") != "bfloat16":
        raise ValueError("--stage-gate requires the bf16 HF reference (--hf-dtype bfloat16)")

    diagnostic_controls = {
        "--inject-hf-prompt-kv": args.inject_hf_prompt_kv,
        "--inject-hf-routing-layers": args.inject_hf_routing_layers,
        "--inject-hf-router-on-tt-hidden": args.inject_hf_router_on_tt_hidden,
        "--inject-hf-router-on-tt-hidden-layers": args.inject_hf_router_on_tt_hidden_layers,
        "--inject-hf-router-on-tt-hidden-steps": args.inject_hf_router_on_tt_hidden_steps,
        "--inject-hf-router-tail-on-tt-norm": args.inject_hf_router_tail_on_tt_norm,
        "--inject-hf-router-norm-on-tt-hidden": args.inject_hf_router_norm_on_tt_hidden,
        "--inject-hf-hidden-layer": args.inject_hf_hidden_layer,
        "--inject-hf-hidden-steps": args.inject_hf_hidden_steps,
        "--inject-hf-attention-layer": args.inject_hf_attention_layer,
        "--teacher-force-all-layer-inputs": args.teacher_force_all_layer_inputs,
        "--teacher-force-prefill-layer-inputs": args.teacher_force_prefill_layer_inputs,
        "--inject-hf-residual-branch-layer": args.inject_hf_residual_branch_layer,
        "--capture-residual-branch-layer": args.capture_residual_branch_layer,
        "--inject-hf-residual-branches": args.inject_hf_residual_branches,
        "--inject-hf-post-attn-norm-on-tt-layer": args.inject_hf_post_attn_norm_on_tt_layer,
        "--inject-hf-post-attn-branch-on-tt-layer": args.inject_hf_post_attn_branch_on_tt_layer,
        "--inject-hf-self-conditioning-signal": args.inject_hf_self_conditioning_signal,
        "--inject-hf-self-conditioning-on-tt-logits": args.inject_hf_self_conditioning_on_tt_logits,
    }
    enabled_controls = [name for name, value in diagnostic_controls.items() if value is not None and value is not False]
    if enabled_controls:
        raise ValueError(f"--stage-gate forbids diagnostic controls: {', '.join(enabled_controls)}")


def _stage_gate_active_step_indices(hf_traj, tt_traj) -> list[int]:
    indices = []
    for step_idx, (hf_step, tt_step) in enumerate(zip(hf_traj.per_step, tt_traj.per_step)):
        indices.append(step_idx)
        if bool(hf_step.accept_mask.all()) and bool(tt_step.accept_mask.all()):
            break
    if not indices:
        raise ValueError("stage gate requires at least one comparable denoise step")
    return indices


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.inject_hf_self_conditioning_signal and args.inject_hf_self_conditioning_on_tt_logits:
        raise ValueError("select only one HF self-conditioning signal control")
    _validate_stage_gate_args(args)
    config = _make_config(args)
    hf_checkpoint = args.hf_checkpoint or args.checkpoint

    hf_dtype = torch.float32 if getattr(args, "hf_dtype", "bfloat16") == "float32" else torch.bfloat16
    tokenizer, hf_model = _load_hf_model(hf_checkpoint, local_files_only=args.local_files_only, dtype=hf_dtype)
    vocab_size = _hf_text_vocab_size(hf_model, tokenizer)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    host_canvas = torch.randint(0, vocab_size, (1, config.canvas_length), dtype=torch.long, generator=generator)
    gumbel_noise, renoise_tokens = _make_replay_noise(
        seed=args.seed,
        steps=config.max_denoise_steps,
        canvas_length=config.canvas_length,
        vocab_size=vocab_size,
        mode=args.noise_mode,
    )
    capture_positions = (
        [int(pos.strip()) for pos in args.capture_positions.split(",") if pos.strip()]
        if args.capture_logits_topk or args.capture_layer_hidden
        else None
    )
    routing_layers = None
    if args.inject_hf_routing_layers:
        routing_layers = (
            set(range(30))
            if args.inject_hf_routing_layers == "all"
            else {int(layer.strip()) for layer in args.inject_hf_routing_layers.split(",") if layer.strip()}
        )
    router_oracle_count = sum(
        (
            args.inject_hf_router_on_tt_hidden,
            bool(args.inject_hf_router_on_tt_hidden_layers),
            args.inject_hf_router_tail_on_tt_norm,
            args.inject_hf_router_norm_on_tt_hidden,
        )
    )
    if router_oracle_count > 1:
        raise ValueError("select only one live-TT HF router oracle mode")
    if router_oracle_count and routing_layers is not None:
        raise ValueError("live-TT HF router controls cannot be combined with --inject-hf-routing-layers")
    hf_router_modules = [layer.router for layer in hf_model.model.decoder.layers] if router_oracle_count else None
    hf_router_live_layers = (
        {int(layer.strip()) for layer in args.inject_hf_router_on_tt_hidden_layers.split(",") if layer.strip()}
        if args.inject_hf_router_on_tt_hidden_layers
        else None
    )
    hf_router_live_steps = (
        {int(step.strip()) for step in args.inject_hf_router_on_tt_hidden_steps.split(",") if step.strip()}
        if args.inject_hf_router_on_tt_hidden_steps
        else None
    )
    hidden_injection_steps = (
        {int(step.strip()) for step in args.inject_hf_hidden_steps.split(",") if step.strip()}
        if args.inject_hf_hidden_steps
        else None
    )
    if args.inject_hf_hidden_steps and args.inject_hf_hidden_layer is None:
        raise ValueError("--inject-hf-hidden-steps requires --inject-hf-hidden-layer")
    if (args.inject_hf_residual_branch_layer is None) != (args.inject_hf_residual_branches is None):
        raise ValueError("residual branch layer and branch selection must be provided together")
    live_branch_options = sum(
        (
            args.inject_hf_post_attn_norm_on_tt_layer is not None,
            args.inject_hf_post_attn_branch_on_tt_layer is not None,
            args.inject_hf_residual_branch_layer is not None,
        )
    )
    if live_branch_options > 1:
        raise ValueError("select only one fixed/live post-attention branch oracle")
    residual_branch_layer = (
        args.inject_hf_post_attn_branch_on_tt_layer
        if args.inject_hf_post_attn_branch_on_tt_layer is not None
        else (
            args.inject_hf_post_attn_norm_on_tt_layer
            if args.inject_hf_post_attn_norm_on_tt_layer is not None
            else (
                args.inject_hf_residual_branch_layer
                if args.inject_hf_residual_branch_layer is not None
                else args.capture_residual_branch_layer
            )
        )
    )
    residual_branches = (
        {"post_attn", "post_ff"}
        if args.inject_hf_residual_branches == "both"
        else ({args.inject_hf_residual_branches} if args.inject_hf_residual_branches else None)
    )
    hf_live_post_attn_norm = (
        hf_model.model.decoder.layers[args.inject_hf_post_attn_norm_on_tt_layer].post_attention_layernorm
        if args.inject_hf_post_attn_norm_on_tt_layer is not None
        else None
    )
    hf_live_post_attn_branch = None
    capture_attention_layer = (
        args.inject_hf_attention_layer if args.inject_hf_attention_layer is not None else args.capture_tp_partials_layer
    )
    capture_residual_branch_layer = (
        args.capture_residual_branch_layer
        if args.capture_residual_branch_layer is not None
        else (
            args.inject_hf_residual_branch_layer
            if args.inject_hf_residual_branch_layer is not None
            else args.capture_tp_partials_layer
        )
    )

    (
        prompt_tokens,
        hf_traj,
        vocab_size,
        hf_logits_topk,
        hf_layer_hidden,
        hf_attention_hidden,
        hf_routing,
        hf_hidden_injection_inputs,
        hf_attention_injection_outputs,
        hf_all_layer_inputs,
        hf_all_encoder_layer_inputs,
        hf_residual_branch_outputs,
        hf_self_conditioning_signals,
        hf_prompt_kv,
    ) = _run_hf_reference(
        hf_model,
        tokenizer,
        args.prompt,
        host_canvas,
        config,
        capture_positions=capture_positions,
        capture_top_k=args.capture_top_k,
        capture_layer_hidden=args.capture_layer_hidden,
        capture_prompt_kv=args.inject_hf_prompt_kv,
        capture_routing=routing_layers is not None,
        capture_hidden_injection_layer=args.inject_hf_hidden_layer,
        capture_attention_injection_layer=capture_attention_layer,
        capture_all_layer_inputs=args.teacher_force_all_layer_inputs,
        capture_all_encoder_layer_inputs=args.teacher_force_prefill_layer_inputs,
        capture_residual_branch_layer=capture_residual_branch_layer,
        capture_self_conditioning_signals=args.inject_hf_self_conditioning_signal,
        gumbel_noise=gumbel_noise,
        renoise_tokens=renoise_tokens,
    )
    if args.inject_hf_post_attn_branch_on_tt_layer is not None:
        layer_idx = args.inject_hf_post_attn_branch_on_tt_layer
        hf_layer = hf_model.model.decoder.layers[layer_idx]
        decoder = hf_model.model.decoder
        layer_type = decoder.text_config.layer_types[layer_idx]
        decoder_position_ids = torch.arange(
            prompt_tokens.shape[1],
            prompt_tokens.shape[1] + config.canvas_length,
            dtype=torch.long,
        ).unsqueeze(0)
        prompt_cache = SimpleNamespace(layers=[SimpleNamespace(keys=key, values=value) for key, value in hf_prompt_kv])

        def hf_live_post_attn_branch(hidden_states):
            normed = hf_layer.input_layernorm(hidden_states)
            position_embeddings = decoder.rotary_emb(normed, decoder_position_ids, layer_type)
            attention_output, _ = hf_layer.self_attn(
                hidden_states=normed,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=decoder_position_ids,
                past_key_values=prompt_cache,
            )
            return hf_layer.post_attention_layernorm(attention_output)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    artifact = {
        "prompt": args.prompt,
        "seed": args.seed,
        "config": config,
        "prompt_tokens": prompt_tokens,
        "host_canvas": host_canvas,
        "hf_traj": hf_traj,
        "hf_logits_topk": hf_logits_topk,
        "hf_residual_branch_counts": {branch: len(outputs) for branch, outputs in hf_residual_branch_outputs.items()},
        "replay_input_hashes": {
            "initial_canvas": _tensor_sha256(host_canvas),
            "gumbel_noise": [_tensor_sha256(noise) for noise in gumbel_noise],
            "renoise_tokens": [_tensor_sha256(tokens) for tokens in renoise_tokens],
        },
    }
    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "hf_checkpoint": str(hf_checkpoint),
        "tt_checkpoint": str(args.checkpoint),
        "canvas_length": config.canvas_length,
        "max_denoising_steps": config.max_denoise_steps,
        **_trajectory_summary("hf", hf_traj, eos_token_id=eos_token_id),
    }

    if not args.hf_only:
        (
            tt_traj,
            tt_logits_topk,
            tt_layer_hidden,
            tt_attention_hidden,
            tt_final_hidden,
            prompt_kv_summary,
            tp_partial_captures,
            tt_residual_branch_outputs,
        ) = _run_tt_replay(
            args,
            args.prompt,
            host_canvas,
            config,
            vocab_size,
            capture_positions=capture_positions,
            capture_top_k=args.capture_top_k,
            capture_layer_hidden=args.capture_layer_hidden,
            hf_prompt_kv=hf_prompt_kv,
            hf_routing=hf_routing,
            hf_router_modules=hf_router_modules,
            hf_router_use_tt_norm=args.inject_hf_router_tail_on_tt_norm,
            hf_router_norm_only=args.inject_hf_router_norm_on_tt_hidden,
            hf_router_live_layers=hf_router_live_layers,
            hf_router_live_steps=hf_router_live_steps,
            routing_layers=routing_layers,
            hidden_injection_layer=args.inject_hf_hidden_layer,
            hf_hidden_injection_inputs=hf_hidden_injection_inputs,
            hidden_injection_steps=hidden_injection_steps,
            attention_injection_layer=args.inject_hf_attention_layer,
            hf_attention_injection_outputs=hf_attention_injection_outputs,
            hf_all_layer_inputs=hf_all_layer_inputs if args.teacher_force_all_layer_inputs else None,
            hf_all_encoder_layer_inputs=(
                hf_all_encoder_layer_inputs if args.teacher_force_prefill_layer_inputs else None
            ),
            residual_branch_layer=residual_branch_layer,
            residual_branches=residual_branches,
            hf_residual_branch_outputs=hf_residual_branch_outputs,
            capture_residual_branch_outputs=args.capture_residual_branch_layer is not None,
            hf_self_conditioning_signals=(
                hf_self_conditioning_signals if args.inject_hf_self_conditioning_signal else None
            ),
            hf_self_conditioning_embedding_weight=(
                hf_model.model.decoder.embed_tokens.weight if args.inject_hf_self_conditioning_on_tt_logits else None
            ),
            hf_self_conditioning_embed_scale=(
                hf_model.model.decoder.embed_tokens.embed_scale
                if args.inject_hf_self_conditioning_on_tt_logits
                else None
            ),
            hf_live_post_attn_norm=hf_live_post_attn_norm,
            hf_live_post_attn_branch=hf_live_post_attn_branch,
            capture_tp_partials_layer=args.capture_tp_partials_layer,
            gumbel_noise=gumbel_noise,
            renoise_tokens=renoise_tokens,
        )
        comparison, summary = _compare_summary(args.prompt, args.seed, hf_traj, tt_traj, eos_token_id=eos_token_id)
        hf_non_eos_mask = (
            hf_traj.committed != eos_token_id
            if eos_token_id is not None
            else torch.ones_like(hf_traj.committed, dtype=torch.bool)
        )
        hf_non_eos_count = int(hf_non_eos_mask.sum())
        summary["committed_match_on_hf_non_eos"] = (
            float((tt_traj.committed[hf_non_eos_mask] == hf_traj.committed[hf_non_eos_mask]).float().mean())
            if hf_non_eos_count
            else None
        )
        summary["noise_mode"] = args.noise_mode
        active_step_indices = (
            _stage_gate_active_step_indices(hf_traj, tt_traj) if hf_traj.per_step and tt_traj.per_step else []
        )
        active_entropy_pcc = [comparison.per_step_entropy_pcc[index] for index in active_step_indices]
        terminal_active_step = active_step_indices[-1] if active_step_indices else None
        stage_gate = {
            "enabled": args.stage_gate,
            "criteria": {
                "committed_match_strictly_greater_than": 0.95,
                "min_active_step_entropy_pcc_strictly_greater_than": 0.95,
                "terminal_active_accept_iou_strictly_greater_than": 0.95,
            },
            "observed": {
                "committed_match": comparison.committed_match,
                "active_step_indices": active_step_indices,
                "min_active_step_entropy_pcc": min(active_entropy_pcc) if active_entropy_pcc else None,
                "terminal_active_accept_iou": (
                    comparison.per_step_accept_iou[terminal_active_step] if terminal_active_step is not None else None
                ),
                "raw_min_per_step_entropy_pcc": (
                    min(comparison.per_step_entropy_pcc) if comparison.per_step_entropy_pcc else None
                ),
                "raw_final_accept_iou": (
                    comparison.per_step_accept_iou[-1] if comparison.per_step_accept_iou else None
                ),
            },
        }
        stage_gate["passed"] = (
            bool(active_step_indices)
            and comparison.committed_match > 0.95
            and min(active_entropy_pcc) > 0.95
            and comparison.per_step_accept_iou[terminal_active_step] > 0.95
        )
        summary["stage_gate"] = stage_gate
        layer_hidden_summary = (
            _layer_hidden_summary(hf_layer_hidden, tt_layer_hidden, capture_positions)
            if args.capture_layer_hidden
            else []
        )
        attention_hidden_summary = (
            _layer_hidden_summary(hf_attention_hidden, tt_attention_hidden, capture_positions)
            if args.capture_layer_hidden
            else []
        )
        residual_branch_summary = {}
        if args.capture_residual_branch_layer is not None:
            for branch in ("post_attn", "shared_ff", "expert_ff", "post_ff"):
                residual_branch_summary[branch] = []
                for hf_branch, tt_branch in zip(hf_residual_branch_outputs[branch], tt_residual_branch_outputs[branch]):
                    hf_branch_f = hf_branch.float()
                    tt_branch_f = tt_branch.float()
                    residual_branch_summary[branch].append(
                        {
                            "pcc": _pcc(hf_branch_f, tt_branch_f),
                            "max_abs": float((hf_branch_f - tt_branch_f).abs().max()),
                        }
                    )
        hf_lm_head_on_tt_hidden_topk = []
        if capture_positions and tt_final_hidden:
            with torch.no_grad():
                lm_head_dtype = next(hf_model.lm_head.parameters()).dtype
                logits = hf_model.lm_head(tt_final_hidden[0].to(lm_head_dtype)).float()
                softcap = float(hf_model.config.text_config.final_logit_softcapping)
                logits = torch.tanh(logits / softcap) * softcap
            hf_lm_head_on_tt_hidden_topk.append(
                _logits_topk_summary(logits, list(range(len(capture_positions))), k=args.capture_top_k, step=0)
            )
            for row, absolute_pos in zip(hf_lm_head_on_tt_hidden_topk[0]["rows"], capture_positions):
                row["pos"] = absolute_pos
        artifact.update(
            {
                "tt_traj": tt_traj,
                "tt_logits_topk": tt_logits_topk,
                "hf_lm_head_on_tt_hidden_topk": hf_lm_head_on_tt_hidden_topk,
                "layer_hidden_summary": layer_hidden_summary,
                "attention_hidden_summary": attention_hidden_summary,
                "residual_branch_summary": residual_branch_summary,
                "prompt_kv_summary": prompt_kv_summary,
                "tp_partial_captures": tp_partial_captures,
                "hf_residual_branch_outputs": hf_residual_branch_outputs,
                "comparison": asdict(comparison),
            }
        )

    artifact["summary"] = summary
    output = Path(args.output)
    torch.save(artifact, output)
    print(json.dumps({"output": str(output), "summary": summary}, indent=2, sort_keys=True))
    if args.stage_gate and not summary["stage_gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
