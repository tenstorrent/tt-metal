# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded full-48-layer ordinary-decode CPU oracle smoke test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38TextModelOracle

OFFICIAL_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."
OFFICIAL_CHAT_SKY_PROMPT = "In one short sentence, explain why the daytime sky usually appears blue."
OFFICIAL_CHAT_SKY_PROMPT_DISCRIMINATOR = "short_sky_explanation_v1"
MAX_CHAT_CONTINUATIONS = 32
MAX_CHAT_TOP_K = 64
EXPECTED_OFFICIAL_SKY_PROMPT_TOKEN_IDS = (
    248_045,
    8_678,
    198,
    2_523,
    513,
    264,
    10_631,
    17_313,
    13,
    248_046,
    198,
    248_045,
    846,
    198,
    623,
    799,
    2_716,
    11_316,
    11,
    10_033,
    3_069,
    279,
    59_024,
    12_515,
    5_802,
    7_701,
    6_105,
    13,
    248_046,
    198,
    248_045,
    74_455,
    198,
    248_068,
    271,
    248_069,
    271,
)
EXPECTED_OFFICIAL_SKY_PROMPT_TOKEN_IDS_SHA256 = "bec9bcc2f17d8e2bcf9dc732c8aa707d5ad32a164dcac1975ce8b8c6a94e7712"
EXPECTED_OFFICIAL_SKY_CONTINUATION_PREFIX = (760, 12_515)
EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_INDICES = (12_515, 59_024, 6_105, 6_797, 8_964)
EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_VALUES = (21.25, 20.125, 16.25, 15.3125, 14.3125)
EXPECTED_OFFICIAL_SKY_EOS_TOKEN_ID = 248_046
EXPECTED_OFFICIAL_SKY_EOS_INDEX = 28


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _record(event: str, **values) -> None:
    print(json.dumps({"event": event, "monotonic_seconds": time.monotonic(), **values}, sort_keys=True), flush=True)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    difference = (actual - expected).abs()
    if actual.numel() < 2 or float(actual.std()) == 0.0 or float(expected.std()) == 0.0:
        correlation = torch.tensor(1.0 if torch.equal(actual, expected) else 0.0)
    else:
        correlation = torch.corrcoef(torch.stack((actual, expected)))[0, 1]
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(torch.quantile(difference, 0.99)),
        "pcc": float(correlation),
    }


def _render_official_sky_prompt(template):
    """Render the same fixed ordinary request used by ``--run-append-once``."""

    from models.demos.blackhole.qwen38_flash_next.chat import PINNED_TOKENIZER_ARTIFACTS

    rendered = template.render(
        (
            {"role": "system", "content": OFFICIAL_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": OFFICIAL_CHAT_SKY_PROMPT},
        ),
        tools=(),
        enable_thinking=False,
        preserve_thinking=True,
        reasoning_effort="low",
    )
    input_ids = getattr(rendered, "input_ids", None)
    prompt_token_ids = (
        tuple(int(token) for token in input_ids.view(-1).tolist()) if isinstance(input_ids, torch.Tensor) else ()
    )
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.device.type != "cpu"
        or input_ids.dtype != torch.int64
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
        or getattr(rendered, "enable_thinking", None) is not False
        or getattr(rendered, "preserve_thinking", None) is not True
        or getattr(rendered, "reasoning_effort", None) != "low"
        or getattr(rendered, "template_sha256", None) != PINNED_TOKENIZER_ARTIFACTS["chat_template.jinja"]
        or not isinstance(getattr(rendered, "text", None), str)
        or OFFICIAL_CHAT_SYSTEM_PROMPT not in rendered.text
        or OFFICIAL_CHAT_SKY_PROMPT not in rendered.text
        or prompt_token_ids != EXPECTED_OFFICIAL_SKY_PROMPT_TOKEN_IDS
    ):
        raise RuntimeError("official sky prompt rendering differs from the append-once chat contract")
    return rendered


def _current_routing(routing) -> tuple[torch.Tensor, torch.Tensor]:
    indices = getattr(routing, "indices", None)
    scores = getattr(routing, "scores", None)
    if (
        not isinstance(indices, torch.Tensor)
        or not isinstance(scores, torch.Tensor)
        or indices.device.type != "cpu"
        or scores.device.type != "cpu"
        or indices.shape != scores.shape
        or indices.ndim < 1
        or indices.shape[-1] == 0
    ):
        raise RuntimeError("CPU oracle routing tensors are malformed")
    current_indices = indices.reshape(-1, indices.shape[-1])[-1:].contiguous()
    current_scores = scores.reshape(-1, scores.shape[-1])[-1:].contiguous()
    if not torch.isfinite(current_scores.float()).all():
        raise RuntimeError("CPU oracle routing scores are non-finite")
    return current_indices, current_scores


def _generate_official_sky_reference(
    *,
    model,
    rendered,
    tokenizer,
    continuation_tokens: int,
    selected_step: int,
    top_k: int,
    expected_layers: int,
    eos_token_ids: tuple[int, ...],
    retained_reference: bool = False,
    release_layer_experts=None,
) -> dict[str, object]:
    """Greedily decode the rendered prompt and retain one exact layerwise anchor."""

    if (
        isinstance(continuation_tokens, bool)
        or not isinstance(continuation_tokens, int)
        or not 1 <= continuation_tokens <= MAX_CHAT_CONTINUATIONS
    ):
        raise ValueError(f"continuation_tokens must be in [1,{MAX_CHAT_CONTINUATIONS}]")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or not 0 <= selected_step < continuation_tokens
    ):
        raise ValueError("selected_step must identify one requested continuation")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_CHAT_TOP_K:
        raise ValueError(f"top_k must be in [1,{MAX_CHAT_TOP_K}]")
    if isinstance(expected_layers, bool) or not isinstance(expected_layers, int) or expected_layers <= 0:
        raise ValueError("expected_layers must be a positive integer")
    if (
        type(eos_token_ids) is not tuple
        or not eos_token_ids
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in eos_token_ids)
    ):
        raise ValueError("eos_token_ids must be one non-empty exact integer tuple")
    if type(retained_reference) is not bool:
        raise TypeError("retained_reference must be boolean")
    if release_layer_experts is not None and not callable(release_layer_experts):
        raise TypeError("release_layer_experts must be callable when supplied")

    prompt_ids = rendered.input_ids
    prompt_length = int(prompt_ids.shape[1])
    state = None
    model_input = prompt_ids
    generated: list[int] = []
    selected_layers: list[int] = []
    stop_reason = "max_continuations"

    for token_index in range(continuation_tokens):
        step_start = time.monotonic()
        input_position = prompt_length - 1 + token_index

        def observe(layer_index, hidden_states, layer_state, aux):
            del layer_state
            try:
                if token_index != selected_step:
                    return
                if not isinstance(hidden_states, torch.Tensor) or hidden_states.device.type != "cpu":
                    raise RuntimeError("CPU oracle layer hidden state is not a host tensor")
                if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[1] == 0:
                    raise RuntimeError("CPU oracle layer hidden state has an invalid shape")
                current_hidden = hidden_states[:, -1:].contiguous()
                if not torch.isfinite(current_hidden.float()).all():
                    raise RuntimeError("CPU oracle layer hidden state is non-finite")
                route_indices, route_scores = _current_routing(aux.routing)
                route_digest = hashlib.sha256()
                route_digest.update(_tensor_bytes(route_indices))
                route_digest.update(_tensor_bytes(route_scores))
                selected_layers.append(int(layer_index))
                _record(
                    "chat_layer",
                    token_index=token_index,
                    input_position=input_position,
                    layer=int(layer_index),
                    hidden_shape=list(current_hidden.shape),
                    hidden_dtype=str(current_hidden.dtype),
                    hidden_sha256=_tensor_sha256(current_hidden),
                    selected_experts=route_indices.view(-1).tolist(),
                    route_scores=route_scores.float().view(-1).tolist(),
                    route_indices_sha256=_tensor_sha256(route_indices),
                    route_scores_sha256=_tensor_sha256(route_scores),
                    route_sha256=route_digest.hexdigest(),
                )
            finally:
                if release_layer_experts is not None:
                    release_layer_experts(int(layer_index))

        output = model.forward(
            model_input,
            state=state,
            return_logits=True,
            logits_to_keep=1,
            layer_observer=observe,
        )
        logits = getattr(output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.device.type != "cpu"
            or tuple(logits.shape[:2]) != (1, 1)
            or logits.shape[-1] < top_k
            or not torch.isfinite(logits.float()).all()
        ):
            raise RuntimeError("official sky CPU oracle did not produce finite B1 current-position logits")
        ranking_k = max(top_k, 5 if retained_reference and token_index == 1 else top_k)
        ranked_values, ranked_indices = logits.float().topk(ranking_k, dim=-1)
        values = ranked_values[..., :top_k]
        indices = ranked_indices[..., :top_k]
        next_token = int(logits.argmax(dim=-1).item())
        retained_expected_token = None
        retained_token_match = None
        retained_observed_top5_indices = None
        retained_observed_top5_values = None
        retained_expected_top5_indices = None
        retained_expected_top5_values = None
        retained_top5_match = None
        retained_gate_applied = False
        retained_failures: list[str] = []
        if retained_reference and token_index < len(EXPECTED_OFFICIAL_SKY_CONTINUATION_PREFIX):
            retained_gate_applied = True
            retained_expected_token = EXPECTED_OFFICIAL_SKY_CONTINUATION_PREFIX[token_index]
            retained_token_match = next_token == retained_expected_token
            if not retained_token_match:
                retained_failures.append(
                    "official sky CPU continuation prefix differs at "
                    f"token {token_index}: observed={next_token} expected={retained_expected_token}"
                )
        if retained_reference and token_index == 1:
            retained_gate_applied = True
            retained_observed_top5_indices = tuple(int(token) for token in ranked_indices.view(-1)[:5].tolist())
            retained_observed_top5_values = tuple(float(value) for value in ranked_values.view(-1)[:5].tolist())
            retained_expected_top5_indices = EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_INDICES
            retained_expected_top5_values = EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_VALUES
            retained_top5_match = (
                retained_observed_top5_indices == retained_expected_top5_indices
                and retained_observed_top5_values == retained_expected_top5_values
            )
            if not retained_top5_match:
                retained_failures.append(
                    "official sky CPU token-one top-5 differs from the retained serial/vectorized reference: "
                    f"observed_indices={retained_observed_top5_indices} "
                    f"expected_indices={retained_expected_top5_indices} "
                    f"observed_values={retained_observed_top5_values} "
                    f"expected_values={retained_expected_top5_values}"
                )
        expected_position = prompt_length + token_index
        if getattr(output.state, "position", None) != expected_position:
            actual_position = getattr(output.state, "position", None)
            raise RuntimeError(f"official sky CPU state position differs: {actual_position} != {expected_position}")
        if token_index == selected_step and selected_layers != list(range(expected_layers)):
            raise RuntimeError("selected CPU continuation did not expose every ordered backbone layer")

        generated.append(next_token)
        output_hidden = getattr(output, "hidden_states", None)
        if (
            not isinstance(output_hidden, torch.Tensor)
            or output_hidden.device.type != "cpu"
            or output_hidden.ndim != 3
            or output_hidden.shape[0] != 1
            or output_hidden.shape[1] == 0
            or not torch.isfinite(output_hidden.float()).all()
        ):
            raise RuntimeError("official sky CPU oracle final-mixer hidden state is malformed")
        current_hidden = output_hidden[:, -1:].contiguous()
        _record(
            "chat_decode",
            token_index=token_index,
            input_position=input_position,
            input_token_id=None if token_index == 0 else int(model_input.item()),
            output_token=next_token,
            topk_indices=indices.view(-1).tolist(),
            topk_values=values.view(-1).tolist(),
            hidden_sha256=_tensor_sha256(current_hidden),
            logits_sha256=_tensor_sha256(logits),
            state_position=output.state.position,
            selected_layer_anchor=token_index == selected_step,
            retained_gate_applied=retained_gate_applied,
            retained_gate_status=("fail" if retained_failures else "pass") if retained_gate_applied else None,
            retained_expected_token=retained_expected_token,
            retained_token_match=retained_token_match,
            retained_observed_top5_indices=retained_observed_top5_indices,
            retained_observed_top5_values=retained_observed_top5_values,
            retained_expected_top5_indices=retained_expected_top5_indices,
            retained_expected_top5_values=retained_expected_top5_values,
            retained_top5_match=retained_top5_match,
            elapsed_seconds=time.monotonic() - step_start,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        gc.collect()
        if retained_failures:
            raise RuntimeError("; ".join(retained_failures))
        state = output.state
        if next_token in eos_token_ids:
            stop_reason = "eos"
            break
        model_input = torch.tensor([[next_token]], dtype=torch.long)

    if selected_layers != list(range(expected_layers)):
        raise RuntimeError("generation stopped before the selected CPU layerwise anchor")
    retained_eos_match = None
    if retained_reference and continuation_tokens > EXPECTED_OFFICIAL_SKY_EOS_INDEX:
        retained_eos_match = (
            len(generated) == EXPECTED_OFFICIAL_SKY_EOS_INDEX + 1
            and generated[EXPECTED_OFFICIAL_SKY_EOS_INDEX] == EXPECTED_OFFICIAL_SKY_EOS_TOKEN_ID
            and stop_reason == "eos"
        )
        if not retained_eos_match:
            raise RuntimeError("official sky CPU EOS boundary differs from the retained full continuation")
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise RuntimeError("official tokenizer did not decode CPU continuation IDs to text")
    summary: dict[str, object] = {
        "requested_continuations": continuation_tokens,
        "generated_continuations": len(generated),
        "generated_token_ids": generated,
        "selected_step": selected_step,
        "selected_input_position": prompt_length - 1 + selected_step,
        "selected_layer_count": len(selected_layers),
        "stop_reason": stop_reason,
        "terminal_eos_token_id": generated[-1] if stop_reason == "eos" else None,
        "decoded_text": decoded,
        "retained_reference_authenticated": retained_reference,
        "retained_eos_match": retained_eos_match,
    }
    _record("chat_complete", **summary)
    return summary


def _run_official_sky_chat(args: argparse.Namespace) -> int:
    from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS, Qwen38OfficialChatTemplate

    torch.set_grad_enabled(False)
    checkpoint = Qwen38Checkpoint(args.checkpoint)
    if args.top_k > checkpoint.config.vocab_size:
        raise ValueError("top_k exceeds the checkpoint vocabulary")
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    template = Qwen38OfficialChatTemplate(args.checkpoint)
    rendered = _render_official_sky_prompt(template)
    model = Qwen38TextModelOracle(checkpoint, placement)

    def release_layer_experts(layer_index: int) -> None:
        cache = getattr(model.layer(layer_index).mlp.weights, "_expert_cache", None)
        if type(cache) is not dict:
            raise RuntimeError("CPU oracle layer expert cache ownership changed")
        cache.clear()

    prompt_token_ids = tuple(int(token) for token in rendered.input_ids.view(-1).tolist())
    token_payload = json.dumps(prompt_token_ids, separators=(",", ":")).encode("ascii")
    _record(
        "start",
        mode="official_sky_chat_cpu_oracle",
        checkpoint=str(args.checkpoint.resolve()),
        config_sha256=checkpoint.config.config_sha256,
        layers=checkpoint.config.num_hidden_layers,
        physical_ids=list(placement.physical_ids),
        execution="CPU integration oracle; no TT device opened",
        greedy=True,
        enable_thinking=False,
        preserve_thinking=True,
        reasoning_effort="low",
        requested_continuations=args.continuation_tokens,
        selected_step=args.selected_step,
        top_k=args.top_k,
        expert_cache_policy="clear_each_layer_after_forward",
    )
    _record(
        "official_prompt",
        system_prompt=OFFICIAL_CHAT_SYSTEM_PROMPT,
        user_prompt=OFFICIAL_CHAT_SKY_PROMPT,
        user_prompt_discriminator=OFFICIAL_CHAT_SKY_PROMPT_DISCRIMINATOR,
        template_sha256=rendered.template_sha256,
        rendered_text=rendered.text,
        rendered_text_sha256=hashlib.sha256(rendered.text.encode("utf-8")).hexdigest(),
        prompt_token_count=len(prompt_token_ids),
        prompt_token_ids=list(prompt_token_ids),
        prompt_token_ids_sha256=hashlib.sha256(token_payload).hexdigest(),
        retained_prompt_token_ids_match=prompt_token_ids == EXPECTED_OFFICIAL_SKY_PROMPT_TOKEN_IDS,
        retained_prompt_token_ids_sha256=EXPECTED_OFFICIAL_SKY_PROMPT_TOKEN_IDS_SHA256,
        retained_continuation_prefix=list(EXPECTED_OFFICIAL_SKY_CONTINUATION_PREFIX),
        retained_token1_top5_indices=list(EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_INDICES),
        retained_token1_top5_values=list(EXPECTED_OFFICIAL_SKY_TOKEN1_TOP5_VALUES),
        retained_eos_token_id=EXPECTED_OFFICIAL_SKY_EOS_TOKEN_ID,
        retained_eos_index=EXPECTED_OFFICIAL_SKY_EOS_INDEX,
    )
    _generate_official_sky_reference(
        model=model,
        rendered=rendered,
        tokenizer=template.tokenizer,
        continuation_tokens=args.continuation_tokens,
        selected_step=args.selected_step,
        top_k=args.top_k,
        expected_layers=checkpoint.config.num_hidden_layers,
        eos_token_ids=EOS_TOKEN_IDS,
        retained_reference=True,
        release_layer_experts=release_layer_experts,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=17)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--verify-prefill-equivalence", action="store_true")
    parser.add_argument("--official-sky-chat", action="store_true")
    parser.add_argument("--continuation-tokens", type=int, default=2)
    parser.add_argument("--selected-step", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.official_sky_chat:
        if args.verify_prefill_equivalence:
            parser.error("--verify-prefill-equivalence is unavailable in official sky chat mode")
        if not 1 <= args.continuation_tokens <= MAX_CHAT_CONTINUATIONS:
            parser.error(f"--continuation-tokens must be in [1,{MAX_CHAT_CONTINUATIONS}]")
        if not 0 <= args.selected_step < args.continuation_tokens:
            parser.error("--selected-step must identify one requested continuation")
        if not 1 <= args.top_k <= MAX_CHAT_TOP_K:
            parser.error(f"--top-k must be in [1,{MAX_CHAT_TOP_K}]")
        return _run_official_sky_chat(args)
    if args.decode_steps not in (1, 2):
        parser.error("this bounded smoke test admits one or two ordinary decode steps")

    torch.set_grad_enabled(False)
    checkpoint = Qwen38Checkpoint(args.checkpoint)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    model = Qwen38TextModelOracle(checkpoint, placement)
    _record(
        "start",
        checkpoint=str(args.checkpoint.resolve()),
        config_sha256=checkpoint.config.config_sha256,
        layers=checkpoint.config.num_hidden_layers,
        physical_ids=list(placement.physical_ids),
        execution="CPU integration oracle; no TT device opened",
    )

    state = None
    token = torch.tensor([[args.token_id]], dtype=torch.long)
    token_history = []
    last_output = None
    tokenwise_layer_hidden = []
    tokenwise_layer_routes = []
    for step in range(args.decode_steps):
        step_start = time.monotonic()
        token_history.append(int(token.item()))

        def observe(layer_index, hidden_states, layer_state, aux):
            del layer_state
            if step == args.decode_steps - 1:
                tokenwise_layer_hidden.append(hidden_states.detach().clone())
                tokenwise_layer_routes.append(aux.routing.indices.detach().clone())
            _record(
                "layer",
                step=step,
                layer=layer_index,
                hidden_sha256=_tensor_sha256(hidden_states),
                selected_experts=aux.routing.indices.view(-1).tolist(),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )

        output = model.forward(token, state=state, layer_observer=observe)
        if output.logits is None or output.logits.shape != (1, 1, checkpoint.config.vocab_size):
            raise RuntimeError("full oracle did not produce exact-vocabulary logits")
        if not torch.isfinite(output.logits.float()).all():
            raise RuntimeError("full oracle produced non-finite logits")
        values, indices = output.logits.float().topk(5, dim=-1)
        next_token = output.logits.argmax(dim=-1)
        _record(
            "decode",
            step=step,
            input_token=int(token.item()),
            output_token=int(next_token.item()),
            top5_indices=indices.view(-1).tolist(),
            top5_values=values.view(-1).tolist(),
            hidden_sha256=_tensor_sha256(output.hidden_states),
            logits_sha256=_tensor_sha256(output.logits),
            state_position=output.state.position,
            elapsed_seconds=time.monotonic() - step_start,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        state = output.state
        token = next_token.to(dtype=torch.long, device="cpu")
        last_output = output

    if args.verify_prefill_equivalence:
        if args.decode_steps != 2:
            raise RuntimeError("prefill equivalence is defined for this smoke test's two tokenwise steps")
        prefill_start = time.monotonic()
        prefill_route_overlaps = []
        prefill_exact_routes = 0
        prefill_top1_routes = 0

        def observe_prefill(layer_index, hidden_states, layer_state, aux):
            del layer_state
            tokenwise_indices = tokenwise_layer_routes[layer_index].view(-1).tolist()
            prefill_indices = aux.routing.indices[-1:].view(-1).tolist()
            overlap = len(set(tokenwise_indices) & set(prefill_indices))
            prefill_route_overlaps.append(overlap)
            nonlocal prefill_exact_routes, prefill_top1_routes
            prefill_exact_routes += int(tokenwise_indices == prefill_indices)
            prefill_top1_routes += int(tokenwise_indices[0] == prefill_indices[0])
            _record(
                "prefill_layer",
                layer=layer_index,
                hidden=_error_stats(tokenwise_layer_hidden[layer_index], hidden_states[:, -1:]),
                route_overlap=overlap,
                route_exact=tokenwise_indices == prefill_indices,
                route_top1_match=tokenwise_indices[0] == prefill_indices[0],
            )

        prefill = model.forward(torch.tensor([token_history], dtype=torch.long), layer_observer=observe_prefill)
        hidden_stats = _error_stats(last_output.hidden_states[:, -1:], prefill.hidden_states[:, -1:])
        logits_stats = _error_stats(last_output.logits, prefill.logits)
        top_token_match = bool(torch.equal(last_output.logits.argmax(dim=-1), prefill.logits.argmax(dim=-1)))
        state_stats = []
        ple_context_matches = 0
        for tokenwise_layer, prefill_layer in zip(last_output.state.layers, prefill.state.layers):
            tokenwise_attention = tokenwise_layer.attention
            prefill_attention = prefill_layer.attention
            if hasattr(tokenwise_attention, "recurrent"):
                state_stats.append(_error_stats(tokenwise_attention.recurrent, prefill_attention.recurrent))
                state_stats.append(_error_stats(tokenwise_attention.conv, prefill_attention.conv))
            elif hasattr(tokenwise_attention, "raw_index_keys"):
                state_stats.append(_error_stats(tokenwise_attention.raw_index_keys, prefill_attention.raw_index_keys))
                state_stats.append(_error_stats(tokenwise_attention.keys, prefill_attention.keys))
                state_stats.append(_error_stats(tokenwise_attention.values, prefill_attention.values))
            if tokenwise_layer.ple is not None:
                ple_context_matches += int(
                    torch.equal(tokenwise_layer.ple.token_context, prefill_layer.ple.token_context)
                )
                state_stats.append(_error_stats(tokenwise_layer.ple.conv, prefill_layer.ple.conv))
        state_min_pcc = min(item["pcc"] for item in state_stats)
        state_max_p99_abs = max(item["p99_abs"] for item in state_stats)
        _record(
            "prefill_equivalence",
            tokens=token_history,
            hidden=hidden_stats,
            logits=logits_stats,
            route_exact_layers=prefill_exact_routes,
            route_top1_layers=prefill_top1_routes,
            route_overlap_min=min(prefill_route_overlaps),
            route_overlap_mean=sum(prefill_route_overlaps) / len(prefill_route_overlaps),
            top_token_match=top_token_match,
            state_min_pcc=state_min_pcc,
            state_max_p99_abs=state_max_p99_abs,
            ple_context_matches=ple_context_matches,
            elapsed_seconds=time.monotonic() - prefill_start,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        if (
            hidden_stats["pcc"] < 0.9998
            or hidden_stats["p99_abs"] > 0.15
            or logits_stats["pcc"] < 0.9998
            or logits_stats["p99_abs"] > 0.1
            or min(prefill_route_overlaps) < 8
            or state_min_pcc < 0.999
            or ple_context_matches != 1
            or not top_token_match
        ):
            raise RuntimeError("full prefill/tokenwise numerical equivalence gate failed")

    _record(
        "complete",
        steps=args.decode_steps,
        final_position=state.position,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
