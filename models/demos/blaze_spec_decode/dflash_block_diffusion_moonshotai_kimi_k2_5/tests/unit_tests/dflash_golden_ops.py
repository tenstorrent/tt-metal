# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DFlashConfig:
    """Port of golden_dflash.config.DFlashConfig for local stage golden tests."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int | None = None
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int | None = None
    rope_scaling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fixture(cls, fixture: dict) -> "DFlashConfig":
        raw = fixture["config"]
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim", int(raw["hidden_size"]) // int(raw["num_attention_heads"]))),
            block_size=int(raw["block_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rope_theta=float(raw["rope_theta"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            attention_bias=bool(raw.get("attention_bias", False)),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            layer_types=list(raw.get("layer_types", ["full_attention"] * int(raw["num_hidden_layers"]))),
            sliding_window=raw.get("sliding_window"),
            target_layer_ids=list(raw.get("target_layer_ids", [])),
            mask_token_id=raw.get("mask_token_id"),
            rope_scaling=dict(raw.get("rope_scaling", {})),
        )

    @classmethod
    def tiny(
        cls,
        *,
        vocab_size: int = 32,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        block_size: int = 4,
        target_layer_ids: list[int] | None = None,
    ) -> "DFlashConfig":
        head_dim = hidden_size // num_attention_heads
        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            layer_types=["full_attention"] * num_hidden_layers,
            target_layer_ids=target_layer_ids or [0, 1],
            mask_token_id=vocab_size - 1,
        )


@dataclass(frozen=True)
class GoldenReplayResult:
    output_token_ids: list[int]
    acceptance_lengths: list[int]
    host_writes: list[dict[str, int | str]]


def load_stage_fixture(path: Path, expected_stage: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        fixture = json.load(f)
    assert fixture["schema_version"] == 1
    assert fixture["approach"] == "dflash_block_diffusion"
    assert fixture["target_model"] == "moonshotai/Kimi-K2.5"
    assert fixture["draft_model"] == "z-lab/Kimi-K2.5-DFlash"
    assert fixture["stage"] == expected_stage
    return fixture


def tensor(value: list, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype)


def assert_close(actual: torch.Tensor, expected: list, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    torch.testing.assert_close(actual, tensor(expected), atol=atol, rtol=rtol)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_norm).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Port of golden_dflash.draft_model.RotaryEmbedding, including YaRN."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        inv_freq, attention_factor = self._compute_inv_freq(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_factor = float(attention_factor)

    @staticmethod
    def _compute_inv_freq(config: DFlashConfig) -> tuple[torch.Tensor, float]:
        scaling = config.rope_scaling or {}
        rope_type = scaling.get("rope_type", scaling.get("type"))
        dim = config.head_dim
        base = config.rope_theta
        if rope_type != "yarn":
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            return inv, 1.0

        factor = float(scaling["factor"])
        mscale = scaling.get("mscale")
        mscale_all_dim = scaling.get("mscale_all_dim")
        original_max = int(scaling.get("original_max_position_embeddings") or config.max_position_embeddings)

        def get_mscale(scale, mscale_value=1.0):
            if scale <= 1:
                return 1.0
            return 0.1 * float(mscale_value) * math.log(scale) + 1.0

        if mscale and mscale_all_dim:
            attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
        else:
            attention_factor = get_mscale(factor)

        beta_fast = float(scaling.get("beta_fast") or 32)
        beta_slow = float(scaling.get("beta_slow") or 1)
        truncate = bool(scaling.get("truncate", True))

        def find_correction_dim(num_rotations):
            return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        low = find_correction_dim(beta_fast)
        high = find_correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0)
        high = min(high, dim - 1)
        if low == high:
            high += 0.001

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_extrapolation = 1.0 / pos_freqs
        inv_interpolation = 1.0 / (factor * pos_freqs)
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        extrapolation_factor = 1 - ramp
        inv = inv_interpolation * (1 - extrapolation_factor) + inv_extrapolation * extrapolation_factor
        return inv, attention_factor

    def forward(
        self,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_factor).to(dtype), (emb.sin() * self.attention_factor).to(dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        groups = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn)


class DFlashMLP(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DFlashAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DFlashMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, target_hidden, position_embeddings)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class KimiDFlashDraftModel(nn.Module):
    """Port of golden_dflash.draft_model.KimiDFlashDraftModel."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.target_layer_ids = config.target_layer_ids
        self.layers = nn.ModuleList([DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        pos = self.rotary_emb(position_ids, hidden_states.dtype, hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, pos)
        return self.norm(hidden_states)


def greedy_sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab = logits.shape
    probs = torch.softmax(logits.reshape(-1, vocab) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def golden_extract_context_feature(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def golden_pre_decoder_fused_stage(fixture: dict) -> dict:
    config = DFlashConfig.from_fixture(fixture)
    inputs = fixture["inputs"]
    weights = fixture["weights"]

    base_hidden = tensor(inputs["base_hidden"])
    target_hidden = tensor(inputs["target_hidden"])
    noise_embedding = tensor(inputs["noise_embedding"])
    position_ids = torch.tensor(inputs["position_ids"], dtype=torch.long)
    lm_head_weight = tensor(weights["target_lm_head_weight"])

    base_norm = RMSNorm(config.hidden_size, config.rms_norm_eps).eval()
    _copy_parameter(base_norm.weight, weights["base_norm_weight"])
    drafter = KimiDFlashDraftModel(config).eval()
    _copy_parameter(drafter.fc.weight, weights["fc_weight"])
    _copy_parameter(drafter.hidden_norm.weight, weights["hidden_norm_weight"])

    with torch.inference_mode():
        base_logits = base_norm(base_hidden) @ lm_head_weight.T
        target_context = drafter.hidden_norm(drafter.fc(target_hidden))
        cos, sin = drafter.rotary_emb(position_ids, noise_embedding.dtype, noise_embedding.device)

    return {
        "base_logits": base_logits,
        "base_token_id": int(greedy_sample(base_logits.unsqueeze(1))[:, 0][0].item()),
        "target_context": target_context,
        "position_cos": cos,
        "position_sin": sin,
        "decoder_input": noise_embedding,
    }


def golden_decoder_layer_stage(fixture: dict) -> torch.Tensor:
    config = DFlashConfig.from_fixture(fixture)
    inputs = fixture["inputs"]
    layer = _decoder_layer_from_fixture(config, fixture["weights"]).eval()

    with torch.inference_mode():
        return layer(
            tensor(inputs["hidden_states"]),
            tensor(inputs["target_context"]),
            (tensor(inputs["position_cos"]), tensor(inputs["position_sin"])),
        )


def golden_post_decoder_fused_stage(fixture: dict) -> dict:
    config = DFlashConfig.from_fixture(fixture)
    inputs = fixture["inputs"]
    weights = fixture["weights"]

    hidden_states = tensor(inputs["hidden_states"])
    final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps).eval()
    _copy_parameter(final_norm.weight, weights["final_norm_weight"])
    lm_head_weight = tensor(weights["target_lm_head_weight"])

    with torch.inference_mode():
        final_hidden = final_norm(hidden_states)
        draft_logits = final_hidden[:, 1 - config.block_size :, :] @ lm_head_weight.T
        draft_token_ids = greedy_sample(draft_logits)

    anchor_position = int(inputs["anchor_position"])
    return {
        "final_hidden": final_hidden,
        "draft_logits": draft_logits,
        "draft_token_ids": draft_token_ids,
        "host_packet": {
            "type": "DRAFT_BLOCK_PROPOSAL",
            "anchor_position": anchor_position,
            "token_ids": draft_token_ids.tolist()[0],
            "positions": list(range(anchor_position + 1, anchor_position + config.block_size)),
        },
    }


def golden_combined_drafter(fixture: dict, fixture_dir: Path) -> dict:
    stage_fixtures = [
        load_stage_fixture(fixture_dir / rel_path, _expected_stage_from_fixture_name(rel_path))
        for rel_path in fixture["stage_fixture_paths"]
    ]
    assert stage_fixtures[0]["stage"] == "pre_decoder_fused"
    assert stage_fixtures[-1]["stage"] == "post_decoder_fused"

    pre = golden_pre_decoder_fused_stage(stage_fixtures[0])
    hidden_states = pre["decoder_input"]
    target_context = pre["target_context"]
    position_cos = pre["position_cos"]
    position_sin = pre["position_sin"]

    for layer_fixture in stage_fixtures[1:-1]:
        layer_fixture = copy.deepcopy(layer_fixture)
        layer_fixture["inputs"]["hidden_states"] = hidden_states.tolist()
        layer_fixture["inputs"]["target_context"] = target_context.tolist()
        layer_fixture["inputs"]["position_cos"] = position_cos.tolist()
        layer_fixture["inputs"]["position_sin"] = position_sin.tolist()
        hidden_states = golden_decoder_layer_stage(layer_fixture)

    post_fixture = copy.deepcopy(stage_fixtures[-1])
    post_fixture["inputs"]["hidden_states"] = hidden_states.tolist()
    return golden_post_decoder_fused_stage(post_fixture)


def golden_accepted_after_anchor(block_token_ids: list[int], target_posterior_token_ids: list[int]) -> int:
    block_output_ids = torch.tensor([block_token_ids], dtype=torch.long)
    posterior = torch.tensor([target_posterior_token_ids], dtype=torch.long)
    matches = block_output_ids[:, 1:] == posterior[:, :-1]
    return int(matches.cumprod(dim=1).sum(dim=1)[0].item())


def golden_host_writes_for_block(block_token_ids: list[int], *, anchor_pos: int) -> list[dict[str, int | str]]:
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


def golden_replay_dflash_lossless_trace(trace: dict) -> GoldenReplayResult:
    params = trace["params"]
    prompt_token_ids = [int(token_id) for token_id in params["prompt_token_ids"]]
    block_size = int(params["block_size"])
    max_new_tokens = int(params["max_new_tokens"])
    max_length = len(prompt_token_ids) + max_new_tokens
    mask_token_id = int(params["mask_token_id"])

    output_ids = torch.full(
        (1, max_length + block_size),
        fill_value=mask_token_id,
        dtype=torch.long,
    )
    output_ids[:, : len(prompt_token_ids)] = torch.tensor([prompt_token_ids], dtype=torch.long)
    start = len(prompt_token_ids)
    output_ids[:, start : start + 1] = int(trace["prefill"]["target_next_token_id"])

    acceptance_lengths: list[int] = []
    host_writes: list[dict[str, int | str]] = []

    for round_idx, round_record in enumerate(trace["rounds"]):
        assert int(round_record["anchor_pos"]) == start, f"round {round_idx} starts at unexpected position"

        block_token_ids = [int(token_id) for token_id in round_record["block_input_token_ids"]]
        target_posterior_token_ids = [int(token_id) for token_id in round_record["target_posterior_token_ids"]]
        block_output_ids = torch.tensor([block_token_ids], dtype=torch.long)
        posterior = torch.tensor([target_posterior_token_ids], dtype=torch.long)
        assert block_output_ids.shape[1] == block_size
        assert posterior.shape[1] == block_size
        assert int(output_ids[0, start].item()) == block_token_ids[0]

        host_writes.extend(golden_host_writes_for_block(block_token_ids, anchor_pos=start))
        accepted_after_anchor = golden_accepted_after_anchor(block_token_ids, target_posterior_token_ids)
        committed = accepted_after_anchor + 1
        assert committed == int(round_record["expected_committed_tokens"])
        acceptance_lengths.append(committed)

        output_ids[:, start : start + committed] = block_output_ids[:, :committed]
        if start + committed < output_ids.shape[1]:
            next_anchor_token_id = int(posterior[0, accepted_after_anchor].item())
            assert next_anchor_token_id == int(round_record["expected_next_anchor_token_id"])
            output_ids[:, start + committed] = next_anchor_token_id

        start += committed
        if start >= max_length:
            break

    output_ids = output_ids[:, : min(start, max_length)]
    return GoldenReplayResult(
        output_token_ids=output_ids[0].tolist(),
        acceptance_lengths=acceptance_lengths,
        host_writes=host_writes,
    )


def _decoder_layer_from_fixture(config: DFlashConfig, weights: dict) -> DFlashDecoderLayer:
    layer = DFlashDecoderLayer(config)
    _copy_parameter(layer.input_layernorm.weight, weights["input_layernorm_weight"])
    _copy_parameter(layer.self_attn.q_proj.weight, weights["q_proj_weight"])
    _copy_parameter(layer.self_attn.k_proj.weight, weights["k_proj_weight"])
    _copy_parameter(layer.self_attn.v_proj.weight, weights["v_proj_weight"])
    _copy_parameter(layer.self_attn.o_proj.weight, weights["o_proj_weight"])
    _copy_parameter(layer.self_attn.q_norm.weight, weights["q_norm_weight"])
    _copy_parameter(layer.self_attn.k_norm.weight, weights["k_norm_weight"])
    _copy_parameter(layer.post_attention_layernorm.weight, weights["post_attention_layernorm_weight"])
    _copy_parameter(layer.mlp.gate_proj.weight, weights["gate_proj_weight"])
    _copy_parameter(layer.mlp.up_proj.weight, weights["up_proj_weight"])
    _copy_parameter(layer.mlp.down_proj.weight, weights["down_proj_weight"])
    return layer


def _copy_parameter(parameter: nn.Parameter, value: list) -> None:
    with torch.no_grad():
        parameter.copy_(tensor(value, dtype=parameter.dtype))


def _expected_stage_from_fixture_name(rel_path: str) -> str:
    name = Path(rel_path).name
    if name == "stage_pre_decoder_fused.json":
        return "pre_decoder_fused"
    if name == "stage_post_decoder_fused.json":
        return "post_decoder_fused"
    prefix = "stage_decoder_layer_"
    suffix = ".json"
    if name.startswith(prefix) and name.endswith(suffix):
        return f"decoder_layer_{int(name[len(prefix) : -len(suffix)])}"
    raise AssertionError(f"Unexpected DFlash stage fixture path: {rel_path}")
