# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-light PyTorch reference for Laguna-XS-2.1 DFlash.

This is deliberately not a serving integration.  It is an executable contract for
the operations that a TT DFlash implementation must reproduce.  The draft
checkpoint does not own token embeddings or an LM head; callers pass the target
model's weights to :meth:`embed_input_ids` and :meth:`compute_logits`.

The implementation is single-request and inference-only.  It can load just a
subset of the five draft layers, which keeps CPU qualification fast while using
the real checkpoint tensors.  Loading all layers gives the complete reference
forward.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

_DFLASH_REVISION = "5c36361aab23c8ed3afbd079c10c426b677bc607"
DFLASH_TARGET_LAYER_IDS = (1, 13, 25, 33, 39)
_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DEFAULT_DFLASH_SNAPSHOT = _HF_HOME / "hub" / "models--poolside--Laguna-XS-2.1-DFlash" / "snapshots" / _DFLASH_REVISION


@dataclass(frozen=True)
class LagunaDFlashConfig:
    """The Laguna-specific subset of the published DFlash config."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    draft_vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    hidden_act: str
    attention_bias: bool
    gating: str
    num_experts: int
    architectures: tuple[str, ...]
    torch_dtype: str
    layer_types: tuple[str, ...]
    aux_hidden_state_layer_ids: tuple[int, ...]
    target_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int
    causal: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "LagunaDFlashConfig":
        path = Path(path)
        with path.open(encoding="utf-8") as config_file:
            raw = json.load(config_file)
        dflash = raw.get("dflash_config") or {}
        config = cls(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            vocab_size=int(raw["vocab_size"]),
            draft_vocab_size=int(raw["draft_vocab_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            sliding_window=int(raw["sliding_window"]),
            hidden_act=str(raw["hidden_act"]),
            attention_bias=bool(raw["attention_bias"]),
            gating=str(raw["gating"]),
            num_experts=int(raw["num_experts"]),
            architectures=tuple(raw["architectures"]),
            torch_dtype=str(raw["torch_dtype"]),
            layer_types=tuple(raw["layer_types"]),
            aux_hidden_state_layer_ids=tuple(int(i) for i in raw["eagle_aux_hidden_state_layer_ids"]),
            target_layer_ids=tuple(int(i) for i in dflash["target_layer_ids"]),
            block_size=int(dflash["block_size"]),
            mask_token_id=int(dflash["mask_token_id"]),
            causal=bool(dflash["causal"]),
        )
        config.validate()
        return config

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def fused_qkv_size(self) -> int:
        return self.q_size + 2 * self.kv_size

    @property
    def num_aux_hidden_states(self) -> int:
        return len(self.target_layer_ids)

    @property
    def max_speculative_tokens(self) -> int:
        # One query slot is the already-known bonus/anchor token.
        return self.block_size - 1

    def validate(self) -> None:
        geometry = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.sliding_window,
            self.block_size,
            self.mask_token_id,
        )
        published_geometry = (2048, 8192, 5, 64, 8, 128, 100352, 262144, 512, 16, 12)
        if geometry != published_geometry:
            raise ValueError(
                "config does not match the published Laguna-XS-2.1 DFlash geometry: "
                f"got {geometry}, expected {published_geometry}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DFlash query heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per draft layer")
        if set(self.layer_types) != {"sliding_attention"}:
            raise ValueError("Laguna DFlash requires five uniform sliding-attention layers")
        if self.gating != "per-head":
            raise ValueError("Laguna DFlash requires softplus per-head attention gating")
        if self.hidden_act != "silu" or self.attention_bias or self.num_experts != 0:
            raise ValueError("Laguna DFlash requires bias-free attention and a dense SwiGLU MLP")
        if self.architectures != ("DFlashLagunaForCausalLM",):
            raise ValueError(f"unexpected Laguna DFlash architecture: {self.architectures}")
        if self.torch_dtype != "bfloat16":
            raise ValueError(f"the published Laguna DFlash checkpoint must be BF16, got {self.torch_dtype}")
        if not self.causal:
            raise ValueError("the published Laguna DFlash checkpoint uses causal attention")
        if len(self.aux_hidden_state_layer_ids) != len(self.target_layer_ids):
            raise ValueError("aux-hidden and target-layer ID counts must match")
        if tuple(i + 1 for i in self.target_layer_ids) != self.aux_hidden_state_layer_ids:
            raise ValueError("DFlash target layer IDs must map to post-layer hidden-state IDs with +1 indexing")
        if self.target_layer_ids != (1, 13, 25, 33, 39):
            raise ValueError(f"unexpected Laguna DFlash target layer IDs: {self.target_layer_ids}")
        if not math.isclose(self.rope_theta, 500_000.0) or not math.isclose(self.rms_norm_eps, 1e-6):
            raise ValueError("unexpected Laguna DFlash RoPE theta or RMSNorm epsilon")
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError("Laguna DFlash must share the target vocabulary and LM head")


def expected_checkpoint_shapes(config: LagunaDFlashConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact unsharded checkpoint layout used by this reference."""

    h = config.hidden_size
    i = config.intermediate_size
    shapes: dict[str, tuple[int, ...]] = {
        **{f"aux_hidden_norms.{idx}.weight": (h,) for idx in range(config.num_aux_hidden_states)},
        "fc.weight": (h, config.num_aux_hidden_states * h),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
    }
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (h,),
                f"{prefix}.self_attn.qkv_proj.weight": (config.fused_qkv_size, h),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.g_proj.weight": (config.num_attention_heads, h),
                f"{prefix}.self_attn.o_proj.weight": (h, config.q_size),
                f"{prefix}.post_attention_layernorm.weight": (h,),
                f"{prefix}.mlp.gate_proj.weight": (i, h),
                f"{prefix}.mlp.up_proj.weight": (i, h),
                f"{prefix}.mlp.down_proj.weight": (h, i),
            }
        )
    return shapes


class LagunaDFlashCheckpoint:
    """Config and safetensors loader with strict shape/name validation."""

    def __init__(self, snapshot: str | Path = DEFAULT_DFLASH_SNAPSHOT):
        self.snapshot = Path(snapshot)
        self.config_path = self.snapshot / "config.json"
        self.weights_path = self.snapshot / "model.safetensors"
        if not self.config_path.is_file() or not self.weights_path.is_file():
            raise FileNotFoundError(f"incomplete Laguna DFlash snapshot: {self.snapshot}")
        self.config = LagunaDFlashConfig.from_json(self.config_path)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        from safetensors import safe_open

        with safe_open(self.weights_path, framework="pt", device="cpu") as weights:
            return {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}

    def tensor_dtypes(self) -> dict[str, str]:
        from safetensors import safe_open

        with safe_open(self.weights_path, framework="pt", device="cpu") as weights:
            return {name: weights.get_slice(name).get_dtype() for name in weights.keys()}

    def validate_layout(self) -> None:
        expected = expected_checkpoint_shapes(self.config)
        actual = self.tensor_shapes()
        dtypes = self.tensor_dtypes()
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        mismatched = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
        wrong_dtype = sorted(name for name, dtype in dtypes.items() if dtype != "BF16")
        if missing or unexpected or mismatched or wrong_dtype:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            if mismatched:
                details.append("shape_mismatch=" + str({name: (actual[name], expected[name]) for name in mismatched}))
            if wrong_dtype:
                details.append(f"non_bf16={wrong_dtype}")
            raise ValueError("invalid Laguna DFlash checkpoint layout: " + "; ".join(details))

    def load_tensors(
        self,
        names: Iterable[str],
        *,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Materialize selected CPU tensors without copying unrelated layers."""

        from safetensors import safe_open

        names = tuple(dict.fromkeys(names))
        with safe_open(self.weights_path, framework="pt", device="cpu") as weights:
            available = set(weights.keys())
            missing = sorted(set(names) - available)
            if missing:
                raise KeyError(f"checkpoint tensors not found: {missing}")
            loaded = {name: weights.get_tensor(name) for name in names}
        if dtype is not None:
            loaded = {name: tensor.to(dtype=dtype) for name, tensor in loaded.items()}
        return loaded

    def load_reference(
        self,
        *,
        layer_indices: Sequence[int] | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LagunaDFlashReference":
        """Load a complete model, or selected layers for an isolated CPU check."""

        self.validate_layout()
        if layer_indices is None:
            layer_indices = tuple(range(self.config.num_hidden_layers))
        layer_indices = tuple(int(index) for index in layer_indices)
        if not layer_indices:
            raise ValueError("at least one draft layer is required")
        if len(set(layer_indices)) != len(layer_indices) or tuple(sorted(layer_indices)) != layer_indices:
            raise ValueError("layer_indices must be unique and increasing")
        if layer_indices[0] < 0 or layer_indices[-1] >= self.config.num_hidden_layers:
            raise ValueError(f"layer index outside [0, {self.config.num_hidden_layers})")

        expected = expected_checkpoint_shapes(self.config)
        shared_names = [
            *(f"aux_hidden_norms.{idx}.weight" for idx in range(self.config.num_aux_hidden_states)),
            "fc.weight",
            "hidden_norm.weight",
            "norm.weight",
        ]
        layer_names = [
            name for name in expected if any(name.startswith(f"layers.{layer_idx}.") for layer_idx in layer_indices)
        ]
        tensors = self.load_tensors((*shared_names, *layer_names), dtype=dtype)
        return LagunaDFlashReference(self.config, tensors, layer_indices=layer_indices)


@dataclass(frozen=True)
class DFlashProposalBlock:
    """One Laguna DFlash parallel query block.

    Slot zero contains the known target bonus token.  Remaining slots contain
    token 12 (the shared target embedding row used as the mask embedding), and
    those rows directly predict their own absolute positions.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    sample_indices: torch.Tensor
    sample_positions: torch.Tensor


@dataclass(frozen=True)
class DFlashDraftArgmaxAccuracy:
    """Host result for TT draft IDs against official raw BF16 logits.

    A unique official maximum requires literal argmax equality.  An exact
    BF16 tie has no numerically preferred token, so the TT selection must be a
    member of the complete exact maximum set.  Target verification and final
    committed tokens are intentionally outside this draft-only contract and
    remain literally exact.
    """

    tt_ids: tuple[int, ...]
    reference_ids: tuple[int, ...]
    tied_rows: tuple[int, ...]
    tied_maximum_ids: tuple[tuple[int, ...], ...]
    non_tied_exact: bool
    tied_membership: bool

    @property
    def passed(self) -> bool:
        return bool(self.non_tied_exact and self.tied_membership)

    @property
    def literal_exact(self) -> bool:
        return self.tt_ids == self.reference_ids


def evaluate_dflash_draft_argmax_accuracy(
    tt_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> DFlashDraftArgmaxAccuracy:
    """Evaluate exact non-tied IDs and exact-tie-set membership.

    Both inputs are raw BF16 draft logits with shape ``[rows, vocab]``.  The
    dtype requirement matters: widening logits and comparing approximately
    would silently invent or erase ties in the official inference contract.
    """

    if tt_logits.ndim != 2 or reference_logits.ndim != 2:
        raise ValueError("DFlash draft accuracy logits must both have shape [rows, vocab]")
    if tuple(tt_logits.shape) != tuple(reference_logits.shape):
        raise ValueError(
            f"DFlash TT/reference logit shapes differ: {tuple(tt_logits.shape)} != " f"{tuple(reference_logits.shape)}"
        )
    if not tt_logits.shape[0] or not tt_logits.shape[1]:
        raise ValueError("DFlash draft accuracy logits must be non-empty")
    if tt_logits.dtype != torch.bfloat16 or reference_logits.dtype != torch.bfloat16:
        raise TypeError(
            "DFlash draft accuracy requires raw BF16 TT and official-reference logits; "
            f"got {tt_logits.dtype} and {reference_logits.dtype}"
        )
    if not bool(torch.isfinite(tt_logits).all()) or not bool(torch.isfinite(reference_logits).all()):
        raise ValueError("DFlash draft accuracy logits must be finite")

    tt_ids_tensor = torch.argmax(tt_logits, dim=-1)
    reference_ids_tensor = torch.argmax(reference_logits, dim=-1)
    tt_ids = tuple(int(token) for token in tt_ids_tensor.tolist())
    reference_ids = tuple(int(token) for token in reference_ids_tensor.tolist())
    tied_rows: list[int] = []
    tied_maximum_ids: list[tuple[int, ...]] = []
    non_tied_exact = True
    tied_membership = True
    for row_index in range(int(reference_logits.shape[0])):
        row = reference_logits[row_index]
        maximum_ids = tuple(int(token) for token in torch.nonzero(row == torch.max(row), as_tuple=False).flatten())
        if len(maximum_ids) == 1:
            non_tied_exact &= tt_ids[row_index] == maximum_ids[0]
            continue
        tied_rows.append(row_index)
        tied_maximum_ids.append(maximum_ids)
        tied_membership &= tt_ids[row_index] in maximum_ids

    return DFlashDraftArgmaxAccuracy(
        tt_ids=tt_ids,
        reference_ids=reference_ids,
        tied_rows=tuple(tied_rows),
        tied_maximum_ids=tuple(tied_maximum_ids),
        non_tied_exact=bool(non_tied_exact),
        tied_membership=bool(tied_membership),
    )


@dataclass(frozen=True)
class DFlashTargetAuxCapture:
    """Five post-target-layer states for a contiguous logical token interval.

    ``hidden_states`` is intentionally an opaque tensor-like object.  The CPU
    reference uses a torch tensor while the serving implementation uses a TT
    tensor with the identical ``[1, rows, 5 * hidden]`` flattened contract.
    Keeping positions as scalar metadata avoids a device-to-host readback in the
    target forward path.
    """

    hidden_states: object
    start_position: int
    row_count: int
    layer_ids: tuple[int, ...] = DFLASH_TARGET_LAYER_IDS

    @property
    def end_position(self) -> int:
        return int(self.start_position) + int(self.row_count) - 1

    def validate(self, config: LagunaDFlashConfig) -> None:
        config.validate()
        if tuple(self.layer_ids) != tuple(config.target_layer_ids):
            raise ValueError(
                f"DFlash auxiliary capture layers {tuple(self.layer_ids)} do not match "
                f"checkpoint target layers {tuple(config.target_layer_ids)}"
            )
        rows = int(self.row_count)
        if rows < 1 or rows > config.sliding_window - 1:
            raise ValueError(f"DFlash auxiliary capture must retain 1..{config.sliding_window - 1} rows, got {rows}")
        shape = tuple(int(dim) for dim in self.hidden_states.shape)
        expected = (1, rows, config.num_aux_hidden_states * config.hidden_size)
        if shape != expected:
            raise ValueError(f"DFlash auxiliary capture shape {shape} does not match {expected}")
        if int(self.start_position) < 0:
            raise ValueError(f"DFlash auxiliary capture starts before position zero: {self.start_position}")


def retain_dflash_context_window(
    config: LagunaDFlashConfig,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain exactly the useful target rows for the next draft query.

    A 512-token inclusive sliding window can see at most 511 already-committed
    context rows because the anchor query itself occupies the final slot.  Rows
    older than that can never affect any anchor+mask output and are discarded.
    """

    if hidden_states.ndim < 2:
        raise ValueError("DFlash context hidden states must have a token dimension")
    if positions.ndim != 1 or positions.numel() != hidden_states.shape[0]:
        raise ValueError("DFlash context positions must contain one entry per hidden row")
    if positions.numel() and not bool((positions[1:] == positions[:-1] + 1).all()):
        raise ValueError("DFlash context positions must be strictly contiguous")
    keep = min(int(hidden_states.shape[0]), config.sliding_window - 1)
    return hidden_states[-keep:], positions[-keep:]


def build_proposal_block(
    config: LagunaDFlashConfig,
    *,
    bonus_token_id: int,
    last_valid_position: int,
    num_speculative_tokens: int | None = None,
) -> DFlashProposalBlock:
    """Build the ``1 + N`` anchor/mask query geometry used by vLLM DFlash."""

    if num_speculative_tokens is None:
        num_speculative_tokens = config.max_speculative_tokens
    num_speculative_tokens = int(num_speculative_tokens)
    if not 1 <= num_speculative_tokens <= config.max_speculative_tokens:
        raise ValueError(
            f"num_speculative_tokens must be in [1, {config.max_speculative_tokens}], " f"got {num_speculative_tokens}"
        )
    if not 0 <= bonus_token_id < config.vocab_size:
        raise ValueError(f"bonus token ID outside target vocabulary: {bonus_token_id}")
    query_count = 1 + num_speculative_tokens
    input_ids = torch.full((query_count,), config.mask_token_id, dtype=torch.int64)
    input_ids[0] = int(bonus_token_id)
    positions = int(last_valid_position) + 1 + torch.arange(query_count, dtype=torch.int64)
    sample_indices = torch.arange(1, query_count, dtype=torch.int64)
    return DFlashProposalBlock(
        input_ids=input_ids,
        positions=positions,
        sample_indices=sample_indices,
        sample_positions=positions[1:].clone(),
    )


@dataclass(frozen=True)
class LayerContextKV:
    key: torch.Tensor  # [context, num_kv_heads, head_dim], QK-normalized + RoPE
    value: torch.Tensor  # [context, num_kv_heads, head_dim]
    positions: torch.Tensor  # [context]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with float32 variance accumulation, matching vLLM/HF inference."""

    if x.shape[-1] != weight.numel():
        raise ValueError(f"RMSNorm width {x.shape[-1]} does not match weight width {weight.numel()}")
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=x.dtype) * weight.to(dtype=x.dtype)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's fused residual-add + RMSNorm rounding contract.

    Normalization consumes the float32 sum, while the residual carried to the
    next block is rounded to the activation dtype.  Keeping this distinction
    avoids a small but systematic error versus collapsing the layer to
    ``rms_norm((x + residual).to(bfloat16))``.
    """

    if x.shape != residual.shape:
        raise ValueError("fused add inputs must have matching shapes")
    if x.shape[-1] != weight.numel():
        raise ValueError("fused RMSNorm weight width does not match the activation")
    summed = x.float() + residual.float()
    carried_residual = summed.to(dtype=x.dtype)
    normalized = summed * torch.rsqrt(summed.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = normalized.to(dtype=weight.dtype) * weight
    return normalized.to(dtype=x.dtype), carried_residual


def split_fused_qkv(qkv: torch.Tensor, config: LagunaDFlashConfig) -> tuple[torch.Tensor, ...]:
    """Split the checkpoint's row-concatenated ``[Q, K, V]`` projection."""

    if qkv.shape[-1] != config.fused_qkv_size:
        raise ValueError(f"fused QKV width must be {config.fused_qkv_size}, got {qkv.shape[-1]}")
    return qkv.split((config.q_size, config.kv_size, config.kv_size), dim=-1)


def apply_neox_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
) -> torch.Tensor:
    """Apply full-dimension NeoX RoPE to ``[sequence, heads, head_dim]``."""

    if x.ndim != 3 or x.shape[-1] % 2:
        raise ValueError("RoPE input must be [sequence, heads, even head_dim]")
    if positions.ndim != 1 or positions.shape[0] != x.shape[0]:
        raise ValueError("RoPE positions must contain one entry per sequence row")
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) * 2 / x.shape[-1]))
    phase = torch.outer(positions.to(device=x.device, dtype=torch.float32), inv_freq)
    cos = phase.cos()[:, None, :]
    sin = phase.sin()[:, None, :]
    left, right = x.float().split(half, dim=-1)
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
    return rotated.to(dtype=x.dtype)


def causal_sliding_attention(
    q: torch.Tensor,
    query_k: torch.Tensor,
    query_v: torch.Tensor,
    query_positions: torch.Tensor,
    context: LayerContextKV,
    *,
    sliding_window: int,
) -> torch.Tensor:
    """Single-request causal GQA over pre-inserted context plus query K/V.

    A window of 512 includes the current key and the preceding 511 absolute
    positions.  Query rows additionally use a lower-triangular order mask, so a
    later mask slot is never visible even if callers provide repeated positions.
    """

    if q.ndim != 3 or query_k.ndim != 3 or query_v.ndim != 3:
        raise ValueError("attention tensors must have [sequence, heads, head_dim] shape")
    if query_k.shape != query_v.shape:
        raise ValueError("query K and V shapes must match")
    if context.key.shape != context.value.shape:
        raise ValueError("context K and V shapes must match")
    if q.shape[0] != query_k.shape[0] or q.shape[0] != query_positions.numel():
        raise ValueError("query tensor and position lengths must match")
    if query_k.shape[1] != context.key.shape[1] or query_k.shape[2] != context.key.shape[2]:
        raise ValueError("context and query KV geometry must match")
    if q.shape[1] % query_k.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")

    repeat = q.shape[1] // query_k.shape[1]
    keys = torch.cat((context.key, query_k), dim=0).repeat_interleave(repeat, dim=1)
    values = torch.cat((context.value, query_v), dim=0).repeat_interleave(repeat, dim=1)
    key_positions = torch.cat((context.positions.to(query_positions.device), query_positions))

    query_count = q.shape[0]
    context_count = context.key.shape[0]
    causal_by_position = key_positions[None, :] <= query_positions[:, None]
    in_window = key_positions[None, :] >= (query_positions[:, None] - (sliding_window - 1))
    query_key_order = (
        torch.arange(query_count, device=q.device)[None, :] <= torch.arange(query_count, device=q.device)[:, None]
    )
    key_order = torch.cat(
        (torch.ones((query_count, context_count), dtype=torch.bool, device=q.device), query_key_order),
        dim=1,
    )
    allowed = causal_by_position & in_window & key_order
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("at least one attention query has no visible key")

    scores = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~allowed[None, :, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
    return torch.einsum("hqk,khd->qhd", probabilities, values)


class LagunaDFlashReference:
    """Functional pure-PyTorch Laguna DFlash model using published weights."""

    def __init__(
        self,
        config: LagunaDFlashConfig,
        weights: Mapping[str, torch.Tensor],
        *,
        layer_indices: Sequence[int],
    ):
        self.config = config
        self.weights = dict(weights)
        self.layer_indices = tuple(layer_indices)
        first_weight = next(iter(self.weights.values()))
        self.dtype = first_weight.dtype
        if any(weight.device.type != "cpu" for weight in self.weights.values()):
            raise ValueError("the lightweight reference is CPU-only")
        if any(weight.dtype != self.dtype for weight in self.weights.values()):
            raise ValueError("all reference weights must have a common dtype")

    def _w(self, name: str) -> torch.Tensor:
        try:
            return self.weights[name]
        except KeyError as error:
            raise KeyError(f"reference weight was not loaded: {name}") from error

    def combine_aux_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize five target hidden slices, concatenate, project, then norm.

        Accepted shapes are ``[tokens, 5, 2048]`` and ``[tokens, 10240]``.
        The slice order is exactly ``target_layer_ids=(1, 13, 25, 33, 39)``.
        """

        h = self.config.hidden_size
        count = self.config.num_aux_hidden_states
        if hidden_states.ndim == 2 and hidden_states.shape[-1] == count * h:
            slices = hidden_states.reshape(hidden_states.shape[0], count, h)
        elif hidden_states.ndim == 3 and hidden_states.shape[-2:] == (count, h):
            slices = hidden_states
        else:
            raise ValueError(f"aux hidden states must be [tokens, {count}, {h}] or [tokens, {count * h}]")
        slices = slices.to(dtype=self.dtype)
        normalized = [
            rms_norm(slices[:, index, :], self._w(f"aux_hidden_norms.{index}.weight"), self.config.rms_norm_eps)
            for index in range(count)
        ]
        combined = F.linear(torch.cat(normalized, dim=-1), self._w("fc.weight"))
        return rms_norm(combined, self._w("hidden_norm.weight"), self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor, target_embedding_weight: torch.Tensor) -> torch.Tensor:
        """Embed draft queries with the target-owned embedding table."""

        if target_embedding_weight.ndim != 2 or target_embedding_weight.shape[1] != self.config.hidden_size:
            raise ValueError(f"target embedding must have shape [vocab, {self.config.hidden_size}]")
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= target_embedding_weight.shape[0]):
            raise ValueError("input ID is outside the supplied target embedding shard/table")
        return F.embedding(input_ids, target_embedding_weight).to(dtype=self.dtype)

    def compute_logits(self, hidden_states: torch.Tensor, target_lm_head_weight: torch.Tensor) -> torch.Tensor:
        """Project with the target-owned LM head (the draft checkpoint has none)."""

        if target_lm_head_weight.ndim != 2 or target_lm_head_weight.shape[1] != self.config.hidden_size:
            raise ValueError(f"target LM head must have shape [vocab, {self.config.hidden_size}]")
        return F.linear(hidden_states.to(dtype=target_lm_head_weight.dtype), target_lm_head_weight)

    def precompute_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> dict[int, LayerContextKV]:
        """Project verifier context into each draft layer's K/V representation.

        Laguna differs from generic DFlash here: each layer first applies that
        layer's *input* RMSNorm, then takes K/V rows from its fused QKV matrix.
        K receives per-head RMSNorm and RoPE; V receives neither.
        """

        if context_states.ndim != 2 or context_states.shape[-1] != self.config.hidden_size:
            raise ValueError(f"context states must be [tokens, {self.config.hidden_size}]")
        if context_positions.ndim != 1 or context_positions.numel() != context_states.shape[0]:
            raise ValueError("context positions must contain one entry per context state")
        context_states = context_states.to(dtype=self.dtype)
        result: dict[int, LayerContextKV] = {}
        for layer_idx in self.layer_indices:
            prefix = f"layers.{layer_idx}"
            normalized = rms_norm(
                context_states,
                self._w(f"{prefix}.input_layernorm.weight"),
                self.config.rms_norm_eps,
            )
            qkv = F.linear(normalized, self._w(f"{prefix}.self_attn.qkv_proj.weight"))
            _, key, value = split_fused_qkv(qkv, self.config)
            key = key.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
            value = value.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
            key = rms_norm(key, self._w(f"{prefix}.self_attn.k_norm.weight"), self.config.rms_norm_eps)
            key = apply_neox_rope(key, context_positions, theta=self.config.rope_theta)
            result[layer_idx] = LayerContextKV(key=key, value=value, positions=context_positions.clone())
        return result

    def _attention(
        self,
        layer_idx: int,
        normalized_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        context: LayerContextKV,
    ) -> torch.Tensor:
        prefix = f"layers.{layer_idx}.self_attn"
        qkv = F.linear(normalized_hidden_states, self._w(f"{prefix}.qkv_proj.weight"))
        query, key, value = split_fused_qkv(qkv, self.config)
        query = query.reshape(-1, self.config.num_attention_heads, self.config.head_dim)
        key = key.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        value = value.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        query = rms_norm(query, self._w(f"{prefix}.q_norm.weight"), self.config.rms_norm_eps)
        key = rms_norm(key, self._w(f"{prefix}.k_norm.weight"), self.config.rms_norm_eps)
        query = apply_neox_rope(query, positions, theta=self.config.rope_theta)
        key = apply_neox_rope(key, positions, theta=self.config.rope_theta)
        attention = causal_sliding_attention(
            query,
            key,
            value,
            positions,
            context,
            sliding_window=self.config.sliding_window,
        )

        # The published checkpoint has one scalar gate per query head.  Softplus
        # is evaluated in float32, then broadcast across all 128 head channels.
        gate = F.softplus(F.linear(normalized_hidden_states, self._w(f"{prefix}.g_proj.weight")).float())
        attention = attention * gate.to(dtype=attention.dtype).unsqueeze(-1)
        return F.linear(attention.reshape(-1, self.config.q_size), self._w(f"{prefix}.o_proj.weight"))

    def _mlp(self, layer_idx: int, normalized_hidden_states: torch.Tensor) -> torch.Tensor:
        prefix = f"layers.{layer_idx}.mlp"
        gate = F.linear(normalized_hidden_states, self._w(f"{prefix}.gate_proj.weight"))
        up = F.linear(normalized_hidden_states, self._w(f"{prefix}.up_proj.weight"))
        return F.linear(F.silu(gate) * up, self._w(f"{prefix}.down_proj.weight"))

    def _forward_query_embeddings_impl(
        self,
        query_embeddings: torch.Tensor,
        query_positions: torch.Tensor,
        context_kv: Mapping[int, LayerContextKV],
        *,
        collect_layer_outputs: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run draft layers, optionally retaining rounded post-layer states."""

        if query_embeddings.ndim != 2 or query_embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(f"query embeddings must be [tokens, {self.config.hidden_size}]")
        if query_positions.ndim != 1 or query_positions.numel() != query_embeddings.shape[0]:
            raise ValueError("query positions must contain one entry per query embedding")
        hidden_states = query_embeddings.to(dtype=self.dtype)
        residual: torch.Tensor | None = None
        layer_outputs: list[torch.Tensor] = []
        for layer_idx in self.layer_indices:
            if layer_idx not in context_kv:
                raise KeyError(f"missing context K/V for draft layer {layer_idx}")
            prefix = f"layers.{layer_idx}"
            if residual is None:
                residual = hidden_states
                normalized = rms_norm(
                    hidden_states,
                    self._w(f"{prefix}.input_layernorm.weight"),
                    self.config.rms_norm_eps,
                )
            else:
                normalized, residual = fused_add_rms_norm(
                    hidden_states,
                    residual,
                    self._w(f"{prefix}.input_layernorm.weight"),
                    self.config.rms_norm_eps,
                )
            attention = self._attention(
                layer_idx,
                normalized,
                query_positions,
                context_kv[layer_idx],
            )
            normalized, residual = fused_add_rms_norm(
                attention,
                residual,
                self._w(f"{prefix}.post_attention_layernorm.weight"),
                self.config.rms_norm_eps,
            )
            hidden_states = self._mlp(layer_idx, normalized)
            if collect_layer_outputs:
                # This is the materialized BF16 state corresponding to a
                # conventional decoder layer's output.  The reference keeps
                # ``hidden_states`` and ``residual`` split internally so the
                # next fused RMSNorm consumes their unrounded FP32 sum.
                layer_outputs.append((hidden_states.float() + residual.float()).to(dtype=self.dtype))
        assert residual is not None
        hidden_states, _ = fused_add_rms_norm(
            hidden_states,
            residual,
            self._w("norm.weight"),
            self.config.rms_norm_eps,
        )
        return hidden_states, tuple(layer_outputs)

    def forward_query_embeddings(
        self,
        query_embeddings: torch.Tensor,
        query_positions: torch.Tensor,
        context_kv: Mapping[int, LayerContextKV],
    ) -> torch.Tensor:
        """Run the selected draft layers over one anchor/mask query block."""

        hidden_states, _ = self._forward_query_embeddings_impl(
            query_embeddings,
            query_positions,
            context_kv,
            collect_layer_outputs=False,
        )
        return hidden_states

    def forward_query_embeddings_with_layer_outputs(
        self,
        query_embeddings: torch.Tensor,
        query_positions: torch.Tensor,
        context_kv: Mapping[int, LayerContextKV],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the final hidden state and five device-comparison stages.

        This default-off diagnostic API does not alter the official forward
        path.  Each retained stage is the BF16-rounded ``MLP + residual``
        state after one layer; the live reference still carries the two terms
        separately to preserve the fused-add RMSNorm rounding contract.
        """

        return self._forward_query_embeddings_impl(
            query_embeddings,
            query_positions,
            context_kv,
            collect_layer_outputs=True,
        )

    def proposal_logits(
        self,
        block: DFlashProposalBlock,
        *,
        target_embedding_weight: torch.Tensor,
        target_lm_head_weight: torch.Tensor,
        context_aux_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run a complete reference proposal and return only the N sampled rows."""

        context_aux_hidden_states, context_positions = retain_dflash_context_window(
            self.config,
            context_aux_hidden_states,
            context_positions,
        )
        context_states = self.combine_aux_hidden_states(context_aux_hidden_states)
        context_kv = self.precompute_context_kv(context_states, context_positions)
        query_embeddings = self.embed_input_ids(block.input_ids, target_embedding_weight)
        hidden_states = self.forward_query_embeddings(query_embeddings, block.positions, context_kv)
        return self.compute_logits(hidden_states[block.sample_indices], target_lm_head_weight)


__all__ = [
    "DEFAULT_DFLASH_SNAPSHOT",
    "DFLASH_TARGET_LAYER_IDS",
    "DFlashDraftArgmaxAccuracy",
    "DFlashProposalBlock",
    "DFlashTargetAuxCapture",
    "LagunaDFlashCheckpoint",
    "LagunaDFlashConfig",
    "LagunaDFlashReference",
    "LayerContextKV",
    "apply_neox_rope",
    "build_proposal_block",
    "causal_sliding_attention",
    "evaluate_dflash_draft_argmax_accuracy",
    "expected_checkpoint_shapes",
    "fused_add_rms_norm",
    "retain_dflash_context_window",
    "rms_norm",
    "split_fused_qkv",
]
