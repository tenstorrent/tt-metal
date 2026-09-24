# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter loader for on-device fuse-mode LoRA on the MiniMax-H3 transformer.

Registers a LoRA safetensors file's A/B pairs into the LoRA-aware Linear modules of a
``MiniMaxH3Transformer3DModel``; ``bind_active`` then merges ``scale * B@A`` into the device weight,
so the forward path stays a plain matmul and costs nothing per step.

Why this differs from the Wan and LTX loaders
---------------------------------------------
Same shape of problem -- the on-device weight is already permuted and fused, so a LoRA ``B`` must be
pre-transformed identically or its delta lands in the wrong rows -- but H3 applies three transforms
where LTX applies two:

  - ``rope_channel_permutation``: Q and K only. Imported from the model rather than re-derived here,
    because a second copy of a channel permutation is a silent wrong-answer waiting to happen.
  - ``_interleave_heads``: folds Q/K/V into the fused ``to_qkv`` with heads interleaved so column
    parallel sharding lands device ``d``'s q, k and v heads contiguously.
  - ``prepare_for_fused_swiglu``: ``ff1`` packs ``[gate|up]`` into interleaved tile pairs. LTX's FFNs
    need no transform; H3's does, and forgetting it swaps which half gets the silu.

``to_out`` and ``ff2`` are untouched by ``_prepare_torch_state``, so their adapters register directly.

``LoRAMixin.register_lora`` takes ``B`` already in the destination weight's layout and does not
transform it, so every transform above has to happen here. Its swiglu guard does not catch ``ff1``
(``ColParallelLinear`` clears ``activation_fn`` once it sets ``fuse_swiglu``), which is why the pack
below is mandatory rather than belt-and-braces.

Scale
-----
``alpha / rank``, taken from a per-target ``.alpha`` tensor when the file carries one (kohya) and
otherwise from the file's ``__metadata__`` (the diffusers publishes of lightx2v's Turbo adapters put
it there). Absent from both, the scale is 1 -- never ``alpha == rank``, which would silently rescale
an adapter that publishes a plain ``W + B@A`` contract.

``.diff`` / ``.diff_b`` direct deltas are unsupported; unmapped keys are raised, never skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from loguru import logger
from safetensors import safe_open

from ...models.transformers.minimax_h3.attention_minimax_h3 import rope_channel_permutation
from ...utils.tensor import prepare_for_fused_swiglu
from .promote import promote_to_lora

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")

# PEFT interposes the adapter name (``.default``) before ``.weight``; kohya spells the slots
# ``lora_down``/``lora_up``. Both reach the same pair.
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}

# The two block stacks an H3 adapter can address, mapped to their attribute path on the transformer.
_BLOCK_RE = re.compile(r"^(?P<stack>transformer_blocks|token_refiner\.refiner_blocks)\.(?P<idx>\d+)\.(?P<rest>.+)$")

# Leaf names as the adapter spells them, against the attribute that holds them on device.
_QKV_SUBS = {"attn.to_q": "q", "attn.to_k": "k", "attn.to_v": "v"}
_SINGLETONS = {
    "attn.to_out.0": ("attn", "to_out", None),
    "ff.net.0.proj": ("ff", "ff1", "swiglu"),
    "ff.net.2": ("ff", "ff2", None),
}


@dataclass
class H3AdapterHandle:
    """Bank indices assigned per module path, so a caller can bind or drop the whole adapter."""

    name: str
    indices: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)


def load_h3_adapter_into(transformer, path: str, *, scale: float = 1.0, name: str = "") -> H3AdapterHandle:
    """Register one adapter file into ``transformer`` and bind it.

    ``scale`` multiplies the adapter's own ``alpha / rank``; it is the caller's strength knob, not a
    substitute for the published scale.
    """
    name = name or path
    raw, metadata = _read(path)
    pairs, alphas = _collect_pairs(raw)
    if not pairs:
        raise RuntimeError(f"no LoRA-style keys (lora_A/lora_B, lora_down/lora_up) in {path}")

    file_alpha = _file_alpha(metadata)
    promoted = promote_to_lora(transformer)

    handle = H3AdapterHandle(name=name)
    bindings: list[tuple[object, int]] = []
    fused: dict[tuple[str, int], dict[str, dict[str, torch.Tensor]]] = {}
    unmapped: list[str] = []

    for base, ab in sorted(pairs.items()):
        target = _parse_target(base)
        if target is None:
            unmapped.append(base)
            continue
        stack, idx, leaf = target
        if leaf in _QKV_SUBS:
            fused.setdefault((stack, idx), {})[_QKV_SUBS[leaf]] = ab
            continue
        module, attr, transform = _SINGLETONS[leaf]
        owner = _resolve(transformer, stack, idx, module)
        linear = getattr(owner, attr)
        b = ab["B"]
        if transform == "swiglu":
            b = _pack_swiglu_rows(b)
        eff = scale * _scale_of(base, ab, alphas, file_alpha)
        bank_idx = linear.register_lora(ab["A"], b, scale=eff, name=name)
        handle.indices[f"{stack}.{idx}.{module}.{attr}"] = bank_idx
        bindings.append((linear, bank_idx))

    for (stack, idx), qkvs in sorted(fused.items()):
        attn = _resolve(transformer, stack, idx, "attn")
        bank_idx = _register_fused(attn, qkvs, scale, name, alphas, file_alpha, stack, idx)
        handle.indices[f"{stack}.{idx}.attn.to_qkv"] = bank_idx
        bindings.append((attn.to_qkv, bank_idx))

    if unmapped:
        sample = ", ".join(unmapped[:5])
        more = "" if len(unmapped) <= 5 else f" (+{len(unmapped) - 5} more)"
        raise RuntimeError(f"{len(unmapped)} adapter target(s) have no H3 destination: {sample}{more}")

    for linear, bank_idx in bindings:
        linear.bind_active(bank_idx)
    logger.info(f"{name}: bound {len(handle)} LoRA targets over {promoted} promoted linears")
    return handle


def _read(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}, dict(handle.metadata() or {})


def _file_alpha(metadata: dict[str, str]) -> float | None:
    """The file-level alpha, which is where a diffusers publish puts it when the tensors do not."""
    raw = metadata.get("alpha") or metadata.get("training_alpha")
    return None if raw is None else float(raw)


def _collect_pairs(raw: dict[str, torch.Tensor]):
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    rejected: list[str] = []
    for key, tensor in raw.items():
        stripped = _strip_prefixes(key)
        match = _LOW_RANK_RE.match(stripped)
        if match:
            pairs.setdefault(match.group("base"), {})[_SLOT_MAP[match.group("slot")]] = tensor
            continue
        if stripped.endswith(".alpha"):
            alphas[stripped[: -len(".alpha")]] = float(tensor.item())
            continue
        rejected.append(key)
    if rejected:
        sample = ", ".join(rejected[:5])
        raise RuntimeError(f"{len(rejected)} key(s) match no LoRA convention: {sample}")
    incomplete = [base for base, ab in pairs.items() if {"A", "B"} - set(ab)]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} half pair(s), e.g. {incomplete[0]}")
    return pairs, alphas


def _strip_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _parse_target(base: str) -> tuple[str, int, str] | None:
    match = _BLOCK_RE.match(base)
    if match is None:
        return None
    leaf = match.group("rest")
    if leaf not in _QKV_SUBS and leaf not in _SINGLETONS:
        return None
    return match.group("stack"), int(match.group("idx")), leaf


def _resolve(transformer, stack: str, idx: int, module: str):
    blocks = (
        transformer.transformer_blocks if stack == "transformer_blocks" else transformer.token_refiner.refiner_blocks
    )
    return getattr(blocks[idx], module)


def _scale_of(base: str, ab: dict[str, torch.Tensor], alphas: dict[str, float], file_alpha: float | None) -> float:
    rank = ab["A"].shape[0]
    alpha = alphas.get(base, file_alpha)
    return 1.0 if alpha is None else alpha / rank


def _register_fused(attn, qkvs, scale, name, alphas, file_alpha, stack, idx) -> int:
    missing = [slot for slot in ("q", "k", "v") if slot not in qkvs]
    if missing:
        raise RuntimeError(f"{stack}.{idx}.attn is missing {missing}; q/k/v share one to_qkv and must arrive together")

    ranks = {qkvs[slot]["A"].shape[0] for slot in ("q", "k", "v")}
    if len(ranks) != 1:
        raise ValueError(f"{stack}.{idx}.attn: q/k/v ranks disagree {sorted(ranks)}")
    rank = ranks.pop()

    perm = rope_channel_permutation(attn.head_dim, attn.rotary_dim)
    a_fused = torch.cat([qkvs[slot]["A"] for slot in ("q", "k", "v")], dim=0)

    # Block-diagonal B over the stacked rank, so each source's delta only reaches its own columns.
    padded = []
    for position, slot in enumerate(("q", "k", "v")):
        b = qkvs[slot]["B"]
        if slot in ("q", "k"):
            b = _permute_rotary_rows(b, attn.num_heads, attn.head_dim, perm)
        block = torch.zeros(attn.inner_dim, 3 * rank, dtype=b.dtype)
        block[:, position * rank : (position + 1) * rank] = b
        padded.append(block)
    b_fused = _interleave_heads_rows(
        padded, attn.n_local_heads, attn.head_dim, attn.parallel_config.tensor_parallel.factor
    )

    alpha = alphas.get(f"{stack}.{idx}.attn.to_q", file_alpha)
    eff = scale * (1.0 if alpha is None else alpha / rank)
    return attn.to_qkv.register_lora(a_fused, b_fused, scale=eff, name=name)


def _permute_rotary_rows(tensor: torch.Tensor, num_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """Mirror of `_permute_rotary`: reorder each head's output channels. `tensor` is [num_heads*head_dim, C]."""
    return tensor.reshape(num_heads, head_dim, -1)[:, perm].reshape(tensor.shape)


def _interleave_heads_rows(tensors, n_local_heads: int, head_dim: int, n_dev: int) -> torch.Tensor:
    """Mirror of `MiniMaxH3Attention._prepare_torch_state._interleave_heads` on the row (out) axis."""
    transposed = [t.T for t in tensors]
    reshaped = [t.reshape(t.shape[0], n_dev, n_local_heads, head_dim) for t in transposed]
    merged = torch.cat(reshaped, dim=2)
    merged = merged.reshape(merged.shape[0], -1)
    return merged.T


def _pack_swiglu_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Mirror of `ColParallelLinear._prepare_torch_state`'s swiglu packing on the row (out) axis.

    `prepare_for_fused_swiglu` reorders the last axis, so the [2N, rank] B is transposed into the
    orientation the base weight has when it is packed and transposed back.
    """
    return prepare_for_fused_swiglu(tensor.T, ndev=1).T
