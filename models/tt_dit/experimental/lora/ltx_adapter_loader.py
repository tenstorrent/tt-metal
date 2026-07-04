# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter loader for on-device fuse-mode LoRA on the LTX-2.3 transformer.

Reads a LoRA safetensors file and registers its A/B pairs into the
LoRA-aware Linear modules of an ``LTXTransformerModel`` built with
``lora_enabled=True``. Swap is then a ``bind_active`` on each module
(``weight.data += scale * A@B`` on device) — no host fuse, no reload.

Why this differs from the Wan loader
-------------------------------------
LTX's ``LTXAttention._prepare_torch_state`` transforms the raw checkpoint
Q/K/V weights before they reach the device:

  - ``_permute_qk``: reorders each head's channels from the checkpoint's
    SPLIT rotary layout to the INTERLEAVED layout the RoPE op expects.
    Applied to Q and K (self-attn) and to the cross-attn ``to_q``.
  - ``_interleave_heads``: folds separate Q/K/V (or K/V) into the fused
    ``to_qkv`` / ``to_kv`` weight with heads interleaved so column-parallel
    sharding lands the right head on each device.

The host-fuse path (``utils.fuse_loras.fuse_loras_into``) is correct
without any of this because it merges the delta into the *raw* checkpoint
weight and lets ``_prepare_torch_state`` permute the merged result. Here
the on-device weight is ALREADY permuted+interleaved, so the LoRA B must
be pre-transformed identically or the merged delta lands in the wrong
layout. ``to_out``/``ff1``/``ff2`` are untouched by ``_prepare_torch_state``,
so their adapters need no transform (the mixin copies W's own sharding).

Scope matches the Wan loader: self-attn ``to_qkv``, cross-attn ``to_q`` +
``to_kv``, ``to_out``, and ``ffn`` (ff1/ff2) — the standard diffusers LoRA
target set. Audio / cross-modal (a2v/v2a) modules and ``.diff``/``.diff_b``
direct deltas are out of scope; unmapped keys are surfaced loudly, never
silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from ...layers.lora import LoRAMixin

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}

# Diffusers LTX checkpoints name blocks "transformer_blocks.<i>"; some exports
# use the shorter "blocks.<i>". Accept both.
_BLOCK_RE = r"(?:transformer_blocks|blocks)\.(\d+)"


@dataclass(frozen=True)
class LTXAdapterHandle:
    name: str
    target_indices: dict[str, int]
    """Map from a canonical tt-dit module path (e.g. ``transformer_blocks.0.attn1.to_qkv``)
    to the bank index returned by ``register_lora`` on that module."""


def load_ltx_adapter_into(
    transformer,  # LTXTransformerModel built with lora_enabled=True
    path: str,
    *,
    scale: float = 1.0,
    name: str = "",
) -> LTXAdapterHandle:
    """Register a single LoRA safetensors file into ``transformer``.

    Returns a handle recording the bank index assigned per module; pass it
    to the pipeline's ``set_active_lora`` to bind it on device.
    """
    raw = load_file(str(path))
    pairs, alphas, skipped_direct, unrecognized = _collect_pairs(raw)
    if not pairs:
        raise RuntimeError(f"no LoRA A/B pairs (lora_A/B or lora_down/up) found in {path}")
    if skipped_direct:
        logger.warning(f"{path}: skipping {skipped_direct} direct (.diff/.diff_b) deltas — not supported.")
    if unrecognized:
        sample = ", ".join(unrecognized[:5])
        more = "" if len(unrecognized) <= 5 else f" (+{len(unrecognized) - 5} more)"
        logger.warning(
            f"{path}: {len(unrecognized)} key(s) matched no known LoRA pattern and were dropped: {sample}{more}"
        )

    # Group Q/K/V that fold into a fused projection vs 1:1 singletons.
    fused: dict[tuple[int, str], dict[str, dict[str, torch.Tensor]]] = {}
    singletons: list[tuple[int, str, str, dict[str, torch.Tensor]]] = []  # (block, attn, sub, ab)
    unmapped = 0

    for base, ab in pairs.items():
        parsed = _parse_target(base)
        if parsed is None:
            unmapped += 1
            continue
        block_idx, attn_name, sub = parsed
        if sub in ("to_q", "to_k", "to_v"):
            is_self = attn_name == "attn1"
            qkv = sub[-1]
            # self-attn: q/k/v all fold into to_qkv. cross-attn: k/v fold into to_kv, q is a singleton.
            if is_self or qkv in ("k", "v"):
                fused.setdefault((block_idx, attn_name), {})[qkv] = ab
                continue
        singletons.append((block_idx, attn_name, sub, ab))

    if unmapped:
        logger.warning(f"{path}: {unmapped} pair(s) targeted a module absent from the LTX name map — dropped.")

    target_indices: dict[str, int] = {}

    for (block_idx, attn_name), qkvs in fused.items():
        attn = _get_attn(transformer, block_idx, attn_name)
        if attn is None:
            logger.warning(f"{path}: no attention module {attn_name} in block {block_idx} — skipping fused group.")
            continue
        idx = _register_fused(attn, qkvs, alphas, scale, name, block_idx, attn_name)
        if idx is None:
            continue
        canonical = f"transformer_blocks.{block_idx}.{attn_name}.{'to_qkv' if attn.is_self else 'to_kv'}"
        target_indices[canonical] = idx

    for block_idx, attn_name, sub, ab in singletons:
        if "A" not in ab or "B" not in ab:
            logger.warning(f"{path}: {attn_name}.{sub} in block {block_idx} missing one of A/B — skipping.")
            continue
        target, canonical, permute_dims = _resolve_singleton(transformer, block_idx, attn_name, sub)
        if target is None:
            logger.warning(f"{path}: unmapped singleton {attn_name}.{sub} in block {block_idx} — skipping.")
            continue
        A = ab["A"]
        B = ab["B"]
        if permute_dims is not None:
            B = _permute_qk_rows(B, *permute_dims)
        rank = A.shape[0]
        alpha = alphas.get(_alpha_key(block_idx, attn_name, sub), float(rank))
        idx = target.register_lora(A, B, scale=scale * (alpha / rank), name=name)
        target_indices[canonical] = idx

    if not target_indices:
        raise RuntimeError(f"{path}: no LoRA targets registered — check the adapter's key naming against LTX modules.")

    return LTXAdapterHandle(name=name or Path(path).stem, target_indices=target_indices)


def iter_lora_modules(root, prefix: str = ""):
    """Yield (dotted_path, module) for every LoRAMixin descendant of ``root``.
    Paths match the loader's canonical keys (e.g. ``transformer_blocks.0.attn1.to_qkv``)."""
    for name, child in root.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRAMixin):
            yield path, child
        yield from iter_lora_modules(child, path)


# --------------------------------------------------------------------
# Key parsing
# --------------------------------------------------------------------
def _strip_prefixes(key: str) -> str:
    for p in _STRIP_PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _collect_pairs(raw: dict[str, torch.Tensor]):
    """Return ({base: {'A','B'}}, {base: alpha}, num_skipped_direct, unrecognized_keys)."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    skipped_direct = 0
    unrecognized: list[str] = []
    for raw_key, tensor in raw.items():
        key = _strip_prefixes(raw_key)
        m = _LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[_SLOT_MAP[m.group("slot")]] = tensor
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(tensor.item())
        elif key.endswith((".diff", ".diff_b")):
            skipped_direct += 1
        else:
            unrecognized.append(raw_key)
    return pairs, alphas, skipped_direct, unrecognized


def _parse_target(base: str) -> tuple[int, str, str] | None:
    """Parse a stripped LoRA base path into (block_idx, attn_or_ff, sub).

    Returns None for paths outside the LoRA-adaptable set (patchify/proj_out/adaln)."""
    m = re.match(rf"^{_BLOCK_RE}\.(attn1|attn2)\.to_([qkv]|out)(?:\.0)?$", base)
    if m:
        sub = m.group(3)
        return int(m.group(1)), m.group(2), ("to_out" if sub == "out" else f"to_{sub}")
    # FFN: diffusers "ff.net.0.proj" -> ff1, "ff.net.2" -> ff2.
    m = re.match(rf"^{_BLOCK_RE}\.ff\.net\.0\.proj$", base)
    if m:
        return int(m.group(1)), "ff", "ff1"
    m = re.match(rf"^{_BLOCK_RE}\.ff\.net\.2$", base)
    if m:
        return int(m.group(1)), "ff", "ff2"
    return None


def _alpha_key(block_idx: int, attn_name: str, sub: str) -> str:
    return f"transformer_blocks.{block_idx}.{attn_name}.{sub}"


# --------------------------------------------------------------------
# Module resolution
# --------------------------------------------------------------------
def _get_attn(transformer, block_idx: int, attn_name: str):
    block = transformer.transformer_blocks[block_idx]
    return getattr(block, attn_name, None)


def _resolve_singleton(transformer, block_idx: int, attn_name: str, sub: str):
    """Return (module, canonical_path, permute_dims) where permute_dims is
    (num_heads, head_dim) if the base weight is _permute_qk'd, else None.
    (None, None, None) when the target does not exist."""
    block = transformer.transformer_blocks[block_idx]
    if sub in ("to_q", "to_out"):
        attn = getattr(block, attn_name, None)
        if attn is None:
            return None, None, None
        target = getattr(attn, sub, None)
        if target is None:
            return None, None, None
        # cross-attn to_q is _permute_qk'd by _prepare_torch_state; to_out is not.
        permute_dims = (attn.num_heads, attn.head_dim) if (sub == "to_q" and not attn.is_self) else None
        return target, f"transformer_blocks.{block_idx}.{attn_name}.{sub}", permute_dims
    if sub in ("ff1", "ff2"):
        ff = getattr(block, "ffn", None)
        if ff is None:
            return None, None, None
        target = getattr(ff, sub, None)
        return target, f"transformer_blocks.{block_idx}.ffn.{sub}", None
    return None, None, None


# --------------------------------------------------------------------
# Fused QKV / KV registration (permute_qk + head-interleave on B)
# --------------------------------------------------------------------
def _register_fused(attn, qkvs, alphas, scale, name, block_idx, attn_name) -> int | None:
    required = ["q", "k", "v"] if attn.is_self else ["k", "v"]
    missing = [r for r in required if r not in qkvs or "A" not in qkvs[r] or "B" not in qkvs[r]]
    if missing:
        logger.warning(f"fused LoRA on block {block_idx}.{attn_name} missing {missing}; skipping (got {list(qkvs)}).")
        return None

    ranks = [qkvs[r]["A"].shape[0] for r in required]
    if len(set(ranks)) != 1:
        raise ValueError(f"QKV LoRA ranks must match on block {block_idx}.{attn_name}; got {ranks}")
    r = ranks[0]
    n = len(required)

    num_heads = attn.num_heads
    head_dim = attn.head_dim
    n_dev = attn.parallel_config.tensor_parallel.factor
    n_local_heads = attn.n_local_heads
    out_per = attn.dim  # each fused source projects to dim

    # A stacked on rank -> [n*r, in]
    A_fused = torch.cat([qkvs[x]["A"] for x in required], dim=0)

    # B: permute Q/K rows (RoPE), pad each into a block-diagonal [out, n*r] tile,
    # then head-interleave the tiles into [n*out, n*r] to match the fused weight.
    B_padded = []
    for i, x in enumerate(required):
        B = qkvs[x]["B"]
        if x in ("q", "k"):
            B = _permute_qk_rows(B, num_heads, head_dim)
        pad = torch.zeros(out_per, n * r, dtype=B.dtype)
        pad[:, i * r : (i + 1) * r] = B
        B_padded.append(pad)
    B_fused = _interleave_heads_rows(B_padded, num_heads, head_dim, n_dev, n_local_heads)

    alpha = alphas.get(_alpha_key(block_idx, attn_name, f"to_{required[0]}"), float(r))
    eff_scale = scale * (alpha / r)
    return attn.register_lora(A_fused, B_fused, scale=eff_scale, name=name)


def _permute_qk_rows(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder each head's output channels SPLIT -> INTERLEAVED, mirroring
    ``LTXAttention._prepare_torch_state._permute_qk``. ``t`` is [num_heads*head_dim, ...]."""
    D = head_dim
    D_half = D // 2
    perm = torch.empty(D, dtype=torch.long)
    perm[0::2] = torch.arange(D_half)
    perm[1::2] = torch.arange(D_half, D)
    rest = t.shape[1:]
    return t.reshape(num_heads, D, *rest).index_select(1, perm).reshape(num_heads * D, *rest)


def _interleave_heads_rows(tensors, num_heads: int, head_dim: int, n_dev: int, n_local_heads: int) -> torch.Tensor:
    """Interleave the OUT (row) dim across sources, mirroring
    ``LTXAttention._prepare_torch_state._interleave_heads``. Each input is
    [out=num_heads*head_dim, C]; returns [n*out, C]."""
    transposed = [t.T for t in tensors]  # [C, out]
    reshaped = [t.reshape(t.shape[0], n_dev, n_local_heads, head_dim) for t in transposed]
    merged = torch.cat(reshaped, dim=2)  # [C, n_dev, n*n_local_heads, head_dim]
    merged = merged.reshape(merged.shape[0], len(tensors) * num_heads * head_dim)  # [C, n*out]
    return merged.T  # [n*out, C]
