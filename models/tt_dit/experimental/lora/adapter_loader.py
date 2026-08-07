# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter loader for the runtime-LoRA Wan 2.2 pipeline.

Reads a LoRA safetensors file, parses A/B pairs using the same key
normalization as ``pipeline_wan_lora.py``, and uploads them into the
LoRA-aware Linear modules of a `WanTransformer3DModel` constructed with
``lora_enabled=True``.

Fused-QKV handling (the hard part):
  Wan attention uses a fused ``to_qkv`` (self-attn) and ``to_kv``
  (cross-attn) projection. The adapter ships separate ``to_q``,
  ``to_k``, ``to_v`` LoRA pairs. We combine them into a single LoRA
  pair on the fused module by:

    - stacking A along the rank dim:  A_fused = [A_q; A_k; A_v]  → [3r, in]
    - building a block-diagonal B with rank=3r and head-interleaving
      it the same way the base weight is head-interleaved:
        B_fused = head_interleave([B_q | 0 | 0,  0 | B_k | 0,  0 | 0 | B_v])

  This triples the LoRA matmul cost on those layers (rank=3r) but keeps
  the math exact and shippable in v0. A follow-up could split this into
  3 independent adapters at the layer level for ~3x lower cost.

`.diff` / `.diff_b` direct deltas are out of scope for v0 — the loader
warns and skips them.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key

# Reused from pipeline_wan_lora.py — kept inline so this module is self-contained.
_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}


def _strip_known_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _kohya_to_lightx2v(key: str) -> str:
    if not key.startswith("lora_unet_"):
        return key
    parts = key.split(".", 1)
    module_path = parts[0][len("lora_unet_") :]
    suffix = f".{parts[1]}" if len(parts) > 1 else ""
    m = re.match(r"blocks_(\d+)_(cross_attn|self_attn)_([a-z]+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"
    m = re.match(r"blocks_(\d+)_(ffn)_(\d+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"
    logger.warning(f"unrecognized kohya key structure: {key}")
    return key


def _normalize_key(raw: str) -> str:
    return _kohya_to_lightx2v(_strip_known_prefixes(raw))


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterHandle:
    name: str
    rank: int  # the "expected bound rank" — may be 3r for fused-QKV targets
    target_indices: dict[str, int]
    """Map from a stable target identifier (e.g. ``'blocks.0.attn1.to_qkv'``)
    to the bank index returned by `register_lora` on that module."""


def load_adapter_into(
    transformer,  # WanTransformer3DModel constructed with lora_enabled=True
    path: str,
    *,
    scale: float = 1.0,
    name: str = "",
) -> AdapterHandle:
    """Load a single LoRA safetensors file into `transformer`.

    Returns an `AdapterHandle` recording the bank index assigned to each
    LoRA-targeted Linear. Use it later with
    ``pipeline.set_active_lora(handle)``.
    """
    raw = load_file(str(path))
    if not any(_is_lora_key(k) for k in raw):
        raise RuntimeError(f"no LoRA-style keys (lora_A/lora_B, lora_down/lora_up) in {path}")

    pairs, alphas, skipped_direct = _collect_pairs(raw)
    if skipped_direct:
        logger.warning(
            f"{path}: skipping {skipped_direct} direct (.diff/.diff_b) deltas "
            "— runtime-LoRA pipeline does not support these yet."
        )

    # Group targets
    fused_qkv: dict[tuple[int, str], dict[str, dict[str, torch.Tensor]]] = {}
    singletons: list[tuple[str, dict[str, torch.Tensor]]] = []  # (diffusers_path, ab)
    skipped_unmapped = 0

    for base_path, ab in pairs.items():
        diff_path = _diffusers_path(base_path)
        if diff_path is None:
            skipped_unmapped += 1
            continue
        m = re.match(r"blocks\.(\d+)\.(attn1|attn2)\.to_([qkv])$", diff_path)
        if m:
            block_idx, attn, qkv = int(m.group(1)), m.group(2), m.group(3)
            # attn1 (self-attn): Q/K/V all fold into to_qkv.
            # attn2 (cross-attn): Q is its own singleton; K/V fold into to_kv.
            if attn == "attn1" or qkv in ("k", "v"):
                fused_qkv.setdefault((block_idx, attn), {})[qkv] = ab
                continue
            # attn2.to_q is a real singleton — fall through.
        # Everything else: 1:1 mapping
        singletons.append((diff_path, ab))

    if skipped_unmapped:
        logger.warning(
            f"{path}: {skipped_unmapped} pairs unmapped — adapter targets a module "
            "that doesn't exist in the tt-dit name map."
        )

    target_indices: dict[str, int] = {}
    fused_ranks: list[int] = []
    singleton_ranks: list[int] = []

    # Fused QKV / KV
    for (block_idx, attn_name), qkvs in fused_qkv.items():
        attn = getattr(transformer.blocks[block_idx], attn_name)
        bank_idx, fused_rank = _register_fused_qkv(attn, block_idx, qkvs, alphas, scale, name)
        if bank_idx is None:
            continue
        key = f"blocks.{block_idx}.{attn_name}.{'to_qkv' if attn.is_self else 'to_kv'}"
        target_indices[key] = bank_idx
        fused_ranks.append(fused_rank)

    # Singletons
    for diff_path, ab in singletons:
        target, canonical = _resolve_singleton(transformer, diff_path)
        if target is None:
            logger.warning(f"unmapped singleton LoRA target: {diff_path}")
            continue
        A = ab["A"]
        B = ab["B"]
        rank = A.shape[0]
        alpha = alphas.get(_lightx2v_for_path(diff_path), float(rank))
        eff_scale = scale * (alpha / rank)
        idx = target.register_lora(A, B, scale=eff_scale, name=name)
        target_indices[canonical] = idx
        singleton_ranks.append(rank)

    # Sanity: every LoRA module in this adapter should have the same
    # bound rank, but fused_QKV produces a 3x larger rank than singletons.
    # The bound-tensor system handles per-module ranks independently, so
    # we just record both for reporting.
    all_ranks = fused_ranks + singleton_ranks
    if not all_ranks:
        raise RuntimeError(f"{path}: no LoRA targets registered")
    representative_rank = max(all_ranks)
    return AdapterHandle(name=name or Path(path).stem, rank=representative_rank, target_indices=target_indices)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _is_lora_key(raw: str) -> bool:
    key = _normalize_key(raw)
    return bool(_LOW_RANK_RE.match(key)) or key.endswith((".diff", ".diff_b"))


def _collect_pairs(raw: dict[str, torch.Tensor]):
    """Returns ({base_path: {'A': T, 'B': T}}, {base_path: alpha}, num_skipped_direct)."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    skipped_direct = 0
    unrecognized: list[tuple[str, str]] = []

    for raw_key, tensor in raw.items():
        key = _normalize_key(raw_key)
        m = _LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[_SLOT_MAP[m.group("slot")]] = tensor
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(tensor.item())
        elif key.endswith(".diff") or key.endswith(".diff_b"):
            skipped_direct += 1
        else:
            unrecognized.append((raw_key, key))

    # Surface dropped keys — a non-standard naming convention silently
    # producing plausible-but-wrong output is the failure mode we cannot
    # afford here.
    if unrecognized:
        sample = ", ".join(f"{raw}→{norm}" for raw, norm in unrecognized[:5])
        more = "" if len(unrecognized) <= 5 else f" (+{len(unrecognized) - 5} more)"
        logger.warning(
            f"adapter loader: {len(unrecognized)} key(s) did not match any known "
            f"pattern (lora_A/B, lora_down/up, .alpha, .diff, .diff_b) and were "
            f"dropped. Samples: {sample}{more}"
        )

    return pairs, alphas, skipped_direct


def _diffusers_path(lightx2v_base: str) -> str | None:
    """Map a lightx2v base path to its diffusers-canonical equivalent
    (without ``.weight``). Returns None if mapping fails."""
    try:
        key = wan_lightx2v_to_diffusers_key(f"{lightx2v_base}.weight")
    except Exception:  # noqa: BLE001
        return None
    if not key.endswith(".weight"):
        return None
    return key[: -len(".weight")]


def _lightx2v_for_path(diffusers_path: str) -> str:
    """Reverse-map (best-effort) a diffusers path back to lightx2v for alpha
    lookups — alphas are keyed in the source naming."""
    # Simple cases that the existing map covers:
    rev = {
        "attn1.to_q": "self_attn.q",
        "attn1.to_k": "self_attn.k",
        "attn1.to_v": "self_attn.v",
        "attn1.to_out.0": "self_attn.o",
        "attn2.to_q": "cross_attn.q",
        "attn2.to_k": "cross_attn.k",
        "attn2.to_v": "cross_attn.v",
        "attn2.to_out.0": "cross_attn.o",
        "ffn.net.0.proj": "ffn.0",
        "ffn.net.2": "ffn.2",
    }
    for diff_seg, lx_seg in rev.items():
        if diff_seg in diffusers_path:
            return diffusers_path.replace(diff_seg, lx_seg)
    return diffusers_path


def _resolve_singleton(transformer, diff_path: str):
    """Resolve a diffusers-style path to (module, tt_dit_dotted_path).

    Returns (None, None) when no match. The tt-dit path is the one
    consumed by the pipeline mixin's bind/unbind walker — it skips
    diffusers-only segments like the trailing ``.0`` in ``to_out.0``.
    """
    m = re.match(r"blocks\.(\d+)\.(attn1|attn2)\.to_q$", diff_path)
    if m:
        block_idx, attn_name = int(m.group(1)), m.group(2)
        attn = getattr(transformer.blocks[block_idx], attn_name)
        if not attn.is_self:
            return attn.to_q, f"blocks.{block_idx}.{attn_name}.to_q"
        return None, None  # self-attn to_q shouldn't arrive here
    m = re.match(r"blocks\.(\d+)\.(attn1|attn2)\.to_out\.0$", diff_path)
    if m:
        block_idx, attn_name = int(m.group(1)), m.group(2)
        return getattr(transformer.blocks[block_idx], attn_name).to_out, (f"blocks.{block_idx}.{attn_name}.to_out")
    m = re.match(r"blocks\.(\d+)\.ffn\.net\.0\.proj$", diff_path)
    if m:
        block_idx = int(m.group(1))
        return transformer.blocks[block_idx].ffn.ff1, f"blocks.{block_idx}.ffn.ff1"
    m = re.match(r"blocks\.(\d+)\.ffn\.net\.2$", diff_path)
    if m:
        block_idx = int(m.group(1))
        return transformer.blocks[block_idx].ffn.ff2, f"blocks.{block_idx}.ffn.ff2"
    return None, None


def _register_fused_qkv(
    attn,
    block_idx: int,
    qkvs: dict[str, dict[str, torch.Tensor]],
    alphas: dict[str, float],
    scale: float,
    name: str,
) -> tuple[int | None, int]:
    """Stack Q/K/V (or K/V for cross-attn) LoRA pairs onto attn.to_qkv (or
    attn.to_kv). Returns (bank_idx, fused_rank) or (None, 0) on skip."""
    if attn.is_self:
        required = ["q", "k", "v"]
        target = attn.to_qkv
    else:
        required = ["k", "v"]
        target = attn.to_kv

    missing = [r for r in required if r not in qkvs]
    if missing:
        logger.warning(
            f"fused QKV LoRA on {target.__class__.__name__} is missing {missing}; "
            "skipping this group (need all of "
            f"{required}, got {list(qkvs)})."
        )
        return None, 0

    A_per = [qkvs[r]["A"] for r in required]
    B_per = [qkvs[r]["B"] for r in required]
    ranks = [A.shape[0] for A in A_per]
    if len(set(ranks)) != 1:
        raise ValueError(f"QKV LoRA ranks must match; got {ranks}")
    r = ranks[0]
    n = len(required)
    in_dim = attn.dim
    out_per = attn.dim  # each of Q/K/V has out=dim

    # A stacked on rank dim → [n*r, in]
    A_fused = torch.cat(A_per, dim=0)
    assert A_fused.shape == (n * r, in_dim), f"A_fused {A_fused.shape}"

    # B: each per-source B is [out_per=dim, r]. We embed each into a
    # [out_per, n*r] tile (zeros for other-source rank columns). Then
    # head-interleave the list of [out_per, n*r] tiles into a single
    # [n*out_per, n*r] tensor — matching the head layout of the fused
    # to_qkv / to_kv base weight.
    n_dev = attn.parallel_config.tensor_parallel.factor
    n_local_heads = attn.n_local_heads
    head_dim = attn.head_dim

    B_per_padded: list[torch.Tensor] = []
    for i, B in enumerate(B_per):
        pad = torch.zeros(out_per, n * r, dtype=B.dtype)
        pad[:, i * r : (i + 1) * r] = B
        B_per_padded.append(pad)

    B_fused = _head_interleave_lora_B(B_per_padded, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
    assert B_fused.shape == (n * out_per, n * r), f"B_fused {B_fused.shape}"

    # Effective scale: pick the first source's alpha (alphas are usually
    # identical across Q/K/V in a single adapter; if not, the choice is
    # safe-ish because we apply scale to all three the same way).
    alpha_keys = [_lightx2v_qkv_key(attn, block_idx, r_name) for r_name in required]
    alpha = next((alphas[k] for k in alpha_keys if k in alphas), float(r))
    eff_scale = scale * (alpha / r)

    bank_idx = target.register_lora(A_fused, B_fused, scale=eff_scale, name=name)
    return bank_idx, n * r


def _lightx2v_qkv_key(attn, block_idx: int, qkv: str) -> str:
    """Build the lightx2v alpha key for a Q/K/V projection inside a
    given WanAttention. Alphas in `_collect_pairs` are keyed by the full
    normalized base path (``blocks.<i>.<side>.<qkv>``), so the block
    index must be threaded in alongside ``attn.is_self``."""
    side = "self_attn" if attn.is_self else "cross_attn"
    return f"blocks.{block_idx}.{side}.{qkv}"


def _head_interleave_lora_B(
    tensors: Sequence[torch.Tensor],
    *,
    n_dev: int,
    n_local_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Mirror of WanAttention._interleave_heads for LoRA B tensors.

    Each input is [out_per_source=num_heads*head_dim, rank_fused].
    Returns a concatenated [N*out_per, rank_fused] tensor with the out
    dim head-interleaved so column-parallel sharding lines up with the
    fused base weight's layout.
    """
    out_per = n_dev * n_local_heads * head_dim
    fused_rank = tensors[0].shape[1]
    assert all(t.shape == (out_per, fused_rank) for t in tensors), (
        f"all B tensors must be [{out_per}, {fused_rank}]; got " f"{[tuple(t.shape) for t in tensors]}"
    )
    n = len(tensors)
    # Transpose each to [fused_rank, out_per]
    transposed = [t.T for t in tensors]
    # Reshape out dim to [fused_rank, n_dev, n_local_heads, head_dim]
    reshaped = [t.reshape(fused_rank, n_dev, n_local_heads, head_dim) for t in transposed]
    # Concat on heads sub-dim → [fused_rank, n_dev, N*n_local_heads, head_dim]
    merged = torch.cat(reshaped, dim=2)
    # Flatten back → [fused_rank, N*n_dev*n_local_heads*head_dim] = [fused_rank, N*out_per]
    merged = merged.reshape(fused_rank, n * out_per)
    # Transpose to [N*out_per, fused_rank]
    return merged.T
