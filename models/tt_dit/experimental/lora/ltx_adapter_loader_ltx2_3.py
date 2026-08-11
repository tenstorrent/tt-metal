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

Scope: every LoRA-adaptable Linear in the transformer — video/audio/cross-modal
attention (``to_qkv``/``to_q``/``to_kv``/``to_out``/``to_gate_logits``), both FFNs
(ff1/ff2), and the globals (``patchify_proj``, ``proj_out``, and the
``adaln_single`` / timestep-embedder linears). ``promote_to_lora`` makes the plain
globals bindable; attention/FFN are already LoRA via ``lora_enabled``. Only
attention q/k/v/q-of-cross carry the rotary permute + head-interleave transform;
everything else registers directly. ``.diff``/``.diff_b`` direct deltas remain
unsupported; unmapped keys are surfaced loudly, never silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from ...layers.lora import LoRAMixin
from .promote import promote_to_lora

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}

# Diffusers LTX checkpoints name blocks "transformer_blocks.<i>"; some exports
# use the shorter "blocks.<i>". Accept both.
_BLOCK_RE = r"(?:transformer_blocks|blocks)\.(\d+)"

# All LTXAttention instances on a block share the same to_q/k/v/out/gate_logits
# structure and fusion rules (self -> to_qkv, cross -> to_q + to_kv), so the loader
# handles them uniformly, reading is_self/heads off each resolved module.
_ATTN_NAMES = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
# Diffusers FFN container name -> tt-dit attribute (ff.net.0.proj/net.2 -> ffX).
_FF_NAME_TO_ATTR = {"ff": "ffn", "audio_ff": "audio_ff"}


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

    # Promote every plain Linear (adaln/proj/patchify/gate globals) to a LoRA
    # variant so it can be bound; attn/ffn are already LoRA via lora_enabled.
    promote_to_lora(transformer)

    # Q/K/V fold into a fused projection; the rest are 1:1 block singletons or
    # direct globals (patchify/proj_out/adaln/timestep-embedder).
    fused: dict[tuple[int, str], dict[str, dict[str, torch.Tensor]]] = {}
    singletons: list[tuple[int, str, str, dict[str, torch.Tensor]]] = []  # (block, container, sub, ab)
    globals_: list[tuple[str, dict[str, torch.Tensor]]] = []

    for base, ab in pairs.items():
        parsed = _parse_target(base)
        if parsed is None:
            globals_.append((base, ab))
            continue
        block_idx, container, sub = parsed
        if sub in ("to_q", "to_k", "to_v"):
            attn = _get_attn(transformer, block_idx, container)
            is_self = attn.is_self if attn is not None else container in ("attn1", "audio_attn1")
            qkv = sub[-1]
            # self-attn: q/k/v all fold into to_qkv. cross-attn: k/v fold into to_kv, q is a singleton.
            if is_self or qkv in ("k", "v"):
                fused.setdefault((block_idx, container), {})[qkv] = ab
                continue
        singletons.append((block_idx, container, sub, ab))

    target_indices: dict[str, int] = {}

    for (block_idx, container), qkvs in fused.items():
        attn = _get_attn(transformer, block_idx, container)
        if attn is None:
            logger.warning(f"{path}: no attention module {container} in block {block_idx} — skipping fused group.")
            continue
        idx = _register_fused(attn, qkvs, alphas, scale, name, block_idx, container)
        if idx is None:
            continue
        canonical = f"transformer_blocks.{block_idx}.{container}.{'to_qkv' if attn.is_self else 'to_kv'}"
        target_indices[canonical] = idx

    for block_idx, container, sub, ab in singletons:
        if "A" not in ab or "B" not in ab:
            logger.warning(f"{path}: {container}.{sub} in block {block_idx} missing one of A/B — skipping.")
            continue
        target, canonical, permute_dims = _resolve_singleton(transformer, block_idx, container, sub)
        if target is None:
            logger.warning(f"{path}: unmapped {container}.{sub} in block {block_idx} — skipping.")
            continue
        A = ab["A"]
        B = ab["B"]
        if permute_dims is not None:
            B = _permute_qk_rows(B, *permute_dims)
        rank = A.shape[0]
        alpha = alphas.get(_alpha_key(block_idx, container, sub), float(rank))
        idx = target.register_lora(A, B, scale=scale * (alpha / rank), name=name)
        target_indices[canonical] = idx

    unmapped_global = 0
    for base, ab in globals_:
        if "A" not in ab or "B" not in ab:
            continue
        target = _resolve_global(transformer, base)
        if not isinstance(target, LoRAMixin):
            unmapped_global += 1
            continue
        A = ab["A"]
        B = ab["B"]
        rank = A.shape[0]
        alpha = alphas.get(base, float(rank))
        idx = target.register_lora(A, B, scale=scale * (alpha / rank), name=name)
        target_indices[base] = idx

    if unmapped_global:
        logger.warning(f"{path}: {unmapped_global} global key(s) matched no LTX module — dropped.")

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
    """Parse a stripped per-block LoRA base path into (block_idx, container, sub).

    Covers every block attention family (video/audio/cross-modal) and both FFN
    containers. Returns None for non-block paths (globals: patchify/proj_out/adaln),
    which ``_resolve_global`` handles by direct path lookup."""
    attn_alt = "|".join(_ATTN_NAMES)
    m = re.match(rf"^{_BLOCK_RE}\.({attn_alt})\.to_(q|k|v|out|gate_logits)(?:\.0)?$", base)
    if m:
        raw = m.group(3)
        sub = {"out": "to_out", "gate_logits": "to_gate_logits"}.get(raw, f"to_{raw}")
        return int(m.group(1)), m.group(2), sub
    ff_alt = "|".join(_FF_NAME_TO_ATTR)
    m = re.match(rf"^{_BLOCK_RE}\.({ff_alt})\.net\.0\.proj$", base)
    if m:
        return int(m.group(1)), m.group(2), "ff1"
    m = re.match(rf"^{_BLOCK_RE}\.({ff_alt})\.net\.2$", base)
    if m:
        return int(m.group(1)), m.group(2), "ff2"
    return None


def _alpha_key(block_idx: int, attn_name: str, sub: str) -> str:
    return f"transformer_blocks.{block_idx}.{attn_name}.{sub}"


# --------------------------------------------------------------------
# Module resolution
# --------------------------------------------------------------------
def _get_attn(transformer, block_idx: int, attn_name: str):
    blocks = transformer.transformer_blocks
    if not (0 <= block_idx < len(blocks)):
        return None
    return getattr(blocks[block_idx], attn_name, None)


def _resolve_singleton(transformer, block_idx: int, container: str, sub: str):
    """Return (module, canonical_path, permute_dims) where permute_dims is
    (num_heads, head_dim) if the base weight is _permute_qk'd, else None.
    (None, None, None) when the target does not exist."""
    blocks = transformer.transformer_blocks
    if not (0 <= block_idx < len(blocks)):
        return None, None, None
    block = blocks[block_idx]
    if sub in ("to_q", "to_out", "to_gate_logits"):
        attn = getattr(block, container, None)
        if attn is None:
            return None, None, None
        target = getattr(attn, sub, None)
        if target is None:
            return None, None, None
        # Only cross-attn to_q is _permute_qk'd; to_out/to_gate_logits are not.
        permute_dims = (attn.num_heads, attn.head_dim) if (sub == "to_q" and not attn.is_self) else None
        return target, f"transformer_blocks.{block_idx}.{container}.{sub}", permute_dims
    if sub in ("ff1", "ff2"):
        ff_attr = _FF_NAME_TO_ATTR.get(container, "ffn")
        ff = getattr(block, ff_attr, None)
        if ff is None:
            return None, None, None
        target = getattr(ff, sub, None)
        return target, f"transformer_blocks.{block_idx}.{ff_attr}.{sub}", None
    return None, None, None


def _resolve_global(transformer, base: str):
    """Resolve a non-block LoRA key (patchify_proj, proj_out, adaln_single.*, ...)
    to the module at that dotted path on the transformer root, or None if absent.
    These are plain replicated/parallel Linears with no rotary layout transform."""
    mod = transformer
    for part in base.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


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
    # register_lora lives on the fused projection Linear (LoRAMixin), not the attention wrapper.
    fused_linear = attn.to_qkv if attn.is_self else attn.to_kv
    return fused_linear.register_lora(A_fused, B_fused, scale=eff_scale, name=name)


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
