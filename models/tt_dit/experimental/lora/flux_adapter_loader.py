# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter loader for FLUX.1 (Kontext) LoRAs, mirroring the Wan 2.2 loader
(``adapter_loader.py``) but for the FLUX transformer's module tree and the
diffusers FLUX key convention.

Reads a LoRA safetensors file, parses A/B pairs, and uploads them into the
LoRA-aware Linear modules of a ``Flux1Transformer`` built with
``lora_enabled=True``.

Fused-QKV handling (identical shape problem to Wan):
  FLUX attention uses a single fused ``to_qkv`` (spatial stream) and, in the
  double-stream blocks, a fused ``add_qkv_proj`` (context stream). Diffusers
  adapters ship separate ``to_q`` / ``to_k`` / ``to_v`` (and
  ``add_q_proj`` / ``add_k_proj`` / ``add_v_proj``) pairs. We combine them into
  one LoRA pair on the fused module by stacking A on the rank dim and building
  a head-interleaved block-diagonal B, matching how
  ``blocks.attention.Attention._reshape_and_merge_qkv`` interleaves the base
  Q/K/V weight. This is done by ``_register_fused`` + ``_head_interleave_lora_B``
  (reused from ``adapter_loader``).

v0 scope / limitations (warn + skip, never silently miswire):
  - Diffusers-style keys only (``transformer_blocks.N.attn.to_q.lora_A.weight``,
    optionally prefixed ``transformer.``/``diffusion_model.``). kohya
    ``lora_unet_*`` FLUX keys are not converted yet.
  - ``single_transformer_blocks.N.proj_out`` (the re-fused-input projection)
    and modulation ``norm*.linear`` adapters are skipped — rare targets whose
    weight layout needs extra handling.
  - Head padding on a LoRA-targeted attention (``padding_config`` with
    ``head_padding > 0``) is unsupported; the loader raises rather than
    produce a shape-mismatched delta. FLUX's common head counts divide the
    usual TP factors, so this does not trigger in practice.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from ...layers.lora import LoRAMixin
from .adapter_loader import AdapterHandle, _head_interleave_lora_B

_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "lora_transformer.", "unet.", "model.")
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}

# diffusers FLUX qkv sub-projection names, per stream, in the fixed q,k,v order
# the fused weight expects.
_SPATIAL_QKV = ("to_q", "to_k", "to_v")
_CONTEXT_QKV = ("add_q_proj", "add_k_proj", "add_v_proj")


def _strip_known_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _is_lora_key(raw: str) -> bool:
    key = _strip_known_prefixes(raw)
    return bool(_LOW_RANK_RE.match(key)) or key.endswith((".diff", ".diff_b"))


def _collect_pairs(raw: dict[str, torch.Tensor]):
    """Returns ({base_path: {'A': T, 'B': T}}, {base_path: alpha}, num_skipped_direct)."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    skipped_direct = 0
    unrecognized: list[str] = []

    for raw_key, tensor in raw.items():
        key = _strip_known_prefixes(raw_key)
        m = _LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[_SLOT_MAP[m.group("slot")]] = tensor
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(tensor.item())
        elif key.endswith((".diff", ".diff_b")):
            skipped_direct += 1
        else:
            unrecognized.append(raw_key)

    if unrecognized:
        sample = ", ".join(unrecognized[:5])
        more = "" if len(unrecognized) <= 5 else f" (+{len(unrecognized) - 5} more)"
        logger.warning(
            f"flux adapter loader: {len(unrecognized)} key(s) matched no known "
            f"pattern (lora_A/B, lora_down/up, .alpha, .diff/.diff_b) and were dropped. "
            f"Samples: {sample}{more}"
        )
    return pairs, alphas, skipped_direct


# --------------------------------------------------------------------
# base-path parsing → structured target
# --------------------------------------------------------------------
_RE_ATTN_QKV = re.compile(r"^transformer_blocks\.(\d+)\.attn\.(to_q|to_k|to_v)$")
_RE_ATTN_ADD_QKV = re.compile(r"^transformer_blocks\.(\d+)\.attn\.(add_q_proj|add_k_proj|add_v_proj)$")
_RE_ATTN_TO_OUT = re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_out\.0$")
_RE_ATTN_TO_ADD_OUT = re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_add_out$")
_RE_FF = re.compile(r"^transformer_blocks\.(\d+)\.ff\.net\.(0\.proj|2)$")
_RE_FF_CTX = re.compile(r"^transformer_blocks\.(\d+)\.ff_context\.net\.(0\.proj|2)$")
_RE_SINGLE_QKV = re.compile(r"^single_transformer_blocks\.(\d+)\.attn\.(to_q|to_k|to_v)$")
_RE_SINGLE_PROJ_MLP = re.compile(r"^single_transformer_blocks\.(\d+)\.proj_mlp$")
_RE_SINGLE_PROJ_OUT = re.compile(r"^single_transformer_blocks\.(\d+)\.proj_out$")


def load_flux_adapter_into(
    transformer,  # Flux1Transformer built with lora_enabled=True
    path: str,
    *,
    scale: float = 1.0,
    name: str = "",
) -> AdapterHandle:
    """Load a single FLUX LoRA safetensors file into ``transformer``.

    Returns an ``AdapterHandle`` recording the bank index assigned to each
    LoRA-targeted Linear (keyed by a stable tt-dit dotted path). Bind it with
    the pipeline's bind walker (``bind_active`` per module) to activate.
    """
    raw = load_file(str(path))
    if not any(_is_lora_key(k) for k in raw):
        raise RuntimeError(f"no LoRA-style keys (lora_A/lora_B, lora_down/lora_up) in {path}")

    pairs, alphas, skipped_direct = _collect_pairs(raw)
    if skipped_direct:
        logger.warning(f"{path}: skipping {skipped_direct} direct (.diff/.diff_b) deltas — not supported.")

    # Grouping: fused QKV pairs collect per (block, stream); everything else is a singleton.
    #   fused key: (kind, block_idx) where kind in {'spatial','context','single'}
    fused: dict[tuple[str, int], dict[str, dict[str, torch.Tensor]]] = {}
    singletons: list[tuple[str, dict[str, torch.Tensor], str]] = []  # (dotted_path, ab, alpha_key)
    skipped_unmapped = 0

    for base_path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            logger.warning(f"{path}: {base_path} has an unpaired A/B ({list(ab)}); skipping")
            continue

        m = _RE_ATTN_QKV.match(base_path)
        if m:
            fused.setdefault(("spatial", int(m.group(1))), {})[m.group(2)] = ab
            continue
        m = _RE_ATTN_ADD_QKV.match(base_path)
        if m:
            fused.setdefault(("context", int(m.group(1))), {})[m.group(2)] = ab
            continue
        m = _RE_SINGLE_QKV.match(base_path)
        if m:
            fused.setdefault(("single", int(m.group(1))), {})[m.group(2)] = ab
            continue

        m = _RE_ATTN_TO_OUT.match(base_path)
        if m:
            i = int(m.group(1))
            singletons.append((f"transformer_blocks.{i}.attn.to_out", ab, base_path))
            continue
        m = _RE_ATTN_TO_ADD_OUT.match(base_path)
        if m:
            i = int(m.group(1))
            singletons.append((f"transformer_blocks.{i}.attn.to_add_out", ab, base_path))
            continue
        m = _RE_FF.match(base_path)
        if m:
            i, which = int(m.group(1)), m.group(2)
            ff_attr = "ff1" if which == "0.proj" else "ff2"
            singletons.append((f"transformer_blocks.{i}.ff.{ff_attr}", ab, base_path))
            continue
        m = _RE_FF_CTX.match(base_path)
        if m:
            i, which = int(m.group(1)), m.group(2)
            ff_attr = "ff1" if which == "0.proj" else "ff2"
            singletons.append((f"transformer_blocks.{i}.ff_context.{ff_attr}", ab, base_path))
            continue
        m = _RE_SINGLE_PROJ_MLP.match(base_path)
        if m:
            i = int(m.group(1))
            singletons.append((f"single_transformer_blocks.{i}.proj_mlp", ab, base_path))
            continue
        if _RE_SINGLE_PROJ_OUT.match(base_path) or base_path.endswith(
            (".norm1.linear", ".norm1_context.linear", ".norm.linear")
        ):
            logger.warning(f"{path}: skipping unsupported v0 LoRA target {base_path!r} (proj_out / modulation).")
            skipped_unmapped += 1
            continue

        skipped_unmapped += 1
        logger.warning(f"{path}: unmapped LoRA target {base_path!r} — no matching module in the FLUX tree.")

    if skipped_unmapped:
        logger.warning(f"{path}: {skipped_unmapped} LoRA target group(s) skipped (see warnings above).")

    target_indices: dict[str, int] = {}
    ranks: list[int] = []

    # ---- fused QKV / add-QKV ----
    for (kind, block_idx), qkvs in fused.items():
        names = _CONTEXT_QKV if kind == "context" else _SPATIAL_QKV
        if kind == "single":
            attn = transformer.single_transformer_blocks[block_idx].attn
            target = attn.to_qkv
            dotted = f"single_transformer_blocks.{block_idx}.attn.to_qkv"
        elif kind == "spatial":
            attn = transformer.transformer_blocks[block_idx].attn
            target = attn.to_qkv
            dotted = f"transformer_blocks.{block_idx}.attn.to_qkv"
        else:  # context
            attn = transformer.transformer_blocks[block_idx].attn
            target = attn.add_qkv_proj
            dotted = f"transformer_blocks.{block_idx}.attn.add_qkv_proj"

        bank_idx, fused_rank = _register_fused(attn, target, dotted, names, qkvs, alphas, scale, name)
        if bank_idx is None:
            continue
        target_indices[dotted] = bank_idx
        ranks.append(fused_rank)

    # ---- singletons ----
    for dotted, ab, alpha_key in singletons:
        target = _path_to_module(transformer, dotted)
        if not isinstance(target, LoRAMixin):
            logger.warning(f"{path}: {dotted} is not a LoRA-aware module (lora_enabled=False?); skipping")
            continue
        A, B = ab["A"], ab["B"]
        rank = A.shape[0]
        alpha = alphas.get(alpha_key, float(rank))
        eff_scale = scale * (alpha / rank)
        idx = target.register_lora(A, B, scale=eff_scale, name=name)
        target_indices[dotted] = idx
        ranks.append(rank)

    if not ranks:
        raise RuntimeError(f"{path}: no LoRA targets registered")

    return AdapterHandle(name=name or Path(path).stem, rank=max(ranks), target_indices=target_indices)


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def _path_to_module(transformer, dotted: str):
    """Resolve a tt-dit dotted path like ``transformer_blocks.0.attn.to_out`` or
    ``single_transformer_blocks.3.proj_mlp`` to the module instance."""
    obj = transformer
    for seg in dotted.split("."):
        obj = obj[int(seg)] if seg.isdigit() else getattr(obj, seg)
    return obj


def _register_fused(attn, target, dotted, names, qkvs, alphas, scale, name):
    """Stack the three (A,B) pairs named in ``names`` onto the fused ``target``
    Linear. Returns (bank_idx, fused_rank) or (None, 0) on skip."""
    if not isinstance(target, LoRAMixin):
        logger.warning(f"{dotted} is not a LoRA-aware module (lora_enabled=False?); skipping fused group")
        return None, 0

    missing = [n for n in names if n not in qkvs]
    if missing:
        logger.warning(f"fused LoRA on {dotted} missing {missing}; need all of {list(names)} — skipping")
        return None, 0

    A_per = [qkvs[n]["A"] for n in names]
    B_per = [qkvs[n]["B"] for n in names]
    rank_set = {A.shape[0] for A in A_per}
    if len(rank_set) != 1:
        raise ValueError(f"{dotted}: QKV LoRA ranks must match; got {[A.shape[0] for A in A_per]}")
    r = A_per[0].shape[0]
    n = len(names)

    in_dim = target.in_features
    per_out = target.out_features // n  # each of Q/K/V contributes 1/n of the fused output

    n_dev = attn.parallel_config.tensor_parallel.factor
    n_local_heads = attn.n_local_heads
    head_dim = attn.head_dim
    expected_out = n_dev * n_local_heads * head_dim
    if per_out != expected_out:
        # Head padding (padding_config.head_padding > 0) makes the fused output
        # larger than the adapter's per-projection output; the interleave below
        # would misalign. Refuse rather than miswire.
        raise NotImplementedError(
            f"{dotted}: fused-QKV LoRA with head padding is unsupported "
            f"(fused per-proj out={per_out} != heads*head_dim={expected_out}); "
            "use a mesh/TP config where num_heads is divisible by the TP factor."
        )
    for i, (A, B) in enumerate(zip(A_per, B_per)):
        if A.shape != (r, in_dim):
            raise ValueError(f"{dotted}: {names[i]} A must be [{r},{in_dim}]; got {tuple(A.shape)}")
        if B.shape != (per_out, r):
            raise ValueError(f"{dotted}: {names[i]} B must be [{per_out},{r}]; got {tuple(B.shape)}")

    # A stacked on the rank dim → [n*r, in]
    A_fused = torch.cat(A_per, dim=0)

    # Each per-source B ([per_out, r]) is embedded into a [per_out, n*r] tile
    # (zeros for other sources' rank columns), then the n tiles are
    # head-interleaved on the output dim → [n*per_out, n*r], matching the fused
    # base weight's head layout.
    B_tiles: list[torch.Tensor] = []
    for i, B in enumerate(B_per):
        tile = torch.zeros(per_out, n * r, dtype=B.dtype)
        tile[:, i * r : (i + 1) * r] = B
        B_tiles.append(tile)
    B_fused = _head_interleave_lora_B(B_tiles, n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
    assert B_fused.shape == (n * per_out, n * r), f"{dotted}: B_fused {B_fused.shape}"

    # alphas are usually identical across Q/K/V; take the first available.
    alpha = next((alphas[k] for k in names if k in alphas), float(r))
    eff_scale = scale * (alpha / r)

    bank_idx = target.register_lora(A_fused, B_fused, scale=eff_scale, name=name)
    return bank_idx, n * r
