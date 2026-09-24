# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-decoder-layer weight loading for DeepSeek-V4-Flash (copied from tt-blaze blaze/weights/deepseek_v4_flash/layer_weights.py; the DSpark ``mtp.*`` drafter branch dropped -- the prefill never loads it).

**The 43 decoder layers are NOT uniform.** Read straight off the checkpoint
index, they fall into four shapes, and a provider that assumes one shape will
silently load the wrong key set:

    layers 0, 1        (2)  SWA, no compressor          hash routing (gate.tid2eid)
    layer  2           (1)  compressor + indexer        hash routing
    layers 3,5,..,41  (20)  compressor, no indexer      learned gate (gate.bias)
    layers 4,6,..,42  (20)  compressor + indexer        learned gate

``num_hash_layers=3`` in config.json is what splits hash from learned routing
(layers 0..2), and 21 of the 43 layers carry an ``attn.indexer`` (DSA sparse
attention). :func:`classify_layer` derives all of this from the index rather
than hardcoding the pattern, so a checkpoint revision that moves a boundary
surfaces as a changed classification instead of a missing-key crash.

Storage: see :mod:`.dequant`. Routed experts are
FP4, attention projections and the shared expert are FP8-block, everything else
is plain bf16/fp32.

**Sizing.** One layer's 256 routed experts are ~12.9 GB at bf16 (256 x 3 x
2048 x 4096 x 2 B), so they are never returned as a dict --
:func:`iter_routed_experts_bf16` streams them one expert at a time and the
caller is expected to move each to device and drop the host copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .dequant import fp4_e2m1_dequant_to_bf16, fp8_e4m3_dequant_to_bf16, read_tensors, read_weight_map

NUM_DECODER_LAYERS = 43
NUM_HASH_LAYERS = 3

# Attention projections stored FP8-E4M3 with 128x128 block scales.
_ATTN_FP8 = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
# w1 -> gate, w2 -> down, w3 -> up (upstream naming), the checkpoint's
# own convention (inference/model.py).
_FFN_TAGS = {"gate": "w1", "down": "w2", "up": "w3"}


@dataclass(frozen=True)
class FlashLayerKind:
    """What a given decoder layer actually contains."""

    layer_id: int
    has_compressor: bool
    has_indexer: bool
    routing: str  # "hash" (gate.tid2eid) | "learned" (gate.weight + gate.bias)

    @property
    def is_hca_only(self) -> bool:
        """True for the variant the validated Dsv4HcaLayer op models (odd layers 3..41)."""
        return self.has_compressor and not self.has_indexer and self.routing == "learned"


FLASH_NUM_LAYERS = 43


def layer_key_prefix(layer_id: int) -> str:
    """Checkpoint key prefix (no trailing dot) of decoder layer ``layer_id``: ``layers.N``."""
    layer_id = int(layer_id)
    if not 0 <= layer_id < FLASH_NUM_LAYERS:
        raise ValueError(
            f"layer {layer_id} is not one of the {FLASH_NUM_LAYERS} decoder layers (the mtp.* drafter is not loaded here)"
        )
    return f"layers.{layer_id}"


def classify_layer(model_dir: str | Path, layer_id: int, *, weight_map=None) -> FlashLayerKind:
    """Derive a layer's shape from the checkpoint index (authoritative)."""
    wm = weight_map if weight_map is not None else read_weight_map(model_dir)
    p = f"{layer_key_prefix(layer_id)}."
    keys = {k[len(p) :] for k in wm if k.startswith(p) and ".experts." not in k}
    if not keys:
        raise KeyError(f"checkpoint has no layer {layer_id}")
    return FlashLayerKind(
        layer_id=layer_id,
        has_compressor=any(k.startswith("attn.compressor.") for k in keys),
        has_indexer=any(k.startswith("attn.indexer.") for k in keys),
        routing="hash" if "ffn.gate.tid2eid" in keys else "learned",
    )


def layer_core_keys(kind: FlashLayerKind) -> list[str]:
    """Every non-expert key for one layer, given its classification."""
    p = f"{layer_key_prefix(kind.layer_id)}"
    keys = [f"{p}.attn.attn_sink", f"{p}.attn.q_norm.weight", f"{p}.attn.kv_norm.weight"]
    keys += [f"{p}.attn.{n}.{s}" for n in _ATTN_FP8 for s in ("weight", "scale")]
    keys += [f"{p}.attn_norm.weight", f"{p}.ffn_norm.weight", f"{p}.ffn.gate.weight"]
    keys += [f"{p}.ffn.shared_experts.{t}.{s}" for t in ("w1", "w2", "w3") for s in ("weight", "scale")]
    keys += [f"{p}.hc_{a}_{b}" for a in ("attn", "ffn") for b in ("base", "fn", "scale")]
    keys += [f"{p}.ffn.gate.tid2eid"] if kind.routing == "hash" else [f"{p}.ffn.gate.bias"]
    if kind.has_compressor:
        keys += [f"{p}.attn.compressor.{s}" for s in ("ape", "norm.weight", "wgate.weight", "wkv.weight")]
    if kind.has_indexer:
        keys += [f"{p}.attn.indexer.compressor.{s}" for s in ("ape", "norm.weight", "wgate.weight", "wkv.weight")]
        keys += [f"{p}.attn.indexer.weights_proj.weight"]
        keys += [f"{p}.attn.indexer.wq_b.weight", f"{p}.attn.indexer.wq_b.scale"]
    return keys


def load_layer_core_bf16(model_dir: str | Path, layer_id: int, *, weight_map=None) -> dict:
    """Load + dequantize one layer's non-expert weights.

    Returned keys are the checkpoint names with the ``layers.N.`` prefix
    stripped, ``.scale`` entries consumed by the dequant, and the shared expert
    renamed/transposed to the ``gate``/``up``/``down`` convention used by the
    dsv4 goldens (gate/up as K x N, down as N x K). Attention projections keep
    their stored orientation, which is what the HCA ops expect.

    Also returns ``"kind"``: the :class:`FlashLayerKind` it was loaded as.
    """
    wm = weight_map if weight_map is not None else read_weight_map(model_dir)
    kind = classify_layer(model_dir, layer_id, weight_map=wm)
    p = f"{layer_key_prefix(layer_id)}"
    raw = read_tensors(model_dir, layer_core_keys(kind), weight_map=wm)

    out: dict = {"kind": kind}

    # FP8 attention projections, stored orientation preserved.
    for n in _ATTN_FP8:
        out[f"attn.{n}"] = fp8_e4m3_dequant_to_bf16(raw[f"{p}.attn.{n}.weight"], raw[f"{p}.attn.{n}.scale"])
    if kind.has_indexer:
        out["attn.indexer.wq_b"] = fp8_e4m3_dequant_to_bf16(
            raw[f"{p}.attn.indexer.wq_b.weight"], raw[f"{p}.attn.indexer.wq_b.scale"]
        )

    # Shared expert: dequantize then transpose into the golden's layout.
    for tag, sub in _FFN_TAGS.items():
        w = fp8_e4m3_dequant_to_bf16(
            raw[f"{p}.ffn.shared_experts.{sub}.weight"], raw[f"{p}.ffn.shared_experts.{sub}.scale"]
        )
        out[f"shared.{tag}"] = w.t().contiguous()

    # Everything else passes through untouched.
    consumed = {f"{p}.attn.{n}.{s}" for n in _ATTN_FP8 for s in ("weight", "scale")}
    consumed |= {f"{p}.ffn.shared_experts.{t}.{s}" for t in ("w1", "w2", "w3") for s in ("weight", "scale")}
    consumed |= {f"{p}.attn.indexer.wq_b.weight", f"{p}.attn.indexer.wq_b.scale"}
    for k, v in raw.items():
        if k not in consumed:
            out[k[len(p) + 1 :]] = v
    return out


def iter_routed_experts_bf16(
    model_dir: str | Path,
    layer_id: int,
    *,
    expert_ids=None,
    num_experts: int = 256,
    weight_map=None,
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Stream one layer's routed experts as ``(expert_id, {gate, up, down})`` bf16.

    A generator on purpose: the full set is ~12.9 GB per layer, so the caller
    moves each expert to device and lets the host copy fall out of scope.
    ``gate``/``up`` come back K x N and ``down`` N x K, matching the shared
    expert and the dsv4 goldens.
    """
    wm = weight_map if weight_map is not None else read_weight_map(model_dir)
    ids = range(num_experts) if expert_ids is None else expert_ids
    for e in ids:
        p = f"{layer_key_prefix(layer_id)}.ffn.experts.{e}"
        keys = [f"{p}.{sub}.{s}" for sub in ("w1", "w2", "w3") for s in ("weight", "scale")]
        raw = read_tensors(model_dir, keys, weight_map=wm)
        got = {}
        for tag, sub in _FFN_TAGS.items():
            got[tag] = fp4_e2m1_dequant_to_bf16(raw[f"{p}.{sub}.weight"], raw[f"{p}.{sub}.scale"]).t().contiguous()
        yield e, got
