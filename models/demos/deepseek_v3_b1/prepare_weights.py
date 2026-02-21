# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (or per-device slice for single-device
tests), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the device or mesh.

Supports save_weights / load_weights for offline preparation and runtime load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights, OverlappedTensor

# Serialization: manifest version and dtype name mapping
_MANIFEST_VERSION = 1
_DTYPE_TO_STR = {
    ttnn.DataType.BFLOAT4_B: "BFLOAT4_B",
    ttnn.DataType.BFLOAT8_B: "BFLOAT8_B",
    ttnn.DataType.UINT32: "UINT32",
    ttnn.DataType.BFLOAT16: "BFLOAT16",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}

# Fusion group name per field (for grouping by fused_tensor)
_FIELD_TO_FUSION_GROUP: dict[str, str] = {
    "q_a_proj": "q_ab_kv_a",
    "q_b_proj": "q_ab_kv_a",
    "kv_a_proj": "q_ab_kv_a",
    "o_proj": "o_proj_gate_mm_norms",
    "gate_mm": "o_proj_gate_mm_norms",
    "attn_norm": "o_proj_gate_mm_norms",
    "q_norm": "o_proj_gate_mm_norms",
    "kv_norm": "o_proj_gate_mm_norms",
    "ffn_norm": "o_proj_gate_mm_norms",
    "kv_b1_proj": "kv_b12",
    "kv_b2_proj": "kv_b12",
    "shared_gate_proj": "gate_up",
    "shared_up_proj": "gate_up",
}


@dataclass
class DeepSeekV3DenseLayerWeights:
    """Weights for a dense layer (0..first_k_dense_replace-1).

    Has the 3 attention fusion groups and o_proj + norms (no gate_mm).
    """

    # From get_tt_q_ab_proj_and_kv_a_proj_weights
    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor

    # From get_tt_o_proj_and_gate_mm_weights (no gate_mm for dense)
    o_proj: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor

    # From get_tt_kv_b12_proj_weights
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor


@dataclass
class DeepSeekV3MoELayerWeights:
    """Weights for an MoE layer (first_k_dense_replace..num_layers-1).

    Extends dense with gate_mm and shared expert projections.
    """

    # From get_tt_q_ab_proj_and_kv_a_proj_weights
    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor

    # From get_tt_o_proj_and_gate_mm_weights (includes gate_mm)
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor

    # From get_tt_kv_b12_proj_weights
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor

    # From get_tt_gate_up_proj_weights (shared expert)
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor


DeepSeekV3LayerWeights = DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights


@dataclass
class DeepSeekV3Weights:
    """Container for all prepared (fused) layer weights."""

    layers: list[DeepSeekV3LayerWeights]


# Constants for kv_b_proj split (HF stores one matrix; we split into kv_b1 and kv_b2).
_NUM_HEADS = 64
_QK_NOPE_HEAD_DIM = 128
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256


def _key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


def _split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HF kv_b_proj (out_features, in_features) into kv_b1 and kv_b2.

    Supports per-device (16384, 512) and full logical (32768, 512).
    out_features = num_heads * (qk_nope_head_dim + v_head_dim) = num_heads * 256.
    Reshape to (num_heads, 256, 512); first 128 dims are k (b1), last 128 are v (b2).
    Only kv_b2 is transposed for blitz.
    """
    out_features, kv_lora_rank = kv_b_proj.shape
    assert kv_lora_rank == _KV_LORA_RANK
    num_heads = out_features // _KV_B_PROJ_HEAD_DIM
    w = kv_b_proj.reshape(num_heads, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    kv_b1 = w[:, :_QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM:, :].reshape(-1, _KV_LORA_RANK).T.contiguous()
    return kv_b1, kv_b2


def _get_layer_raw_tensors(
    state_dict: dict[str, torch.Tensor], layer_idx: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract and transform raw tensors for one layer from the state dict.

    Expects full logical HF shapes. We transpose HF
    (out_features, in_features) to (K, N); norms unsqueeze(0) to
    (1, W); kv_b_proj is split into kv_b1 and kv_b2 (see _split_kv_b_proj).

    Transformation (HF full logical -> transform -> passed to BlitzDecodeWeights):

        Weight        | HF key (under model.layers.{i}.)     | HF shape      | Transform   | To blitz
        --------------|-------------------------------------|---------------|-------------|------------------
        q_b_proj      | self_attn.q_b_proj.weight            | (24576, 1536) | .T          | (1536, 24576)
        o_proj        | self_attn.o_proj.weight              | (7168, 16384) | .T          | (16384, 7168)
        kv_b_proj     | self_attn.kv_b_proj.weight           | (32768, 512)  | split       | kv_b1, kv_b2
        q_a_proj      | self_attn.q_a_proj.weight            | (1536, 7168)  | .T          | (7168, 1536)
        kv_a_proj     | self_attn.kv_a_proj_with_mqa.weight  | (576, 7168)   | .T          | (7168, 576)
        norms         | input_layernorm, q_a_layernorm, etc. | (7168,), …    | unsqueeze(0)| (1, 7168), …

    MoE-only (gate_mm, shared_gate_proj, shared_up_proj) are read in
    prepare_moe_decoder_layer_weights.

    Returns:
        (q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm).
    """
    q_a = state_dict[_key(layer_idx, "self_attn.q_a_proj.weight")].T.contiguous()
    q_b = state_dict[_key(layer_idx, "self_attn.q_b_proj.weight")].T.contiguous()
    kv_a = state_dict[_key(layer_idx, "self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
    kv_b1, kv_b2 = _split_kv_b_proj(state_dict[_key(layer_idx, "self_attn.kv_b_proj.weight")])
    o_proj = state_dict[_key(layer_idx, "self_attn.o_proj.weight")].T.contiguous()

    attn_norm = state_dict[_key(layer_idx, "input_layernorm.weight")].unsqueeze(0)
    q_norm = state_dict[_key(layer_idx, "self_attn.q_a_layernorm.weight")].unsqueeze(0)
    kv_norm = state_dict[_key(layer_idx, "self_attn.kv_a_layernorm.weight")].unsqueeze(0)
    ffn_norm = state_dict[_key(layer_idx, "post_attention_layernorm.weight")].unsqueeze(0)

    return q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm


def prepare_dense_decoder_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
) -> DeepSeekV3DenseLayerWeights:
    """Prepare fused weights for a single dense decoder layer."""
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = _get_layer_raw_tensors(
        state_dict, layer_idx
    )
    # Single device: state dict is per-device slice. Multi device: full logical; blitz shards internally. No expansion.
    q_a_proj, q_b_proj, kv_a_proj = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a, q_b, kv_a)
    kv_b1_proj, kv_b2_proj = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2)

    # TODO: replace with actual gate_mm when available.
    gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
    o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm)
    o_proj_ot, _gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms

    return DeepSeekV3DenseLayerWeights(
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        o_proj=o_proj_ot,
        attn_norm=attn_norm_ot,
        q_norm=q_norm_ot,
        kv_norm=kv_norm_ot,
        ffn_norm=ffn_norm_ot,
        kv_b1_proj=kv_b1_proj,
        kv_b2_proj=kv_b2_proj,
    )


def prepare_moe_decoder_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
) -> DeepSeekV3MoELayerWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = _get_layer_raw_tensors(
        state_dict, layer_idx
    )
    q_a_proj, q_b_proj, kv_a_proj = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a, q_b, kv_a)
    kv_b1_proj, kv_b2_proj = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2)

    gate_mm = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
    shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
    shared_up = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()

    o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm)
    o_proj_ot, gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
    shared_gate_proj, shared_up_proj = bdw.get_tt_gate_up_proj_weights(shared_gate, shared_up)

    return DeepSeekV3MoELayerWeights(
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        o_proj=o_proj_ot,
        gate_mm=gate_mm_ot,
        attn_norm=attn_norm_ot,
        q_norm=q_norm_ot,
        kv_norm=kv_norm_ot,
        ffn_norm=ffn_norm_ot,
        kv_b1_proj=kv_b1_proj,
        kv_b2_proj=kv_b2_proj,
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
    )


def prepare_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    num_layers: int = 61,
    first_k_dense_replace: int = 3,
) -> DeepSeekV3Weights:
    """Build fused weights from a HuggingFace-style state dict.

    State dict should use full logical HF shapes when device is a mesh (e.g. 4x2);
    internally we shard them across the mesh.

    Args:
        state_dict: Weights keyed by model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*, etc.
        device: MeshDevice to place weights on.
        num_layers: Total number of layers (default 61).
        first_k_dense_replace: Number of dense layers before MoE (default 3).

    Returns:
        DeepSeekV3Weights with one entry per layer; dense vs MoE type by layer index.
    """
    bdw = BlitzDecodeWeights(device)
    layers: list[DeepSeekV3LayerWeights] = []

    for i in range(num_layers):
        is_moe = i >= first_k_dense_replace
        if is_moe:
            layers.append(prepare_moe_decoder_layer_weights(bdw, state_dict, i))
        else:
            layers.append(prepare_dense_decoder_layer_weights(bdw, state_dict, i))

    return DeepSeekV3Weights(layers=layers)


def deallocate_weights(weights: DeepSeekV3Weights) -> None:
    """Release device memory for all fused tensors in prepared weights.

    Call this before loading a new set of weights onto the same device to avoid
    OOM (the original and loaded weights would otherwise both reside on device).
    """
    seen: set[int] = set()
    for layer in weights.layers:
        for _name, ot in _layer_overlapped_tensor_fields(layer):
            fid = id(ot.fused_tensor)
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)


def _core_range_set_to_list(crs: ttnn.CoreRangeSet) -> list[list[list[int]]]:
    """Serialize CoreRangeSet to JSON-serializable list of [[sx, sy], [ex, ey]]."""
    result = []
    for r in crs.ranges():
        start, end = r.start, r.end
        result.append([[start.x, start.y], [end.x, end.y]])
    return result


def _core_range_set_from_list(lst: list[list[list[int]]]) -> ttnn.CoreRangeSet:
    """Deserialize list of [[sx, sy], [ex, ey]] to CoreRangeSet."""
    ranges = [
        ttnn.CoreRange(
            ttnn.CoreCoord(pair[0][0], pair[0][1]),
            ttnn.CoreCoord(pair[1][0], pair[1][1]),
        )
        for pair in lst
    ]
    return ttnn.CoreRangeSet(ranges)


def _overlapped_tensor_to_json(ot: OverlappedTensor) -> dict:
    """Serialize one OverlappedTensor's metadata to a JSON-serializable dict."""
    dtype_str = _DTYPE_TO_STR.get(ot.dtype)
    if dtype_str is None:
        dtype_str = str(ot.dtype)
    return {
        "tensor_shape": list(ot.tensor_shape),
        "shard_shape": list(ot.shard_shape),
        "core_range_set": _core_range_set_to_list(ot.core_range_set),
        "dtype": dtype_str,
        "tile_shape": list(ot.tile_shape),
        "byte_offset": ot.byte_offset,
    }


def _overlapped_tensor_from_dict(
    fused_tensor: ttnn.Tensor,
    d: dict,
) -> OverlappedTensor:
    """Reconstruct one OverlappedTensor from loaded fused tensor and manifest dict."""
    dtype = _STR_TO_DTYPE.get(d["dtype"])
    if dtype is None:
        raise ValueError(f"Unknown dtype in manifest: {d['dtype']}")
    return OverlappedTensor(
        fused_tensor=fused_tensor,
        tensor_shape=tuple(d["tensor_shape"]),
        shard_shape=tuple(d["shard_shape"]),
        core_range_set=_core_range_set_from_list(d["core_range_set"]),
        dtype=dtype,
        tile_shape=tuple(d["tile_shape"]),
        byte_offset=d["byte_offset"],
    )


def _layer_overlapped_tensor_fields(
    layer: DeepSeekV3LayerWeights,
) -> list[tuple[str, OverlappedTensor]]:
    """Return (field_name, OverlappedTensor) for every OverlappedTensor field on the layer."""
    out = []
    for f in fields(layer):
        val = getattr(layer, f.name)
        if isinstance(val, OverlappedTensor):
            out.append((f.name, val))
    return out


def save_layer(
    layer: DeepSeekV3LayerWeights,
    path: str | Path,
    layer_idx: int,
    *,
    hf_model_name: str,
    hf_state_dict_name: str,
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Serialize a single layer to <path>/layer_{layer_idx:03d}/.

    Creates one directory with manifest.json and per-fusion-group .tensorbin files.
    Caller must provide hf_model_name and hf_state_dict_name for the manifest.
    """
    logger.info(f"Saving layer {layer_idx} to {path}...")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layer_dir = path / f"layer_{layer_idx:03d}"
    logger.info(f"Saving layer {layer_idx} to {layer_dir}...")

    layer_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fields_list = _layer_overlapped_tensor_fields(layer)
    by_fused: dict[int, list[tuple[str, OverlappedTensor]]] = {}
    for name, ot in fields_list:
        fid = id(ot.fused_tensor)
        if fid not in by_fused:
            by_fused[fid] = []
        by_fused[fid].append((name, ot))

    logger.info(f"Creating fusion groups for layer {layer_idx}...")
    fusion_groups: dict[str, dict] = {}
    for fid, group_fields in by_fused.items():
        group_name = _FIELD_TO_FUSION_GROUP.get(group_fields[0][0])
        if group_name is None:
            raise KeyError(f"Unknown field for fusion group: {group_fields[0][0]}")
        tensorbin_name = f"{group_name}.tensorbin"

        logger.info(f"Saving {tensorbin_name}...")
        ttnn.dump_tensor(layer_dir / tensorbin_name, group_fields[0][1].fused_tensor)
        logger.info(f"Saved {tensorbin_name} to {layer_dir / tensorbin_name}...")

        fusion_groups[group_name] = {
            "tensorbin": tensorbin_name,
            "fields": {name: _overlapped_tensor_to_json(ot) for name, ot in group_fields},
        }
    logger.info(f"Created fusion groups for layer {layer_idx}...")

    is_moe = isinstance(layer, DeepSeekV3MoELayerWeights)
    manifest = {
        "version": _MANIFEST_VERSION,
        "created_time": created,
        "hf_model_name": hf_model_name,
        "hf_state_dict_name": hf_state_dict_name,
        "device_mesh_shape": list(device_mesh_shape),
        "layer_idx": layer_idx,
        "layer_type": "moe" if is_moe else "dense",
        "fusion_groups": fusion_groups,
    }
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_layer(
    path: str | Path,
    device,
    layer_idx: int,
) -> DeepSeekV3LayerWeights:
    """Deserialize a single layer from <path>/layer_{layer_idx:03d}/."""
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    manifest_path = layer_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("version", 0) > _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")

    layer_type = manifest["layer_type"]
    fusion_groups = manifest["fusion_groups"]

    if layer_type == "dense":
        logger.info(f"Loading all dense tensors for layer {layer_idx}...")
        q_ab = fusion_groups["q_ab_kv_a"]
        fused_q = ttnn.load_tensor(layer_dir / q_ab["tensorbin"], device=device)
        q_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_a_proj"])
        q_b_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_b_proj"])
        kv_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["kv_a_proj"])

        o_grp = fusion_groups["o_proj_gate_mm_norms"]
        fused_o = ttnn.load_tensor(layer_dir / o_grp["tensorbin"], device=device)
        o_proj = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["o_proj"])
        attn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["attn_norm"])
        q_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["q_norm"])
        kv_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["kv_norm"])
        ffn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["ffn_norm"])

        kv_grp = fusion_groups["kv_b12"]
        fused_kv = ttnn.load_tensor(layer_dir / kv_grp["tensorbin"], device=device)
        kv_b1_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b1_proj"])
        kv_b2_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b2_proj"])
        logger.info(f"Loaded all dense tensors for layer {layer_idx}...")

        return DeepSeekV3DenseLayerWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_proj,
            attn_norm=attn_norm,
            q_norm=q_norm,
            kv_norm=kv_norm,
            ffn_norm=ffn_norm,
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
        )
    else:
        logger.info(f"Loading all dense tensors for layer {layer_idx}...")
        q_ab = fusion_groups["q_ab_kv_a"]
        fused_q = ttnn.load_tensor(layer_dir / q_ab["tensorbin"], device=device)
        q_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_a_proj"])
        q_b_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_b_proj"])
        kv_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["kv_a_proj"])

        o_grp = fusion_groups["o_proj_gate_mm_norms"]
        fused_o = ttnn.load_tensor(layer_dir / o_grp["tensorbin"], device=device)
        o_proj = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["o_proj"])
        gate_mm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["gate_mm"])
        attn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["attn_norm"])
        q_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["q_norm"])
        kv_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["kv_norm"])
        ffn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["ffn_norm"])

        kv_grp = fusion_groups["kv_b12"]
        fused_kv = ttnn.load_tensor(layer_dir / kv_grp["tensorbin"], device=device)
        kv_b1_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b1_proj"])
        kv_b2_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b2_proj"])

        gu_grp = fusion_groups["gate_up"]
        fused_gu = ttnn.load_tensor(layer_dir / gu_grp["tensorbin"], device=device)
        shared_gate_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_gate_proj"])
        shared_up_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_up_proj"])
        logger.info(f"Loaded all MoE tensors for layer {layer_idx}...")

        return DeepSeekV3MoELayerWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_proj,
            gate_mm=gate_mm,
            attn_norm=attn_norm,
            q_norm=q_norm,
            kv_norm=kv_norm,
            ffn_norm=ffn_norm,
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            shared_gate_proj=shared_gate_proj,
            shared_up_proj=shared_up_proj,
        )


def save_weights(
    weights: DeepSeekV3Weights,
    path: str | Path,
    *,
    hf_model_name: str,
    hf_state_dict_name: str,
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Serialize all layers to disk. Convenience wrapper around save_layer."""
    path = Path(path)
    for layer_idx, layer in enumerate(weights.layers):
        save_layer(
            layer,
            path,
            layer_idx,
            hf_model_name=hf_model_name,
            hf_state_dict_name=hf_state_dict_name,
            device_mesh_shape=device_mesh_shape,
        )


def load_weights(
    path: str | Path,
    device,
    num_layers: int = 61,
) -> DeepSeekV3Weights:
    """Deserialize layers from disk. Convenience wrapper: load_layer for each index."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Weights path is not a directory: {path}")
    layers = [load_layer(path, device, i) for i in range(num_layers)]
    return DeepSeekV3Weights(layers=layers)
