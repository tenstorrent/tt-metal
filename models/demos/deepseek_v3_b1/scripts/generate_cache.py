# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate weight cache for DeepSeek V3 from HuggingFace safetensors.

Runs in fast dispatch; weights are prepared and saved to disk (no device placement).

Modes:
  dense     - Full dense layer (layers 0-2).
  moe       - Full MoE layer (layers 3-60): attention + shared experts + routed experts.
  embedding - Embedding layer (model.embed_tokens). No --layer-num needed.
  lm_head   - LM head + final RMSNorm. No --layer-num needed.
  mtp       - MTP (Multi-Token Prediction) speculative decode weights (layer 61). No --layer-num needed.

Usage:
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --layer-num 0 --type dense
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --layer-num 3 4 5 6 --type moe
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --type embedding
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --type lm_head
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --type mtp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

import ttnn

# Same mesh device setup as test_prepare_weights.py (bh_2d_mesh_device fixture).
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.prepare_weights import (
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    load_dense_decoder_layer,
    load_embedding_weights,
    load_lm_head_weights,
    load_moe_decoder_layer,
    load_mtp_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
    save_decoder_layer,
    save_embedding_weights,
    save_lm_head_weights,
    save_mtp_weights,
)

NUM_LAYERS = 61
FIRST_K_DENSE_REPLACE = 3
DEVICE_MESH_SHAPE = (4, 2)
MANIFEST_VERSION = 1
HF_MODEL_NAME = "deepseek-ai/DeepSeek-V3"
HF_STATE_DICT_NAME = "lazy"

# Expected tensor_topology() placements for 4x2 mesh (mla_tp=2, moe_tp=8); used by verify device load.
_PLACEMENTS_SHARD_NONE_1 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]  # q_ab_kv_a, o_proj_gate_mm_norms
_PLACEMENTS_SHARD_NONE_0 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)]  # kv_b12
_PLACEMENTS_SHARD_0_1 = [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]  # gate_up, shared_down_proj, dense routed
_PLACEMENTS_REPLICATE = [ttnn.PlacementReplicate()]  # MoE routed experts (per expert)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate weight cache for one or more DeepSeek V3 layers from HuggingFace weights.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local directory containing model.safetensors.index.json and shard files (required for generate; optional for --verify)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory to write cache (layer_NNN subdirs created inside)",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        nargs="+",
        default=None,
        help="Layer index(es) (0-60); one or more, required for dense/moe, ignored for embedding/lm_head",
    )
    parser.add_argument(
        "--type",
        dest="mode",
        choices=("dense", "moe", "embedding", "lm_head", "mtp"),
        required=True,
        help="Cache type: dense (layers 0-2), moe (full layer for 3-60), embedding, lm_head, mtp",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache files for this layer/mode",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing cache (manifest + files, optional device load) instead of generating",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and exit with message on failure."""
    layer_nums = args.layer_num
    mode = args.mode
    output_path = args.output_path.resolve()

    if mode in ("dense", "moe"):
        if not layer_nums:
            logger.error("--layer-num is required for type={}", mode)
            sys.exit(1)
        for layer_num in layer_nums:
            if layer_num < 0 or layer_num >= NUM_LAYERS:
                logger.error("layer-num must be in [0, {}], got {}", NUM_LAYERS - 1, layer_num)
                sys.exit(1)
            if mode == "dense" and layer_num >= FIRST_K_DENSE_REPLACE:
                logger.error(
                    "type=dense requires layer-num < {} (dense layers are 0-{}), got {}",
                    FIRST_K_DENSE_REPLACE,
                    FIRST_K_DENSE_REPLACE - 1,
                    layer_num,
                )
                sys.exit(1)
            if mode == "moe" and layer_num < FIRST_K_DENSE_REPLACE:
                logger.error(
                    "type=moe requires layer-num >= {} (MoE layers are {}-{}), got {}",
                    FIRST_K_DENSE_REPLACE,
                    FIRST_K_DENSE_REPLACE,
                    NUM_LAYERS - 1,
                    layer_num,
                )
                sys.exit(1)

    if args.verify:
        if not output_path.exists():
            logger.error("output-path must exist for verify: {}", output_path)
            sys.exit(1)
        if mode == "embedding":
            manifest_path = output_path / "embedding" / "manifest.json"
            if not manifest_path.is_file():
                logger.error("embedding/manifest.json not found for verify: {}", manifest_path)
                sys.exit(1)
        elif mode == "lm_head":
            manifest_path = output_path / "lm_head" / "manifest.json"
            if not manifest_path.is_file():
                logger.error("lm_head/manifest.json not found for verify: {}", manifest_path)
                sys.exit(1)
        elif mode == "mtp":
            manifest_path = output_path / "layer_061" / "manifest.json"
            if not manifest_path.is_file():
                logger.error("layer_061/manifest.json not found for verify: {}", manifest_path)
                sys.exit(1)
        else:
            for layer_num in layer_nums:
                layer_dir = output_path / f"layer_{layer_num:03d}"
                if not layer_dir.is_dir():
                    logger.error("Layer directory must exist for verify: {}", layer_dir)
                    sys.exit(1)
                manifest_path = layer_dir / "manifest.json"
                if not manifest_path.is_file():
                    logger.error("manifest.json not found for verify: {}", manifest_path)
                    sys.exit(1)
        return

    if args.model_path is None:
        logger.error("--model-path is required for generate (omit --verify to generate)")
        sys.exit(1)
    model_path = args.model_path.resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        logger.error("model-path must contain model.safetensors.index.json; not found at {}", index_path)
        sys.exit(1)

    if not output_path.parent.exists():
        logger.error("output-path parent must exist: {}", output_path.parent)
        sys.exit(1)

    if not args.force:
        if mode == "embedding":
            emb_dir = output_path / "embedding"
            if emb_dir.is_dir() and (emb_dir / "manifest.json").is_file():
                logger.error("embedding cache already exists. Use --force to overwrite.")
                sys.exit(1)
        elif mode == "lm_head":
            lm_dir = output_path / "lm_head"
            if lm_dir.is_dir() and (lm_dir / "manifest.json").is_file():
                logger.error("lm_head cache already exists. Use --force to overwrite.")
                sys.exit(1)
        elif mode == "mtp":
            mtp_layer_dir = output_path / "layer_061"
            if mtp_layer_dir.is_dir() and (mtp_layer_dir / "manifest.json").is_file():
                logger.error("MTP layer cache already exists (layer_061). Use --force to overwrite.")
                sys.exit(1)
        else:
            for layer_num in layer_nums:
                layer_dir = output_path / f"layer_{layer_num:03d}"
                manifest = layer_dir / "manifest.json"
                if manifest.exists():
                    logger.error(
                        "Layer {} cache already exists (manifest.json). Use --force to overwrite.",
                        layer_num,
                    )
                    sys.exit(1)


def _check_file(path: Path, layer_dir: Path) -> bool:
    """Return True if path exists and has size > 0. path can be relative to layer_dir or absolute."""
    if not path.is_absolute():
        path = layer_dir / path
    if not path.exists():
        logger.error("File missing or not readable: {}", path)
        return False
    if path.stat().st_size == 0:
        logger.error("File empty: {}", path)
        return False
    return True


def _check_on_device(tensor: ttnn.Tensor, name: str) -> bool:
    """Return True if tensor is on device (4x2 grid)."""
    if tensor.storage_type() != ttnn.StorageType.DEVICE:
        logger.error("{}: expected storage DEVICE, got {}", name, tensor.storage_type())
        return False
    return True


def _check_topology(
    tensor: ttnn.Tensor,
    expected_placements: list,
    name: str,
) -> bool:
    """Return True if tensor_topology().placements() matches expected for 4x2."""
    actual = list(tensor.tensor_topology().placements())
    if len(actual) != len(expected_placements):
        logger.error(
            "{}: expected {} placements, got {}",
            name,
            len(expected_placements),
            len(actual),
        )
        return False
    for a, e in zip(actual, expected_placements):
        if type(a) != type(e):
            logger.error("{}: placement type mismatch: {} vs {}", name, type(a).__name__, type(e).__name__)
            return False
        if isinstance(e, ttnn.PlacementShard) and a.dim != e.dim:
            logger.error("{}: shard dim mismatch: {} vs {}", name, a.dim, e.dim)
            return False
    return True


def _verify_layer_on_4x2_grid(
    loaded: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights,
) -> bool:
    """Verify all tensors are on device and have correct tensor_topology() for 4x2 mesh. Returns False on first failure."""
    seen_fused: set[int] = set()
    # q_ab_kv_a
    if not _check_on_device(loaded.q_a_proj.fused_tensor, "q_a_proj.fused_tensor"):
        return False
    fid = id(loaded.q_a_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        if not _check_topology(loaded.q_a_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_1, "q_ab_kv_a"):
            return False
    # o_proj_gate_mm_norms
    if not _check_on_device(loaded.o_proj.fused_tensor, "o_proj.fused_tensor"):
        return False
    fid = id(loaded.o_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        if not _check_topology(loaded.o_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_1, "o_proj_gate_mm_norms"):
            return False
    # kv_b12
    if not _check_on_device(loaded.kv_b1_proj.fused_tensor, "kv_b1_proj.fused_tensor"):
        return False
    fid = id(loaded.kv_b1_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        if not _check_topology(loaded.kv_b1_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_0, "kv_b12"):
            return False
    # gate_up
    if not _check_on_device(loaded.shared_gate_proj.fused_tensor, "shared_gate_proj.fused_tensor"):
        return False
    fid = id(loaded.shared_gate_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        if not _check_topology(loaded.shared_gate_proj.fused_tensor, _PLACEMENTS_SHARD_0_1, "gate_up"):
            return False
    # shared_down_proj
    if not _check_on_device(loaded.shared_down_proj, "shared_down_proj"):
        return False
    if not _check_topology(loaded.shared_down_proj, _PLACEMENTS_SHARD_0_1, "shared_down_proj"):
        return False
    # Routed experts
    if isinstance(loaded, DeepSeekV3DenseLayerWeights):
        if not _check_on_device(loaded.routed_gate_proj, "routed_gate_proj"):
            return False
        if not _check_on_device(loaded.routed_up_proj, "routed_up_proj"):
            return False
        if not _check_on_device(loaded.routed_down_proj, "routed_down_proj"):
            return False
        if not _check_topology(loaded.routed_gate_proj, _PLACEMENTS_SHARD_0_1, "routed_gate_proj"):
            return False
        if not _check_topology(loaded.routed_up_proj, _PLACEMENTS_SHARD_0_1, "routed_up_proj"):
            return False
        if not _check_topology(loaded.routed_down_proj, _PLACEMENTS_SHARD_0_1, "routed_down_proj"):
            return False
    else:
        assert isinstance(loaded, DeepSeekV3MoELayerWeights)
        for e in range(len(loaded.routed_gate_proj)):
            if not _check_on_device(loaded.routed_gate_proj[e], f"routed_gate_proj[{e}]"):
                return False
            if not _check_on_device(loaded.routed_up_proj[e], f"routed_up_proj[{e}]"):
                return False
            if not _check_on_device(loaded.routed_down_proj[e], f"routed_down_proj[{e}]"):
                return False
            if not _check_topology(loaded.routed_gate_proj[e], _PLACEMENTS_REPLICATE, f"routed_gate_proj[{e}]"):
                return False
            if not _check_topology(loaded.routed_up_proj[e], _PLACEMENTS_REPLICATE, f"routed_up_proj[{e}]"):
                return False
            if not _check_topology(loaded.routed_down_proj[e], _PLACEMENTS_REPLICATE, f"routed_down_proj[{e}]"):
                return False
    return True


def _verify_embedding_cache(output_path: Path) -> bool:
    """Verify embedding cache: manifest, file existence, and optionally load to device. Returns True if all checks pass."""
    emb_dir = output_path / "embedding"
    manifest_path = emb_dir / "manifest.json"
    logger.info("Verifying embedding cache...")

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load manifest: {}", e)
        return False
    version = manifest.get("version", 0)
    if version > MANIFEST_VERSION:
        logger.error("Unsupported manifest version {} (max {})", version, MANIFEST_VERSION)
        return False
    logger.info("Manifest OK (version={})", version)

    if not _check_file(Path("embedding.tensorbin"), emb_dir):
        return False
    logger.info("All embedding files present")

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            loaded = load_embedding_weights(output_path, submesh)
        if not isinstance(loaded, DeepSeekV3EmbeddingLayerWeights):
            logger.error("Expected DeepSeekV3EmbeddingLayerWeights, got {}", type(loaded).__name__)
            return False
        if loaded.embedding.shape != (129280, 7168):
            logger.error("embedding shape mismatch: expected (129280, 7168), got {}", loaded.embedding.shape)
            return False
        if not _check_on_device(loaded.embedding, "embedding"):
            return False
        logger.info("Device load OK (shape and on-device checks passed)")
    except Exception as e:
        logger.error("Device load failed: {}", e)
        return False

    logger.info("Verify OK")
    return True


def _verify_lm_head_cache(output_path: Path) -> bool:
    """Verify LM head cache: manifest, file existence, and optionally load to device. Returns True if all checks pass."""
    lm_dir = output_path / "lm_head"
    manifest_path = lm_dir / "manifest.json"
    logger.info("Verifying lm_head cache...")

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load manifest: {}", e)
        return False
    version = manifest.get("version", 0)
    if version > MANIFEST_VERSION:
        logger.error("Unsupported manifest version {} (max {})", version, MANIFEST_VERSION)
        return False
    logger.info("Manifest OK (version={})", version)

    if not _check_file(Path("lm_head.tensorbin"), lm_dir):
        return False
    if not _check_file(Path("final_norm.tensorbin"), lm_dir):
        return False
    logger.info("All lm_head files present")

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            loaded = load_lm_head_weights(output_path, submesh)
        if not isinstance(loaded, DeepSeekV3LMHeadWeights):
            logger.error("Expected DeepSeekV3LMHeadWeights, got {}", type(loaded).__name__)
            return False
        if not _check_on_device(loaded.lm_head, "lm_head"):
            return False
        if not _check_on_device(loaded.final_norm, "final_norm"):
            return False
        logger.info("Device load OK (on-device checks passed)")
    except Exception as e:
        logger.error("Device load failed: {}", e)
        return False

    logger.info("Verify OK")
    return True


def _verify_mtp_cache(output_path: Path) -> bool:
    """Verify MTP cache: manifest, file existence, and optionally load to device. Returns True if all checks pass."""
    layer_dir = output_path / "layer_061"
    manifest_path = layer_dir / "manifest.json"
    logger.info("Verifying MTP cache (layer_061)...")

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load manifest: {}", e)
        return False
    version = manifest.get("version", 0)
    if version > MANIFEST_VERSION:
        logger.error("Unsupported manifest version {} (max {})", version, MANIFEST_VERSION)
        return False
    logger.info("Manifest OK (version={})", version)

    for fname in (
        "mtp_h_gamma.tensorbin",
        "mtp_e_gamma.tensorbin",
        "mtp_eh_projection.tensorbin",
    ):
        if not _check_file(Path(fname), layer_dir):
            return False
    logger.info("All MTP files present")

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            loaded = load_mtp_weights(output_path, submesh)
        if not isinstance(loaded, DeepSeekV3MTPWeights):
            logger.error("Expected DeepSeekV3MTPWeights, got {}", type(loaded).__name__)
            return False
        if not _check_on_device(loaded.h_gamma, "mtp.h_gamma"):
            return False
        if not _check_on_device(loaded.e_gamma, "mtp.e_gamma"):
            return False
        if not _check_on_device(loaded.eh_projection, "mtp.eh_projection"):
            return False
        if not _check_on_device(loaded.decoder.q_a_proj.fused_tensor, "mtp.decoder.q_a_proj"):
            return False
        if not _check_on_device(loaded.decoder.shared_down_proj, "mtp.decoder.shared_down_proj"):
            return False
        logger.info("Device load OK (on-device checks passed)")
    except Exception as e:
        logger.error("Device load failed: {}", e)
        return False

    logger.info("Verify OK")
    return True


def _verify_cache(output_path: Path, layer_num: int, mode: str) -> bool:
    """Verify existing cache: manifest, file existence, and optionally load to device. Returns True if all checks pass."""
    layer_dir = output_path / f"layer_{layer_num:03d}"
    manifest_path = layer_dir / "manifest.json"
    logger.info("Verifying layer {} (mode={})...", layer_num, mode)

    # 1. Manifest checks
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load manifest: {}", e)
        return False
    version = manifest.get("version", 0)
    if version > MANIFEST_VERSION:
        logger.error("Unsupported manifest version {} (max {})", version, MANIFEST_VERSION)
        return False
    logger.info("Manifest OK (version={})", version)
    layer_type = manifest.get("layer_type")
    if mode == "dense":
        if layer_type != "dense":
            logger.error("Expected layer_type 'dense', got '{}'", layer_type)
            return False
    else:
        assert mode == "moe"
        if layer_type != "moe":
            logger.error("Expected layer_type 'moe', got '{}'", layer_type)
            return False

    fusion_groups = manifest.get("fusion_groups", {})
    standalone_tensors = manifest.get("standalone_tensors", {})

    # 2. File-existence checks by mode
    if mode == "dense":
        required_fusion = ("q_ab_kv_a", "o_proj_gate_mm_norms", "kv_b12", "gate_up")
        for name in required_fusion:
            if name not in fusion_groups:
                logger.error("Missing fusion_groups['{}']", name)
                return False
            grp = fusion_groups[name]
            tensorbin = grp.get("tensorbin")
            if not tensorbin or not _check_file(Path(tensorbin), layer_dir):
                return False
        required_standalone = ("shared_down_proj", "routed_gate_proj", "routed_up_proj", "routed_down_proj")
        for name in required_standalone:
            if name not in standalone_tensors:
                logger.error("Missing standalone_tensors['{}']", name)
                return False
            if not _check_file(Path(standalone_tensors[name]), layer_dir):
                return False
        logger.info("All dense layer files present")
    else:
        # Full MoE layer: fusion groups + standalone (incl. gate_bias) + routed experts
        for name in ("q_ab_kv_a", "o_proj_gate_mm_norms", "kv_b12", "gate_up"):
            if name not in fusion_groups:
                logger.error("Missing fusion_groups['{}']", name)
                return False
            if not _check_file(Path(fusion_groups[name]["tensorbin"]), layer_dir):
                return False
        for name in ("shared_down_proj", "gate_bias"):
            if name not in standalone_tensors:
                logger.error("Missing standalone_tensors['{}']", name)
                return False
            if not _check_file(Path(standalone_tensors[name]), layer_dir):
                return False
        routed = manifest.get("routed_experts", {})
        num_experts = routed.get("num_experts", 0)
        if num_experts != NUM_ROUTED_EXPERTS:
            logger.error("Expected routed_experts.num_experts={}, got {}", NUM_ROUTED_EXPERTS, num_experts)
            return False
        experts_dir = layer_dir / "experts"
        if not experts_dir.is_dir():
            logger.error("experts/ directory missing: {}", experts_dir)
            return False
        for e in range(NUM_ROUTED_EXPERTS):
            expert_dir = experts_dir / f"e_{e:03d}"
            if not expert_dir.is_dir():
                logger.error("Expert dir missing: {}", expert_dir)
                return False
            for fname in ("gate_proj.tensorbin", "up_proj.tensorbin", "down_proj.tensorbin"):
                if not _check_file(expert_dir / fname, layer_dir):
                    return False
        logger.info("All MoE layer files present (attn+shared+experts)")

    # 3. Optional device load when cache is a complete layer
    if True:
        logger.info("Cache is complete layer; loading to device for sanity check...")
        if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
            os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
        device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
        try:
            with bh_2d_mesh_device_context(device_params) as mesh_device:
                submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
                layer_type = manifest.get("layer_type")
                if layer_type == "dense":
                    loaded = load_dense_decoder_layer(output_path, submesh, layer_num)
                else:
                    loaded = load_moe_decoder_layer(output_path, submesh, layer_num)
            if manifest.get("layer_type") == "dense":
                if not isinstance(loaded, DeepSeekV3DenseLayerWeights):
                    logger.error("Expected DeepSeekV3DenseLayerWeights, got {}", type(loaded).__name__)
                    return False
                # One shape check from manifest
                expected_shape = tuple(fusion_groups["q_ab_kv_a"]["fields"]["q_a_proj"]["tensor_shape"])
                if loaded.q_a_proj.tensor_shape != expected_shape:
                    logger.error(
                        "q_a_proj shape mismatch: expected {}, got {}",
                        expected_shape,
                        loaded.q_a_proj.tensor_shape,
                    )
                    return False
            else:
                if not isinstance(loaded, DeepSeekV3MoELayerWeights):
                    logger.error("Expected DeepSeekV3MoELayerWeights, got {}", type(loaded).__name__)
                    return False
                expected_shape = tuple(fusion_groups["q_ab_kv_a"]["fields"]["q_a_proj"]["tensor_shape"])
                if loaded.q_a_proj.tensor_shape != expected_shape:
                    logger.error(
                        "q_a_proj shape mismatch: expected {}, got {}",
                        expected_shape,
                        loaded.q_a_proj.tensor_shape,
                    )
                    return False
            # Verify tensors are on 4x2 grid with correct topology
            if not _verify_layer_on_4x2_grid(loaded):
                logger.error("4x2 grid / tensor_topology() verification failed")
                return False
            logger.info("Device load OK (type, shape, on-device, and topology checks passed)")
        except Exception as e:
            logger.error("Device load failed: {}", e)
            return False

    logger.info("Verify OK")
    return True


def main() -> int:
    parser = _create_parser()
    args = parser.parse_args()
    _validate_args(args)

    layer_nums = args.layer_num or []
    mode = args.mode
    output_path = args.output_path.resolve()

    if args.verify:
        if mode == "embedding":
            ok = _verify_embedding_cache(output_path)
        elif mode == "lm_head":
            ok = _verify_lm_head_cache(output_path)
        elif mode == "mtp":
            ok = _verify_mtp_cache(output_path)
        else:
            ok = True
            for layer_num in layer_nums:
                if not _verify_cache(output_path, layer_num, mode):
                    ok = False
                    break
        return 0 if ok else 1

    model_path = args.model_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

    logger.info(
        "Generating cache: mode={}, layers={}, model_path={}, output_path={}",
        mode,
        layer_nums if layer_nums else "N/A",
        model_path,
        output_path,
    )

    t0 = time.perf_counter()
    with LazyStateDict(model_path) as state_dict:
        logger.info("LazyStateDict initialized in {:.3f}s", time.perf_counter() - t0)

        if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
            os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
            logger.info("Set TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=30000 (fabric init may be slow)")
        device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
        logger.info("Opening 4x2 mesh device (via bh_2d_mesh_device_context)...")
        t0 = time.perf_counter()
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            logger.info("Mesh device opened in {:.3f}s", time.perf_counter() - t0)

            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            logger.info("Creating BlitzDecodeWeights on 4x2 submesh...")
            bdw = BlitzDecodeWeights(submesh)

            manifest_kw = dict(
                hf_model_name=HF_MODEL_NAME,
                hf_state_dict_name=HF_STATE_DICT_NAME,
                device_mesh_shape=DEVICE_MESH_SHAPE,
            )

            if mode == "dense":
                for layer_num in layer_nums:
                    layer_t0 = time.perf_counter()
                    logger.info("Preparing dense decoder layer {} weights...", layer_num)
                    t0 = time.perf_counter()
                    layer = prepare_dense_layer_weights(bdw, state_dict, layer_num)
                    logger.info("prepare_dense_layer_weights took {:.3f}s", time.perf_counter() - t0)
                    logger.info("Saving dense layer {} to disk...", layer_num)
                    t0 = time.perf_counter()
                    save_decoder_layer(
                        layer,
                        output_path,
                        layer_num,
                        hf_model_name=manifest_kw["hf_model_name"],
                        hf_state_dict_name=manifest_kw["hf_state_dict_name"],
                        device_mesh_shape=manifest_kw["device_mesh_shape"],
                    )
                    logger.info("save_decoder_layer took {:.3f}s", time.perf_counter() - t0)
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - layer_t0)
            elif mode == "moe":
                for layer_num in layer_nums:
                    layer_t0 = time.perf_counter()
                    logger.info("Preparing full MoE layer {} weights...", layer_num)
                    t0 = time.perf_counter()
                    layer = prepare_moe_layer_weights(bdw, state_dict, layer_num)
                    logger.info("prepare_moe_layer_weights took {:.3f}s", time.perf_counter() - t0)
                    logger.info("Saving MoE layer {} to disk...", layer_num)
                    t0 = time.perf_counter()
                    save_decoder_layer(
                        layer,
                        output_path,
                        layer_num,
                        hf_model_name=manifest_kw["hf_model_name"],
                        hf_state_dict_name=manifest_kw["hf_state_dict_name"],
                        device_mesh_shape=manifest_kw["device_mesh_shape"],
                    )
                    logger.info("save_decoder_layer took {:.3f}s", time.perf_counter() - t0)
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - layer_t0)
            elif mode == "embedding":
                logger.info("Preparing embedding weights...")
                t0 = time.perf_counter()
                weights = prepare_embedding_weights(state_dict, submesh)
                logger.info("prepare_embedding_weights took {:.3f}s", time.perf_counter() - t0)
                logger.info("Saving embedding weights...")
                t0 = time.perf_counter()
                save_embedding_weights(weights, output_path, **manifest_kw)
                logger.info("save_embedding_weights took {:.3f}s", time.perf_counter() - t0)
            elif mode == "lm_head":
                logger.info("Preparing LM head and final norm weights...")
                t0 = time.perf_counter()
                weights = prepare_lm_head_weights(state_dict, submesh)
                logger.info("prepare_lm_head_weights took {:.3f}s", time.perf_counter() - t0)
                logger.info("Saving LM head weights...")
                t0 = time.perf_counter()
                save_lm_head_weights(weights, output_path, **manifest_kw)
                logger.info("save_lm_head_weights took {:.3f}s", time.perf_counter() - t0)
            elif mode == "mtp":
                logger.info("Preparing MTP weights...")
                t0 = time.perf_counter()
                weights = prepare_mtp_weights(bdw, state_dict, submesh)
                logger.info("prepare_mtp_weights took {:.3f}s", time.perf_counter() - t0)
                logger.info("Saving MTP weights...")
                t0 = time.perf_counter()
                save_mtp_weights(weights, output_path, **manifest_kw)
                logger.info("save_mtp_weights took {:.3f}s", time.perf_counter() - t0)

    elapsed = time.perf_counter() - total_t0
    if mode in ("embedding", "lm_head", "mtp"):
        logger.info("Cache generation complete (mode={}) in {:.3f}s", mode, elapsed)
    else:
        logger.info("Cache generation complete for layers {} (mode={}) in {:.3f}s", layer_nums, mode, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
