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

Usage:
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --layer-num 0 --type dense
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --layer-num 3 4 5 6 --type moe
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --type embedding
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --output-path /path/to/cache --type lm_head
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger

import ttnn

# Same mesh device setup as test_prepare_weights.py (bh_2d_mesh_device fixture).
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.prepare_weights import (
    CACHE_TYPE_OVERLAPPED,
    CACHE_TYPE_TENSOR,
    CACHE_TYPE_TENSOR_LIST,
    embedding_fingerprint,
    layer_fingerprints,
    lm_head_fingerprints,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
)
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, TensorCache

NUM_LAYERS = 61
FIRST_K_DENSE_REPLACE = 3
DEVICE_MESH_SHAPE = (4, 2)
HF_MODEL_NAME = "deepseek-ai/DeepSeek-V3"
HF_REVISION = "main"


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
        choices=("dense", "moe", "embedding", "lm_head"),
        required=True,
        help="Cache type: dense (layers 0-2), moe (full layer for 3-60), embedding, lm_head",
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


def _check_on_device(tensor: ttnn.Tensor, name: str) -> bool:
    """Return True if tensor is on device (4x2 grid)."""
    if tensor.storage_type() != ttnn.StorageType.DEVICE:
        logger.error("{}: expected storage DEVICE, got {}", name, tensor.storage_type())
        return False
    return True


def _make_cache_config(output_path: Path) -> tuple[TensorCache, CacheConfig]:
    """Create TensorCache and CacheConfig for the given output path."""
    cache = TensorCache(output_path)
    return cache, CacheConfig(cache=cache, hf_model_id=HF_MODEL_NAME, hf_revision=HF_REVISION)


def _verify_embedding_cache(output_path: Path) -> bool:
    """Verify embedding cache: artifact existence and device load. Returns True if all checks pass."""
    logger.info("Verifying embedding cache...")
    cache, cc = _make_cache_config(output_path)
    fp = embedding_fingerprint(cc, DEVICE_MESH_SHAPE)
    if not cache.has_tensor(fp):
        logger.error("Embedding artifact not found in cache (fingerprint {})", fp.artifact_id()[:12])
        return False
    logger.info("Embedding artifact present")

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            loaded_tensor = cache.load_tensor(fp, device=submesh)
        if loaded_tensor.shape != (129280, 7168):
            logger.error("embedding shape mismatch: expected (129280, 7168), got {}", loaded_tensor.shape)
            return False
        if not _check_on_device(loaded_tensor, "embedding"):
            return False
        logger.info("Device load OK (shape and on-device checks passed)")
    except Exception as e:
        logger.error("Device load failed: {}", e)
        return False

    logger.info("Verify OK")
    return True


def _verify_lm_head_cache(output_path: Path) -> bool:
    """Verify LM head cache: artifact existence and device load. Returns True if all checks pass."""
    logger.info("Verifying lm_head cache...")
    cache, cc = _make_cache_config(output_path)
    fps = lm_head_fingerprints(cc, DEVICE_MESH_SHAPE)
    for name, (fp, _ctype) in fps.items():
        if not cache.has_tensor(fp):
            logger.error("{} artifact not found in cache (fingerprint {})", name, fp.artifact_id()[:12])
            return False
    logger.info("All lm_head artifacts present")

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            lm_head_fp, _ = fps["lm_head"]
            norm_fp, _ = fps["final_norm"]
            lm_head_tensor = cache.load_tensor(lm_head_fp, device=submesh)
            norm_tensor = cache.load_tensor(norm_fp, device=submesh)
        if not _check_on_device(lm_head_tensor, "lm_head"):
            return False
        if not _check_on_device(norm_tensor, "final_norm"):
            return False
        logger.info("Device load OK (on-device checks passed)")
    except Exception as e:
        logger.error("Device load failed: {}", e)
        return False

    logger.info("Verify OK")
    return True


def _verify_cache(output_path: Path, layer_num: int, mode: str) -> bool:
    """Verify existing layer cache: artifact existence and device load. Returns True if all checks pass."""
    logger.info("Verifying layer {} (mode={})...", layer_num, mode)
    is_moe = mode == "moe"
    cache, cc = _make_cache_config(output_path)
    fps = layer_fingerprints(cc, DEVICE_MESH_SHAPE, layer_num, is_moe=is_moe)

    _HAS_FN = {
        CACHE_TYPE_OVERLAPPED: cache.has_overlapped,
        CACHE_TYPE_TENSOR: cache.has_tensor,
        CACHE_TYPE_TENSOR_LIST: cache.has_tensor_list,
    }
    for name, (fp, ctype) in fps.items():
        has_fn = _HAS_FN.get(ctype)
        if has_fn and not has_fn(fp):
            logger.error("Missing artifact '{}' (type={}, fingerprint {})", name, ctype, fp.artifact_id()[:12])
            return False
    logger.info("All {} layer artifacts present", mode)

    logger.info("Loading to device for sanity check...")
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            for name, (fp, ctype) in fps.items():
                if ctype == CACHE_TYPE_OVERLAPPED:
                    views = cache.load_overlapped(fp, device=submesh)
                    for vname, ot in views.items():
                        if not _check_on_device(ot.fused_tensor, f"{name}.{vname}.fused_tensor"):
                            return False
                elif ctype == CACHE_TYPE_TENSOR:
                    t = cache.load_tensor(fp, device=submesh)
                    if not _check_on_device(t, name):
                        return False
                elif ctype == CACHE_TYPE_TENSOR_LIST:
                    tensor_list = cache.load_tensor_list(fp, device=submesh)
                    for i, t in enumerate(tensor_list):
                        if not _check_on_device(t, f"{name}[{i}]"):
                            return False
        logger.info("Device load OK (on-device checks passed)")
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
        else:
            ok = True
            for layer_num in layer_nums:
                if not _verify_cache(output_path, layer_num, mode):
                    ok = False
                    break
        return 0 if ok else 1

    model_path = args.model_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    _cache, cc = _make_cache_config(output_path)

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

            if mode == "dense":
                for layer_num in layer_nums:
                    layer_t0 = time.perf_counter()
                    logger.info("Preparing and caching dense decoder layer {} weights...", layer_num)
                    prepare_dense_layer_weights(submesh, state_dict, layer_num, cache_config=cc)
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - layer_t0)
            elif mode == "moe":
                for layer_num in layer_nums:
                    layer_t0 = time.perf_counter()
                    logger.info("Preparing and caching MoE layer {} weights...", layer_num)
                    prepare_moe_layer_weights(submesh, state_dict, layer_num, cache_config=cc)
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - layer_t0)
            elif mode == "embedding":
                logger.info("Preparing and caching embedding weights...")
                t0 = time.perf_counter()
                prepare_embedding_weights(state_dict, submesh, cache_config=cc)
                logger.info("Embedding done in {:.3f}s", time.perf_counter() - t0)
            elif mode == "lm_head":
                logger.info("Preparing and caching LM head and final norm weights...")
                t0 = time.perf_counter()
                prepare_lm_head_weights(state_dict, submesh, cache_config=cc)
                logger.info("LM head done in {:.3f}s", time.perf_counter() - t0)

    elapsed = time.perf_counter() - total_t0
    if mode in ("embedding", "lm_head"):
        logger.info("Cache generation complete (mode={}) in {:.3f}s", mode, elapsed)
    else:
        logger.info("Cache generation complete for layers {} (mode={}) in {:.3f}s", layer_nums, mode, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
