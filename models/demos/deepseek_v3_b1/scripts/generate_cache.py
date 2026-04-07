# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Warm TensorCache on disk for DeepSeek V3 B1 (CAS layout under ``cache-root/objects/``).

Legacy ``layer_NNN/manifest.json`` + per-layer ``.tensorbin`` trees are not supported; use this
script or runtime ``CacheWeightProvider`` to populate the cache.

Uses the same ``CacheContext`` fields as ``CacheWeightProvider`` (``schema_version``, ``hf_model_id``,
``hf_revision``, ``transform_version``, ``mesh_shape``).

Examples:

  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --layer-num 0 --type dense
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --layer-num 3 4 5 --type moe
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --type embedding
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --type lm_head
  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --type mtp

  python generate_cache.py --model-path /path/to/DeepSeek-V3 --cache-root /path/to/cache --type embedding --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger

import ttnn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.weights.adapter import (
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
)
from models.demos.deepseek_v3_b1.weights.catalog import CURRENT_TRANSFORM_VERSION
from models.demos.deepseek_v3_b1.weights.types import (
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3MoELayerWeights,
)

NUM_LAYERS = 62
FIRST_K_DENSE_REPLACE = 3
DEVICE_MESH_SHAPE = (4, 2)


def _cache_config(
    cache_root: Path,
    submesh: ttnn.MeshDevice,
    *,
    hf_model_id: str,
    hf_revision: str,
    schema_version: int,
) -> CacheConfig:
    return CacheConfig(
        cache=TensorCache(cache_root),
        context=CacheContext(
            schema_version=schema_version,
            hf_model_id=hf_model_id,
            hf_revision=hf_revision,
            transform_version=CURRENT_TRANSFORM_VERSION,
            mesh_shape=(submesh.shape[0], submesh.shape[1]),
        ),
    )


def _release_decoder_layer(layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights) -> None:
    seen: set[int] = set()
    for f in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "o_proj",
        "attn_norm",
        "q_norm",
        "kv_norm",
        "ffn_norm",
        "kv_b1_proj",
        "kv_b2_proj",
        "shared_gate_proj",
        "shared_up_proj",
    ):
        ot = getattr(layer, f, None)
        if ot is not None and hasattr(ot, "fused_tensor"):
            fid = id(ot.fused_tensor)
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
    ttnn.deallocate(layer.shared_down_proj, force=True)
    if isinstance(layer, DeepSeekV3MoELayerWeights):
        gm = layer.gate_mm
        gid = id(gm.fused_tensor)
        if gid not in seen:
            seen.add(gid)
            ttnn.deallocate(gm.fused_tensor, force=True)
        ttnn.deallocate(layer.gate_bias, force=True)
        for t in layer.routed_gate_proj:
            ttnn.deallocate(t, force=True)
        for t in layer.routed_up_proj:
            ttnn.deallocate(t, force=True)
        for t in layer.routed_down_proj:
            ttnn.deallocate(t, force=True)
    else:
        ttnn.deallocate(layer.routed_gate_proj, force=True)
        ttnn.deallocate(layer.routed_up_proj, force=True)
        ttnn.deallocate(layer.routed_down_proj, force=True)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm TensorCache for DeepSeek V3 B1 weights (prepare_* + CAS on disk).",
    )
    cache = parser.add_mutually_exclusive_group(required=True)
    cache.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="TensorCache local root (contains objects/ after warming).",
    )
    cache.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Deprecated alias for --cache-root.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="HF model directory (model.safetensors.index.json + shards). Required unless --verify-only checks are skipped.",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        nargs="+",
        default=None,
        help="Layer index(es) 0..61; required for dense/moe.",
    )
    parser.add_argument(
        "--type",
        dest="mode",
        choices=("dense", "moe", "embedding", "lm_head", "mtp"),
        required=True,
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="Cache key hf_model_id (default: basename of --model-path).",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="local",
        help="Cache key hf_revision (default: local).",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        default=1,
        help="Cache schema_version (default: 1).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After warm: load twice via CacheWeightProvider (second load is a cache hit).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip warm; only run the double-load check (use on an already-populated cache-root).",
    )
    return parser


def _cache_root_from_args(args: argparse.Namespace) -> Path:
    if args.cache_root is not None:
        return args.cache_root.resolve()
    assert args.output_path is not None
    return args.output_path.resolve()


def _validate_args(args: argparse.Namespace, cache_root: Path) -> None:
    mode = args.mode
    layer_nums = args.layer_num or []

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
                    "type=dense requires layer-num < {} (dense layers 0..{}), got {}",
                    FIRST_K_DENSE_REPLACE,
                    FIRST_K_DENSE_REPLACE - 1,
                    layer_num,
                )
                sys.exit(1)
            if mode == "moe" and layer_num < FIRST_K_DENSE_REPLACE:
                logger.error(
                    "type=moe requires layer-num >= {} (MoE layers {}..{}), got {}",
                    FIRST_K_DENSE_REPLACE,
                    FIRST_K_DENSE_REPLACE,
                    NUM_LAYERS - 1,
                    layer_num,
                )
                sys.exit(1)

    if args.model_path is None:
        logger.error("--model-path is required")
        sys.exit(1)
    model_path = args.model_path.resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        logger.error("model-path must contain model.safetensors.index.json; not found at {}", index_path)
        sys.exit(1)

    cache_root.mkdir(parents=True, exist_ok=True)


def _ensure_fabric_env() -> None:
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
        logger.info("Set TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=30000")


def _warm(
    *,
    model_path: Path,
    cache_root: Path,
    mode: str,
    layer_nums: list[int],
    hf_model_id: str,
    hf_revision: str,
    schema_version: int,
) -> None:
    _ensure_fabric_env()
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    with LazyStateDict(model_path) as state_dict:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            cfg = _cache_config(
                cache_root,
                submesh,
                hf_model_id=hf_model_id,
                hf_revision=hf_revision,
                schema_version=schema_version,
            )
            bdw = BlitzDecodeWeights(submesh)
            if mode == "dense":
                for layer_num in layer_nums:
                    t0 = time.perf_counter()
                    logger.info("Warming TensorCache: dense layer {}", layer_num)
                    prepare_dense_layer_weights(
                        bdw,
                        state_dict,
                        layer_num,
                        move_to_device=True,
                        cache_config=cfg,
                    )
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - t0)
            elif mode == "moe":
                for layer_num in layer_nums:
                    t0 = time.perf_counter()
                    logger.info("Warming TensorCache: MoE layer {}", layer_num)
                    prepare_moe_layer_weights(
                        bdw,
                        state_dict,
                        layer_num,
                        num_routed_experts=NUM_ROUTED_EXPERTS,
                        move_to_device=True,
                        cache_config=cfg,
                    )
                    logger.info("Layer {} done in {:.3f}s", layer_num, time.perf_counter() - t0)
            elif mode == "embedding":
                t0 = time.perf_counter()
                logger.info("Warming TensorCache: embedding")
                prepare_embedding_weights(state_dict, submesh, move_to_device=True, cache_config=cfg)
                logger.info("Embedding done in {:.3f}s", time.perf_counter() - t0)
            elif mode == "lm_head":
                t0 = time.perf_counter()
                logger.info("Warming TensorCache: lm_head")
                prepare_lm_head_weights(state_dict, submesh, move_to_device=True, cache_config=cfg)
                logger.info("LM head done in {:.3f}s", time.perf_counter() - t0)
            else:
                assert mode == "mtp"
                t0 = time.perf_counter()
                logger.info("Warming TensorCache: mtp")
                prepare_mtp_weights(state_dict, submesh, move_to_device=True, cache_config=cfg)
                logger.info("MTP done in {:.3f}s", time.perf_counter() - t0)


def _verify(
    *,
    model_path: Path,
    cache_root: Path,
    mode: str,
    layer_nums: list[int],
    hf_model_id: str,
    hf_revision: str,
    schema_version: int,
) -> bool:
    if not cache_root.is_dir():
        logger.error("cache-root must exist for --verify: {}", cache_root)
        return False
    _ensure_fabric_env()
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    try:
        provider = CacheWeightProvider(
            cache_root,
            model_path,
            hf_model_id=hf_model_id,
            hf_revision=hf_revision,
            schema_version=schema_version,
        )
    except Exception as e:
        logger.error("CacheWeightProvider init failed: {}", e)
        return False

    try:
        with bh_2d_mesh_device_context(device_params) as mesh_device:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*DEVICE_MESH_SHAPE))
            if mode == "embedding":
                w = provider.load_embedding(submesh)
                shape = w.embedding.shape
                ttnn.deallocate(w.embedding, force=True)
                w2 = provider.load_embedding(submesh)
                if w2.embedding.shape != shape:
                    logger.error("embedding shape mismatch on second load: {} vs {}", w2.embedding.shape, shape)
                    return False
            elif mode == "lm_head":
                w = provider.load_lm_head(submesh)
                ls, ns = w.lm_head.shape, w.final_norm.shape
                ttnn.deallocate(w.lm_head, force=True)
                ttnn.deallocate(w.final_norm, force=True)
                w2 = provider.load_lm_head(submesh)
                if w2.lm_head.shape != ls or w2.final_norm.shape != ns:
                    logger.error("lm_head second load shape mismatch")
                    return False
            elif mode == "mtp":
                w = provider.load_mtp(submesh)
                hs, es, ehs = w.h_gamma.shape, w.e_gamma.shape, w.eh_projection.shape
                ttnn.deallocate(w.h_gamma, force=True)
                ttnn.deallocate(w.e_gamma, force=True)
                ttnn.deallocate(w.eh_projection, force=True)
                w2 = provider.load_mtp(submesh)
                if w2.h_gamma.shape != hs or w2.e_gamma.shape != es or w2.eh_projection.shape != ehs:
                    logger.error("mtp second load shape mismatch")
                    return False
            elif mode == "dense":
                for lid in layer_nums:
                    w = provider.load_dense_layer(lid, submesh)
                    q_shape = w.q_a_proj.tensor_shape
                    _release_decoder_layer(w)
                    w2 = provider.load_dense_layer(lid, submesh)
                    if w2.q_a_proj.tensor_shape != q_shape:
                        logger.error("dense layer {} second load shape mismatch", lid)
                        return False
                    _release_decoder_layer(w2)
            else:
                assert mode == "moe"
                for lid in layer_nums:
                    w = provider.load_moe_layer(lid, submesh)
                    q_shape = w.q_a_proj.tensor_shape
                    _release_decoder_layer(w)
                    w2 = provider.load_moe_layer(lid, submesh)
                    if w2.q_a_proj.tensor_shape != q_shape:
                        logger.error("moe layer {} second load shape mismatch", lid)
                        return False
                    _release_decoder_layer(w2)
    except Exception as e:
        logger.error("Verify failed: {}", e)
        return False

    objs = cache_root / "objects"
    n = len(list(objs.rglob("data.tensorbin"))) if objs.is_dir() else 0
    logger.info("Verify OK (TensorCache objects with data.tensorbin: {})", n)
    return True


def main() -> int:
    parser = _create_parser()
    args = parser.parse_args()
    if args.verify and args.verify_only:
        logger.error("Use either --verify (warm then check) or --verify-only (check only), not both.")
        return 2
    cache_root = _cache_root_from_args(args)
    _validate_args(args, cache_root)

    model_path = args.model_path.resolve()
    hf_model_id = args.hf_model_id or model_path.name
    layer_nums = list(args.layer_num or [])

    if not args.verify_only:
        t_all = time.perf_counter()
        _warm(
            model_path=model_path,
            cache_root=cache_root,
            mode=args.mode,
            layer_nums=layer_nums,
            hf_model_id=hf_model_id,
            hf_revision=args.hf_revision,
            schema_version=args.schema_version,
        )
        logger.info("Warm complete in {:.3f}s", time.perf_counter() - t_all)

    if args.verify or args.verify_only:
        ok = _verify(
            model_path=model_path,
            cache_root=cache_root,
            mode=args.mode,
            layer_nums=layer_nums,
            hf_model_id=hf_model_id,
            hf_revision=args.hf_revision,
            schema_version=args.schema_version,
        )
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
