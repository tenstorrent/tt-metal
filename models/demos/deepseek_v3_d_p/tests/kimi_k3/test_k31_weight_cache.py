# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Builds one layer of Kimi-K3.1's TTNN weight cache (8x4, TP axis 1) from the real checkpoint.

Not a test of behaviour, a producer: constructing `TtKimiK3Transformer` with real weights and a
`weight_cache_path` writes every tensorbin and, per layer, the completion marker
(`mark_layer_cached`). Nothing runs forward. The cache is the one the K3.1 runner reads
(`cache_root(..., KimiK31Adapter)`, keyed by checkpoint), so a layer already marked is a cache hit and
costs only its load; a marked attention layer missing its indexer is rebuilt attention-only.

The layer is `$K31_CACHE_LAYER`, one per run, so a 93-layer build runs as short device jobs that release
the device lock between them; a cold MoE layer holds ~59 GB of routed experts in host memory. Run it under
`umask 0002`: the cache is on a shared tree, and only files 664 / directories 2775 are readable by others.

    umask 0002; K31_CACHE_LAYER=3 scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_weight_cache.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
    cache_root,
    layer_is_cached,
    load_layer_state_dict_cached,
    load_tensors,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

SP_AXIS, TP_AXIS = 0, 1
# Weights do not depend on the sequence length; this only sizes construction-time buffers.
SEQ_LEN = 5120
LAYER_ENV = "K31_CACHE_LAYER"


def _layer() -> int | None:
    spec = os.getenv(LAYER_ENV)
    if not spec:
        return None
    if not spec.isdigit() or int(spec) >= KimiK31Config.NUM_LAYERS:
        raise ValueError(f"${LAYER_ENV}={spec!r} is not one layer index in 0..{KimiK31Config.NUM_LAYERS - 1}")
    return int(spec)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": KimiK31Config.L1_SMALL_SIZE},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        )
    ],
    indirect=True,
)
def test_build_k31_weight_cache(mesh_device, device_params):
    layer = _layer()
    if layer is None:
        pytest.skip(f"a cache build: set ${LAYER_ENV} to the layer to build")
    checkpoint = Path(os.getenv(KimiK31Adapter.env_var) or KimiK31Adapter.hf_model_default)
    if not (checkpoint / "model.safetensors.index.json").is_file():
        pytest.skip(f"K3.1 checkpoint not on this box ({checkpoint}); set ${KimiK31Adapter.env_var}")
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS, KimiK31Adapter)
    assert cache is not None, f"caching is disabled (${KimiK31Adapter.ttnn_cache_env} is empty)"
    logger.info(f"K3.1 weight cache: building layer {layer} into {cache}")

    root = resolve_model_root(checkpoint)
    is_last = layer == KimiK31Config.NUM_LAYERS - 1
    # Every layer is built as a first rank, because a non-first rank must start on an AttnRes block
    # edge. The model-level tensors are therefore always passed in full: each is written once, by the
    # first layer to build it, and is a cache hit (its torch tensor ignored) for every later one.
    globals_ = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    state_dict = {
        "embed_weight": globals_["embed_weight"].float(),
        "norm_weight": globals_["norm_weight"] if is_last else None,
        "attn_res_weights": load_attn_res_state_dict(checkpoint, KimiK31Config.NUM_LAYERS, root),
        "attn_res_prefix": root,
        "layers": [load_layer_state_dict_cached(checkpoint, layer, cache, model_cfg=KimiK31Config)],
    }

    TtKimiK3Transformer(
        mesh_device,
        kimi_k3_1_hf_config(max_seq=SEQ_LEN),
        KimiK31Config,
        state_dict,
        num_layers=1,
        seq_len=SEQ_LEN,
        first_layer_idx=layer,
        is_first_rank=True,
        is_last_rank=is_last,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )

    assert layer_is_cached(cache, layer, KimiK31Config), f"layer {layer} has no complete cache after the build"
    # The runner reads `{root}/{name}_bh_32dev/{sp}x{tp}`; point it at this checkpoint's cache, as K3's is.
    # Re-pointed when it names another checkpoint's cache (a re-export gets a new id), so the runner never
    # loads stale tensors; until this checkpoint's cache is complete, its completeness check then fails.
    link = cache.parent.parent / f"{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    if link.is_symlink() and link.resolve() != cache.resolve():
        logger.warning(f"re-pointing {link} from {link.resolve()} to {cache}")
        link.unlink()
    if not link.is_symlink():
        link.symlink_to(cache.relative_to(cache.parent.parent))
    logger.info(f"K3.1 weight cache: layer {layer} complete at {cache}")
