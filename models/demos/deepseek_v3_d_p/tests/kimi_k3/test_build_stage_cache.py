# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build one pipeline stage's TTNN weight cache for the mesh this process opens.

The depth ladder only ever builds a stack starting at layer 0, so a cache for a later stage (layers
12-23 of a two-stage galaxy) has no producer. This builds `[first, first + count)` exactly the way a
rank of that stage constructs it, which is also what writes the tensorbins and the per-layer
`.layer_N.complete` markers, plus the stage's AttnRes queries (the runner builds every weight from the
cache, AttnRes included; the depth ladder loads AttnRes from the checkpoint and never caches it).
Layers already marked complete are not re-read from the checkpoint.

    KIMI_K3_STAGE_FIRST_LAYER=12 KIMI_K3_STAGE_NUM_LAYERS=12 \
        scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_build_stage_cache.py -rs

The cache root comes from `TT_KIMI_K3_PREFILL_TTNN_CACHE` (see `kimi_k3/weights.py:cache_root`).
"""

import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_x_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import resolve_checkpoint
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import AttnResWeights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
    cache_root,
    layer_is_cached,
    load_layer_state_dict_cached,
    load_tensors,
)

SP_AXIS, TP_AXIS = 0, 1
FIRST_LAYER = int(os.getenv("KIMI_K3_STAGE_FIRST_LAYER", "0"))
NUM_LAYERS = int(os.getenv("KIMI_K3_STAGE_NUM_LAYERS", "12"))
# The depth the stage belongs to; only decides whether this stage owns the final norm.
TOTAL_LAYERS = int(os.getenv("KIMI_K3_STAGE_TOTAL_LAYERS", "24"))
SEQ_LEN = int(os.getenv("KIMI_K3_TEST_SEQ_LEN", "5120"))

PLACEMENTS = [
    pytest.param(
        (4, 4),
        torus_x_device_params(l1_small_size=KimiK3Config.L1_SMALL_SIZE),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-x-4x4",
    ),
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": KimiK3Config.L1_SMALL_SIZE},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    ),
]


@pytest.mark.timeout(4 * 3600)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_build_stage_cache(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("needs KIMI_K3_CKPT (dequantized checkpoint)")
    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)
    assert cache is not None, "no writable cache root; set TT_KIMI_K3_PREFILL_TTNN_CACHE"

    layers = range(FIRST_LAYER, FIRST_LAYER + NUM_LAYERS)
    missing = [idx for idx in layers if not layer_is_cached(cache, idx)]
    attn_res_cached = AttnResWeights.check_cache_complete(
        cache, "attn_res", num_layers=NUM_LAYERS, first_layer_idx=FIRST_LAYER, dtype=ttnn.bfloat16
    )
    logger.info(f"stage cache {cache}: layers {list(layers)}, missing {missing}, attn_res cached {attn_res_cached}")
    if not missing and attn_res_cached:
        return

    is_first = FIRST_LAYER == 0
    is_last = FIRST_LAYER + NUM_LAYERS == TOTAL_LAYERS
    model = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    state_dict = {
        "embed_weight": model["embed_weight"].float(),
        "norm_weight": model["norm_weight"],
        "layers": [load_layer_state_dict_cached(checkpoint, idx, cache) for idx in layers],
        # Keyed by GLOBAL layer; the transformer takes its window with `first_layer_idx`.
        "attn_res_weights": load_attn_res_state_dict(checkpoint, FIRST_LAYER + NUM_LAYERS, root),
        "attn_res_prefix": root,
    }
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        state_dict,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        first_layer_idx=FIRST_LAYER,
        is_first_rank=is_first,
        is_last_rank=is_last,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )
    still_missing = [idx for idx in layers if not layer_is_cached(cache, idx)]
    assert not still_missing, f"layers not marked complete after the build: {still_missing}"
    assert AttnResWeights.check_cache_complete(
        cache, "attn_res", num_layers=NUM_LAYERS, first_layer_idx=FIRST_LAYER, dtype=ttnn.bfloat16
    ), "AttnRes cache still incomplete after the build"
