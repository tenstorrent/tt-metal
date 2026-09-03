# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Populate the TTNN weight cache for every Kimi-K3 layer, one layer at a time.

Not a test — a cache generator that happens to need the `mesh_device` fixture, because the mesh
mapper needs the mesh shape. It asserts nothing about the model and runs no forward pass.

**Why one layer at a time.** The full 93-layer model does not fit in Galaxy DRAM: 92 MoE layers x 28
experts/chip x 33.0 M params at bfloat4_b is 47.9 GB/chip of routed experts against 34.2 GB/chip
available. Building the whole stack to populate its cache is therefore impossible. A single layer is
~739 MiB/chip, so building and freeing one at a time is bounded regardless of depth.

**Why through the real constructors** rather than the `build_ttnn_cache` statics. Those statics
exist and are host-only, but assembling the right call for each of norms / MLA-or-KDA / MoE-or-dense
means restating every dtype, shard dim and cache-name prefix the loader will later expect. Getting
one of them subtly wrong produces a cache that loads without error and yields a wrong model —
`ttnn.as_tensor` writes whatever tensor it is handed when the file is absent, and
`TtDistributedRmsNorm` hands it `torch.empty` (#54841). Driving the same constructor the loader uses
makes that class of mistake impossible: whatever it writes is by definition what it later reads.

The per-layer completion marker is written by `TtKimiK3Transformer` itself, only after the layer has
been built end to end from real weights, so an interrupted run keeps every layer it finished.
"""

import gc
from pathlib import Path

import pytest
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import resolve_checkpoint
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import PLACEMENTS, SEQ_LEN, SP_AXIS, TP_AXIS
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
    cache_root,
    layer_is_cached,
    load_layer_state_dict_cached,
    load_tensors,
)


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_build_full_cache(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("needs KIMI_K3_HF_MODEL")
    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)
    assert cache is not None, "weight caching is disabled; unset TT_KIMI_K3_PREFILL_TTNN_CACHE"
    logger.info(f"cache root: {cache}")

    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    model_bits = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    already = [n for n in range(KimiK3Config.NUM_LAYERS) if layer_is_cached(cache, n)]
    todo = [n for n in range(KimiK3Config.NUM_LAYERS) if not layer_is_cached(cache, n)]
    logger.info(f"{len(already)} layers already cached; building {len(todo)}: {todo[:5]}...{todo[-3:] if todo else ''}")

    for done, layer_idx in enumerate(todo, start=1):
        state_dict = {
            "embed_weight": model_bits["embed_weight"].float(),
            "norm_weight": model_bits["norm_weight"],
            "layers": [load_layer_state_dict_cached(checkpoint, layer_idx, cache)],
        }
        # One layer, positioned at its real global index so the schedule, the MLA/KDA choice and every
        # cache-name prefix match what a full stack would produce for it.
        model = TtKimiK3Transformer(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            num_layers=1,
            first_layer_idx=layer_idx,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            max_seq_len=SEQ_LEN,
            weight_cache_path=cache,
        )
        kind = "MLA" if layer_idx in KimiK3Config.mla_layer_ids() else "KDA"
        ffn = "dense" if layer_idx < KimiK3Config.NUM_DENSE_LAYERS else "MoE"
        logger.info(f"cached layer {layer_idx:2d} ({kind}, {ffn}) — {done}/{len(todo)}")

        if model.kda_states is not None:
            model.kda_states.deallocate()
        del model, state_dict
        gc.collect()

    missing = [n for n in range(KimiK3Config.NUM_LAYERS) if not layer_is_cached(cache, n)]
    logger.info(f"cache complete for {KimiK3Config.NUM_LAYERS - len(missing)}/{KimiK3Config.NUM_LAYERS} layers")
    assert not missing, f"layers still uncached: {missing}"
