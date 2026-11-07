# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1's Gated DSA layers run inside the Kimi-K3 stack: finite and deterministic on 8x4.

K3.1 is K3 with its 24 full-attention layers turned into Gated DSA (`ttMLA` with K3.1's NoPE
indexer, scaled-FP8 KVPE cache and a bf8 indexer key cache). `ttMLA` itself is covered against the
CPU truth by `sparse_mla/test_sparse_mla.py`; this test covers the seam K3.1 adds to the K3 stack:
`TtKimiK3Transformer` threading the sparse cache format and the index cache through
`TtKimiK3Block` to `TtK3MlaAttention`, in the rank-local slot space K3's schedule assigns.

There is no K3.1 CPU reference or golden, so the bar is the K3 layer tests' structural one: two
5120-token chunks of chunked prefill (the second chunk's indexer selects its top-2048 from a 10240
token prefix and reads the first chunk's cache), every output and cache finite, and a second run of
the same sequence bit-identical to the first.

Synthetic weights, with one deliberate substitution: the FFN of every layer is K3's dense FFN
(`NUM_DENSE_LAYERS` covers the window). A synthetic LatentMoE layer is 896 routed experts at 33 M
parameters each, ~59 GB per layer, and the MoE is unchanged from K3 and covered by K3's tests; the
attention, norms, AttnRes and KDA are the model's own.

Windows: the first DSA layer alone (layer 3, a first rank with its embedding); the same layer as a
pipeline stage's KV-only last layer (layer 92's role); and layers 0-3, so the KDA carries and the DSA
index cache meet in one forward.

Caches are allocated here in GLM-5.3's serving layout (TP-deduped scaled-FP8
KVPE and bf8 index-key caches, one slot per DSA layer of the rank), without the paging bank alignment,
which needs a multiple of 8192 tokens; two 5120-token chunks are 10240.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import query_weight_names
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights as random_kda_weights
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import attach_kda_states
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import PLACEMENTS, SP_AXIS, TP_AXIS, _compose
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import CHECKPOINT_PREFIX
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

CHUNK = 5120
NUM_CHUNKS = 2
MAX_SEQ_LEN = CHUNK * NUM_CHUNKS
STD = 0.02


class _DenseFfnK31(KimiK31Config):
    """K3.1 with every layer's FFN dense: the one substitution this test makes (see the module docstring)."""

    NUM_DENSE_LAYERS = KimiK31Config.NUM_LAYERS


# (first_layer_idx, num_layers, kv_only_last_layer)
WINDOWS = {
    "dsa_L3": (3, 1, False),
    "dsa_L3_kv_only": (3, 1, True),
    "hybrid_L0_3": (0, 4, False),
}


def _lin(generator, out_features, in_features):
    return (torch.randn(out_features, in_features, generator=generator) * STD).to(torch.bfloat16)


def _synthetic_state_dict(config, first_layer_idx: int, num_layers: int) -> dict:
    """Embedding, final norm, AttnRes queries, and per layer: norms, DSA or KDA, dense FFN."""
    generator = torch.Generator().manual_seed(2026)
    hidden, inter = config.hidden_size, config.intermediate_size
    mla_layers = set(KimiK31Config.mla_layer_ids())
    layers = []
    for layer_idx in range(first_layer_idx, first_layer_idx + num_layers):
        layer = {
            "attn_norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
            "ffn_norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
            "ffn_weights": {
                "gate_proj": _lin(generator, inter, hidden),
                "up_proj": _lin(generator, inter, hidden),
                "down_proj": _lin(generator, hidden, inter),
            },
        }
        if layer_idx in mla_layers:
            layer["mla_weights"] = random_mla_weights(config, seed=layer_idx)
        else:
            layer["kda_weights"] = random_kda_weights(kimi_k3_kda_config())
        layers.append(layer)

    attn_res = {}
    for name in query_weight_names(num_layers, CHECKPOINT_PREFIX, first_layer_idx=first_layer_idx):
        if name.endswith("_norm.weight"):
            attn_res[name] = torch.ones(hidden, dtype=torch.float32)
        else:
            attn_res[name] = torch.randn(1, hidden, generator=generator) * STD
    return {
        "embed_weight": torch.randn(config.vocab_size, hidden, generator=generator, dtype=torch.bfloat16),
        "norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
        "attn_res_weights": attn_res,
        "layers": layers,
    }


def _shards(mesh_device, tensor) -> torch.Tensor:
    """Every device's shard, as stored (no dtype conversion): what bit-identity compares."""
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(tensor, mesh_composer=composer)


def _run_sequence(model, mesh_device, token_ids, kvpe, index) -> list[torch.Tensor | None]:
    outputs = []
    for chunk in range(NUM_CHUNKS):
        start = chunk * CHUNK
        tokens_tt = prepare_prefill_input_tensor(
            token_ids[start : start + CHUNK],
            mesh_device,
            tuple(mesh_device.shape)[SP_AXIS],
            False,
            tuple(mesh_device.shape),
            SP_AXIS,
        )
        logger.info(f"chunk {chunk}: forward at actual_start={start}")
        out = model.forward(
            tokens_tt,
            kvpe_cache=kvpe,
            index_kv_cache=index,
            actual_start=start,
            actual_end=start + CHUNK,
        )
        ttnn.deallocate(tokens_tt)
        if out is None:
            outputs.append(None)
        else:
            outputs.append(_compose(mesh_device, out))
            ttnn.deallocate(out)
    return outputs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("window", list(WINDOWS), ids=list(WINDOWS))
def test_k31_dsa_block_finite_and_deterministic(mesh_device, device_params, window):
    first_layer_idx, num_layers, kv_only = WINDOWS[window]
    config = kimi_k3_1_hf_config(max_seq=MAX_SEQ_LEN)
    cache_format = MlaKvCacheFormat.SCALED_FP8

    logger.info(f"{window}: building synthetic weights for layers [{first_layer_idx}, {first_layer_idx + num_layers})")
    state_dict = _synthetic_state_dict(config, first_layer_idx, num_layers)
    logger.info(f"{window}: building the model")
    model = TtKimiK3Transformer(
        mesh_device,
        config,
        _DenseFfnK31,
        state_dict,
        num_layers=num_layers,
        seq_len=CHUNK,
        first_layer_idx=first_layer_idx,
        kv_only_last_layer=kv_only,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=MAX_SEQ_LEN,
        is_chunked=True,
        sparse_kv_cache_format=cache_format,
    )
    dsa_layers = [layer for layer in model.layers if layer.attention.writes_kv]
    assert dsa_layers and all(layer.attention.mla._has_indexer for layer in dsa_layers), "K3.1 layers must be DSA"

    num_slots = model.schedule.num_mla_layers
    cache_kwargs = dict(
        mesh_device=mesh_device,
        seq_len=MAX_SEQ_LEN,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_slots,
        num_users=1,
        tp_axis=TP_AXIS,
    )
    kvpe = init_mla_kv_cache(cache_format=cache_format, hf_config=config, **cache_kwargs)
    index = init_kvpe_cache(kvpe_cache_head_dim=config.index_head_dim, dtype=ttnn.bfloat8_b, **cache_kwargs)
    slabs = attach_kda_states(mesh_device, model, sp_axis=SP_AXIS, tp_axis=TP_AXIS)

    token_ids = torch.randint(0, config.vocab_size, (MAX_SEQ_LEN,), generator=torch.Generator().manual_seed(7))
    runs = []
    try:
        for run in range(2):
            logger.info(f"{window}: run {run}")
            outputs = _run_sequence(model, mesh_device, token_ids.tolist(), kvpe, index)
            runs.append((outputs, _shards(mesh_device, kvpe.storage), _shards(mesh_device, index)))
    finally:
        if slabs is not None:
            slabs.deallocate()
        model.release_sub_device_managers()

    (outputs, kvpe_raw, index_raw), (outputs_again, kvpe_raw_again, index_raw_again) = runs

    for chunk, out in enumerate(outputs):
        if kv_only:
            assert out is None, "a KV-only last layer produces no output"
            continue
        assert torch.isfinite(out).all(), f"chunk {chunk}: non-finite output"
        assert out.float().std() > 0, f"chunk {chunk}: constant output"

    kvpe_logical = kvpe.unpack_host(kvpe_raw)
    assert torch.isfinite(kvpe_logical).all(), "non-finite KVPE cache"
    assert torch.isfinite(index_raw.float()).all(), "non-finite index cache"
    # Both chunks of every DSA slot were written: no all-zero row survives in either cache.
    assert (kvpe_logical.abs().amax(dim=-1) > 0).all(), "a KVPE cache row was never written"
    assert (index_raw.float().abs().amax(dim=-1) > 0).all(), "an index cache row was never written"

    for chunk, (out, again) in enumerate(zip(outputs, outputs_again)):
        if out is not None:
            assert torch.equal(out, again), f"chunk {chunk}: output differs between identical runs"
    assert torch.equal(kvpe_raw.view(torch.uint8), kvpe_raw_again.view(torch.uint8)), "KVPE cache not deterministic"
    assert torch.equal(index_raw, index_raw_again), "index cache not deterministic"
