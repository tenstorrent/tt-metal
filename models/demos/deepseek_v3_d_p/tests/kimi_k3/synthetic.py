# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 and Kimi-K3.1 stacks on synthetic weights, with the caches the runner allocates for them.

For the tests that have no golden: K3.1 has neither a CPU reference nor a vLLM trace, so its variants
of K3's depth, chunked-prefill and pipeline-split tests check what the stack wrote (every cache slot
its own, carries stored, rows past a prompt's end untouched), that a repeated request is bit-identical,
and that a split stack matches the undivided one. The same tests run K3 through the identical path --
K3.1 is K3 with Gated DSA on its 24 full-attention layers -- as the baseline.

One deliberate substitution: every layer's FFN is K3's dense FFN (`dense_ffn_config`). A synthetic
LatentMoE layer is 896 routed experts at 33 M parameters each, ~59 GB per layer; the MoE is the same
in K3 and K3.1 and covered by K3's own tests. Attention, norms, AttnRes and KDA are the model's own.

Weights are drawn per GLOBAL layer index, so every window sees the same weights: a pipeline rank
built from layers [12, 16) holds exactly what an undivided [0, 16) stack holds there.
"""

import pytest
import torch

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import layer_query_names, output_query_names
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights as random_kda_weights
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import CHECKPOINT_PREFIX
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural

SP_AXIS, TP_AXIS = 0, 1
# The KDA recurrence's sequence quantum on 8x4: TILE(32) * KDA_SUMMARY_GROUP_CHUNKS(20) * SP(8).
CHUNK = 5120
# Cache capacity: a multiple of the allocator's 8192-token paging and of CHUNK (the cache write), as
# the runner's max_seq_len must be for K3.1.
CACHE_LEN = 40960
STD = 0.02

MODEL_CONFIGS = {"kimi_k3": KimiK3Config, "kimi_k3_1": KimiK31Config}
HF_CONFIGS = {"kimi_k3": kimi_k3_hf_config, "kimi_k3_1": kimi_k3_1_hf_config}


def placements(models=tuple(MODEL_CONFIGS), **device_params) -> list:
    """`(model, mesh_device, device_params)` on 8x4 Fabric2D, each model at its own L1_SMALL; extra
    `device_params` (e.g. a trace test's `trace_region_size`) are added to every placement."""
    return [
        pytest.param(
            model,
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "l1_small_size": MODEL_CONFIGS[model].L1_SMALL_SIZE,
                **device_params,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id=f"{model}-fabric2d-8x4",
        )
        for model in models
    ]


def dense_ffn_config(model_cfg: type) -> type:
    """`model_cfg` with every layer's FFN dense: the one substitution (see the module docstring)."""
    return type(f"{model_cfg.__name__}DenseFfn", (model_cfg,), {"NUM_DENSE_LAYERS": model_cfg.NUM_LAYERS})


def _lin(generator, out_features, in_features):
    return (torch.randn(out_features, in_features, generator=generator) * STD).to(torch.bfloat16)


def synthetic_state_dict(config, first_layer_idx: int, num_layers: int) -> dict:
    """Embedding, final norm, AttnRes queries, and per layer: norms, MLA/DSA or KDA, dense FFN."""
    hidden, inter = config.hidden_size, config.intermediate_size
    mla_layers = set(KimiK3Config.mla_layer_ids())
    has_indexer = resolve_has_indexer(config)
    layers, attn_res = [], {}
    for layer_idx in range(first_layer_idx, first_layer_idx + num_layers):
        generator = torch.Generator().manual_seed(2026 + layer_idx)
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
            # K3 is the same attention minus the indexer, whose fields K3's config does not carry.
            mla = random_mla_weights(config if has_indexer else kimi_k3_1_hf_config(), seed=layer_idx)
            layer["mla_weights"] = mla if has_indexer else {k: v for k, v in mla.items() if "indexer." not in k}
        else:
            layer["kda_weights"] = random_kda_weights(kimi_k3_kda_config())
        layers.append(layer)
        for name in layer_query_names(layer_idx, CHECKPOINT_PREFIX):
            attn_res[name] = _query(name, hidden, generator)

    generator = torch.Generator().manual_seed(2025)
    for name in output_query_names(CHECKPOINT_PREFIX):
        attn_res[name] = _query(name, hidden, generator)
    return {
        "embed_weight": torch.randn(config.vocab_size, hidden, generator=generator, dtype=torch.bfloat16),
        "norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
        "attn_res_weights": attn_res,
        "layers": layers,
    }


def _query(name, hidden, generator):
    if name.endswith("_norm.weight"):
        return torch.ones(hidden, dtype=torch.float32)
    return torch.randn(1, hidden, generator=generator) * STD


def window(state_dict: dict, first_layer_idx: int, num_layers: int) -> dict:
    """Layers [first_layer_idx, +num_layers) of a state dict that starts at global layer 0."""
    return {**state_dict, "layers": state_dict["layers"][first_layer_idx : first_layer_idx + num_layers]}


def build_stack(
    mesh_device,
    model: str,
    state_dict: dict,
    first_layer_idx: int,
    num_layers: int,
    *,
    seq_len: int = CHUNK,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
    kv_only_last_layer: bool = False,
    num_users: int = 1,
):
    """A rank's `TtKimiK3Transformer` and its caches from the serving allocator (`allocate_kv_cache`).

    Dense bf8 kvpe for K3; TP-deduped scaled-FP8 kvpe plus the bf8 indexer key cache for K3.1; the KDA
    slabs for both, attached to the model as the runner's compile() does. `num_users` is the slot count of
    every cache and slab and the model's user slots.
    """
    adapter = get_adapter(model)
    config = HF_CONFIGS[model](CACHE_LEN)
    params = PrefillRunParams(
        mesh_shape=tuple(mesh_device.shape),
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        max_seq_len=CACHE_LEN,
        chunk_size=seq_len,
        num_users=num_users,
        capacity_factor=2,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=kv_only_last_layer,
        weight_cache_path=None,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        tp_shard_kv=True,
        sparse_kv_cache_format=None,
    )
    transformer = TtKimiK3Transformer(
        mesh_device,
        config,
        dense_ffn_config(MODEL_CONFIGS[model]),
        state_dict,
        num_layers=num_layers,
        seq_len=seq_len,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        kv_only_last_layer=kv_only_last_layer,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=CACHE_LEN,
        is_chunked=True,
        slot_num=num_users,
        sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
        tp_shard_kv=True,
    )
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=config, params=params)
    transformer.kda_states = caches.kda_states
    return transformer, caches


def release(transformer, caches) -> None:
    if caches.kda_states is not None:
        caches.kda_states.deallocate()
    transformer.release_sub_device_managers()


def token_ids(num_tokens: int, seed: int = 7) -> list[int]:
    return torch.randint(
        0, kimi_k3_hf_config().vocab_size, (num_tokens,), generator=torch.Generator().manual_seed(seed)
    ).tolist()


def upload_tokens(mesh_device, ids: list[int]) -> ttnn.Tensor:
    """One chunk's ids, SP-sharded the way the runner places them (natural order, contiguous per chip)."""
    shape = tuple(mesh_device.shape)
    return prepare_prefill_input_tensor(ids, mesh_device, shape[SP_AXIS], False, shape, SP_AXIS)


def compose(mesh_device, tensor: ttnn.Tensor, rows: int) -> torch.Tensor:
    """A `[1, 1, seq/sp, hidden/tp]`-sharded activation as host `[rows, hidden]` in token order."""
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(tensor, mesh_composer=composer).reshape(-1, KimiK3Config.EMB_SIZE)[:rows]


def cache_rows(mesh_device, cache, chunk: int, rows: int) -> torch.Tensor:
    """Every slot of a TP-deduped kvpe (`MlaKvCache`) or index cache as logical float32 `[slots, rows, dim]`
    in token order, un-rotated from the block-cyclic layout `chunk`-token writes leave. The write grid
    need not divide the cache's per-chip rows."""
    gathered, stripes = gather_cache_natural(cache, mesh_device, tp_shard_kv=True)
    positions = blockcyclic_positions(stripes, chunk, gathered.shape[1])
    held = positions < rows
    natural = torch.zeros(gathered.shape[0], rows, gathered.shape[-1])
    natural[:, positions[held]] = gathered[:, held]
    return natural
