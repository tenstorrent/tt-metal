# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1 stacks on synthetic weights, with the caches the runner allocates for them.

For the K3.1 checks that need no reference: what the stack wrote (carries stored, rows past a prompt's
end untouched), that a repeated request is bit-identical, and that a split stack matches the undivided
one. K3.1 is K3 with Gated DSA on its 24 full-attention layers, run through
K3's stack; accuracy against vLLM is the golden tests' job (`test_k31_golden.py`, the depth ladder's
K3.1 rungs).

One deliberate substitution: every layer's FFN is K3's dense FFN (`DENSE_FFN_CONFIG`). A synthetic
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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights as random_kda_weights
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import CHECKPOINT_PREFIX
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural

SP_AXIS, TP_AXIS = 0, 1
# The KDA recurrence's sequence quantum on 8x4: TILE(32) * KDA_SUMMARY_GROUP_CHUNKS(20) * SP(8).
CHUNK = 5120
# Cache capacity: a multiple of the allocator's 8192-token paging and of CHUNK (the cache write), as
# the runner's max_seq_len must be for K3.1.
CACHE_LEN = 40960
STD = 0.02

# K3.1's config with every layer's FFN dense: the one substitution (see the module docstring).
DENSE_FFN_CONFIG = type("KimiK31ConfigDenseFfn", (KimiK31Config,), {"NUM_DENSE_LAYERS": KimiK31Config.NUM_LAYERS})


def hf_config():
    """K3.1's HF config at the synthetic stacks' cache capacity."""
    return kimi_k3_1_hf_config(CACHE_LEN)


def placements(**device_params) -> list:
    """`(mesh_device, device_params)` on 8x4 Fabric2D at K3.1's L1_SMALL; extra `device_params` (e.g. a
    trace test's `trace_region_size`) are added."""
    return [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "l1_small_size": KimiK31Config.L1_SMALL_SIZE,
                **device_params,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        )
    ]


def _lin(generator, out_features, in_features):
    return (torch.randn(out_features, in_features, generator=generator) * STD).to(torch.bfloat16)


def synthetic_state_dict(first_layer_idx: int, num_layers: int) -> dict:
    """Embedding, final norm, AttnRes queries, and per layer: norms, DSA or KDA, dense FFN."""
    config = hf_config()
    hidden, inter = config.hidden_size, config.intermediate_size
    mla_layers = set(KimiK3Config.mla_layer_ids())
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
            layer["mla_weights"] = random_mla_weights(config, seed=layer_idx)
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

    TP-deduped scaled-FP8 kvpe, the bf8 indexer key cache, and the KDA slabs, attached to the model as
    the runner's compile() does. `num_users` is the slot count of every cache and slab and the model's
    user slots.
    """
    adapter = get_adapter("kimi_k3_1")
    config = hf_config()
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
        DENSE_FFN_CONFIG,
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
    if is_first_rank and first_layer_idx > 0:
        # The transformer reads a first rank past layer 0 as a capacity probe (PREFILL_PP_LAYER_OFFSET) and
        # gives it a one-tile random stand-in table, into which every id >= 32 clamps to one row. These windows
        # are real first ranks that embed their tokens, so they take the layer-0 rank's embedding instead.
        ttnn.deallocate(transformer.embed.weight)
        transformer.embed = TtParallelEmbedding(
            mesh_device=mesh_device,
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            torch_weight=state_dict["embed_weight"],
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
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
        0, KimiK31Config.VOCAB_SIZE, (num_tokens,), generator=torch.Generator().manual_seed(seed)
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
