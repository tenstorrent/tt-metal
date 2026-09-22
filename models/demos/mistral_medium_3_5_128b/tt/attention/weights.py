# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading and layout. Ported from ``gpt_oss_d_p/tt/attention/weights.py``.

Two transforms happen on the host, before anything reaches the device:

1. **HF -> Meta row permutation** of ``q_proj`` and ``k_proj``. The checkpoint stores rotary pairs
   half-split (``x[i]`` pairs with ``x[i + head_dim/2]``); ``rotary_embedding_llama`` expects them
   interleaved. Permuting the projection **rows** once at load time is equivalent to permuting the
   activations every step. ``v_proj`` and ``o_proj`` are untouched — no rotation touches them.
   The permutation is ``reference.modeling.hf_to_meta_head_perm``, the same function the reference
   uses, so the two sides cannot drift.
2. **QKV fusion.** ``[q; k; v]`` concatenated on the output dim into one ``wqkv``, so the block
   issues one matmul instead of three. The concat order fixes what
   ``operations.split_qkv_heads_prefill`` must undo, and it is interleaved **per TP shard**: chip
   ``t`` gets ``[q_heads[t*24:(t+1)*24]; k_heads[t*2:(t+1)*2]; v_heads[t*2:(t+1)*2]]`` so that a
   plain ``shard_heads_tp`` split lands the right heads together.

Sharding: ``wqkv`` is column-parallel (heads over TP), ``o_proj`` row-parallel. Unlike the source
there is **no output padding** — ``hidden_size / tp = 12288 / 4 = 3072`` is already a multiple of
the 32-wide tile, so the ``pad_size`` plumbing the source needs for its odd head count is dropped.
There are **no biases** anywhere in this checkpoint (``attention_bias`` absent, verified over the
safetensors index), so the dataclass carries only the two weights.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.modeling import hf_to_meta_head_perm
from models.demos.mistral_medium_3_5_128b.utils.general_utils import get_cache_file_name
from models.demos.mistral_medium_3_5_128b.utils.substate import substate


@dataclass(frozen=True)
class AttentionWeights:
    """On-device attention weights.

    Attributes:
        wqkv: fused ``[q; k; v]`` projection, column-parallel over TP.
            Per chip ``[1, 1, hidden_size, (num_heads + 2*num_kv_heads) * head_dim / tp]``
            = ``[1, 1, 12288, 3584]`` at target.
        o_proj: output projection, row-parallel over TP.
            Per chip ``[1, 1, num_heads * head_dim / tp, hidden_size]`` = ``[1, 1, 3072, 12288]``.
    """

    wqkv: "ttnn.Tensor"
    o_proj: "ttnn.Tensor"


def fuse_qkv_host(state_dict, config, tp: int):
    """Host-side ``[q; k; v]`` fusion with the HF -> Meta permutation applied to q and k.

    Args:
        state_dict: ``{q,k,v}_proj.weight`` in HF ``[out_features, in_features]`` layout.
        config: the model config (head counts and ``head_dim``).
        tp: tensor-parallel width; the fused output is ordered so a plain column split over ``tp``
            keeps each chip's q/k/v heads contiguous and matched.

    Returns:
        A torch tensor ``[hidden_size, (num_heads + 2*num_kv_heads) * head_dim]``, transposed into
        the ``[in, out]`` layout ttnn matmuls want.
    """
    head_dim = config.head_dim
    q = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
    k = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
    v = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

    assert config.num_heads % tp == 0, f"num_heads ({config.num_heads}) must divide tp ({tp})"
    assert config.num_kv_heads % tp == 0, f"num_kv_heads ({config.num_kv_heads}) must divide tp ({tp})"

    # HF -> Meta on the ROWS of q and k. Each head's head_dim rows are permuted independently; v is
    # never rotated so it is left alone.
    perm = torch.tensor(hf_to_meta_head_perm(head_dim), dtype=torch.long)

    def _permute_rows(w):
        out_features = w.shape[0]
        return w.reshape(out_features // head_dim, head_dim, -1)[:, perm, :].reshape(out_features, -1)

    q = _permute_rows(q)
    k = _permute_rows(k)

    # Interleave per TP shard so that a plain column split over `tp` keeps each chip's q/k/v heads
    # together and matched. The order inside a shard is [q; k; v] — what split_qkv_heads_prefill
    # undoes via ttnn.experimental.nlp_create_qkv_heads.
    shards = []
    for t in range(tp):
        wq = torch.chunk(q, tp, dim=0)[t]
        wk = torch.chunk(k, tp, dim=0)[t]
        wv = torch.chunk(v, tp, dim=0)[t]
        shards.append(torch.cat([wq, wk, wv], dim=0).transpose(-2, -1))  # [hidden, local_qkv]

    return torch.cat(shards, dim=-1)


def load_attention_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config,
    *,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> AttentionWeights:
    """Fuse, permute, shard and push the attention weights onto the mesh.

    Args:
        mesh_device: the open mesh.
        config: an :class:`~.config.AttentionConfig`.
        state_dict: ``{q,k,v,o}_proj.weight``. Empty dict => load from ``tensor_cache_path`` only.
        mesh_config: :class:`~...tt.config.MeshConfig`; supplies the TP axis and shard mappers.
        weight_dtype: on-device dtype (the spec's ``dataformats.weights``: bfloat8_b).
        tensor_cache_path: directory for the tilized-weight cache, or None.
    """
    if state_dict:
        assert config.hidden_size % (ttnn.TILE_SIZE * mesh_config.tp) == 0, (
            f"hidden_size/tp ({config.hidden_size // mesh_config.tp}) must be tile-aligned; this "
            f"model's 12288/4 = 3072 is, which is why no o_proj padding is carried here"
        )
        qkv = fuse_qkv_host(state_dict, config, mesh_config.tp).unsqueeze(0).unsqueeze(0)
        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    else:
        qkv, o_proj = None, None

    return AttentionWeights(
        wqkv=ttnn.as_tensor(
            qkv,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        o_proj=ttnn.as_tensor(
            o_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mesh_config.row_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
