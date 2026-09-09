# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading and sharding.

Borrowed structurally from `minimax_m3/tt/attention/weights.py`.

Llama's four projections, and how each shards under TP:

| Weight | HF shape `[out, in]` | Parallelism | Per chip at TP=4 |
|---|---|---|---|
| `q_proj` | [4096, 4096]  (32 heads x 128) | column-parallel | 8 q heads |
| `k_proj` | [1024, 4096]  (8 kv heads x 128) | column-parallel | **2 kv heads** |
| `v_proj` | [1024, 4096]  (8 kv heads x 128) | column-parallel | **2 kv heads** |
| `o_proj` | [4096, 4096] | row-parallel + TP collective | contracts over 1024 |

All four are bias-free (`attention_bias: false`).

## The q/k permutation

`q_proj` and `k_proj` weight rows are permuted from HF half-split to **Meta interleaved** head
layout on the way in, because `rotary_embedding_llama` and `rotary_embedding_indexed` consume
Meta-format cos/sin tables. `v_proj` and `o_proj` are untouched: v is never rotated, and the
attention output inherits v's column order. See `utils/weight_conversion.py` for why permuting only
q and k is exact rather than approximate.

Skipping this permutation does not raise. It measures PCC 0.75 — plausible output, silently wrong.

## Two things this model does NOT need, which the loaders it was borrowed from do

* **No dequantization.** The checkpoint is unquantized bf16, so there are no scales to apply and no
  packed blocks to unpack; the single dtype exit is the whole conversion.
* **No QKV fusion.** Llama ships q/k/v as three separate tensors, and they are kept separate. Fusing
  them is a perf choice whose per-head column order after the split has to be pinned by its own
  test, and perf tuning is out of scope for this bring-up.
"""

from types import SimpleNamespace

import ttnn
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b_d_p.utils.weight_conversion import hf_to_meta_qk


def load_attention_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config,
    *,
    weight_dtype=None,
    tensor_cache_path=None,
):
    """Tilize q/k/v (column-parallel) and o (row-parallel) onto the mesh.

    Args:
        state_dict: this attention's substate — `{q,k,v,o}_proj.weight` — or `{}` for cache-only.

    Returns a namespace of device tensors: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    """
    weight_dtype = weight_dtype or ttnn.bfloat8_b
    col_mapper = mesh_config.column_parallel(mesh_device)  # shard the output (head) dim across TP
    row_mapper = mesh_config.row_parallel(mesh_device)  # shard the input (contraction) dim

    # Permute q/k rows HF -> Meta BEFORE the transpose, while the head dim is still dim 0. The head
    # count differs per projection and passing the wrong one mis-permutes silently.
    permute_heads = {"q_proj": config.num_heads, "k_proj": config.num_kv_heads}

    def prep(name):
        key = f"{name}.weight"
        if not state_dict or key not in state_dict:
            return None
        w = state_dict[key]
        if name in permute_heads:
            w = hf_to_meta_qk(w, permute_heads[name])
        # HF stores Linear as [out, in]; ttnn.linear wants [in, out].
        return w.transpose(-1, -2).unsqueeze(0).unsqueeze(0)

    def load(name, mapper):
        weight = prep(name)
        if weight is None and not tensor_cache_path:
            return None
        return ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mapper,
            # The cache key names the convention: a cache written before the permutation existed
            # must not be reused, and its contents are indistinguishable from a correct one.
            cache_file_name=get_cache_file_name(tensor_cache_path, f"{name}_meta_rope" if name in permute_heads else name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return SimpleNamespace(
        q_proj=load("q_proj", col_mapper),
        k_proj=load("k_proj", col_mapper),
        v_proj=load("v_proj", col_mapper),
        o_proj=load("o_proj", row_mapper),
    )
