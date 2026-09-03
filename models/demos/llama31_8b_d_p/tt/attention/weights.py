# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B attention projection weights: Meta-swizzle, transpose, shard, tilize.

Template: ``models/demos/gpt_oss_d_p/tt/attention/weights.py:23`` (the dataclass), ``:38``
(``load_attention_weights``), ``:74-78`` (the substate reads and the ``o_proj`` transpose),
``:145-146`` (the two mappers), ``:149`` (``as_tensor`` with ``cache_file_name``).

**Three separate Q/K/V weights, not a fused ``wqkv`` (``DEC-011``).** gpt-oss fuses (``:31``) and
must therefore pre-interleave per device (``:83-100`` builds ``cat([q_0,k_0,v_0, q_1,k_1,v_1, ...])``
so a naive equal split hands each chip its own Q|K|V triple). That loop is the most error-prone code
in the template and its failure mode is **invisible at TP=1**, i.e. at every gate P5 can run. Three
column-parallel weights are shard-correct by construction: at TP=8, 4096/8 = 512 = 4 Q heads and
1024/8 = 128 = 1 KV head, no interleave. Cost: one ``ttnn.concat`` per layer at runtime
(``operations.apply_qkv_projection``). Fused QKV is the perf follow-up.

**Deletions vs the template** (``03_OUTLINE.md`` §3.8) — Llama has none of these:

* ``wqkv_bias`` (``:32``), ``o_proj_bias`` (``:34``) and the bias-fusion loops (``:102-115``,
  ``:130-133``). ``attention_bias: false``, so the bias keys are **asserted absent**, not branched on.
* ``sinks`` (``:35``), the sink pre-division (``:117-120``) and the ``sinks_tt`` load (``:192``).
* The ``o_proj`` tile-alignment padding (``:64-70``, ``:122-128``). gpt-oss needs it because
  ``2880/8 = 360`` is not tile-aligned; for Llama ``4096/TP`` is tile-aligned for every admissible
  TP (``00_MODEL_CARD.md`` §4.3), so the path is dead code and is replaced by an assert.

**The Meta RoPE swizzle is applied HERE (``DEC-033``).** ``tt/rope.py`` emits Meta/interleaved
cos/sin for ``ttnn.experimental.rotary_embedding_llama``, which requires the Q/K **projection
weights** to be ``reverse_permute``d at load time
(``models/tt_transformers/tt/load_checkpoints.py:891``; the dict-walking wrapper
``convert_hf_qkv_to_meta_format`` at ``:451`` keys off ``"q_proj.weight"`` / ``"k_proj.weight"``).
Doing it in the loader — rather than expecting every caller to remember, as
``models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:197`` does — means this module takes
plain HF weights and P6's weight loading has nothing extra to know.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama31_8b_d_p.utils.substate import substate
from models.tt_transformers.tt.load_checkpoints import reverse_permute

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """The four projection weights, already sharded and tilized. No biases, no sinks."""

    wq: ttnn.Tensor
    wk: ttnn.Tensor
    wv: ttnn.Tensor
    o_proj: ttnn.Tensor


def _meta_swizzle(w, head_dim: int):
    """``reverse_permute`` one HF ``[out, in]`` Q or K projection into the Meta interleaved layout.

    ``x_meta[2i] = x_hf[i]``, ``x_meta[2i+1] = x_hf[i + D/2]`` per head — the same relation
    ``tests/unit/test_rope_vs_ref.py:60`` derives between the two cos/sin conventions.
    """
    n_heads = w.shape[0] // head_dim
    assert n_heads * head_dim == w.shape[0], f"projection out-dim {w.shape[0]} is not a multiple of head_dim {head_dim}"
    return reverse_permute(w, n_heads, w.shape[0], w.shape[1])


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    *,
    mesh_config,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
    meta_swizzle=True,
) -> AttentionWeights:
    """Load ``q/k/v/o_proj`` from an already-stripped ``self_attn.*`` sub-dict.

    Args:
        mesh_device: the ttnn mesh device.
        config: :class:`~.config.AttentionConfig`.
        state_dict: the ``self_attn.*`` sub-dict in **HF layout** (``[out, in]``), keys
            ``{q,k,v,o}_proj.weight``. ``{}`` means cache-only mode (``tensor_cache_path`` must be
            set), matching ``models/demos/minimax_m3/tt/dense_mlp.py:62``.
        mesh_config: ``MeshConfig``; supplies the column/row-parallel mappers.
        weight_dtype: on-device weight dtype (default ``bfloat8_b``).
        tensor_cache_path: directory for the tilized weight cache, or ``None``.
        meta_swizzle: apply the Q/K ``reverse_permute`` (``DEC-033``). Only a test that deliberately
            breaks the RoPE convention — the ``G-ATTN`` negative control — passes ``False``.

    Returns:
        :class:`AttentionWeights`.
    """
    hidden_size = config.hidden_size
    # The gpt-oss o_proj padding path is deleted, so this must hold rather than be worked around.
    if mesh_config.tp > 1:
        assert (hidden_size // mesh_config.tp) % ttnn.TILE_SIZE == 0, (
            f"hidden_size/tp = {hidden_size}/{mesh_config.tp} = {hidden_size // mesh_config.tp} is "
            f"not tile-aligned; this package deletes gpt-oss's o_proj padding path because Llama "
            f"never needs it (03_OUTLINE.md §3.8)"
        )
        assert (config.num_kv_heads % mesh_config.tp) == 0, (
            f"num_kv_heads {config.num_kv_heads} is not divisible by tp {mesh_config.tp}; "
            f"column-parallel k/v sharding would split a head"
        )

    if state_dict:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sub = substate(state_dict, name)
            assert "weight" in sub, f"{name}.weight missing from the attention state dict"
            # attention_bias: false. Assert, do not branch (03_OUTLINE.md §1 convention 12).
            assert "bias" not in sub, (
                f"{name} carries a bias, but Llama-3.1 has attention_bias: false; this module has "
                f"no bias path (03_OUTLINE.md §3.8)"
            )

        wq = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden]
        wk = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden]
        wv = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden]
        o_proj = substate(state_dict, "o_proj")["weight"]  # [hidden, num_heads * head_dim]

        # Meta / interleaved RoPE convention: Q and K only. V and o_proj are convention-free.
        if meta_swizzle:
            wq = _meta_swizzle(wq, config.head_dim)
            wk = _meta_swizzle(wk, config.head_dim)

        # HF [out, in] -> ttnn [in, out], once, at load time (03_OUTLINE.md §1 convention 6).
        def _prep(w):
            return w.transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wq, wk, wv, o_proj = _prep(wq), _prep(wk), _prep(wv), _prep(o_proj)
    else:
        assert tensor_cache_path, (
            "load_attention_weights got an empty state_dict and no tensor_cache_path; there is "
            "nothing to load from (cache-only mode needs the cache)"
        )
        wq = wk = wv = o_proj = None

    col_mapper = mesh_config.column_parallel(mesh_device)  # shard the head/output dim
    row_mapper = mesh_config.row_parallel(mesh_device)  # shard the contraction dim

    def _load(name, weight, mapper):
        return ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return AttentionWeights(
        # The cache key records the swizzle, so a swizzled and an unswizzled build can never share
        # a cache file (that would be a silent wrong-RoPE load in cache-only mode).
        wq=_load("wq_meta" if meta_swizzle else "wq_hf", wq, col_mapper),
        wk=_load("wk_meta" if meta_swizzle else "wk_hf", wk, col_mapper),
        wv=_load("wv", wv, col_mapper),
        o_proj=_load("o_proj", o_proj, row_mapper),
    )
