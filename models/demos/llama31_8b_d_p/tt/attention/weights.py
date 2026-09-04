# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading for Llama-3.1-8B: three separate Q/K/V projections plus O, no biases.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention`'s
`q_proj`/`k_proj`/`v_proj`/`o_proj`. **Template:**
`models/demos/gpt_oss_d_p/tt/attention/weights.py:38` for the loader shape, the mapper choice
(`:145-146`) and the cache-only `None` branch (`:135-142`).

**Four deletions from the template, all because Llama does not have the feature**
(`bringup_log/00_MODEL_CARD.md` §3):

1. **No fused QKV.** The template pre-fuses Q|K|V per TP shard into one weight
   (`models/demos/gpt_oss_d_p/tt/attention/weights.py:83-100`). This loads three separate weights
   (`DEC-016`, corrected by `DEC-019`) — the head split then takes `(q, cat(k, v))`.
2. **No biases.** `attention_bias: false`, so `wqkv_bias`, `o_proj_bias` and the row-parallel
   bias-replication trick at `models/demos/gpt_oss_d_p/tt/attention/weights.py:130-133` all go.
3. **No `sinks`.**
4. **No o_proj padding.** gpt-oss needs it because `2880/8 = 360` is not tile-aligned
   (`models/demos/gpt_oss_d_p/tt/attention/weights.py:64-70`); here `4096/8 = 512 = 16*32` and
   `1024/8 = 128 = 4*32`, so every shard is tile-aligned and the pad branch — plus the matching
   un-pad slice in the collective tail — is deleted rather than left as a no-op.

**The Meta swizzle happens HERE, and only here.** The device runs
`ttnn.experimental.rotary_embedding_llama`, the interleaved-pair (Meta) convention, so `q_proj` and
`k_proj` must be `reverse_permute`d at load: `models/tt_transformers/tt/load_checkpoints.py:451`
`convert_hf_qkv_to_meta_format` -> `:891` `reverse_permute` (`DEC-011`, `tt/rope.py`). Doing it
inside the loader is the point — a weight cannot reach the device un-swizzled via a path that
forgot. `v_proj` and `o_proj` are **not** permuted: V is never rotated, and O consumes attention
output in head-major order, not rotary pairs. `G-ATTN`'s negative control is exactly this permute
omitted, and it scores ~0.95 — high enough that only the control catches it.
"""

from dataclasses import dataclass

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from .config import AttentionConfig

# The four projections, and which mesh mapper each takes. Q/K/V are column-parallel (shard the
# output/feature dim, i.e. heads); O is row-parallel (shard the input/contraction dim, i.e. heads
# again, but on the other side of the matmul) — `bringup_log/04_CCL_PLAN.md` §4.
_COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj")
_ROW_PARALLEL = ("o_proj",)

# The two weights `convert_hf_qkv_to_meta_format` rewrites
# (`models/tt_transformers/tt/load_checkpoints.py:457`).
_META_SWIZZLED = ("q_proj", "k_proj")


def _cache_name(name: str) -> str:
    return f"{name}_meta" if name in _META_SWIZZLED else name


@dataclass(frozen=True)
class AttentionWeights:
    """The four projection weights, already transposed to `[1, 1, in, out]`, sharded and tilized."""

    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    *,
    mesh_config,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> AttentionWeights:
    """Load, Meta-swizzle, transpose, shard and tilize `q/k/v/o_proj`.

    Args:
        mesh_device: the open mesh.
        config: `AttentionConfig` — only `head_dim` is read, by the Meta swizzle.
        state_dict: already stripped to this attention's own keys, i.e.
            `{"q_proj.weight": ..., "k_proj.weight": ..., "v_proj.weight": ..., "o_proj.weight": ...}`.
            Empty dict -> cache-only load, which requires `tensor_cache_path`.
        mesh_config: the model's `MeshConfig`, for the two mappers.
        weight_dtype: on-device weight dtype, `bfloat8_b` (`DEC-022`).
        tensor_cache_path: where `ttnn.as_tensor` persists / reloads the tilized weights.

    Shapes at TP=8 (`bringup_log/03_OUTLINE.md` §2.7): `q_proj` `[1,1,4096,512]`,
    `k_proj`/`v_proj` `[1,1,4096,128]` (**one** KV head per chip — the equality that forces TP=8),
    `o_proj` `[1,1,512,4096]`.
    """
    if state_dict:
        # The Meta swizzle, on the HF `[out, in]` tensors and BEFORE the transpose: `reverse_permute`
        # reshapes dim 0 into (n_heads, 2, head_dim/2) and it is dim 0 that carries the heads in HF
        # layout. It matches on the key substrings "q_proj.weight" / "k_proj.weight"
        # (`models/tt_transformers/tt/load_checkpoints.py:457`), so v_proj and o_proj pass through
        # untouched and no explicit exclusion list is needed.
        swizzled = convert_hf_qkv_to_meta_format(state_dict, config.head_dim)
        assert set(swizzled) == set(state_dict), "convert_hf_qkv_to_meta_format changed the key set"

        def _prep(name):
            # HF `[out, in]` -> ttnn `[in, out]`, transposed at LOAD time and never at runtime.
            return substate(swizzled, name)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        torch_weights = {name: _prep(name) for name in _COLUMN_PARALLEL + _ROW_PARALLEL}
    elif not tensor_cache_path:
        # Fail loud rather than build four `None` projections (Appendix B, "cache-only build
        # silently wrong"). Attention has no optional weights here — no biases, no sinks, no
        # QK-norm — so every one of the four must have a source.
        raise ValueError(
            "load_attention_weights needs either a state_dict with q/k/v/o_proj or a tensor_cache_path to load from"
        )
    else:
        torch_weights = {name: None for name in _COLUMN_PARALLEL + _ROW_PARALLEL}

    mappers = {name: mesh_config.column_parallel(mesh_device) for name in _COLUMN_PARALLEL}
    mappers.update({name: mesh_config.row_parallel(mesh_device) for name in _ROW_PARALLEL})

    def _load(name):
        return ttnn.as_tensor(
            torch_weights[name],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mappers[name],
            # Q and K carry a `_meta` cache key because the persisted tensor is the SWIZZLED
            # weight: a cache written by an HF-convention loader would be silently wrong here, and
            # the symptom would be "one layer runs on garbage" (Appendix B). V and O are unswizzled
            # and keep their plain names.
            cache_file_name=get_cache_file_name(tensor_cache_path, _cache_name(name)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return AttentionWeights(**{name: _load(name) for name in _COLUMN_PARALLEL + _ROW_PARALLEL})
