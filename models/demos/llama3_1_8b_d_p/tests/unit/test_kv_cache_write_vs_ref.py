# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: KV cache contents after a write through the production prefill seam, at SP=8 x TP=4.

Drives one ``Attention`` through ``attention_forward`` with an externally-allocated ``LlamaKVCache``,
then reads the cache back and PCCs it against the torch reference's K/V — i.e. the write landed at
the right slot, the right heads and the right offset, and survived the bf8 round-trip.

Two Llama-specific things this pins that neither donor could:

* **2 KV heads per chip.** At TP=4 with 8 KV heads, each chip's cache carries dim1 == 2, so column
  ``c`` holds global KV heads ``2c`` and ``2c+1``. Both donors have exactly one head per chip, so
  this head-gathering order is new code and is exactly where an off-by-one would hide.
* **K is stored Meta-swizzled.** The cache holds post-RoPE K computed from ``reverse_permute``d q/k
  projections, so the reference K must be put through the same head permutation before comparison.
  V is raw (v_proj is not swizzled and V is not rotated), so V is compared unpermuted — if a
  regression permuted both, this test would still catch it.

``kv_actual=0`` with a cache sized to the prompt makes the block-cyclic layout the identity, so
readback is in natural order. The block-cyclic ordering itself is covered by
``test_kv_cache_gqa_sp_vs_ref``.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention import (
    Attention,
    AttentionConfig,
    ProgramConfig,
    allocate_kv_cache,
)

from ..test_factory import (
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    shard_seq_on_sp,
)

PCC = 0.99


def gather_kv_cache(mesh_device, cache_tensor, n_kv_local, slot_row=0, chunk_local=None):
    """Reassemble a packed per-chip cache into ``[1, n_kv_global, S, head_dim]``, in natural order.

    Column ``c`` holds global KV heads ``[c*n_kv_local, (c+1)*n_kv_local)``; SP row ``r`` holds its
    block-cyclic sequence shard. Concatenate rows on the seq dim, then columns on the head dim.

    ``chunk_local``: rows per chip per chunk. Pass it whenever the cache holds MORE than one chunk —
    the row-concatenation above is natural order only for a single chunk (see
    ``block_cyclic_to_natural``). ``None`` means one chunk, where the two coincide.
    """
    from ..test_factory import block_cyclic_to_natural

    rows, cols = tuple(mesh_device.shape)
    dts = ttnn.get_device_tensors(cache_tensor)
    per_col = []
    for c in range(cols):
        seq_parts = [ttnn.to_torch(dts[r * cols + c])[slot_row : slot_row + 1].float() for r in range(rows)]
        per_col.append(torch.cat(seq_parts, dim=2))  # [1, n_kv_local, S, hd]
    out = torch.cat(per_col, dim=1)  # [1, n_kv_global, S, hd]
    if chunk_local is not None and out.shape[2] // rows != chunk_local:
        out = block_cyclic_to_natural(out, rows, chunk_local, seq_dim=2)
    return out


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])
def test_kv_cache_write_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp
    n_kv_local = cfg.num_key_value_heads // tp
    assert n_kv_local == 2, f"this test pins the 2-KV-heads-per-chip case; got {n_kv_local}"

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_attn = ref.LlamaAttention(cfg, layer_idx=0)
    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    _, ref_k, ref_v = ref_attn(x, (cos_hf, sin_hf))  # k post-RoPE, v raw; [1, 8, S, 128]

    # The cache stores K in Meta-swizzled head layout (reverse_permute'd projections + interleaved
    # RoPE). V is untouched.
    ref_k = ref.hf_to_meta_head_perm(ref_k, hd)

    state_dict = hf_to_meta_qk(
        {
            "q_proj.weight": ref_attn.q_proj.weight.data,
            "k_proj.weight": ref_attn.k_proj.weight.data,
            "v_proj.weight": ref_attn.v_proj.weight.data,
            "o_proj.weight": ref_attn.o_proj.weight.data,
        },
        hd,
    )

    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            max_seq_len=seq_len,
            sequence_parallel=True,
        ),
        state_dict=state_dict,
        ccl_manager=make_ccl(mesh_device),
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": tt_rope.build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )

    # Whole-cache indexed rope: cache sized to the prompt, so chunk 0 covers every position.
    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=seq_len, chunk_size=seq_len, sp_axis=mesh_config.sp_axis
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=seq_len,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    x_tt = shard_seq_on_sp(mesh_device, x.reshape(1, 1, seq_len, cfg.hidden_size), mesh_config)
    attn(x_tt, rope_mats=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=0, indexed_rope=True)
    ttnn.synchronize_device(mesh_device)

    host_k = gather_kv_cache(mesh_device, kv_cache.k, n_kv_local)
    host_v = gather_kv_cache(mesh_device, kv_cache.v, n_kv_local)

    ok_k, pcc_k = comp_pcc(ref_k, host_k, PCC)
    ok_v, pcc_v = comp_pcc(ref_v, host_v, PCC)
    logger.info(f"KV write s={seq_len}: K pcc={pcc_k}  V pcc={pcc_v}")
    assert ok_k, f"K cache mismatch: {pcc_k}"
    assert ok_v, f"V cache mismatch: {pcc_v}"
