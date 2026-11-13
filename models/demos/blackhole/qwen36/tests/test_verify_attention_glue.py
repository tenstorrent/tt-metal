# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The MTP verify's attention (TPAttention._forward_verify) against the previous op sequence, on device.

One full-attention layer of the checkpoint, at the mesh's TP. Both sequences run on the same input rows and on two
copies of one paged bf8 KV cache with random contents; the attention output and both caches must be equal byte for
byte. The reference is forward_decode's full verify body (_forward_verify_ok forced False), with
apply_partial_rope_decode as it was before (the RoPE pass-through slice lands in L1, then a copy to DRAM) and cos / sin
as [1, R, 1, rope_dim]. Shapes: (users, T) = (8, 4), (1, 4), (4, 8), (2, 12). Each user has its own shuffled blocks
and its own long-context position; some users' T rows cross a 32-row tile or a block boundary.

Run: MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_verify_attention_glue.py -v -s
"""

import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
)
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

BLOCK = 64


def _previous_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim):
    """apply_partial_rope_decode before the verify's glue cuts, op for op."""
    hd = x.shape[-1]
    B = batch_size
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
    x_rope_t = ttnn.transpose(x_rope, 1, 2)
    ttnn.deallocate(x_rope)
    cos_p = ttnn.reshape(cos_tt, (1, 1, B, rope_dim))
    sin_p = ttnn.reshape(sin_tt, (1, 1, B, rope_dim))
    roped_t = ttnn.experimental.rotary_embedding_hf(
        x_rope_t, cos_p, sin_p, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(x_rope_t)
    roped = ttnn.to_memory_config(ttnn.transpose(roped_t, 1, 2), ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(roped_t)
    x_pass = ttnn.to_memory_config(ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd)), ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result


def _rm_int(mesh_device, t):
    return ttnn.from_torch(
        t.to(torch.int32).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _host(t, mesh_device):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("n_users, T", [(8, 4), (1, 4), (4, 8), (2, 12)], ids=["u8T4", "u1T4", "u4T8", "u2T12"])
def test_verify_attention_matches_the_previous_op_sequence(
    mesh_device, n_users, T, monkeypatch, reset_seeds, ensure_gc
):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=n_users, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HD, rd = args.n_local_kv_heads, args.head_dim, args.rope_head_dim
    R = n_users * T

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    tw = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, li), args)

    # User u's T rows sit at p_u .. p_u + T - 1: across a block boundary for odd u, across a 32-row tile for even u.
    starts = [BLOCK * (31 + 2 * u) + (BLOCK - 2 if u % 2 else 29) for u in range(n_users)]
    row_pos = torch.tensor([p + j for p in starts for j in range(T)], dtype=torch.int32)
    nb = (max(starts) + T) // BLOCK + 1
    blocks = torch.randperm(n_users * nb + 1)[: n_users * nb].reshape(n_users, nb).to(torch.int32)

    probe = TPAttention(mesh_device, args, tw, tt_ccl)
    groups = probe._spec_sdpa_plan(T, n_users)[1]
    pt_rows = _rm_int(mesh_device, blocks.repeat_interleave(T, dim=0))
    pt_users = _rm_int(mesh_device, blocks)
    pt_groups = _rm_int(mesh_device, blocks.repeat_interleave(groups // n_users, dim=0))
    cur_pos = _rm_int(mesh_device, row_pos)

    model = SimpleNamespace(args=args, rope=SimpleNamespace(rope_delta=0))
    model._rope_tp_cos_sin_decode_torch = lambda p: Qwen36Model._rope_tp_cos_sin_decode_torch(model, p)
    rope_old = [replicate_to_device(mesh_device, t) for t in Qwen36Model._rope_tp_cos_sin_decode_torch(model, row_pos)]
    rope_new = [replicate_to_device(mesh_device, t) for t in Qwen36Model._verify_cos_sin_torch(model, row_pos)]
    assert tuple(rope_old[0].shape) == (1, R, 1, rd) and tuple(rope_new[0].shape) == (1, 1, R, rd)

    cache_init = torch.randn(n_users * nb + 1, NKV, BLOCK, HD, dtype=torch.bfloat16)
    x = replicate_to_device(mesh_device, torch.randn(1, 1, R, args.dim, dtype=torch.bfloat16) * 2)

    def caches():
        return [
            ttnn.from_torch(
                cache_init,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

    def run(attn, rope, **kw):
        return attn.forward_decode(
            x,
            cur_pos,
            *rope,
            page_table=pt_rows,
            alias_kv_write=True,
            spec_verify_mode=True,
            spec_page_table=pt_groups,
            spec_n_users=n_users,
            spec_user_page_table=pt_users,
            **kw,
        )

    a_old = probe
    a_old.set_paged_kv_cache(*caches())
    with monkeypatch.context() as m:
        m.setattr(TPAttention, "_forward_verify_ok", lambda self, *a: False)
        m.setattr(tp_mod, "apply_partial_rope_decode", _previous_partial_rope_decode)
        out_old = run(a_old, rope_old)
    a_new = TPAttention(mesh_device, args, tw, tt_ccl)
    a_new.set_paged_kv_cache(*caches())
    assert a_new._forward_verify_ok(R, pt_rows, True, pt_groups, n_users, pt_users)
    pos_groups = [ttnn.slice(cur_pos, (j,), (R,), (T,)) for j in range(T)]
    out_new = run(a_new, rope_new, spec_pos_groups=pos_groups)

    checks = {
        "attention output": (out_old, out_new),
        "K cache": (a_old.paged_k, a_new.paged_k),
        "V cache": (a_old.paged_v, a_new.paged_v),
    }
    for name, (a, b) in checks.items():
        ha, hb = _host(a, mesh_device), _host(b, mesh_device)
        same = tuple(ha.shape) == tuple(hb.shape) and torch.equal(ha, hb)
        logger.info(f"users={n_users} T={T} {name} {tuple(ha.shape)}: bit-identical={same}")
        assert same, f"users={n_users} T={T}: {name} differs (max abs {(ha.float() - hb.float()).abs().max()})"
    ref_cache = _host(
        ttnn.from_torch(
            cache_init,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        mesh_device,
    )
    assert not torch.equal(ref_cache, _host(a_new.paged_k, mesh_device)), "the verify wrote no K row"
