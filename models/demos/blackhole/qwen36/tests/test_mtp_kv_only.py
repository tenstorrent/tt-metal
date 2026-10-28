# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for QWEN36_MTP_KV_ONLY: TPAttention.forward_decode(kv_only=True) writes the SAME paged KV cache bytes as
the full decode step, for every shape the MTP drafter's KV-maintenance forwards use:

  reseed_b8   32 rows = 8 users x T=4, alias_kv_write, spec_n_users=8  (the batched reseed inside the drafter trace)
  reseed_b1    4 rows = 1 user  x T=4, alias_kv_write, spec_n_users=1  (B == 1 engine: per-row aliased write)
  warm_b1      1 row, independent                                      (eager reseed / single-row join warm; lean path)
  warm_b8      8 rows, independent users                               (batched join warm; lean path)

Both caches start from the same random content, so a stray write anywhere in the cache also shows up. The comparison is
torch.equal on every device's cache shard, not a PCC.

Run (inside the tt-metal container, on a Blackhole Galaxy / P150x8 TP mesh):
    MESH_DEVICE=P150x8 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_mtp_kv_only.py -v -s
"""
import os

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
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

BLOCK, BPU = 64, 4  # 4 blocks of 64 per user


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize(
    "users, T, aliased",
    [(8, 4, True), (1, 4, True), (1, 1, False), (8, 1, False)],
    ids=["reseed_b8", "reseed_b1", "warm_b1", "warm_b8"],
)
def test_mtp_kv_only_cache_bytes(mesh_device, users, T, aliased, cache_dtype, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=1024)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HD = args.n_local_kv_heads, args.head_dim
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    tw = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, li), args)

    R = users * T
    nblocks = users * BPU
    g = torch.Generator().manual_seed(7)
    init_k = torch.randn(nblocks, NKV, BLOCK, HD, generator=g).to(torch.bfloat16)
    init_v = torch.randn(nblocks, NKV, BLOCK, HD, generator=g).to(torch.bfloat16)
    x = torch.randn(1, 1, R, args.dim, generator=g).to(torch.bfloat16)
    # Row u*T + i is user u's candidate i at position base_u + i (user-major, as reseed_row_plan lays them out).
    base = [40 + 23 * u for u in range(users)]
    pos = torch.tensor([base[r // T] + r % T for r in range(R)], dtype=torch.int32)
    pt_rows = torch.tensor([list(range((r // T) * BPU, (r // T + 1) * BPU)) for r in range(R)], dtype=torch.int32)

    def cache(t):
        return ttnn.from_torch(
            t,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rep_rm(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def run(kv_only):
        a = TPAttention(mesh_device, args, tw, tt_ccl)
        a.decode_sdpa_max_cores = 64  # as Qwen36MTP sets it
        kc, vc = cache(init_k), cache(init_v)
        a.set_paged_kv_cache(kc, vc)
        cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, pos)
        kw = dict(alias_kv_write=True, spec_n_users=users) if aliased else {}
        out = a.forward_decode(
            replicate_to_device(mesh_device, x),
            rep_rm(pos),
            cos,
            sin,
            page_table=rep_rm(pt_rows),
            kv_only=kv_only,
            **kw,
        )
        assert (out is None) == kv_only
        if out is not None:
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        got = (ttnn.to_torch(kc, mesh_composer=comp), ttnn.to_torch(vc, mesh_composer=comp))
        ttnn.deallocate(kc)
        ttnn.deallocate(vc)
        return got

    full_k, full_v = run(False)
    lean_k, lean_v = run(True)
    # The step wrote something (else the comparison proves nothing): the written rows differ from the initial cache.
    init_k_dev = cache(init_k)
    k0 = ttnn.to_torch(init_k_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(init_k_dev)
    assert not torch.equal(full_k, k0), "the full step left the K cache unchanged"
    assert torch.equal(full_k, lean_k), f"K cache differs: {int((full_k != lean_k).sum())} elements"
    assert torch.equal(full_v, lean_v), f"V cache differs: {int((full_v != lean_v).sum())} elements"
    logger.info(f"kv_only cache bytes identical: {users} users x T={T}, aliased={aliased}, cache {cache_dtype}")
