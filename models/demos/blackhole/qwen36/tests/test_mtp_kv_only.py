# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check for QWEN36_MTP_KV_ONLY: TPAttention.forward_decode(kv_only=True) writes the SAME paged KV cache bytes as
the full decode step, for every shape the MTP drafter's KV-maintenance forwards use:

  reseed_b8   32 rows = 8 users x T=4, alias_kv_write, spec_n_users=8  (the batched reseed inside the drafter trace)
  reseed_b1    4 rows = 1 user  x T=4, alias_kv_write, spec_n_users=1  (B == 1 engine: per-row aliased write)
  warm_b1      1 row, independent                                      (eager reseed / single-row join warm; lean path)
  warm_b8      8 rows, independent users                               (batched join warm; lean path)

Two checks per case, on every device's cache shard:
  * full vs kv_only: torch.equal, not a PCC.
  * each mode vs an oracle: the initial cache with exactly the intended rows replaced by the K/V rows the KV write was
    handed (read back when the write starts: _write_kv_aliased for the aliased legacy path, the fused cache update for
    the lean path). Both caches start from the same random content, so a stray write anywhere shows up, and so does a
    write that stores other bytes than it was given in BOTH modes alike, which full vs kv_only alone cannot see.
    The pad-view bug was of that kind: ttnn.pad of the [1, B, 1, HD] K/V returns a view of their buffer, the legacy
    path freed the source, and the grouped write's own L1 slices and shards could then land on V's freed buffer and
    overwrite tiles before they were read. Whether they did depends on the allocation map: here the full step was
    clean and the kv_only step (q and gate freed earlier) lost 16 of V's 256 tiles per device. bf16 caches compare
    exactly; bf8 rows that were written compare within the bf8 quantization step (untouched rows stay exact).

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


def _bf8_close(got, exp):
    """Whether bf16 rows ``exp`` stored in a bfloat8_b cache read back as ``got``: bfloat8_b keeps 7 mantissa bits
    under one exponent per 16 consecutive values of a row, so each value is within 2^-6 of its group's largest one."""
    g = exp.float().reshape(*exp.shape[:-1], -1, 16)
    tol = g.abs().amax(dim=-1, keepdim=True) * 2.0**-6
    return bool(((got.float().reshape(g.shape) - g).abs() <= tol).all())


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize(
    "users, T, aliased",
    [(8, 4, True), (1, 4, True), (1, 1, False), (8, 1, False)],
    ids=["reseed_b8", "reseed_b1", "warm_b1", "warm_b8"],
)
def test_mtp_kv_only_cache_bytes(mesh_device, users, T, aliased, cache_dtype, monkeypatch, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=1024)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HD = args.n_local_kv_heads, args.head_dim
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    tw = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, li), args)

    R = users * T
    nblocks = users * BPU
    ndev = mesh_device.get_num_devices()
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
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

    def read_back(t):
        return ttnn.to_torch(t, mesh_composer=comp)

    def rep_rm(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # The oracle's input: the K/V rows the KV write is handed, read back before it allocates anything. Each device's
    # rows come back stacked on dim 0: [ndev, R, >= NKV, HD]; row r's head h sits at [d, r, h].
    handed = {}
    _aliased_write = TPAttention._write_kv_aliased
    _fused_update = ttnn.experimental.paged_fused_update_cache

    def spy_aliased_write(self, keys, values, k_p, v_p, *a, **kw):
        handed["k"], handed["v"] = read_back(k_p), read_back(v_p)
        return _aliased_write(self, keys, values, k_p, v_p, *a, **kw)

    def spy_fused_update(k_cache, k_in, v_cache, v_in, *a, **kw):
        handed["k"], handed["v"] = read_back(k_in), read_back(v_in)
        return _fused_update(k_cache, k_in, v_cache, v_in, *a, **kw)

    monkeypatch.setattr(TPAttention, "_write_kv_aliased", spy_aliased_write)
    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", spy_fused_update)

    def expected(stored_init, rows):
        exp = stored_init.clone()
        for d in range(ndev):
            for r in range(R):
                p = int(pos[r])
                blk = d * nblocks + int(pt_rows[r][p // BLOCK])
                exp[blk, :, p % BLOCK, :] = rows[d, r, :NKV, :]
        return exp

    def written_mask():
        m = torch.zeros(ndev * nblocks, NKV, BLOCK, dtype=torch.bool)
        for d in range(ndev):
            for r in range(R):
                p = int(pos[r])
                m[d * nblocks + int(pt_rows[r][p // BLOCK]), :, p % BLOCK] = True
        return m

    def run(kv_only):
        handed.clear()
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
        assert set(handed) == {"k", "v"}, "the KV write was not intercepted (a new write path needs a spy here)"
        got = (read_back(kc), read_back(vc))
        ttnn.deallocate(kc)
        ttnn.deallocate(vc)
        return got, (handed["k"], handed["v"])

    # The initial caches as stored (bf8 rounds them), to build the oracle from.
    stored = []
    for t in (init_k, init_v):
        dev = cache(t)
        stored.append(read_back(dev))
        ttnn.deallocate(dev)
    mask = written_mask()

    full_kv, full_rows = run(False)
    lean_kv, lean_rows = run(True)
    for mode, kv, rows in (("full", full_kv, full_rows), ("kv_only", lean_kv, lean_rows)):
        for name, got, init, handed_rows in zip("KV", kv, stored, rows):
            exp = expected(init, handed_rows)
            assert torch.equal(got[~mask], init[~mask]), f"{mode}: {name} cache changed outside the written rows"
            if cache_dtype == ttnn.bfloat16:
                bad = int((got[mask] != exp[mask]).sum())
                assert bad == 0, f"{mode}: {name} cache rows differ from the rows the KV write was handed ({bad} values)"
            else:
                assert _bf8_close(got[mask], exp[mask]), f"{mode}: {name} cache rows are not the handed rows (bf8)"
    # The step wrote something (else the comparison proves nothing): the written rows differ from the initial cache.
    assert not torch.equal(full_kv[0], stored[0]), "the full step left the K cache unchanged"
    full_k, full_v = full_kv
    lean_k, lean_v = lean_kv
    assert torch.equal(full_k, lean_k), f"K cache differs: {int((full_k != lean_k).sum())} elements"
    assert torch.equal(full_v, lean_v), f"V cache differs: {int((full_v != lean_v).sum())} elements"
    logger.info(f"kv_only cache bytes identical: {users} users x T={T}, aliased={aliased}, cache {cache_dtype}")
