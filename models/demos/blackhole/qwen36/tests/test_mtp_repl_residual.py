# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks for QWEN36_MTP_REPL_RESIDUAL (K5: the MTP drafter's decode forwards on the replicated RES40 residual,
tp_common block comment "QWEN36_MTP_REPL_RESIDUAL"). Not bit-identical by design (the fc partials are rounded to bf16
per device before the all-reduce, and wo / down partials are summed in the all-reduce's order), so every check is
against the fractured path on the SAME head, weights and inputs: the replicated head keeps every fractured-path
attribute, and Qwen36MTP.forward_decode(mtp, ...) (the class method) still runs today's path while mtp.forward_decode
(the instance route _init_repl_residual installs) runs K5.

  test_k5_fuse          _fuse_repl (row-parallel fc + one all-reduce, RES40) vs the gather of the fractured _fuse
                        (column-parallel fc), 8 and 32 rows, both embedding-norm modes; replicas identical
  test_k5_enorm_modes   table (sharded post-all-gather with the per-token stats row) vs gather (all-gathered RES40
                        norm): the fc input differs only there
  test_k5_draft_chain   K=3 chained draft steps (need_logits=True) on 8 users over a random 448-position MTP KV
                        cache: per-step logits PCC + argmax agreement except near ties, next_hidden PCC
  test_k5_reseed_kv     the batched KV-only reseed (32 rows = 8 users x T=4, alias_kv_write, spec_n_users=8): cache
                        rows written by K5 vs the fractured path (bf8 cache), untouched rows byte-identical
  test_k5_trace         one K5 draft step captured in a trace and replayed twice == the eager step, bit for bit

Run (inside the tt-metal container, on a Blackhole Galaxy P150x8 TP row, holding the device lock):
    MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 HF_MODEL=<Qwen3.8-27B> \
      pytest models/demos/blackhole/qwen36/tests/test_mtp_repl_residual.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.mtp_torch_ref import load_head_sd
from models.demos.blackhole.qwen36.tests.test_factory import (
    assert_argmaxes_match_except_near_ties,
    compute_pcc,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
from models.tt_transformers.tt.common import Mode

USERS, T, BLOCK, NB = 8, 4, 64, 8  # 8 users, K=3 (T=4) candidates, 8 blocks of 64 positions per user


def _build(mesh_device, monkeypatch, enorm="table"):
    """An n_layers=0 model (embedding + LM head + final norm + MTP head) with the K5 drafter built."""
    monkeypatch.setenv("QWEN36_MTP_REPL_RESIDUAL", "1")
    monkeypatch.setenv("QWEN36_MTP_REPL_ENORM", enorm)
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=USERS, max_seq_len=1024)
    sd = load_head_sd(args.CKPT_DIR)
    args.n_layers = 0
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model(mesh_device, args, sd, tensor_cache_path=args.weight_cache_path())
    mtp = model.mtp
    assert mtp is not None and mtp.repl_residual, "QWEN36_MTP_REPL_RESIDUAL did not engage (DECODE_ALLREDUCE / mesh?)"
    assert mtp._repl_enorm == enorm
    return model, mtp, args


def _rep(mesh_device, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _full(mesh_device, t, rows):
    """A fractured [1,1,*,dim/tp] (or vocab/tp) tensor -> the full torch [rows, width] (device-major concat)."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))[0, 0, :rows].float()


def _replica(mesh_device, t, rows):
    """A replicated [1,1,*,dim] tensor -> device 0's torch [rows, dim]; asserts every replica is identical."""
    reps = [ttnn.to_torch(d)[0, 0, :rows].float() for d in ttnn.get_device_tensors(t)]
    for i, r in enumerate(reps[1:], 1):
        assert torch.equal(r, reps[0]), f"replica {i} differs from replica 0"
    return reps[0]


def _inputs(mesh_device, args, rows, seed):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, 1, rows, args.dim, generator=g).to(torch.bfloat16)
    tokens = torch.randint(0, args.vocab_size, (rows, 1), generator=g, dtype=torch.int32)
    return shard_to_device(mesh_device, hidden, dim=-1), _rep(mesh_device, tokens, ttnn.uint32), tokens


def _kv_cache(mesh_device, args, seed):
    """A random bf8 paged MTP KV cache (NB blocks per user + the reseed's scratch block), bound to the MTP attention."""
    g = torch.Generator().manual_seed(seed)
    shape = [USERS * NB + 1, args.n_local_kv_heads, BLOCK, args.head_dim]
    k = torch.randn(*shape, generator=g).to(torch.bfloat16)
    v = torch.randn(*shape, generator=g).to(torch.bfloat16)

    def mk(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return mk(k), mk(v)


def _positions(rows_per_user):
    """User u's rows sit at 100 + 37u + i (user-major, i < rows_per_user); page-table rows name the user's blocks."""
    pos = torch.tensor([100 + 37 * u + i for u in range(USERS) for i in range(rows_per_user)], dtype=torch.int32)
    pt = torch.tensor([list(range(u * NB, (u + 1) * NB)) for u in range(USERS) for _ in range(rows_per_user)],
                      dtype=torch.int32)
    return pos, pt


def _rope(mesh_device, args, pos):
    return rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, pos)


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("enorm", ["table", "gather"])
@pytest.mark.parametrize("rows", [8, 32])
def test_k5_fuse(mesh_device, enorm, rows, monkeypatch, reset_seeds, ensure_gc):
    _, mtp, args = _build(mesh_device, monkeypatch, enorm)
    h_frac, tok, _ = _inputs(mesh_device, args, rows, seed=1)
    emb = mtp._drafter_embed(tok)
    ref = _full(mesh_device, mtp._fuse(emb, h_frac, Mode.DECODE), rows)  # today's column-parallel fc, gathered
    h_rep = mtp.replicate_hidden(h_frac)
    assert _replica(mesh_device, h_rep, rows).equal(_full(mesh_device, h_frac, rows)), "the hidden gather is not exact"
    x0 = mtp._fuse_repl(emb, h_rep, tok)
    got = _replica(mesh_device, x0, rows)
    pcc = compute_pcc(ref, got)
    rel = float(((got - ref).norm() / ref.norm()))
    logger.info(f"[k5 fuse {enorm} rows={rows}] pcc={pcc:.6f} rel_l2={rel:.3e}")
    assert pcc >= 0.9999 and rel < 1e-2, f"K5 fc output drifted: pcc {pcc}, rel {rel}"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("rows", [8, 32])
def test_k5_enorm_modes(mesh_device, rows, monkeypatch, reset_seeds, ensure_gc):
    _, mtp, args = _build(mesh_device, monkeypatch, "table")
    h_frac, tok, _ = _inputs(mesh_device, args, rows, seed=2)
    emb = mtp._drafter_embed(tok)
    h_rep = mtp.replicate_hidden(h_frac)
    x_table = _replica(mesh_device, mtp._fuse_repl(emb, h_rep, tok), rows)
    mtp._repl_enorm = "gather"
    x_gather = _replica(mesh_device, mtp._fuse_repl(emb, h_rep, tok), rows)
    mtp._repl_enorm = "table"
    pcc = compute_pcc(x_gather, x_table)
    logger.info(f"[k5 enorm rows={rows}] table vs gather pcc={pcc:.7f} maxabs={float((x_table - x_gather).abs().max()):.3e}")
    assert pcc >= 0.99999, f"table-mode embedding norm differs from the gathered norm: pcc {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_k5_draft_chain(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """K=3 chained draft steps: the fractured class method vs the K5 route, same tokens (the fractured step's argmax)
    fed to both so only numerics differ. Each path rewrites the same cache slots before its own SDPA reads them."""
    _, mtp, args = _build(mesh_device, monkeypatch, "table")
    k_cache, v_cache = _kv_cache(mesh_device, args, seed=3)
    mtp.attention.set_paged_kv_cache(k_cache, v_cache)
    h_frac, tok, _ = _inputs(mesh_device, args, USERS, seed=4)
    pos, pt = _positions(1)
    pt_tt = _rep(mesh_device, pt, ttnn.int32)
    h_ref, h_k5, tok_tt = h_frac, h_frac, tok
    for k in range(3):
        p = pos + k
        cos, sin = _rope(mesh_device, args, p)
        pos_tt = _rep(mesh_device, p, ttnn.int32)
        lg_ref, n_ref = Qwen36MTP.forward_decode(mtp, h_ref, tok_tt, pos_tt, cos, sin, pt_tt)
        lg_k5, n_k5 = mtp.forward_decode(h_k5, tok_tt, pos_tt, cos, sin, pt_tt)
        ref = _full(mesh_device, lg_ref, USERS)[:, : args.vocab_size]
        got = _full(mesh_device, lg_k5, USERS)[:, : args.vocab_size]
        pcc = compute_pcc(ref, got)
        hpcc = compute_pcc(_full(mesh_device, n_ref, USERS), _replica(mesh_device, n_k5, USERS))
        agree = int((ref.argmax(-1) == got.argmax(-1)).sum())
        logger.info(f"[k5 chain step {k}] logits pcc={pcc:.6f} next_hidden pcc={hpcc:.6f} argmax {agree}/{USERS}")
        assert pcc >= 0.999 and hpcc >= 0.999, f"step {k}: logits pcc {pcc}, next_hidden pcc {hpcc}"
        assert_argmaxes_match_except_near_ties(
            [ref[i] for i in range(USERS)], [got[i] for i in range(USERS)], f"K5 draft step {k}", near_tie_gap=1.0
        )
        assert tuple(n_k5.shape)[-1] == args.dim, "the K5 chain value must be the replicated head-norm output"
        tok_tt = _rep(mesh_device, ref.argmax(-1).reshape(USERS, 1).to(torch.int32), ttnn.uint32)
        h_ref, h_k5 = n_ref, n_k5


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("kv_only", ["1", "0"], ids=["kv_only", "full"])
def test_k5_reseed_kv(mesh_device, kv_only, monkeypatch, reset_seeds, ensure_gc):
    monkeypatch.setenv("QWEN36_MTP_KV_ONLY", kv_only)
    _, mtp, args = _build(mesh_device, monkeypatch, "table")
    assert mtp.kv_only_maint == (kv_only == "1")
    R = USERS * T
    h_frac, tok, _ = _inputs(mesh_device, args, R, seed=5)
    pos, pt = _positions(T)
    cos, sin = _rope(mesh_device, args, pos)
    pos_tt, pt_tt = _rep(mesh_device, pos, ttnn.int32), _rep(mesh_device, pt, ttnn.int32)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    caches = {}
    for name, fwd in (("ref", lambda *a, **k: Qwen36MTP.forward_decode(mtp, *a, **k)), ("k5", mtp.forward_decode)):
        kc, vc = _kv_cache(mesh_device, args, seed=6)  # identical initial content for both paths
        mtp.attention.set_paged_kv_cache(kc, vc)
        _, hn = fwd(h_frac, tok, pos_tt, cos, sin, pt_tt, need_logits=False, alias_kv_write=True, spec_n_users=USERS)
        if hn is not None:
            ttnn.deallocate(hn)
        caches[name] = (ttnn.to_torch(kc, mesh_composer=comp).float(), ttnn.to_torch(vc, mesh_composer=comp).float())
    nblk = USERS * NB + 1
    written = torch.zeros(caches["ref"][0].shape[:-1], dtype=torch.bool)  # [ndev*nblk, NKV, BLOCK]
    for d in range(mesh_device.get_num_devices()):
        for r in range(R):
            p = int(pos[r])
            written[d * nblk + int(pt[r][p // BLOCK]), :, p % BLOCK] = True
    for i, which in enumerate("kv"):
        a, b = caches["ref"][i], caches["k5"][i]
        assert torch.equal(a[~written], b[~written]), f"{which}: K5 wrote outside the reseed rows"
        pcc = compute_pcc(a[written], b[written])
        logger.info(f"[k5 reseed {kv_only=}] {which} written rows pcc={pcc:.6f}")
        assert pcc >= 0.999, f"{which}: reseed KV rows drifted (pcc {pcc})"


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=64 << 20)
def test_k5_trace(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """The K5 draft step is trace-safe: no host round trip, no program compiled inside the capture, and the replays
    reproduce the eager step exactly (persistent inputs, fixed allocation order)."""
    _, mtp, args = _build(mesh_device, monkeypatch, "table")
    k_cache, v_cache = _kv_cache(mesh_device, args, seed=7)
    mtp.attention.set_paged_kv_cache(k_cache, v_cache)
    h_frac, tok, _ = _inputs(mesh_device, args, USERS, seed=8)
    pos, pt = _positions(1)
    cos, sin = _rope(mesh_device, args, pos)
    pos_tt, pt_tt = _rep(mesh_device, pos, ttnn.int32), _rep(mesh_device, pt, ttnn.int32)

    def step():
        lg, hn = mtp.forward_decode(h_frac, tok, pos_tt, cos, sin, pt_tt)
        return lg, hn

    lg0, hn0 = step()  # eager: compiles every program
    eager = (_full(mesh_device, lg0, USERS), _replica(mesh_device, hn0, USERS))
    ttnn.deallocate(lg0)
    ttnn.deallocate(hn0)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    lg_t, hn_t = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for rep in range(2):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = (_full(mesh_device, lg_t, USERS), _replica(mesh_device, hn_t, USERS))
        assert torch.equal(got[0], eager[0]) and torch.equal(got[1], eager[1]), f"replay {rep} != eager"
    ttnn.release_trace(mesh_device, tid)
    logger.info("[k5 trace] two replays == eager, bit for bit")
