# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced short-prompt prefill for ANY length < 2048 (change 1.4B) vs the eager masked-bucket prefill.

Device tests (TP mesh, MESH_DEVICE=P150x4; the model is truncated to 8 layers = full-attn + GDN):

* ``test_prefill_trace_any_len``       B=1: for lengths 1,7,100,128,129,500,1000,2047 compare the traced prefill
                                       with the eager one: last-row logits (PCC >= 0.999), post-prefill GDN state
                                       (rec_state, conv_states, _conv_hist_rm), KV rows [0, len), 8 decode steps.
* ``test_prefill_trace_request_order`` a long prompt then a short one (and the reverse): no state leaks across requests.
* ``test_prefill_trace_trash_block``   the padded K/V of a bucket never touches blocks other than the request's own
                                       and the trash block (sentinel-filled KV cache).
* ``test_prefill_paged_slots_traced``  B=8 mixed lengths through prefill_paged_slots (traced + device slot write) vs the
                                       eager host-snapshot path (QWEN36_PREFILL_BUCKET_TRACE=0), slot by slot.

The first group of tests (``test_host_*``) needs no device.

Run:
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
    pytest -svq models/demos/blackhole/qwen36/tests/test_prefill_trace_any_len.py
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK = 64
LENS = [1, 7, 100, 128, 129, 500, 1000, 2047]
PCC = 0.999


# --------------------------------------------------------------------------- #
# Host-only tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", LENS + [2, 3, 4, 1024, 1025])
def test_host_mask_math(length):
    """conv_sel x conv-input == [3 history zeros ; real rows][-3:] exactly; m_bg / sel are the right one-hots."""
    bucket = Qwen36Model._mask_bucket_for(length)
    bg, conv_sel, sel = Qwen36Model._short_trace_host_torch(bucket, length, 3)
    x = torch.randn(1, bucket, 16).to(torch.bfloat16)
    ref = torch.cat([torch.zeros(1, 3, 16, dtype=torch.bfloat16), x[:, :length]], dim=1)[:, -3:]
    got = (conv_sel.to(torch.bfloat16).float() @ x.float()).to(torch.bfloat16)
    assert torch.equal(ref, got)
    assert int(bg.sum()) == length and bool(bg[0, :length].all()) and not bool(bg[0, length:].any())
    assert sel.sum() == 1 and int(sel.argmax()) == length - 1


def test_host_flags(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_BUCKET_TRACE", raising=False)
    assert Qwen36Model.short_prefill_trace_enabled()
    monkeypatch.setenv("QWEN36_PREFILL_BUCKET_TRACE", "0")
    assert not Qwen36Model.short_prefill_trace_enabled()
    monkeypatch.setenv("QWEN36_PREFILL_TRACE_BUCKETS", "128,300,2048")
    assert Qwen36Model.short_prefill_trace_buckets(Qwen36Model.__new__(Qwen36Model)) == (128, 2048)


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def _gdn(model):
    return [l.attention for l in model.layers if not l.is_full_attention]


def _host(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _snapshot_state(model, mesh):
    """Host copy of every GDN layer's rec_state / conv_states / fused-decode conv history (slot 0 of dim 0 per device)."""
    out = []
    for dn in _gdn(model):
        hist = dn._conv_hist_rm
        out.append(
            {
                "rec": _host(mesh, dn.rec_state),
                "conv": [_host(mesh, c) for c in dn.conv_states],
                "hist": None if hist is None else _host(mesh, hist),
            }
        )
    return out


def _snapshot_kv(model, mesh, blocks, length):
    """Host K/V rows [0, length) of every attention layer, read through the page-table blocks `blocks`."""
    nd = mesh.get_num_devices()
    rows = []
    for k_cache, v_cache in model._paged_kv_caches:
        per = []
        for c in (k_cache, v_cache):
            h = _host(mesh, c)  # [nd * num_blocks, nkv, block, hd]
            h = h.reshape(nd, -1, *h.shape[1:])[:, list(blocks)]  # [nd, nb, nkv, block, hd]
            h = h.permute(0, 2, 1, 3, 4).reshape(nd, h.shape[2], -1, h.shape[-1])[:, :, :length]  # [nd,nkv,len,hd]
            per.append(h)
        rows.append(per)
    return rows


def _pcc(a, b):
    return float(comp_pcc(a.reshape(-1).float(), b.reshape(-1).float(), 0.0)[1])


def _assert_state_close(ref, got, tag, thr=PCC):
    for li, (r, g) in enumerate(zip(ref, got)):
        p = _pcc(r["rec"], g["rec"])
        assert p >= thr, f"{tag}: layer {li} rec_state PCC {p}"
        assert torch.equal(r["conv"][0], g["conv"][0]), f"{tag}: layer {li} conv_states[0] must be the zero tap"
        for j in range(1, len(r["conv"])):
            if float(r["conv"][j].abs().max()) == 0.0:
                assert float(g["conv"][j].abs().max()) == 0.0, f"{tag}: layer {li} conv_states[{j}] should be zero"
            else:
                p = _pcc(r["conv"][j], g["conv"][j])
                assert p >= thr, f"{tag}: layer {li} conv_states[{j}] PCC {p}"
        if r["hist"] is not None:
            if float(r["hist"].abs().max()) == 0.0:
                assert float(g["hist"].abs().max()) == 0.0, f"{tag}: layer {li} _conv_hist_rm should be zero"
            else:
                p = _pcc(r["hist"], g["hist"])
                assert p >= thr, f"{tag}: layer {li} _conv_hist_rm PCC {p}"


def _assert_kv_close(ref, got, tag, thr=PCC):
    for li, (r, g) in enumerate(zip(ref, got)):
        for name, a, b in (("K", r[0], g[0]), ("V", r[1], g[1])):
            p = _pcc(a, b)
            assert p >= thr, f"{tag}: attn layer {li} {name} rows PCC {p}"


def _decode_chain(model, mesh, first_tok, pos, page_table, n=8, forced=None):
    """n eager decode steps (teacher-forced with `forced` when given); returns (tokens, [logits])."""
    vocab = model.args.vocab_size
    toks, lgs = [first_tok], []
    cur = first_tok
    for i in range(n):
        dev = model.prepare_inputs_decode(
            torch.tensor([[cur]], dtype=torch.int32), torch.tensor([pos + i], dtype=torch.int32), page_table
        )
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        lg = model.process_output_decode(out, 1).reshape(-1)[:vocab].float()
        lgs.append(lg)
        nxt = int(torch.argmax(lg))
        cur = forced[i + 1] if forced is not None else nxt
        toks.append(nxt)
    return toks, lgs


def _logits_host(model, lg):
    """Host [vocab] float from a prefill_traced_chunked result (device tensor or host torch)."""
    if isinstance(lg, torch.Tensor):
        return lg.reshape(-1)[: model.args.vocab_size].float()
    return _host(model.mesh_device, lg).reshape(-1, model.args.vocab_size)[0]


def _setup_b1(mesh, n_layers=8, num_blocks=64):
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=4096, n_layers=n_layers)
    args = model.args
    kv_shape = (num_blocks, args.n_local_kv_heads, BLOCK, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    model.prefill_trash_block = num_blocks - 1  # spare block: lengths <= 2047 own blocks [0, 32)
    return model, page_table


def _eager_reference(model, mesh, prompt, page_table, length):
    """Eager masked-bucket prefill (traces not captured yet) -> (logits, state, kv, decode tokens/logits)."""
    toks = torch.tensor([prompt[:length]], dtype=torch.long)
    lg = _logits_host(model, model.prefill_traced_chunked(toks, page_table, actual_len=length))
    state = _snapshot_state(model, mesh)
    kv = _snapshot_kv(model, mesh, range((length + BLOCK - 1) // BLOCK), length)
    first = int(torch.argmax(lg))
    dec_t, dec_l = _decode_chain(model, mesh, first, length, page_table)
    return {"logits": lg, "state": state, "kv": kv, "tokens": dec_t, "dec_logits": dec_l}


# --------------------------------------------------------------------------- #
# Device tests
# --------------------------------------------------------------------------- #
@torch.no_grad()
@parametrize_mesh_tp()
def test_prefill_trace_any_len(mesh_device, reset_seeds, ensure_gc):
    mesh = mesh_device
    assert mesh.get_num_devices() > 1
    model, page_table = _setup_b1(mesh)
    vocab = model.args.vocab_size
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (2048,)).tolist()

    # Eager references FIRST (no trace parked yet: the eager path may compile freely).
    model.capture_prefill_trace_chunked(mesh, page_table, chunk_size=2048, capture_chunk_trace=False)
    refs = {L: _eager_reference(model, mesh, prompt, page_table, L) for L in LENS}

    model.capture_prefill_traces_short(mesh, page_table)  # all buckets 128..2048
    for L in LENS:
        toks = torch.tensor([prompt[:L]], dtype=torch.long)
        got_logits = _logits_host(model, model.prefill_traced_chunked(toks, page_table, actual_len=L))
        ref = refs[L]
        # Was the trace really used? (the eager path would also pass; the trace must exist for this bucket)
        assert model._short_trace_for(L) is not None, f"no short trace serves length {L}"
        p = _pcc(ref["logits"], got_logits)
        assert p >= PCC, f"L={L}: last-row logits PCC {p}"
        if L == 1 or float(ref["logits"].topk(2).values.diff().abs()) > 0.1:
            assert int(torch.argmax(got_logits)) == int(torch.argmax(ref["logits"])), f"L={L}: first token differs"
        _assert_state_close(ref["state"], _snapshot_state(model, mesh), f"L={L} state")
        _assert_kv_close(ref["kv"], _snapshot_kv(model, mesh, range((L + BLOCK - 1) // BLOCK), L), f"L={L} kv")
        # 8 teacher-forced decode steps from the traced state.
        _, dec_l = _decode_chain(model, mesh, int(torch.argmax(ref["logits"])), L, page_table, forced=ref["tokens"])
        for i, (a, b) in enumerate(zip(ref["dec_logits"], dec_l)):
            pd = _pcc(a, b)
            assert pd >= 0.99, f"L={L}: decode step {i} logits PCC {pd}"
        logger.info(f"L={L}: logits PCC {p:.5f}, 8 decode steps OK")


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("order", [(500, 100), (100, 500), (7, 1000, 129, 1)], ids=["500_100", "100_500", "mixed"])
def test_prefill_trace_request_order(mesh_device, order, reset_seeds, ensure_gc):
    """Replaying a bucket after a request of ANOTHER length must equal that request run on its own (no mask / state /
    page-table leakage between requests)."""
    mesh = mesh_device
    model, page_table = _setup_b1(mesh)
    torch.manual_seed(1)
    prompts = {L: torch.randint(0, model.args.vocab_size, (L,)).tolist() for L in set(order)}
    model.capture_prefill_trace_chunked(mesh, page_table, chunk_size=2048, capture_chunk_trace=False)
    refs = {L: _eager_reference(model, mesh, prompts[L], page_table, L) for L in set(order)}
    model.capture_prefill_traces_short(mesh, page_table, buckets=(128, 256, 512, 1024))
    for L in order:
        toks = torch.tensor([prompts[L]], dtype=torch.long)
        got = _logits_host(model, model.prefill_traced_chunked(toks, page_table, actual_len=L))
        assert _pcc(refs[L]["logits"], got) >= PCC, f"order {order}: L={L} logits"
        _assert_state_close(refs[L]["state"], _snapshot_state(model, mesh), f"order {order} L={L}")
        _assert_kv_close(
            refs[L]["kv"], _snapshot_kv(model, mesh, range((L + BLOCK - 1) // BLOCK), L), f"order {order} L={L}"
        )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("L", [1, 100, 129, 1000], ids=lambda x: f"L{x}")
def test_prefill_trace_trash_block(mesh_device, L, reset_seeds, ensure_gc):
    """Padded K/V of the bucket must land ONLY in the trash block: sentinel-fill the whole KV cache, replay, and check that
    every block outside {request blocks, trash} still holds the sentinel."""
    mesh = mesh_device
    model, _ = _setup_b1(mesh, num_blocks=64)
    nreal = (L + BLOCK - 1) // BLOCK
    trash = 63
    # The request owns blocks 40..40+nreal-1 (NOT 0..): other blocks 0..39 / 40+nreal..62 hold "other users" data.
    page_table = torch.zeros(1, 64, dtype=torch.int32)
    page_table[0, :nreal] = torch.arange(40, 40 + nreal, dtype=torch.int32)
    model.prefill_trash_block = trash
    pt_warm = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    model.capture_prefill_trace_chunked(mesh, pt_warm, chunk_size=2048, capture_chunk_trace=False)
    model.capture_prefill_traces_short(mesh, pt_warm, buckets=(Qwen36Model._mask_bucket_for(L),))
    sentinel = 3.0
    for k_cache, v_cache in model._paged_kv_caches:
        for c in (k_cache, v_cache):
            shp = _host(mesh, c).shape
            fill = ttnn.from_torch(
                torch.full(shp, sentinel),
                dtype=c.dtype,
                layout=c.layout,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            ttnn.copy(fill, c)
            ttnn.deallocate(fill)
    toks = torch.randint(0, model.args.vocab_size, (1, L), dtype=torch.long)
    assert model.prefill_short_traced_logits(toks, page_table, L) is not None
    nd = mesh.get_num_devices()
    allowed = set(range(40, 40 + nreal)) | {trash}
    for k_cache, v_cache in model._paged_kv_caches:
        for c in (k_cache, v_cache):
            h = _host(mesh, c)
            h = h.reshape(nd, -1, *h.shape[1:])
            for b in range(h.shape[1]):
                if b in allowed:
                    continue
                assert bool((h[:, b] == sentinel).all()), f"L={L}: block {b} (not the request's / trash) was written"
        # The request's own blocks were really written.
    assert not bool(
        (_host(mesh, model._paged_kv_caches[0][0]).reshape(nd, -1, *tuple(c.shape)[1:])[:, 40] == sentinel).all()
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8], ids=["B8"])
def test_prefill_paged_slots_traced(mesh_device, B, monkeypatch, reset_seeds, ensure_gc):
    """B=8 mixed lengths via prefill_paged_slots: traced short prefill + device-side slot write vs the eager host-snapshot
    path (QWEN36_PREFILL_BUCKET_TRACE=0), slot by slot (logits, rec_state / conv_states of the slot, KV rows)."""
    mesh = mesh_device
    lens = [1, 7, 100, 128, 129, 500, 1000, 1500]
    assert len(lens) == B
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=B, max_seq_len=4096, n_layers=8)
    args, vocab = model.args, model.args.vocab_size
    bpu = 32  # >= 2048 / 64: the bucket-2048 chunk page table is 32 wide
    total_blocks = B * bpu + 1
    model.allocate_kv_caches((total_blocks, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=B)
    model.prefill_trash_block = B * bpu  # spare block after the users' blocks
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    torch.manual_seed(2)
    toks = [torch.randint(0, vocab, (1, L), dtype=torch.long) for L in lens]
    # A permuted slot assignment (what the plugin's empty_slots may look like): request u -> slot perm[u].
    perm = [3, 0, 5, 1, 7, 2, 6, 4]
    nd = mesh.get_num_devices()

    def _slot_state(slot):
        out = []
        for dn in _gdn(model):
            rec = _host(mesh, dn.rec_state).reshape(nd, B, *tuple(dn.rec_state.shape)[1:])[:, slot]
            conv = [_host(mesh, c).reshape(nd, 1, B, -1)[:, :, slot] for c in dn.conv_states]
            out.append((rec, conv))
        return out

    def _run():
        host = model.prefill_paged_slots(toks, page_table, perm, valid_lens=lens)
        res = []
        for u in range(B):
            blocks = page_table[u, : (lens[u] + BLOCK - 1) // BLOCK].tolist()
            res.append(
                (host[u].reshape(-1)[:vocab].float(), _slot_state(perm[u]), _snapshot_kv(model, mesh, blocks, lens[u]))
            )
        return res

    monkeypatch.setenv("QWEN36_PREFILL_BUCKET_TRACE", "0")
    ref = _run()  # eager masked-bucket + host round trip
    monkeypatch.setenv("QWEN36_PREFILL_BUCKET_TRACE", "1")
    warm_pt = torch.cat([torch.arange(total_blocks, dtype=torch.int32), torch.zeros(31, dtype=torch.int32)])
    warm_pt = warm_pt[: ((total_blocks + 31) // 32) * 32].reshape(1, -1)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_traces_short(mesh, warm_pt)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    got = _run()
    for u in range(B):
        p = _pcc(ref[u][0], got[u][0])
        assert p >= PCC, f"slot {perm[u]} (len {lens[u]}): logits PCC {p}"
        for li, ((rr, rc), (gr, gcv)) in enumerate(zip(ref[u][1], got[u][1])):
            assert _pcc(rr, gr) >= PCC, f"slot {perm[u]} layer {li} rec_state"
            for j in range(1, len(rc)):
                if float(rc[j].abs().max()) > 0:
                    assert _pcc(rc[j], gcv[j]) >= PCC, f"slot {perm[u]} layer {li} conv_states[{j}]"
        _assert_kv_close(ref[u][2], got[u][2], f"slot {perm[u]} len {lens[u]} kv")
    del model
    gc.collect()
