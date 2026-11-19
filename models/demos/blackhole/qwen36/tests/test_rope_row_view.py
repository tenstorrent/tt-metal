# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the row view of the MTP verify's partial RoPE (rope_tp._partial_rope_decode_row_view) against the transposed
path it replaces (cos / sin [1, 1, R, 64], two transposes per tensor).

* test_the_row_view_lookup_repeats_each_rope_row_over_its_tile: Qwen36Model.rope_lookup(row_view=True) holds the host
  rows of _rope_tp_cos_sin_decode_torch, row r in tile rows r * 32 .. r * 32 + 31, and the plain lookup's rows, byte
  for byte; R = 4 and 32.
* test_the_row_view_rope_gives_the_transposed_bytes: apply_partial_rope_decode of random q [1, R, NH, 256] and k
  [1, R, NKV, 256] in L1 (as the verify hands them over), row-view cos / sin against [1, 1, R, 64]: equal byte for
  byte; R = 4 and 32.
* test_the_verify_attention_row_view_gives_the_same_bytes: one checkpoint attention layer's TPAttention._forward_verify
  with row-view cos / sin against [1, 1, R, 64] ones, on two copies of one random paged bf8 KV cache: q and k after
  RoPE, the attention output (the wo partial before the all-reduce) and both caches equal byte for byte;
  (users, T) = (1, 4) and (8, 4).
* test_spec_generation_with_the_row_view_matches_the_transposed_rope: greedy SpeculativeDecoder generation that reads
  every verify's logits, once as built and once with the verify's cos / sin in the transposed path's layout
  (Qwen36Model.rope_lookup without row_view): per-user tokens, acceptance counters, every verify's ids and logits, and
  the full-attention and drafter KV caches after the run equal byte for byte (the drafter cache without its scratch
  block); B * T = 4 (1 user) and 32 (8 users), K = 3.
* test_the_row_view_replays_faster: the RoPE of one verify (q and k of 16 attention layers, plus the row view's two
  cos / sin repeats) captured as a trace in each layout; mean replay time; R = 4 and 32.

Run:
    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_rope_row_view.py -v -s
"""

import hashlib
import os
import random
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
)
from models.demos.blackhole.qwen36.tests.test_spec_batched import _batch_prompts, _batched_kv, _build_model, _fresh_kv
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW
from models.demos.blackhole.qwen36.tests.test_spec_step import (
    MAX_SEQ_LEN,
    RD,
    _bits,
    _capture,
    _host,
    _parametrize,
    _rep,
    _replay_us,
    _rope_model,
)
from models.demos.blackhole.qwen36.tests.test_verify_attention_glue import BLOCK, _KeepPartial, _rm_int
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    apply_partial_rope_decode,
    rope_row_repeat_matrix,
    rope_rows_to_row_view,
)
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.spec_step import rope_rows

HD = 256
# Qwen3.8-27B attention: 24 q heads and 4 KV heads, split over the mesh's TP devices (at least one KV head each).
N_HEADS, N_KV_HEADS = 24, 4
ATTENTION_LAYERS = 16


def _local_heads(mesh):
    tp = mesh.get_num_devices()
    return N_HEADS // tp, max(1, N_KV_HEADS // tp)


def _lookup_rows(mesh, model, R, seed):
    """[1, R] uint32 RoPE table rows of R random positions (the last one at the table's end), and the positions."""
    rng = random.Random(seed)
    pos = [rng.randrange(0, MAX_SEQ_LEN) for _ in range(R)]
    pos[-1] = MAX_SEQ_LEN - 1
    idx = rope_rows(pos, [0] * R, model.rope_table_rows())
    return _rep(mesh, idx.reshape(1, R).to(torch.int32), ttnn.uint32), pos


def _mesh_host(t, mesh):
    """Every device's copy, concatenated on dim 0."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("R", [4, 32])
def test_the_row_view_lookup_repeats_each_rope_row_over_its_tile(mesh_device, R):
    model = _rope_model(mesh_device)
    dev_idx, pos = _lookup_rows(mesh_device, model, R, seed=R)
    plain = model.rope_lookup(dev_idx, R)
    rows = model.rope_lookup(dev_idx, R, row_view=True)
    want = model._rope_tp_cos_sin_decode_torch(torch.tensor(pos, dtype=torch.int32), [0] * R)
    for p, r, w, name in zip(plain, rows, want, ("cos", "sin")):
        assert tuple(r.shape) == (1, 1, R * ttnn.TILE_SIZE, RD) and r.layout == ttnn.TILE_LAYOUT
        got = _bits(_host(r).reshape(R * ttnn.TILE_SIZE, RD))
        assert torch.equal(got, _bits(w.reshape(R, RD).repeat_interleave(ttnn.TILE_SIZE, dim=0))), name
        assert torch.equal(got, _bits(_host(p).reshape(R, RD).repeat_interleave(ttnn.TILE_SIZE, dim=0))), name
    for t in (dev_idx, *plain, *rows):
        ttnn.deallocate(t)


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("R", [4, 32])
def test_the_row_view_rope_gives_the_transposed_bytes(mesh_device, R):
    model = _rope_model(mesh_device)
    dev_idx, pos = _lookup_rows(mesh_device, model, R, seed=100 + R)
    transposed = model.rope_lookup(dev_idx, R)
    row_view = model.rope_lookup(dev_idx, R, row_view=True)
    cos_h, sin_h = (t.reshape(R, RD).float() for t in model._rope_tp_cos_sin_decode_torch(torch.tensor(pos), [0] * R))
    g = torch.Generator().manual_seed(R)
    for heads in _local_heads(mesh_device):
        x = (torch.randn(1, R, heads, HD, generator=g) * 2).to(torch.bfloat16)
        x_dev = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        outs = [_host(apply_partial_rope_decode(x_dev, *rope, heads, R, RD)) for rope in (transposed, row_view)]
        assert tuple(outs[0].shape) == tuple(outs[1].shape) == (1, R, heads, HD)
        same = torch.equal(_bits(outs[0]), _bits(outs[1]))
        # Both are the rotation (HF rotate-half on the first 64 channels, the rest passed through).
        xf = x.float()
        rot = torch.cat([-xf[..., RD // 2 : RD], xf[..., : RD // 2]], dim=-1)
        ref = torch.cat([xf[..., :RD] * cos_h[None, :, None] + rot * sin_h[None, :, None], xf[..., RD:]], dim=-1)
        close, pcc = comp_pcc(ref, outs[1].float(), 0.999)
        logger.info(f"R={R} heads={heads}: row view bit-identical to the transposed path: {same}, PCC vs torch {pcc}")
        assert same, f"R={R} heads={heads}: max abs {(outs[0].float() - outs[1].float()).abs().max()}"
        assert torch.equal(_bits(outs[1][..., RD:]), _bits(x[..., RD:])), "the pass-through channels moved"
        assert close, f"R={R} heads={heads}: PCC vs torch {pcc}"
        ttnn.deallocate(x_dev)
    for t in (dev_idx, *transposed, *row_view):
        ttnn.deallocate(t)


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("n_users, T", [(1, 4), (8, 4)], ids=["rows4", "rows32"])
def test_the_verify_attention_row_view_gives_the_same_bytes(
    mesh_device, n_users, T, monkeypatch, reset_seeds, ensure_gc
):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=n_users, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, rd = args.n_local_kv_heads, args.rope_head_dim
    R = n_users * T
    tw = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, li), args)

    # User u's T rows sit at p_u .. p_u + T - 1: across a block boundary for odd u, across a 32-row tile for even u.
    starts = [BLOCK * (31 + 2 * u) + (BLOCK - 2 if u % 2 else 29) for u in range(n_users)]
    row_pos = torch.tensor([p + j for p in starts for j in range(T)], dtype=torch.int32)
    nb = (max(starts) + T) // BLOCK + 1 + 4
    blocks = torch.randperm(n_users * nb + 1)[: n_users * nb].reshape(n_users, nb).to(torch.int32)
    groups = TPAttention(mesh_device, args, tw, None)._spec_sdpa_plan(T, n_users)[1]
    pt_rows = _rm_int(mesh_device, blocks.repeat_interleave(T, dim=0))
    pt_users = _rm_int(mesh_device, blocks)
    pt_groups = _rm_int(mesh_device, blocks.repeat_interleave(groups // n_users, dim=0))
    cur_pos = _rm_int(mesh_device, row_pos)

    model = SimpleNamespace(args=args, rope=SimpleNamespace(rope_delta=0))
    model._rope_tp_cos_sin_decode_torch = lambda p, deltas=None: Qwen36Model._rope_tp_cos_sin_decode_torch(
        model, p, deltas
    )
    transposed = [replicate_to_device(mesh_device, t) for t in Qwen36Model._verify_cos_sin_torch(model, row_pos)]
    repeat = replicate_to_device(mesh_device, rope_row_repeat_matrix(R).reshape(1, 1, R * ttnn.TILE_SIZE, R))
    row_view = [rope_rows_to_row_view(t, repeat) for t in transposed]
    assert tuple(transposed[0].shape) == (1, 1, R, rd) and tuple(row_view[0].shape) == (1, 1, R * ttnn.TILE_SIZE, rd)

    cache_init = torch.randn(n_users * nb + 1, NKV, BLOCK, HD, dtype=torch.bfloat16)
    x = replicate_to_device(mesh_device, torch.randn(1, 1, R, args.dim, dtype=torch.bfloat16) * 2)

    def cache():
        return ttnn.from_torch(
            cache_init,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    roped = []
    rope_fn = tp_mod.apply_partial_rope_decode

    def spy(t, cos_tt, sin_tt, n_heads, batch_size, rope_dim):
        out = rope_fn(t, cos_tt, sin_tt, n_heads, batch_size, rope_dim)
        roped.append(_mesh_host(out, mesh_device))
        return out

    monkeypatch.setattr(tp_mod, "apply_partial_rope_decode", spy)

    def run(rope):
        attn = TPAttention(mesh_device, args, tw, None)
        attn.set_paged_kv_cache(cache(), cache())
        assert attn._forward_verify_ok(R, pt_rows, True, pt_groups, n_users, pt_users)
        roped.clear()
        out = attn.forward_decode(
            x,
            cur_pos,
            *rope,
            page_table=pt_rows,
            alias_kv_write=True,
            spec_verify_mode=True,
            spec_page_table=pt_groups,
            spec_n_users=n_users,
            spec_user_page_table=pt_users,
            decode_ar=_KeepPartial(),
        )
        assert len(roped) == 2, "_forward_verify rotates q and k once each"
        return attn, out, list(roped)

    a_old, out_old, (q_old, k_old) = run(transposed)
    a_new, out_new, (q_new, k_new) = run(row_view)
    checks = {
        "q after RoPE": (q_old, q_new),
        "k after RoPE": (k_old, k_new),
        "attention output": (_mesh_host(out_old, mesh_device), _mesh_host(out_new, mesh_device)),
        "K cache": (_mesh_host(a_old.paged_k, mesh_device), _mesh_host(a_new.paged_k, mesh_device)),
        "V cache": (_mesh_host(a_old.paged_v, mesh_device), _mesh_host(a_new.paged_v, mesh_device)),
    }
    for name, (a, b) in checks.items():
        same = tuple(a.shape) == tuple(b.shape) and torch.equal(_bits(a), _bits(b))
        logger.info(f"users={n_users} T={T} {name} {tuple(a.shape)}: bit-identical={same}")
        assert same, f"users={n_users} T={T}: {name} differs (max abs {(a.float() - b.float()).abs().max()})"
    ref_cache = _mesh_host(cache(), mesh_device)
    assert not torch.equal(ref_cache, checks["K cache"][1]), "the verify wrote no K row"


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _cache_digest(cache, blocks):
    """SHA-256 of every device's copy of a paged KV cache, each cut to its first ``blocks`` blocks. The drafter's cache
    has one more block, the scratch block that the reseed's padding rows all write: the order of those writes, and so
    the block's bytes, change from run to run of the same layout."""
    return _digest(torch.cat([ttnn.to_torch(d)[:blocks] for d in ttnn.get_device_tensors(cache)]))


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch", [1, 8], ids=["rows4", "rows32"])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_generation_with_the_row_view_matches_the_transposed_rope(mesh_device, batch, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

    B, K = batch, 3
    mesh_device.enable_program_cache()
    model = _build_model(mesh_device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)
    attentions = [layer.attention for layer in model.layers if layer.is_full_attention] + [model.mtp.attention]
    lookup = Qwen36Model.rope_lookup
    decode_forward = MTPSpecEngine.decode_forward

    def run(layout):
        verifies = []

        def recording(self, *a, **kw):
            res = decode_forward(self, *a, **kw)
            verifies.append((res.argmax_ids.clone(), res.logits.clone()))
            return res

        with monkeypatch.context() as m:
            m.setattr(MTPSpecEngine, "decode_forward", recording)
            if layout == "transposed":
                m.setattr(Qwen36Model, "rope_lookup", lambda self, rows, n, row_view=False: lookup(self, rows, n))
            _fresh_kv(model, kv_shape, B)  # shuts the engine down: the decoder below captures under this layout
            dec = SpeculativeDecoder(model, page_tables, draft_len=K)
            dec.read_verify_logits = True
            outs = dec.generate(prompts, MAX_NEW)
            caches = [_cache_digest(c, kv_shape[0]) for a in attentions for c in (a.paged_k, a.paged_v)]
        counters = (dec.total_accepted, dec.iters, dec.total_drafted)
        logger.info(f"B={B} {layout}: {len(verifies)} verifies, (accepted, iters, drafted) {counters}")
        return outs, counters, verifies, caches

    a_outs, a_cnt, a_ver, a_kv = run("row view")
    b_outs, b_cnt, b_ver, b_kv = run("transposed")
    assert a_outs == b_outs, f"tokens differ:\nrow view   {a_outs}\ntransposed {b_outs}"
    assert a_cnt == b_cnt, f"(accepted, iters, drafted) {a_cnt} != {b_cnt}"
    assert len(a_ver) == len(b_ver) > 1
    for step, ((a_ids, a_lg), (b_ids, b_lg)) in enumerate(zip(a_ver, b_ver)):
        assert torch.equal(a_ids, b_ids), f"verify {step}: ids {a_ids.tolist()} != {b_ids.tolist()}"
        assert torch.equal(
            _bits(a_lg), _bits(b_lg)
        ), f"verify {step}: logits differ, max abs {(a_lg - b_lg).abs().max()}"
    bad = [i for i, (a, b) in enumerate(zip(a_kv, b_kv)) if a != b]
    assert not bad, f"KV caches differ (index = 2 * attention module + (0 K, 1 V); the drafter's is last): {bad}"
    logger.info(f"B={B}: {len(a_ver)} verifies, ids, logits and {len(a_kv)} KV caches bit-identical")


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("R", [4, 32])
def test_the_row_view_replays_faster(mesh_device, R):
    model = _rope_model(mesh_device)
    dev_idx, _ = _lookup_rows(mesh_device, model, R, seed=200 + R)
    cos, sin = model.rope_lookup(dev_idx, R)
    repeat = model._rope_row_repeat(R)
    g = torch.Generator().manual_seed(R)
    xs = [
        (
            ttnn.from_torch(
                torch.randn(1, R, heads, HD, generator=g).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            heads,
        )
        for heads in _local_heads(mesh_device)
    ]

    def layers(c, s):
        for _ in range(ATTENTION_LAYERS):
            for x, heads in xs:
                ttnn.deallocate(apply_partial_rope_decode(x, c, s, heads, R, RD))

    def transposed():
        layers(cos, sin)

    def row_view():
        c, s = rope_rows_to_row_view(cos, repeat), rope_rows_to_row_view(sin, repeat)
        layers(c, s)
        ttnn.deallocate(c)
        ttnn.deallocate(s)

    us = {}
    for name, body in (("transposed", transposed), ("row view", row_view)):
        body()  # compiles every program, so the capture compiles nothing
        ttnn.synchronize_device(mesh_device)
        trace = _capture(mesh_device, body)
        us[name] = _replay_us(mesh_device, trace)
        ttnn.release_trace(mesh_device, trace)
    logger.info(
        f"R={R}: RoPE of one verify ({ATTENTION_LAYERS} layers, q and k): transposed {us['transposed']:.1f} us, "
        f"row view {us['row view']:.1f} us, saved {us['transposed'] - us['row view']:.1f} us per verify"
    )
    assert us["row view"] < us["transposed"]
    for t in (dev_idx, cos, sin, *(x for x, _ in xs)):
        ttnn.deallocate(t)
