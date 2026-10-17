# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_VERIFY_SPEC_STEP=1: the fused gdn_spec_step spec verify against the composite verify and against plain decode.

Layer level (random GDN weights, TP mesh, no checkpoint beyond config.json via HF_MODEL):
  * the fused T = 1 SEED (gdn_spec_step over rec_state / _conv_win_buf) vs one plain decode step of a layer on the
    gdn_decode_step op (fast_mode + fp32_state): rec_state and the conv shift register torch.equal, outputs equal;
  * a (B, T) VERIFY with the flag ON vs OFF (the composite _verify_fullbatch chain) from the same seeded state: ring
    contents and layer outputs within PCC (different kernels, ~1e-5), the new conv window rows torch.equal;
  * the same verify vs T chained plain decode steps: ring block t of every user torch.equal the decode state after
    token t (the per-token arithmetic IS one gdn_decode_step), outputs equal; a second verify at a non-zero accepted
    index mi per user (the carry rows E_prev[mi+1 : mi+K]) again ON vs OFF.
Model level (checkpoint): spec decode with the flag ON must follow plain greedy decode token for token up to bf16
near-ties (test_spec_lossless's gate); with one GDN kernel family the GDN layers no longer contribute flips.

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_spec_verify_specstep.py -sv
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    model_path,
    parametrize_mesh_tp,
    random_gdn_state_dict,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import (
    TPGatedDeltaNet,
    load_gdn_weights_tp,
    spec_conv_sel,
    spec_ctrl_page,
    spec_state_blk_idx,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

PCC = 0.999


def _cat0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _rep_u32(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _build(mesh, B, monkeypatch):
    """Three layers sharing one random weight set: `g_on` (QWEN36_VERIFY_SPEC_STEP=1: gdn_spec_step verify/seed),
    `g_off` (the composite verify/seed) and `g_dec` (plain decode on the gdn_decode_step op, fast_mode + fp32_state)."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh, max_batch_size=B, max_seq_len=256)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = random_gdn_state_dict(args, seed=li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh) if mesh.get_num_devices() > 1 else None
    monkeypatch.setenv("QWEN36_GDN_FUSED_DECODE", "1")
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "1")
    monkeypatch.setenv("QWEN36_GDN_STEP_FP32_STATE", "1")
    tw = load_gdn_weights_tp(mesh, sd, args)
    assert "conv_taps_rows" in tw, "load_gdn_weights_tp did not build the gdn_spec_step taps"
    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", "1")
    g_on = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", "0")
    g_off = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    g_dec = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    assert g_on._spec_fused and not g_off._spec_fused and not g_dec._spec_fused
    assert g_on._decode_op and g_dec._decode_op and g_dec._decode_step_fp32_state
    return args, g_on, g_off, g_dec


def _seed(mesh, g, rec0, hist0):
    """Identical durable state in every layer: rec0 [B] x [1, Nv, Dk, Dv] fp32, hist0 [B] x [1, 3, C] bf16."""
    g._stable_state = True
    g.reset_state()
    rec = [
        ttnn.from_torch(
            r, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        for r in rec0
    ]
    g.assemble_batched_state(rec, [replicate_to_device(mesh, h) for h in hist0])


def _hist_rm(mesh, g, nd, B, C):
    """The plain decode layer's conv history [nd, B, 3, C] (last three inputs, oldest first) in the RM format."""
    if g._conv_fmt == "packed":
        g._unpack_conv_hist()
    return _cat0(mesh, g._conv_hist_rm).reshape(nd, B, 3, C)


def _ring(mesh, g, nd, T, B, Nv, Dk, Dv):
    return _cat0(mesh, g._spec_ring).reshape(nd, T, B, Nv, Dk, Dv)


def _rec(mesh, g, nd, B, Nv, Dk, Dv):
    return _cat0(mesh, g.rec_state).reshape(nd, B, Nv, Dk, Dv)


def _win_rows(mesh, g, nd, B):
    """The current E_prev window [nd, B, L, C] (flag on: the pair half the next verify reads)."""
    w = g.verify_win_cur()
    return _cat0(mesh, w).reshape(nd, B, w.shape[1], w.shape[2])


def _cmp(tag, a, b, exact):
    eq = torch.equal(a, b)
    d = (a - b).abs().max().item()
    p = compute_pcc(a, b)
    logger.info(f"[specstep] {tag}: torch.equal={eq} max|d|={d:.3e} pcc={p:.7f}")
    if exact:
        assert eq, f"{tag}: not bit-identical (max|d|={d:.3e}, pcc={p:.7f})"
    else:
        assert p >= PCC, f"{tag}: pcc {p:.6f} < {PCC}"
    return eq


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B,T", [(1, 4), (8, 4), (4, 8), (2, 12)])
def test_layer_verify_specstep_vs_composite_and_decode(mesh_device, B, T, reset_seeds, ensure_gc, monkeypatch):
    mesh = mesh_device
    args, g_on, g_off, g_dec = _build(mesh, B, monkeypatch)
    nd = mesh.get_num_devices()
    Nv, Dk, Dv, C, K = g_on.Nv, g_on.Dk, g_on.Dv, g_on.qkv_dim_tp, g_on.K
    comp = tp_composer(mesh)
    rec0 = [0.1 * torch.randn(1, Nv, Dk, Dv) for _ in range(B)]
    hist0 = [torch.randn(1, 3, C, dtype=torch.bfloat16) for _ in range(B)]
    for g in (g_on, g_off, g_dec):
        _seed(mesh, g, rec0, hist0)

    def dec_step(x_rows):
        """One plain decode step (gdn_decode_step op) over [1, 1, B, dim]; returns the layer output [1,1,B,dim]."""
        g_dec.prepare_decode_width(B)
        assert g_dec._conv_fmt == "packed"
        return ttnn.to_torch(g_dec.forward_decode(replicate_to_device(mesh, x_rows)), mesh_composer=comp).float()

    # ---- 1. the fused T = 1 seed vs one plain decode step (vs the composite seed within PCC) ----
    x_seed = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    out_seed_on = ttnn.to_torch(
        g_on.forward_seed_recurrent(replicate_to_device(mesh, x_seed), B, pre_gathered=True, n_users=B),
        mesh_composer=comp,
    ).float()
    out_seed_off = ttnn.to_torch(
        g_off.forward_seed_recurrent(replicate_to_device(mesh, x_seed), B, pre_gathered=True, n_users=B),
        mesh_composer=comp,
    ).float()
    out_seed_dec = dec_step(x_seed)
    _cmp(
        "seed: rec_state fused-seed vs decode op",
        _rec(mesh, g_on, nd, B, Nv, Dk, Dv),
        _rec(mesh, g_dec, nd, B, Nv, Dk, Dv),
        True,
    )
    _cmp(
        "seed: conv register fused-seed vs decode op",
        _cat0(mesh, g_on._conv_win_buf).reshape(nd, B, K, C)[:, :, 1:],
        _hist_rm(mesh, g_dec, nd, B, C),
        True,
    )
    eq_out = _cmp("seed: layer output fused-seed vs decode op", out_seed_on, out_seed_dec, False)
    _cmp(
        "seed: rec_state fused-seed vs composite seed",
        _rec(mesh, g_on, nd, B, Nv, Dk, Dv),
        _rec(mesh, g_off, nd, B, Nv, Dk, Dv),
        False,
    )
    _cmp("seed: layer output fused-seed vs composite seed", out_seed_on, out_seed_off, False)

    # ---- 2. the (B, T) verify at mi = 0: flag on vs off, and vs T chained plain decode steps ----
    for g in (g_on, g_off):
        g.prepare_spec_verify(B, T)
        g.seed_spec_state()
    idx0 = _rep_u32(mesh, spec_state_blk_idx([0] * B, B, Nv))
    sel0 = replicate_to_device(mesh, spec_conv_sel([0] * B, B, T, K))
    ctrl0 = _rep_u32(mesh, spec_ctrl_page([0] * B, B, Nv, parity=g_on._spec_win_par))
    x = torch.randn(1, 1, B * T, args.dim, dtype=torch.bfloat16)  # user-major rows u*T + t
    out_on = ttnn.to_torch(
        g_on.forward_verify_recurrent(
            replicate_to_device(mesh, x), B * T, pre_gathered=True, n_users=B, spec_ctrl=ctrl0
        ),
        mesh_composer=comp,
    ).float()
    g_on._spec_win_par ^= 1  # what model.verify_traced does after a replay
    out_off = ttnn.to_torch(
        g_off.forward_verify_recurrent(
            replicate_to_device(mesh, x), B * T, pre_gathered=True, n_users=B, state_blk_idx=idx0, conv_sel=sel0
        ),
        mesh_composer=comp,
    ).float()
    ring_on = _ring(mesh, g_on, nd, T, B, Nv, Dk, Dv)
    ring_off = _ring(mesh, g_off, nd, T, B, Nv, Dk, Dv)
    for t in range(T):
        _cmp(f"verify t={t}: ring on vs off", ring_on[:, t], ring_off[:, t], False)
    _cmp("verify: layer output on vs off", out_on[:, :, : B * T], out_off[:, :, : B * T], False)
    win_on, win_off = _win_rows(mesh, g_on, nd, B), _win_rows(mesh, g_off, nd, B)
    _cmp("verify: new conv window on vs off", win_on, win_off, True)
    # vs plain decode, token by token (the per-token arithmetic of the fused verify IS one gdn_decode_step)
    outs_dec = []
    for t in range(T):
        x_t = x[:, :, [u * T + t for u in range(B)], :]
        outs_dec.append(dec_step(x_t))
        _cmp(f"verify t={t}: ring block vs decode state", ring_on[:, t], _rec(mesh, g_dec, nd, B, Nv, Dk, Dv), True)
    out_on_tok = torch.stack([out_on[0, 0, [u * T + t for u in range(B)]] for t in range(T)])  # [T, B, dim]
    out_dec_tok = torch.stack([o[0, 0, :B] for o in outs_dec])
    eq_out &= _cmp("verify: layer outputs vs decode steps", out_on_tok, out_dec_tok, False)
    # the new window's last K-1 rows are the decode layer's conv history
    _cmp("verify: window tail vs decode history", win_on[:, :, T : T + K - 1], _hist_rm(mesh, g_dec, nd, B, C), True)
    logger.info(
        f"[specstep] layer outputs bit-identical to plain decode (out-proj + reduce at B*T vs B rows): {eq_out}"
    )

    # ---- 3. a second verify at a non-zero accepted index per user: flag on vs off ----
    mi = [(u + 1) % T for u in range(B)]
    ctrl1 = _rep_u32(mesh, spec_ctrl_page(mi, B, Nv, parity=g_on._spec_win_par))
    idx1 = _rep_u32(mesh, spec_state_blk_idx(mi, B, Nv))
    sel1 = replicate_to_device(mesh, spec_conv_sel(mi, B, T, K))
    x2 = torch.randn(1, 1, B * T, args.dim, dtype=torch.bfloat16)
    out2_on = ttnn.to_torch(
        g_on.forward_verify_recurrent(
            replicate_to_device(mesh, x2), B * T, pre_gathered=True, n_users=B, spec_ctrl=ctrl1
        ),
        mesh_composer=comp,
    ).float()
    g_on._spec_win_par ^= 1
    out2_off = ttnn.to_torch(
        g_off.forward_verify_recurrent(
            replicate_to_device(mesh, x2), B * T, pre_gathered=True, n_users=B, state_blk_idx=idx1, conv_sel=sel1
        ),
        mesh_composer=comp,
    ).float()
    ring2_on, ring2_off = _ring(mesh, g_on, nd, T, B, Nv, Dk, Dv), _ring(mesh, g_off, nd, T, B, Nv, Dk, Dv)
    for t in range(T):
        _cmp(f"verify2 (mi={mi}) t={t}: ring on vs off", ring2_on[:, t], ring2_off[:, t], False)
    _cmp("verify2: layer output on vs off", out2_on[:, :, : B * T], out2_off[:, :, : B * T], False)
    _cmp("verify2: new conv window on vs off", _win_rows(mesh, g_on, nd, B), _win_rows(mesh, g_off, nd, B), True)
    assert torch.isfinite(out2_on).all()


@run_for_blackhole()
@pytest.mark.parametrize("flag", ["1"])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_specstep_matches_plain_greedy(mesh_device, flag, monkeypatch):
    """test_spec_lossless's gate with QWEN36_VERIFY_SPEC_STEP=1: the spec trajectory vs plain decode teacher-forced
    down it. Plain decode runs the gdn_decode_step op here (set_gdn_fused_decode(True) is a no-op under the flag), the
    same arithmetic the fused verify runs, so confident mismatches are a bug and near-tie flips should be fewer than
    the composite path's (logged).

      MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 pytest models/demos/blackhole/qwen36/tests/test_spec_verify_specstep.py -k greedy -sv
    """
    from models.demos.blackhole.qwen36.tests.test_spec_lossless import (
        _N_LAYERS,
        MAX_NEW,
        NUM_BLOCKS,
        PROMPT_LEN,
        _assert_lossless_up_to_near_ties,
        _reference_greedy,
    )
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4 / P150x8")
    from transformers import AutoTokenizer

    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", flag)
    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, n_layers=_N_LAYERS
    )
    assert model.mtp is not None, "MTP head not built"
    assert model.gdn_verify_spec_step, "QWEN36_VERIFY_SPEC_STEP=1 did not take (see the [GDN] log lines)"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = _get_prompt(PROMPT_LEN, tokenizer)[0].tolist()
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)

    # The usual spec setup call: under the flag it must leave plain decode on the gdn_decode_step op.
    model.set_gdn_fused_decode(True)
    assert not model.gdn_fused_decode, "set_gdn_fused_decode(True) must be ignored under QWEN36_VERIFY_SPEC_STEP=1"

    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = SpeculativeDecoder(model, pt)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[specstep] spec {len(spec)} tokens: {spec}")
    logger.info(f"[specstep] spec text: {tokenizer.decode(spec)!r}")
    dec.log_stats(prefix="specstep")
    model.free_kv_caches()
    assert len(spec) == MAX_NEW, f"expected {MAX_NEW} tokens from spec decode, got {len(spec)}"

    ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, spec)
    logger.info(f"[specstep] plain-decode argmax at each spec position: {ref}")
    n_mis = sum(1 for i in range(len(spec)) if spec[i] != ref[i])
    logger.info(f"[specstep] mismatches vs plain greedy (near-tie or not): {n_mis}/{len(spec)}")
    _assert_lossless_up_to_near_ties(spec, ref, gaps, tokenizer, tag="spec-step")
