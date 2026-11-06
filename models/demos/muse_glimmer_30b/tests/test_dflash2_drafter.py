# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: TT DFlash2 drafter context pass (K/V ring contents) vs the float32 torch reference."""

import os

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.reference.dflash2 import (
    GroupedDynamicCausalConv,
    load_dflash2,
    rope_cos_sin,
    rotate_half,
)
from models.demos.muse_glimmer_30b.tt.ccl import MuseCCL
from models.demos.muse_glimmer_30b.tt.common import RopeTables, as_tt, shard_mapper
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir, dflash_checkpoint_dir

pytestmark = pytest.mark.skipif(
    not (
        os.path.exists(os.path.join(checkpoint_dir(), "config.json"))
        and os.path.exists(os.path.join(dflash_checkpoint_dir(), "model.safetensors"))
    ),
    reason="target or DFlash2 checkpoint not available",
)

PCC = 0.99
CONV_PCC = 0.999
H = 6656
ROWS = 16
_CONV_CKPT = os.path.join(dflash_checkpoint_dir(), "model.safetensors")
_mesh_params = mesh_test_params()


def _to_tt(x, mesh_device):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_cache(cache_tt, mesh_device, n_ring, block, positions):
    """Per-device ring cache -> [C, 8 kv heads, hd] float for the given positions (device d holds heads 2d, 2d+1)."""
    devs = ttnn.get_device_tensors(cache_tt)
    outs = []
    for d in devs:
        outs.append(ttnn.to_torch(d).float())  # [n_ring, kv_local, block, hd]
    rows = []
    for pos in positions:
        blk, row = (pos // block) % n_ring, pos % block
        rows.append(torch.cat([o[blk, :, row, :] for o in outs], dim=0))  # [8, hd]
    return torch.stack(rows)


@_mesh_params
@pytest.mark.timeout(1800)
def test_context_pass(mesh_device, tp):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.drafter import TtDFlash2Drafter

    target_args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, max_seq_len=8192, max_batch_size=1)
    ccl = MuseCCL(sub)
    rope = RopeTables(sub, target_args.head_dim, target_args.rope_theta, target_args.max_seq_len)
    drafter = TtDFlash2Drafter(sub, target_args, ccl, rope)
    logger.info(f"drafter init: {drafter.init_seconds:.1f}s")

    ref, cfg = load_dflash2(dflash_checkpoint_dir(), torch.float32)
    H, hd = cfg.hidden_size, cfg.head_dim
    C, P0 = 40, 3000
    torch.manual_seed(0)
    taps = torch.randn(1, C, 5 * H) * 0.5
    positions = list(range(P0, P0 + C))

    for start, n in ((0, 32), (32, C - 32)):
        tap_tt = []
        for t in range(5):
            x = torch.zeros(1, 1, 32, H)
            x[0, 0, :n] = taps[0, start : start + n, t * H : (t + 1) * H]
            tap_tt.append(_to_tt(x.bfloat16(), sub))
        rope_idx = drafter.make_rope_idx(positions[start : start + n])
        widx = drafter.make_write_idxs(P0 + start, n if n == 32 else 8, valid=n)
        drafter.context_pass(tap_tt, rope_idx, widx)
    ttnn.synchronize_device(sub)

    with torch.no_grad():
        ctx = ref.context_features(taps)
        cos, sin = rope_cos_sin(torch.tensor([positions]), hd, cfg.rope_theta, torch.float32)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)  # [1, C, 1, hd]
    failures = []
    for li in (0, 4):
        attn = ref.layers[li].self_attn
        with torch.no_grad():
            k = attn.k_norm(attn.k_proj(ctx).view(1, C, 8, hd))
            k = k * cos + rotate_half(k) * sin
            v = attn.v_proj(ctx).view(1, C, 8, hd)
        k_cache, v_cache = drafter.kv_caches[li]
        for name, ref_t, cache in (("K", k, k_cache), ("V", v, v_cache)):
            got = _read_cache(cache, sub, drafter.n_ring, drafter.block_size, positions)
            ok, msg = comp_pcc(ref_t[0], got, PCC)
            logger.info(f"layer {li} {name}: {msg}")
            if not ok:
                failures.append((li, name, msg))
    assert not failures, failures


ANCHOR, MASK = 1234, 201818


def _build(sub, num_layers=None):
    from models.demos.muse_glimmer_30b.tt.dflash.drafter import TtDFlash2Drafter

    target_args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, max_seq_len=8192, max_batch_size=1)
    ccl = MuseCCL(sub)
    rope = RopeTables(sub, target_args.head_dim, target_args.rope_theta, target_args.max_seq_len)
    return target_args, TtDFlash2Drafter(sub, target_args, ccl, rope, num_layers=num_layers)


def _run_context(drafter, sub, taps, P0):
    C, H = taps.shape[1], taps.shape[2] // 5
    for start in range(0, C, 32):
        n = min(32, C - start)
        tap_tt = []
        for t in range(5):
            x = torch.zeros(1, 1, 32, H)
            x[0, 0, :n] = taps[0, start : start + n, t * H : (t + 1) * H]
            tap_tt.append(_to_tt(x.bfloat16(), sub))
        rope_idx = drafter.make_rope_idx(range(P0 + start, P0 + start + n))
        widx = drafter.make_write_idxs(P0 + start, n, valid=n)
        drafter.context_pass(tap_tt, rope_idx, widx)


def _block_inputs(drafter, sub, noise_rows, C, P0, H):
    x = torch.zeros(1, 1, 32, H)
    x[0, 0, :16] = noise_rows
    x_tt = _to_tt(x.bfloat16(), sub)
    rope_idx = drafter.make_rope_idx(range(P0 + C, P0 + C + 16))
    widx = drafter.make_write_idxs(P0 + C, 16)
    sdpa_pos = drafter.make_sdpa_pos(P0 + C + 15)
    return x_tt, rope_idx, widx, sdpa_pos


@_mesh_params
@pytest.mark.timeout(1800)
# bf16 activations with large outlier channels; acceptance length is the real quality metric
@pytest.mark.parametrize("C,P0,min_pcc", [(48, 0, 0.98), (2100, 3000, 0.95)], ids=["short", "long_wrap"])
def test_block_forward(mesh_device, tp, C, P0, min_pcc):
    sub = skip_and_submesh(mesh_device, tp)
    target_args, drafter = _build(sub)
    ref, cfg = load_dflash2(dflash_checkpoint_dir(), torch.float32)
    H = cfg.hidden_size
    table = target_args.load_embedding_weight()
    noise = table[[ANCHOR] + [MASK] * 15].float()[None]  # [1, 16, H]
    torch.manual_seed(0)
    taps = torch.randn(1, C, 5 * H) * 0.5
    with torch.no_grad():
        expected = ref(noise, taps, torch.arange(P0, P0 + C + 16)[None])  # [1, 16, H]

    _run_context(drafter, sub, taps, P0)
    x_tt, rope_idx, widx, sdpa_pos = _block_inputs(drafter, sub, noise[0], C, P0, H)
    out = drafter.block_forward(x_tt, rope_idx, widx, sdpa_pos)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, :16]
    ok, msg = comp_pcc(expected[0], got, min_pcc)
    logger.info(f"block_forward C={C} P0={P0}: {msg}")
    assert ok, msg


@_mesh_params
@pytest.mark.timeout(1800)
def test_embed_block(mesh_device, tp):
    sub = skip_and_submesh(mesh_device, tp)
    target_args, drafter = _build(sub, num_layers=1)
    emb = target_args.load_embedding_weight()
    table_tt = as_tt(
        emb.reshape(1, 1, *emb.shape),
        sub,
        dtype=target_args.embed_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=shard_mapper(sub, -1),
        cache_file_name=target_args.cache_file("embed_tokens", target_args.embed_dtype, "v1"),
    )
    toks = torch.zeros(1, 32, dtype=torch.int32)
    toks[0, 0] = ANCHOR
    toks[0, 1:16] = MASK
    tokens_tt = ttnn.from_torch(
        toks,
        device=sub,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = drafter.embed_block(tokens_tt, table_tt)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, :16]
    ok, msg = comp_pcc(emb[toks[0, :16].long()].float(), got, 0.9999)
    logger.info(f"embed_block: {msg}")
    assert ok, msg


@_mesh_params
@pytest.mark.timeout(1800)
def test_draft_head(mesh_device, tp):
    from models.demos.muse_glimmer_30b.tt.attention import MuseLinear
    from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector
    from models.demos.muse_glimmer_30b.tt.weight_packing import pack_lm_head

    sub = skip_and_submesh(mesh_device, tp)
    target_args, drafter = _build(sub)
    ref, cfg = load_dflash2(dflash_checkpoint_dir(), torch.float32)
    H, C, P0 = cfg.hidden_size, 48, 0
    W = target_args.load_lm_head_weight()
    lm = pack_lm_head(W, target_args.padded_vocab_size)
    lm_head = MuseLinear(  # built as tt/model.py builds it (same cache file)
        lm,
        sub,
        shard_dim=-1,
        k=target_args.hidden_size,
        n=target_args.vocab_local,
        dtype=target_args.lm_head_dtype,
        cache_file_name=target_args.cache_file("lm_head", target_args.lm_head_dtype, "v2ws"),
        sharded=True,
        group="attn",
        readers=3,
        block=1,
    )
    del lm
    table = target_args.load_embedding_weight()
    noise = table[[ANCHOR] + [MASK] * 15].float()[None]
    torch.manual_seed(0)
    taps = torch.randn(1, C, 5 * H) * 0.5
    with torch.no_grad():
        h_ref = ref(noise, taps, torch.arange(P0, P0 + C + 16)[None])  # [1, 16, H]

    _run_context(drafter, sub, taps, P0)
    x_tt, rope_idx, widx, sdpa_pos = _block_inputs(drafter, sub, noise[0], C, P0, H)
    h_tt = drafter.block_forward(x_tt, rope_idx, widx, sdpa_pos)
    h_host = ttnn.to_torch(ttnn.get_device_tensors(h_tt)[0]).float()[0, 0, :16]

    vals, ids, hp_tt = drafter.draft_head(h_tt, lm_head)
    scores, cands, hp = drafter.read_draft(vals, ids, hp_tt)

    # Isolated head check on the TT hidden.
    w_hp = ref.candidate_selector.hidden_projection.weight.float()
    with torch.no_grad():
        logits = ref.compute_logits(h_host[1:16].float(), W.float())
        t_scores, t_cands = torch.topk(logits, 16, dim=-1)
        hp_ref = h_host[1:16] @ w_hp.T
    overlap = sum(len(set(cands[i].tolist()) & set(t_cands[i].tolist())) / 16 for i in range(15)) / 15
    top1 = int((cands[:, 0] == t_cands[:, 0]).sum())
    hp_ok, hp_msg = comp_pcc(hp_ref, hp, 0.999)
    logger.info(f"head check: overlap={overlap:.3f} top1={top1}/15 hp: {hp_msg}")

    # End-to-end quality (log only).
    with torch.no_grad():
        logits_e = ref.compute_logits(h_ref[0, 1:16], W.float())
        e_top1 = logits_e.argmax(-1)
        ref_path, _ = ref.propose_greedy(h_ref[:, 1:16], logits_e[None], torch.tensor([ANCHOR]))
    selector = HostCandidateSelector.from_checkpoint()
    path = selector.select(scores, cands, hp, ANCHOR)
    e2e_top1 = int((cands[:, 0] == e_top1).sum())
    eq_path = int((torch.tensor(path) == ref_path[0]).sum())
    logger.info(f"e2e: top1 agreement {e2e_top1}/15, selector path equal {eq_path}/15")

    assert overlap >= 0.75 and top1 >= 13 and hp_ok, (overlap, top1, hp_msg)


def _to_torch(t, mesh_device):
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[0:1].float()  # replicated: take rank 0 -> [1, 1, S, H]


@_mesh_params
@pytest.mark.timeout(900)
@pytest.mark.parametrize("prefix", ["layers.0.attention_conv", "layers.4.mlp_conv"])
def test_dynamic_causal_conv(mesh_device, tp, prefix):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.conv import TtDynamicCausalConv

    with safe_open(_CONV_CKPT, framework="pt") as f:
        base = f.get_tensor(f"{prefix}.base_kernel").float()
        proj = f.get_tensor(f"{prefix}.kernel_projection.weight").float()

    ref = GroupedDynamicCausalConv(H, 2, 16).float()
    ref.load_state_dict({"base_kernel": base, "kernel_projection.weight": proj})
    ref.eval()

    torch.manual_seed(0)
    h = torch.randn(1, 32, H)
    x = torch.randn(1, 32, H)
    with torch.no_grad():
        ref_out, ref_dyn = ref.prepare(h[:, :ROWS])
        ref_fin = ref.finish(x[:, :ROWS], ref_dyn)

    conv = TtDynamicCausalConv(sub, base, proj, group_size=16, block_rows=ROWS)
    h_tt = _to_tt(h.bfloat16().view(1, 1, 32, H), sub)
    x_tt = _to_tt(x.bfloat16().view(1, 1, 32, H), sub)
    out, dyn = conv.prepare(h_tt)
    fin = conv.finish(x_tt, dyn)

    out_t = _to_torch(out, sub)[0, 0, :ROWS]
    fin_t = _to_torch(fin, sub)[0, 0, :ROWS]
    ok1, msg1 = comp_pcc(ref_out[0], out_t, CONV_PCC)
    ok2, msg2 = comp_pcc(ref_fin[0], fin_t, CONV_PCC)
    logger.info(f"{prefix} prepare: {msg1}")
    logger.info(f"{prefix} finish: {msg2}")
    assert ok1, f"prepare: {msg1}"
    assert ok2, f"finish: {msg2}"
