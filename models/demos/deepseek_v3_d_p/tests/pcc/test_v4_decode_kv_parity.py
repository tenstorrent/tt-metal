# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""M9 g2-lite -- does the prefill's exported KV equal what the DECODE ring expects in its caches?

The tt-blaze decode probe (tests/blaze/fused_ops/dsv4_hca_layer/test_csa_stage_pcc.py) seeds a real layer's caches from a
host rollout of the decode-branch formulas: window K rows = kv_norm(wkv(x_p)) roped at p, compressed entries from the
token-by-token compressor state machine (roped at the window's first position), indexer keys likewise and then
HADAMARD-ROTATED (k @ H128 / sqrt(128)) before they enter the decode indexer's key cache. With DSV4_KV_SEED_DUMP set,
the probe writes those seed tensors and its history hidden states to a file.

This test runs OUR attention module for the same real layer on the same history, exports into the contract's unified
caches exactly as the prefill runtime does, and compares row for row. INPUT CONVENTION (DS4F-0254): the probe's ``hist_x`` is
the PRE-layernorm activation (the mHC ``collapsed``); the decode ring folds the checkpoint's ``attn_norm.weight`` (HF
``input_layernorm.weight``) into its projection / compressor weights and feeds ``hist_x`` raw, so the attention input our
module must see is the reference's ``input_layernorm(hist_x) = rmsnorm(hist_x) * gamma``. Feeding ``hist_x`` raw reads as
entries PCC 0.62 / index keys 0.55 on a layer whose gamma is 0.04..0.27 (MEASURED, host). The decode ring also stores the
indexer keys Hadamard-rotated (``k @ H128 / sqrt(128)``); our export rotates them the same way, so ``index_keys_plain`` is
the one that must match.
  window ring rows [0, 128)  vs  seed[:128]           (a 128-token forward: ring row p holds token p)
  entries rows [128, 128+n)  vs  seed[128:128+n]      (a cur_pos+1-token forward)
  index keys [0, n)          vs  idx_keys             (plain and H128-rotated -- one of the two must match)
PCC >= 0.99 per group is the gate; a layout / rope / rotation convention mismatch reads as PCC ~0 on the whole group.

Run after the probe:
  DSV4_KV_SEED_DUMP=/path/seed_csa2.pt  (tt-blaze: run_probe29.sh -k csa-2 with DSV4_CSA_CUR_POS=511 DSV4_CSA_INDEXER=1)
  V4_KV_SEED=/path/seed_csa2.pt pytest tests/pcc/test_v4_decode_kv_parity.py   (2x4: TT_VISIBLE_DEVICES=0..7 + MGD)
"""

import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_block import _MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_kv_table_device import _one_chip
from models.demos.deepseek_v3_d_p.tt.v4 import kv_contract as kc
from models.demos.deepseek_v3_d_p.tt.v4.block import build_attention
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import CSA, HCA, SLIDING, layer_kinds
from models.demos.deepseek_v3_d_p.tt.v4.weights import hf_names

_MODEL = os.environ.get("PREFILL_HF_MODEL", "/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731")
_SEED_FILE = os.environ.get("V4_KV_SEED", "")
_PCC = 0.99


def _to_device_h(mesh_device, h: torch.Tensor):
    """The attention input [1, 1, S, D] sharded S over SP (dim 2) and D over TP (dim 3), bf16 TILE."""
    return ttnn.from_torch(
        h.to(torch.bfloat16).view(1, 1, h.shape[0], h.shape[1]),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )


def _export_for(kind, caches, batch=0):
    if kind == HCA:
        return (caches.hca_unified, batch)
    if kind == SLIDING:
        return (caches.swa_window, batch)
    return (caches.csa_unified, caches.csa_index_k, batch, caches.csa_pending)


def _pcc(a, b):
    return comp_pcc(a.float(), b.float())[1]


def _host_k_rows(cfg, w, rot, rope_layer_type, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The tt-metal reference's sliding-window K rows for attention input ``h`` [n, D] at ``positions`` [n]:
    kv_norm(kv_proj(h)) then the interleaved RoPE on the trailing rope slice (modeling_deepseek_v4 lines 822-823)."""
    kv = h.float() @ w["self_attn.kv_proj.weight"].float().T
    gamma = w["self_attn.kv_norm.weight"].float()
    kv = kv * torch.rsqrt(kv.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * gamma
    cos, sin = rot(kv, position_ids=positions.view(1, -1), layer_type=rope_layer_type)
    return apply_rotary_pos_emb(kv.view(1, 1, kv.shape[0], kv.shape[1]), cos, sin)[0, 0]


@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_prefill_kv_matches_decode_seed(mesh_device, device_params):
    if not _SEED_FILE or not os.path.exists(_SEED_FILE):
        pytest.skip("set V4_KV_SEED to a DSV4_KV_SEED_DUMP file from the tt-blaze decode probe")
    seed = torch.load(_SEED_FILE)
    layer = int(seed["layer_id"])
    hist_raw = seed["hist_x"].float()  # [n_hist, D]: the PRE-layernorm history (decode folds gamma into its weights)
    window, ratio = int(seed["window"]), int(seed["ratio"])
    n_hist = hist_raw.shape[0]
    sp, tp = mesh_device.shape
    cfg = deepseek_v4_flash_hf_config(max_seq=max(n_hist, 2 * window))
    kind = layer_kinds(cfg)[layer]
    logger.info(
        f"[parity] layer {layer} kind {kind} variant {seed['attn_variant']} cur_pos {seed['cur_pos']} hist {tuple(hist_raw.shape)}"
    )
    assert (
        (kind == CSA and seed["attn_variant"] == "csa")
        or (kind == HCA and seed["attn_variant"] == "hca")
        or (kind == SLIDING and seed["attn_variant"] == "swa")
    )

    wm = hf_names.read_weight_map(_MODEL)
    w = hf_names.layer_torch_dict(_MODEL, layer, weight_map=wm)
    gamma = w["input_layernorm.weight"].float()
    hist = hist_raw * torch.rsqrt(hist_raw.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * gamma  # input_layernorm
    logger.info(f"[parity] input_layernorm gamma: min {gamma.min():.4f} max {gamma.max():.4f} mean {gamma.mean():.4f}")
    rot = DeepseekV4RotaryEmbedding(cfg)
    attn = build_attention(mesh_device, cfg, layer, w, rot)

    results = {}

    def run_and_export(n_tokens: int):
        """Fresh state, one chunk of the first n_tokens history tokens, exported into fresh unified caches."""
        params = SimpleNamespace(
            max_seq_len=max(n_tokens, kc.ENGINE_CHUNK_TOKENS),
            sp_factor=sp,
            first_layer_idx=layer,
            num_layers=1,
            mesh_shape=(sp, tp),
            sp_axis=0,
            num_users=1,
        )
        caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
        state = attn.alloc_state(n_tokens, chunk_tokens=n_tokens)
        h_dev = _to_device_h(mesh_device, hist[:n_tokens])
        attn(h_dev, seq_len_actual=n_tokens, state=state, export=_export_for(kind, caches))
        ttnn.synchronize_device(mesh_device)
        exp = _export_for(kind, caches)
        unified = _one_chip(mesh_device, exp[0]).float()[0, 0]  # [rows, 512]
        keys = _one_chip(mesh_device, exp[1]).float()[0, 0] if kind == CSA else None  # [rows, 128]
        if kind == CSA and exp[3] is not None:
            run_and_export.pending = _one_chip(mesh_device, exp[3]).float()[0, 0]  # [32, 1024]
        return unified, keys

    # (1) the window ring. The decode seed holds tokens 0..127 (rows p); our smallest exportable chunk on 2x4 is 256 tokens
    # (the ring export needs a 256-row slab), whose ring holds tokens 128..255. So: (a) the tt-metal K-row formula vs the
    # decode seed on tokens 0..127 -- do the two sides' conventions (kv_norm, rope kind, [nope|rope] layout) agree? --
    # and (b) our exported ring rows vs that formula on tokens 128..255 -- does the export write what the formula says?
    seed_win = seed["kv_window_seed"][:window].float()
    formula_0 = _host_k_rows(cfg, w, rot, attn.rope_layer_type, hist[:window], torch.arange(window))
    pcc_conv = _pcc(seed_win, formula_0)
    results["window_convention(decode seed vs tt-metal formula)"] = pcc_conv
    logger.info(f"[parity] window CONVENTION tokens 0..127: decode seed vs tt-metal K formula PCC {pcc_conv:.6f}")
    unified, _ = run_and_export(2 * window)
    formula_1 = _host_k_rows(
        cfg, w, rot, attn.rope_layer_type, hist[window : 2 * window], torch.arange(window, 2 * window)
    )
    pcc_win = _pcc(formula_1, unified[:window])  # ring row p % 128 = p - 128 for tokens 128..255
    results["window_export(ring vs formula)"] = pcc_win
    logger.info(
        f"[parity] window ring rows [0,{window}) = tokens 128..255: ours vs tt-metal K formula PCC {pcc_win:.6f} "
        f"(ours |row| {unified[:window].norm(dim=-1).mean():.3f}, formula {formula_1.norm(dim=-1).mean():.3f}, decode seed {seed_win.norm(dim=-1).mean():.3f})"
    )

    # (2) entries + index keys: the whole history in one chunk
    n_tok = (n_hist // (ratio * sp)) * (ratio * sp)
    n_entries = n_tok // ratio
    unified, keys = run_and_export(n_tok)
    if os.environ.get("V4_KV_EXPORT_DUMP"):
        # g2-lite step 2: hand the PREFILL's exported rows to the decode probe (harness DSV4_KV_SEED_FROM), which seeds its
        # live caches from them instead of the host rollout and runs the decode step at cur_pos = n_tok - 1.
        torch.save(
            {
                "layer_id": layer,
                "attn_variant": seed["attn_variant"],
                "n_tok": n_tok,
                "window": window,
                "ratio": ratio,
                "hist_x": seed["hist_x"].clone(),
                "ring_rows": unified[
                    :window
                ].clone(),  # ring row r = token p with p % window == r, p in [n_tok-window, n_tok)
                "entries": unified[window : window + n_entries].clone(),
                "index_keys": None if keys is None else keys[:n_entries].clone(),  # stored convention (H128-rotated)
            },
            os.environ["V4_KV_EXPORT_DUMP"],
        )
        logger.info(f"[parity] wrote the prefill export dump {os.environ['V4_KV_EXPORT_DUMP']} ({n_tok} tokens)")
    seed_entries = seed["kv_window_seed"][window : window + n_entries].float()
    valid = seed_entries.norm(dim=-1) > 0
    pcc_ent = _pcc(seed_entries[valid], unified[window : window + n_entries][valid])
    results["entries"] = pcc_ent
    logger.info(
        f"[parity] compressed entries [{window},{window + n_entries}) ({int(valid.sum())} seeded rows): PCC {pcc_ent:.6f}  "
        f"(ours |row| {unified[window:window + n_entries][valid].norm(dim=-1).mean():.3f}, decode seed {seed_entries[valid].norm(dim=-1).mean():.3f})"
    )
    if kind == CSA and getattr(run_and_export, "pending", None) is not None:
        # contract config 4: the compressor overlap state for the decode ring's next entry (DS4F-0242)
        h4 = hist[n_tok - ratio : n_tok]
        wc, gc = w["self_attn.compressor.kv_proj.weight"].float(), w["self_attn.compressor.gate_proj.weight"].float()
        wi, gi = (
            w["self_attn.compressor.indexer.kv_proj.weight"].float(),
            w["self_attn.compressor.indexer.gate_proj.weight"].float(),
        )
        d, di = wc.shape[0] // 2, wi.shape[0] // 2
        # the stored Ca gate carries the position bias (csa_math.pool_entries keeps the BIASED gate); S % 4 == 0, so the
        # last 4 tokens are bias rows 0..3 in order
        apec = w["self_attn.compressor.position_bias"].float()[:ratio, :d]
        apei = w["self_attn.compressor.indexer.position_bias"].float()[:ratio, :di]
        exp_main = torch.cat([(h4 @ wc.T)[:, :d], (h4 @ gc.T)[:, :d] + apec], dim=1)  # [4, 1024] Ca [kv | gate + ape]
        exp_idx = torch.cat([(h4 @ wi.T)[:, :di], (h4 @ gi.T)[:, :di] + apei], dim=1)  # [4, 256]
        got = run_and_export.pending
        pcc_main = _pcc(exp_main, got[:ratio])
        pcc_idx = _pcc(exp_idx, got[ratio : 2 * ratio, : 2 * di])
        results["pending_main_ca"], results["pending_indexer_ca"] = pcc_main, pcc_idx
        logger.info(
            f"[parity] csa_pending rows 0..3 (main Ca [kv|gate]) PCC {pcc_main:.6f}; rows 4..7 (indexer Ca) PCC {pcc_idx:.6f}; "
            f"rows 8..31 |max| {got[2 * ratio:].abs().max():.3g}"
        )
    if kind == CSA and seed.get("idx_keys") is not None:
        idx = seed["idx_keys"].float()
        n = min(idx.shape[0], n_entries)
        ours = keys[:n]
        ours_rot = ours @ kc.index_key_rotation()  # R is its own inverse: this UN-rotates the exported rows
        pcc_plain, pcc_rot = _pcc(idx[:n], ours), _pcc(idx[:n], ours_rot)
        results["index_keys_plain"], results["index_keys_h128"] = pcc_plain, pcc_rot
        logger.info(
            f"[parity] indexer keys [0,{n}): PCC as exported {pcc_plain:.6f} / rotated once more (= un-rotated) {pcc_rot:.6f} "
            f"(decode stores k @ H128 / sqrt(128); the export rotates, so the first must match)"
        )
    logger.info(f"[parity] SUMMARY layer {layer} ({kind}): {results}")
    assert results["window_export(ring vs formula)"] >= _PCC, results
    assert results["window_convention(decode seed vs tt-metal formula)"] >= _PCC, results
    assert results["entries"] >= _PCC, results
    if "index_keys_h128" in results:
        assert results["index_keys_plain"] >= _PCC, results
    if "pending_main_ca" in results:
        assert results["pending_main_ca"] >= _PCC and results["pending_indexer_ca"] >= _PCC, results
