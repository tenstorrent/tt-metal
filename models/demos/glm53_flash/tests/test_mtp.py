# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP module (checkpoint layer 45, ``tt/mtp.py``) on ONE chip vs the CPU reference, real weights.

Inputs: the ``L45_S128`` host dump (``prompt_ids`` + the layer-44 output streams of the prefill and the two
decode steps = the main model's hidden states for 130 positions, so no 45-layer pass is needed). Reference: the
fp32 from-scratch MTP of ``_agents/mtp_head/mtp_reference.py`` (its check A: HF-module parity, block PCC
0.9998), embedded below in its dense-causal form (below 2048 positions every k-pool selection is the causal set,
check D of the same script).

1. prefill geometry: rows 0..15 (``T_pad`` 32) through ``prefill_forward`` for each of 2 user slots: glue PCC,
   block-output PCC >= 0.99, top-1 agreement of the logits over the 16 rows (one near-tie flip tolerated).
2. the draft step: one ``decode_forward`` at MTP position 16 for both slots (same prefix) vs row 16 of the
   reference's 17-row causal forward: row PCC >= 0.99, greedy id equal (or the reference's runner-up at a
   logit gap < 1), the two slots bit-equal, the device argmax equal to the host argmax of the device logits.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_mtp.py -s --timeout 1800

Env: ``GLM53_FULLMODEL_CACHE`` = tensorbin cache root (default ``~/ashai/glm53/_cache/fullmodel``; the layer-45
experts land in ``<root>/layer45`` on the first run, ``none`` disables).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_layer_raw,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.embedding import padded_rows
from models.demos.glm53_flash.tt.mtp import MTP_LAYER, TtMTPBlock

DUMP = Path("bringup/references/glm53_flash/L45_S128")
ROWS = 16  # prefill rows graded (positions 0..15); the decode step runs at position ROWS
_cache_env = os.environ.get("GLM53_FULLMODEL_CACHE", "/home/ttuser/ashai/glm53/_cache/fullmodel")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
PCC_BAR = 0.99
NEAR_TIE_GAP = 1.0


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


class MTPHostReference:
    """fp32 MTP forward from the raw layer-45 tensors (``_agents/mtp_head/mtp_reference.MTPReference`` with the
    dense causal mask: exact below 2048 positions, where the indexer selects every visible token)."""

    def __init__(self, cfg: GLM53FlashConfig, raw: dict[str, torch.Tensor], embed: torch.Tensor, lm_head: torch.Tensor):
        self.cfg, self.raw, self.embed, self.lm_head = cfg, raw, embed, lm_head
        self.eps = float(cfg.rms_norm_eps)
        self.n_heads, self.dq, self.dv = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
        self.scale = cfg.mla_softmax_scale

    def w(self, name: str) -> torch.Tensor:
        return self.raw[name].float()

    def project(self, next_tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        e = _rms_norm(self.embed[next_tokens], self.w("enorm.weight"), self.eps)
        h = _rms_norm(hidden, self.w("hnorm.weight"), self.eps)
        return torch.cat([e, h], dim=-1) @ self.w("eh_proj.weight").t()

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[0]
        q_resid = _rms_norm(
            x @ self.w("self_attn.q_a_proj.weight").t(), self.w("self_attn.q_a_layernorm.weight"), self.eps
        )
        q = (q_resid @ self.w("self_attn.q_b_proj.weight").t()).view(S, self.n_heads, self.dq)
        latent = _rms_norm(
            x @ self.w("self_attn.kv_a_proj_with_mqa.weight").t(), self.w("self_attn.kv_a_layernorm.weight"), self.eps
        )
        kv = (latent @ self.w("self_attn.kv_b_proj.weight").t()).view(S, self.n_heads, self.dq + self.dv)
        k, v = kv[..., : self.dq], kv[..., self.dq :]
        scores = torch.einsum("shd,thd->hst", q, k) * self.scale
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None], torch.finfo(torch.float32).min)
        o = torch.einsum("hst,thd->shd", scores.softmax(-1), v).reshape(S, -1)
        return o @ self.w("self_attn.o_proj.weight").t()

    def mlp(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        lim = float(self.cfg.swiglu_limit)
        gate = (x @ self.w(f"{prefix}.gate_proj.weight").t()).clamp(max=lim)
        up = (x @ self.w(f"{prefix}.up_proj.weight").t()).clamp(-lim, lim)
        return (F.silu(gate) * up) @ self.w(f"{prefix}.down_proj.weight").t()

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        scores = (x @ self.w("mlp.gate.weight").t()).sigmoid()
        top = (scores + self.w("mlp.gate.e_score_correction_bias")).topk(self.cfg.num_experts_per_tok, dim=-1).indices
        wts = scores.gather(1, top)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * float(self.cfg.routed_scaling_factor)
        out = torch.zeros_like(x)
        for e in top.unique().tolist():
            rows, slots = torch.where(top == e)
            out[rows] += self.mlp(x[rows], f"mlp.experts.{e}") * wts[rows, slots, None]
        return out + self.mlp(x, "mlp.shared_experts")

    def forward(self, next_tokens: torch.Tensor, hidden: torch.Tensor) -> dict:
        x = self.project(next_tokens, hidden)
        out = x + self.attention(_rms_norm(x, self.w("input_layernorm.weight"), self.eps))
        out = out + self.moe(_rms_norm(out, self.w("post_attention_layernorm.weight"), self.eps))
        y = _rms_norm(out, self.w("shared_head.norm.weight"), self.eps)
        return dict(x=x, out=out, normed=y, logits=y @ self.lm_head.float().t())


def _dump_hidden(cfg: GLM53FlashConfig, final_norm_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens [130], H [129, 4096] bf16): the main model's normed final hidden per position, the HF head path
    (unweighted bf16 stream mean, then the final norm), as ``mtp_reference.py`` derives it from the dump."""
    pre = torch.load(DUMP / "prefill_input.pt", map_location="cpu", weights_only=False)
    l44 = torch.load(DUMP / f"layer{cfg.num_hidden_layers - 1}_prefill.pt", map_location="cpu", weights_only=False)
    d0 = torch.load(DUMP / f"layer{cfg.num_hidden_layers - 1}_decode0.pt", map_location="cpu", weights_only=False)
    d1 = torch.load(DUMP / f"layer{cfg.num_hidden_layers - 1}_decode1.pt", map_location="cpu", weights_only=False)
    tokens = torch.cat([pre["prompt_ids"].reshape(-1), d0["token"].reshape(-1), d1["token"].reshape(-1)])
    streams = torch.cat([l44["x_out"][0], d0["x_out"][0]], dim=0)  # [129, 4, H] bf16
    H = _rms_norm(streams.mean(dim=1), final_norm_w, cfg.rms_norm_eps).to(torch.bfloat16)
    return tokens, H


def _greedy_ok(ref_logits: torch.Tensor, tt_id: int) -> tuple[bool, float]:
    """Equal argmax, or the reference's runner-up at a near tie (gap < NEAR_TIE_GAP); returns (ok, gap)."""
    top = ref_logits.topk(2)
    gap = float(top.values[0] - top.values[1])
    if tt_id == int(top.indices[0]):
        return True, gap
    return tt_id == int(top.indices[1]) and gap < NEAR_TIE_GAP, gap


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mtp_block_vs_reference(mesh_device):
    if not (DUMP / "prefill_input.pt").exists():
        pytest.skip(f"host dump {DUMP} missing (reference/hf_layer_reference.py --layers 0-44 --stream)")
    try:
        snapshot = default_snapshot_dir()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    headers = shard_headers(snapshot)
    w = load_tensors(snapshot, [EMBED_KEY, LM_HEAD_KEY, FINAL_NORM_KEY], headers)
    tokens, H = _dump_hidden(cfg, w[FINAL_NORM_KEY])
    t0 = time.perf_counter()
    raw = load_layer_raw(snapshot, MTP_LAYER, headers)  # the full layer (the reference needs the experts)
    logger.info(f"layer {MTP_LAYER} raw: {len(raw)} tensors in {time.perf_counter() - t0:.0f} s")

    # ---- CPU reference: rows 0..ROWS (ROWS prefill rows + the decode row), one causal forward
    R = ROWS + 1
    next_tok = tokens[1 : R + 1]  # row i embeds t_{i+1}
    t0 = time.perf_counter()
    ref = MTPHostReference(cfg, raw, w[EMBED_KEY], w[LM_HEAD_KEY]).forward(next_tok, H[:R].float())
    logger.info(f"reference forward on {R} rows: {time.perf_counter() - t0:.1f} s")

    # ---- device block: 2 user slots, the shared embedding / head built by the loader
    t0 = time.perf_counter()
    mtp = TtMTPBlock.from_checkpoint(
        mesh_device, cfg, snapshot, headers, raw=raw, max_batch=2, max_seq_len=ROWS + 64, weight_cache_root=CACHE_ROOT
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"TtMTPBlock built in {time.perf_counter() - t0:.0f} s (expert cache root {CACHE_ROOT})")

    # ---- 1. prefill geometry: rows 0..ROWS-1 for both slots (the second one re-checks the slot addressing)
    T_pad = padded_rows(ROWS)
    ids_dev = mtp.tokens_to_device(next_tok[:ROWS], pad_to=T_pad)
    hid_dev = mtp.hidden_to_device(H[:ROWS], pad_to=T_pad)
    for user in range(2):
        x_tt = mtp.project(ids_dev, hid_dev)
        x_host = ttnn.to_torch(x_tt).float()[0, 0, :ROWS]
        out_tt = mtp.prefill_rows(x_tt, user=user, seq_len=ROWS)
        normed_tt, logits_tt, ids_tt = mtp.head_forward(out_tt)
        out_host = ttnn.to_torch(out_tt).float()[0, 0, :ROWS]
        logits_host = ttnn.to_torch(logits_tt).float()[0, 0, :ROWS]
        dev_ids = ttnn.to_torch(ids_tt).view(torch.int32).reshape(-1)[:ROWS]
        for t in (out_tt, normed_tt, logits_tt, ids_tt):
            ttnn.deallocate(t)
    p_glue = pcc(ref["x"][:ROWS], x_host)
    p_out = pcc(ref["out"][:ROWS], out_host)
    p_logits = pcc(ref["logits"][:ROWS], logits_host)
    ref_top1 = ref["logits"][:ROWS].argmax(-1)
    tt_top1 = logits_host.argmax(-1)
    agree = int((ref_top1 == tt_top1).sum())
    flips = [(i, _greedy_ok(ref["logits"][i], int(tt_top1[i]))[1]) for i in range(ROWS) if ref_top1[i] != tt_top1[i]]
    logger.info(
        f"prefill {ROWS} rows: glue pcc {p_glue:.6f}, block out pcc {p_out:.6f}, logits pcc {p_logits:.6f}, "
        f"top-1 agreement {agree}/{ROWS} (flips at rows / ref gaps: {flips}); device argmax == host argmax: "
        f"{int((dev_ids == tt_top1).sum())}/{ROWS}"
    )
    assert p_glue > 0.999, f"glue pcc {p_glue:.6f}"
    assert p_out > PCC_BAR, f"block output pcc {p_out:.6f}"
    assert agree >= ROWS - 1 and all(gap < NEAR_TIE_GAP for _, gap in flips), f"top-1 agreement {agree}/{ROWS}: {flips}"
    assert torch.equal(dev_ids, tt_top1), "device argmax differs from the host argmax of the device logits"

    # ---- 2. the draft step: one decode row per slot at MTP position ROWS (row ROWS of the reference)
    dp = mtp.make_decode_positions([ROWS, ROWS])
    ids_d = mtp.tokens_to_device(next_tok[ROWS].repeat(2), pad_to=2)
    hid_d = mtp.hidden_to_device(H[ROWS : ROWS + 1].repeat(2, 1))
    out_d, normed_d, logits_d, ids_dd = mtp.decode_forward(ids_d, hid_d, dp)
    out_rows = ttnn.to_torch(out_d).float()[0, 0, :2]
    logits_rows = ttnn.to_torch(logits_d).float()[0, 0, :2]
    dev_ids = ttnn.to_torch(ids_dd).view(torch.int32).reshape(-1)[:2].tolist()
    p_dec = pcc(ref["out"][ROWS], out_rows[0])
    p_dec_logits = pcc(ref["logits"][ROWS], logits_rows[0])
    ok, gap = _greedy_ok(ref["logits"][ROWS], dev_ids[0])
    logger.info(
        f"decode step at position {ROWS}: block out pcc {p_dec:.6f}, logits pcc {p_dec_logits:.6f}, device greedy "
        f"{dev_ids} vs reference {int(ref['logits'][ROWS].argmax())} (ref top-2 gap {gap:.3f}); slot 0 == slot 1: "
        f"{torch.equal(out_rows[0], out_rows[1])}"
    )
    assert p_dec > PCC_BAR, f"decode block output pcc {p_dec:.6f}"
    assert ok, f"decode draft {dev_ids[0]} vs reference greedy {int(ref['logits'][ROWS].argmax())} (gap {gap:.3f})"
    assert dev_ids[0] == dev_ids[1] and torch.equal(out_rows[0], out_rows[1]), "the two slots (same prefix) differ"
    assert dev_ids[0] == int(logits_rows[0].argmax()), "device argmax differs from the host argmax of the device logits"
    for t in (out_d, normed_d, logits_d, ids_dd):
        ttnn.deallocate(t)
