# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device module tests (CONTRACT §3, D0 amendment 3, PLAN v1.12 / v1.24): each module alone on the test goldens'
inputs, compared with the reference's captured output (or the CPU dataflow for intermediates the reference does
not expose: rotated q / k and the attention sublayers), every comparison asserted against MODULE_FLOORS in both
precisions. Attention outputs are compared before the residual add (v1.12).

    embed        packed patch embedding vs `input_patch_embedding`
    rope_qk      rotated q and k of encoder blocks 0 and 3 (half order) vs the dataflow's pair-order rope, permuted
    enc_block    each encoder block alone, fed the golden input of that block
    enc_chain    the encoder run through on the device, each block's output vs the golden (error growth by layer)
    enc_out      the encoder output (final norm) of the device chain
    cross_attn   decoder cross-attention, q 32 rows x k 96 rows, blocks 0, 1, 5
    dec_self     decoder self-attention at blocks 1 and 5 (block 0 has v0 = v1 and cannot show a mask error)
"""
import pytest
import torch

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.helpers import MODULE_FLOORS
from models.experimental.kairos_forecast.tests.pcc.test_dataflow import GOLD, fronts
from models.experimental.kairos_forecast.tests.pcc.test_model import pcc
from models.experimental.kairos_forecast.tt.common import D_PAD, S_PAD, Precision, upload
from models.experimental.kairos_forecast.tt.model import HALF, N_TOK, TtKairos

CASES = ["ragged4", "single_gift", "forced_mixed"]


def _fires(rec, name):
    return [o for n, _a, _k, o in rec["firings"] if n == name]


def enc_dev(h):
    """(B, N + 1, 512) reference layout -> (1, 1, B * 96, 512): tokens in rows 0..N-1, [REG] in row 64."""
    B, L, D = h.shape
    out = torch.zeros(B, S_PAD, D)
    out[:, : L - 1], out[:, N_TOK] = h[:, :-1], h[:, -1]
    return out.reshape(1, 1, B * S_PAD, D)


def enc_ref(t, B, N):
    """Device (1, 1, B * 96, d) -> the reference's (B, N + 1, d); device heads (B, 8, 96, 64) -> (B, 8, N + 1, 64)."""
    if t.dim() == 4 and t.shape[1] == 8:  # heads
        return torch.cat([t[:, :, :N], t[:, :, N_TOK : N_TOK + 1]], 2)
    t = t.reshape(B, S_PAD, -1)
    return torch.cat([t[:, :N], t[:, N_TOK : N_TOK + 1]], 1)


def dec_dev(h):
    B, L, D = h.shape
    out = torch.zeros(B, D_PAD, D)
    out[:, :L] = h
    return out.reshape(1, 1, B * D_PAD, D)


def dec_ref(t, B):
    return t.reshape(B, D_PAD, -1)[:, :2]


class Checker:
    def __init__(self, case, prec):
        self.case, self.prec, self.rows = case, prec, []

    def __call__(self, module, got, exp, where=""):
        p = pcc(got, exp)
        floor = MODULE_FLOORS[module][self.prec]
        print(f"MODULE {self.case}/{self.prec} {module}{where}: pcc {p:.6f} (floor {floor}) "
              f"max|d| {float((got - exp).abs().max()):.3e}", flush=True)
        self.rows.append((module + where, p, floor))

    def assert_all(self):
        bad = [r for r in self.rows if not r[1] >= r[2]]
        assert not bad, bad


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
@pytest.mark.parametrize("case", CASES)
def test_modules(device, W, case, prec):
    import ttnn

    to = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    rec = torch.load(GOLD / f"{case}.pt", weights_only=False)
    m = TtKairos(device, W, getattr(Precision, prec)())
    up = lambda t: upload(device, t)  # noqa: E731
    chk = Checker(case, prec)
    emb_g, enc_in = _fires(rec, "input_patch_embedding"), [_fires(rec, f"encoder.block.{i}") for i in range(6)]
    enc_out_g = _fires(rec, "encoder.final_layer_norm")
    dec_g = [_fires(rec, f"decoder.block.{i}") for i in range(6)]
    for i, f in enumerate(fronts(rec)[:2]):
        B, N = f["attn_mask"].shape
        s = f["s_features"]
        inp = m.prepare(f)
        # embedding
        chk("embed", to(m.embed(inp["xs"])).view(B, N_TOK, -1)[:, :N], emb_g[i], f" fwd{i}")
        # encoder inputs per block: block 0 gets embedding + [REG]; block b > 0 the golden output of block b - 1
        ins = [torch.cat([emb_g[i], W["shared.weight"][1].expand(B, 1, 512)], 1)] + [enc_in[b][i][0] for b in range(5)]
        mask_add = ((1.0 - torch.cat([f["attn_mask"].float(), torch.ones(B, 1)], 1)) * DF.NEG)[:, None, None, :]
        for b in (0, 3):
            pre = f"encoder.block.{b}.layer.0"
            q, k, _v = m._qkv(m._norm(up(enc_dev(ins[b])), m.enc[b]["n1"]), m.enc[b], inp["enc_rope"][b], B, S_PAD)
            n = DF.rms_norm(ins[b], W[f"{pre}.layer_norm.weight"])
            fr = DF.rope_freqs(W, f"{pre}.SelfAttention", s)
            for name, dev, w in (("q", q, "q"), ("k", k, "k")):
                ref = DF.rope((n @ W[f"{pre}.SelfAttention.{w}.weight"].T).view(B, -1, 8, 64).transpose(1, 2), fr)
                chk("rope_qk", enc_ref(to(dev), B, N), ref[..., HALF], f" {name} block{b} fwd{i}")
        for b in range(6):
            got = to(m.enc_block(up(enc_dev(ins[b])), m.enc[b], inp["enc_rope"][b], inp["enc_mask"], B))
            chk("enc_block", enc_ref(got, B, N), enc_in[b][i][0], f" {b} fwd{i}")
        # chained: where the device error grows
        e = ttnn.concat([ttnn.reshape(m.embed(inp["xs"]), (B, N_TOK, 512)), inp["reg"]], dim=1)
        h = ttnn.reshape(e, (1, 1, B * S_PAD, 512))
        for b in range(6):
            h = m.enc_block(h, m.enc[b], inp["enc_rope"][b], inp["enc_mask"], B)
            chk("enc_chain", enc_ref(to(h), B, N), enc_in[b][i][0], f" {b} fwd{i}")
        enc = m._norm(h, m.enc_norm)
        chk("enc_out", enc_ref(to(enc), B, N), enc_out_g[i], f" fwd{i}")
        # decoder sublayers on golden inputs; the encoder states are the golden encoder output
        enc_g = up(enc_dev(enc_out_g[i]))
        dins = [W["shared.weight"][0].expand(B, 2, 512).clone()] + [dec_g[b][i][0] for b in range(5)]
        causal = torch.triu(torch.full((2, 2), DF.NEG), 1)[None, None]
        for b in (0, 1, 5):
            p = f"decoder.block.{b}.layer"
            cs_self, cs_q, cs_k = inp["dec_rope"][b]
            sa = DF.attention(W, f"{p}.0.SelfAttention", DF.rms_norm(dins[b], W[f"{p}.0.layer_norm.weight"]),
                              DF.rms_norm(dins[b], W[f"{p}.0.layer_norm.weight"]), causal, s)
            if b >= 1:
                got = to(m.self_attention(up(dec_dev(dins[b])), m.dec[b], cs_self, m._consts(B)["causal"], B, D_PAD))
                chk("dec_self", dec_ref(got, B), sa, f" block{b} fwd{i}")
            h1 = dins[b] + sa
            n = DF.rms_norm(h1, W[f"{p}.1.layer_norm.weight"])
            ref = DF.attention(W, f"{p}.1.EncDecAttention", n, enc_out_g[i], mask_add, s)
            got = to(m.cross_attention(up(dec_dev(h1)), enc_g, m.dec[b], cs_q, cs_k, inp["cross_mask"], B))
            chk("cross_attn", dec_ref(got, B), ref, f" block{b} fwd{i}")
    chk.assert_all()
