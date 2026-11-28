# Fused GDN chunk preprocessing + gated_delta_attn_seq vs the shared ttnn_delta_rule_seq path (same scan kernel).
import pytest
import torch

import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as seq
import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill import chunk_prep


@pytest.mark.parametrize("BH, T", [(4, 256), (48, 2048)])
def test_chunk_prep(device, BH, T):
    torch.manual_seed(0)
    K = V = 128
    q = torch.nn.functional.normalize(torch.randn(BH, T, K), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(BH, T, K), dim=-1)
    v = torch.randn(BH, T, V)
    beta = torch.sigmoid(torch.randn(BH, T, 1))
    g = -torch.nn.functional.softplus(torch.randn(BH, T)) * 0.3
    s0 = torch.randn(BH, K, V) * 0.1
    F = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    args = lambda: (F(q), F(k), F(v), F(beta), F(g))
    o_ref_t, s_ref_t = seq.chunk_gated_delta_rule_seq(*args(), chunk_size=128, initial_state=F(s0), mesh_device=device)
    o_ref, s_ref = ttnn.to_torch(o_ref_t).float(), ttnn.to_torch(s_ref_t).float()
    ttnn.deallocate(o_ref_t)
    ttnn.deallocate(s_ref_t)
    saved, chunk_prep._ORIG = chunk_prep._ORIG, seq.chunk_gated_delta_rule_seq
    o, s = chunk_prep.chunk_gated_delta_rule_seq_fused(*args(), chunk_size=128, initial_state=F(s0), mesh_device=device)
    chunk_prep._ORIG = saved
    o, s = ttnn.to_torch(o).float(), ttnn.to_torch(s).float()
    ro = ((o - o_ref).norm() / o_ref.norm()).item()
    rs = ((s - s_ref).norm() / s_ref.norm()).item()
    print(f"PREP BH={BH} T={T} o rel {ro:.5f} state rel {rs:.5f}")
    assert ro < 0.01 and rs < 0.01


@pytest.mark.parametrize("gscale", [1.0, 20.0])
def test_chunk_prep_big_decay(device, gscale):
    torch.manual_seed(1)
    BH, T, K, V = 8, 512, 128, 128
    q = torch.nn.functional.normalize(torch.randn(BH, T, K), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(BH, T, K), dim=-1)
    v = torch.randn(BH, T, V)
    beta = torch.sigmoid(torch.randn(BH, T, 1))
    g = -torch.nn.functional.softplus(torch.randn(BH, T) * 3) * gscale
    F = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    prep = chunk_prep.ChunkPrep(device)
    outs = prep(ttnn.multiply(F(q), K**-0.5), F(k), F(v), F(beta), F(g), 0.25)
    names = ["L_unit", "v_beta_sc", "k_bd_sc", "intra", "q_decay", "k_decay_t", "dl_exp", "L_inv"]
    for nm, t in zip(names, outs):
        x = ttnn.to_torch(t).float()
        print(
            f"BIG g*{gscale} {nm}: finite {torch.isfinite(x).all().item()} max|x| {x[torch.isfinite(x)].abs().max().item():.3g}"
        )


def test_chunk_prep_adapter(device):
    """Through the shared adapter the model calls (GQA 16 -> 48 heads, bf16 [B, T, H, D] inputs)."""
    torch.manual_seed(2)
    B, T, Hq, H, D = 1, 2048, 16, 48, 128
    B16 = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q, k = torch.randn(B, T, Hq, D), torch.randn(B, T, Hq, D)
    v = torch.randn(B, T, H, D)
    beta = torch.sigmoid(torch.randn(B, T, H))
    g = -torch.nn.functional.softplus(torch.randn(B, T, H) * 3) * 5
    s0 = torch.randn(B, H, D, D) * 0.05
    run = lambda: seq.chunk_gated_delta_rule_seq_adapter(
        B16(q), B16(k), B16(v), B16(beta), B16(g), chunk_size=128, initial_state=B16(s0), device=device
    )
    o_r, s_r = run()
    o_r, s_r = ttnn.to_torch(o_r).float(), ttnn.to_torch(s_r).float()
    chunk_prep.install()
    try:
        o, s = run()
    finally:
        seq.chunk_gated_delta_rule_seq = chunk_prep._ORIG
        chunk_prep._ORIG = None
    o, s = ttnn.to_torch(o).float(), ttnn.to_torch(s).float()
    print(
        f"ADAPT finite o {torch.isfinite(o).all().item()} s {torch.isfinite(s).all().item()} "
        f"o rel {((o - o_r).norm() / o_r.norm()).item():.5f} s rel {((s - s_r).norm() / s_r.norm()).item():.5f}"
    )
    assert torch.isfinite(o).all() and torch.isfinite(s).all()
