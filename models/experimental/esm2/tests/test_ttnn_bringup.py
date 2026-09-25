# SPDX-License-Identifier: MIT
"""Tiny on-device bring-up checks (random weights, small L; no checkpoint).

1. gelu op: device ttnn.gelu (default variant=Accurate == exact erf) vs torch
   exact-erf and tanh forms on identical bf16-rounded inputs.
2. rotary: device sliced-half rotary (both layouts: q [B,H,L,D], k^T
   [B,H,D,L]) vs reference_layers.apply_rotary (fp32).
3. host embedding path (token dropout) vs reference Esm2Embeddings (exact).
4. one full encoder layer on device (manual attention: matmul + additive
   mask + softmax + matmul) vs fp32 CPU twin with identical weights.

Run from the model root on a TT host: python tests/test_ttnn_bringup.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from tt.esm2.config import Esm2TTConfig  # noqa: E402
from tt.esm2.reference_layers import (  # noqa: E402
    Esm2Embeddings,
    RotaryTables,
    additive_attention_mask,
    apply_rotary,
    position_ids_from_input_ids,
)
from tt.esm2.ttnn_backend import TtnnEsm2  # noqa: E402

from tests.util import nrmse  # noqa: E402

_ALIAS = {
    "lm_dense.weight": "lm.dense.weight",
    "lm_dense.bias": "lm.dense.bias",
    "lm_ln.weight": "lm.ln.weight",
    "lm_ln.bias": "lm.ln.bias",
}


def make_twin(cfg: Esm2TTConfig, seed: int):
    """Random-weight CPU twin (reference_layers) + its canonical weight dict."""
    torch.manual_seed(seed)
    from tt.esm2.reference_layers import Esm2Model

    model = Esm2Model(cfg)
    model.eval()
    canon = {}
    for name, p in model.named_parameters():
        canon[_ALIAS.get(name, name)] = p.detach().clone()
    # plain-tensor attrs (not nn.Parameter until load_canonical is called)
    canon["embeddings.word_embeddings.weight"] = model.embeddings.weight.detach().clone()
    canon["lm.bias"] = model.lm_bias.detach().clone()
    return model, canon


def main() -> int:
    import ttnn

    cfg = Esm2TTConfig(
        num_hidden_layers=1,
        hidden_size=1280,
        num_attention_heads=20,
        intermediate_size=5120,
        vocab_size=33,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        mask_token_id=32,
        max_position_embeddings=1026,
    )
    B, L = 2, 16
    torch.manual_seed(0)
    ids = torch.randint(4, 24, (B, L))
    ids[:, 0] = 0  # cls
    ids[:, -1] = 2  # eos
    ids[0, 6:9] = cfg.pad_token_id
    ids[1, 10:12] = cfg.pad_token_id
    ids[:, [3, 5]] = cfg.mask_token_id
    am = ids.ne(cfg.pad_token_id).long()

    device = ttnn.open_device(device_id=0)
    model = None
    ok = True
    try:
        twin_model, canon = make_twin(cfg, seed=7)
        model = TtnnEsm2(cfg, canon, device=device, precision="bf16")
        model.build()

        # ---- 1) gelu variant check -----------------------------------
        x = torch.linspace(-4.0, 4.0, 512)
        xb = x.to(torch.bfloat16)
        dev = ttnn.to_device(ttnn.from_torch(xb.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), device)
        g = ttnn.gelu(dev)
        gt = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(g), ttnn.ROW_MAJOR_LAYOUT)).float().flatten()
        ref_erf = torch.nn.functional.gelu(xb.float())
        ref_tanh = torch.nn.functional.gelu(xb.float(), approximate="tanh")
        e_erf = nrmse(ref_erf, gt)
        d_tanh = (gt - ref_tanh).abs().max().item()
        d_erf = (gt - ref_erf).abs().max().item()
        print(f"[gelu  ] NRMSE vs erf={e_erf:.3e}  max|dev-erf|={d_erf:.3e}  " f"max|dev-tanh|={d_tanh:.3e}")
        ok &= e_erf < 2e-2

        # ---- 2) rotary on device (q layout and k^T layout) ------------
        pos = position_ids_from_input_ids(ids, cfg.pad_token_id)
        cos, sin = RotaryTables(cfg).cos_sin(pos)
        q = torch.randn(B, cfg.num_attention_heads, L, cfg.head_dim) * 0.5
        qb = q.to(torch.bfloat16)
        q_dev = model._dev(qb)
        q_tables, k_tables = model.rotary_tensors(ids)
        qr_dev = model._host(model._rotary_apply(q_dev, *q_tables))
        ref_q, ref_k = apply_rotary(qb.float(), qb.float(), cos, sin)
        e_rot = nrmse(ref_q, qr_dev)
        # k layout: rotate the transposed twin and compare
        kt_dev = model._dev(qb.transpose(-1, -2).contiguous())
        kr_dev = model._host(model._rotary_apply_t(kt_dev, *k_tables))
        e_rot_t = nrmse(ref_k.transpose(-1, -2), kr_dev)
        print(f"[rotary] q NRMSE={e_rot:.3e}  k^T NRMSE={e_rot_t:.3e}")
        ok &= e_rot < 1e-2 and e_rot_t < 1e-2

        # ---- 3) host embedding (token dropout) -----------------------
        h0 = model.host_embedding(ids, am)
        ref_h = Esm2Embeddings(cfg, canon["embeddings.word_embeddings.weight"])(ids, am)
        d_emb = (h0 - ref_h).abs().max().item()
        print(f"[emb   ] max|host-ref|={d_emb:.3e}")
        ok &= d_emb == 0.0

        # ---- 4) single encoder layer on device -----------------------
        x0 = torch.randn(B, L, cfg.hidden_size)
        twin_layer = twin_model.layers[0]
        attn_bias = additive_attention_mask(am)
        with torch.no_grad():
            ref_out = twin_layer(x0.clone(), attn_bias, cos, sin)
        x_dev = model._dev(x0)
        out = model._host(model._layer_forward(x_dev, 0, q_tables, k_tables, model.mask_tensor(am)))
        e_layer = nrmse(ref_out, out)
        print(f"[layer ] NRMSE={e_layer:.3e}  shape={tuple(out.shape)}")
        ok &= e_layer < 3e-2

        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if model is not None:
            model.close()
        ttnn.close_device(device)


if __name__ == "__main__":
    raise SystemExit(main())
