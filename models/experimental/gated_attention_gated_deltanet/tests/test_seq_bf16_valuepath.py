# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Numerical check for Option 2 (bf16 value-path) of `gated_delta_attn_seq`.

Runs `chunk_gated_delta_rule_seq` twice on identical random inputs — once all-fp32
(reference), once with the value-path tensors (v_beta_sc, intra_attn, q_decay,
k_decay_t) fed to the C++ kernel as bf16 — and asserts the outputs (o, final_state)
match by PCC. This isolates the kernel dtype change from the rest of the model.

Run:
    python models/experimental/gated_attention_gated_deltanet/tests/test_seq_bf16_valuepath.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _pcc(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if a.std() < 1e-12 and b.std() < 1e-12:
        return 1.0
    da = a - a.mean()
    db = b - b.mean()
    return (da * db).sum().item() / (torch.sqrt((da**2).sum()) * torch.sqrt((db**2).sum()) + 1e-12).item()


def main():
    import ttnn
    from tt import ttnn_delta_rule_seq as seq

    torch.manual_seed(0)
    BH, NC, C, K, V = 12, 8, 128, 128, 128
    T = NC * C

    q = torch.randn(BH, T, K, dtype=torch.float32) * 0.1
    k = torch.randn(BH, T, K, dtype=torch.float32) * 0.1
    v = torch.randn(BH, T, V, dtype=torch.float32) * 0.1
    beta = torch.rand(BH, T, 1, dtype=torch.float32)
    g = -torch.rand(BH, T, dtype=torch.float32) * 0.1  # decay gates <= 0

    device = ttnn.open_device(device_id=0)
    try:

        def _dev(t, shape=None):
            return ttnn.from_torch(
                t if shape is None else t.reshape(shape),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        masks = seq.create_chunk_masks_seq(C, device)

        def run():
            q_t, k_t, v_t = _dev(q), _dev(k), _dev(v)
            beta_t = _dev(beta)
            g_t = _dev(g)
            o, s = seq.chunk_gated_delta_rule_seq(
                q_t,
                k_t,
                v_t,
                beta_t,
                g_t,
                chunk_size=C,
                initial_state=None,
                mesh_device=device,
                cached_masks=masks,
            )
            return ttnn.to_torch(o).to(torch.float32), ttnn.to_torch(s).to(torch.float32)

        seq._BF16_VALUEPATH = False
        o_ref, s_ref = run()

        seq._BF16_VALUEPATH = True
        o_bf, s_bf = run()

        pcc_o = _pcc(o_ref, o_bf)
        pcc_s = _pcc(s_ref, s_bf)
        max_o = (o_ref - o_bf).abs().max().item()
        max_s = (s_ref - s_bf).abs().max().item()
        print(f"o : PCC={pcc_o:.6f}  max|diff|={max_o:.3e}")
        print(f"S : PCC={pcc_s:.6f}  max|diff|={max_s:.3e}")

        assert pcc_o >= 0.99, f"output PCC {pcc_o:.6f} < 0.99"
        assert pcc_s >= 0.99, f"final_state PCC {pcc_s:.6f} < 0.99"
        print("PASS: bf16 value-path matches fp32 (PCC >= 0.99)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
