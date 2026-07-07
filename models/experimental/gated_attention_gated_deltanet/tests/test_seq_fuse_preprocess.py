# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end numerical check for `QWEN_GDN_FUSE_PREPROCESS`.

Runs `chunk_gated_delta_rule_seq` twice on identical random inputs — once with the Python
preprocessing preamble (reference), once with the fused C++ `gated_delta_attn_preprocess` op — and
asserts the outputs (o, final_state) match by PCC. This exercises the full fused preamble + scan.

Run:
    python models/experimental/gated_attention_gated_deltanet/tests/test_seq_fuse_preprocess.py
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

        def _dev(t):
            return ttnn.from_torch(
                t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        masks = seq.create_chunk_masks_seq(C, device)

        def run(fuse):
            seq._FUSE_PREPROCESS = fuse
            o, s = seq.chunk_gated_delta_rule_seq(
                _dev(q),
                _dev(k),
                _dev(v),
                _dev(beta.clone()),
                _dev(g.clone()),
                chunk_size=C,
                initial_state=None,
                mesh_device=device,
                cached_masks=masks,
            )
            return ttnn.to_torch(o).to(torch.float32), ttnn.to_torch(s).to(torch.float32)

        o_ref, s_ref = run(False)
        o_fused, s_fused = run(True)

        pcc_o = _pcc(o_ref, o_fused)
        pcc_s = _pcc(s_ref, s_fused)
        max_o = (o_ref - o_fused).abs().max().item()
        max_s = (s_ref - s_fused).abs().max().item()
        print(f"o : PCC={pcc_o:.6f}  max|diff|={max_o:.3e}")
        print(f"S : PCC={pcc_s:.6f}  max|diff|={max_s:.3e}")

        assert pcc_o >= 0.999, f"output PCC {pcc_o:.6f} < 0.999"
        assert pcc_s >= 0.999, f"final_state PCC {pcc_s:.6f} < 0.999"
        print("PASS: fused preprocess path matches Python preamble (PCC >= 0.999)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
