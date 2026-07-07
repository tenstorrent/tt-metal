# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-tensor validation for the fused C++ `ttnn.transformer.gated_delta_attn_preprocess` op.

Runs the op on random inputs and compares each of its 8 outputs against a self-contained
torch reference that mirrors the Python preamble in `chunk_gated_delta_rule_seq`.

This is the incremental-debugging tool for the decay-path kernel: it isolates the preprocess
op from the scan, so a wrong output tensor is pinpointed directly.

Env:
    QWEN_TEST_G     "0" -> g=0 (no-decay slice), else random negative gates (default random)
    QWEN_TEST_ALPHA diagonal regularization alpha (default 0.25; use 0.0 for no-decay baseline)
    QWEN_TEST_BH / QWEN_TEST_NC  problem size (default 4 / 2 for a fast run)

Run:
    python models/experimental/gated_attention_gated_deltanet/tests/test_preprocess_op.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

C = 128
K = 128
V = 128


def _pcc(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if a.std() < 1e-12 and b.std() < 1e-12:
        return 1.0
    da = a - a.mean()
    db = b - b.mean()
    return (da * db).sum().item() / (torch.sqrt((da**2).sum()) * torch.sqrt((db**2).sum()) + 1e-12).item()


def _reference(q, k, v, beta, g, alpha):
    """Torch reference for the 8 preprocess outputs. Inputs are [BH, L, *] (q already scaled)."""
    BH, L, _ = q.shape
    NC = L // C
    b = BH * NC
    qc = q.reshape(b, C, K).double()
    kc = k.reshape(b, C, K).double()
    vc = v.reshape(b, C, V).double()
    betac = beta.reshape(b, C, 1).double()
    gc = g.reshape(b, C).double()

    tril = torch.tril(torch.ones(C, C, dtype=torch.double))
    eye = torch.eye(C, dtype=torch.double)

    decay_raw = torch.cumsum(gc, dim=1)  # [b, C] inclusive prefix sum
    decay_exp = torch.exp(torch.clamp(decay_raw, min=-20.0, max=0.0))  # [b, C]

    Ldiff = decay_raw[:, :, None] - decay_raw[:, None, :]  # [b, C, C]
    Lmask = torch.exp(torch.clamp(Ldiff, min=-20.0, max=0.0)) * tril  # premask absorbed by final *tril

    kbeta = kc * betac  # [b, C, K]
    kk = kbeta @ kc.transpose(1, 2)  # [b, C, C]
    kk_lmask = kk * Lmask
    kk_diag = kk_lmask * eye
    kk_reg = kk_lmask - (1.0 - alpha) * kk_diag
    Lmat = eye + kk_reg
    Dmat = Lmat * eye
    Ddiag = Dmat.sum(-1)  # [b, C]
    Dinv = 1.0 / Ddiag  # [b, C]
    Lstrict = Lmat - Dmat
    N = Dinv[:, :, None] * Lstrict
    Lunit = eye + N

    vbeta_sc = Dinv[:, :, None] * (vc * betac)
    kbd_sc = Dinv[:, :, None] * (kbeta * decay_exp[:, :, None])
    q_decay = qc * decay_exp[:, :, None]

    decay_diff = decay_raw[:, -1:] - decay_raw  # [b, C]
    decay_diff_exp = torch.exp(torch.clamp(decay_diff, min=-20.0, max=0.0))
    kdecay_t = (kc * decay_diff_exp[:, :, None]).transpose(1, 2)  # [b, K, C]

    dl_exp = torch.exp(torch.clamp(decay_raw[:, -1], min=-20.0, max=0.0))  # [b]

    intra = (qc @ kc.transpose(1, 2)) * Lmask  # lower_causal absorbed into Lmask

    # L_inv: 4 diagonal 32x32 block inverses stacked along the C (row) dim -> [b, C, 32]
    Linv = torch.zeros(b, C, 32, dtype=torch.double)
    for blk in range(C // 32):
        s = blk * 32
        Linv[:, s : s + 32, :] = torch.linalg.inv(Lunit[:, s : s + 32, s : s + 32])

    def r4(t, d1, d2):
        return t.reshape(BH, NC, d1, d2).float()

    return {
        "L_unit": r4(Lunit, C, C),
        "v_beta_sc": r4(vbeta_sc, C, V),
        "k_bd_sc": r4(kbd_sc, C, K),
        "intra_attn": r4(intra, C, C),
        "q_decay": r4(q_decay, C, K),
        "k_decay_t": r4(kdecay_t, K, C),
        "dl_exp": dl_exp.reshape(BH, NC, 1, 1).float(),
        "L_inv": r4(Linv, C, 32),
    }


def main():
    import ttnn
    from tt import ttnn_delta_rule_seq as seq

    torch.manual_seed(0)
    BH = int(os.environ.get("QWEN_TEST_BH", "4"))
    NC = int(os.environ.get("QWEN_TEST_NC", "2"))
    alpha = float(os.environ.get("QWEN_TEST_ALPHA", "0.25"))
    g_mode = os.environ.get("QWEN_TEST_G", "rand")
    L = NC * C

    q = torch.randn(BH, L, K, dtype=torch.float32) * 0.1
    k = torch.randn(BH, L, K, dtype=torch.float32) * 0.1
    v = torch.randn(BH, L, V, dtype=torch.float32) * 0.1
    beta = torch.rand(BH, L, 1, dtype=torch.float32)
    if g_mode == "0":
        g = torch.zeros(BH, L, 1, dtype=torch.float32)
    else:
        g = -torch.rand(BH, L, 1, dtype=torch.float32) * 0.1  # decay gates <= 0

    ref = _reference(q, k, v, beta, g, alpha)

    names = ["L_unit", "v_beta_sc", "k_bd_sc", "intra_attn", "q_decay", "k_decay_t", "dl_exp", "L_inv"]

    device = ttnn.open_device(device_id=0)
    try:

        def _dev(t):
            return ttnn.from_torch(
                t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        masks = seq.create_chunk_masks_seq(C, device)

        outs = ttnn.transformer.gated_delta_attn_preprocess(
            _dev(q),
            _dev(k),
            _dev(v),
            _dev(beta),
            _dev(g),
            masks["triu_ones"],
            masks["tril_mask"],
            masks["eye"],
            masks["lower_causal"],
            masks["eye_32"],
            chunk_size=C,
            diag_alpha=alpha,
            bf16_value_path=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print(f"config: BH={BH} NC={NC} alpha={alpha} g={g_mode}")
        all_ok = True
        for i, name in enumerate(names):
            got = ttnn.to_torch(outs[i]).to(torch.float32)
            want = ref[name]
            pcc = _pcc(want, got)
            maxd = (want - got).abs().max().item()
            ok = pcc >= 0.999
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'} {name:12s} PCC={pcc:.6f} max|diff|={maxd:.3e} shape={tuple(got.shape)}"
            )
        print("ALL PASS" if all_ok else "SOME FAILED")
        return 0 if all_ok else 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
