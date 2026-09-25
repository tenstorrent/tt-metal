# SPDX-License-Identifier: MIT
"""Self-contained portable test: CPU FP32 twin invariants.

No TT device, no /weights checkpoint, no prepared /input files — torch CPU
only. Runnable on any machine with the package on sys.path. Run from the
model root: python tests/test_portable_model.py

Note: Esm2Model(cfg) without a weights dict intentionally starts from a
zero embedding table (production loads canonical weights); this test
initializes that table randomly so input-dependence is meaningful.

Checks:
1. determinism (bitwise) of the FP32 twin;
2. contract output shapes (logits [B,L,vocab], hidden [B,L,hidden]);
3. trailing-pad invariance: appending pad rows must not change real-position
   outputs (pads are arange-positioned and attention-masked; contributions
   are exact zeros, so equality is bitwise);
4. input sensitivity: changing a residue id (incl. to the mask token, whose
   embedding the token-dropout policy zeroes) changes that position's logits.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from tt.esm2.config import Esm2TTConfig  # noqa: E402
from tt.esm2.reference_layers import Esm2Model  # noqa: E402


def small_cfg() -> Esm2TTConfig:
    return Esm2TTConfig(
        num_hidden_layers=2, hidden_size=128, num_attention_heads=4,
        intermediate_size=256, vocab_size=33, layer_norm_eps=1e-5,
        pad_token_id=1, mask_token_id=32, max_position_embeddings=1026,
    )


def main() -> int:
    torch.manual_seed(0)
    cfg = small_cfg()
    model = Esm2Model(cfg).eval()
    model.embeddings.weight.normal_(0.0, 0.05)  # zero table -> random (see note)
    B, L = 2, 37
    ids = torch.randint(4, 24, (B, L))
    ids[:, 0] = 0   # cls
    ids[:, -1] = 2  # eos
    ids[0, 9:12] = cfg.pad_token_id
    ids[1, 20:24] = cfg.pad_token_id
    am = ids.ne(cfg.pad_token_id).long()

    with torch.no_grad():
        l1, h1 = model(ids, am)
        l2, h2 = model(ids, am)

    ok = True

    det = max((l1 - l2).abs().max().item(), (h1 - h2).abs().max().item())
    print(f"[determinism] max|delta| = {det:.3e}")
    ok &= det == 0.0

    shapes_ok = (tuple(l1.shape) == (B, L, cfg.vocab_size)
                 and tuple(h1.shape) == (B, L, cfg.hidden_size))
    print(f"[shapes      ] logits {tuple(l1.shape)} hidden {tuple(h1.shape)}")
    ok &= shapes_ok

    ids_p = torch.cat([ids, torch.full((B, 5), cfg.pad_token_id, dtype=ids.dtype)], dim=1)
    am_p = torch.cat([am, torch.zeros((B, 5), dtype=am.dtype)], dim=1)
    with torch.no_grad():
        l_p, h_p = model(ids_p, am_p)
    pad_delta = max((l1 - l_p[:, :L]).abs().max().item(),
                    (h1 - h_p[:, :L]).abs().max().item())
    print(f"[pad-invar   ] max|delta| real positions = {pad_delta:.3e}")
    ok &= pad_delta == 0.0

    ids_m = ids.clone()
    ids_m[0, 5] = cfg.mask_token_id
    with torch.no_grad():
        l_m, _ = model(ids_m, am)
    sens = (l_m[0, 5] - l1[0, 5]).abs().max().item()
    print(f"[mask-sens   ] max|delta| at masked pos = {sens:.3e}")
    ok &= sens > 0.0

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
