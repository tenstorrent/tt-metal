# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU only. NOISE FLOOR of the decode indexer: attention-output PCC / top-512 agreement / attention-mass coverage between the reference with its fp4-simulated
indexer q (V0) and a reference variant whose indexer arithmetic differs only by skipping that q fp4 simulation (V1), on the SAME cache.

    python -m models.demos.blackhole.deepseek_v41_flash.tests.idx_noise_floor --layer 2 --S 2048 [--real DIR]"""

import argparse

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.idx_state import Capture, make_state


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--real", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    ref_kernels.FAKE_QUANT = True
    mod = R.load_model_module()
    blk = R.build_layer(a.layer, max_batch_size=1, max_seq_len=a.S + 64)
    x = make_state(blk, a.S, a.real, a.layer)
    cap = Capture(mod)
    pos = a.S - 1
    o0 = blk.attn(x, pos).float().reshape(-1)
    ids0, p0 = cap.ids.clone(), cap.p_comp.clone()
    orig_q = mod.fp4_act_quant

    def skip_q(t, *args, **kw):  # the indexer's q is the only 4-D (b, s, heads, d) tensor quantised
        return t if t.dim() == 4 else orig_q(t, *args, **kw)

    mod.fp4_act_quant = skip_q
    o1 = blk.attn(x, pos).float().reshape(-1)
    mod.fp4_act_quant = orig_q
    ids1 = cap.ids.clone()
    agree = len(set(ids0.tolist()) & set(ids1.tolist())) / len(ids0)
    # dense-mass coverage under V0's probabilities (cap.p_comp holds the V1 call's; recompute by restoring p0)
    cap.p_comp = p0
    print(
        f"NOISE layer {a.layer} S={a.S} {'REAL' if a.real else 'synthetic'}: attention-output PCC V0 (fp4 q) vs V1 (no q fp4): {R.pcc(o0, o1):.5f}; "
        f"top-512 set agreement {agree:.4f}; attention-mass coverage of V0 set {cap.coverage(ids0):.4f}, of V1 set {cap.coverage(ids1):.4f}; "
        f"total (window+selected) coverage V1 {cap.total_coverage(ids1):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
