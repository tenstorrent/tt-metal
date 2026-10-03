# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer router calibration: the typical top-6 selection cutoff of ``score + correction_bias``.

The device gate ranks in bf16, so the bias is shifted by this constant (see tt/router.py). It only has to
be close: across tokens the cutoff varies by ~0.1 while the bias spans ~10.
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


@torch.no_grad()
def calibrate_gate_cutoff(layer_id: int, n_batches: int = 4, seq: int = 64, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    sh = _Shards()
    w = sh.get(f"layers.{layer_id}.ffn.gate.weight").float()
    b = sh.get(f"layers.{layer_id}.ffn.gate.bias").float()
    vals = []
    for _ in range(n_batches):
        tok = torch.randint(1000, 100000, (2, seq), generator=g)
        h, pm = R.embed_tokens(tok)
        blk(h, 0, pm, None)
        x = cap["x"].reshape(-1, w.size(1)).float()
        rank = torch.nn.functional.softplus(x @ w.T).sqrt() + b
        vals.append(rank.topk(7, -1).values[:, 5])
        blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)  # fresh caches for the next batch
        blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    return float(torch.cat(vals).mean())
