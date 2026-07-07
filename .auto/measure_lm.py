# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM planner (acestep-5Hz-lm-1.7B) PCC benchmark + batch>1 check.

Primary metric: lm_pcc = MIN over seq lengths of the TT-vs-HF last_hidden_state PCC (higher better).
The 28-layer Qwen3 LM has massive activations (absmax ~205) that bf16 mis-represents -> baseline
~0.58-0.78 PCC. The goal is to raise this toward the 0.97 gate WITHOUT overfitting (no per-seq tuning;
the metric is the MIN across seqs so a fix must generalize) and to add batch>1 support (batch_pcc:
the PCC of a batch-2 forward vs two batch-1 forwards, gating that batching is numerically correct).

Outputs METRIC lines. Compares against the genuine HF Qwen3Model reference (fp32 last_hidden_state).
"""
import sys
import torch
import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.tt.model_config import build_lm_planner

SEQS = [32, 64, 128]


def _rope(hf, seq, dev, batch=1):
    rope = Qwen3RotaryEmbedding(hf.config)
    pos = torch.arange(seq).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq, hf.config.hidden_size), pos)
    ct = ttnn.from_torch(cos.reshape(1, 1, seq, -1), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    st = ttnn.from_torch(sin.reshape(1, 1, seq, -1), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return ct, st


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        tt_lm, hf = build_lm_planner(dev)
        V = hf.config.vocab_size
        pccs = []
        for seq in SEQS:
            torch.manual_seed(seq)
            ids = torch.randint(0, V, (1, seq))
            with torch.no_grad():
                ref = hf(input_ids=ids).last_hidden_state
            ct, st = _rope(hf, seq, dev)
            out = ttnn.to_torch(tt_lm.forward(ids, ct, st)).float().reshape(ref.shape)
            p = comp_pcc(ref, out, 0.0)[1]
            pccs.append(p)
            print(f"LM seq={seq}: pcc={p:.4f} ref_absmax={float(ref.abs().max()):.1f}")
            print(f"METRIC lm_pcc_seq{seq}={p:.6f}")
        print(f"METRIC lm_pcc={min(pccs):.6f}")
        print(f"METRIC lm_pcc_mean={sum(pccs)/len(pccs):.6f}")

        # batch>1 correctness: batch-2 forward must equal two independent batch-1 forwards.
        batch_pcc = 1.0
        try:
            seq = 64
            torch.manual_seed(1)
            ids2 = torch.randint(0, V, (2, seq))
            ct, st = _rope(hf, seq, dev)
            outs_single = []
            for b in range(2):
                o = ttnn.to_torch(tt_lm.forward(ids2[b : b + 1], ct, st)).float().reshape(1, seq, -1)
                outs_single.append(o)
            ref_single = torch.cat(outs_single, dim=0)
            try:
                ob = ttnn.to_torch(tt_lm.forward(ids2, ct, st)).float().reshape(2, seq, -1)
                batch_pcc = comp_pcc(ref_single, ob, 0.0)[1]
                print(f"LM batch2 vs 2x batch1: pcc={batch_pcc:.4f}")
            except Exception as e:
                batch_pcc = 0.0
                print(f"LM batch2 UNSUPPORTED: {str(e)[:120]}")
        except Exception as e:
            batch_pcc = 0.0
            print(f"LM batch check ERR: {str(e)[:120]}")
        print(f"METRIC batch_pcc={batch_pcc:.6f}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    sys.exit(main())
