# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN chunk-prefill vs recurrent-decode CROSS-PATH consistency (the last unexamined systematic
prefill/decode difference). test_gdn_drift compared recurrent-vs-torch-recurrent; this compares the
CHUNK kernel (prefill) to the RECURRENT kernel (decode) on the SAME input sequence. A causal linear
attention must give identical per-position outputs either way. If chunk(prefill) and recurrent(decode)
diverge, every downstream layer is fed a diverged hidden state at decode -> compounds to the GSM8K drift.

Feeds ONE real GDN layer (layer 0) an S-token sequence:
  - prefill: forward_prefill(x[1,1,S,dim]) -> out_pf[S, dim] (chunk kernel, from scratch)
  - decode:  reset_state(); for pos: forward_decode(x[:,:,pos]) -> out_dec[pos] (recurrent, step-by-step)
Reports PCC(out_dec[pos], out_pf[pos]) binned by position.
Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_gdn_chunk_vs_rec.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_gdn_chunk_vs_rec(mesh_device):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    # n_layers=1 -> just layer 0, which is a GDN (linear-attn) layer.
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, n_layers=1)
    layer = model.layers[0]
    assert not layer.is_full_attention, "layer 0 should be GDN"
    gdn = layer.attention
    dim = model.args.dim
    S = 128
    torch.manual_seed(0)

    x = (torch.randn(1, 1, S, dim) * 1.0).to(torch.bfloat16).float()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    # Output is TP-fractured along dim=3 (the trailing all_reduce is over the size-1 mesh axis 0, a
    # no-op), so each device holds a 1280-wide hidden shard. Both paths share the SAME output tail →
    # comparing device-0's shard is a valid consistency check (covers 1/4 of the value heads).
    def host0(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    # ---- prefill (chunk kernel), from scratch ----
    gdn.reset_state()
    h = host0(gdn.forward_prefill(dev(x), chunk_size=128, valid_len=S, capture_state=False))
    W = h.shape[-1]  # per-device hidden shard width
    out_pf = h.reshape(-1, W)[:S]  # [S, W]

    # ---- recurrent (decode kernel), one token at a time from scratch ----
    gdn.reset_state()
    dec_rows = []
    for pos in range(S):
        xp = x[:, :, pos:pos + 1, :]  # [1,1,1,dim]
        o = gdn.forward_decode(dev(xp))
        dec_rows.append(host0(o).reshape(-1)[:W])
    out_dec = torch.stack(dec_rows, dim=0)  # [S, W]

    logger.info(f"GDNXPATH dim={dim} S={S} overall_PCC={_pcc(out_dec, out_pf):.5f}")
    for lo in range(0, S, 32):
        hi = min(lo + 32, S)
        rows = [_pcc(out_dec[p], out_pf[p]) for p in range(lo, hi)]
        logger.info(f"GDNXPATH pos[{lo}-{hi-1}] meanPCC={sum(rows)/len(rows):.5f} "
                    f"min={min(rows):.5f} first={rows[0]:.5f}")
