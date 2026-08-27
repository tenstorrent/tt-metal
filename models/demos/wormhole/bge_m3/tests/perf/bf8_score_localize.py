# SPDX-License-Identifier: Apache-2.0
"""Localize the bf8-score encoder-SDPA PCC collapse using REALISTIC activation
statistics (NOT peaked random, which hides the failure).

Prior probes used torch.randn Q/K/V -> peaked softmax -> bf8 looks fine (PCC~1).
The full model fails at 0.31 only with real, flatter-softmax activations. This
harness synthesizes Q/K/V whose QK^T score distribution matches the real deep
layers (small |S|, K channel-bias), runs the ACTUAL encoder_sdpa op on device
with bf8 vs bf16 score, and compares BOTH against an fp32 torch flash reference.

If bf16-score matches fp32 but bf8-score collapses -> the op's bf8 CB_QK path is
the true culprit and we localize WHERE (score vs prob vs reduction). If bf8-score
also matches fp32 here -> the collapse needs the exact real tensors (capture path).

Usage: python .../bf8_score_localize.py
"""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import build_encoder_sdpa_descriptor

B, HQ, HKV, SQ, SK, DH = 6, 32, 16, 4096, 8192, 64


def torch_flash_ref(q, k, v):
    # q:[B,HQ,SQ,DH] k,v:[B,HKV,SK,DH]; HQ heads map to HKV via head-fold groups.
    # Here HQ=32 is head-folded (16 heads x 2 seq-chunks). For the reference we
    # just do full attention per the folded layout the op expects: treat each of
    # HQ as attending to its corresponding HKV head. The op internally remaps;
    # for a reference we replicate stock behavior: group = HQ//HKV.
    g = HQ // HKV
    out = torch.empty(B, HQ, SQ, DH, dtype=torch.float32)
    scale = 1.0 / (DH**0.5)
    for h in range(HQ):
        kh = k[:, h // g]
        vh = v[:, h // g]
        s = torch.matmul(q[:, h].float(), kh.float().transpose(-1, -2)) * scale
        p = torch.softmax(s, dim=-1)
        out[:, h] = torch.matmul(p, vh.float())
    return out


def main():
    torch.manual_seed(23)
    # Realistic deep-layer stats: modest Q/K magnitude + shared K channel bias.
    q_t = torch.randn(B, HQ, SQ, DH) * 0.9
    k_t = torch.randn(B, HKV, SK, DH) * 0.9 + torch.randn(1, 1, 1, DH) * 4.0
    v_t = torch.randn(B, HKV, SK, DH) * 0.9

    ref = torch_flash_ref(q_t, k_t, v_t)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(2, 1), trace_region_size=40_000_000)
    try:

        def to_dev(x, dt):
            return ttnn.from_torch(
                x.to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        q = to_dev(q_t, ttnn.bfloat8_b)
        k = to_dev(k_t, ttnn.bfloat4_b)
        v = to_dev(v_t, ttnn.bfloat8_b)

        outs = {}
        for label, bf8 in (("bf16-score", False), ("bf8-score", True)):
            cfg = EncoderSDPAConfig(
                q_chunk_size=128, k_chunk_size=2048, fp32_dest_acc_en=False, score_cb_bf8=bf8
            )
            build = build_encoder_sdpa_descriptor(q, k, v, config=cfg)
            ttnn.generic_op(build.io_tensors, build.descriptor)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(build.output, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B].float()
            outs[label] = got
            _, msg = comp_pcc(ref, got, 0.90)
            print(f"RESULT {label}: PCC vs (broken) fp32-torch = {msg}", flush=True)
        # The decisive metric: bf8-score vs bf16-score (same op, same inputs).
        # If this is ~1.0, bf8 is harmless on these inputs (need real activations).
        # If it collapses, bf8 CB_QK is intrinsically lossy -> localizes the fix.
        _, msg = comp_pcc(outs["bf16-score"], outs["bf8-score"], 0.90)
        print(f"RESULT bf8-vs-bf16 (decisive): PCC = {msg}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
