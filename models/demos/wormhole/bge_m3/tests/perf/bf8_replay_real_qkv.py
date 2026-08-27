# SPDX-License-Identifier: Apache-2.0
"""Replay REAL captured layer-23 Q/K/V through encoder_sdpa: bf8 vs bf16 score.

Loads the exact activations captured from a real full-model forward (via
BGE_CAPTURE_QKV_LAYER) and runs the actual op both ways, comparing bf8-score vs
bf16-score output directly. Real activations are what made the full model collapse
to 0.31; synthetic inputs gave bf8-vs-bf16 PCC 1.0. If bf8 diverges HERE, this is
the ground-truth localization of the failure (and lets us iterate a fix in ~10s
instead of the 200s full model).

Usage: python .../bf8_replay_real_qkv.py
"""
import numpy as np
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import build_encoder_sdpa_descriptor

DIR = "/localdev/gtobar/bge_optimization/tt-metal/.auto/qkv_capture"


def main():
    q_t = torch.from_numpy(np.load(f"{DIR}/q.npy"))
    k_t = torch.from_numpy(np.load(f"{DIR}/k.npy"))
    v_t = torch.from_numpy(np.load(f"{DIR}/v.npy"))
    print(f"loaded q={tuple(q_t.shape)} k={tuple(k_t.shape)} v={tuple(v_t.shape)}", flush=True)
    print(f"K stats: mean={k_t.mean():.4f} |max|={k_t.abs().max():.2f} per-chan-bias-max={k_t.mean(dim=2).abs().max():.3f}", flush=True)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(2, 1), trace_region_size=40_000_000)
    try:
        # The captured tensors are already the concatenated 2-shard global batch.
        # Re-shard across the mesh dim0 exactly as the model does.
        def to_dev(x, dt):
            return ttnn.from_torch(
                x.to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
            )

        q = to_dev(q_t, ttnn.bfloat8_b)
        k = to_dev(k_t, ttnn.bfloat4_b)
        v = to_dev(v_t, ttnn.bfloat8_b)
        local_b = q.shape[0]
        print(f"local per-device batch = {local_b}", flush=True)

        import os as _os
        variants = [
            ("bf16", dict(score_cb_bf8=False, fp32_dest_acc_en=False, k_chunk_size=2048)),
            ("bf8", dict(score_cb_bf8=True, fp32_dest_acc_en=False, k_chunk_size=2048)),
        ]
        if _os.environ.get("BGE_REPLAY_EXTRA", "0") == "1":
            variants += [
                ("bf8_fp32dest", dict(score_cb_bf8=True, fp32_dest_acc_en=True, k_chunk_size=2048)),
                ("bf8_k8192", dict(score_cb_bf8=True, fp32_dest_acc_en=False, k_chunk_size=8192)),
            ]
        outs = {}
        for label, kw in variants:
            cfg = EncoderSDPAConfig(batch=local_b, q_chunk_size=128, **kw)
            build = build_encoder_sdpa_descriptor(q, k, v, config=cfg)
            ttnn.generic_op(build.io_tensors, build.descriptor)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(build.output, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)).float()
            outs[label] = got
            ninf = torch.isinf(got).sum().item()
            print(f"  variant {label}: inf={ninf}", flush=True)

        for lab in ("bf16", "bf8"):
            o = outs[lab]
            print(f"  {lab}: nan={torch.isnan(o).sum().item()} inf={torch.isinf(o).sum().item()} "
                  f"min={o[~torch.isnan(o)&~torch.isinf(o)].min():.3f} max={o[~torch.isnan(o)&~torch.isinf(o)].max():.3f}", flush=True)
        # Localize the Inf: which (batch,head,q) positions overflow?
        o8 = outs["bf8"]
        bad = ~torch.isfinite(o8)  # [B,HQ,SQ,DH]
        bad_rows = bad.any(dim=-1)  # [B,HQ,SQ]
        idx = torch.nonzero(bad_rows)[:8]
        print(f"  Inf rows (b,h,q) first 8: {idx.tolist()}", flush=True)
        # For the offending positions, examine the real score row S=q.kh^T stats.
        g = q_t.shape[1] // k_t.shape[1]
        for (bb, hh, qq) in idx[:4].tolist():
            qrow = q_t[bb, hh, qq].float()
            krow = k_t[bb, hh // g].float()
            s = (qrow @ krow.transpose(0, 1)) / (q_t.shape[-1] ** 0.5)
            print(f"    (b{bb},h{hh},q{qq}): S min={s.min():.2f} max={s.max():.2f} "
                  f"|max|={s.abs().max():.2f} range={s.max()-s.min():.2f}", flush=True)
        # HYPOTHESIS TEST: if only the Inf positions are broken, clamping them to a
        # finite value should recover PCC vs bf16. If PCC recovers -> a HW clamp on
        # the overflow site is the fix. If not -> the whole output is corrupted.
        o8 = outs["bf8"].clone()
        o8_clamped = torch.nan_to_num(o8, nan=0.0, posinf=0.0, neginf=0.0)
        _, msg_clamp = comp_pcc(outs["bf16"], o8_clamped, 0.90)
        print(f"  bf8 with Inf->0 clamp: PCC vs bf16 = {msg_clamp}", flush=True)
        # also try clamp to the finite output range instead of 0
        finite_max = outs["bf16"].abs().max().item() * 2
        o8_sat = torch.clamp(torch.nan_to_num(o8, nan=0.0, posinf=finite_max, neginf=-finite_max), -finite_max, finite_max)
        _, msg_sat = comp_pcc(outs["bf16"], o8_sat, 0.90)
        print(f"  bf8 with Inf->+-{finite_max:.1f} sat: PCC vs bf16 = {msg_sat}", flush=True)
        _, msg = comp_pcc(outs["bf16"], outs["bf8"], 0.90)
        a = outs["bf16"].double().flatten()
        b = outs["bf8"].double().flatten()
        am, bm = a - a.mean(), b - b.mean()
        pcc_full = (torch.dot(am, bm) / (am.norm() * bm.norm())).item()
        rel = (a - b).norm().item() / (a.norm().item() + 1e-12)
        maxerr = (a - b).abs().max().item()
        print(f"RESULT bf8-vs-bf16 REAL layer-23: PCC={msg}", flush=True)
        print(f"  high-precision PCC={pcc_full:.8f}  rel-L2-err={rel:.6e}  max|abs-err|={maxerr:.4e}", flush=True)
        print(f"  1-PCC={1-pcc_full:.3e}  (compounds over 24 layers ~ {(pcc_full**24):.4f} if independent)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
