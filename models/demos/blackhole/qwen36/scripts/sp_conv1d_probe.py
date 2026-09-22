# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Depthwise conv1d (K=4, HEIGHT_SHARDED) correctness + timing probe: fused-SiLU vs a
separate ttnn.silu, PCC/max-abs-err against a torch reference, at a given T. Backs the
coalescing-wall and fused-SiLU-numerics findings in SP_PREFILL_HANDOFF.md sec 5.

Run: env per SP_PREFILL_HANDOFF.md sec 6, then
  python sp_conv1d_probe.py --T 1024 --fused 1
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

import ttnn

K = 4
CW = 3072


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--packer_l1_acc", type=int, default=1)
    ap.add_argument("--fp32_dest_acc_en", type=int, default=1)
    ap.add_argument("--act_block_h_override", type=int, default=0)
    ap.add_argument("--dump_errors", type=int, default=0)
    ap.add_argument("--timing", type=int, default=1)
    args = ap.parse_args()

    T = args.T
    LIN = (K - 1) + T
    cw = CW
    fused = bool(args.fused)

    torch.manual_seed(0)
    x = (torch.randn(1, LIN, cw) * 0.1).to(torch.bfloat16).float()
    w = (torch.randn(cw, 1, K) * 0.1).to(torch.bfloat16).float()
    ref = F.conv1d(x.transpose(1, 2), w, groups=cw).transpose(1, 2)  # [1,T,cw]
    if fused:
        ref = F.silu(ref)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=32 * 1024 * 1024)
    result = {
        "T": T,
        "cw": cw,
        "fused": fused,
        "packer_l1_acc": bool(args.packer_l1_acc),
        "fp32_dest_acc_en": bool(args.fp32_dest_acc_en),
        "act_block_h_override": args.act_block_h_override,
    }
    try:
        _dram = ttnn.DRAM_MEMORY_CONFIG
        xin = ttnn.from_torch(
            x.reshape(1, LIN, 1, cw),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=_dram,
        )
        w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=bool(args.fp32_dest_acc_en),
            packer_l1_acc=bool(args.packer_l1_acc),
        )

        kw = dict(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        if fused:
            kw["activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        if args.act_block_h_override:
            kw["act_block_h_override"] = args.act_block_h_override
        conv_cfg = ttnn.Conv1dConfig(**kw)

        wprep = ttnn.prepare_conv_weights(
            weight_tensor=w_tt,
            input_memory_config=_dram,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=cw,
            out_channels=cw,
            batch_size=1,
            input_height=1,
            input_width=LIN,
            kernel_size=(1, K),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=False,
            groups=cw,
            device=dev,
            input_dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=cc,
        )

        def run_once():
            return ttnn.conv1d(
                input_tensor=xin,
                weight_tensor=wprep,
                device=dev,
                in_channels=cw,
                out_channels=cw,
                batch_size=1,
                input_length=LIN,
                kernel_size=K,
                stride=1,
                padding=0,
                dilation=1,
                groups=cw,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )

        out = run_once()
        out_i = ttnn.reshape(ttnn.sharded_to_interleaved(out, _dram), (1, T, cw))
        out_torch = ttnn.to_torch(out_i).float()
        p = pcc(out_torch, ref)

        err = (out_torch - ref).abs()  # [1, T, cw]
        max_abs_err = float(err.max())
        frac_gt = float((err > 0.05).float().mean())

        result.update(status="ok", pcc=p, max_abs_err=max_abs_err, frac_gt_0p05=frac_gt)

        if args.dump_errors:
            err2 = err[0]  # [T, cw]
            row_max = err2.max(dim=1).values  # [T]
            chan_max = err2.max(dim=0).values  # [cw]
            # top offending rows / channels
            n_top = 20
            top_rows = torch.topk(row_max, min(n_top, T))
            top_chans = torch.topk(chan_max, min(n_top, cw))
            bad_row_mask = row_max > 0.05
            bad_chan_mask = chan_max > 0.05
            result["err_dump"] = {
                "n_bad_rows": int(bad_row_mask.sum()),
                "n_bad_chans": int(bad_chan_mask.sum()),
                "bad_row_idx_sample": bad_row_mask.nonzero().flatten()[:40].tolist(),
                "top_row_idx": top_rows.indices.tolist(),
                "top_row_val": [round(v, 4) for v in top_rows.values.tolist()],
                "top_chan_idx": top_chans.indices.tolist(),
                "top_chan_val": [round(v, 4) for v in top_chans.values.tolist()],
            }

        ttnn.deallocate(out)
        ttnn.deallocate(out_i)

        us = None
        method = None
        tid = None
        if args.timing:
            try:
                ttnn.synchronize_device(dev)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                traced_out = run_once()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.synchronize_device(dev)
                for _ in range(2):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                us = sorted(times)[len(times) // 2] * 1e6
                method = "trace_replay5_median"
                ttnn.release_trace(dev, tid)
                tid = None
                ttnn.deallocate(traced_out)
            except Exception as te:
                tmsg = str(te).splitlines()
                tmsg = tmsg[-1] if tmsg else repr(te)
                if tid is not None:
                    try:
                        ttnn.release_trace(dev, tid)
                    except Exception:
                        pass
                method = f"timing skipped: {tmsg[:80]}"

        result.update(us=us, method=method)
    except Exception as e:
        msg = str(e).strip().splitlines()
        msg = msg[-1] if msg else repr(e)
        result.update(status="error", error=msg[:300])

    print("RESULT_JSON " + json.dumps(result), flush=True)

    try:
        ttnn.close_mesh_device(dev)
    except Exception:
        pass


if __name__ == "__main__":
    main()
