# Causal prefill SDPA for the DP4 x TP8 layout (8 q heads, 1 kv head per chip, full sequence) on one Blackhole chip:
# time and accuracy (vs torch fp32) over chunk sizes, fp32 accumulation and the approximate exp.
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp4_sdpa_sweep.py -s        env: SDPA_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

NQ, NKV, D = int(os.environ.get("SDPA_NQ") or 8), 1, 128  # SDPA_NQ: q heads per chip (8 in the model)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dp4_sdpa_sweep(device):
    grid = device.compute_with_storage_grid_size()
    for M in [int(m) for m in (os.environ.get("SDPA_M") or "4096,16384").split(",")]:
        torch.manual_seed(0)
        q = torch.randn(1, NQ, M, D)
        k = torch.randn(1, NKV, M, D)
        v = torch.randn(1, NKV, M, D)
        gold = None
        if M <= 4096:
            kk, vv = k.expand(1, NQ, M, D), v.expand(1, NQ, M, D)
            gold = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        tq = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tk = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tv = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tq8 = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        # SDPA_QS (q16,q8), SDPA_FP32 (1,0), SDPA_APPROX (0,1), SDPA_CHUNKS ("128x128,...")
        qsel = (os.environ.get("SDPA_QS") or "q16,q8").split(",")
        fsel = [x == "1" for x in (os.environ.get("SDPA_FP32") or "1,0").split(",")]
        asel = [x == "1" for x in (os.environ.get("SDPA_APPROX") or "0,1").split(",")]
        csel = [
            tuple(int(v) for v in c.split("x"))
            for c in (os.environ.get("SDPA_CHUNKS") or "128x128,128x256,256x256,128x512,256x512,512x256").split(",")
        ]
        # SDPA_FID (HiFi2): comma list of math fidelities
        fids = [getattr(ttnn.MathFidelity, f) for f in (os.environ.get("SDPA_FID") or "HiFi2").split(",")]
        for (qname, tqq), fid, fp32 in [
            (x, fd, f3) for x in (("q16", tq), ("q8", tq8)) if x[0] in qsel for fd in fids for f3 in fsel
        ]:
            if True:
                for approx in asel:
                    for qc, kc in csel:
                        ck = ttnn.init_device_compute_kernel_config(
                            device.arch(),
                            math_fidelity=fid,
                            math_approx_mode=approx,
                            fp32_dest_acc_en=fp32,
                            packer_l1_acc=True,
                            dst_full_sync_en=os.environ.get("SDPA_FULLSYNC") == "1",
                        )
                        cfg = ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=grid,
                            q_chunk_size=qc,
                            k_chunk_size=kc,
                            exp_approx_mode=approx,
                        )
                        f = lambda: ttnn.transformer.scaled_dot_product_attention(
                            tqq,
                            tk,
                            tv,
                            is_causal=True,
                            program_config=cfg,
                            compute_kernel_config=ck,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        try:
                            o = f()
                            ttnn.synchronize_device(device)
                            t0 = time.perf_counter()
                            for _ in range(5):
                                ttnn.deallocate(f())
                            ttnn.synchronize_device(device)
                            ms = (time.perf_counter() - t0) / 5 * 1e3
                            p = _pcc(gold, ttnn.to_torch(o)) if gold is not None else float("nan")
                            ttnn.deallocate(o)
                            logger.info(
                                f"SDPASWEEP M={M} {qname} {fid} fp32={fp32} approx={approx} q{qc} k{kc}: {ms:.3f} ms pcc {p:.5f}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"SDPASWEEP M={M} {qname} fp32={fp32} approx={approx} q{qc} k{kc}: ERROR {str(e)[:120]}"
                            )
