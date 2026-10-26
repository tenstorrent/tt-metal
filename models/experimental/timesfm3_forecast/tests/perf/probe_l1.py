"""Probe which M give a 2D QKV config over L1 (D4 cond. 3): one ttnn.linear per M, pass/fail plus the config."""
import json, math, sys
import torch, ttnn
from models.experimental.timesfm3_forecast.tt import program_configs as pc
from models.experimental.timesfm3_forecast.tt.common import TtPrecision

dev = ttnn.open_device(device_id=0)
try:

    from models.experimental.timesfm3_forecast.tt.common import TtPrecision as P
    prec = P.accuracy()
    kcfg = prec.matmul_config()
    K, N = 1536, 4608
    w = ttnn.from_torch(torch.randn(K, N) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out = []
    for mt in [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(16, 2049, 8):
        x = ttnn.from_torch(torch.randn(1, 1, mt * 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = pc.matmul_2d_config((11, 10), mt, K // 32, N // 32, out_block_cap=8)
        rec = dict(m_tiles=mt, out_block_h=cfg.out_block_h, per_core_M=cfg.per_core_M, per_core_N=cfg.per_core_N, in0_block_w=cfg.in0_block_w, sub=(cfg.out_subblock_h, cfg.out_subblock_w))
        try:
            y = ttnn.linear(x, w, program_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kcfg)
            ttnn.deallocate(y); rec["ok"] = True
        except RuntimeError as e:
            rec["ok"] = False; rec["err"] = str(e)[str(e).find("grow to"):][:60]
        ttnn.deallocate(x)
        out.append(rec)
    json.dump(out, open("/tmp/timesfm3_probe_l1.json", "w"))
    print("FAILS", [r for r in out if not r["ok"]])
    print("OKS_SAMPLE", [ (r["m_tiles"], r["out_block_h"], r["per_core_M"], r["in0_block_w"]) for r in out if r["ok"]][:5])
finally:
    ttnn.close_device(dev)
