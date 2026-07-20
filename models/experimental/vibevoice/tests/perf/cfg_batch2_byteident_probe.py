"""CFG batch-2 BYTE-IDENTITY probe (ISOLATED — touches no production files).

Decides whether a B=2 LM decode forward can be made BIT-IDENTICAL (per row) to two separate B=1
forwards — which determines the acceptance tier for CFG batch-2 fusion:
  maxabsdiff == 0  on every matmul  => byte-identical => Tier-0 (long-form-safe by construction)
  any nonzero                       => math-changing  => Tier-2 (must pass full 100-min render)

The LM lays the batch in dim 0 ([B,1,1,H]); auto matmuls SHOULD treat each batch row independently
(bit-identical), but the width-sharded wq/wo progcfg (_QO_DECODE_PROGCFG: fuse_batch=True,
per_core_M=1) is B=1-only.  This probe measures, at VibeVoice's exact decode dims + real configs,
row-0 of a B=2 matmul vs the B=1 matmul, for every LM decode matmul shape.

Run: python models/experimental/vibevoice/tests/perf/cfg_batch2_byteident_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

ROOT = "/home/ubuntu/vibe-voice/tt-metal"
sys.path.insert(0, ROOT)
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    _QO_DECODE_PROGCFG,
    _QO_DECODE_OUT_MEMCFG,
    _HIFI4,
)

H, N_HEADS, N_KV, HD, FFN = 1536, 12, 2, 128, 8960
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
torch.manual_seed(0)


def tt(a, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.as_tensor(a, device=dev, dtype=dt, layout=lay, memory_config=mc)


def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.abs(a - b).max())


# A B=2 variant of _QO_DECODE_PROGCFG: identical in0_block_w / mcast / subblock, per_core_M=2 so it
# is valid for M=2 (2 batch rows folded into M).  If the K-reduction (in0_block_w) is preserved,
# per-row output should be bit-identical to the B=1 per_core_M=1 config.
_QO_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)


def run_matmul(x_t, w_t, progcfg, out_mc):
    x = tt(x_t)
    w = tt(w_t)
    y = ttnn.linear(x, w, compute_kernel_config=_HIFI4, program_config=progcfg, memory_config=out_mc)
    out = ttnn.to_torch(y).float()
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    ttnn.deallocate(y)
    return out


def test(name, K, Nout, progcfg_b1, out_mc_b1, progcfg_b2, out_mc_b2):
    w_t = torch.randn(1, 1, K, Nout, dtype=torch.bfloat16)
    x1 = torch.randn(1, 1, 1, K, dtype=torch.bfloat16)
    x2r0 = x1.clone()
    x2r1 = torch.randn(1, 1, 1, K, dtype=torch.bfloat16)
    x2 = torch.cat([x2r0, x2r1], dim=0)  # [2,1,1,K], row0 == the B=1 input
    y1 = run_matmul(x1, w_t, progcfg_b1, out_mc_b1)  # [1,1,1,Nout]
    y2 = run_matmul(x2, w_t, progcfg_b2, out_mc_b2)  # [2,1,1,Nout]
    d = maxdiff(y1[0, 0, 0], y2[0, 0, 0])  # row0 B=1 vs B=2
    print(f"  {name:22s} K={K:5d} N={Nout:6d}  row0 maxabsdiff={d:.6e}  {'IDENTICAL' if d == 0 else 'DIFFERS'}")
    return d


D = ttnn.DRAM_MEMORY_CONFIG
print("=== B=1 vs B=2 per-row byte-identity (row0), real LM decode configs ===")
res = {}
# wq/wo 1536x1536 — custom width-sharded progcfg (B1) vs per_core_M=2 variant (B2)
res["wq_wo_customcfg"] = test(
    "wq/wo custom->B2cfg", H, N_HEADS * HD, _QO_DECODE_PROGCFG, _QO_DECODE_OUT_MEMCFG, _QO_B2, _QO_DECODE_OUT_MEMCFG
)
# same 1536x1536 but AUTO for both (fallback option if custom-cfg batching differs)
res["wq_wo_auto"] = test("wq/wo auto both", H, N_HEADS * HD, None, D, None, D)
# wk/wv 1536x256 auto
res["wk_wv_auto"] = test("wk/wv auto", H, N_KV * HD, None, D, None, D)
# FFN gate/up 1536x8960 auto
res["ffn_gate_auto"] = test("ffn gate/up auto", H, FFN, None, D, None, D)
# FFN down 8960x1536 auto
res["ffn_down_auto"] = test("ffn down auto", FFN, H, None, D, None, D)

print("\n=== SUMMARY ===")
allid = all(v == 0 for v in res.values())
for k, v in res.items():
    print(f"  {k:22s} {'IDENTICAL' if v == 0 else f'DIFFERS ({v:.2e})'}")
print(
    f"\n  auto matmuls byte-identical B1==B2 row0? "
    f"{all(res[k] == 0 for k in ['wk_wv_auto','ffn_gate_auto','ffn_down_auto','wq_wo_auto'])}"
)
print(f"  wq/wo custom-cfg -> per_core_M=2 byte-identical? {res['wq_wo_customcfg'] == 0}")
print(f"  ==> CFG batch-2 can be Tier-0 (all identical): {allid}")
ttnn.close_mesh_device(dev)
