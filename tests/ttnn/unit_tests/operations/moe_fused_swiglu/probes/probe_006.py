"""Accuracy comparison: moe_fused_swiglu vs unified_routed_expert_ffn.

K 7168 / N 2048 / bfp4 weights / bf16 ROW_MAJOR x, nd_shard weights, 88 cores.

Both ops quantize x to bfp8 internally and both emit bfp8 TILE, so those are SHARED error
floors, not differences. Two references separate the terms:

  REF_IDEAL  fp32 math on the original bf16 weights   -> total error a caller sees
  REF_QUANT  fp32 math on the DEQUANTIZED bfp4 weights and the bfp8-quantized x
             -> what the device actually holds, so this isolates each op's own arithmetic
  FLOOR      REF_QUANT round-tripped through bfp8 TILE -> best any op could score given a
             bfp8 output; anything at the floor is limited by the output format, not the op
"""
import sys, torch, ttnn
import torch.nn.functional as F
import ttnn.operations.moe_fused_swiglu.moe_fused_swiglu  # noqa: F401

M = sys.modules["ttnn.operations.moe_fused_swiglu.moe_fused_swiglu"]
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

EMB, HIDDEN, CAP = 7168, 2048, 5120
NG, NL, LID, GID = 256, 8, 3, 137
GRID, RE_GRID_X, TILE = (11, 8), 11, 32
COUNTS = [32, 128, 512, 2048, 5120]
torch.set_num_threads(16)


def metrics(got, ref):
    g, r = got.flatten().double(), ref.flatten().double()
    pcc = torch.corrcoef(torch.stack([g, r]))[0, 1].item()
    err = g - r
    relL2 = (err.norm() / r.norm()).item()
    big = r.abs() > 0.1 * r.abs().mean()  # skip near-zeros for the rel-err stat
    return {
        "pcc": pcc,
        "relL2": relL2,
        "max_abs": err.abs().max().item(),
        "mean_abs": err.abs().mean().item(),
        "max_rel": (err[big].abs() / r[big].abs()).max().item(),
        "rms_ref": r.pow(2).mean().sqrt().item(),
    }


device = ttnn.open_device(device_id=0)
try:

    def nd_mc(n_dim, gx):
        per_core = (n_dim // TILE + gx - 1) // gx
        d = device.dram_grid_size()
        return ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, per_core * TILE]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(d.x - 1, d.y - 1))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    torch.manual_seed(0)
    wg = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
    wu = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(HIDDEN, EMB, dtype=torch.bfloat16) * 0.02
    xt = torch.zeros(CAP, EMB, dtype=torch.bfloat16)
    xt[: max(COUNTS)] = torch.randn(max(COUNTS), EMB, dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(
        xt.reshape(1, 1, CAP, EMB),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Weight tensors: identical values, per-op ND shard width.
    W = {}
    for tag, gx in (("fused", None), ("routed", RE_GRID_X)):
        if gx is None:
            gu, dm = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
        else:
            gu, dm = nd_mc(HIDDEN, gx), nd_mc(EMB, gx)
        W[tag] = [
            ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            for t, mc in ((wg, gu), (wu, gu), (wd, dm))
        ]

    # What the device ACTUALLY holds: dequantized bfp4 weights, and x as bfp8.
    dq = [ttnn.to_torch(t).float() for t in W["fused"]]
    x_q = ttnn.to_torch(
        ttnn.from_torch(xt.reshape(1, 1, CAP, EMB), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    )[0, 0].float()
    print(
        f"weight quantization bf16->bfp4: relL2 gate = " f"{((dq[0]-wg.float()).norm()/wg.float().norm()).item():.5f}",
        flush=True,
    )
    print(
        f"activation quantization bf16->bfp8: relL2 x = " f"{((x_q-xt.float()).norm()/xt.float().norm()).item():.5f}",
        flush=True,
    )

    c0 = torch.zeros(NG, dtype=torch.int32)
    ix = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
    ix[LID] = GID
    tt_idx = ttnn.from_torch(ix, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    cfg_noapprox = M.default_compute_kernel_config()
    cfg_noapprox.math_approx_mode = False

    for count in COUNTS:
        c = c0.clone()
        c[GID] = count
        tt_c = ttnn.from_torch(c, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        xs, xq = xt[:count].float(), x_q[:count]
        ref_ideal = (F.silu(xs @ wg.float()) * (xs @ wu.float())) @ wd.float()
        ref_quant = (F.silu(xq @ dq[0]) * (xq @ dq[1])) @ dq[2]
        floor = ttnn.to_torch(
            ttnn.from_torch(
                ref_quant.reshape(1, 1, count, EMB).bfloat16(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        )[0, 0].float()

        outs = {}
        o = M.moe_fused_swiglu(tt_x, *W["fused"], tt_c, tt_idx, LID, core_grid=GRID)
        outs["fused"] = ttnn.to_torch(o)[0, 0, :count].float()
        ttnn.deallocate(o)

        o = M.moe_fused_swiglu(tt_x, *W["fused"], tt_c, tt_idx, LID, core_grid=GRID, compute_kernel_config=cfg_noapprox)
        outs["fused(approx=off)"] = ttnn.to_torch(o)[0, 0, :count].float()
        ttnn.deallocate(o)

        ro = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, CAP, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        r = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
            tt_x, *W["routed"], tt_c, tt_idx, LID, output=ro, x_is_row_major=True
        )
        outs["routed"] = ttnn.to_torch(r)[0, 0, :count].float()
        ttnn.deallocate(ro)

        print(f"\n===== M = {count}  (ref rms {ref_ideal.pow(2).mean().sqrt().item():.4f}) =====", flush=True)
        hdr = f"{'variant':20s} {'vs':10s} {'PCC':>10s} {'relL2':>9s} {'max_abs':>9s} {'mean_abs':>9s} {'max_rel':>8s}"
        print(hdr, flush=True)
        for nm, arr in [("FLOOR bfp8(ref)", floor)] + list(outs.items()):
            for rname, rr in (("REF_IDEAL", ref_ideal), ("REF_QUANT", ref_quant)):
                m = metrics(arr, rr)
                print(
                    f"{nm:20s} {rname:10s} {m['pcc']:10.6f} {m['relL2']:9.5f} "
                    f"{m['max_abs']:9.4f} {m['mean_abs']:9.5f} {m['max_rel']:8.4f}",
                    flush=True,
                )
        m = metrics(outs["fused"], outs["routed"])
        print(
            f"{'fused vs routed':20s} {'each other':10s} {m['pcc']:10.6f} {m['relL2']:9.5f} "
            f"{m['max_abs']:9.4f} {m['mean_abs']:9.5f} {m['max_rel']:8.4f}",
            flush=True,
        )
        ttnn.deallocate(tt_c)
finally:
    ttnn.close_device(device)
