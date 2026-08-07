"""Does routed's ~22% arithmetic-error advantage become VISIBLE end-to-end at wider weights?

At bfp4 the weight-quantization floor (relL2 0.111) swamps both ops' arithmetic error
(0.039-0.050), so end-to-end PCC was identical to 1.4e-4. Prediction: as the weight dtype
widens the floor drops below the arithmetic error and routed should pull ahead against
REF_IDEAL. Run every (shape, weight dtype) where FUSED FITS L1; skip the rest with the op's
own refusal. bfp4 is included at the same shapes as the control.
"""
import sys, torch, ttnn
import torch.nn.functional as F
import ttnn.operations.moe_fused_swiglu.moe_fused_swiglu  # noqa: F401

M = sys.modules["ttnn.operations.moe_fused_swiglu.moe_fused_swiglu"]
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

CAP, NG, NL, LID, GID = 5120, 256, 8, 3, 137
GRID, RE_GRID_X, TILE = (11, 8), 11, 32
SHAPES = [(6144, 2048), (7168, 1024)]
WDT = [("bfp4", ttnn.bfloat4_b), ("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)]
COUNTS = [512, 5120]
torch.set_num_threads(16)


def metrics(got, ref):
    g, r = got.flatten().double(), ref.flatten().double()
    err = g - r
    return {
        "pcc": torch.corrcoef(torch.stack([g, r]))[0, 1].item(),
        "relL2": (err.norm() / r.norm()).item(),
        "max_abs": err.abs().max().item(),
        "mean_abs": err.abs().mean().item(),
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

    for EMB, HIDDEN in SHAPES:
        torch.manual_seed(0)
        wg = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
        wu = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
        wd = torch.randn(HIDDEN, EMB, dtype=torch.bfloat16) * 0.02
        xt = torch.zeros(CAP, EMB, dtype=torch.bfloat16)
        xt[:CAP] = torch.randn(CAP, EMB, dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(
            xt.reshape(1, 1, CAP, EMB),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_q = ttnn.to_torch(
            ttnn.from_torch(xt.reshape(1, 1, CAP, EMB), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        )[0, 0].float()
        ix = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
        ix[LID] = GID
        tt_idx = ttnn.from_torch(ix, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ref_ideal = {
            c: (F.silu(xt[:c].float() @ wg.float()) * (xt[:c].float() @ wu.float())) @ wd.float() for c in COUNTS
        }

        for dt_name, dt in WDT:
            gu_f, dm_f = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
            Wf = [
                ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                for t, mc in ((wg, gu_f), (wu, gu_f), (wd, dm_f))
            ]
            Wr = [
                ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                for t, mc in (
                    (wg, nd_mc(HIDDEN, RE_GRID_X)),
                    (wu, nd_mc(HIDDEN, RE_GRID_X)),
                    (wd, nd_mc(EMB, RE_GRID_X)),
                )
            ]
            dq = [ttnn.to_torch(t).float() for t in Wf]
            wfloor = ((dq[0] - wg.float()).norm() / wg.float().norm()).item()
            print(
                f"\n########## K {EMB} · N {HIDDEN} · {dt_name} weights "
                f"(weight quant relL2 = {wfloor:.5f}) ##########",
                flush=True,
            )

            for c in COUNTS:
                cc = torch.zeros(NG, dtype=torch.int32)
                cc[GID] = c
                tt_c = ttnn.from_torch(cc, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                xq = x_q[:c]
                ref_q = (F.silu(xq @ dq[0]) * (xq @ dq[1])) @ dq[2]
                rows = []
                try:
                    o = M.moe_fused_swiglu(tt_x, *Wf, tt_c, tt_idx, LID, core_grid=GRID)
                    rows.append(("fused", ttnn.to_torch(o)[0, 0, :c].float()))
                    ttnn.deallocate(o)
                except RuntimeError as e:
                    msg = str(e).split(chr(10))[0][:110]
                    print(f"  M={c:5d}  fused  REFUSED: {msg}", flush=True)
                ro = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, CAP, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
                )
                r = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
                    tt_x, *Wr, tt_c, tt_idx, LID, output=ro, x_is_row_major=True
                )
                rows.append(("routed", ttnn.to_torch(r)[0, 0, :c].float()))
                ttnn.deallocate(ro)

                if rows:
                    print(f"  --- M = {c} ---", flush=True)
                    print(
                        f"  {'variant':10s} {'vs':10s} {'PCC':>10s} {'relL2':>9s} {'max_abs':>9s} {'mean_abs':>9s}",
                        flush=True,
                    )
                    for nm, arr in rows:
                        for rn, rr in (("REF_IDEAL", ref_ideal[c]), ("REF_QUANT", ref_q)):
                            m = metrics(arr, rr)
                            print(
                                f"  {nm:10s} {rn:10s} {m['pcc']:10.6f} {m['relL2']:9.5f} "
                                f"{m['max_abs']:9.4f} {m['mean_abs']:9.5f}",
                                flush=True,
                            )
                    if len(rows) == 2:
                        m = metrics(rows[0][1], rows[1][1])
                        print(
                            f"  {'f vs r':10s} {'each other':10s} {m['pcc']:10.6f} {m['relL2']:9.5f} "
                            f"{m['max_abs']:9.4f} {m['mean_abs']:9.5f}",
                            flush=True,
                        )
                ttnn.deallocate(tt_c)
            for t in (*Wf, *Wr):
                ttnn.deallocate(t)
        for t in (tt_x, tt_idx):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
