"""Confirm the diagnosis: routed's ND-shard weight read is bfp4-only (hardcoded 576 B/tile),
while its INTERLEAVED path is dtype-correct. Then get the bfp8/bf16 accuracy comparison from
the path that works."""
import sys, torch, ttnn
import torch.nn.functional as F
import ttnn.operations.moe_fused_swiglu.moe_fused_swiglu  # noqa: F401

M = sys.modules["ttnn.operations.moe_fused_swiglu.moe_fused_swiglu"]
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

CAP, NG, NL, LID, GID, C = 5120, 256, 8, 3, 137, 512
GRID, RE_GRID_X, TILE = (11, 8), 11, 32
torch.set_num_threads(16)


def met(got, ref):
    g, r = got.flatten().double(), ref.flatten().double()
    e = g - r
    bad = (~torch.isfinite(g)).sum().item()
    if bad:
        return f"NONFINITE {bad}/{g.numel()} elements"
    return (
        f"PCC {torch.corrcoef(torch.stack([g,r]))[0,1].item():10.6f}  "
        f"relL2 {(e.norm()/r.norm()).item():8.5f}  mean_abs {e.abs().mean().item():8.5f}"
    )


device = ttnn.open_device(device_id=0)
try:

    def nd_mc(n_dim):
        per_core = (n_dim // TILE + RE_GRID_X - 1) // RE_GRID_X
        d = device.dram_grid_size()
        return ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, per_core * TILE]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(d.x - 1, d.y - 1))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    for EMB, HIDDEN in [(6144, 2048), (7168, 1024)]:
        torch.manual_seed(0)
        wg = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
        wu = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
        wd = torch.randn(HIDDEN, EMB, dtype=torch.bfloat16) * 0.02
        xt = torch.randn(CAP, EMB, dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(
            xt.reshape(1, 1, CAP, EMB),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_q = ttnn.to_torch(
            ttnn.from_torch(xt.reshape(1, 1, CAP, EMB), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        )[0, 0].float()[:C]
        ix = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
        ix[LID] = GID
        tt_idx = ttnn.from_torch(ix, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        cc = torch.zeros(NG, dtype=torch.int32)
        cc[GID] = C
        tt_c = ttnn.from_torch(cc, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        xs = xt[:C].float()
        ref_ideal = (F.silu(xs @ wg.float()) * (xs @ wu.float())) @ wd.float()

        for dn, dt in (("bfp4", ttnn.bfloat4_b), ("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)):
            print(
                f"\n##### K {EMB} · N {HIDDEN} · {dn} weights · M {C} " f"(tile {ttnn.tile_size(dt)} B) #####",
                flush=True,
            )
            dq = None
            for place in ("nd_shard", "interleaved"):
                if place == "nd_shard":
                    gu_f, dm_f = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
                    gu_r, dm_r = nd_mc(HIDDEN), nd_mc(EMB)
                else:
                    gu_f = dm_f = gu_r = dm_r = ttnn.DRAM_MEMORY_CONFIG
                Wf = [
                    ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                    for t, mc in ((wg, gu_f), (wu, gu_f), (wd, dm_f))
                ]
                Wr = [
                    ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                    for t, mc in ((wg, gu_r), (wu, gu_r), (wd, dm_r))
                ]
                if dq is None:
                    dq = [ttnn.to_torch(t).float() for t in Wf]
                    ref_q = (F.silu(x_q @ dq[0]) * (x_q @ dq[1])) @ dq[2]
                try:
                    o = M.moe_fused_swiglu(tt_x, *Wf, tt_c, tt_idx, LID, core_grid=GRID)
                    print(
                        f"  fused  {place:12s} vs IDEAL  {met(ttnn.to_torch(o)[0,0,:C].float(), ref_ideal)}", flush=True
                    )
                    print(f"  fused  {place:12s} vs QUANT  {met(ttnn.to_torch(o)[0,0,:C].float(), ref_q)}", flush=True)
                    ttnn.deallocate(o)
                except RuntimeError as e:
                    print(f"  fused  {place:12s} REFUSED: {str(e).splitlines()[0][:95]}", flush=True)
                ro = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, CAP, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
                )
                r = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
                    tt_x, *Wr, tt_c, tt_idx, LID, output=ro, x_is_row_major=True
                )
                print(f"  routed {place:12s} vs IDEAL  {met(ttnn.to_torch(r)[0,0,:C].float(), ref_ideal)}", flush=True)
                print(f"  routed {place:12s} vs QUANT  {met(ttnn.to_torch(r)[0,0,:C].float(), ref_q)}", flush=True)
                ttnn.deallocate(ro)
                for t in (*Wf, *Wr):
                    ttnn.deallocate(t)
        for t in (tt_x, tt_idx, tt_c):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
