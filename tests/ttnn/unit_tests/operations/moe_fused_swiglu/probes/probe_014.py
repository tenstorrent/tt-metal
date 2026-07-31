"""Refinement 1b: the UNBEATABLE format ceiling vs the acceptance gate 0.98,
on the acceptance test's exact fixture, for all 5 SHAPES x both formats.

floor_w  = torch fp32 chain, ONLY the bfp4_b weight quantization (what helpers.py
           calls the number "no correct implementation can beat")
floor_op = the same chain PLUS the two format steps the op's OWN CONTRACT forces:
           `h` requantized to bfp8_b and the output bfp8_b. This is the real
           ceiling for THIS op, since its signature pins bfloat8_b TILE output.
"""
import torch, ttnn
import tests.ttnn.unit_tests.operations.moe_fused_swiglu.test_moe_fused_swiglu as T
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rt(t, dt):
    tt = ttnn.from_torch(
        t.to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.to_torch(tt).to(torch.float32)


GATE = 0.98
print(
    f"\n{'fmt':>9} {'emb':>5} {'cap':>5} {'cnt':>5} | {'floor_w':>9} {'floor_op':>9} |"
    f" {'device':>9} | {'dev-vs-floor_op':>15} | gate 0.98 vs ceiling"
)
rows = []
for fmt in ("bf16_rm", "bfp8_tile"):
    for emb, capacity, count in T.SHAPES:
        x_rows, (wg, wu, wd), tt_x, tt_w, tt_c, tt_i = T._build_inputs(emb, capacity, count, fmt, device)
        ref = T._reference(x_rows, wg, wu, wd)
        xr = x_rows.to(torch.float32)
        g4, u4, d4 = rt(wg, ttnn.bfloat4_b), rt(wu, ttnn.bfloat4_b), rt(wd, ttnn.bfloat4_b)
        h = torch.nn.functional.silu(xr @ g4) * (xr @ u4)
        f_w = pcc(ref, h @ d4)
        f_op = pcc(ref, rt(rt(h, ttnn.bfloat8_b) @ d4, ttnn.bfloat8_b))
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_c, tt_i, T.LOCAL_EXPERT_ID)
        dev = pcc(ref, ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32))
        verdict = "UNREACHABLE" if f_op < GATE else "reachable"
        print(
            f"{fmt:>9} {emb:>5} {capacity:>5} {count:>5} | {f_w:>9.5f} {f_op:>9.5f} |"
            f" {dev:>9.5f} | {f_op-dev:>15.2e} | {verdict}"
        )
        rows.append((f_w, f_op, dev))

print(f"\nfloor_w  range: {min(r[0] for r in rows):.5f} .. {max(r[0] for r in rows):.5f}")
print(f"floor_op range: {min(r[1] for r in rows):.5f} .. {max(r[1] for r in rows):.5f}")
print(f"device   range: {min(r[2] for r in rows):.5f} .. {max(r[2] for r in rows):.5f}")
print(
    f"\ncells whose OP CEILING is below the 0.98 acceptance gate: "
    f"{sum(1 for r in rows if r[1] < GATE)} / {len(rows)}"
)
print(f"max kernel-attributable dpcc (ceiling - device): {max(r[1]-r[2] for r in rows):.2e}")
print(
    f"golden suite gate is now 0.975 -> device clears it by " f"{min(r[2] for r in rows) - 0.975:.2e} on the worst cell"
)
