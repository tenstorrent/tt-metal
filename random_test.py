"""Reproduce and characterize the bf16 -> fp8_e4m3fn -> fp32 round-trip error.

Shows how absolute error grows with input magnitude while relative error stays
roughly constant at ~6.25% (one half-ULP / value), since fp8_e4m3fn has 3 mantissa
bits → max relative round-to-nearest error = 2^(-4) = 1/16.
"""

import torch

# Match dispatch test scale: dispatch_group_size=8, seq_len_per_chip=32, emb_dim=7168.
SHAPE = (8, 32, 7168)
SEED = 42


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return ((a * b).sum() / (torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum()))).item()


def fp8_bin_histogram(x: torch.Tensor) -> None:
    """Print how many elements land in each fp8_e4m3fn exponent bin.

    The 'max half-ULP' column is the worst-case round-to-nearest error any element
    in that bin can produce. The bin holding your observed max abs error is the bin
    that produced it.
    """
    a = x.abs().to(torch.float32)
    edges = [0.0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 448, float("inf")]
    print(f"{'bin':>16}  {'count':>10}  {'max half-ULP':>14}")
    print("-" * 46)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (a >= lo) & (a < hi)
        n = mask.sum().item()
        if n == 0:
            continue
        if hi <= 448:
            half_ulp_str = f"{hi / 16:.4f}"  # ULP at top of bin / 2
        else:
            half_ulp_str = "OVERFLOW->NaN"
        print(f"  [{lo:>6}, {hi:>6}) {n:>10}  {half_ulp_str:>14}")


def round_trip(x_bf16: torch.Tensor):
    x_ref = x_bf16.to(torch.float32)
    x_fp8 = x_bf16.to(torch.float8_e4m3fn).to(torch.float32)
    abs_err = (x_fp8 - x_ref).abs()

    # Relative error is only bounded by 1/16 in fp8's NORMAL range.
    # Smallest fp8_e4m3fn normal = 2^-6 ≈ 0.01562; below that, subnormal rounding
    # to zero blows up relative error. Filter to normals to see the real bound.
    fp8_smallest_normal = 2**-6
    in_normal_range = (x_ref.abs() > fp8_smallest_normal) & ~torch.isnan(x_fp8)
    if in_normal_range.any():
        rel_err = (abs_err[in_normal_range] / x_ref[in_normal_range].abs()).max().item()
    else:
        rel_err = float("nan")

    return {
        "max_val": x_ref.abs().max().item(),
        "max_abs_err": abs_err.max().item(),
        "max_rel_err": rel_err,
        "pcc": pcc(x_fp8, x_ref),
        "n_nan": torch.isnan(x_fp8).sum().item(),
    }


print(f"shape           : {SHAPE} ({torch.prod(torch.tensor(SHAPE)).item():,} elements)")
print(f"fp8_e4m3fn max  : {torch.finfo(torch.float8_e4m3fn).max}")
print()
print(f"{'scale':>8}  {'|max(x)|':>10}  {'max_abs_err':>12}  {'max_rel_err':>12}  {'PCC':>10}  {'NaN':>6}")
print("-" * 74)

for scale in [1.0, 2.0, 4.0, 8.0, 50.0, 100.0, 500.0]:
    torch.manual_seed(SEED)
    x_bf16 = (torch.randn(SHAPE, dtype=torch.bfloat16) * scale).to(torch.bfloat16)
    r = round_trip(x_bf16)
    print(
        f"{scale:>8.1f}  {r['max_val']:>10.4f}  {r['max_abs_err']:>12.6f}  "
        f"{r['max_rel_err']:>12.6f}  {r['pcc']:>10.6f}  {r['n_nan']:>6d}"
    )

print()
print("Notes:")
print("  - max_abs_err scales with magnitude (ULP doubles every exponent bin).")
print("  - max_rel_err stays bounded by 1/16 = 0.0625 (half-ULP / value).")
print("  - At scale=500, |x| exceeds fp8's ±448 range -> NaN appears, PCC collapses.")
print("  - At scale=1.0, allclose needs atol >= ~0.125; at scale=8.0, atol >= ~1.0.")
print()
print("=" * 60)
print("FP8 bin distribution for randn(0,1) (scale=1.0):")
print("=" * 60)
torch.manual_seed(SEED)
x_unscaled = torch.randn(SHAPE, dtype=torch.bfloat16)
fp8_bin_histogram(x_unscaled)
