import math
import torch, ttnn

DTYPE_TORCH = torch.float32
DTYPE_TTNN = ttnn.float32  # try ttnn.bfloat16 too
LAYOUT = ttnn.ROW_MAJOR_LAYOUT
ATOL, RTOL = 1e-5, 1e-5  # relax to (1e-3, 1e-2) for bf16 ?
PRINT_MISMATCHES = 40  # how many element-level diffs to print
TEST_SHAPES = [
    (1, 1, 32, 32),  # clean tile
    (1, 1, 16, 16),  # smallezr than tile (fix small one sas well div_up ?)
    (1, 1, 48, 48),  # multi-tile
    (1, 1, 35, 35),  # not a multiple of 32 (padding case).
]

TEST_SHAPES.extend(
    [
        (2, 1, 32, 32),  # batch > 1, single channel
        (1, 3, 32, 32),  # channel > 1, batch 1
        (2, 3, 32, 32),  # both > 1
        (4, 2, 17, 19),  # multi-N/C + weird H/W (PASS!), fuckign werid when i include this, passes indiviulat,
    ]
)

TEST_SHAPES.extend(
    [
        (1, 1, 2048, 1),
        (1, 1, 1, 2048),
        (1, 1, 3, 1025),  # crosses 1024 boundary in W
        (1, 1, 33, 64),  # weird, multi-row & multi-tile in both dims
    ]
)

TEST_SHAPES.extend(
    [
        (1, 1, 31, 31),
        (1, 1, 33, 33),
        (1, 1, 64, 64),
        (1, 1, 63, 63),
        (1, 1, 65, 65),
    ]
)


USE_ARANGE_INPUT = True  # deterministic values; set False for random , both
SEED = 0


def pretty_idx(idx):
    return f"(n={idx[0]}, c={idx[1]}, h={idx[2]}, w={idx[3]})"


def compare_and_report(tt_out, ref, label="add"):
    """
    tt_out: torch.Tensor from ttnn result (on cpu)
    ref:    torch.Tensor reference (torch add)
    """
    # Ensure contiguous CPU float32 for fair comparison
    tt = tt_out.to(torch.float32).contiguous()
    rf = ref.to(torch.float32).contiguous()

    if tt.shape != rf.shape:
        print(f"[{label}] SHAPE MISMATCH: TTNN {tt.shape} vs REF {rf.shape}")
        return

    diff = tt - rf
    absdiff = diff.abs()

    # Error mask with mixed abs/rel criterion
    thresh = ATOL + RTOL * rf.abs()
    bad = absdiff > thresh
    num_bad = bad.sum().item()
    total = bad.numel()

    print(f"\n=== [{label}] shape={list(tt.shape)} dtype=fp32 compare ===")
    print(f"max|diff| = {absdiff.max().item():.6g}")
    print(f"mean|diff|= {absdiff.mean().item():.6g}")
    print(f"RMSE      = {math.sqrt((diff.pow(2).mean().item())):.6g}")
    print(f"tolerances: ATOL={ATOL} RTOL={RTOL}")
    print(f"mismatches: {num_bad}/{total} " f"({100.0 * num_bad/total:.4f}%) using |a-b| > ATOL + RTOL*|b|")

    if num_bad == 0:
        print("All elements within tolerance!!")
        return 69

    # Per-row/col summary (over H,W)
    # Collapse N and C so we can see spatial structure
    # (works for N=C=1 as well)
    spatial_abs = absdiff.view(-1, *tt.shape[-2:])
    per_row_max = spatial_abs.amax(dim=[0, 2])  # H
    per_col_max = spatial_abs.amax(dim=[0, 1])  # W

    top_rows = torch.topk(per_row_max, k=min(8, per_row_max.numel())).indices.tolist()
    top_cols = torch.topk(per_col_max, k=min(8, per_col_max.numel())).indices.tolist()
    print("\nTop-rows by max|diff| (H indices):", top_rows)
    print("Top-cols by max|diff| (W indices):", top_cols)

    # First K mismatches with full indices & values
    print(f"\nFirst {min(PRINT_MISMATCHES, num_bad)} mismatches:")
    nz = torch.nonzero(bad, as_tuple=False)
    for i in range(min(PRINT_MISMATCHES, num_bad)):
        n, c, h, w = nz[i].tolist()
        print(
            f"  idx {pretty_idx((n,c,h,w))}: "
            f"TTNN={tt[n,c,h,w].item():.6g}  "
            f"REF={rf[n,c,h,w].item():.6g}  "
            f"diff={diff[n,c,h,w].item():.6g}"
        )

    return nz


def run_case(dev, shape):
    N, C, H, W = shape
    if USE_ARANGE_INPUT:
        base = torch.arange(1, N * C * H * W + 1, dtype=DTYPE_TORCH).reshape(N, C, H, W)
        a_t = base.clone()
        # b_t = a_t.clone() this not work better?
        b_t = torch.ones_like(a_t, dtype=torch.float32)  # base.clone()

    else:
        g = torch.Generator().manual_seed(SEED)
        a_t = torch.randn(shape, generator=g, dtype=DTYPE_TORCH)
        b_t = torch.randn(shape, generator=g, dtype=DTYPE_TORCH)

    print("shape: ", shape, "a_t pytorch: ", a_t)
    # Upload to device (Row-major layout requested)
    a = ttnn.from_torch(a_t, dtype=DTYPE_TTNN, layout=LAYOUT, device=dev)
    b = ttnn.from_torch(b_t, dtype=DTYPE_TTNN, layout=LAYOUT, device=dev)

    # Compute
    out = ttnn.add(a, b, use_legacy=None)

    # Bring back to Torch (CPU)
    out_t = ttnn.to_torch(out)
    ref_t = a_t + b_t

    # Compare both: full tensor and cropped "true" region.
    print("\n" + "-" * 80)
    print(f"Testing shape {shape} | Layout={LAYOUT} | TTNN dtype={DTYPE_TTNN}")
    x = compare_and_report(out_t, ref_t, label="full (raw)")

    # Optional: spot-check a small window for pretty printing
    # (useful when there are many diffs)
    h0, w0 = 0, 0
    hh, ww = min(8, H), min(8, W)
    print("\nSample window [0,0 : h<{}, w<{}] of TTNN vs REF:")
    # print("TTNN:\n", out_t[0, 0, h0 : h0 + hh, w0 : w0 + ww])
    # print("REF:\n", (a_t + b_t)[0, 0, h0 : h0 + hh, w0 : w0 + ww])
    print("TTNN:\n", out_t)
    print("REF:\n", (a_t + b_t))
    sol_t = a_t + b_t
    if type(x) == int:
        return x

    for bol in x:
        n, c, h, w = bol.tolist()

        print(bol)
        print(out_t[n, c, h, w])
        print(sol_t[n, c, h, w])

    return 1


def main():
    dev = ttnn.open_device(device_id=0)
    total = len(TEST_SHAPES)
    val = 0
    failed = []
    try:
        for shape in TEST_SHAPES:
            nz = run_case(dev, shape)
            #   ttnn.synchronize_device(dev)
            if nz == 69:
                val += 1
            else:
                failed.append(shape)

    #            break
    finally:
        ttnn.close_device(dev)

    print(val, "/", total, " Are correct!")
    print("Failed:", len(failed), "new again, no pading, sure")
    print(failed)


if __name__ == "__main__":
    main()

"""

"""
