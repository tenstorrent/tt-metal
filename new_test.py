import os, torch, ttnn
import math

# === CONFIGURATION ===
DTYPE_TORCH = torch.float32
DTYPE_TTNN = ttnn.float32
LAYOUT = ttnn.TILE
ATOL, RTOL = 1e-5, 1e-5
PRINT_MISMATCHES = 40
USE_ARANGE_INPUT = True
SEED = 0
EARLY_EXIT_ON_FAIL = True

TEST_SHAPES = [
    (1, 1, 32, 32),  # clean tile
    (1, 1, 16, 16),  # smaller than tile
    (1, 1, 48, 48),  # multi-tile (Fail for scal only ? numrow ? , curent shit ? )
    # # we nee dthe othe rbcast, dimeiosn info ? realy man fuck , yamb maybe es
    (1, 1, 35, 35),  # not multiple of 32 (padding)
    (5, 1, 32, 32),  # batch >1
    (1, 3, 32, 32),  # channel >1
    (2, 3, 32, 32),  # both >1
    (4, 2, 17, 19),  # multi N/C + weird H/W . (FAIL)
    (1, 1, 2048, 1),  # tall skinny
    (1, 1, 1, 2048),  # wide skinny
    (1, 1, 3, 1025),  # crosses 1024 in W
    (1, 1, 33, 64),  # multi-row, multi-tile
    (1, 1, 31, 31),
    (1, 1, 33, 33),
    (1, 1, 64, 64),
    (1, 1, 63, 63),
    (1, 1, 65, 65),
    (1, 1, 1, 1),  # tiny (this stuck? )
    (
        1,
        1,
        1024,  # this one curent is stuck for scal abcast man
        1024,
    ),  # large square (stuck ? this one ? ). Need to try diff version of thi where w is > 1024 abd reakly huge tensors
    # very gua dn veyr weird ones man
    (1, 1, 2048, 2048),  # very large (memory stress) (this one ? )
    (1, 1, 3, 2048),  # skewed small H large W
    (1, 1, 2048, 3),  # skewed large H small W
    (2, 2, 1024, 1024),  # multi batch/channel large (STUCK man ?gemini)
    (1, 1, 4096, 1),  # extreme tall
    (1, 1, 1, 4096),
    # is treated as row major, when doign scalr  # extreme wide (scalar stuck ? ), is this row bcast or scalr bcast ?
    # (1,1,2045,2045)
    # (3,4,2049,2043)
    #  (3333,3333,3333,3333) # would this wrok ,haha hsoudl work, HAHA
    # (3333,3333,3333,3333,3333) would this wrok ,haha hsoudl work
    # we also need very big ones, which fill al the cores in use and then loop aroudn
    # and nee dot set aht modul o properly as well
]

# # Examples from test cases:
# [[1, 1, 320, 320], [1, 1, 1, 320]]  # Single batch/channel
# [[1, 4, 320, 320], [1, 1, 1, 320]]  # Multiple channels
# [[4, 4, 320, 320], [1, 1, 1, 320]]  # Multiple batches and channels

# For fallback, filter shapes where H>1 and W>1
TEST_SHAPES_FALLBACK = [s for s in TEST_SHAPES if s[2] > 1 and s[3] > 1]


def pretty_idx(idx):
    return f"(n={idx[0]}, c={idx[1]}, h={idx[2]}, w={idx[3]})"


def compare_and_report(tt_out, ref, label="add", out_tt=None, layout_expected_rowmajor=True):
    """
    tt_out: torch.Tensor from ttnn result (on cpu)
    ref:    torch.Tensor reference (torch add)
    out_tt: ttnn.Tensor for layout check
    """
    # Ensure contiguous CPU float32 for fair comparison
    tt = tt_out.to(torch.float32).contiguous()
    rf = ref.to(torch.float32).contiguous()

    if tt.shape != rf.shape:
        print(f"[{label}] SHAPE MISMATCH: TTNN {tt.shape} vs REF {rf.shape}")
        return None

    diff = tt - rf
    absdiff = diff.abs()

    # Error mask with mixed abs/rel criterion
    thresh = ATOL + RTOL * rf.abs()
    bad = absdiff > thresh
    num_bad = bad.sum().item()
    total = bad.numel()

    print(f"\n=== [{label}] shape={list(tt.shape)} dtype=fp32 compare ===")
    if out_tt is not None:
        print(" out_tt.layout =", out_tt.layout)
        if layout_expected_rowmajor:
            print(" EXPECT row-major fast path -> layout should be ROW_MAJOR_LAYOUT")
        else:
            print(" EXPECT fallback allowed -> layout may NOT be ROW_MAJOR_LAYOUT")
    print(f"max|diff| = {absdiff.max().item():.6g}")
    print(f"mean|diff|= {absdiff.mean().item():.6g}")
    print(f"RMSE      = {math.sqrt((diff.pow(2).mean().item())):.6g}")
    print(f"tolerances: ATOL={ATOL} RTOL={RTOL}")
    print(f"mismatches: {num_bad}/{total} " f"({100.0 * num_bad/total:.4f}%) using |a-b| > ATOL + RTOL*|b|")

    if num_bad == 0:
        print("All elements within tolerance!!")
        return 69

    # Per-row/col summary (over H,W)
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
    print(tt)
    print(rf)

    return nz


def _make(dev, shape, dtype=DTYPE_TORCH, start=1):
    N, C, H, W = shape
    numel = N * C * H * W
    if numel == 0:
        base = torch.empty(shape, dtype=dtype)
    else:
        base = torch.arange(start, numel + start, dtype=dtype).reshape(shape)
    tt = ttnn.from_torch(base, dtype=DTYPE_TTNN, layout=LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return base, tt


def run_case(dev, a_shape, b_shape, label, layout_expected_rowmajor=True, a_start=1, b_start=100000):
    print("\n" + "-" * 80)
    print(f"Starting test: {label} | A shape={a_shape} B shape={b_shape} | Layout={LAYOUT} | TTNN dtype={DTYPE_TTNN}")

    a_pt, a_tt = _make(dev, a_shape, start=a_start)
    b_pt, b_tt = _make(dev, b_shape, start=b_start)

    print("Pytorhc tensors: ")
    print(a_pt)
    print(b_pt)

    out_tt = ttnn.add(a_tt, b_tt, use_legacy=None)
    out_t = ttnn.to_torch(out_tt)
    ref_t = a_pt + b_pt  # torch broadcasts automatically

    x = compare_and_report(out_t, ref_t, label=label, out_tt=out_tt, layout_expected_rowmajor=layout_expected_rowmajor)

    # Spot-check small window
    H, W = out_t.shape[-2:]
    hh, ww = min(8, H), min(8, W)
    print(f"\nSample window [0,0 : h<{hh}, w<{ww}] of TTNN vs REF:")
    print("TTNN:\n", out_t[0, 0, :hh, :ww])
    print("REF:\n", ref_t[0, 0, :hh, :ww])

    print(f"Finished test: {label}")

    if isinstance(x, int):  # 69 good
        return x

    # Print values at bad indices
    # return 1
    print("\nPrinting values at mismatch indices:")
    max_l = 100
    for bol in x:
        n, c, h, w = bol.tolist()
        print(bol)
        print("TTNN:", out_t[n, c, h, w].item())
        print("REF :", ref_t[n, c, h, w].item())
        max_l -= 1
        if max_l == 0:
            break

    return 1  # bad


def run_fallback_case(dev, out_shape, label, layout_expected_rowmajor=False):
    print("\n" + "-" * 80)
    print(
        f"Starting test: {label} | Out shape={out_shape} (fallback via reshape) | Layout={LAYOUT} | TTNN dtype={DTYPE_TTNN}"
    )

    N, C, H, W = out_shape
    a_start = 1
    b_start = 100000

    # A full
    a_pt, a_tt = _make(dev, out_shape, start=a_start)

    # B as (N,C,W,1), then reshape to (N,C,1,W)
    b_raw_shape = (N, C, W, 1)
    b_pt_raw = torch.arange(b_start, N * C * W * 1 + b_start, dtype=DTYPE_TORCH).reshape(b_raw_shape)
    b_tt_raw = ttnn.from_torch(
        b_pt_raw, dtype=DTYPE_TTNN, layout=LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt_view = ttnn.reshape(b_tt_raw, ttnn.Shape((N, C, 1, W)))

    # Expected: broadcast reshaped B
    b_pt_view = b_pt_raw.reshape(N, C, 1, W)
    ref_t = a_pt + b_pt_view

    out_tt = ttnn.add(a_tt, b_tt_view, use_legacy=None)
    out_t = ttnn.to_torch(out_tt)

    x = compare_and_report(out_t, ref_t, label=label, out_tt=out_tt, layout_expected_rowmajor=layout_expected_rowmajor)

    # Spot-check
    hh, ww = min(8, H), min(8, W)
    print(f"\nSample window [0,0 : h<{hh}, w<{ww}] of TTNN vs REF:")
    print("TTNN:\n", out_t[0, 0, :hh, :ww])
    print("REF:\n", ref_t[0, 0, :hh, :ww])

    print(f"Finished test: {label}")

    if isinstance(x, int):
        return x

    print("\nPrinting values at mismatch indices:")
    for bol in x:
        n, c, h, w = bol.tolist()
        print(bol)
        print("TTNN:", out_t[n, c, h, w].item())
        print("REF :", ref_t[n, c, h, w].item())

    return 1


def core_suite_minimal():
    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    total_tests = 0
    passed_tests = 0
    failed = []

    categories = [
        "no-bcast",
        "scalar-bcast",
        "row-bcast-B",
        "col-bcast-A",
        "mixed-colA-rowB",
        "mixed-rowA-colB",
    ]
    # print("PYTHON FUCKER WT ARE YOU DONG ?")
    try:
        for category in categories:
            print(f"\n{'=' * 80}\nTesting category: {category}\n{'=' * 80}")
            for shape in TEST_SHAPES:
                N, C, H, W = shape
                layout_expected = True
                if category == "no-bcast":
                    a_sh = (N, C, H, W)
                    b_sh = (N, C, H, W)
                elif category == "scalar-bcast":
                    a_sh = (N, C, H, W)
                    b_sh = (N, C, 1, 1)
                elif category == "col-bcast-A":
                    a_sh = (N, C, H, 1)
                    b_sh = (N, C, H, W)
                elif category == "row-bcast-B":
                    a_sh = (N, C, H, W)
                    b_sh = (N, C, 1, W)
                elif category == "mixed-colA-rowB":
                    a_sh = (N, C, H, 1)
                    b_sh = (N, C, 1, W)
                elif category == "mixed-rowA-colB":
                    a_sh = (N, C, 1, W)
                    b_sh = (N, C, H, 1)
                else:
                    continue

                # Skip if incompatible (e.g., H=1 for col bcast needing H>1)
                if (H == 1 and ("col" in category or "H" in category)) or (
                    W == 1 and ("row" in category or "W" in category)
                ):
                    print(f"Skipping incompatible shape {shape} for {category}")
                    continue

                total_tests += 1
                label = f"{category} shape={shape}"
                nz = run_case(dev, a_sh, b_sh, label=label, layout_expected_rowmajor=layout_expected)
                if nz == 69:
                    passed_tests += 1
                else:
                    failed.append((category, shape))
                    print("Fail:", category, shape)
                    if EARLY_EXIT_ON_FAIL:
                        print("Early exit due to failure")
                        return 1

        # Fallback cases
        # print(f"\n{'=' * 80}\nTesting category: fallback-mixed\n{'=' * 80}")
        # for shape in TEST_SHAPES_FALLBACK:
        #     total_tests += 1
        #     label = f"fallback-mixed shape={shape}"
        #     nz = run_fallback_case(dev, shape, label=label, layout_expected_rowmajor=False)
        #     if nz == 69:
        #         passed_tests += 1
        #     else:
        #         failed.append(("fallback-mixed", shape))
        #         if EARLY_EXIT_ON_FAIL:
        #             print("Early exit due to failure")
        #            return 1

    finally:
        ttnn.close_device(dev)

    print(f"\nSummary: {passed_tests} / {total_tests} tests passed!")
    print("Failed tests:", len(failed))
    print(failed)
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    exit(core_suite_minimal())
