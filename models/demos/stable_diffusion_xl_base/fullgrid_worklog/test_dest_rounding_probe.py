# Deterministic probe of the 16-bit DEST accumulate rounding in matmul (fp32_dest_acc_en=False, packer_l1_acc=False).
# One 32x32 output tile on one core. in0 rows carry a multiplier f_r, in1 columns carry a base (K-tile 0) and a
# delta d_c (every later K-tile, one nonzero row per tile). Output cell = base_c + n_adds * f_r * d_c, with the
# discarded bits placed at an exact bf16 tie, just above, or just below, for both signs. Every cell is compared to
# simulated accumulators: RNE / ties-away / truncate / round-up-in-magnitude after every tile add, and a wide
# accumulator rounded once at the end.
import numpy as np
import pytest
import torch

import ttnn

# (base, delta) per output column. Bases sit at bf16 ULP 2 (256..512) or 32 (4096..8192) or 2^-7 (1..2).
COLS = [
    (256.0, 1.0),  # tie, even base
    (258.0, 1.0),  # tie, odd base
    (256.0, 0.75),  # below tie
    (256.0, 1.25),  # above tie
    (256.0, 0.5),  # quarter ULP
    (256.0, 1.5),  # 3/4 ULP
    (-256.0, -1.0),  # negative tie, even
    (-258.0, -1.0),  # negative tie, odd
    (-256.0, -0.75),
    (-256.0, -1.25),
    (256.0, -1.0),  # 255: exactly representable (ULP 1 below 256)
    (256.0, -0.5),  # 255.5: tie between 255 and 256
    (256.0, 2.0),  # exact
    (256.0, 0.0),  # control
    (4096.0, 16.0),  # tie at ULP 32
    (4128.0, 16.0),  # tie, odd base
    (4096.0, 12.0),
    (4096.0, 20.0),
    (4096.0, 8.0),
    (4096.0, 4.0),
    (4096.0, 2.0),
    (-4096.0, -16.0),
    (-4096.0, -12.0),
    (-4096.0, -20.0),
    (1.0, 2.0**-8),  # tie at ULP 2^-7
    (1.0078125, 2.0**-8),  # odd base
    (1.0, 3 * 2.0**-9),
    (1.0, 2.0**-9),
    (-1.0, -(2.0**-8)),
    (1.0, -(2.0**-8)),  # 1 - 2^-8: tie between 1-2^-7 and 1 (ULP below 1 is 2^-8, so exact)
    (0.0, 1.0),  # pure sum of ones (exact until 256, then ties)
    (0.0, 0.75),
]
# multiplier per output row (bf16-exact); product f*d is exact in HiFi4.
ROWS = [
    1.0,
    1 + 2.0**-7,
    1 - 2.0**-7,
    1 + 2.0**-6,
    1 - 2.0**-6,
    1 + 2.0**-5,
    1 - 2.0**-5,
    1 + 2.0**-4,
    1 - 2.0**-4,
]
ROWS += [1 + 2.0**-3, 1 - 2.0**-3, 1.25, 0.75, 0.5, 1.5, 2.0, 3.0, 0.25, 4.0, 0.125]
ROWS += [1.0] * (32 - len(ROWS))
assert len(COLS) == 32 and len(ROWS) == 32


def _bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _from_bits(b):
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def rnd_bf16(x, mode):
    """Round float64 -> bf16 (as float64) with the given mode; values must be within float32 exactly first."""
    x32 = np.asarray(x, dtype=np.float64).astype(np.float32)
    b = _bits(x32)
    low = b & 0xFFFF
    hi = b & 0xFFFF0000
    if mode == "trunc":
        return _from_bits(hi)
    if mode == "away":  # round half away from zero (on magnitude)
        return _from_bits(((b + 0x8000) & 0xFFFF0000))
    if mode == "up_mag":  # any discarded bit -> round magnitude up
        return _from_bits(np.where(low != 0, hi + 0x10000, hi))
    if mode == "rne":
        lsb = (b >> 16) & 1
        return _from_bits(((b + 0x7FFF + lsb) & 0xFFFF0000))
    raise ValueError(mode)


def simulate(base, f, d, n, mode):
    acc = float(base)
    if mode == "wide":  # exact accumulate, single final RNE to bf16
        return float(rnd_bf16(acc + n * f * d, "rne"))
    for _ in range(n):
        acc = float(rnd_bf16(acc + f * d, mode))
    return acc


MODES = ("rne", "away", "trunc", "up_mag", "wide")


def build(n_adds):
    Kt = 1 + n_adds
    K = 32 * Kt
    a = torch.zeros(1, 1, 32, K, dtype=torch.float32)
    w = torch.zeros(1, 1, K, 32, dtype=torch.float32)
    a[..., :, 0] = 1.0  # tile 0 row 0: base multiplier 1
    for t in range(1, Kt):
        for r, f in enumerate(ROWS):
            a[..., r, 32 * t] = f
    for c, (base, d) in enumerate(COLS):
        w[..., 0, c] = base
        for t in range(1, Kt):
            w[..., 32 * t, c] = d
    # everything must be bf16-exact
    assert torch.equal(a, a.to(torch.bfloat16).float()) and torch.equal(w, w.to(torch.bfloat16).float())
    return a, w, Kt


def run_mm(device, a, w, Kt, fp32, l1acc, fid="HiFi4"):
    ta = ttnn.from_torch(a, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tw = ttnn.from_torch(w, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=Kt,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    out = ttnn.matmul(
        ta,
        tw,
        program_config=pc,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=l1acc,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return ttnn.to_torch(out).float().reshape(32, 32).double().numpy()


@pytest.mark.parametrize("n_adds", [1, 8, 40])
@pytest.mark.parametrize("fp32", [False, True])
def test_dest_rounding(device, n_adds, fp32):
    a, w, Kt = build(n_adds)
    dev = run_mm(device, a, w, Kt, fp32=fp32, l1acc=False)
    sims = {m: np.zeros((32, 32)) for m in MODES}
    for r, f in enumerate(ROWS):
        for c, (base, d) in enumerate(COLS):
            for m in MODES:
                sims[m][r, c] = simulate(base, f, d, n_adds, m)
    print(f"\n=== DEST probe: n_adds={n_adds} fp32_dest_acc={fp32} l1acc=0 HiFi4 ===")
    for m in MODES:
        match = np.isclose(dev, sims[m], rtol=0, atol=0)
        print(f"MODEL {m:7s} matches {match.sum():4d}/1024 cells")
    # cells where the models disagree with each other are the informative ones: print them
    print("row_f      base       delta   n  | device       rne          away         trunc        up_mag       wide")
    shown = 0
    for r, f in enumerate(ROWS):
        for c, (base, d) in enumerate(COLS):
            vals = [sims[m][r, c] for m in MODES]
            if len(set(vals)) > 1 or not np.isclose(dev[r, c], vals[0]):
                if shown < 140:
                    print(
                        f"{f:<9.6g} {base:<10.6g} {d:<8.6g} {n_adds:3d} | {dev[r,c]:<12.7g} "
                        + " ".join(f"{v:<12.7g}" for v in vals)
                    )
                shown += 1
    print(f"({shown} informative cells)")
    # sanity: exact cells (delta 0 or exact sums) must be exact on device
    exact_col = COLS.index((256.0, 0.0))
    assert np.all(dev[:, exact_col] == 256.0), dev[:, exact_col]


def test_random_gain_one_core(device):
    """Reproduce the magnitude bias on the same 1-core path with random data (sanity that this path is the biased one)."""
    torch.manual_seed(0)
    for Kt in (2, 8, 40):
        K = 32 * Kt
        a = torch.randn(1, 1, 32, K)
        w = torch.randn(1, 1, K, 32) / K**0.5
        ab, wb = a.to(torch.bfloat16).float(), w.to(torch.bfloat16).float()
        ref = (ab.double() @ wb.double()).reshape(32, 32).numpy()
        for fp32 in (False, True):
            dev = run_mm(device, ab, wb, Kt, fp32=fp32, l1acc=False)
            gain = (dev * ref).sum() / (ref * ref).sum()
            print(
                f"RANDOM Kt={Kt:3d} fp32={int(fp32)}: gain={gain:.5f} std_ratio={dev.std()/ref.std():.5f} "
                f"bias={(dev-ref).mean():+.2e} rel_rms={np.sqrt(((dev-ref)**2).mean())/ref.std():.5f}"
            )


@pytest.mark.parametrize("fid", ["HiFi4", "LoFi"])
def test_per_product_vs_per_tile(device, fid):
    """Is the 16-bit DEST rounding applied once per K-tile SOP or once per product (or per sub-chunk)?
    Column groups: the same tile-sum delta (1.0 on base 256 = exact tie) delivered as 1 product, 2 products of 0.5,
    4 of 0.25, 8 of 0.125, 16 of 0.0625, 32 of 0.03125 per K-tile. If the whole tile SOP is rounded once, every
    group is a tie and rounds away (258 per tile, 256 -> 256+2n). If products are accumulated one at a time into
    16-bit DEST, the spread groups add sub-half-ULP amounts that each round back to 256 and the sum stays 256.
    Also base 4096 (ULP 32) with tile-sum 16, and below-tie sums 0.75 spread."""
    n_adds = 8
    Kt = 1 + n_adds
    K = 32 * Kt
    a = torch.zeros(1, 1, 32, K, dtype=torch.float32)
    w = torch.zeros(1, 1, K, 32, dtype=torch.float32)
    a[..., :, 0] = 1.0
    a[..., :, 32:] = 1.0  # all rows identical multiplier 1; products = w values
    cases = []  # (base, tile_sum, n_products)
    for base, ts in ((256.0, 1.0), (4096.0, 16.0), (256.0, 0.75), (-256.0, -1.0), (256.0, 1.25)):
        for npd in (1, 2, 4, 8, 16, 32):
            cases.append((base, ts, npd))
    cases = cases[:32]
    for c, (base, ts, npd) in enumerate(cases):
        w[..., 0, c] = base
        for t in range(1, Kt):
            for j in range(npd):
                w[..., 32 * t + j, c] = ts / npd
    assert torch.equal(w, w.to(torch.bfloat16).float())
    dev = run_mm(device, a, w, Kt, fp32=False, l1acc=False, fid=fid)
    print(f"\n=== per-product vs per-tile probe ({fid}, fp32 off, l1acc off, {n_adds} K-tiles) ===")
    print("base     tile_sum  n_products | device(row0)   exact_sum   once-per-tile(away)")
    for c, (base, ts, npd) in enumerate(cases):
        once = simulate(base, 1.0, ts, n_adds, "away")
        print(f"{base:<8g} {ts:<9g} {npd:<10d} | {dev[0,c]:<14.7g} {base+n_adds*ts:<11g} {once:<g}")
    assert np.all(dev == dev[0:1, :]), "rows should be identical"


def test_half_tile_and_guard_grid(device):
    """Confirm (a) the K-tile is accumulated into DEST in two 16-row halves, (b) products are rounded (ties away)
    onto a grid 6 bits below the bf16 ULP of the DEST value before summation."""
    n_adds = 8
    Kt = 1 + n_adds
    K = 32 * Kt
    a = torch.zeros(1, 1, 32, K, dtype=torch.float32)
    w = torch.zeros(1, 1, K, 32, dtype=torch.float32)
    a[..., :, :] = 1.0
    a[..., :, 1:32] = 0.0  # tile 0: only row 0 carries the base
    # (base, [(row, value), ...]) per column; all rows of in0 in later tiles are 1.0 so products = values
    cases = {
        "A 16x(1/16) rows 0-15": (256.0, [(r, 1 / 16) for r in range(0, 16)]),
        "B 16x(1/16) rows 16-31": (256.0, [(r, 1 / 16) for r in range(16, 32)]),
        "C 16x(1/16) rows 8-23": (256.0, [(r, 1 / 16) for r in range(8, 24)]),
        "D 8x(1/8) rows 0-7 + 8x(1/8) rows 16-23 (0.5 per half... =1.0 each? no: 1.0 per half)": (
            256.0,
            [(r, 1 / 8) for r in range(0, 8)] + [(r, 1 / 8) for r in range(16, 24)],
        ),
        "E 32x(1/32) all rows": (256.0, [(r, 1 / 32) for r in range(32)]),
        "F 16x(1/32) rows 0-15 + 16x(1/32) rows 16-31 = E": (256.0, [(r, 1 / 32) for r in range(32)]),
        "G grid: 16 x 0.046875 (=1.5 grid) rows 0-15": (256.0, [(r, 0.046875) for r in range(16)]),
        "H grid: 8 x 0.09375 (=3 grid) rows 0-7": (256.0, [(r, 0.09375) for r in range(8)]),
        "I grid: 12 x 0.0625 (=2 grid) rows 0-11": (256.0, [(r, 0.0625) for r in range(12)]),
        "J grid: 16 x 0.0390625 (=1.25 grid) rows 0-15": (256.0, [(r, 0.0390625) for r in range(16)]),
        "K grid: 16 x 0.0546875 (=1.75 grid) rows 0-15": (256.0, [(r, 0.0546875) for r in range(16)]),
        "L grid: 16 x 0.015625 (=0.5 grid) rows 0-15": (256.0, [(r, 0.015625) for r in range(16)]),
        "M grid: 16 x 0.0078125 (=0.25 grid) rows 0-15": (256.0, [(r, 0.0078125) for r in range(16)]),
        "N base 4096 (ULP 32, grid 0.5): 16 x 0.75 (=1.5 grid) rows 0-15": (4096.0, [(r, 0.75) for r in range(16)]),
        "O base 4096: 16 x 1.0 (=2 grid) rows 0-15 (sum 16 = tie)": (4096.0, [(r, 1.0) for r in range(16)]),
        "P base 4096: 16 x 0.5 (=1 grid) rows 0-15 (sum 8)": (4096.0, [(r, 0.5) for r in range(16)]),
        "Q neg: -256, 16 x -0.046875 rows 0-15": (-256.0, [(r, -0.046875) for r in range(16)]),
        "R single 0.984375 row 0 (7 bits below tie)": (256.0, [(0, 0.984375)]),
        "S single 0.96875 row 0 (6 bits below tie)": (256.0, [(0, 0.96875)]),
        "T single 0.9921875 row 5": (256.0, [(5, 0.9921875)]),
    }
    names = list(cases)
    for c, nm in enumerate(names):
        base, rows = cases[nm]
        w[..., 0, c] = base
        for t in range(1, Kt):
            for r, v in rows:
                w[..., 32 * t + r, c] = v
    assert torch.equal(w, w.to(torch.bfloat16).float())
    dev = run_mm(device, a, w, Kt, fp32=False, l1acc=False, fid="HiFi4")
    print(f"\n=== half-tile / guard-grid probe (HiFi4, fp32 off, l1acc off, {n_adds} K-tiles) ===")
    for c, nm in enumerate(names):
        base, rows = cases[nm]
        ts = sum(v for _, v in rows)
        print(
            f"{nm:75s} tile_sum={ts:<9g} exact={base + n_adds*ts:<9g} device={dev[0,c]:<9g} "
            f"({'UP' if abs(dev[0,c]) > abs(base) else 'flat'})"
        )
