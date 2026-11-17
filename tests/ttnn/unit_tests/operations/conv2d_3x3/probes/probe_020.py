import ttnn, importlib

pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
TILE = 32


def blk(name, H, W, c_in, c_out, p, s, d, sh, grid, bias=True):
    gx, gy = grid
    n_rows, n_cols = gy, gx
    c_in_p = sh[1]
    h_out = (H + 2 * p - 2 * d - 1) // s + 1
    w_out = (W + 2 * p - 2 * d - 1) // s + 1
    g = pd.Conv3x3Geometry(
        input_height=H,
        input_width=W,
        c_in_p=c_in_p,
        c_out=c_out,
        padding=p,
        stride=s,
        dilation=d,
        output_height=h_out,
        output_width=w_out,
        c_in=c_in,
        tile_input=True,
        in0_tile_bytes=1088,
        weight_tile_bytes=1088,
        out_tile_bytes=1088,
        has_bias=bias,
        sharded_output=True,
        k_cols=n_cols,
    )
    core_m_tiles, _ = pd.m_split(g, n_rows)
    resident = (sh[0] // 32) * (sh[1] // 32) * 1088 + core_m_tiles * g.out_n_tiles * 1088
    budget = pd.l1_cb_budget_bytes(resident)
    b = pd.select_blocking(g, (n_rows, 1), budget)
    print(
        name,
        "core_m",
        b.core_m_tiles,
        "bm",
        b.block_m_tiles,
        "bk",
        b.block_k_tiles,
        "/",
        g.tensor_kkh_tiles,
        "bn",
        b.block_n_tiles,
        "ns",
        g.out_n_tiles,
        "kcols",
        g.k_cols,
        "ub",
        b.untilize_batch_rows,
        "slab_rows",
        b.slab_tile_rows,
        "fp",
        b.footprint_bytes // 1024,
        "budget",
        budget // 1024,
    )
    # what does bm=core_m need?
    for bm in sorted({d for d in range(1, b.core_m_tiles + 1) if b.core_m_tiles % d == 0}, reverse=True):
        f = pd._footprint(g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles)
        print("   L bm", bm, "fp KB", f["footprint_bytes"] // 1024)


blk("res4_0", 64, 128, 256, 256, 1, 2, 1, [1024, 32], (8, 8))
blk("res4", 32, 64, 256, 256, 1, 1, 1, [256, 32], (8, 8))
blk("res5_0", 32, 64, 512, 512, 2, 1, 2, [224, 64], (8, 10))
blk("res5_2", 32, 64, 512, 512, 8, 1, 8, [224, 64], (8, 10))
blk("aspp1", 32, 64, 2048, 256, 6, 1, 6, [256, 192], (11, 8))
blk("aspp3", 32, 64, 1024, 256, 18, 1, 18, [256, 96], (11, 8), bias=False)
