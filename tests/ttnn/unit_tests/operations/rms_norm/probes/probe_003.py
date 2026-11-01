import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import derive_geometry, select_blocking, _Sizes

device = ttnn.open_device(device_id=0)


def run(shape, gamma=False, layout=ttnn.ROW_MAJOR_LAYOUT):
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    g = None
    tg = None
    if gamma:
        g = torch.randn(shape[-1]).to(torch.bfloat16)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = ttnn.to_torch(rms_norm(tx, gamma=tg)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    if g is not None:
        ref = ref * g.float()
    geo = derive_geometry(shape)
    grid = device.compute_with_storage_grid_size()
    b = select_blocking(
        geo,
        _Sizes(ttnn.bfloat16, ttnn.bfloat16, layout == ttnn.ROW_MAJOR_LAYOUT, gamma),
        grid.x,
        grid.y,
        ttnn.get_max_worker_l1_unreserved_size(),
        2,
    )
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
    # per-column-chunk error
    err = (y - ref).abs().reshape(-1, shape[-1]).amax(0)
    bad_cols = (err > 0.1).nonzero().flatten()
    bad_rows = ((y - ref).abs().reshape(-1, shape[-1]).amax(1) > 0.1).nonzero().flatten()
    print(
        f"CASE {shape} gamma={gamma} k={b.k} C={b.rm_chunk_w_tiles} pitch={b.x_row_pitch} b={b.block_row_tiles} d={b.x_block_depth} pcc={pcc:.5f} badcols={bad_cols[:3].tolist()}..{bad_cols[-3:].tolist()} n={len(bad_cols)} badrows={bad_rows[:3].tolist()} n={len(bad_rows)}"
    )


run((1, 1, 6400, 64), layout=ttnn.TILE_LAYOUT)
run((1, 1, 6400, 1280), layout=ttnn.TILE_LAYOUT)
ttnn.close_device(device)
