import torch, ttnn
from loguru import logger
from ttnn.operations.rms_norm.perf_experiments.pass2_batch_rows import run_op, create_sharded_memory_config

TILE = 32


def make_case(device, per_w_t, ht_local):
    torch.manual_seed(7)
    m, n = ht_local * TILE, per_w_t * TILE
    x = (torch.rand(m, n) * 2 - 1).to(torch.bfloat16).to(torch.float32)
    rstd_col = (torch.rand(m) * 1.5 + 0.25).to(torch.float32)
    gamma_row = (torch.rand(n) * 2 - 1).to(torch.bfloat16).to(torch.float32)
    rstd_full = rstd_col[:, None].expand(m, TILE).contiguous().to(torch.float32)
    gamma_full = gamma_row[None, :].expand(TILE, n).contiguous().to(torch.float32)
    expected = x * rstd_col[:, None] * gamma_row[None, :]
    xd = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, n)),
    )
    rd = ttnn.from_torch(
        rstd_full,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, TILE)),
    )
    gd = ttnn.from_torch(
        gamma_full.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((TILE, n)),
    )
    return xd, rd, gd, expected


def pcc(a, e):
    return torch.corrcoef(torch.stack([a.flatten().double(), e.flatten().double()]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    x, rstd, gamma, expected = make_case(device, 4, 32)
    for ki in (2, 3):
        out = run_op(
            x, rstd, gamma, variant="batch_both", per_w_t=4, ht_local=32, c_rows=8, has_gamma=True, kernel_iters=ki
        )
        p = pcc(ttnn.to_torch(out).float(), expected)
        logger.info(f"batch_both c_rows=8 ki={ki} PCC={p:.5f}")
finally:
    ttnn.close_device(device)
