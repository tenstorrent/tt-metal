import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}

# The L1-tight corner: a 1-tile-row HEIGHT shard of a WIDE tensor puts the whole
# input AND output shard in one core's L1 on top of every CB.
cases = [
    (s, dt, g)
    for s in [
        (1, 1, 32, 4096),
        (1, 1, 128, 4096),
        (2, 1, 64, 4096),
        (1, 32, 4096),
        (32, 4096),
        (1, 1, 32, 8192),
        (1, 32, 8192),
        (128, 8192),
    ]
    for dt in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)
    for g in (True,)
]
for shape, dt, has_g in cases:
    try:
        mc = auto_shard_config(list(shape), HS, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
        x = torch.randn(shape, dtype=TD[dt])
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tg = (
            ttnn.from_torch(
                torch.randn(1, 1, 1, shape[-1], dtype=TD[dt]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device
            )
            if has_g
            else None
        )
        out = rms_norm(tx, gamma=tg, memory_config=mc)
        ht, wt = pd._tile_geometry(tx)
        p = pd._select_placement(device, device.compute_with_storage_grid_size(), tx, ht, wt, True)
        blk = pd._derive_blocking(tx, tg, 110, p, sharded_in=True, sharded_out=True)
        print(
            f"OK   {shape} {dt.name} shard={list(mc.shard_spec.shape)} cores={p.num_cores} chunk={blk.wt_chunk} nw={blk.nw} cb={blk.cb_total_bytes}",
            flush=True,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(tx)
    except Exception as e:
        print(f"FAIL {shape} {dt.name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
ttnn.close_device(device)
