import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
RM = ttnn.ROW_MAJOR_LAYOUT
for shape, dt in [
    ((1, 1, 256, 512), ttnn.bfloat16),
    ((1, 1, 32, 50), ttnn.bfloat16),
    ((1, 1, 32, 8192), ttnn.bfloat16),
    ((4, 8, 32, 256), ttnn.bfloat16),
    ((128, 512), ttnn.bfloat16),
    ((1, 1, 17, 50), ttnn.bfloat16),
    ((1, 1, 32, 64), ttnn.float32),
]:
    cfg = auto_shard_config(list(shape), HEIGHT, layout=RM, dtype=dt, device=device)
    ss = cfg.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    ncores = ss.grid.num_cores()
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)
    total_rows = NC * int(shape[-2])
    xt = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
        dtype=dt,
        layout=RM,
        device=device,
        memory_config=cfg,
    )
    print(
        f"shape={shape} -> shard[{sh},{sw}] ncores={ncores} total_rows={total_rows} page={xt.buffer_aligned_page_size()} elem={xt.element_size()} W={shape[-1]} Ht_local={-(-sh//32)}"
    )
ttnn.close_device(device)
