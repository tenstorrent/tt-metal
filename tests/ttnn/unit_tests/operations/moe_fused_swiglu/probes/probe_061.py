import torch, ttnn
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import weight_memory_configs, worker_grid

device = ttnn.open_device(device_id=0)
try:
    print("worker grid:", worker_grid(device))
    for emb in (7168, 6144):
        gu_mc, dn_mc = weight_memory_configs(device, emb, 2048)
        print(
            f"emb={emb} gate/up shard={list(gu_mc.nd_shard_spec.shard_shape)} down shard={list(dn_mc.nd_shard_spec.shard_shape)}"
        )
        for name, shape, mc in (("w_gate", (emb, 2048), gu_mc), ("w_down", (2048, emb), dn_mc)):
            i = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            s = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mc,
            )
            bi, bs = i.buffer().size(), s.buffer().size()
            print(f"  {name}: interleaved {bi:>10,} B   sharded {bs:>10,} B   overhead {100*(bs-bi)/bi:+.2f}%")
            ttnn.deallocate(i)
            ttnn.deallocate(s)
finally:
    ttnn.close_device(device)
