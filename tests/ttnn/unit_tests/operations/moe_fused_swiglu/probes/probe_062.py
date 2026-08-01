import torch, ttnn
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import weight_memory_configs

device = ttnn.open_device(device_id=0)


def dram_used():
    mv = ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


try:
    for emb in (7168, 6144):
        gu_mc, dn_mc = weight_memory_configs(device, emb, 2048)
        for name, shape, mc in (("w_gate", (emb, 2048), gu_mc), ("w_down", (2048, emb), dn_mc)):
            sizes = {}
            for tag, m in (("interleaved", ttnn.DRAM_MEMORY_CONFIG), ("sharded", mc)):
                before = dram_used()
                t = ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=m,
                )
                sizes[tag] = dram_used() - before
                ttnn.deallocate(t)
            bi, bs = sizes["interleaved"], sizes["sharded"]
            print(f"emb={emb} {name}: interleaved {bi:>12,} B  sharded {bs:>12,} B  overhead {100*(bs-bi)/bi:+.2f}%")
finally:
    ttnn.close_device(device)
