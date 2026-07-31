"""VERIFIER: per-core L1 CB budget of the shipped descriptor, both activation formats."""
import torch, ttnn
from ttnn.operations.moe_fused_swiglu import default_compute_kernel_config
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import create_program_descriptor, make_mailbox

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print(f"grid = {grid.x} x {grid.y} = {grid.x*grid.y} cores")
    for fmt, (dt, lay) in (
        ("bf16_rm", (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)),
        ("bfp8_tile", (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)),
    ):
        emb, cap = 7168, 5120
        x = ttnn.from_torch(
            torch.zeros((1, 1, cap, emb), dtype=torch.bfloat16),
            dtype=dt,
            layout=lay,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = [
            ttnn.from_torch(
                torch.zeros(s, dtype=torch.bfloat16),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for s in ((emb, 2048), (emb, 2048), (2048, emb))
        ]
        c = ttnn.from_torch(
            torch.zeros(256, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        i = ttnn.from_torch(
            torch.zeros(8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, cap, emb]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        mb = make_mailbox(device, int(grid.x) * int(grid.y))
        d = create_program_descriptor(
            x,
            w[0],
            w[1],
            w[2],
            c,
            i,
            out,
            mb,
            local_expert_id=3,
            input_m_tiles=cap // 32,
            compute_kernel_config=default_compute_kernel_config(),
        )
        tot = sum(cb.total_size for cb in d.cbs)
        print(
            f"{fmt}: {len(d.cbs)} CBs, total per-core L1 = {tot} B = {tot/1024:.1f} KB "
            f"of 1427.1 KB budget ({100*tot/1461376:.1f}%)"
        )
        for cb in d.cbs:
            f = cb.format_descriptors[0]
            print(
                f"    idx {f.buffer_index:2d}  {cb.total_size//f.page_size:4d} pages x {f.page_size:5d} B = {cb.total_size/1024:7.1f} KB"
            )
finally:
    ttnn.close_device(device)
