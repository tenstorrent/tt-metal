# Why is bt/b13 0 on the interleaved multi-block regimes? Dump the whole plan.
import torch, ttnn
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 8192, 32), (1, 1, 4096, 64)]:
        t = torch.zeros(shape).bfloat16()
        d = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        probe = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        p = tpd.build_plan(d, probe, device)
        print(
            shape,
            {
                k: p[k]
                for k in (
                    "path",
                    "bank_table",
                    "stateful_read",
                    "split_read",
                    "prefetch_blocks",
                    "read_group",
                    "stagger",
                    "fanin_mode",
                    "coalesce_rows",
                    "blocks_row_major",
                    "addr_probe",
                    "blocks_per_core",
                    "chunk_row_bytes",
                    "row_page_stride",
                    "depth",
                    "ncores",
                    "nt_h",
                    "chunk_wt",
                )
            },
            flush=True,
        )
finally:
    ttnn.close_device(device)
