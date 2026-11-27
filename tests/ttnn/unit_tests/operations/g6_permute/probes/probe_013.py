import torch, ttnn
from ttnn.operations.g6_permute import g6_permute_plan

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 64, 128, 32, 2, 2), (1, 32, 64, 32, 2, 2)]:
        for mc in [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]:
            t = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=mc,
            )
            p = g6_permute_plan(t, (0, 1, 4, 2, 5, 3))
            N, C, H, W, I, J = p.shape
            inp = p.input_depth * p.block_h * W * I * p.page_stride
            stg = p.staging_depth * p.num_banks * p.staging_rounds * p.slot_bytes
            rows = (
                (p.write_group_depth * I * p.num_banks * p.group_slots * p.out_slot_bytes)
                if p.write_group_units
                else p.block_h * I * J * p.out_row_bytes
            )
            print(
                f"PLAN {shape} {mc.buffer_type} regime={p.regime} cores={p.num_cores} units_g1={p.units_per_core_g1} block_h={p.block_h} blocks={p.max_blocks_per_core} cap={p.block_cap} depth={p.input_depth} banks={p.num_banks} slot={p.slot_bytes} stride={p.page_stride} rounds={p.staging_rounds} sdepth={p.staging_depth} alt={p.read_alt_share} wg={p.write_group_units} wgd={p.write_group_depth} gslots={p.group_slots} oslot={p.out_slot_bytes} L1: in={inp} staging={stg} rows={rows} total={inp+stg+rows}"
            )
            t.deallocate()
finally:
    ttnn.close_device(device)
