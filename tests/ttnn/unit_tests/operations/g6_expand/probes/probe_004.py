import torch, ttnn
import ttnn.operations.g6_expand.g6_expand_program_descriptor as pdm
from ttnn.operations.g6_expand import g6_expand

device = ttnn.open_device(device_id=0)


def run(shape, sizes, label):
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xt = ttnn.fill_implicit_tile_padding(xt, 1000.0)
    out = g6_expand(xt, sizes)
    got = ttnn.to_torch(out)
    exp = x.expand(*sizes).contiguous()
    p = g6_expand.last_plan
    same = torch.equal(got.view(torch.int16), exp.view(torch.int16))
    print(
        f"CASE {label}: {'ok' if same else 'FAIL'} plan cores={p.num_cores} groups={p.pattern_groups}x{p.replica_groups} block={p.block_pattern_tiles} depth={p.pattern_cb_depth_blocks} blocks0={p.num_blocks(0)}"
    )


try:
    run([2, 3, 1, 96], [2, 3, 64, 96], "multi_image_ragged")
    pdm.L1_OP_BUDGET_BYTES = pdm.L1_RESERVED_BYTES + 2048
    run([1, 1, 1, 512], [1, 1, 512, 512], "two_blocks_depth1_mcast")
    run([1, 1, 1, 207], [1, 1, 207, 207], "one_block_depth1")
    pdm.PATTERN_CB_DEPTH_BLOCKS = 2
    pdm.L1_OP_BUDGET_BYTES = pdm.L1_RESERVED_BYTES + 2 * 2048
    run([1, 1, 1, 512], [1, 1, 512, 512], "two_blocks_depth2_mcast")
    pdm.MAX_PATTERN_GROUPS = 3
    run([1, 1, 1, 207], [1, 1, 207, 207], "ragged_blocks_depth2")
    pdm.MAX_REPLICA_GROUPS = 1
    run([1, 1, 1, 512], [1, 1, 512, 512], "no_receivers_absent_mcast")
    run([1, 1, 1, 196], [1, 1, 1, 196], "R1_single_row")
finally:
    ttnn.close_device(device)
