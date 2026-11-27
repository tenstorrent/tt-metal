# Untested paths from op_design.md Key Risks: ragged W, idle tail cores, uneven split, multi-tile
# blocks, multi-block depth-2 slot reuse, rank 1 and rank 3. Bit-exact compare, NaN input padding.
import torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze, plan_for
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as desc

device = ttnn.open_device(device_id=0)
torch.manual_seed(7)
SPECIAL = torch.tensor([0x7FC12345, 0x00000001, 0x80000000 - (1 << 32), 0x7F800001], dtype=torch.int64).to(torch.int32)


def run(shape, max_cores=None):
    desc.MAX_CORES = max_cores
    x = torch.randn(shape, dtype=torch.float32)
    flat = x.view(-1).view(torch.int32)
    n = flat[::5].numel()
    flat[::5] = SPECIAL.repeat(n // 4 + 1)[:n]
    t = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    t = ttnn.fill_implicit_tile_padding(t, float("nan"))
    plan = plan_for(t, max_cores)
    out = g6_unsqueeze(t, -1)
    got = ttnn.to_torch(out)
    want = torch.unsqueeze(x, -1)
    ok = tuple(got.shape) == tuple(want.shape) and torch.equal(
        got.contiguous().view(torch.int32), want.contiguous().view(torch.int32)
    )
    roles = {}
    for c in plan["cores"]:
        roles[c["core_w_tiles"]] = roles.get(c["core_w_tiles"], 0) + 1
    print(
        f"CASE {shape} max_cores={max_cores} rect={plan['rectangle']} ranges={plan['num_core_ranges']} "
        f"tiles/core->count={roles} block={plan['block_w_tiles']} depth={plan['staging_depth_blocks']} "
        f"l1={plan['l1_bytes_per_core']} -> {'OK' if ok else 'FAIL'}"
    )


run((1, 32 * 17 + 5))  # 17 tiles, ragged last tile, idle tail core
run((40,))  # rank 1, 2 tiles, second half-row padding
run((1, 1, 100))  # rank 3, ragged
run((1, 1, 7))  # rank 3, one partial tile, lanes 7..31 padding
run((1, 32 * 300 + 7))  # 301 tiles over the grid: 2-3 tiles per core, one block each
run((1, 32 * 17 + 5), 4)  # uneven groups 5/4/4/4
run((1, 32 * 400 + 9), 1)  # one core, 401 tiles: depth 2, 3 blocks, slot reuse, ragged last block
run((1, 32 * 700), 2)  # two cores, 350 tiles each: depth 2, 3 blocks of 117 (balanced, 350 = 117+117+116)
run((1, 512))
run((1, 256))
ttnn.close_device(device)
