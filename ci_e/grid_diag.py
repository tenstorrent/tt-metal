# Replicates test_grid_sample_bilinear_weight_format for chosen configs and dumps output, golden and mismatch positions.
import os, sys
if not os.environ.get("HWLOCK_HELD"):  # CI: the job sets HWLOCK_HELD=ci
    sys.exit("not under hwlock")
import sys, os, json, torch, ttnn
from ttnn.operations.pool import golden_grid_sample

out_dir, tag = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
CFGS = [
    # channels, align_corners, sharded_grid, grid_kind, input_dtype
    (384, True, True, "float32", torch.bfloat16),
    (384, True, True, "bfloat16", torch.float32),
    (384, True, True, "precomputed", torch.bfloat16),
    (384, False, True, "float32", torch.bfloat16),
    (384, True, False, "float32", torch.bfloat16),
]
if len(sys.argv) > 3 and sys.argv[3] == "first":
    CFGS = CFGS[:1]
device = ttnn.open_device(device_id=0)
res = {}
for channels, align_corners, sharded_grid, grid_kind, input_dtype in CFGS:
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 8, 16, channels).to(input_dtype)
    grid = torch.randint(-12, 13, (1, 2, 32, 2)).float() / 8
    expected = golden_grid_sample(input_tensor, grid, align_corners=align_corners)
    pre = grid_kind == "precomputed"
    grid_dtype = ttnn.float32 if grid_kind == "float32" else ttnn.bfloat16
    grid_host = ttnn.from_torch(grid, dtype=ttnn.float32 if pre else grid_dtype)
    if pre:
        grid_host = ttnn.prepare_grid_sample_grid(grid_host, list(input_tensor.shape), mode="bilinear",
                                                  align_corners=align_corners, output_dtype=ttnn.bfloat16)
    epp = 6 if pre else 2
    grid_host = ttnn.reshape(grid_host, (1, 2, 8, 4 * epp))
    grid_device = ttnn.to_device(grid_host, device)
    if sharded_grid:
        mc = ttnn.create_sharded_memory_config((8, 4 * epp), ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.HEIGHT,
                                               ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        grid_device = ttnn.to_memory_config(grid_device, mc)
    input_device = ttnn.from_torch(input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    outs = []
    for rep in range(2):
        o = ttnn.grid_sample(input_device, grid_device, mode="bilinear", align_corners=align_corners,
                             use_precomputed_grid=pre)
        outs.append(ttnn.to_torch(o).float())
        if rep == 0:
            try:
                res_addr = o.buffer_address()
            except Exception:
                res_addr = None
    out = outs[0]
    exp = expected.float()
    back = ttnn.to_torch(input_device).float()
    clob = (back != input_tensor.float())
    clob_sticks = sorted({(i[1], i[2]) for i in clob.nonzero().tolist()})
    clob_info = {f"{h},{w}": [int(c) for c in clob[0, h, w].nonzero().flatten().tolist()][:1] + [int(clob[0, h, w].sum())] for h, w in clob_sticks}
    try:
        addrs = {"input": input_device.buffer_address(), "grid": grid_device.buffer_address(), "out": res_addr}
    except Exception as e:
        addrs = {"err": str(e)}
    bad = ~torch.isclose(out, exp, atol=0.02, rtol=0.02)
    key = f"c{channels}_ac{int(align_corners)}_sh{int(sharded_grid)}_{grid_kind}_{str(input_dtype).split('.')[-1]}"
    idx = bad.nonzero().tolist()
    pts = sorted({(i[1], i[2]) for i in idx})
    per_pt = {f"{h},{w}": sorted(i[3] for i in idx if (i[1], i[2]) == (h, w)) for h, w in pts}
    res[key] = {"mismatch": int(bad.sum()), "repeat_equal": bool(torch.equal(outs[0], outs[1])),
                "points": {k: [len(v), v[0], v[-1]] for k, v in per_pt.items()},
                "input_clobbered": clob_info, "addrs": addrs}
    torch.save({"out": outs, "exp": exp, "input": input_tensor, "grid": grid}, f"{out_dir}/{tag}_{key}.pt")
    print(tag, key, json.dumps(res[key]), flush=True)
json.dump(res, open(f"{out_dir}/{tag}_summary.json", "w"), indent=1)
ttnn.close_device(device)
print("DIAG DONE", tag)
