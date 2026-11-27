import ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
pd.BANK_LOCAL_DRAM_BYTES_PER_CYCLE = 10**6
a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 200, 2048]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
b = ttnn.allocate_tensor_on_device(ttnn.Shape([2048, 3008]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 200, 3008]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
r = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 3008]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
geom = pd._geometry(a, b, o, True, "bias_relu", r, None, None)
print("OUT", geom.b_banks, pd.bank_eligible(geom), pd.bank_classes(geom))
budget = pd.l1_cb_budget(dev, [a, b, o, r])
for g in pd.bank_candidates(geom, 11, 10):
    blk = pd.choose_blocking(geom, g, budget)
    print("OUT", g, blk, pd.operand_mode(geom, g, blk), [round(t / 1350, 1) for t in pd.split_cost_terms(geom, g, blk)])
grid, blk = pd.choose_split(geom, 11, 10, budget)
print(
    "OUT best",
    grid,
    [round(t / 1350, 1) for t in pd.split_cost_terms(geom, grid, blk)],
    pd.read_traffic(geom, grid, blk),
)
ttnn.close_device(dev)
