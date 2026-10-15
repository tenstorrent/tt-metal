# The DRAM-bank-adjacent workers (get_optimal_dram_bank_to_logical_worker_assignment) on NoC 0 / 1 and the logical ->
# virtual core map, to place a NoC hotspot relative to the DRAM columns (NEGATIVE_RESULTS 73). Usage: dram_workers.py
import ttnn

D = ttnn.open_device(device_id=0)
try:
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        ws = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(D, noc)
        print(f"RES {noc}: bank -> logical worker " + ", ".join(f"{i}:({c.x},{c.y})" for i, c in enumerate(ws)))
    g = D.compute_with_storage_grid_size()
    print(f"RES grid {g.x}x{g.y}")
    try:
        row = [D.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)) for x in range(g.x)]
        print("RES logical x -> virtual x (row 0): " + " ".join(f"{x}:{c.x}" for x, c in enumerate(row)))
        col = [D.worker_core_from_logical_core(ttnn.CoreCoord(0, y)) for y in range(g.y)]
        print("RES logical y -> virtual y (col 0): " + " ".join(f"{y}:{c.y}" for y, c in enumerate(col)))
    except Exception as e:
        print("RES no worker_core_from_logical_core:", e)
finally:
    ttnn.close_device(D)
