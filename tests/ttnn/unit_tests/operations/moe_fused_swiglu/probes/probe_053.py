import importlib, torch, ttnn
from loguru import logger

T = importlib.import_module("ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.test_scatter_matmul")
B = importlib.import_module("ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.bench")

device = ttnn.open_device(device_id=0)
try:
    geo = T.geo_of()
    for i in range(4):
        logger.info(f"--- direct dispatch {i} ---")
        pcc, _ = T.run_cell(device, "direct", geo, "addchain", 1, measure=False)
        logger.info(f"direct dispatch {i}: pcc={pcc}")
finally:
    ttnn.close_device(device)
