import importlib, ttnn
from loguru import logger

T = importlib.import_module("ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.test_scatter_matmul")
device = ttnn.open_device(device_id=0)
try:
    for shape in ("scatter", "tree", "direct"):
        geo = T.geo_of()
        logger.info(f"PROBE start {shape} m=8 N=6")
        pcc, _, _ = T.run_cell(device, shape, geo, "addchain", 1, measure=False)
        logger.info(f"PROBE ok    {shape}: pcc={pcc}")
finally:
    ttnn.close_device(device)
