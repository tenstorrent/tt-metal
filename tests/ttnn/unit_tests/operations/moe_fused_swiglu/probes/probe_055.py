import importlib, ttnn
from loguru import logger

T = importlib.import_module("ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.test_scatter_matmul")
device = ttnn.open_device(device_id=0)
try:
    for m, n in ((8, 6), (4, 6), (1, 6), (2, 2), (1, 2)):
        geo = T.geo_of(m=m, n=n)
        logger.info(f"PROBE direct m={m} N={n} T={geo.t}")
        pcc, _, _ = T.run_cell(device, "direct", geo, "addchain", 1, measure=False)
        logger.info(f"PROBE direct m={m} N={n} T={geo.t}: pcc={pcc}")
finally:
    ttnn.close_device(device)
