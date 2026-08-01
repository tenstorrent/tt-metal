import importlib, ttnn
from loguru import logger

T = importlib.import_module("ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.test_scatter_matmul")
device = ttnn.open_device(device_id=0)
try:
    for m, n in ((1, 2), (2, 2), (1, 6), (2, 6)):
        geo = T.geo_of(m=m, n=n)
        logger.info(f"PROBE start direct T={geo.t} (4 dispatches)")
        pcc, ns, s = T.run_cell(device, "direct", geo, "addchain", 1, measure=True)
        logger.info(f"PROBE ok    direct T={geo.t}: pcc={pcc} ns={ns} samples={s}")
finally:
    ttnn.close_device(device)
