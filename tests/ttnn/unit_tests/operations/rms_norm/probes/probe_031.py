import ttnn
from ttnn.operations.rms_norm.perf_experiments.fused_scale import regime_flip
dev = ttnn.open_device(device_id=0)
regime_flip.report(dev)
ttnn.close_device(dev)
