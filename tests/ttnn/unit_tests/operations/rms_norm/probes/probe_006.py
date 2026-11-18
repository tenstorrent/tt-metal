import sys; sys.path.insert(0, ".")
import torch, ttnn
from eval.golden_tests.rms_norm.helpers import _build_memory_config
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
device = ttnn.open_device(device_id=0)
mc = _build_memory_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, [128, 8192], device, dtype=ttnn.float32)
x = ttnn.from_torch(torch.randn(128, 8192), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
y = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config())
print("addr", x.buffer_address(), y.buffer_address(), mc.shard_spec.shape, mc.shard_spec.grid)
orig = pd.select_blocking
def sb(*a, **k):
    print("budget", k.get("budget"))
    r = orig(*a, **k); print("blk", r); return r
pd.select_blocking = sb
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
try:
    pd.create_program_descriptor(x, None, y, 1e-6, cfg)
except Exception as e:
    print("ERR", e)
ttnn.close_device(device)
