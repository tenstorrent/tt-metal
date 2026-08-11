import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
try:
    for shape in [(1, 224, 11008)]:
        mc = auto_shard_config(
            list(shape), ML.HEIGHT_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        print("MSG spec", list(mc.shard_spec.shape), mc.shard_spec.grid.num_cores())
        t = torch.randn(shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        gm = ttnn.from_torch(
            torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        probe = ttnn.cb_descriptor_from_sharded_tensor(1, x)
        print("MSG bank_bytes", int(probe.total_size), "page", int(probe.format_descriptors[0].page_size))
        print("MSG input addr", x.buffer_address(), "budget_l1", pd._l1_cb_budget())
        try:
            out = rms_norm(x, gamma=gm, memory_config=mc)
            print("MSG OK")
        except Exception as e:
            print("MSG ERR", str(e).replace("\n", " | ")[:600])
finally:
    ttnn.close_device(device)
