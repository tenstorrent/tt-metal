import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
try:
    for shape in [(7, 224, 3072), (1, 1, 224, 3072)]:
        try:
            mc = auto_shard_config(
                list(shape), ML.HEIGHT_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device
            )
            print("MSG spec", list(mc.shard_spec.shape), "ncores", mc.shard_spec.grid.num_cores())
            tx = ttnn.from_torch(
                torch.randn(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=mc,
            )
            tg = ttnn.from_torch(
                torch.randn(shape[-1], dtype=torch.bfloat16).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            rms_norm(tx, gamma=tg, memory_config=tx.memory_config())
            print("MSG ok", shape)
        except Exception as e:
            for line in str(e).splitlines()[:6]:
                print("MSG", shape, line[:230])
finally:
    ttnn.close_device(device)
