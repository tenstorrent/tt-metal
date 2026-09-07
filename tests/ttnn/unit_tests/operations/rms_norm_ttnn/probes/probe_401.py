import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import sys
sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn")
import importlib
torch = importlib.import_module("torch")
import ttnn
from eval.sharding import shard_config
_ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
try:
    import test_rms_norm_ttnn_perf as T
    for label, shape, ml, shard in (
        ("8192x1024 INT", (1, 1, 8192, 1024), _ML.INTERLEAVED, None),
        ("8192x1024 BLOCK", (1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    ):
        dtype = ttnn.bfloat16
        mc = ttnn.DRAM_MEMORY_CONFIG if ml == _ML.INTERLEAVED else shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=dev)
        x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, dev, mc)
        g = ttnn.from_torch(torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = T._config()
        seed = T.seed_descriptor(x, out, gamma=g, epsilon=1e-12, compute_kernel_config=cfg)
        mine = T.ttnn_descriptor(x, out, weight=g, epsilon=1e-12, compute_kernel_config=cfg, program_config=T._PC_NONE)
        sa = list(seed.kernels[0].compile_time_args)[:21]
        ma = list(mine.kernels[0].compile_time_args)[:21]
        print(f"RESULT {label:18s} reader diffs={[(i, sa[i], ma[i]) for i in range(21) if sa[i] != ma[i]]}", flush=True)
finally:
    ttnn.close_device(dev)
