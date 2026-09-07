"""Perf 3 -- WHICH writer/compute CT args moved away from the seed, and what are they?"""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import sys
sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn")
import importlib
torch = importlib.import_module("torch")
import ttnn
from eval.sharding import shard_config, auto_shard_config
_ML = ttnn.TensorMemoryLayout
CASES = [
    ("8192x1024 INT gamma", (1, 1, 8192, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None, "gamma"),
    ("32x7168 INT gamma", (1, 1, 32, 7168), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None, "gamma"),
    ("1024x512 WIDTH gamma", (1, 1, 1024, 512), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, ([1024, 64], (8, 1)), "gamma"),
]
dev = ttnn.open_device(device_id=0)
try:
    import test_rms_norm_ttnn_perf as T
    for label, shape, layout, ml, shard, mode in CASES:
        dtype = ttnn.bfloat16
        if ml == _ML.INTERLEAVED:
            mc = ttnn.DRAM_MEMORY_CONFIG
        elif shard is not None:
            mc = shard_config(shard[0], shard[1], ml, layout=layout, dtype=dtype, device=dev)
        else:
            mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=dev)
        x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=layout, device=dev, memory_config=mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, dev, mc)
        g = ttnn.from_torch(torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=dtype, layout=layout, device=dev) if mode == "gamma" else None
        cfg = T._config()
        seed = T.seed_descriptor(x, out, gamma=g, epsilon=1e-12, compute_kernel_config=cfg)
        mine = T.ttnn_descriptor(x, out, weight=g, epsilon=1e-12, compute_kernel_config=cfg, program_config=T._PC_NONE)
        for ki, kname in ((1, "writer"), (2, "compute")):
            sa = list(seed.kernels[ki].compile_time_args)
            ma = list(mine.kernels[ki].compile_time_args)
            diffs = [(i, sa[i], ma[i]) for i in range(min(len(sa), len(ma))) if sa[i] != ma[i]]
            print(f"RESULT {label:22s} {kname:8s} seed_n={len(sa):3d} mine_n={len(ma):3d} diffs={diffs}", flush=True)
finally:
    ttnn.close_device(dev)
