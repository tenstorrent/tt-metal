"""Perf 3 -- what exactly diverged in the stale seed-structure pin, and by how much L1."""
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
    ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 32, 7168), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 256, 512), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, None),
    ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
]
dev = ttnn.open_device(device_id=0)
try:
    import test_rms_norm_ttnn_perf as T
    for shape, layout, ml, shard in CASES:
        dtype = ttnn.bfloat16
        if ml == _ML.INTERLEAVED:
            mc = ttnn.DRAM_MEMORY_CONFIG
        elif shard is not None:
            mc = shard_config(shard[0], shard[1], ml, layout=layout, dtype=dtype, device=dev)
        else:
            mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=dev)
        x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=layout, device=dev, memory_config=mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, dev, mc)
        cfg = T._config()
        seed = T.seed_descriptor(x, out, gamma=None, epsilon=1e-12, compute_kernel_config=cfg)
        mine = T.ttnn_descriptor(x, out, weight=None, epsilon=1e-12, compute_kernel_config=cfg, program_config=T._PC_NONE)
        def tot(d):
            return sum(cb.total_size for cbs in [d.cbs] for cb in cbs) if hasattr(d, "cbs") else None
        ss, ms = T._cb_signature(seed, drop=T._TREE_RING_CBS), T._cb_signature(mine, drop=T._TREE_RING_CBS)
        sb = sum(t for t, _ in ss.values()) if isinstance(ss, dict) else sum(v[0] for v in ss)
        mb = sum(t for t, _ in ms.values()) if isinstance(ms, dict) else sum(v[0] for v in ms)
        print(f"RESULT {str(shape):22s} {str(ml).split('.')[-1][:6]:7s} seed_L1={sb:8d} mine_L1={mb:8d} delta={mb-sb:+8d} same={ss==ms}", flush=True)
finally:
    ttnn.close_device(dev)
