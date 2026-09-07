"""Perf 3 -- dump the op's OWN program on the cells the pin newly flags, so pre/post can
be diffed directly instead of inferred from the seed."""
import os, json
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_TRACE_BLOCKING"] = "1"
import sys
sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn")
import importlib
torch = importlib.import_module("torch")
import ttnn
from eval.sharding import auto_shard_config
_ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
try:
    import test_rms_norm_ttnn_perf as T
    for label, shape, layout, ml, mode in (
        ("32x7168_INT_nog", (1, 1, 32, 7168), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, "no_gamma"),
        ("256x512_HEIGHT_nog", (1, 1, 256, 512), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, "no_gamma"),
        ("256x512_RM_WIDTH_nog", (1, 1, 256, 512), ttnn.ROW_MAJOR_LAYOUT, _ML.WIDTH_SHARDED, "no_gamma"),
    ):
        dtype = ttnn.bfloat16
        mc = ttnn.DRAM_MEMORY_CONFIG if ml == _ML.INTERLEAVED else auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=dev)
        x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=layout, device=dev, memory_config=mc)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, dev, mc)
        g = ttnn.from_torch(torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=dtype, layout=layout, device=dev) if mode == "gamma" else None
        cfg = T._config()
        mine = T.ttnn_descriptor(x, out, weight=g, epsilon=1e-12, compute_kernel_config=cfg, program_config=T._PC_NONE)
        cbs = sorted(cb.format_descriptors[0].buffer_index for cb in mine.cbs)
        print(f"RESULT {label:22s} cbs={cbs}", flush=True)
        print(f"RESULT {label:22s} compute={list(mine.kernels[2].compile_time_args)}", flush=True)
finally:
    ttnn.close_device(dev)
