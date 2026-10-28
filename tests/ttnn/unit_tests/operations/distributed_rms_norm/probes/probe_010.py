import ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import plan_summary

cache = _FabricMeshCache()
mesh = cache.get()
ML = ttnn.TensorMemoryLayout
for shape, ml, dt, acc in [
    ((1, 1, 32, 4096), ML.HEIGHT_SHARDED, ttnn.bfloat16, True),
    ((128, 8192), ML.HEIGHT_SHARDED, ttnn.bfloat8_b, True),
    ((1, 1, 32, 8192), ML.HEIGHT_SHARDED, ttnn.bfloat16, True),
    ((1, 1, 4096, 1024), ML.WIDTH_SHARDED, ttnn.bfloat16, True),
    ((1, 1, 4096, 1024), ML.WIDTH_SHARDED, ttnn.bfloat8_b, True),
    ((1, 1, 640, 1792), ML.INTERLEAVED, ttnn.bfloat16, True),
    ((1, 1, 2048, 1280), ML.INTERLEAVED, ttnn.bfloat16, True),
]:
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if ml == ML.INTERLEAVED
        else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=mesh)
    )
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, mesh, mc)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, mesh, mc)
    g = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, shape[-1]]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        print(
            "PROBE",
            shape,
            ml,
            dt,
            plan_summary(
                t,
                cluster_axis=1,
                gamma=g,
                num_links=1,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc
                ),
            ),
        )
    except Exception as e:
        print("PROBE", shape, ml, dt, "ERR", e)
    t.deallocate()
    o.deallocate()
    g.deallocate()
