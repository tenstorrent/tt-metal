import torch, ttnn
from eval.golden_tests.all_gather_rms_norm import helpers
from eval.golden_tests.all_gather_rms_norm.topology import FABRIC_CONFIG, system_mesh_shape
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm as opm

ttnn.set_fabric_config(FABRIC_CONFIG)
md = ttnn.open_mesh_device(ttnn.MeshShape(*system_mesh_shape()))


def st(tag):
    try:
        s = md.get_memory_allocation_statistics(ttnn.BufferType.L1)
        print("STAT", tag, s.total_allocated_bytes, s.largest_free_block_bytes)
    except Exception as e:
        print("ERR stat", e)


try:
    st("start")
    call, check = helpers.prepare_all_gather_rms_norm(
        ((128, 8192),),
        mesh_device=md,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        beta_mode="beta",
        norm_type="layer_norm",
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        cluster_axis=0,
        topology=ttnn.Topology.Ring,
        num_links=2,
    )
    st("prepared")
    orig = ttnn.generic_op

    def g(io, desc):
        for t in io:
            print("ADDR io", t.memory_config().buffer_type, ttnn.get_device_tensors(t)[0].buffer_address())
        st("before generic")
        return orig(io, desc)

    ttnn.generic_op = g
    try:
        call()
    except Exception as e:
        print("ERR", str(e).splitlines()[1][:200])
finally:
    ttnn.close_mesh_device(md)
