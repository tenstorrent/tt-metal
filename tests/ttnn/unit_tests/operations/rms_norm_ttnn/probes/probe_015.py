import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as seedpd
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as newpd
from eval.sharding import shard_config

device = ttnn.open_device(device_id=0)


def build(shape, shard, ml, dtype=ttnn.bfloat16):
    mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mc)
    W = shape[-1]
    g = ttnn.from_torch(
        torch.zeros(1, 1, 1, W, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    a = seedpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-12, compute_kernel_config=cfg)
    b = newpd.create_program_descriptor(
        x, out, weight=g, epsilon=1e-12, compute_kernel_config=cfg, program_config=newpd._PC_NONE
    )
    print(f"--- {shape} {shard} {str(ml).split('.')[-1]} ---")
    for name, i in (("reader", 0), ("writer", 1), ("compute", 2)):
        sa = list(a.kernels[i].compile_time_args)
        sb = list(b.kernels[i].compile_time_args)
        same = sb[: len(sa)] == sa if name == "writer" else None
        print(f"  {name}: seed n={len(sa)} new n={len(sb)}")
        if name == "compute":
            print(f"    seed: {sa}")
            print(f"    new : {sb}")
        elif name == "writer":
            print(f"    identical={sa==sb}")
    ca = {cb.format_descriptors[0].buffer_index: (cb.total_size, cb.format_descriptors[0].page_size) for cb in a.cbs}
    cbb = {cb.format_descriptors[0].buffer_index: (cb.total_size, cb.format_descriptors[0].page_size) for cb in b.cbs}
    print(f"  CBs identical={ca==cbb}  seed={sorted(ca.items())}")
    if ca != cbb:
        print(f"                       new ={sorted(cbb.items())}")


build((1, 1, 32, 7168), ([32, 256], (7, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED)
build((1, 1, 32, 1024), ([32, 128], (8, 1)), ttnn.TensorMemoryLayout.WIDTH_SHARDED)
build((1, 1, 32, 5120), ([32, 160], (8, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED)
ttnn.close_device(device)
