import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor as seed_d
from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import create_program_descriptor as my_d, _PC_NONE
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 256, 512)
    mc = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    print("shard spec:", mc)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    s = seed_d(x, out, gamma=None, epsilon=1e-12, compute_kernel_config=cfg)
    m = my_d(x, out, weight=None, epsilon=1e-12, compute_kernel_config=cfg, program_config=_PC_NONE)

    def sig(d):
        return {c.format_descriptors[0].buffer_index: (c.total_size, c.format_descriptors[0].page_size) for c in d.cbs}

    ss, ms = sig(s), sig(m)
    for k in sorted(set(ss) | set(ms)):
        if ss.get(k) != ms.get(k):
            print("DIFF cb", k, "seed", ss.get(k), "mine", ms.get(k))
    for G in range(2, 40):
        t = PD._combine_tree_arity(G, 1)
        if t:
            print("tree", G, t)
finally:
    ttnn.close_device(device)
