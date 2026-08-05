import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd
from ttnn.operations.rms_norm.rms_norm import rms_norm
from eval.sharding import shard_config

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 8192, 1024)
    W = 1024
    mc = shard_config(
        [1024, 128],
        (8, 8),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    g = ttnn.from_torch(
        torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    cap = {}
    orig = rpd.create_program_descriptor

    def spy(*a, **k):
        pd = orig(*a, **k)
        cap["op"] = (
            [list(kk.compile_time_args) for kk in pd.kernels],
            [
                (
                    int(cb.total_size),
                    sorted((c.x, c.y) for c in rpd._cores_in(cb.core_ranges))[:1],
                    len(rpd._cores_in(cb.core_ranges)),
                )
                for cb in pd.cbs
            ],
            [list(pd.kernels[i].runtime_args[0][0]) for i in range(3)],
            [str(kk.kernel_source) for kk in pd.kernels],
            [list(kk.defines) for kk in pd.kernels],
        )
        return pd

    rpd.create_program_descriptor = spy
    y = rms_norm(x, gamma=g, compute_kernel_config=cfg, memory_config=mc)
    rpd.create_program_descriptor = orig
    print("op eps/kwargs captured. out mc == in mc:", y.memory_config() == x.memory_config())

    pd2 = rpd.create_program_descriptor(x, y, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
    bench = (
        [list(kk.compile_time_args) for kk in pd2.kernels],
        [
            (
                int(cb.total_size),
                sorted((c.x, c.y) for c in rpd._cores_in(cb.core_ranges))[:1],
                len(rpd._cores_in(cb.core_ranges)),
            )
            for cb in pd2.cbs
        ],
        [list(pd2.kernels[i].runtime_args[0][0]) for i in range(3)],
        [str(kk.kernel_source) for kk in pd2.kernels],
        [list(kk.defines) for kk in pd2.kernels],
    )
    names = ["reader_ct", "writer_ct", "compute_ct"]
    for i in range(3):
        a, b = cap["op"][0][i], bench[0][i]
        print(f"{names[i]}: {'SAME' if a==b else 'DIFF'}")
        if a != b:
            print("   op   ", a)
            print("   bench", b)
    print("cbs SAME" if cap["op"][1] == bench[1] else "cbs DIFF")
    if cap["op"][1] != bench[1]:
        for u, v in zip(cap["op"][1], bench[1]):
            if u != v:
                print("   ", u, v)
    for i in range(3):
        if cap["op"][2][i] != bench[2][i]:
            print(f"rt[{i}]@(0,0) DIFF op={cap['op'][2][i]} bench={bench[2][i]}")
    print("defines op", cap["op"][4], "bench", bench[4])
finally:
    ttnn.close_device(device)
