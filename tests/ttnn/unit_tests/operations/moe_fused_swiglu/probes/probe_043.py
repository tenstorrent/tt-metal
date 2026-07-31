import ttnn
import torch

from ttnn.operations.moe_fused_swiglu.perf_experiments.root_epilogue_fusion.program_descriptor_with_inline_kernels import (
    VARIANTS,
    create_sharded_memory_config,
    run_op,
)

TILE = 32

device = ttnn.open_device(device_id=0)
try:
    m_eff, hn_pad = 8, 6
    m, n = m_eff * TILE, hn_pad * TILE
    mem_cfg = create_sharded_memory_config((m, n))

    def _dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg)

    gate_acc = _dev(torch.zeros(m, n))
    up_acc = _dev(torch.zeros(m, n))
    reduce_gate_in = _dev(torch.zeros(m, n))
    reduce_up_in = _dev(torch.zeros(m, n))
    inputs = [gate_acc, up_acc, reduce_gate_in, reduce_up_in]

    for variant in VARIANTS:
        print(f"RUNNING variant={variant}", flush=True)
        out = run_op(inputs, m_eff=m_eff, hn_pad=hn_pad, variant=variant, kernel_iters=1)
        ttnn.synchronize_device(device)
        print(f"DONE variant={variant}", flush=True)
    print("ALL VARIANTS OK")
finally:
    ttnn.close_device(device)
