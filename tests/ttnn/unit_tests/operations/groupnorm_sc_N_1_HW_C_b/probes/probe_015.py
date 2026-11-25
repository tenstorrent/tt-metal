# R2: in-place plans with a clean L1 per case (the returned tensor aliases the input: free it before the next case).
import gc, torch, ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, pytorch_groupnorm_sc_N_1_HW_C
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b import feature_spec as fs
from eval.metrics import compute_metrics

device = ttnn.open_device(device_id=0)
try:
    print(f"FREE budget_empty={pd.l1_cb_budget(device)}")
    for name, (shape, G, grid), layout, in_place, eps in [c for c in fs._PERF_CASES if len(c[1]) == 3 and c[3]][-2:]:
        torch.manual_seed(0)
        C = shape[-1]
        xt = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        exp = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
        mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
        p = pd.plan_or_raise(x, G, has_gamma=True, has_beta=True, in_place=True)
        print(
            f"PLAN {name} W={p.W} h={p.h} T={p.T} rows_core={p.rows_core_max} B={p.B_max} K_max={p.K_max} K={p.K_alloc} fp={p.footprint} budget={p.l1_budget} margin={p.l1_budget - p.footprint}"
        )
        addr = x.buffer_address()
        y = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=True)
        m = compute_metrics(y, exp)
        print(f"ACC {name} pcc={m.pcc:.6f} rms={m.rms:.5f} same_addr={y.buffer_address() == addr}")
        ttnn.deallocate(y, force=True)
        del x, y
        gc.collect()
        print(f"FREE after {name}: budget={pd.l1_cb_budget(device)}")
finally:
    ttnn.close_device(device)
