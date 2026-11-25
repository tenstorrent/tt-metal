# R2: which sharded RM plans change K under the K_full / K_part residency accounting (old: K = min(B, K_part)).
import torch, ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b import tag_shard_fit
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b import feature_spec as fs

device = ttnn.open_device(device_id=0)
try:
    shapes = {i for i in fs.INPUTS if len(i) == 3} | {c[1] for c in fs._PERF_CASES if len(c[1]) == 3}
    n = changed = 0
    for inp in sorted(shapes):
        if tag_shard_fit(inp, {"layout": ttnn.ROW_MAJOR_LAYOUT}) != "even":
            continue
        shape, G, grid = inp
        mem = block_shard_memory_config(shape, grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mem,
        )
        for ip in (False, True):
            for aff in ((True, True), (True, False), (False, False)):
                try:
                    p = pd.plan_or_raise(x, G, has_gamma=aff[0], has_beta=aff[1], in_place=ip)
                except NotImplementedError as e:
                    print(f"REFUSE {shape} {grid} ip={ip} aff={aff}: {e}")
                    continue
                n += 1
                fp_kw = dict(
                    W=p.W,
                    T=p.T,
                    R=p.R,
                    is_rm=True,
                    has_gamma=aff[0],
                    has_beta=aff[1],
                    map_period=p.map_period,
                    sharded=True,
                )
                k_old = min(
                    p.B_max, (p.l1_budget - pd.footprint_bytes(K_alloc=0, stream=True, **fp_kw)) // (2048 * p.T)
                )
                tag = "CHG " if k_old != p.K_alloc else "PLAN"
                changed += k_old != p.K_alloc
                if k_old != p.K_alloc or aff == (True, True):
                    print(
                        f"{tag} {shape} {grid} ip={int(ip)} aff={int(aff[0])}{int(aff[1])} W={p.W} h={p.h} B={p.B_max} K_old={k_old} K={p.K_alloc} fp={p.footprint} budget={p.l1_budget}"
                    )
        x.deallocate()
    print(f"SUMMARY plans={n} changed={changed}")
finally:
    ttnn.close_device(device)
