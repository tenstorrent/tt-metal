import ttnn
from ttnn.operations.rms_norm import _bench_rms_norm as b
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd

device = ttnn.open_device(device_id=0)
try:
    print("BUDGET", ttnn.get_max_worker_l1_unreserved_size() - opd.L1_RESERVED_BYTES)
    rows = []
    for name, config, dtype, gamma in b.GATESET:
        shape, shape_dtype, layout = b.BENCH_SHAPES[name]
        dt = dtype or shape_dtype
        x, g = b._make(device, shape, dt, layout, b.BENCH_GAMMA_LAYOUT.get(name))
        if not gamma:
            g = None
        cfg = b.BENCH_CONFIGS[config]()
        cfg = opd._apply_precision_levers(cfg, None)
        p = opd.blocking_plan(x, g, x, device, cfg, None)
        gpages = [n for i, n, _, _ in p.cb_layout if i == opd.CB_GAMMA_TILES]
        gdepth = (gpages[0] // p.WT_SCALE_BLOCK) if gpages else 0
        inpages = [n for i, n, _, _ in p.cb_layout if i == opd.CB_INPUT_TILES]
        tag = f"{name}/{config}" + (f"/{str(dt).split('.')[-1]}" if dtype else "") + ("" if gamma else "/no_gamma")
        rows.append(
            f"{tag:36s} G={p.group_size:3d} reg={p.regime} Wt={p.Wt:4d} Wtc={p.Wt_core:4d} "
            f"wr={p.WT_REDUCE_BLOCK:4d} ws={p.WT_SCALE_BLOCK:4d} bht={p.BLOCK_HT} "
            f"in={p.IN_BUF_DEPTH} out={p.OUT_BUF_DEPTH} rm={p.RM_BUF_DEPTH} gd={gdepth} "
            f"inpg={inpages[0] if inpages else 0} nrb={p.num_row_blocks} gu={p.groups_used} "
            f"L1={p.working_set_bytes()}"
        )
        ttnn.deallocate(x)
        if g is not None:
            ttnn.deallocate(g)
    print("=== PLANS ===")
    for r in rows:
        print(r)
finally:
    ttnn.close_device(device)
