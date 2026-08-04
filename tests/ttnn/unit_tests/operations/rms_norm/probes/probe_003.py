"""Introspect the derived knobs: grid fill, BLOCK_ROWS, WT_CHUNK, regime."""
import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print(
        f"grid = {grid.x}x{grid.y} = {grid.x*grid.y} cores; L1 budget = {int(ttnn.get_max_worker_l1_unreserved_size()*pd.L1_SAFETY_FRACTION)}"
    )
    print(
        f"{'shape':22} {'lay':4} {'dt':5} {'g':3} {'Rt':>5} {'cores':>5} {'BLK':>4} {'Wt':>5} {'WTC':>5} {'nchk':>5} regime"
    )

    real_pd = pd.create_program_descriptor
    cap = {}

    def spy(inp, out, *, gamma=None, epsilon=1e-6, compute_kernel_config=None):
        # re-derive by calling the real factory then reading back what it chose
        return real_pd(inp, out, gamma=gamma, epsilon=epsilon, compute_kernel_config=compute_kernel_config)

    def knobs(shape, layout, dt, with_gamma):
        W = shape[-1]
        Wt = (W + 31) // 32
        n = 1
        if layout == ttnn.TILE_LAYOUT:
            for d in shape[:-2]:
                n *= d
            Rt = n * ((shape[-2] + 31) // 32)
        else:
            for d in shape[:-1]:
                n *= d
            Rt = (n + 31) // 32
        is_tile = layout == ttnn.TILE_LAYOUT
        bt = ttnn.tile_size(dt)
        gt = ttnn.tile_size(dt) if with_gamma else 0
        st = ttnn.tile_size(ttnn.bfloat16)
        ft = ttnn.tile_size(ttnn.float32)
        cbx = 2 if is_tile else 1
        cbo = 2 if is_tile else 1
        mult = cbx + 1 + (1 if with_gamma else 0) + cbo
        pw = W % 32
        sb = st * (2 if pw else 1)
        budget = int(ttnn.get_max_worker_l1_unreserved_size() * pd.L1_SAFETY_FRACTION)
        fixed = Wt * gt + 0 + (2 * pd.CB_RM_STAGE_DEPTH * Wt * bt if not is_tile else 0) + sb
        brmax = max(0, (budget - fixed) // (Wt * bt * mult + ft))
        nc, allc, g1, g2, r1, r2 = ttnn.split_work_to_cores(pd._core_range_set_full_grid(device), Rt, True)
        if brmax >= 1:
            return Rt, nc, min(max(r1, r2), brmax), Wt, Wt, 1, "RESIDENT"
        per = bt * mult + (gt * 1 if with_gamma else 0) + (2 * pd.CB_RM_STAGE_DEPTH * bt if not is_tile else 0)
        wtc = pd._largest_divisor_at_most(Wt, max(1, (budget - (sb + ft)) // per))
        return Rt, nc, 1, Wt, wtc, Wt // wtc, "STREAM"

    cases = [
        ((1, 1, 32, 32), 0),
        ((1, 1, 64, 128), 1),
        ((1, 1, 32, 256), 1),
        ((1, 1, 256, 32), 1),
        ((2, 4, 64, 128), 1),
        ((4, 128, 512), 1),
        ((128, 512), 1),
        ((1, 1, 64, 64), 1),
        ((1, 1, 128, 512), 1),
        ((1, 1, 64, 500), 1),
        ((1, 1, 32, 4096), 1),
        ((1, 1, 64, 4096), 1),
        ((1, 1, 32, 4000), 1),
        ((1, 1, 2048, 256), 1),
        ((8192, 1024), 1),
        ((1, 1, 224, 72), 1),
    ]
    for shape, wg in cases:
        for layout, ln in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "rm")):
            for dt, dn in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
                Rt, nc, blk, Wt, wtc, nchk, reg = knobs(shape, layout, dt, bool(wg))
                print(
                    f"{str(shape):22} {ln:4} {dn:5} {'g' if wg else '-':3} {Rt:5} {nc:5} {blk:4} {Wt:5} {wtc:5} {nchk:5} {reg}"
                )
finally:
    ttnn.close_device(device)
