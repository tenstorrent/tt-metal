import ttnn, torch
from ttnn.operations.rms_norm import _bench_rms_norm as b
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd

CASES = {
    "w_nonalign(1,1,32,4095)": (1, 1, 32, 4095),
    "w_nonalign(1,1,32,3071)": (1, 1, 32, 3071),
    "focus(1,1,32,7168)": (1, 1, 32, 7168),
}
ARMS = {
    "default": None,
    "OFF:double_buffer": dict(double_buffer=0),
    "OFF:coarse_chunk": dict(coarse_chunk=0),
    "wt_block=32": dict(wt_block=32),
    "OFF:w_split": dict(w_split=0),
}
device = ttnn.open_device(device_id=0)
try:
    for label, shape in CASES.items():
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        g = ttnn.from_torch(
            torch.randn((1, 1, 1, shape[-1])), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        print(f"--- {label}")
        for arm, levers in ARMS.items():
            cfg = opd._apply_precision_levers(b._cfg_loose(), levers)
            p = opd.blocking_plan(x, g, x, device, cfg, levers)
            print(
                f"   {arm:20s} G={p.group_size:3d} reg={p.regime} Wtc={p.Wt_core:4d} wr={p.WT_REDUCE_BLOCK:4d} "
                f"ws={p.WT_SCALE_BLOCK:4d} bht={p.BLOCK_HT} in={p.IN_BUF_DEPTH} out={p.OUT_BUF_DEPTH} "
                f"rm={p.RM_BUF_DEPTH} gd={p.GAMMA_DEPTH} L1={p.working_set_bytes()}"
            )
        ttnn.deallocate(x)
        ttnn.deallocate(g)
finally:
    ttnn.close_device(device)
