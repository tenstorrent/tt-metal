import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from ttnn.operations.rms_norm import rms_norm

captured = []
_orig = pdmod.ttnn.KernelDescriptor


def spy(**kw):
    if "compute" in str(kw.get("kernel_source", "")):
        captured.append(list(kw.get("compile_time_args", [])))
    return _orig(**kw)


pdmod.ttnn.KernelDescriptor = spy
device = ttnn.open_device(device_id=0)
try:
    cands = [
        ((1, 1, 3552, 3072), ttnn.TILE_LAYOUT),
        ((1, 1, 3552, 3072 - 24), ttnn.TILE_LAYOUT),
        ((1, 1, 64, 4096), ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 64, 4000), ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 32, 4096), ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 128, 6144), ttnn.ROW_MAJOR_LAYOUT),
    ]
    for shape, lay in cands:
        W = shape[-1]
        x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=lay, device=device)
        g = ttnn.from_torch(
            torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=lay, device=device
        )
        captured.clear()
        out = rms_norm(x, gamma=g)
        ct = captured[-1]
        wtc, nwc, br, xres = ct[1], ct[2], ct[3], ct[15]
        regime = "RESIDENT" if nwc == 1 else ("ROW_RESIDENT" if xres else "STREAM")
        print(
            f"SHAPE {shape} lay={'T' if lay==ttnn.TILE_LAYOUT else 'RM'} Wt={(W+31)//32} {regime} wtc={wtc} nwc={nwc} br={br}"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
