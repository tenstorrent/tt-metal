import ttnn
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
bf16, fp32, bf8b = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b

shapes = {
    "tile_aligned": (1, 1, 64, 128),
    "w_non_aligned": (1, 1, 64, 17),
    "h_non_aligned": (1, 1, 50, 128),
    "multi_image": (2, 4, 128, 512),
    "3d": (4, 128, 512),
    "2d": (128, 512),
}

# (label, dtype, layout, gamma_mode, gamma_dtype, gamma_layout, fp32_acc)
combos = [
    ("bf16-TILE-g", bf16, TILE, "gamma", bf16, TILE, True),
    ("bf16-TILE-ng", bf16, TILE, "no_gamma", "none", "none", True),
    ("bf16-RM-g", bf16, RM, "gamma", bf16, RM, True),
    ("bf16-RM-ng", bf16, RM, "no_gamma", "none", "none", True),
    ("fp32-TILE-g", fp32, TILE, "gamma", fp32, TILE, True),
    ("fp32-RM-g", fp32, RM, "gamma", fp32, RM, True),
    ("bf8b-TILE-g", bf8b, TILE, "gamma", bf8b, TILE, True),
    ("bf16-T-False", bf16, TILE, "gamma", bf16, TILE, False),
]

device = ttnn.open_device(device_id=0)
results = []
try:
    for sname, shape in shapes.items():
        for label, dt, lay, gm, gdt, glay, acc in combos:
            axes = dict(
                dtype=dt,
                layout=lay,
                gamma_mode=gm,
                gamma_dtype=gdt,
                gamma_layout=glay,
                memory_layout=HS,
                fp32_dest_acc_en=acc,
            )
            try:
                run_rms_norm((shape,), device=device, **axes)
                results.append((sname, label, "PASS", ""))
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:90]}"
                results.append((sname, label, "FAIL", msg))
finally:
    ttnn.close_device(device)

print("\n===== HEIGHT_SHARDED cartesian landscape =====")
for sname, label, status, msg in results:
    print(f"  {sname:14s} {label:14s} {status}  {msg}")
npass = sum(1 for r in results if r[2] == "PASS")
print(f"\nTOTAL: {npass}/{len(results)} pass")
