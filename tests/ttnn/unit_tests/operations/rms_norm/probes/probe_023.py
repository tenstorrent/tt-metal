import ttnn
from eval.golden_tests.rms_norm.helpers import run_rms_norm
from eval.oom import OOMError

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
bf16, fp32 = ttnn.bfloat16, ttnn.float32

# The large / wide-W INPUTS shapes that stress L1 (resident shard) + core count.
shapes = [
    (1, 1, 2048, 256),  # 64 tile-rows -> 64 cores
    (4, 1, 512, 512),  # 64 tile-rows
    (1, 1, 32, 8192),  # 1 tile-row, W=256 tiles resident (~0.5MB bf16 / 1MB fp32)
    (1, 1, 128, 4096),  # 4 tile-rows, W=128 tiles
    (128, 8192),  # 4 tile-rows, W=256 tiles
    (1024, 1024),  # 32 tile-rows, W=32 tiles
    (1, 32, 8192),  # 3D, 1 tile-row, wide
    (2, 512, 1024),  # 3D, 32 tile-rows
]
combos = [("bf16-TILE-g", bf16, TILE, bf16), ("fp32-TILE-g", fp32, TILE, fp32), ("bf16-RM-g", bf16, RM, bf16)]

device = ttnn.open_device(device_id=0)
results = []
try:
    for shape in shapes:
        for label, dt, lay, gdt in combos:
            glay = TILE if lay == TILE else RM
            axes = dict(
                dtype=dt,
                layout=lay,
                gamma_mode="gamma",
                gamma_dtype=gdt,
                gamma_layout=glay,
                memory_layout=HS,
                fp32_dest_acc_en=True,
            )
            try:
                run_rms_norm((shape,), device=device, **axes)
                results.append((shape, label, "PASS", ""))
            except OOMError as e:
                results.append((shape, label, "SKIP-OOM", str(e)[:60]))
            except Exception as e:
                results.append((shape, label, "FAIL", f"{type(e).__name__}: {str(e)[:70]}"))
finally:
    ttnn.close_device(device)

print("\n===== wide/large HEIGHT_SHARDED =====")
for shape, label, status, msg in results:
    print(f"  {str(shape):18s} {label:12s} {status}  {msg}")
npass = sum(1 for r in results if r[2] == "PASS")
print(f"\nPASS {npass} / {len(results)} (SKIP-OOM is a harness-infeasible cell, not a fail)")
