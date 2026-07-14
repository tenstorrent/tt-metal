import ttnn, pytest
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
bf16, fp32 = ttnn.bfloat16, ttnn.float32
Skipped = pytest.skip.Exception

shapes = [
    (1, 1, 2048, 256),
    (4, 1, 512, 512),
    (1, 1, 32, 8192),
    (1, 1, 128, 4096),
    (128, 8192),
    (1024, 1024),
    (1, 32, 8192),
    (2, 512, 1024),
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
            except Skipped as e:
                results.append((shape, label, "SKIP", str(e)[:50]))
            except Exception as e:
                results.append((shape, label, "FAIL", f"{type(e).__name__}: {str(e)[:70]}"))
finally:
    ttnn.close_device(device)

print("\n===== wide/large HEIGHT_SHARDED =====")
for shape, label, status, msg in results:
    print(f"  {str(shape):18s} {label:12s} {status}  {msg}")
npass = sum(1 for r in results if r[2] == "PASS")
nfail = sum(1 for r in results if r[2] == "FAIL")
print(f"\nPASS {npass} / {len(results)}  (FAIL {nfail}; SKIP = harness-infeasible L1, not charged)")
