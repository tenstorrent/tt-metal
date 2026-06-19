import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from tests.ttnn.unit_tests.operations.rms_norm.test_rms_norm_perf import measure_device_kernel_ns

dev = device
grid = dev.compute_with_storage_grid_size()
print("GRID", grid.x, grid.y, "total", grid.x * grid.y)

# Inspect _select_k routing for candidate shapes (num_row_groups, Wt)
cfg = desc._resolve_compute_config(None)
fp32 = True


def route(Wt, nrg):
    K = desc._select_k(Wt, nrg, grid, grid.x * grid.y, False, ttnn.bfloat16, fp32, ttnn.bfloat16)
    return K


for label, Wt, nrg in [
    ("(1,1,32,8192)", 256, 1),
    ("(1,1,64,8192)", 256, 2),
    ("(256,8192)", 256, 8),
    ("(512,8192)", 256, 16),
    ("(1,1,32,16384)", 512, 1),
    ("(256,16384)", 512, 8),
]:
    K = route(Wt, nrg)
    used = nrg * K if K else None
    print(f"{label}: Wt={Wt} nrg={nrg} -> K={K} used_cores={used}")

# Measure device time for candidate multi-row-group B shapes (TILE bf16 no-gamma)
print("=== device time (us, best of 7) ===")
for shape in [(1, 1, 32, 8192), (1, 1, 64, 8192), (1, 256, 8192), (1, 512, 8192), (1, 1, 32, 16384), (1, 256, 16384)]:
    x = torch.randn(*shape)
    t = measure_device_kernel_ns(dev, x, None, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    print(f"{shape}: {t/1000:.2f} us" if t else f"{shape}: None")
