import ttnn, torch
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

c = _FabricMeshCache()
mesh = c.get()
print("mesh", tuple(mesh.shape))
view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
print("total", view.total_bytes_per_bank, "largest", view.largest_contiguous_bytes_free_per_bank)
for b in view.block_table[:20]:
    print(dict(b))
print("budget", pd.l1_budget(mesh))
x = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 131072, 3584]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
)
sz = pd._sizes(x, x, False, 2)
sz = pd.Sizes(T_in=2048, T_out=2048, T_g=2048, T_i=2048, has_gamma=True, G=2)
print(sz)
grid = mesh.compute_with_storage_grid_size()
print("grid", grid)
for L in (1, 2):
    ports = pd.port_cores(grid, 2 * L)
    pset = {(p.x, p.y) for p in ports}
    workers = [ttnn.CoreCoord(xx, yy) for yy in range(grid.y) for xx in range(grid.x) if (xx, yy) not in pset]
    print("L", L, "workers", len(workers))
    for S in (112, 56, 28, 16, 8, 5):
        G_w = -(-112 // S)
        G_h = min(4096, len(workers) // G_w)
        rm = -(-4096 // G_h)
        b = pd._wave_plan("D", False, S, G_w, G_h, rm, False, False)
        b12 = pd.rewave(b, pd.WAVES_MAX_PASSES)
        P = b12.G_h * b12.owners
        n_pub = -(-P // L)
        seg = pd._align_up(n_pub * b12.q * 128, 64)
        print(S, G_w, G_h, rm, "nw", b12.num_waves, "q_w", b12.q_w, "q", b12.q, "landing", b12.num_waves * 2 * seg)
c.close()
