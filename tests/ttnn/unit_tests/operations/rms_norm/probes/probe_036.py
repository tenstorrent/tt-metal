import ttnn, math

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
nc = g.x * g.y
print("COMPUTE_GRID", g.x, g.y, "num_cores", nc)


def pick(Wt, ncores):
    per = -(-Wt // ncores)
    K = -(-Wt // per)
    return per, K


print("=== decode interleaved (logical W-split) ===")
for W in [1024, 2304, 5120, 7168]:
    Wt = math.ceil(W / 32)
    per, K = pick(Wt, nc)
    print(f"W={W} Wt={Wt} per_w_t={per} K={K}")
print("=== WIDTH/BLOCK sharded perf geometries ===")
for W, sh, sw, gx, gy in [
    (1024, 32, 128, 8, 1),
    (2304, 32, 256, 9, 1),
    (5120, 32, 160, 8, 4),
    (7168, 32, 256, 7, 4),
    (1024, 1024, 128, 8, 8),
]:
    Wt = math.ceil(W / 32)
    per_w_t = sw // 32
    K = gx  # WIDTH: K=all cores; BLOCK: K=cols
    print(f"W={W} shard[{sh},{sw}] grid({gx},{gy}) Wt={Wt} per_w_t={per_w_t} cores={gx*gy}")
ttnn.close_device(device)
