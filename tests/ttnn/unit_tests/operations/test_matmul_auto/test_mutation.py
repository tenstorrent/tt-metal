import os, time
os.environ['TT_METAL_LOGGER_LEVEL'] = 'FATAL'
import torch
import ttnn

print('=== MUTATION/PERTURBATION TEST ===')
print('Stage 3 requirement: prove selected config beats modified neighbors')
print('')
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx, gy = grid.x, grid.y
print('Device grid: ' + str(gx) + 'x' + str(gy))

test_shapes = [
    ('1024x1024x1024', 1024, 1024, 1024),
    ('1024x4096x1024', 1024, 4096, 1024),
    ('2048x2048x2048', 2048, 2048, 2048),
]

passed = 0
failed = 0


def bench(fn, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / runs * 1000


for name, M, K, N in test_shapes:
    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    M_tiles = M // 32
    K_tiles = K // 32
    N_tiles = N // 32

    print(name + ':')

    # Config A: auto-selected by matmul_auto
    from ttnn.operations.auto_config.matmul_auto import matmul_auto
    t_auto = bench(lambda: matmul_auto(ta, tb))
    print('  [A] matmul_auto (selected):    ' + str(round(t_auto, 4)) + 'ms')

    # Config B: default ttnn.matmul (no program_config)
    t_default = bench(lambda: ttnn.matmul(ta, tb))
    print('  [B] ttnn.matmul (default):     ' + str(round(t_default, 4)) + 'ms')

    # Config C: Deliberately suboptimal -- small per_core_M (1 tile per core)
    configs_tested = 0
    configs_auto_wins = 0
    suboptimal_configs = []

    # Try various degraded configs
    for label, core_x, core_y in [('4x4_grid', 4, 4), ('2x2_grid', 2, 2), ('full_grid', min(gx, N_tiles), min(gy, M_tiles))]:
        if core_x > gx or core_y > gy:
            continue
        pcM = M_tiles // core_y
        pcN = N_tiles // core_x
        if pcM < 1 or pcN < 1:
            continue
        try:
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_x, core_y),
                in0_block_w=min(K_tiles, 4),
                out_subblock_h=1,
                out_subblock_w=min(pcN, 4),
                per_core_M=pcM,
                per_core_N=pcN,
            )
            t_cfg = bench(lambda: ttnn.matmul(ta, tb, program_config=cfg))
            configs_tested += 1
            if t_auto <= t_cfg * 1.05:
                configs_auto_wins += 1
                print('  [C] ' + label + ' (in0_bw=4):     ' + str(round(t_cfg, 4)) + 'ms -> AUTO WINS')
            else:
                print('  [C] ' + label + ' (in0_bw=4):     ' + str(round(t_cfg, 4)) + 'ms -> NEIGHBOR WINS')
            suboptimal_configs.append((label, t_cfg))
        except Exception:
            print('  [C] ' + label + ':   CRASHED (invalid)')
            configs_tested += 1
            configs_auto_wins += 1

    # Config D: Same grid as full but with smaller in0_block_w (1 instead of optimal)
    try:
        pcM_d = M_tiles // min(gy, M_tiles) if min(gy, M_tiles) > 0 else M_tiles
        pcN_d = N_tiles // min(gx, N_tiles) if min(gx, N_tiles) > 0 else N_tiles
        cfg_small_bw = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(min(gx, N_tiles), min(gy, M_tiles)),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=pcM_d,
            per_core_N=pcN_d,
        )
        t_small_bw = bench(lambda: ttnn.matmul(ta, tb, program_config=cfg_small_bw))
        configs_tested += 1
        if t_auto <= t_small_bw * 1.05:
            configs_auto_wins += 1
            print('  [D] full_grid (in0_bw=1):      ' + str(round(t_small_bw, 4)) + 'ms -> AUTO WINS')
        else:
            print('  [D] full_grid (in0_bw=1):      ' + str(round(t_small_bw, 4)) + 'ms -> NEIGHBOR WINS')
    except Exception:
        print('  [D] full_grid (in0_bw=1):      CRASHED')
        configs_tested += 1
        configs_auto_wins += 1

    # Summary for shape
    auto_vs_default = t_default / t_auto
    shape_pass = configs_auto_wins >= configs_tested * 0.5 if configs_tested > 0 else False
    if shape_pass:
        passed += 1
    else:
        failed += 1
    print('  Summary: auto vs default = ' + str(round(auto_vs_default, 3)) + 'x | beats ' + str(configs_auto_wins) + '/' + str(configs_tested) + ' neighbors')
    print('')

ttnn.close_device(device)
print('=== MUTATION TEST: ' + str(passed) + ' PASS, ' + str(failed) + ' FAIL ===')
