import ttnn, torch, json
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
cases = {
    "vae_dram_1048576x128": ((1, 1, 1048576, 128), 32),
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32),
    "vae_dram_262144x256": ((1, 1, 262144, 256), 32),
    "vae_dram_262144x512": ((1, 1, 262144, 512), 32),
    "vae_dram_1048576x256": ((1, 1, 1048576, 256), 32),
    "refiner_dram_16384x1152": ((1, 1, 16384, 1152), 32),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16),
    "flux1_dram_1048576x64": ((1, 1, 1048576, 64), 16),
    "mochi_dram_8x407040x128": ((8, 1, 407040, 128), 32),
}
meas = json.load(open("ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/_round2/baseline_full.json"))
for name, (shape, G) in cases.items():
    x = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT), dev)
    p = pd.plan_or_raise(x, G, has_gamma=True, has_beta=True)
    X = shape[0] * shape[2] * shape[3] * 2
    # per-core traffic: rows_core tiles rows * W; reads B chunks + (B-K) re-reads; writes B
    tot_r = tot_w = 0
    for wk in p.work:
        B = -(-wk.rows_core // p.h)
        K = min(p.K_alloc, B)
        rows_reread = max(0, wk.rows_core - K * p.h) if K < B else 0
        tot_r += (wk.rows_core + rows_reread) * wk.num_blocks * p.W * 2048
        tot_w += wk.rows_core * wk.num_blocks * p.W * 2048
    ns = meas[name]
    print(
        f"{name:26s} gx={p.gx} W={p.W} U={p.U} cpi={p.cpi} cores={len(p.work)} h={p.h} T={p.T} B={p.B_max} K={p.K_alloc} "
        f"rd={tot_r/1e6:7.1f}MB wr={tot_w/1e6:7.1f}MB  {(tot_r+tot_w)/ns:5.1f} GB/s  floor@471={((tot_r+tot_w)/471)/1e3:8.0f}us meas={ns/1e3:8.0f}us"
    )
    x.deallocate()
ttnn.close_device(dev)
