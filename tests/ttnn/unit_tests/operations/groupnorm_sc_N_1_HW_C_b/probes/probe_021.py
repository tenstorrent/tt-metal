import ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
for name, shape, G in [
    ("vae_dram_1048576x256", (1, 1, 1048576, 256), 32),
    ("mochi", (8, 1, 407040, 128), 32),
    ("vae_dram_1048576x128", (1, 1, 1048576, 128), 32),
    ("vae_dram_262144x512", (1, 1, 262144, 512), 32),
]:
    x = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT), dev)
    p = pd.plan_or_raise(x, G, has_gamma=True, has_beta=True)
    T = p.T
    W = p.W
    R = p.R
    f_fixed = pd.footprint_bytes(W=W, T=T, R=R, K_alloc=0, stream=True, is_rm=False, has_gamma=True, has_beta=True)
    ring_stream = pd.D_IN * T * 2048
    out_ring = pd.output_ring_bytes(T)
    print(
        f"{name}: l1_budget={p.l1_budget} f_fixed={f_fixed} (stream {ring_stream}, out ring {out_ring}, rest {f_fixed-ring_stream-out_ring}) chunk={T*2048} K_max={p.K_max} K={p.K_alloc} B={p.B_max} footprint={p.footprint}"
    )
    x.deallocate()
ttnn.close_device(dev)
