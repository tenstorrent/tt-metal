import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import plan_or_raise, Formats

dev = ttnn.open_device(device_id=0)
try:
    for name, shape, G in [
        ("vae_dram_65536x512", (1, 1, 65536, 512), 32),
        ("refiner_dram_16384x1152", (1, 1, 16384, 1152), 32),
        ("flux1_dram_16384x256", (1, 1, 16384, 256), 16),
        ("vae_dram_1048576x128", (1, 1, 1048576, 128), 32),
    ]:
        for dt in (ttnn.bfloat16, ttnn.float32):
            spec = ttnn.TensorSpec(shape, dt, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
            t = ttnn.allocate_tensor_on_device(spec, dev)
            p = plan_or_raise(t, G, has_gamma=True, has_beta=True, fmt=Formats(in_dtype=dt, aff_dtype=dt))
            tag = "bf16" if dt == ttnn.bfloat16 else "f32 "
            print(
                f"{tag} {name}: gx={p.gx} W={p.W} h={p.h} T={p.T} B_max={p.B_max} K_max={p.K_max} K_alloc={p.K_alloc} stream={p.stream} out_w_chunks={p.out_writer_chunks} cores={len(p.work)} footprint={p.footprint} budget={p.l1_budget}"
            )
            t.deallocate()
finally:
    ttnn.close_device(dev)
