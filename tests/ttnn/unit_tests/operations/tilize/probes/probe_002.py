import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 128).bfloat16()

    def run():
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return tilize(x, use_multicore=False)

    o1 = run()
    after1 = dev.num_program_cache_entries()
    o2 = run()
    after2 = dev.num_program_cache_entries()
    ok1 = torch.equal(ttnn.to_torch(o1), t)
    ok2 = torch.equal(ttnn.to_torch(o2), t)
    print(
        f"PROBE cache entries after 1st call={after1} after 2nd={after2} "
        f"hit={after1 == after2} identity_1={ok1} identity_2={ok2}"
    )
finally:
    ttnn.close_device(dev)
