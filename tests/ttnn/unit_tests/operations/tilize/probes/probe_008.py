import torch, ttnn
from ttnn.operations.tilize import SUPPORTED, tilize

dev = ttnn.open_device(device_id=0)
try:
    SUPPORTED["buffer"] = ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"]
    SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8]
    SUPPORTED["output_dtype"] = [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.uint32,
        ttnn.int32,
        ttnn.uint16,
        ttnn.uint8,
    ]
    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    shape = (1, 1, 64, 128)
    for inmc, outmc, name in [(DR, L1, "dram_to_l1"), (L1, L1, "l1_to_l1"), (L1, DR, "l1_to_dram")]:
        try:
            torch.manual_seed(3)
            ti = torch.randn(shape, dtype=torch.float32).bfloat16()
            tt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=inmc)
            out = tilize(tt, outmc)
            print(
                f"PROBE buffer {name}: exact={torch.equal(ttnn.to_torch(out).float(), ti.float())} buf={out.memory_config().buffer_type}"
            )
        except Exception as e:
            print(f"PROBE buffer {name}: RAISED {type(e).__name__}: {str(e)[:140]}")

    for d, od, name in [
        (ttnn.float32, None, "fp32->fp32"),
        (ttnn.float32, ttnn.bfloat16, "fp32->bf16"),
        (ttnn.bfloat16, ttnn.float32, "bf16->fp32"),
        (ttnn.bfloat16, ttnn.bfloat8_b, "bf16->bf8b"),
        (ttnn.bfloat16, ttnn.bfloat4_b, "bf16->bf4b"),
        (ttnn.uint32, None, "uint32->uint32"),
        (ttnn.int32, None, "int32->int32"),
        (ttnn.uint16, None, "uint16->uint16"),
        (ttnn.uint8, None, "uint8->uint8"),
    ]:
        try:
            torch.manual_seed(3)
            if d in (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8):
                ti = torch.randint(0, 100, shape, dtype=torch.int32)
            else:
                ti = torch.randn(shape, dtype=torch.float32)
                if d == ttnn.bfloat16:
                    ti = ti.bfloat16().float()
            tt = ttnn.from_torch(ti, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DR)
            out = tilize(tt, dtype=od) if od is not None else tilize(tt)
            got = ttnn.to_torch(out).float()
            err = float((got - ti.float()).abs().max())
            print(f"PROBE dtype {name}: out_dtype={out.dtype} max_abs_err={err:.6g}")
        except Exception as e:
            print(f"PROBE dtype {name}: RAISED {type(e).__name__}: {str(e)[:140]}")
finally:
    ttnn.close_device(dev)
