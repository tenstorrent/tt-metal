"""Refinement 5, first device pass: the 18 legal (dtype, output_dtype) pairs,
with the uint8 EXCLUSIONS lifted so the SrcB ALU-format repair can be judged."""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")

device = ttnn.open_device(device_id=0)
device.disable_and_clear_program_cache()

M.EXCLUSIONS = [e for e in M.EXCLUSIONS if e not in M._UINT8_EXCLUSIONS]

DT = [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8]
OD = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8]
INT = (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8)
FLOAT_OUT = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b)
WIDTH = {ttnn.uint32: 4, ttnn.int32: 4, ttnn.uint16: 2, ttnn.uint8: 1}


def legal(i, o):
    return (o in INT and WIDTH[i] == WIDTH[o]) if i in INT else (o in FLOAT_OUT)


TORCH = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
    ttnn.bfloat4_b: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint8: torch.uint8,
}


def mk(dt, shape):
    if dt == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dt in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    if dt == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dt == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if torch.equal(a, b):
        return 1.0
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va * vb).sum() / d) if d > 0 else 1.0


torch.manual_seed(0)
for shape in [(1, 1, 64, 128), (2, 3, 64, 96)]:
    print(f"===== shape {shape} =====", flush=True)
    for i in DT:
        for o in OD:
            if not legal(i, o):
                continue
            try:
                x = mk(i, shape)
                tt = ttnn.from_torch(
                    x, dtype=i, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                out = M.tilize(tt, dtype=o)
                got = ttnn.to_torch(out).to(torch.float64)
                exp = x.to(TORCH[o]).to(torch.float64)
                nm = int((got != exp).sum())
                print(
                    f"  {str(i)[9:]:10s} -> {str(o)[9:]:10s} exact={nm == 0} pcc={pcc(got, exp):.7f} "
                    f"max_abs={float((got - exp).abs().max()):.6g}",
                    flush=True,
                )
            except Exception as e:
                print(f"  {str(i)[9:]:10s} -> {str(o)[9:]:10s} EXC {type(e).__name__}: {str(e)[:220]}", flush=True)
ttnn.close_device(device)
