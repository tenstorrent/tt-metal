# Device time of the int32 scalar compares (ttnn.unary_chain) and of the int32 add and sub users, every case in one
# process under `python -m tracy -r -p --no-web-server -o <dir> prof_int.py <cases>`; a signpost names each case before
# its R3_NMEAS measured calls (2 warm-up calls first). R_BITS=<json>: instead, one call per case, outputs checked against
# torch and hashed.
import gc, hashlib, json, os, sys
import torch
import ttnn

try:
    from tracy import signpost
except ImportError:
    signpost = None

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "10"))
EUP = ttnn._ttnn.activation.EltwiseUnaryWithParam
IMIN, IMAX = -(2**31), 2**31 - 1


def ints(shape, seed, lo=IMIN, hi=IMAX):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)
    edge = torch.tensor([IMIN, IMIN + 1, -6, -5, -4, -1, 0, 1, 4, 5, 6, IMAX - 1, IMAX], dtype=torch.int64)
    x.view(-1)[: 4096] = edge[torch.arange(4096) % len(edge)]
    return x.to(torch.int32)


def dram(x, d):
    return ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=d)


def hs(x, d, tiles_h=8):
    grid = d.compute_with_storage_grid_size(); n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(crs, (32 * tiles_h, x.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem), mem


def hs_shape(d, tiles_h=8, w=256):
    g = d.compute_with_storage_grid_size()
    return (1, 1, 32 * tiles_h * g.x * g.y, w)


CMP = {"lt": "UNARY_LT", "gt": "UNARY_GT", "le": "UNARY_LE", "ge": "UNARY_GE", "eq": "UNARY_EQ", "ne": "UNARY_NE"}
TORCH = {"lt": torch.lt, "gt": torch.gt, "le": torch.le, "ge": torch.ge, "eq": torch.eq, "ne": torch.ne}
CASES = {}
for name, op in CMP.items():
    for s in (5, -5):
        CASES[f"cmp_{name}_{s}_dram"] = (
            lambda d: (dram(ints((1, 1, 1024, 1024), 7), d),),
            lambda d, x, op=op, s=s: ttnn.unary_chain(x, [EUP(getattr(ttnn.UnaryOpType, op), s)]),
            lambda x, name=name, s=s: TORCH[name](x, s).to(torch.int32))
    CASES[f"cmp_{name}_5_hs"] = (
        lambda d: hs(ints(hs_shape(d), 8), d),
        lambda d, x, m, op=op: ttnn.unary_chain(x, [EUP(getattr(ttnn.UnaryOpType, op), 5)], memory_config=m),
        lambda x, name=name: TORCH[name](x, 5).to(torch.int32))
SMALL = (-(2**30), 2**30)
CASES["add_dram"] = (lambda d: (dram(ints((1, 1, 2048, 2048), 1, *SMALL), d), dram(ints((1, 1, 2048, 2048), 2, *SMALL), d)),
                     lambda d, a, b: ttnn.add(a, b), lambda a, b: a + b)
CASES["sub_dram"] = (lambda d: (dram(ints((1, 1, 2048, 2048), 3, *SMALL), d), dram(ints((1, 1, 2048, 2048), 4, *SMALL), d)),
                     lambda d, a, b: ttnn.subtract(a, b), lambda a, b: a - b)
CASES["add_hs"] = (lambda d: (*hs(ints(hs_shape(d), 5, *SMALL), d), hs(ints(hs_shape(d), 6, *SMALL), d)[0]),
                   lambda d, a, m, b: ttnn.add(a, b, memory_config=m), lambda a, b: a + b)
CASES["add_relu_dram"] = (lambda d: (dram(ints((1, 1, 1024, 1024), 9, *SMALL), d), dram(ints((1, 1, 1024, 1024), 10, *SMALL), d)),
                          lambda d, a, b: ttnn.add(a, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]),
                          lambda a, b: torch.relu(a + b))
CASES["cumsum_int32"] = (lambda d: (dram(ints((1, 1, 512, 1024), 11, -1000, 1000), d),),
                         lambda d, x: ttnn.cumsum(x, dim=-1), lambda x: torch.cumsum(x, dim=-1).to(torch.int32))
CASES["sum_int32"] = (lambda d: (dram(ints((1, 1, 512, 1024), 12, -1000, 1000), d),),
                      lambda d, x: ttnn.sum(x, dim=-1), None)
CASES["add_float_dram"] = (lambda d: (ttnn.from_torch(torch.randn(1, 1, 1024, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d),) * 2,
                           lambda d, a, b: ttnn.add(a, b), None)


def to_host(y):
    return ttnn.to_torch(y)


def bits(dev, names, out):
    res = {}
    for name in names:
        setup, run, ref = CASES[name]
        args = setup(dev)
        y = to_host(run(dev, *args))
        h = hashlib.sha256(y.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]
        ok = None
        if ref is not None:
            host = [to_host(a) if isinstance(a, ttnn.Tensor) else a for a in args]
            host = [h_ for h_ in host if isinstance(h_, torch.Tensor)]
            r = ref(*host)
            ok = bool(torch.equal(y.reshape(r.shape).to(torch.int64), r.to(torch.int64)))
        res[name] = [h, ok]
    json.dump(res, open(out, "w"), indent=0)
    print("saved", len(res), "golden fails", [k for k, v in res.items() if v[1] is False])


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0)
    try:
        if os.environ.get("R_BITS"):
            bits(dev, names, os.environ["R_BITS"])
            return
        for name in names:
            setup, run, _ = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                print(f"CASE {name} ok", flush=True)
                del args
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
