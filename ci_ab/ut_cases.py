# Scratch, not for merge. Untilize bits coverage: every case prints "BITS <tag> <case> <i> <status> <sha>" per readback.
import hashlib
import os
import sys

import torch
import ttnn

TAG = os.environ.get("BITS_TAG", "x")
TILE = 32
DT = {"bf16": (ttnn.bfloat16, torch.bfloat16), "bfp8": (ttnn.bfloat8_b, torch.float32),
      "bfp4": (ttnn.bfloat4_b, torch.float32), "fp32": (ttnn.float32, torch.float32)}


def wide(shape, seed, tdt):
    """Wide-range data: every 16 datums of a row (one block-float exponent group) get their own random power of two."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(shape, generator=g)
    e = torch.randint(-30, 30, shape[:-1] + (shape[-1] // 16,), generator=g)
    return (m * (2.0 ** e.float()).repeat_interleave(16, dim=-1)).to(tdt)


def sha(t):
    x = t.detach().contiguous().cpu()
    iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()]
    return hashlib.sha256((x.view(iv) if x.is_floating_point() else x).numpy().tobytes()).hexdigest()[:16]


def emit(case, i, status, t):
    print(f"BITS {TAG} {case} #{i} {status} {sha(t)}", flush=True)


def to_tt(dev, x, dt, mc=None, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=DT[dt][0], layout=layout, device=dev, memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)


def check_untilize(dev, case, x, dt, mc=None, out_mc=None, unpad=None):
    t = to_tt(dev, x, dt, mc)
    ref = ttnn.to_torch(t)
    if unpad is None:
        u = ttnn.untilize(t, memory_config=out_mc or t.memory_config())
        o = ttnn.to_torch(u)
        emit(case, 0, "PASS" if torch.equal(o, ref) else "FAIL", o)
    else:
        u = ttnn.untilize_with_unpadding(t, unpad)
        o = ttnn.to_torch(u)
        r = ref[tuple(slice(0, n + 1) for n in unpad)]
        emit(case, 0, "PASS" if torch.equal(o, r) else "FAIL", o)


def sharded_mc(layout, grid, shape):
    spec = ttnn.ShardSpec(grid, list(shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, spec)


def grid(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


CASES = {}


def case(f):
    CASES[f.__name__] = f
    return f


def reg(name, fn):
    CASES[name] = fn


# interleaved untilize, DRAM and L1, wide-range data
for _dt in ("bf16", "bfp8", "fp32"):
    for _ct in (2, 8, 9, 12, 16, 17, 24, 33, 64):
        for _l1 in (False, True):
            if _dt == "fp32" and _ct not in (9, 16, 33):
                continue

            def _f(dev, _dt=_dt, _ct=_ct, _l1=_l1):
                x = wide((1, 2, 3 * TILE, _ct * TILE), 100 + _ct, DT[_dt][1])
                mc = ttnn.L1_MEMORY_CONFIG if _l1 else ttnn.DRAM_MEMORY_CONFIG
                check_untilize(dev, f"il_{_dt}_ct{_ct}_{'l1' if _l1 else 'dram'}", x, _dt, mc)
            reg(f"il_{_dt}_ct{_ct}_{'l1' if _l1 else 'dram'}", _f)

# untilize with unpadding
for _dt in ("bf16", "bfp8"):
    for _ct in (9, 12, 16):
        def _f(dev, _dt=_dt, _ct=_ct):
            x = wide((1, 1, 2 * TILE, _ct * TILE), 200 + _ct, DT[_dt][1])
            check_untilize(dev, f"unpad_{_dt}_ct{_ct}", x, _dt, unpad=[0, 0, 49, _ct * TILE - 6])
        reg(f"unpad_{_dt}_ct{_ct}", _f)

# sharded untilize: input sharded (output sharded), and interleaved to sharded
for _dt in ("bf16", "bfp8"):
    for _ct in (9, 12, 16):
        W = _ct * TILE

        def _h(dev, _dt=_dt, W=W, _ct=_ct):
            x = wide((1, 1, 4 * 2 * TILE, W), 300 + _ct, DT[_dt][1])
            mc = sharded_mc(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, grid(4, 1), (2 * TILE, W))
            check_untilize(dev, f"sh_h_{_dt}_ct{_ct}", x, _dt, mc)
        reg(f"sh_h_{_dt}_ct{_ct}", _h)

        def _w(dev, _dt=_dt, W=W, _ct=_ct):
            x = wide((1, 1, 2 * TILE, 4 * W), 400 + _ct, DT[_dt][1])
            mc = sharded_mc(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid(4, 1), (2 * TILE, W))
            check_untilize(dev, f"sh_w_{_dt}_ct{_ct}", x, _dt, mc)
        reg(f"sh_w_{_dt}_ct{_ct}", _w)

        def _b(dev, _dt=_dt, W=W, _ct=_ct):
            x = wide((1, 1, 2 * 2 * TILE, 2 * W), 500 + _ct, DT[_dt][1])
            mc = sharded_mc(ttnn.TensorMemoryLayout.BLOCK_SHARDED, grid(2, 2), (2 * TILE, W))
            check_untilize(dev, f"sh_b_{_dt}_ct{_ct}", x, _dt, mc)
        reg(f"sh_b_{_dt}_ct{_ct}", _b)

        def _i2s(dev, _dt=_dt, W=W, _ct=_ct):
            x = wide((1, 1, 4 * 2 * TILE, W), 600 + _ct, DT[_dt][1])
            mc = sharded_mc(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, grid(4, 1), (2 * TILE, W))
            check_untilize(dev, f"i2s_{_dt}_ct{_ct}", x, _dt, None, out_mc=mc)
        reg(f"i2s_{_dt}_ct{_ct}", _i2s)


def _cache_tensor(dev, cache, dt):
    return ttnn.Tensor(cache, DT[dt][0]).to(ttnn.TILE_LAYOUT).to(dev)


# update_cache and fill_cache with a block-float cache, head_dim over 256 (Wt over 8)
for _dt in ("bfp8", "bf16"):
    for _hd in (320, 512):
        if _dt == "bf16" and _hd == 512:
            continue

        def _uc(dev, _dt=_dt, _hd=_hd):
            users, heads, seq, idx = 32, 1, 64, 5
            cache = wide((users, heads, seq, _hd), 700 + _hd, torch.bfloat16).float()
            x = wide((users, heads, 1, _hd), 800 + _hd, torch.bfloat16).float()
            ct = _cache_tensor(dev, cache, _dt)
            xt = ttnn.Tensor(x.permute(2, 1, 0, 3), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(dev)
            ct = ttnn.update_cache(ct, xt, idx, batch_offset=0)
            emit(f"update_cache_{_dt}_hd{_hd}", 0, "RUN", ct.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())
        reg(f"update_cache_{_dt}_hd{_hd}", _uc)

        def _fc(dev, _dt=_dt, _hd=_hd):
            users, heads, seq = 4, 1, 64
            cache = wide((users, heads, seq, _hd), 900 + _hd, torch.bfloat16).float()
            ct = _cache_tensor(dev, cache, _dt)
            for u in range(users):
                x = wide((1, heads, 32, _hd), 950 + _hd + u, torch.bfloat16).float()
                xt = ttnn.Tensor(x, DT[_dt][0]).to(ttnn.TILE_LAYOUT).to(dev)
                ct = ttnn.fill_cache(ct, xt, u)
            emit(f"fill_cache_{_dt}_hd{_hd}", 0, "RUN", ct.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())
        reg(f"fill_cache_{_dt}_hd{_hd}", _fc)


def _paged_fused(dev, case, dt, hd, paged, heads):
    users, block, seq = 8, 64, 128
    idx0 = 3
    c1 = wide((users, heads, seq, hd), 1000 + hd, torch.bfloat16).float()
    c2 = wide((users, heads, seq, hd), 1100 + hd, torch.bfloat16).float()
    nb = seq // block
    if paged:
        perm = torch.randperm(users * nb, generator=torch.Generator().manual_seed(7))
        table = torch.argsort(perm).reshape(users, nb)
        pt = ttnn.Tensor(table, ttnn.int32).to(dev)

        def prep(c):
            return c.reshape(users, heads, nb, block, hd).transpose(1, 2).reshape(users * nb, heads, block, hd)[perm]
        c1, c2 = prep(c1), prep(c2)
    else:
        pt = None
    ct1, ct2 = _cache_tensor(dev, c1, dt), _cache_tensor(dev, c2, dt)
    xs = []
    for s in (1200, 1300):
        x = wide((1, users, heads, hd), s + hd, torch.bfloat16).float()
        xp = torch.nn.functional.pad(x, (0, 0, 0, 32 - heads))
        xs.append(ttnn.Tensor(xp, ttnn.bfloat16).to(ttnn.TILE_LAYOUT))
    cg = dev.compute_with_storage_grid_size()
    avail = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, cg.y - 1))})
    g1 = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(0, 0), users, avail, True)
    avail = avail.subtract(g1)
    g2 = ttnn.num_cores_to_corerangeset_in_subcoregrids(avail.ranges()[0].start, users, avail, True)
    outs = []
    for xt, g in zip(xs, (g1, g2)):
        spec = ttnn.ShardSpec(g, [xt.volume() // xt.padded_shape[-1] // users, xt.padded_shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
        outs.append(xt.to(dev, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)))
    idxs = [idx0 + 17 * i for i in range(users)]
    ct1, ct2 = ttnn.experimental.paged_fused_update_cache(
        ct1, outs[0], ct2, outs[1], update_idxs_tensor=ttnn.Tensor(torch.tensor(idxs), ttnn.int32).to(dev), page_table=pt)
    emit(case, 0, "RUN", ct1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())
    emit(case, 1, "RUN", ct2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())


for _dt in ("bfp8",):
    for _hd in (320, 512):
        for _paged in (False, True):
            for _heads in (1, 8):
                _n = f"pfuc_{_dt}_hd{_hd}_{'paged' if _paged else 'flat'}_h{_heads}"
                reg(_n, lambda dev, _n=_n, _dt=_dt, _hd=_hd, _paged=_paged, _heads=_heads: _paged_fused(dev, _n, _dt, _hd, _paged, _heads))


def run_all(dev, names=None):
    ok = 0
    for n, f in CASES.items():
        if names and not any(s in n for s in names):
            continue
        try:
            f(dev)
            ok += 1
        except Exception as e:
            print(f"BITS {TAG} {n} #0 ERR {str(e)[:60].replace(chr(10), ' ')}", flush=True)
    return ok


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        print("CASES", len(CASES), "ran", run_all(dev, sys.argv[1:]), flush=True)
    finally:
        ttnn.close_device(dev)
