#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""rf_walk.py: N7, run.sh's copy-only register-file walk, right after N2 (hc/fleet/chunk_load.py rw_after).

Root cause: a matmul mismatch can be a Dest/Src register bit or the FPU's math. N7 moves data through the
plainest unpack -> Src -> Dest -> pack path and nothing else, so a bit it finds wrong is the register file's (or the
unpacker's/packer's), never the FPU's. The op is ttnn.clone with a dtype change, which runs the eltwise copy compute
kernel (compute_kernel_lib::copy -> copy_tile: unpack a tile to Src, datacopy to Dest, pack it out); a clone with the
same dtype skips compute and would test nothing here. Two legs per pass:
  bf16   bfloat16 in, float32 out, 16-bit Dest (fp32_dest_acc_en off)
  fp32   float32 in (values bf16 can hold), bfloat16 out, 32-bit Dest (fp32_dest_acc_en on)
Both conversions are exact for these values, so the readback is compared bit for bit with the host's copy.

Data: one bf16 tensor HEIGHT_SHARDED in L1 over the compute grid, TILE layout, one shard of TILES 32x32 tiles per Tensix
core (shard i on core (i % gx, i // gx), as l1_march.mem()). Values are walking-1 and walking-0 words over fields that
never give a denormal, inf or NaN: the sign bit, each mantissa bit at exponent 127, the exponent walking 1 (1, 2, ...,
128) and walking 0 (0xFE, 0xFD, ..., 0x7F), and each word's bitwise inverse (also legal). Element (row, col) of a tile
takes word (col + 3*row + 5*tile + 7*shard + pass/2) of that list, and odd passes invert every word, so within two passes
each Src/Dest position holds both a 1 and a 0 in every bit.

Vote across chips as l1_march: a diff that more than half of the chips show identically is the test's own (a WARN
that names nothing); a chip that differs alone is its N7 FAIL (fleet/chunk_sched.rf_lines), one line per core and leg
naming BDF, tray, core x,y, leg and the flipped bits. An op that raises leaves that leg 'not checked' (a WARN), never
a crash. Passes loop until the wall cap (always at least one).
"""
import time

TILES = 4  # 32x32 tiles per core
KEEP = 8  # example words kept per chip
LEGS = ("bf16", "fp32")


def words():
    """The walking-1/0 bf16 words (as ints 0..0xFFFF) and their inverses: every exponent is 1..254."""
    one = 0x3F80  # 1.0: exponent 127, mantissa 0
    w = [one, one | 0x8000]  # sign
    w += [one | 1 << b for b in range(7)]  # mantissa walking 1
    w += [one | (0x7F ^ 1 << b) for b in range(7)]  # mantissa walking 0
    w += [(1 << b) << 7 for b in range(8)]  # exponent walking 1: 1, 2, ..., 128
    w += [(0xFF ^ 1 << b) << 7 for b in range(8)]  # exponent walking 0: 0xFE, ..., 0x7F
    return w + [v ^ 0xFFFF for v in w]


def gold_bits(torch, n, k):
    """Pass k's (n * TILES * 32, 32) bf16 words as int32 (0..0xFFFF), shard by shard."""
    w = torch.tensor(words(), dtype=torch.int64)
    rr = torch.arange(32).view(1, 1, 32, 1)
    cc = torch.arange(32).view(1, 1, 1, 32)
    tt = torch.arange(TILES).view(1, TILES, 1, 1)
    ss = torch.arange(n).view(n, 1, 1, 1)
    g = w[(cc + 3 * rr + 5 * tt + 7 * ss + k // 2) % len(w)]
    if k % 2:
        g = g ^ 0xFFFF
    return g.reshape(n * TILES * 32, 32).to(torch.int32)


def grid(mesh):
    try:
        g = mesh.compute_with_storage_grid_size()
    except Exception:  # noqa: BLE001
        g = mesh.get_devices()[0].compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def mem(ttnn, gx, gy):
    """HEIGHT_SHARDED L1 over the grid, TILES tiles per core: shard i on core (i % gx, i // gx)."""
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [TILES * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )


def kcfg(ttnn, fp32):
    try:
        return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32)
    except Exception:  # noqa: BLE001
        return None


def to_host(torch, bits, leg):
    """The input tensor for leg: bf16 words as torch.bfloat16 (bf16 leg) or as float32 with zero low halves (fp32)."""
    b16 = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    if leg == "bf16":
        return b16.view(torch.bfloat16)
    return _f32(torch, bits)


def _f32(torch, bits):
    v = bits.to(torch.int64) << 16
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).view(torch.float32)


def out_bits(torch, t, leg):
    """A readback's words as int64 in the output's own width: float32 bits (bf16 leg), bf16 bits (fp32 leg)."""
    t = t.contiguous()
    if leg == "bf16":
        return t.to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return t.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF


def want_bits(bits, leg):
    """The words a clean readback holds, as out_bits() reads them."""
    return bits.long() << 16 if leg == "bf16" else bits.long()


def inject(got, chips, k, leg, gx, n):
    """With TRAY_INJECT naming N7 and a chip: one bit flipped in that chip's readback of this leg, in the first word of
    the spec's core's shard (default core 0,0)."""

    import tray_inject

    for i in range(len(got)):
        s = tray_inject.want("N7", chips.get(i, {}).get("pci"), k)
        if s is None:
            continue
        x, y = s["core"] or (0, 0)
        e = (y * gx + x) % n
        tray_inject.applied(s, core=(e % gx, e // gx), where=f"leg={leg}")
        got[i] = tray_inject.flip_tensor(s, got[i], e * TILES * 32 * 32)


def rf_check(ttnn, torch, mesh, chips, wall_s, beat=None, say=None):
    """N7: both legs above on every chip until wall_s s. chips: mesh index -> dict(pci, tray, dev). Returns dict(passes,
    legs, tiles_per_core, cores, grid, wall_s, test_side, errors={leg: error}, chips={mesh index: dict(chip, passes,
    bad_passes, bad (words), first_pass, first_t, xor ({leg: mask}), cores={"x,y": {leg: dict(words, xor)}},
    examples)})."""
    gx, gy = grid(mesh)
    n = gx * gy
    rows = n * TILES * 32
    mc = mem(ttnn, gx, gy)
    shard = torch.arange(rows).view(rows, 1).expand(rows, 32) // (TILES * 32)
    new = lambda: dict(  # noqa: E731
        passes=0, bad_passes=0, bad=0, first_pass=None, first_t=None, xor={}, cores={}, examples=[]
    )
    st = {i: new() for i in chips}
    errors, test_side = {}, []
    t0 = time.monotonic()
    k, last = 0, 0.0
    while time.monotonic() - t0 + last < wall_s or k == 0:
        p0 = time.monotonic()
        bits = gold_bits(torch, n, k)
        bad_pass = set()
        for leg in LEGS:
            if leg in errors:
                continue
            fp32 = leg == "fp32"
            x = y = None
            try:
                x = ttnn.from_torch(
                    to_host(torch, bits, leg),
                    dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mc,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                y = ttnn.clone(
                    x,
                    dtype=ttnn.bfloat16 if fp32 else ttnn.float32,
                    memory_config=mc,
                    compute_kernel_config=kcfg(ttnn, fp32),
                )
                ttnn.synchronize_device(mesh)
                got = [ttnn.to_torch(d).reshape(rows, 32) for d in ttnn.get_device_tensors(y.cpu())]
            except Exception as e:  # noqa: BLE001  that leg is not checked; the other leg and the stage go on
                errors[leg] = f"{type(e).__name__}: {str(e)[:200]}"
                if say:
                    say(f"leg {leg} not checked: {errors[leg]}")
                continue
            finally:
                for t in (x, y):
                    if t is not None:
                        try:
                            ttnn.deallocate(t)
                        except Exception:  # noqa: BLE001
                            pass
            inject(got, chips, k, leg, gx, n)  # TRAY_INJECT: test-only flip of one chip's host copy, before compare
            want = want_bits(bits, leg)
            ob = [out_bits(torch, g, leg) for g in got]
            diffs = {i: o != want for i, o in enumerate(ob) if bool((o != want).any())}
            sig = {}
            for i, d in diffs.items():
                f = d.reshape(-1)
                key = (int(f.sum()), int(torch.nonzero(f)[0]), int(ob[i].reshape(-1)[f][0]))
                sig.setdefault(key, []).append(i)
            # a diff most chips share is the test's own; below 3 chips there is no majority, so each chip is judged
            # against the fixed expected pattern alone
            common = [ii for ii in sig.values() if len(ii) * 2 > len(got)] if len(got) >= 3 else []
            if common:
                w = int(diffs[common[0][0]].sum())
                test_side.append(dict(pass_=k, leg=leg, chips=len(common[0]), of=len(got), words=w))
                if say:
                    say(
                        f"leg {leg} pass {k}: {len(common[0])} of {len(got)} chips mismatched the same way: "
                        "test-side, not judged"
                    )
                for i in common[0]:
                    diffs.pop(i)
            for i, d in diffs.items():
                s = st.setdefault(i, new())
                bad_pass.add(i)
                nb = int(d.sum())
                s["bad"] += nb
                if s["first_pass"] is None:
                    s["first_pass"], s["first_t"] = k, round(time.time(), 1)
                xb = (ob[i] ^ want)[d]
                sh = shard[d]
                for e in torch.unique(sh).tolist():
                    m = 0
                    for v in torch.unique(xb[sh == e]).tolist():
                        m |= int(v)
                    c = s["cores"].setdefault(f"{e % gx},{e // gx}", {}).setdefault(leg, dict(words=0, xor=0))
                    c["words"] += int((sh == e).sum())
                    c["xor"] |= m
                    s["xor"][leg] = s["xor"].get(leg, 0) | m
                r, cc = torch.nonzero(d, as_tuple=True)
                for j in range(min(KEEP - len(s["examples"]), int(r.numel()))):
                    e = int(r[j]) // (TILES * 32)
                    s["examples"].append(
                        dict(
                            leg=leg,
                            k=k,
                            core=f"{e % gx},{e // gx}",
                            row=int(r[j]),
                            col=int(cc[j]),
                            got=int(ob[i][r[j], cc[j]]),
                            want=int(want[r[j], cc[j]]),
                        )
                    )
                if say:
                    ch = chips.get(i, {})
                    say(f"BAD mesh{i} {ch.get('pci')} tray {ch.get('tray') or '?'}: {nb} words, leg {leg}, pass {k}")
        if len(errors) == len(LEGS):
            break
        for i in st:
            st[i]["passes"] += 1
            st[i]["bad_passes"] += i in bad_pass
        k += 1
        last = time.monotonic() - p0
        if beat:
            beat(f"rf pass {k}")
    el = time.monotonic() - t0
    return dict(
        passes=k,
        legs=[L for L in LEGS if L not in errors],
        tiles_per_core=TILES,
        cores=n,
        grid=f"{gx}x{gy}",
        wall_s=round(el, 1),
        test_side=test_side,
        errors=errors,
        chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()},
    )
