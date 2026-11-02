# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host check of the PackedS layout: replay the A3 v2 kernel's scale reads (BRISC order, then the
DST rows scale_acc uses) over pack_scales() bytes and compare with the original scales.
    python -m models.demos.blackhole.bonsai2.native.a3.check_pack_scales"""
import numpy as np

from models.demos.blackhole.bonsai2.native.a3.op import SUPER, pack_scales, tail_group


def untile(b):  # 1024 fp32 in face order -> [32, 32]
    return b.reshape(2, 2, 16, 16).swapaxes(1, 2).reshape(32, 32)


def check(k, n, banks=8, wpb=4, cols=1):
    rng = np.random.default_rng(k + n)
    num_g = k // 32 // 32
    s = rng.random((num_g * SUPER, n), dtype=np.float32)
    cpc = n // 32 // banks // wpb
    t = tail_group(num_g, cpc)
    s_full, rem = num_g // 4, num_g % 4
    words = pack_scales(s, banks, t).reshape(banks, -1).view(np.float32)
    per_core = (cpc // t * (t * s_full + (1 if rem else 0))) * 1024
    for b in range(banks):
        for i in range(wpb):
            buf, off, tail = words[b], i * per_core, None
            for c in range(cpc):
                gcol = (b * wpb + i) * cpc + c
                if rem and c % t == 0:
                    tail, off = untile(buf[off:off + 1024]), off + 1024
                mains = [untile(buf[off + j * 1024: off + (j + 1) * 1024]) for j in range(s_full)]
                off += s_full * 1024
                for G in range(num_g):
                    if G < 4 * s_full:
                        rows = mains[G // 4][(G % 4) * 8:(G % 4) * 8 + 8]
                    else:
                        r0 = ((c % t) * rem + G - 4 * s_full) * 8
                        rows = tail[r0:r0 + 8]
                    want = s[G * SUPER:(G + 1) * SUPER, gcol * 32:(gcol + 1) * 32]
                    assert np.array_equal(rows, want), (k, n, b, i, c, G)
    return t, s_full, rem


if __name__ == "__main__":
    for k, n in ((5120, 2 * 17408), (17408, 5120), (6144, 5120), (5120, 16384), (5120, 14336), (5120, 248832)):
        t, f, r = check(k, n)
        print(f"K={k:5d} N={n:6d}: tail_t={t} s_full={f} rem={r} ok")
