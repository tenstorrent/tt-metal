# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 40: skinny_contract attribution on the transposed operands (one process per variant, the variant a
define the factory reads from TTML_SC_ATTRIB: ATTRIB_SKIP_SMALL / ATTRIB_SKIP_BIG / ATTRIB_SKIP_MATH), at the three
step shapes: gate_up dB (big g^T [16384, T], small h [T, 32], transpose_out), w2 dA (big x^T [8256, T], small s_dh^T
[32, T] small_transposed) and w2 dB (big g^T [5120, T], small h^T [32, T] small_transposed, transpose_out).

    for v in '' ATTRIB_SKIP_SMALL ATTRIB_SKIP_BIG ATTRIB_SKIP_MATH; do TTML_SC_ATTRIB=$v python iter40_contract_attrib.py; done
"""

import os
import time

import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
device = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
T = 1024


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a.astype(np.float32)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def timed(fn, reps=30):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps


h = dev(np.pad(rng.standard_normal((1, 1, T, 16)), [(0, 0), (0, 0), (0, 0), (0, 16)]))
h_t = ttnn.transpose(h, -2, -1)
g16k = dev(rng.standard_normal((1, 1, 16384, T)) * 0.01)
x_t = dev(rng.standard_normal((1, 1, 8256, T)))
g5k = dev(rng.standard_normal((1, 1, 5120, T)) * 0.01)
cases = {
    "gate_up dB  op(h, g^T[16384,T], tr_out, big_tr)": lambda: ttml.ops.metal.skinny_contract(
        h, g16k, 2.0, True, big_transposed=True
    ),
    "w2 dA       op(s_dh^T, x^T[8256,T], big_tr, small_tr)": lambda: ttml.ops.metal.skinny_contract(
        h_t, x_t, 1.0, False, big_transposed=True, small_transposed=True
    ),
    "w2 dB       op(h^T, g^T[5120,T], tr_out, big_tr, small_tr)": lambda: ttml.ops.metal.skinny_contract(
        h_t, g5k, 2.0, True, big_transposed=True, small_transposed=True
    ),
}
variant = os.environ.get("TTML_SC_ATTRIB", "") or "as is"
times = {k: [] for k in cases}
names = list(cases)
for _ in range(3):
    for k in names + names[::-1]:
        times[k].append(timed(cases[k]))
for k in names:
    print(f"{variant:18s} {k:58s} median {np.median(times[k]):6.0f} us  min {min(times[k]):6.0f}")
ttml.autograd.AutoContext.get_instance().close_device()
