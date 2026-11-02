import sys

sys.path.insert(0, "tests/python")
import numpy as np
import ttml
from tensor_helpers import as_stored, to_device, to_host, sigmoid


def tr(a):
    return np.ascontiguousarray(np.swapaxes(a, -1, -2))


def ref_bw(gate, up, dh):
    s = sigmoid(gate)
    return dh * up * (s * (1.0 + gate * (1.0 - s))), dh * gate * s


for inner, seq in [(64, 32), (128, 64)]:
    rng = np.random.default_rng(1)
    gate = rng.uniform(-4, 4, (1, 1, seq, inner))
    up = rng.uniform(-2, 2, (1, 1, seq, inner))
    dh = rng.uniform(-1, 1, (1, 1, seq, inner))
    for a_scale in (0.0, 0.5):
        a = rng.uniform(-a_scale, a_scale, (1, 1, 16, inner)) if a_scale else np.zeros((1, 1, 16, inner))
        s_dh = rng.uniform(-1, 1, (1, 1, seq, 16))
        tail = np.zeros((1, 1, seq, 64))
        tail[..., :16] = s_dh
        packed = np.concatenate([gate, up], -1)
        dpt = ttml.ops.metal.swiglu_packed_bw(
            to_device(tr(packed)).get_value(),
            to_device(tr(np.concatenate([dh, tail], -1))).get_value(),
            lora_a=to_device(a).get_value(),
            lowrank_scale=0.25,
            transposed_io=True,
        )
        got = tr(to_host(dpt))
        dh_full = as_stored(as_stored(dh) + 0.25 * as_stored(s_dh) @ as_stored(a))
        ref = to_host(ttml.ops.metal.swiglu_packed_bw(to_device(packed).get_value(), to_device(dh_full).get_value()))
        d = np.abs(got - ref)
        print(
            f"I={inner} T={seq} a_scale={a_scale}: max|got-ref| dgate {d[..., :inner].max():.3e} dup {d[..., inner:].max():.3e}; got dup[0,:4] {got[0,0,0,inner:inner+4]} ref {ref[0,0,0,inner:inner+4]}"
        )
        # which rows/cols differ?
        bad = np.argwhere(d[0, 0] > 1e-2)
        if len(bad):
            print("   first bad (row, col):", bad[:6].tolist(), " count", len(bad), " of", d[0, 0].size)
