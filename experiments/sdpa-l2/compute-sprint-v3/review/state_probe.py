"""Independent group-two recurrence model; NOT a hardware SDPA emulator.

Chunks/corrections are prescribed BF16 values. Scores, exp, PV accumulation,
reciprocal and final output packing are not modeled. The probe establishes
state invariants and looks for arithmetic-association sensitivity only.
"""

import json
import math
import random
import struct


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    """SFPU round macro: nearest, ties away from zero (finite inputs)."""
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits = (bits + 0x8000) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def split(x):
    hi = bf16(x)
    return hi, bf16(f32(x - hi))


def canonical(chunks, corrections):
    hi, lo = chunks[0], 0.0
    for chunk, correction in zip(chunks[1:], corrections[1:]):
        hi, lo = split(f32(math.fma(f32(hi + lo), correction, chunk)))
    return f32(hi + lo)


def grouped(chunks, corrections):
    hi, lo, local = chunks[0], 0.0, 0.0
    for index in range(1, len(chunks)):
        chunk, correction = chunks[index], corrections[index]
        boundary = index % 2 == 0 or index == len(chunks) - 1
        if correction != 1.0:
            hi, lo = split(f32(math.fma(f32(f32(hi + lo) + local), correction, chunk)))
            local = 0.0
        elif boundary:
            hi, lo = split(f32(f32(hi + lo) + f32(local + chunk)))
            local = 0.0
        else:
            assert local == 0.0, "Would overwrite an unflushed contribution"
            local = chunk
    assert local == 0.0, "Final partial group was not drained"
    return f32(hi + lo)


def exact(chunks, corrections):
    value = chunks[0]
    for chunk, correction in zip(chunks[1:], corrections[1:]):
        value = value * correction + chunk
    return value


def run():
    assert bf16(1.00390625) == 1.0078125
    assert bf16(-1.00390625) == -1.0078125
    rng = random.Random(20260918)
    records = []
    for count in (1, 2, 3, 4, 5, 16, 64, 512):
        for distribution in ("coherent", "constant", "cancellation", "normal"):
            for change in ("identity", "every", "alternating", "sparse"):
                changes = []
                reference_errors = [[], []]
                constant_ratio_errors = [[], []]
                for trial in range(64):
                    weights = [bf16(rng.uniform(0.25, 2.0)) for _ in range(count)]
                    if distribution == "constant":
                        chunks = weights
                    elif distribution == "coherent":
                        chunks = [bf16(w * (32.0 + rng.uniform(-0.1, 0.1))) for w in weights]
                    elif distribution == "cancellation":
                        chunks = [bf16((-1.0 if i % 2 else 1.0) * rng.uniform(0.5, 1.5)) for i in range(count)]
                    else:
                        chunks = [bf16(rng.gauss(0.0, 1.0)) for _ in range(count)]
                    corrections = [1.0 if change == "identity" or
                                   (change == "alternating" and i % 2) or
                                   (change == "sparse" and i % 7) else
                                   bf16(rng.uniform(0.25, 0.99)) for i in range(count)]
                    base, candidate = canonical(chunks, corrections), grouped(chunks, corrections)
                    ref = exact(chunks, corrections)
                    changes.append(abs(candidate - base))
                    reference_errors[0].append(abs(base - ref))
                    reference_errors[1].append(abs(candidate - ref))
                    if distribution == "constant":
                        denom = canonical(weights, corrections)
                        constant_ratio_errors[0].append(abs(base / denom - 1.0))
                        constant_ratio_errors[1].append(abs(candidate / denom - 1.0))
                records.append(dict(k_chunks=count, distribution=distribution,
                                    max_change_pattern=change, trials=64,
                                    max_candidate_baseline_abs=max(changes),
                                    max_baseline_reference_abs=max(reference_errors[0]),
                                    max_candidate_reference_abs=max(reference_errors[1]),
                                    max_constant_v_ratio_error=(max(constant_ratio_errors[1])
                                                               if constant_ratio_errors[1] else None)))
    print(json.dumps(dict(note=__doc__, total_trials=len(records)*64, records=records), indent=2))


if __name__ == "__main__":
    run()
