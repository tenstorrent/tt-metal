# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small scalar recurrence model, NOT an attention/hardware accuracy oracle."""

import argparse
import json
import math
import random
import struct


def fp32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    # Explicit nearest/ties-away magnitude rounding. This model studies the
    # recurrence; it does not claim to emulate final TT reciprocal/multiply.
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits = (bits + 0x8000) & 0xffff0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def split(x):
    high = bf16(x)
    return high, bf16(fp32(x - high))


def metrics(actual, expected):
    error = math.fsum((a-b)**2 for a, b in zip(actual, expected))
    norm = math.fsum(b*b for b in expected)
    return {"l2_pct": 100*math.sqrt(error/norm) if norm else None,
            "absolute_rms": math.sqrt(error/len(actual)),
            "max_abs": max(abs(a-b) for a, b in zip(actual, expected))}


def probe(case, chunks, group, seed):
    rng = random.Random(seed)
    rows, channels = 64, 2
    size = rows*channels
    baseline_hi, baseline_lo = [0.0]*size, [0.0]*size
    root_hi, root_lo, local = [0.0]*size, [0.0]*size, [0.0]*size
    reference, denominator = [0.0]*size, [0.0]*rows
    maxima = [-math.inf]*rows
    identity_count, folds = 0, 0
    for k in range(chunks):
        correction = []
        for row in range(rows):
            if case == "changing":
                value = bf16((k//2)*0.03125)
            elif case == "rare_large_rescale":
                value = bf16((k//127)*4.0)
            elif case == "random_max":
                value = bf16(rng.gauss(3.0, 0.3))
            else:
                value = 0.0
            new_max = max(maxima[row], value)
            correction.append(0.0 if k == 0 else bf16(math.exp(maxima[row]-new_max)))
            maxima[row] = new_max
        identity = all(c == 1.0 for c in correction)
        identity_count += int(k > 0 and identity)
        fold = k > 0 and (k % group == 0 or k == chunks-1 or not identity)
        folds += int(fold)
        for row in range(rows):
            weight = bf16(1.0 if case in ("constant", "changing", "rare_large_rescale")
                          else math.exp(rng.gauss(0, 0.3)))
            denominator[row] = denominator[row]*correction[row]+weight
            for channel in range(channels):
                index = row*channels+channel
                if case in ("constant", "changing", "rare_large_rescale"):
                    chunk = weight
                elif case == "common_v":
                    chunk = bf16(weight*(32.0+rng.gauss(0, 1)))
                elif case == "cancellation":
                    chunk = bf16((1 if k % 2 else -1)*weight)
                else:
                    chunk = bf16(rng.gauss(0, 1)*weight)
                c = correction[row]
                reference[index] = reference[index]*c+chunk
                old = fp32(baseline_hi[index]+baseline_lo[index])
                baseline_hi[index], baseline_lo[index] = split(fp32(old*c+chunk))
                if k == 0:
                    root_hi[index], root_lo[index], local[index] = chunk, 0.0, 0.0
                elif fold:
                    root = fp32(root_hi[index]+root_lo[index])
                    if identity:
                        total = fp32(root+fp32(local[index]+chunk))
                    else:
                        total = fp32(fp32(root+local[index])*c+chunk)
                    root_hi[index], root_lo[index] = split(total)
                    local[index] = 0.0
                else:
                    local[index] = bf16(fp32(local[index]+chunk))
        # K0 already bootstraps root; one-chunk inputs need no recurrent fold.
    expected = [value/denominator[i//channels] for i, value in enumerate(reference)]
    baseline = [bf16(fp32(value/denominator[i//channels])) for i, value in enumerate(baseline_hi)]
    candidate = [bf16(fp32(value/denominator[i//channels])) for i, value in enumerate(root_hi)]
    bm, cm = metrics(baseline, expected), metrics(candidate, expected)
    gate = None if bm["l2_pct"] is None else cm["l2_pct"] <= 1.05*bm["l2_pct"]+0.0001
    return dict(case=case, chunks=chunks, group=group, baseline=bm, candidate=cm,
                numerical_gate=gate, candidate_baseline=metrics(candidate, baseline),
                all_row_identity_steps=identity_count, protected_folds=folds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    cases = ("normal", "constant", "common_v", "cancellation", "changing", "rare_large_rescale", "random_max")
    result = {"scope": "Scalar FP32/BF16 recurrence study, same quantized chunk/correction streams; not original-input attention qualification",
              "rounding": "Explicit BF16 nearest/ties-away; final normalization is simplified and is not TT hardware emulation",
              "results": [probe(case, args.chunks, group, args.seed) for group in (2, 4) for case in cases]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
