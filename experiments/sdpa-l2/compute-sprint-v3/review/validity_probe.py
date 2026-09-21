"""Independent logical-validity probe with deliberately uncleared physical local state."""

import json
import math
import random

from state_probe import bf16, f32, grouped, split


def valid_grouped(chunks, corrections, prior_physical=float("nan")):
    hi, lo, physical, valid = chunks[0], 0.0, prior_physical, False
    paths = {"retain": 0, "empty": 0, "identity_fold": 0, "changed_fold": 0}
    for index in range(1, len(chunks)):
        odd = index % 2 == 1
        chunk, correction = chunks[index], corrections[index]
        identity = correction == 1.0
        boundary = not odd or index == len(chunks) - 1
        if odd:
            assert not valid, "Even step failed to invalidate local"
            physical = chunk  # Direct odd PV overwrites the stale low plane.
        has_local = not odd and valid
        if identity and not boundary:
            assert odd
            paths["retain"] += 1
        elif not has_local:
            # Must not read physical here, including EVEN IDENTITY cases.
            hi, lo = split(f32(math.fma(f32(hi + lo), correction, chunk)))
            paths["empty"] += 1
        elif identity:
            hi, lo = split(f32(f32(hi + lo) + f32(physical + chunk)))
            paths["identity_fold"] += 1
        else:
            hi, lo = split(f32(math.fma(f32(f32(hi + lo) + physical), correction, chunk)))
            paths["changed_fold"] += 1
        valid = odd and identity and not boundary
        # Do NOT clear physical after a fold. The model intentionally leaves
        # stale nonzero data, including across distinct Q jobs below.
    assert not valid
    assert math.isfinite(hi) and math.isfinite(lo)
    return f32(hi + lo), physical, paths


def run():
    rng = random.Random(20260919)
    paths = {"retain": 0, "empty": 0, "identity_fold": 0, "changed_fold": 0}
    total = changed = 0
    physical = float("nan")
    max_delta = 0.0
    for count in (1, 2, 3, 4, 5, 8, 16, 64, 512):
        for trial in range(256):
            # Four independently changing row groups; sequential calls also
            # carry deliberately stale physical data between distinct Q jobs.
            for row in range(4):
                chunks = [bf16(rng.gauss(0, 1) + (32 if trial % 3 == 0 else 0)) for _ in range(count)]
                corrections = [1.0 if rng.random() < (trial % 5)/4 else bf16(rng.uniform(0.25, 0.99))
                               for _ in range(count)]
                expected = grouped(chunks, corrections)
                actual, physical, used = valid_grouped(chunks, corrections, physical)
                delta = abs(actual - expected)
                max_delta = max(max_delta, delta)
                changed += actual != expected
                total += 1
                for key, value in used.items():
                    paths[key] += value
                assert actual == expected, "Validity optimization changed modeled group2 result"
    # Negative control: odd changed-max leaves a stale chunk in low; even
    # identity must take empty-local update, not fold stale low again.
    expected, _, _ = valid_grouped([1.0, 2.0, 3.0], [1.0, 0.5, 1.0])
    stale_reused_result = (1.0*0.5 + 2.0) + 2.0 + 3.0
    assert expected == 5.5 and stale_reused_result == 7.5
    print(json.dumps(dict(trials=total, modeled_differences=changed, max_abs_delta=max_delta,
                          paths=paths, stale_local_negative_control=dict(correct=expected,
                          incorrect_even_identity=stale_reused_result),
                          note="State-machine simulation, not device precision qualification."), indent=2))


if __name__ == "__main__":
    run()
