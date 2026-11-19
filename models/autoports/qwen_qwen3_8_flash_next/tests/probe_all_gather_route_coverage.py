# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU enumeration of multicast route coverage; not a fabric/device test.

Mirrors the factory's original and alternate E/W lines plus N/S rectangles.
Hardware evidence belongs to the separate 1x4 Blackhole model probes.
"""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path


def hops(size, coordinate, ring):
    return (size // 2, size - 1 - size // 2) if ring else (size - 1 - coordinate, coordinate)


def routes(ny, nx, sy, sx, ring_y, ring_x, alternate):
    """Return destination/path pairs; a rectangle includes its spine column."""
    east, west = hops(nx, sx, ring_x)
    south, north = hops(ny, sy, ring_y)
    if alternate and ring_x and nx % 2 == 0:
        east, west = west, east
    if alternate and ring_y and ny % 2 == 0:
        south, north = north, south
    result = []
    for sign, distance, direction in ((1, east, "E"), (-1, west, "W")):
        for step in range(1, distance + 1):
            result.append(((sy, (sx + sign * step) % nx), direction))
    for sign, distance, direction in ((1, south, "S"), (-1, north, "N")):
        for step in range(1, distance + 1):
            y = (sy + sign * step) % ny
            result.append(((y, sx), direction))
            for xsign, xdistance, xdirection in ((1, east, "E"), (-1, west, "W")):
                for xstep in range(1, xdistance + 1):
                    result.append(((y, (sx + xsign * xstep) % nx), direction + xdirection))
    return result


def analyze():
    configurations = 0
    alternating_configurations = 0
    original_missing_path_pairs = 0
    for ny, nx in product(range(1, 9), repeat=2):
        if ny * nx == 1:
            continue
        for ring_y, ring_x in product((False, True), repeat=2):
            # Factory topology uses genuine torus axes: lengths 1/2 are linear.
            if (ring_y and ny < 3) or (ring_x and nx < 3):
                continue
            for sy, sx in product(range(ny), range(nx)):
                original = routes(ny, nx, sy, sx, ring_y, ring_x, False)
                alternate = routes(ny, nx, sy, sx, ring_y, ring_x, True)
                expected = set(product(range(ny), range(nx))) - {(sy, sx)}
                for route_set in (original, alternate):
                    credits = Counter(destination for destination, _ in route_set)
                    assert set(credits) == expected
                    assert set(credits.values()) == {1}
                both = Counter(destination for destination, _ in original + alternate)
                assert set(both.values()) == {2}
                # Path coverage, not merely arrival count: old completion reaches
                # each rank but may omit the alternate physical incoming path.
                payload_paths = set(original) | set(alternate)
                assert payload_paths <= set(original + alternate)
                original_missing_path_pairs += len(payload_paths - set(original))
                alternating_configurations += int(original != alternate)
                configurations += 1
    assert configurations == 4760
    original = routes(1, 4, 0, 1, False, True, False)
    alternate = routes(1, 4, 0, 1, False, True, True)
    destination = (0, 3)
    old_paths = {path for target, path in original if target == destination}
    payload_paths = {path for target, path in original + alternate if target == destination}
    assert old_paths == {"E"} and payload_paths == {"E", "W"}

    # An untagged counter cannot distinguish a next-epoch credit. This example
    # is excluded by the startup barrier on the measured model path, but not by
    # separate startup/done addresses or subtraction in isolation.
    threshold = 2 * (4 - 1)
    current_by_source = {"source0": 2, "source1": 0, "source2": 2}
    future_by_source = {"source0": 2, "source1": 0, "source2": 0}
    observed = sum(current_by_source.values()) + sum(future_by_source.values())
    assert observed == threshold and sum(current_by_source.values()) < threshold
    return {
        "passed": True,
        "scope": "CPU route/count enumeration and ordering analysis only; hardware validation limited to separate 1x4 Blackhole probes",
        "axis_lengths": [1, 8],
        "topologies": "linear or genuine rings of length >=3, independently per axis; all source coordinates",
        "configurations": configurations,
        "alternating_configurations": alternating_configurations,
        "old_completion_missing_payload_path_pairs": original_missing_path_pairs,
        "new_completion": "Each complete route set contributes one credit per remote rank; alternating runs send both and wait/subtract 2*(N-1). Nonalternating runs retain N-1.",
        "antipodal_counterexample": {
            "mesh": [1, 4],
            "source": [0, 1],
            "destination": [0, 3],
            "payload_paths": sorted(payload_paths),
            "old_completion_paths": sorted(old_paths),
            "missing_old_completion_paths": sorted(payload_paths - old_paths),
            "new_completion_paths": sorted(payload_paths),
        },
        "epoch_ordering": {
            "measured_model_uses_init_barrier": True,
            "proof": [
                "Current completion credits may arrive in either order; exactly 2*(N-1) are emitted for this receiver in an alternating invocation.",
                "A destination waiting for current completion cannot start its next program or send its next startup credit.",
                "Every remote reader needs that destination's next startup credit before producing next-invocation payload; its writer is gated by the reader's CB publication.",
                "Consequently no remote next-invocation completion can replace a current credit while this destination waits, provided the startup barrier runs.",
                "Separate startup/done semaphore addresses prevent early next startup credits from satisfying current completion. Atomic subtraction preserves legitimately early startup credits.",
            ],
            "separate_done_and_subtraction_alone_sufficient": False,
            "persistent_output_caveat": "Factory skips startup when persistent_output_tensor is supplied. Without another epoch fence, a fast sender's next completion can replace a slow sender's missing current credit. This pre-existing generic-path gap is outside the measured model call, which omits persistent_output_tensor.",
            "no_startup_counterexample": {
                "num_devices": 4,
                "threshold": threshold,
                "current_credits_by_source": current_by_source,
                "next_credits_by_source": future_by_source,
                "observed_counter": observed,
                "current_completion_missing": 2,
            },
        },
        "sources": [
            "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp",
            "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp",
            "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {result['configurations']} CPU topology/source configurations -> {args.output}")


if __name__ == "__main__":
    main()
