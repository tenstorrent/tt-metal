# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Compare two OLMO_DUMP_TOKENS dumps from text_olmo_demo.py user by user.

    python models/demos/olmo_galaxy/scripts/compare_decode_tokens.py BASE.json NEW.json [--min-prefix 64]

Prints, per user, the index of the first generated token that differs. Exit code 0 when every
user matches for at least --min-prefix tokens (default: the full length, i.e. bit-identical).
A run compared against itself in one build must be bit-identical (determinism); an optimization
that changes numerics may flip near-ties late in a sequence, so judge those divergences by eye.
"""

import argparse
import json
import sys


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--min-prefix", type=int, default=None, help="tokens that must match per user")
    args = parser.parse_args()

    base, new = json.load(open(args.base)), json.load(open(args.new))
    if base["prompts"] != new["prompts"]:
        sys.exit("prompts differ: dumps are not comparable")

    n_tokens = len(base["generated"][0])
    min_prefix = n_tokens if args.min_prefix is None else args.min_prefix
    identical, failing = 0, []
    for user, (a, b) in enumerate(zip(base["generated"], new["generated"])):
        div = first_divergence(a, b)
        if div is None:
            identical += 1
            continue
        print(f"user {user:2d}: first divergence at generated token {div}")
        if div < min_prefix:
            failing.append(user)

    print(f"{identical}/{len(base['generated'])} users bit-identical over {n_tokens} tokens")
    if failing:
        print(f"FAIL: users {failing} diverge before token {min_prefix}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
