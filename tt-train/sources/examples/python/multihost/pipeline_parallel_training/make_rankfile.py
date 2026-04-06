#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Generate an MPI rankfile from scontrol show hostnames output.

Usage:
    scontrol show hostnames | python make_rankfile.py -n <ranks_per_host> [-o <output_file>]
    python make_rankfile.py -n <ranks_per_host> [-o <output_file>] < hostnames.txt
"""

import argparse
import re
import sys


def _parse_host(hostname):
    """Parse hostname into (host_num, u_num) or None if unrecognised."""
    match = re.search(r"(\d+)u(\d{2})$", hostname)
    if not match:
        logger.warning(
            f"Hostname '{hostname}' does not match pattern '<digits>u<2 digits>'; placing last in canonical order"
        )
        return None
    return int(match.group(1)), int(match.group(2))


def canonicalize_hosts(hosts: list[str]) -> list[str]:
    """
    Sort hostnames into canonical pipeline order (snake/zigzag):
    low_host u08, low_host u02, high_host u02, high_host u08.
    Example: c09u08, c09u02, c10u02, c10u08.

    Within even-indexed host-number groups (0th, 2nd, …) u is descending (08 before 02).
    Within odd-indexed host-number groups (1st, 3rd, …) u is ascending (02 before 08).
    """
    parsed = [(h, _parse_host(h)) for h in hosts]
    unrecognised = [h for h, p in parsed if p is None]
    recognised = [(h, p) for h, p in parsed if p is not None]

    from collections import defaultdict

    groups = defaultdict(list)
    for h, (host_num, u_num) in recognised:
        groups[host_num].append((u_num, h))

    result = []
    for group_idx, host_num in enumerate(sorted(groups)):
        entries = groups[host_num]
        reverse = group_idx % 2 == 0
        entries.sort(key=lambda e: e[0], reverse=reverse)
        result.extend(h for _, h in entries)

    result.extend(unrecognised)
    return result


def make_rankfile(hosts: list[str], ranks_per_host: int) -> str:
    lines = []
    rank = 0
    for host in hosts:
        for slot in range(ranks_per_host):
            lines.append(f"rank {rank}={host} slot={slot}")
            rank += 1
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate MPI rankfile from SLURM hostnames")
    parser.add_argument(
        "-n",
        "--ranks-per-host",
        type=int,
        required=True,
        help="Number of MPI ranks to assign per host",
    )
    parser.add_argument("-o", "--output", default="-", help="Output file path (default: stdout)")
    args = parser.parse_args()

    raw_hosts = [line.strip() for line in sys.stdin if line.strip()]
    if not raw_hosts:
        print("error: no hostnames provided on stdin", file=sys.stderr)
        sys.exit(1)

    hosts = canonicalize_hosts(raw_hosts)
    rankfile = make_rankfile(hosts, args.ranks_per_host)

    if args.output == "-":
        sys.stdout.write(rankfile)
    else:
        with open(args.output, "w") as f:
            f.write(rankfile)
        print(
            f"Wrote rankfile to {args.output} ({len(hosts)} hosts x {args.ranks_per_host} ranks = {len(hosts) * args.ranks_per_host} total ranks)"
        )


if __name__ == "__main__":
    main()
