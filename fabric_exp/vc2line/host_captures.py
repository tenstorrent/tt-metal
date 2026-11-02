#!/usr/bin/env python3
"""host_captures.py PREFIX CASE [CASE ...]: mock captures (CPU only) of coverage cases under the names PREFIX + CASE in
this worktree's coverage directory, three at a time, with this worktree's host and router.

For host-change identity checks: capture the same cases before and after a host change (two prefixes), build both with
the same router sources (vc2l_offline.py build), and compare each case's set of router images (host_identity.py).
A new host arg changes a config's hash but, unread by the router, not its image."""
import concurrent.futures
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc2l_offline as off  # noqa: E402


def main():
    prefix, cases = sys.argv[1], sys.argv[2:]
    off.cov.ensure_root()
    for c in cases:
        off.cov.CASES[prefix + c] = off.cov.CASES[c]
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda c: off.run_mock(prefix + c), cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
