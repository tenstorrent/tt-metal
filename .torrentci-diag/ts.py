#!/usr/bin/env python3
# TEMPORARY (torrentci#2377 diagnostic): prefix each stdin line with the
# seconds elapsed since this filter started.
import sys
import time

start = time.monotonic()
for line in sys.stdin:
    sys.stdout.write(f"[{time.monotonic() - start:7.1f}] {line}")
    sys.stdout.flush()
