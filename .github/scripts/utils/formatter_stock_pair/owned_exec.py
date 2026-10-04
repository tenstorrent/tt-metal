#!/usr/bin/env python3
"""Gated direct exec: publish PID/birth before starting a bounded owned CLI."""
import os
import sys

assert sys.stdin.readline() == "GO\n", "Controller must publish ownership first"
assert len(sys.argv) > 1
os.execvp(sys.argv[1], sys.argv[1:])
