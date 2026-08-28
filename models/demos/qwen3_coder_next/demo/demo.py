# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Canonical entrypoint: runs the real emitted pipeline demo (repointed by emit-e2e)."""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, 'demo_text_generation.py')
sys.argv[0] = _TARGET
runpy.run_path(_TARGET, run_name="__main__")
