# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Copy this file as conftest.py next to a copy of a tt-blaze test to run it outside the blaze tree.
# Scratch conftest: loads blaze's root and tests/blaze conftests so probe copies run outside the blaze tree.
import importlib.util
import sys

_B = "/localdev/gchoudhary/tt-blaze"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_root = _load("_blaze_root_conftest", f"{_B}/conftest.py")
_tests = _load("_blaze_tests_conftest", f"{_B}/tests/blaze/conftest.py")
for _m in (_root, _tests):
    globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
