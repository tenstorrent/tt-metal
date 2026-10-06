# SPDX-License-Identifier: Apache-2.0
"""Load the retained pre-repair source only for historical diagnostic repros."""

import importlib.util
from importlib.machinery import SourceFileLoader

from . import run_coverage

path = run_coverage.ROOT / "doc/functional_decoder/before_numerical_fix/functional_decoder.py.txt"
loader = SourceFileLoader("kolibri_diagnostic_baseline", str(path))
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
FunctionalDecoder = module.FunctionalDecoder
run_coverage.FunctionalDecoder = FunctionalDecoder
