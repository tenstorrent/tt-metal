# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep experiments reproducible after production defaults are consolidated."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

name = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tt._multichip_initial"
path = Path(__file__).resolve().parents[1] / "doc/multichip_decoder/candidates/initial.py.txt"
loader = importlib.machinery.SourceFileLoader(name, str(path))
spec = importlib.util.spec_from_loader(name, loader)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
loader.exec_module(module)
MultichipDecoder = module.MultichipDecoder
