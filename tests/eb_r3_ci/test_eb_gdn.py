# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): gated_delta_attn.cpp through the gated deltanet chunked-mode validation test, which
imports `tests.test_gated_deltanet` from its own directory (a name the repository's `tests` package shadows), loaded here
by path."""
import importlib.util
import os
import sys

GD = os.path.abspath("models/experimental/gated_attention_gated_deltanet")
sys.path.insert(0, GD)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sys.modules["tests.test_gated_deltanet"] = _load("eb_gd_test_gated_deltanet", os.path.join(GD, "tests", "test_gated_deltanet.py"))
_val = _load("eb_gd_validation", os.path.join(GD, "tests", "test_ttnn_validation.py"))


def test_gated_deltanet_chunked():
    _val.test_gated_deltanet_chunked_ttnn()
