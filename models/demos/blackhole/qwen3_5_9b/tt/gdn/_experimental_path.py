# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""sys.path shim for the experimental gated_deltanet backend.

The experimental gated_deltanet module uses `from tt.ttnn_delta_rule_ops import ...`
which requires its parent directory on sys.path so `tt` resolves as a package.
Importing this module (as the FIRST import, before any experimental
`ttnn_gated_deltanet` / `ttnn_delta_rule_*` import) installs that path. This
mirrors the shim that lived at the top of the original single-file GDN module
(since split into tt/gdn/).
"""
import os
import sys

_EXPERIMENTAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "experimental", "gated_attention_gated_deltanet"
)
_EXPERIMENTAL_DIR = os.path.abspath(_EXPERIMENTAL_DIR)
if _EXPERIMENTAL_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTAL_DIR)
