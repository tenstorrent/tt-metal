# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference implementations and architecture factories.

This package is intentionally test-side only — it never imports any
``TTNN``-side production code. The factories return fresh ``nn.Module``
instances with deterministic random bf16 weights so module-level tests
can share the SAME state_dict between the PyTorch reference and the
TTNN module under test (User decision #3: dummy weights).
"""
