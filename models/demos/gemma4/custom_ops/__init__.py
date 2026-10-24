# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-local fused ops for Gemma 4 (``ttnn.generic_op`` programs with their kernels beside them).

Each op lives in its own folder: ``<op>/op.py`` (a class with ``golden()`` in torch and ``op()`` building the
``ttnn.ProgramDescriptor``), ``<op>/kernels/`` (reader / compute / writer), and a unit test under
``tests/unit/custom_ops``. They replace chains of data-movement ops in the lane decode step (see README.md).
"""
