# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exo-lang based kernel generation for TT-Metal.

This package uses the Exo scheduling language to generate reader, compute,
and writer kernels for Tenstorrent hardware. Exo separates algorithm
definition from hardware-specific optimization, enabling scheduling
exploration with compiler-guaranteed correctness.

Requires: pip install exo-lang
"""
