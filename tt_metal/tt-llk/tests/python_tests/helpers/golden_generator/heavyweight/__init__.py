# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Heavyweight golden generators: block-by-block models of the hardware pipeline.

Each subpackage is one category of block. :mod:`.data_transfer` covers the
L1 <-> register-file moves (unpack and pack).
"""
