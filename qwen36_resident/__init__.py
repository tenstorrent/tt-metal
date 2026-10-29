# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident Qwen3.6-27B on Tenstorrent QB2: persistent decode and pipeline-parallel prefill."""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# the tt-metal checkout this package builds against (the submodule by default): kernel headers and data
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.join(os.path.dirname(PKG_DIR), "tt-metal"))
