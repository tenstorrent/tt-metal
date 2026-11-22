# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode prototype for Gemma-4-31B on QB2 (step 3: serving engine; copied from the frozen resident_s2).

Design and kernels adapted from changh95/qwen3.8-27b-p300x2 (code/qwen36_resident, Apache-2.0).
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
