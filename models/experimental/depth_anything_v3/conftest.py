# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for Depth Anything V3."""

import os

import torch

# Use every physical core for the CPU reference path. PyTorch defaults to
# half of nproc on this host (32/64), leaving the rest idle during inference.
_n = os.cpu_count() or 1
torch.set_num_threads(_n)
