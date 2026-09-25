# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host (PyTorch/CPU) reference models for the Qwen3.5/3.6 Blackhole port.

Nothing here touches ttnn or a device — these are the golden implementations the
``tt/`` modules are validated against, and the place a new architecture is proven
runnable before any device bring-up starts.
"""
