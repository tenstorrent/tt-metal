# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

# Per-kernel JIT compile logging (tt_metal/jit_build/build.cpp) is off by default so non-DiT
# workloads that compile 10k+ kernels — and CI — stay quiet. A DiT run is its own process, so
# importing any tt_dit model opts this process in, turning a multi-minute silent cold start
# (weight load + first-run JIT) into a visible stream. setdefault: an explicit
# TT_METAL_LOG_KERNEL_COMPILE=0 from the caller still wins.
os.environ.setdefault("TT_METAL_LOG_KERNEL_COMPILE", "1")
