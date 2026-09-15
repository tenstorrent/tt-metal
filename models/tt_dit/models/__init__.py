# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

# Opt this process into per-kernel JIT compile logging (tt_metal/jit_build/build.cpp, off by default so
# non-DiT workloads that compile 10k+ kernels and CI stay quiet). Placed on the DiT *model* package, not
# the tt_dit root: non-DiT models (Qwen, DeepSeek, SDXL) import tt_dit.utils/parallel but never
# tt_dit.models, so a shared-util import can't flip logging on for them — only importing an actual DiT
# model does. setdefault: an explicit TT_METAL_LOG_KERNEL_COMPILE=0 from the caller still wins.
os.environ.setdefault("TT_METAL_LOG_KERNEL_COMPILE", "1")
