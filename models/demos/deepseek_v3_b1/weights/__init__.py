# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 weight preparation stack.

Package layout:

- ``weights.overlap`` — generic overlap primitives (OverlappedTensorSpec, overlap_tensors).
- ``weights.cache`` — content-addressed tensor cache (TensorCache, fingerprinting).
- ``weights.specs`` — DeepSeek-specific overlap configs and fusion group constants.
- ``weights.transforms`` — DeepSeek-specific shuffle/preprocess/fuse helpers.
- ``weights.prepare`` — high-level prepare functions (state dict → device tensors).
- ``weights.versioning`` — cache-invalidation version constant.
"""
