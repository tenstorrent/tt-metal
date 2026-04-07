# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 weight preparation stack.

Package layout:

- ``weights.overlap`` — generic overlap primitives (OverlappedTensorSpec, overlap_tensors).
- ``weights.cache`` — content-addressed tensor cache (TensorCache, fingerprinting).
- ``weights.specs`` — DeepSeek-specific overlap configs (each produces its own FusionGroupSpec).
- ``weights.transforms`` — DeepSeek-specific shuffle/preprocess/fuse helpers.
- ``weights.prepare`` — high-level prepare functions (state dict → device tensors).

Cache-invalidation versioning is per-target: each ``*_SingleDeviceOverlapSpec``
(for fusion groups) and each ``TensorTarget`` (for standalone tensors) carries
its own ``transform_version`` field.  Bump it when the associated
shuffle/preprocess logic changes.
"""
