# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 B1 **weights** adapter (layer 2) on top of generic **tensor_cache** (layer 1).

- ``catalog`` — **ArtifactTarget** layouts and cache-version constants for DeepSeek.
- ``preprocessing`` — deterministic torch **preprocess** from HF tensors.
- ``fusion_runtime`` — **pack/fuse** into device buffers (also re-exported via ``tensor_cache.fuse``).
- ``types`` — **assemble**d layer dataclasses and MoE helpers.
- ``adapter`` — orchestration: cache ``get_or_create`` + assembly.

Prefer :mod:`models.demos.deepseek_v3_b1.prepare_weights` if you need the historical import path.
"""
