# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's copy of the ttKDA prefill driver (``models/demos/deepseek_v3_d_p/tt/kda`` at tt-metal f8fabeb9b9c,
vendored 2026-10-29 at the user's direction so its grouping geometry, L1 budgets and chronology contract are tuned
here, inside the model folder; the ttnn ``experimental.kda`` ops stay shared). The HF-side reference helpers
(``deepseek_v3_d_p/reference/kda``) are still imported from the sibling.

Local tweaks (keep this list current):
- kda.py / recurrence.py take a host-side ``sequence_parallel_size`` (1 on a multi-row mesh = a row-local
  chronology per mesh row for the waves: the single-rank scan paths; the TP collective axis follows the mesh's SP
  extent). The ttnn ops stay stock: a row-local chronology reaches their kernels through the bounds (row r's slice
  ``[r rows, r rows + valid)`` of the global chronology, ``TtKDA.local_bounds``), not through an op argument
  (the 2026-10-29 ttnn override was retired 2026-11-02).
- config.py KDA_PREPARATION_L1_BYTES_PER_CORE 320 -> 256 KiB: the 1024-1088-local-row sequence-parallel geometry kept
  its chunk terms in L1 and clashed with the exclusive scan's circular buffers (2026-10-29).
- recurrence.py keeps the gathered SP prefix (all_gather + chain_affine_transforms) and the grouped SP scan; main's
  copy moved to the SP relay with one group per head (relay_affine_prefix + entry_correction, #63310), which is not
  ported. On main's op contracts (2026-11-18): the packed complement-form [S, N, K, K+V] summaries of [E | B]
  (E = A - I, #62896, the chunk decay as expm1, #62895), the tail slot whenever the mesh's SP axis has more than one
  device, and reduce_affine_transforms (removed by #63310) replaced by compose_group_transforms, a ttnn composition
  of each rank's head-slot group summaries. select_final_carry is gone too (2026-11-19, #66101: main's relay forms
  the carry): the SP final carry is the prefix's final state that chain_affine_transforms leaves on every SP rank,
  select_final_carry's unsplit case (GLM's SP chunks start rank-aligned, at 0).
"""
"""Public Kimi Delta Attention layer API."""

from models.demos.glm53_flash.tt.ttkda.kda import KdaState, ttKDA

__all__ = ["ttKDA", "KdaState"]
