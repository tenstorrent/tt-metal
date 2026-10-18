# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests in this directory exercise the throughput-experts fused ops (TopKRouter
fused kernel, all_to_all_dispatch / combine, moe_gpt, etc.) and are pinned to
a 4×8 Galaxy mesh with FABRIC_1D_RING. On systems with fewer devices — e.g.
the 1×1 Blackhole P150 dev box — pytest's mesh_device fixture would TT_FATAL
when trying to open a 4×8 mesh and crash the entire test session. Skip the
whole subdirectory cleanly when the system can't host a 4×8 mesh.
"""


import ttnn

collect_ignore_glob = []

if ttnn.get_num_devices() < 32:
    collect_ignore_glob = ["test_gpt_oss_*.py"]
