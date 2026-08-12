# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def test_composed_gather_partition_goldens_host_only():
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    all_gather = ttnn.get_golden_function(ttnn.all_gather)
    gathered = all_gather(value, dim=-1, cluster_axis=0)
    assert gathered is not value
    assert torch.equal(gathered, value)

    mesh_partition = ttnn.get_golden_function(ttnn.mesh_partition)
    partitioned = mesh_partition(value, dim=1, cluster_axis=0)
    assert partitioned is not value
    assert torch.equal(partitioned, value)
