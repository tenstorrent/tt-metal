# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn

from ..test_attention_wan import run_test_wan_attention


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1], ids=["num_links_1"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
@pytest.mark.parametrize(
    "sp_axis, tp_axis, topology",
    [
        (1, 0, ttnn.Topology.Linear),
    ],
)
@pytest.mark.parametrize(
    "T, H, W",
    [
        (31, 40, 80),
        (21, 60, 104),
        (21, 90, 160),
    ],
    ids=["5b-720p", "14b-480p", "14b-720p"],
)
@pytest.mark.parametrize("prompt_seq_len", [None, 26, 126], ids=["no_prompt", "short_prompt", "long_prompt"])
@pytest.mark.parametrize("is_fsdp", [True], ids=["yes_fsdp"])
def test_wan_attention_exabox(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    run_test_wan_attention(
        mesh_device, sp_axis, tp_axis, num_links, T, H, W, prompt_seq_len, is_fsdp, topology, chunk_size=64
    )
