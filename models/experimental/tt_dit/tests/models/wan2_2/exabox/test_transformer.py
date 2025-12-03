# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import ttnn

from ..test_transformer_wan import (
    run_test_wan_transformer,
    run_test_wan_transformer_model,
    run_test_wan_transformer_model_caching,
)


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1], ids=["num_links_1"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
@pytest.mark.parametrize(
    "sp_axis, tp_axis, topology",
    [
        (1, 0, ttnn.Topology.Ring),
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
@pytest.mark.parametrize("prompt_seq_len", [118], ids=["prompt_118"])
@pytest.mark.parametrize("is_fsdp", [True], ids=["yes_fsdp"])
def test_wan_transformer_exabox(
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
    run_test_wan_transformer(
        mesh_device, sp_axis, tp_axis, num_links, T, H, W, prompt_seq_len, is_fsdp, topology, chunk_size=128
    )


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [3], ids=["num_links_3"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
@pytest.mark.parametrize(
    "sp_axis, tp_axis, topology",
    [
        (1, 0, ttnn.Topology.Ring),
    ],
)
@pytest.mark.parametrize(
    ("B, T, H, W, prompt_seq_len"),
    [
        (1, 31, 40, 80, 118),
        (1, 21, 60, 104, 118),
        (1, 21, 90, 160, 118),
    ],
    ids=["5b-720p", "14b-480p", "14b-720p"],
)
@pytest.mark.parametrize("load_cache", [True, False], ids=["yes_load_cache", "no_load_cache"])
def test_wan_transformer_model_exabox(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    topology: ttnn.Topology,
    load_cache: bool,
    reset_seeds,
) -> None:
    run_test_wan_transformer_model(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        B=B,
        T=T,
        H=H,
        W=W,
        prompt_seq_len=prompt_seq_len,
        load_cache=load_cache,
        topology=topology,
        chunk_size=128,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [3], ids=["num_links_3"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
@pytest.mark.parametrize(
    "sp_axis, tp_axis, topology",
    [
        (1, 0, ttnn.Topology.Ring),
    ],
)
@pytest.mark.parametrize("subfolder", ["transformer", "transformer_2"], ids=["transformer_1", "transformer_2"])
def test_wan_transformer_model_caching_exabox(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    subfolder: str,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    run_test_wan_transformer_model_caching(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        subfolder=subfolder,
        topology=topology,
        chunk_size=128,
    )
