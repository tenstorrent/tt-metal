# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free (CPU preparation) KDA weight caches equal device-written ones on every LoudBox placement."""

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights

pytestmark = run_for_blackhole()


def _cache_artifact_names(path: Path) -> set[str]:
    return {artifact.name for artifact in path.glob("*.tensorbin")}


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param((2, 4), 1, fabric_1d_device_params(), id="mesh2x4-tpaxis1"),
        pytest.param((8, 1), 0, fabric_1d_device_params(), id="mesh8x1-tpaxis0"),
        pytest.param((8, 1), 1, fabric_1d_device_params(), id="mesh8x1-tpaxis1"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_device_free_cache_matches_device_written_cache(
    mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict, tmp_path: Path
) -> None:
    """The preparation step's device-free tensorbins equal the device-mapper ones and load bit-identically."""
    config = KDAConfig(hidden_size=128, num_heads=8, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, norm_eps=1e-5)
    state_dict = random_weights(config)
    prefix = "layer_1.kda"
    device_free, device_written = tmp_path / "device_free", tmp_path / "device_written"
    KDAWeights.build_ttnn_cache(
        state_dict, device_free, prefix, config, tuple(mesh_device.shape), tensor_parallel_axis=tensor_parallel_axis
    )
    written = load_kda_weights(
        mesh_device,
        config,
        state_dict,
        device_written,
        cache_name_prefix=prefix,
        tensor_parallel_axis=tensor_parallel_axis,
    )

    names = _cache_artifact_names(device_free)
    assert len(names) == 10 and names == _cache_artifact_names(device_written)
    assert all((device_free / name).read_bytes() == (device_written / name).read_bytes() for name in names)
    loaded = KDAWeights.from_cache(device_free, prefix, config, mesh_device, tensor_parallel_axis=tensor_parallel_axis)
    for name in ("input_projection", "decay_output_projection", "output_projection", "norm"):
        expected = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(ttnn.from_device(getattr(written, name)))]
        actual = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(ttnn.from_device(getattr(loaded, name)))]
        assert len(actual) == mesh_device.get_num_devices()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), name
