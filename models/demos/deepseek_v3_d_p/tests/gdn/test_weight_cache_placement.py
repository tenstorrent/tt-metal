# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test: device-free GDN weight caches equal device-written ones on the LoudBox placements, cache-only loading
reproduces the device-written tensors, and a cache miss fails before anything is placed."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tt.gdn.weights import GDNWeights, load_gdn_weights

pytestmark = run_for_blackhole()

_CONFIG = replace(TINY, hidden_size=128, num_key_heads=4, num_value_heads=12, head_k_dim=32, head_v_dim=32)


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(ttnn.from_device(tensor))]


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param((2, 4), 1, fabric_1d_device_params(), id="LB-A-mesh2x4-tpaxis1"),
        pytest.param((8, 1), 1, fabric_1d_device_params(), id="LB-B-mesh8x1-tpaxis1"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_device_free_cache_matches_device_written_cache(
    mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict, tmp_path: Path, expect_error
) -> None:
    state_dict = random_weights(_CONFIG)
    prefix = "layer_0.gdn"
    device_free, device_written = tmp_path / "device_free", tmp_path / "device_written"
    with expect_error(FileNotFoundError, "incomplete GDN TTNN cache"):
        load_gdn_weights(mesh_device, _CONFIG, None, device_free, cache_name_prefix=prefix)
    GDNWeights.build_ttnn_cache(
        state_dict, device_free, prefix, _CONFIG, tuple(mesh_device.shape), tensor_parallel_axis=tensor_parallel_axis
    )
    written = load_gdn_weights(
        mesh_device,
        _CONFIG,
        state_dict,
        device_written,
        cache_name_prefix=prefix,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    names = {path.name for path in device_free.glob("*.tensorbin")}
    assert len(names) == 9 and names == {path.name for path in device_written.glob("*.tensorbin")}
    assert all((device_free / name).read_bytes() == (device_written / name).read_bytes() for name in names)
    loaded = load_gdn_weights(
        mesh_device, _CONFIG, None, device_free, cache_name_prefix=prefix, tensor_parallel_axis=tensor_parallel_axis
    )
    for name in ("input_projection", "output_projection", "decay_scale", "decay_bias", "norm"):
        actual, expected = _host_shards(getattr(loaded, name)), _host_shards(getattr(written, name))
        assert len(actual) == mesh_device.get_num_devices()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), name
    assert loaded.decay_scale.dtype == ttnn.float32 and loaded.input_projection.dtype == ttnn.bfloat16
