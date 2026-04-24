# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.cpu_reference import combine_routed_experts, compressor_prefill, swiglu_expert
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_flash.expert_abi import load_packed_expert_weight
from models.demos.deepseek_v4_flash.ttnn_prefill_compressor import TtPrefillCompressor, load_prefill_compressor_weights
from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP


@pytest.fixture(scope="module")
def tiny_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_device_smoke_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_device_smoke")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=3)
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture(scope="module")
def device():
    try:
        num_devices = ttnn.GetNumAvailableDevices()
    except Exception as exc:
        pytest.skip(f"Unable to query TT devices: {exc}")
    if num_devices < 1:
        pytest.skip("No TT devices available")

    device_id = int(os.environ.get("TTNN_DEVICE_ID", "0"))
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        yield device
    finally:
        ttnn.close_device(device)


def test_tiny_hyperconnection_residual_shape_roundtrips_on_device(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    hidden_size = manifest["config"]["hidden_size"]
    hc_mult = manifest["config"]["hc_mult"]
    torch_tensor = torch.linspace(-1.0, 1.0, steps=hc_mult * hidden_size, dtype=torch.float32).reshape(
        1, 1, hc_mult, hidden_size
    )
    torch_tensor = torch_tensor.to(torch.bfloat16)

    tt_tensor = ttnn.from_torch(torch_tensor, device=device, dtype=ttnn.bfloat16)
    torch_output = ttnn.to_torch(tt_tensor)

    torch.testing.assert_close(torch_output, torch_tensor)


def test_tiny_shared_expert_w1_projection_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    non_expert_path = tiny_tt_preprocessed_checkpoint / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_path, framework="pt", device="cpu") as handle:
        w1 = handle.get_tensor("layers.0.ffn.shared_experts.w1.weight").to(torch.bfloat16)

    hidden_size = w1.shape[-1]
    intermediate_size = w1.shape[0]
    torch_input = torch.linspace(-0.5, 0.5, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    torch_weight = w1.T.contiguous().reshape(1, 1, hidden_size, intermediate_size)
    expected = torch.matmul(torch_input.float(), torch_weight.float())

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.matmul(tt_input, tt_weight)
    torch_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(torch_output.float(), expected, rtol=2e-2, atol=2e-2)


def test_tiny_shared_expert_mlp_module_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    non_expert_path = tiny_tt_preprocessed_checkpoint / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_path, framework="pt", device="cpu") as handle:
        w1 = handle.get_tensor("layers.0.ffn.shared_experts.w1.weight").to(torch.bfloat16)
        w2 = handle.get_tensor("layers.0.ffn.shared_experts.w2.weight").to(torch.bfloat16)
        w3 = handle.get_tensor("layers.0.ffn.shared_experts.w3.weight").to(torch.bfloat16)

    hidden_size = w1.shape[-1]
    torch_input = torch.linspace(-0.25, 0.25, steps=32 * hidden_size, dtype=torch.float32).reshape(
        1, 1, 32, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    expected = swiglu_expert(
        torch_input.reshape(-1, hidden_size),
        w1,
        w2,
        w3,
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).view_as(torch_input)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtSharedExpertMLP.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device)
    torch_output = ttnn.to_torch(module(tt_input))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_tiny_routed_expert_mlp_module_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    expert_id = 2
    weights = {
        projection: load_packed_expert_weight(
            tiny_tt_preprocessed_checkpoint, layer=0, expert=expert_id, projection=projection
        ).dequantize(dtype=torch.bfloat16)
        for projection in ("w1", "w2", "w3")
    }

    hidden_size = weights["w1"].shape[-1]
    intermediate_size = weights["w1"].shape[0]
    torch_input = torch.linspace(-0.2, 0.2, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    route_weights = torch.linspace(0.25, 1.0, steps=32, dtype=torch.float32).reshape(1, 32, 1)
    route_indices = torch.full((1, 32, 1), expert_id, dtype=torch.int64)
    expected = combine_routed_experts(
        torch_input.reshape(1, 32, hidden_size),
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"])},
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_route_weights = ttnn.from_torch(
        route_weights.reshape(1, 1, 32, 1).expand(1, 1, 32, intermediate_size).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    module = TtRoutedExpertMLP.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, expert=expert_id)
    torch_output = ttnn.to_torch(module(tt_input, route_weight=tt_route_weights))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_tiny_prefill_compressor_overlap_uses_host_ratio_pooling_and_matches_torch(
    tiny_tt_preprocessed_checkpoint: Path, device
):
    """Smoke the first compressor prefill slice; overlap pooling is still host-side."""

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    layer = 2
    weights = load_prefill_compressor_weights(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=layer)
    hidden_size = manifest["config"]["hidden_size"]
    head_dim = manifest["config"]["head_dim"]
    compress_ratio = manifest["config"]["compress_ratios"][layer]
    seq_len = 32
    torch_input = torch.linspace(-0.1, 0.1, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    expected = compressor_prefill(
        torch_input[:, 0],
        weights.wkv,
        weights.wgate,
        weights.ape,
        weights.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        norm_eps=float(manifest["config"]["rms_norm_eps"]),
        overlap=True,
    )

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtPrefillCompressor.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=layer)
    torch_output = ttnn.to_torch(module(tt_input))[:, 0]

    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=8e-2, atol=8e-2)
