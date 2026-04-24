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
from models.demos.deepseek_v4_flash.cpu_reference import swiglu_expert
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

ttnn = pytest.importorskip("ttnn")

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

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=1)
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
