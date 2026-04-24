# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.expert_abi import load_packed_expert_weight
from models.demos.deepseek_v4_flash.fp4 import FP4_E2M1_TABLE
from models.demos.deepseek_v4_flash.mesh_config import MeshConfig, ModeConfig, mesh_1x8, mesh_2x4, mesh_for_shape
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint


def test_mesh_config_t3k_and_n150_semantics():
    n150 = mesh_1x8()
    assert n150.mesh_shape == (1, 8)
    assert n150.decode == ModeConfig(tp=8, ep=1, sp=1)
    assert n150.prefill == ModeConfig(tp=8, ep=1, sp=1)
    assert n150.data_parallel("decode") == 1
    assert n150.shard_size(4096) == 512

    t3k = mesh_2x4()
    assert t3k.decode == ModeConfig(tp=4, ep=2, sp=1)
    assert t3k.prefill == ModeConfig(tp=4, ep=1, sp=2)
    assert t3k.data_parallel("decode") == 1
    assert t3k.data_parallel("prefill") == 2
    assert t3k.to_manifest_dict()["prefill"] == {"tp": 4, "ep": 1, "sp": 2, "dp": 2}
    assert "decode[TP=4, EP=2, SP=1, DP=1]" in repr(t3k)

    assert mesh_for_shape((2, 4)).to_manifest_dict() == t3k.to_manifest_dict()


def test_mesh_config_rejects_invalid_axis_usage():
    try:
        MeshConfig((2, 4), decode=ModeConfig(tp=8, ep=1))
    except ValueError as exc:
        assert "TP(8)" in str(exc)
    else:
        raise AssertionError("Expected invalid TP to fail")

    try:
        MeshConfig((2, 4), decode=ModeConfig(tp=4, ep=3))
    except ValueError as exc:
        assert "EP(3)" in str(exc)
    else:
        raise AssertionError("Expected invalid EP to fail")


def test_packed_expert_abi_loads_and_dequantizes_tiny_fixture(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=1)
    preprocessed = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    packed = load_packed_expert_weight(preprocessed, layer=0, expert=0, projection="w1")

    assert packed.abi == "deepseek_v4_flash.fp4_e2m1fn_x2.block32.v1"
    assert packed.weight_packed.dtype == torch.uint8
    assert packed.scale.shape == (32, 1)

    dequant = packed.dequantize(dtype=torch.float32)
    assert dequant.shape == (32, 32)
    expected_first_row = FP4_E2M1_TABLE[torch.arange(32) % 16]
    torch.testing.assert_close(dequant[0], expected_first_row)
