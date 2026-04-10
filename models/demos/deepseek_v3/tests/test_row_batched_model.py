#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight

pytestmark = pytest.mark.t3k_compat


@dataclass(frozen=True)
class _FakeMeshDevice:
    shape: tuple[int, int]


def _make_hf_config() -> PretrainedConfig:
    hf_config = PretrainedConfig()
    hf_config.first_k_dense_replace = 1
    hf_config.num_hidden_layers = 2
    hf_config.num_nextn_predict_layers = 1
    return hf_config


def test_row_batched_model_reuses_base_embedding_and_head_for_mtp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hf_config = _make_hf_config()
    mesh_device = _FakeMeshDevice(shape=(2, 4))

    embedding_weight_cfg = {
        "weight": SavedWeight(path=Path("embedding/base.tensorbin"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    }
    head_weight_cfg = {
        "input_tensor_b": SavedWeight(path=Path("lm_head/base.tensorbin"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    }
    captured_reuse_cfgs: dict[str, object] = {}

    def fake_embedding_convert_weights(_hf_config, _state_dicts, _output_path, _mesh_device):
        return embedding_weight_cfg

    def fake_decoder_convert_weights(_hf_config, _state_dicts, _output_path, _mesh_device):
        return {"decoder": SavedWeight(path=Path("decoder.tensorbin"), memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    def fake_norm_convert_weights(_hf_config, _state_dicts, _output_path, _mesh_device):
        return {"norm": SavedWeight(path=Path("norm.tensorbin"), memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    def fake_head_convert_weights(_hf_config, _state_dicts, _output_path, _mesh_device):
        return head_weight_cfg

    def fake_mtp_convert_weights(
        _hf_config,
        _state_dicts,
        _output_path,
        _mesh_device,
        *,
        reuse_embedding_weight_cfg=None,
        reuse_head_weight_cfg=None,
    ):
        captured_reuse_cfgs["embedding"] = reuse_embedding_weight_cfg
        captured_reuse_cfgs["head"] = reuse_head_weight_cfg
        return {"embedding": reuse_embedding_weight_cfg, "head": reuse_head_weight_cfg}

    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.Embedding2D.convert_weights",
        fake_embedding_convert_weights,
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.DecoderBlock2D.convert_weights",
        fake_decoder_convert_weights,
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.MoEDecoderBlock2D.convert_weights",
        fake_decoder_convert_weights,
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.DistributedRMSNorm.convert_weights",
        fake_norm_convert_weights,
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.LMHead1D.convert_weights",
        fake_head_convert_weights,
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.model.row_batched_model.MTP2D.convert_weights",
        fake_mtp_convert_weights,
    )

    state_dict = {
        "model.embed_tokens.weight": torch.empty(1),
        "model.layers.2.eh_proj.weight": torch.empty(1),
        "lm_head.weight": torch.empty(1),
    }

    weight_cfg = RowBatchedModel.convert_weights(hf_config, [state_dict], tmp_path, mesh_device)

    assert captured_reuse_cfgs["embedding"] is embedding_weight_cfg
    assert captured_reuse_cfgs["head"] is head_weight_cfg
    assert weight_cfg["mtp"]["embedding"] is weight_cfg["embedding"]
    assert weight_cfg["mtp"]["head"] is weight_cfg["lm_head"]
