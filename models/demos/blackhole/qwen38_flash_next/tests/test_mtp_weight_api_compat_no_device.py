# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device proof for MTP hyper-connection weight API compatibility."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as ttnn_final_mixer
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as ttnn_gr

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))


class _ShapeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = torch.bfloat16


class _RecordingCheckpoint:
    def __init__(self, config: object) -> None:
        self.config = config
        self.names: list[str] = []

    def tensor(self, name: str) -> _ShapeTensor:
        self.names.append(name)
        suffix = name.rsplit(".", 2)[-2:]
        if suffix == ["hc_norm", "weight"]:
            shape = (self.config.residual_width,)
        elif suffix == ["input_mix_weight_down", "weight"]:
            shape = (self.config.residual_rank, self.config.residual_width)
        elif suffix == ["input_mix_weight_up", "weight"]:
            shape = (self.config.residual_width, self.config.residual_rank)
        elif suffix == ["block_inject_weight", "weight"]:
            shape = (self.config.residual_branches, self.config.residual_width)
        else:
            raise AssertionError(f"unexpected tensor request {name}")
        return _ShapeTensor(shape)


def _fixture() -> tuple[_RecordingCheckpoint, SimpleNamespace]:
    config = SimpleNamespace(
        num_hidden_layers=48,
        mtp_layers=1,
        residual_width=10_240,
        residual_rank=320,
        residual_branches=4,
    )
    return _RecordingCheckpoint(config), SimpleNamespace(config=config)


def test_mtp_gr_and_final_mixer_resolve_exact_released_prefixes() -> None:
    checkpoint, placement = _fixture()

    attention = Qwen38GatedResidualWeights.from_mtp_checkpoint(
        checkpoint,
        placement,
        mtp_layer_index=0,
        block="attn",
    )
    mlp = Qwen38GatedResidualWeights.from_mtp_checkpoint(
        checkpoint,
        placement,
        mtp_layer_index=0,
        block="mlp",
    )
    final = Qwen38FinalMixerWeights.from_mtp_checkpoint(checkpoint, placement)

    gr_fields = (
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    )
    expected = [f"mtp.layers.0.attn_hyper_connection.{field}" for field in gr_fields]
    expected += [f"mtp.layers.0.mlp_hyper_connection.{field}" for field in gr_fields]
    expected += [
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    ]
    assert checkpoint.names == expected
    assert (attention.layer_index, attention.block) == (0, "attn")
    assert (mlp.layer_index, mlp.block) == (0, "mlp")
    assert final.placement is placement


def test_real_checkpoint_exposes_every_bound_mtp_hyper_connection_tensor() -> None:
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    expected = {
        "hc_norm.weight": (10_240,),
        "input_mix_weight_down.weight": (320, 10_240),
        "input_mix_weight_up.weight": (10_240, 320),
        "block_inject_weight.weight": (4, 10_240),
    }
    for block in ("attn", "mlp"):
        prefix = f"mtp.layers.0.{block}_hyper_connection."
        for suffix, shape in expected.items():
            metadata = checkpoint.metadata(prefix + suffix)
            assert metadata.dtype == "BF16"
            assert metadata.shape == shape
    final_expected = {
        "hc_norm.weight": (10_240,),
        "input_mix_weight_down.weight": (320, 10_240),
        "input_mix_weight_up.weight": (10_240, 320),
    }
    for suffix, shape in final_expected.items():
        metadata = checkpoint.metadata("mtp.hyper_connection_mixer." + suffix)
        assert metadata.dtype == "BF16"
        assert metadata.shape == shape


def test_real_checkpoint_mtp_weight_constructors_load_the_exact_dense_tensors() -> None:
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(1, 0, 2, 3))

    attention = Qwen38GatedResidualWeights.from_mtp_checkpoint(
        checkpoint,
        placement,
        mtp_layer_index=0,
        block="attn",
    )
    mlp = Qwen38GatedResidualWeights.from_mtp_checkpoint(
        checkpoint,
        placement,
        mtp_layer_index=0,
        block="mlp",
    )
    final = Qwen38FinalMixerWeights.from_mtp_checkpoint(checkpoint, placement)

    for weights in (attention, mlp):
        assert weights.norm.shape == (10_240,)
        assert weights.down.shape == (320, 10_240)
        assert weights.up.shape == (10_240, 320)
        assert weights.inject.shape == (4, 10_240)
        assert all(value.dtype == torch.bfloat16 for value in weights.transformers_state_dict().values())
    assert final.norm.shape == (10_240,)
    assert final.down.shape == (320, 10_240)
    assert final.up.shape == (10_240, 320)
    assert all(value.dtype == torch.bfloat16 for value in final.transformers_state_dict().values())


def test_invalid_mtp_layer_and_block_fail_before_tensor_access() -> None:
    checkpoint, placement = _fixture()

    for layer_index, block in ((1, "attn"), (0, "invalid")):
        try:
            Qwen38GatedResidualWeights.from_mtp_checkpoint(
                checkpoint,
                placement,
                mtp_layer_index=layer_index,
                block=block,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid MTP GR request {(layer_index, block)} passed")
    assert checkpoint.names == []


def test_ttnn_callers_match_the_released_cpu_weight_signatures() -> None:
    gr_parameters = inspect.signature(Qwen38GatedResidualWeights.from_mtp_checkpoint).parameters
    final_parameters = inspect.signature(Qwen38FinalMixerWeights.from_mtp_checkpoint).parameters
    assert set(gr_parameters) == {"checkpoint", "placement", "mtp_layer_index", "block"}
    assert set(final_parameters) == {"checkpoint", "placement"}

    gr_source = Path(ttnn_gr.__file__).read_text(encoding="utf-8")
    final_source = Path(ttnn_final_mixer.__file__).read_text(encoding="utf-8")
    assert "Qwen38GatedResidualWeights.from_mtp_checkpoint(" in gr_source
    assert "mtp_layer_index=layer_index" in gr_source
    assert "Qwen38FinalMixerWeights.from_mtp_checkpoint(checkpoint, placement)" in final_source
