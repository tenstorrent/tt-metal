# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The release config, the 1x4 placement, the checkpoint's headers and lazy reads, the per-device weight budget
(``tools/checkpoint_budget.py``) and the MTP hyper-connection weight API."""

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import (
    BF4_TILE_BYTES,
    TensorRecord,
    bfp_tile_storage_bytes,
    checkpoint_category,
    plan_a_per_device_weight_bytes,
    vision_resident_layout,
)
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as ttnn_final_mixer
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as ttnn_gr


def test_exact_release_config_and_layer_schedule(checkpoint_dir):
    config = Qwen38Config.from_checkpoint(checkpoint_dir)

    assert config.hidden_size == 2560
    assert config.residual_width == 10240
    assert config.hidden_act == "silu"
    assert config.num_hidden_layers == 48
    assert config.layer_types == ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 12
    assert config.ple_checkpoint_layer == 1
    assert config.gdn_qk_heads == 16
    assert config.gdn_value_heads == 48
    assert config.gdn_output_gate == "sigmoid"
    assert config.qsa_query_heads == 24
    assert config.qsa_kv_heads == 2
    assert config.num_experts == 512
    assert config.top_k == 10
    assert config.norm_topk_prob is True
    assert config.mtp_layers == 1


def test_config_fails_closed_on_shape_only_substitution(tmp_path, expect_error, checkpoint_dir):
    source = json.loads((checkpoint_dir / "config.json").read_text())
    source["text_config"]["linear_num_value_heads"] = 32
    (tmp_path / "config.json").write_text(json.dumps(source))

    with expect_error(ValueError, "linear_num_value_heads"):
        Qwen38Config.from_checkpoint(tmp_path)


def test_exact_tp4_placement_contract(release_placement):
    placement = release_placement
    assert placement.hidden_ranges == ((0, 640), (640, 1280), (1280, 1920), (1920, 2560))
    assert placement.vocab_ranges[-1] == (186240, 248320)
    assert placement.expert_ranges == ((0, 128), (128, 256), (256, 384), (384, 512))
    assert placement.gdn_qk_head_ranges[-1] == (12, 16)
    assert placement.gdn_value_head_ranges[-1] == (36, 48)
    assert placement.qsa_query_head_ranges == ((0, 6), (6, 12), (12, 18), (18, 24))
    assert placement.qsa_kv_device_groups == ((0, 1), (2, 3))
    assert placement.index_query_head_ranges == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert placement.index_key_replicas == (0, 1, 2, 3)
    assert placement.ple_result_ranges == placement.hidden_ranges


@pytest.mark.parametrize(
    ("shape", "physical_ids", "message"),
    [((1, 2), (0, 1), "exactly four"), ((2, 2), (0, 1, 2, 3), "1x4"), ((1, 4), (0, 1, 1, 3), "distinct")],
)
def test_placement_rejects_wrong_or_ambiguous_mesh(shape, physical_ids, message, expect_error, release_checkpoint):
    with expect_error(ValueError, message):
        Qwen38Placement(release_checkpoint.config, mesh_shape=shape, physical_ids=physical_ids)


def test_checkpoint_headers_and_exact_release_totals(release_checkpoint):
    summary = release_checkpoint.validate_all_headers()

    assert summary.shard_count == 131
    assert summary.tensor_count == 1658
    assert summary.tensor_bytes == 359_999_963_128
    assert summary.file_bytes == 360_000_192_888
    assert summary.int64_elements == 35

    metadata = release_checkpoint.metadata("model.language_model.layers.0.linear_attn.A_log")
    assert metadata.shape == (48,)
    assert metadata.dtype == "BF16"


def test_checkpoint_lazy_slice_does_not_materialize_full_expert_tensor(release_checkpoint):
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    metadata = release_checkpoint.metadata(name)
    assert metadata.shape == (512, 1280, 2560)

    expert_zero = release_checkpoint.tensor_slice(name, (slice(0, 1), slice(None), slice(None)))
    assert expert_zero.shape == (1, 1280, 2560)
    assert expert_zero.dtype == torch.bfloat16


def test_ple_table_is_identified_as_host_resident(release_checkpoint, release_placement):
    ple_names = release_checkpoint.names_with_prefix(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    )

    assert len(ple_names) == 128
    assert all(release_placement.classify_tensor(name) == "host_ple" for name in ple_names)


# -- the per-device weight budget --------------------------------------------------------------------------------------


def _tensor(name: str, shape: tuple[int, ...], *, dtype: str = "BF16") -> TensorRecord:
    elements = math.prod(shape)
    return TensorRecord(name, dtype, shape, elements, elements * {"BF16": 2, "I64": 8}[dtype])


def test_every_checkpoint_domain_is_explicit(expect_error):
    cases = {
        "model.visual.blocks.0.norm1.weight": "vision",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "ple_host_table",
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": "ple_host_metadata",
        "model.language_model.layers.1.ple.key_proj.weight": "ple_device_non_table",
        "model.language_model.layers.0.mlp.experts.down_proj": "routed_down",
        "mtp.layers.0.mlp.experts.gate_up_proj": "routed_gate_up",
        "model.language_model.embed_tokens.weight": "token_embedding",
        "lm_head.weight": "lm_head",
        "mtp.fc_hidden.weight": "mtp_nonexpert",
        "model.language_model.layers.0.linear_attn.A_log": "backbone_nonexpert",
    }
    assert {name: checkpoint_category(name) for name in cases} == cases
    with expect_error(ValueError, "unclassified tensor"):
        checkpoint_category("unexpected.weight")


def test_bfp_storage_uses_tenstorrent_tile_payload(expect_error):
    routed_down = _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2560, 640))
    assert bfp_tile_storage_bytes(routed_down, BF4_TILE_BYTES) == 471_859_200
    misaligned = _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2559, 640))
    with expect_error(ValueError, "tile aligned"):
        bfp_tile_storage_bytes(misaligned, BF4_TILE_BYTES)


def test_plan_a_models_tp4_grouped_kv_and_replicated_vectors():
    records = [
        _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2560, 640)),
        _tensor("model.language_model.layers.3.self_attn.k_proj.weight", (512, 2560)),
        _tensor("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight", (640, 2560)),
        _tensor("model.language_model.layers.3.self_attn.q_norm.weight", (256,)),
        _tensor("model.language_model.layers.0.linear_attn.out_proj.weight", (2560, 6144)),
        _tensor("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", (2_500_012, 160)),
        _tensor("model.visual.pos_embed", (2304, 1152)),
    ]

    result = plan_a_per_device_weight_bytes(records, mesh_size=4, ring_size=8)

    assert result["routed_expert_bf4_ep"] == 148_635_648
    assert result["routed_expert_bf4_stream_slot"] == 148_635_648
    assert result["qsa_kv_grouped_bf16"] == 1_310_720
    # Four query index heads are sharded; the single index key head is replicated.
    assert result["qsa_indexer_split_bf16"] == 1_310_720
    assert result["replicated_bf16"] == 512
    assert result["tp4_bf16"] == 7_864_320
    assert result["host_only"] == 800_003_840
    assert result["vision_omitted"] == 5_308_416
    assert result["vision_bf16_resident"] == 0

    resident = plan_a_per_device_weight_bytes(records, mesh_size=4, ring_size=8, vision_resident=True)
    layout = vision_resident_layout(mesh_size=4)
    assert resident["vision_omitted"] == 0
    assert resident["vision_bf16_resident"] == layout["device_total"]
    assert resident["host_only"] == 800_003_840 + layout["host_position_table"]
    assert resident["device_weight_total"] == result["device_weight_total"] + layout["device_total"]


def test_vision_resident_layout_is_the_tile_padded_device_tower():
    layout = vision_resident_layout(mesh_size=4)
    # 27 blocks: qkv 1152 x 4608, proj 1536 x 1152, fc1 1152 x 4320, fc2 4320 x 1152 (BF16), plus the vectors.
    block_matrices = 27 * 2 * (1152 * 4608 + 1536 * 1152 + 1152 * 4320 + 4320 * 1152)
    assert layout["blocks"] > block_matrices
    assert layout["blocks"] - block_matrices < 27 * 32 * 2 * (4 * 1152 + 4608 + 1152 + 4320 + 1152) + 1
    assert layout["host_position_table"] == 2304 * 1152 * 2
    assert layout["device_total"] == layout["patch_embed"] + layout["blocks"] + layout["merger"] + 2048
    assert vision_resident_layout(mesh_size=1)["merger"] > layout["merger"]


def test_expert_parallel_fails_if_expert_axis_is_not_divisible(expect_error):
    with expect_error(ValueError, "expert axis"):
        plan_a_per_device_weight_bytes(
            [_tensor("mtp.layers.0.mlp.experts.down_proj", (510, 2560, 640))], mesh_size=4, ring_size=8
        )


def test_ring_aware_packed_expert_budget():
    records = [
        _tensor("model.language_model.layers.0.mlp.experts.gate_up_proj", (512, 1280, 2560)),
        _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2560, 640)),
    ]
    ring7 = plan_a_per_device_weight_bytes(records, ring_size=7)
    ring8 = plan_a_per_device_weight_bytes(records, ring_size=8)
    assert ring7["routed_expert_bf4_ep"] == 476_872_704
    assert ring8["routed_expert_bf4_ep"] == 544_997_376
    assert ring7["device_weight_total_streamed"] == ring7["device_weight_total"]


# -- the MTP hyper-connection weight API -------------------------------------------------------------------------------

# The tensors the MTP GR and final-mixer constructors bind, in their request order.
MTP_HYPER_CONNECTION_SHAPES = {
    **{
        f"mtp.layers.0.{block}_hyper_connection.{field}": shape
        for block in ("attn", "mlp")
        for field, shape in (
            ("hc_norm.weight", (10_240,)),
            ("input_mix_weight_down.weight", (320, 10_240)),
            ("input_mix_weight_up.weight", (10_240, 320)),
            ("block_inject_weight.weight", (4, 10_240)),
        )
    },
    "mtp.hyper_connection_mixer.hc_norm.weight": (10_240,),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10_240),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (10_240, 320),
}


class _RecordingCheckpoint:
    """Serves each requested hyper-connection tensor's shape (BF16) and records the names in order."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=48, mtp_layers=1, residual_width=10_240, residual_rank=320, residual_branches=4
        )
        self.names: list[str] = []

    def tensor(self, name: str) -> SimpleNamespace:
        self.names.append(name)
        return SimpleNamespace(shape=MTP_HYPER_CONNECTION_SHAPES[name], dtype=torch.bfloat16)


def _mtp_weights(checkpoint, placement):
    """The MTP attention GR, MLP GR and final mixer weights."""

    return (
        Qwen38GatedResidualWeights.from_mtp_checkpoint(checkpoint, placement, mtp_layer_index=0, block="attn"),
        Qwen38GatedResidualWeights.from_mtp_checkpoint(checkpoint, placement, mtp_layer_index=0, block="mlp"),
        Qwen38FinalMixerWeights.from_mtp_checkpoint(checkpoint, placement),
    )


def test_mtp_gr_and_final_mixer_resolve_exact_released_prefixes() -> None:
    checkpoint = _RecordingCheckpoint()
    placement = SimpleNamespace(config=checkpoint.config)
    attention, mlp, final = _mtp_weights(checkpoint, placement)

    assert checkpoint.names == list(MTP_HYPER_CONNECTION_SHAPES)
    assert (attention.layer_index, attention.block) == (0, "attn")
    assert (mlp.layer_index, mlp.block) == (0, "mlp")
    assert final.placement is placement


def test_real_checkpoint_exposes_every_bound_mtp_hyper_connection_tensor(release_checkpoint) -> None:
    for name, shape in MTP_HYPER_CONNECTION_SHAPES.items():
        metadata = release_checkpoint.metadata(name)
        assert (metadata.dtype, metadata.shape) == ("BF16", shape), name


def test_real_checkpoint_mtp_weight_constructors_load_the_exact_dense_tensors(release_checkpoint) -> None:
    placement = Qwen38Placement(release_checkpoint.config, mesh_shape=(1, 4), physical_ids=(1, 0, 2, 3))
    attention, mlp, final = _mtp_weights(release_checkpoint, placement)

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


def test_invalid_mtp_layer_and_block_fail_before_tensor_access(expect_error) -> None:
    checkpoint = _RecordingCheckpoint()
    placement = SimpleNamespace(config=checkpoint.config)
    for layer_index, block in ((1, "attn"), (0, "invalid")):
        with expect_error(ValueError):
            Qwen38GatedResidualWeights.from_mtp_checkpoint(
                checkpoint, placement, mtp_layer_index=layer_index, block=block
            )
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
