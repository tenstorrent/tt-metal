# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    dflash_full_drafter_device_fit,
    load_full_dflash_reference,
)


APPROACH = "dflash"
TARGET_MODEL = "moonshotai/Kimi-K2.5"
DRAFT_MODEL = "z-lab/Kimi-K2.5-DFlash"
REFERENCE_DIR = Path(__file__).parents[1] / "references" / "dflash"
REFERENCE_PATH = REFERENCE_DIR / "reference_kimi_k2_5_dflash.pt"


def require_keys(value: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    assert not missing, f"{context} missing required key(s): {missing}"


def load_validated_dflash_reference(path: Path = REFERENCE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            "DFlash reference is a local artifact. Generate it with "
            f"{REFERENCE_DIR / 'capture_real_reference.py'} --num-tokens 8 before running these tests."
        )
    reference = load_full_dflash_reference(path)
    assert_dflash_reference_metadata(reference)
    return reference


def assert_dflash_identity(value: dict, context: str) -> None:
    assert value["approach"] == APPROACH, context
    assert value["target_model"] == TARGET_MODEL, context
    assert value["draft_model"] == DRAFT_MODEL, context


def assert_dflash_reference_metadata(reference: dict) -> None:
    require_keys(reference, ("schema_version", "metadata", "config", "prompt", "host_trace", "stages"), "reference")
    metadata = reference["metadata"]
    config = reference["config"]
    prompt = reference["prompt"]
    host_trace = reference["host_trace"]
    passes = _reference_passes(reference)
    num_layers = int(config["num_hidden_layers"])
    block_size = int(config["runtime_block_size"])

    assert reference["schema_version"] == 1
    assert metadata["approach"] == APPROACH
    assert metadata["target_model"] == TARGET_MODEL
    assert metadata["draft_model"] == DRAFT_MODEL
    assert metadata["dataset"] == "openai/openai_humaneval"
    assert int(metadata["max_new_tokens"]) > 0
    assert metadata["weights_policy"].startswith("Drafter and target weights are loaded from checkpoint/state dict")
    assert int(metadata["block_size"]) == block_size
    assert_stage_config_metadata(config, "reference.config")

    require_keys(prompt, ("text", "rendered_chat_prompt", "input_ids"), "reference.prompt")
    assert isinstance(prompt["text"], str) and prompt["text"]
    assert isinstance(prompt["rendered_chat_prompt"], str) and prompt["rendered_chat_prompt"]
    assert prompt["input_ids"].ndim == 2
    assert int(prompt["input_ids"].shape[0]) == 1

    require_keys(
        host_trace,
        (
            "params",
            "num_input_tokens",
            "num_output_tokens",
            "generated_token_ids",
            "output_token_ids",
            "acceptance_lengths",
            "verification_passes",
            "committed_tokens",
            "num_accepts",
            "num_rejects",
            "average_committed_tokens",
            "host_writes",
            "draft_blocks",
            "target_verification_packets",
        ),
        "reference.host_trace",
    )
    assert host_trace["params"]["prompt_token_ids"] == prompt["input_ids"][0].tolist()
    assert int(host_trace["params"]["max_new_tokens"]) == int(metadata["max_new_tokens"])
    assert int(host_trace["params"]["block_size"]) == block_size
    assert int(host_trace["num_input_tokens"]) == int(prompt["input_ids"].shape[1])
    assert int(host_trace["num_output_tokens"]) == len(host_trace["generated_token_ids"])
    assert int(host_trace["num_output_tokens"]) <= int(metadata["max_new_tokens"])
    assert len(host_trace["output_token_ids"]) == int(host_trace["num_input_tokens"]) + int(
        host_trace["num_output_tokens"]
    )
    assert len(host_trace["acceptance_lengths"]) == int(host_trace["verification_passes"])
    assert len(host_trace["draft_blocks"]) == int(host_trace["verification_passes"])
    assert len(host_trace["target_verification_packets"]) == int(host_trace["verification_passes"])
    assert len(passes) == int(host_trace["verification_passes"])
    assert int(host_trace["num_accepts"]) == sum(
        int(packet["accepted_after_anchor"]) for packet in host_trace["target_verification_packets"]
    )
    assert int(host_trace["num_rejects"]) == sum(
        int(packet["accepted_after_anchor"]) < block_size - 1
        for packet in host_trace["target_verification_packets"]
    )

    for pass_idx, pass_record in enumerate(passes):
        assert int(pass_record["pass_index"]) == pass_idx
        _assert_pass_metadata(pass_record, num_layers=num_layers, block_size=block_size)
        draft_block = host_trace["draft_blocks"][pass_idx]
        packet = host_trace["target_verification_packets"][pass_idx]
        assert int(draft_block["pass_index"]) == pass_idx
        assert int(packet["pass_index"]) == pass_idx
        assert int(draft_block["anchor_pos"]) == int(pass_record["anchor_position"])
        assert int(packet["anchor_pos"]) == int(pass_record["anchor_position"])


def assert_stage_config_metadata(config: dict, context: str) -> None:
    require_keys(
        config,
        (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "block_size",
            "runtime_block_size",
            "target_layer_ids",
        ),
        context,
    )
    assert int(config["vocab_size"]) > 0
    assert int(config["hidden_size"]) > 0
    assert int(config["intermediate_size"]) > 0
    assert int(config["num_hidden_layers"]) > 0
    assert int(config["num_attention_heads"]) > 0
    assert int(config["num_key_value_heads"]) > 0
    assert int(config["block_size"]) > 1
    assert int(config["runtime_block_size"]) > 1
    assert config["target_layer_ids"]


def expected_decode_manager_outputs(reference: dict) -> dict[str, Any]:
    host_trace = reference["host_trace"]
    block_size = int(reference["config"]["runtime_block_size"])
    packets = host_trace["target_verification_packets"]
    return {
        "generated_tokens": host_trace["generated_token_ids"],
        "emitted_tokens": host_trace["generated_token_ids"],
        "output_token_ids": host_trace["output_token_ids"],
        "acceptance_lengths": host_trace["acceptance_lengths"],
        "writes": [_normalize_host_write(write) for write in host_trace["host_writes"]],
        "read_count": int(host_trace["verification_passes"]),
        "stats": {
            "num_accepts": int(host_trace["num_accepts"]),
            "num_rejects": int(host_trace["num_rejects"]),
            "average_committed_tokens": float(host_trace["average_committed_tokens"]),
        },
    }


def assert_decode_manager_outputs_match(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual["generated_tokens"] == expected["generated_tokens"]
    assert actual["emitted_tokens"] == expected["emitted_tokens"]
    assert actual["output_token_ids"] == expected["output_token_ids"]
    assert actual["acceptance_lengths"] == expected["acceptance_lengths"]
    assert actual["writes"] == expected["writes"]
    assert actual["read_count"] == expected["read_count"]
    assert actual["stats"]["num_accepts"] == expected["stats"]["num_accepts"]
    assert actual["stats"]["num_rejects"] == expected["stats"]["num_rejects"]
    assert math.isclose(actual["stats"]["average_committed_tokens"], expected["stats"]["average_committed_tokens"])


def expected_pre_decoder_outputs(reference: dict) -> list[dict[str, Any]]:
    return [
        {
            "pass_index": int(pass_record["pass_index"]),
            "anchor_position": int(pass_record["anchor_position"]),
            "target_context": pass_record["pre_decoder_fused"]["expected"]["target_context"],
            "position_cos": pass_record["pre_decoder_fused"]["expected"]["position_cos"],
            "position_sin": pass_record["pre_decoder_fused"]["expected"]["position_sin"],
            "decoder_input": pass_record["pre_decoder_fused"]["expected"]["decoder_input"],
        }
        for pass_record in _reference_passes(reference)
    ]


def expected_decoder_layer_outputs(reference: dict) -> list[dict[str, Any]]:
    expected = []
    for pass_record in _reference_passes(reference):
        for layer_record in pass_record["decoder_layers"]:
            expected.append(
                {
                    "pass_index": int(pass_record["pass_index"]),
                    "anchor_position": int(pass_record["anchor_position"]),
                    "layer_idx": int(layer_record["layer_idx"]),
                    "hidden_states": layer_record["expected"]["hidden_states"],
                }
            )
    return expected


def expected_post_decoder_outputs(reference: dict) -> list[dict[str, Any]]:
    return [
        _stage_output_from_expected(pass_record, pass_record["post_decoder_fused"]["expected"])
        for pass_record in _reference_passes(reference)
    ]


def expected_combined_drafter_outputs(reference: dict) -> list[dict[str, Any]]:
    return [
        _stage_output_from_expected(pass_record, pass_record["combined_drafter"]["expected"])
        for pass_record in _reference_passes(reference)
    ]


def assert_stage_sequence_outputs_match(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    tensor_fields: tuple[str, ...],
    exact_fields: tuple[str, ...] = (),
) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        assert actual_item["pass_index"] == expected_item["pass_index"]
        if "anchor_position" in expected_item:
            assert actual_item["anchor_position"] == expected_item["anchor_position"]
        if "layer_idx" in expected_item:
            assert actual_item["layer_idx"] == expected_item["layer_idx"]
        for field in tensor_fields:
            assert_tensor_matches_reference(actual_item[field], expected_item[field])
        for field in exact_fields:
            assert normalize_reference_value(actual_item[field]) == normalize_reference_value(expected_item[field])


def expected_full_drafter_outputs(reference: dict) -> dict[str, Any]:
    host_trace = reference["host_trace"]
    return {
        "device_fit": dflash_full_drafter_device_fit(reference["config"]),
        "host_trace": {
            "generated_token_ids": host_trace["generated_token_ids"],
            "output_token_ids": host_trace["output_token_ids"],
            "acceptance_lengths": host_trace["acceptance_lengths"],
            "verification_passes": host_trace["verification_passes"],
            "average_committed_tokens": host_trace["average_committed_tokens"],
        },
        "stage_outputs": expected_combined_drafter_outputs(reference),
    }


def assert_full_drafter_outputs_match(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual["device_fit"] == expected["device_fit"]
    assert actual["device_fit"]["full_model_stage_count"] >= 3
    assert actual["device_fit"]["stage_mesh_shape"] == [4, 2]
    assert actual["device_fit"]["galaxies_required"] >= 1
    assert actual["host_trace"] == expected["host_trace"]
    assert_stage_sequence_outputs_match(
        actual["stage_outputs"],
        expected["stage_outputs"],
        tensor_fields=("final_hidden", "draft_logits", "draft_token_ids"),
        exact_fields=("host_packet",),
    )


def assert_tensor_matches_reference(actual: Any, expected: Any, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    actual_tensor = actual if isinstance(actual, torch.Tensor) else torch.tensor(actual)
    expected_tensor = expected if isinstance(expected, torch.Tensor) else torch.tensor(expected)
    torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)


def normalize_reference_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _reference_passes(reference: dict) -> list[dict[str, Any]]:
    return list(reference["stages"]["passes"])


def _assert_pass_metadata(pass_record: dict, *, num_layers: int, block_size: int) -> None:
    require_keys(
        pass_record,
        (
            "pass_index",
            "anchor_position",
            "pre_decoder_fused",
            "decoder_layers",
            "post_decoder_fused",
            "combined_drafter",
            "tensor_shapes",
        ),
        f"pass_{pass_record.get('pass_index', 'unknown')}",
    )
    require_keys(
        pass_record["pre_decoder_fused"]["expected"],
        ("target_context", "position_cos", "position_sin", "decoder_input"),
        "pre_decoder_fused.expected",
    )
    assert len(pass_record["decoder_layers"]) == num_layers
    assert [int(layer["layer_idx"]) for layer in pass_record["decoder_layers"]] == list(range(num_layers))
    for layer_record in pass_record["decoder_layers"]:
        require_keys(layer_record["expected"], ("hidden_states",), f"decoder_layer_{layer_record['layer_idx']}")

    combined = pass_record["combined_drafter"]["expected"]
    post = pass_record["post_decoder_fused"]["expected"]
    require_keys(
        post,
        ("final_hidden", "draft_logits", "draft_token_ids", "host_packet"),
        "post_decoder_fused.expected",
    )
    require_keys(combined, ("final_hidden", "draft_logits", "draft_token_ids", "host_packet"), "combined.expected")
    for field_name in ("final_hidden", "draft_logits", "draft_token_ids"):
        torch.testing.assert_close(combined[field_name], post[field_name], atol=0, rtol=0)
        assert list(combined[field_name].shape) == pass_record["tensor_shapes"][field_name]["shape"]
    assert combined["host_packet"] == post["host_packet"]
    assert len(post["host_packet"]["token_ids"]) == block_size - 1
    assert len(post["host_packet"]["positions"]) == block_size - 1


def _stage_output_from_expected(pass_record: dict, expected: dict) -> dict[str, Any]:
    return {
        "pass_index": int(pass_record["pass_index"]),
        "anchor_position": int(pass_record["anchor_position"]),
        "final_hidden": expected["final_hidden"],
        "draft_logits": expected["draft_logits"],
        "draft_token_ids": expected["draft_token_ids"],
        "host_packet": expected["host_packet"],
    }


def _normalize_host_write(write: dict) -> dict[str, int | str]:
    return {
        "token_id": int(write["token_id"]),
        "type": str(write["token_type"]),
        "pos": int(write["position_id"]),
        "user_id": int(write["user_id"]),
        "prefill_id": int(write["prefill_token_id"]),
    }
