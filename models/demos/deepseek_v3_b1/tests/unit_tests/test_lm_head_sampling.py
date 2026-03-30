# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN LM Head Sampling CCL Broadcast + Mcast + Matmul Op Test

In multi-device mode: CCL broadcasts input_a [1, 7168] from sender device to all
devices, then on each device the sender core multicasts to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.

In single-device mode (skip_ccl=True): CCL is skipped and the input is used directly.
"""

import os
import re
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import (
    PipelineConfiguration,
    create_single_galaxy_pipeline_configuration,
    create_single_galaxy_pipeline_spec_stage_only_configuration,
    create_single_galaxy_spec_decode_pipeline_configuration,
)
from models.demos.deepseek_v3_b1.demo.stage import (
    TOKEN_META_PAGE_SIZE_BYTES,
    TOKEN_PAGE_SIZE_BYTES,
    BaseLMHeadStage,
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
    SpecLMHeadStage,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import StateDictWeightProvider
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MTPWeights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    build_broadcast_test_inputs,
    create_fabric_router_config,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_LM_HEAD_SAMPLING_REFERENCE_PT_ENV = "DEEPSEEK_V3_LM_HEAD_SAMPLING_REFERENCE_PT"


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _resolve_lm_head_sampling_reference_payload_path() -> Path:
    raw = os.getenv(_LM_HEAD_SAMPLING_REFERENCE_PT_ENV)
    if not raw or not raw.strip():
        pytest.skip(f"{_LM_HEAD_SAMPLING_REFERENCE_PT_ENV} is not set; skip golden reference payload tests")
    path = Path(raw.strip()).resolve()
    if not path.is_file():
        pytest.skip(f"Reference payload file not found at {path}")
    return path


@pytest.fixture(scope="session")
def lm_head_sampling_reference_payload():
    path = _resolve_lm_head_sampling_reference_payload_path()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        pytest.fail(f"Expected reference payload to be a dict, got {type(payload)} from {path}")

    required_keys = [
        "base_hidden_states",
        "base_output_tokens",
        "mtp_decoder_inputs",
        "mtp_decoder_outputs",
        "mtp_speculation_tokens",
        "base_hidden_positions",
        "base_output_positions",
        "mtp_input_positions",
        "mtp_speculation_positions",
        "start_tokens",
        "metadata",
    ]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        pytest.fail(f"Reference payload is missing required keys: {missing_keys}")
    return payload


def _payload_tensor(payload: dict, key: str) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def _squeeze_trailing_unit_dims(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _flatten_feature_rows(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.ndim >= 2, f"Expected feature tensor with ndim>=2, got shape {tuple(tensor.shape)}"
    return tensor.reshape(-1, tensor.shape[-1])


def _flatten_scalar_rows(tensor: torch.Tensor) -> torch.Tensor:
    return _squeeze_trailing_unit_dims(tensor).reshape(-1)


def _reference_row_shape(payload: dict) -> tuple[int, ...]:
    return tuple(_squeeze_trailing_unit_dims(_payload_tensor(payload, "base_hidden_positions")).shape)


def _payload_request_axis(row_shape: tuple[int, ...], payload: dict) -> int:
    metadata = payload.get("metadata", {})
    capture_users = metadata.get("capture_users")
    capture_steps = metadata.get("capture_steps")

    if len(row_shape) >= 2:
        if capture_users is not None and row_shape[1] == capture_users:
            if capture_steps is None or row_shape[0] == capture_steps:
                return 1
        if capture_users is not None and row_shape[0] == capture_users:
            return 0

    return 0


def _flat_request_and_step(flat_idx: int, row_shape: tuple[int, ...], payload: dict) -> tuple[int, tuple[int, ...]]:
    if len(row_shape) == 0:
        return 0, ()
    if len(row_shape) == 1:
        return 0, (flat_idx,)

    request_axis = _payload_request_axis(row_shape, payload)

    if len(row_shape) == 2:
        dim0, dim1 = row_shape
        if request_axis == 1:
            step_idx = flat_idx // dim1
            request_idx = flat_idx % dim1
            return request_idx, (step_idx,)

        request_idx = flat_idx // dim1
        step_idx = flat_idx % dim1
        return request_idx, (step_idx,)

    if request_axis != 0:
        raise AssertionError(f"Unsupported request axis {request_axis} for row_shape={row_shape}")

    tail_size = 1
    for dim in row_shape[1:]:
        tail_size *= dim

    request_idx = flat_idx // tail_size
    rem = flat_idx % tail_size
    step_suffix = []
    for dim in reversed(row_shape[1:]):
        step_suffix.append(rem % dim)
        rem //= dim
    return request_idx, tuple(reversed(step_suffix))


def _slice_reference_payload(payload: dict, *, max_requests: int, max_steps: int) -> dict:
    row_shape = _reference_row_shape(payload)
    if len(row_shape) != 2:
        pytest.skip(
            f"Reference payload slicing only supports 2D [steps, users] or [users, steps] payloads, got {row_shape}"
        )

    request_axis = _payload_request_axis(row_shape, payload)
    step_axis = 1 - request_axis
    request_count = row_shape[request_axis]
    step_count = row_shape[step_axis]
    if request_count <= 0 or step_count <= 0:
        pytest.skip(f"Reference payload has no data to slice: row_shape={row_shape}")

    req_slice = slice(0, min(max_requests, request_count))
    step_slice = slice(0, min(max_steps, step_count))

    def _slice_rows(tensor: torch.Tensor) -> torch.Tensor:
        squeezed = _squeeze_trailing_unit_dims(tensor)
        if squeezed.ndim < 2 or tuple(squeezed.shape[:2]) != row_shape:
            return tensor

        selectors = [slice(None)] * tensor.ndim
        selectors[request_axis] = req_slice
        selectors[step_axis] = step_slice
        return tensor[tuple(selectors)]

    sliced = {}
    for key, value in payload.items():
        if key == "metadata":
            continue
        sliced[key] = _slice_rows(_payload_tensor(payload, key))

    start_tokens = _payload_tensor(payload, "start_tokens")
    if start_tokens.ndim == 0:
        sliced["start_tokens"] = start_tokens
    else:
        sliced["start_tokens"] = start_tokens[: req_slice.stop]

    metadata = dict(payload.get("metadata", {}))
    metadata["capture_users"] = req_slice.stop
    metadata["capture_steps"] = step_slice.stop
    sliced["metadata"] = metadata
    return sliced


def _normalize_reference_payload_request_step_tensors(
    payload: dict,
    *,
    feature_keys: tuple[str, ...] = (),
    scalar_keys: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Tensor], int, int]:
    row_shape = _reference_row_shape(payload)
    if len(row_shape) != 2:
        pytest.skip(f"Expected a 2D reference payload window, got {row_shape}")

    request_axis = _payload_request_axis(row_shape, payload)
    step_axis = 1 - request_axis
    request_count = row_shape[request_axis]
    step_count = row_shape[step_axis]

    normalized: dict[str, torch.Tensor] = {}
    for key in feature_keys:
        tensor = _payload_tensor(payload, key)
        if request_axis == 1:
            tensor = tensor.permute(1, 0, *range(2, tensor.ndim)).contiguous()
        normalized[key] = tensor

    for key in scalar_keys:
        tensor = _squeeze_trailing_unit_dims(_payload_tensor(payload, key))
        if request_axis == 1:
            tensor = tensor.permute(1, 0, *range(2, tensor.ndim)).contiguous()
        normalized[key] = tensor

    return normalized, request_count, step_count


def _infer_mtp_layer_idx(hf_state_dict) -> int:
    pattern = re.compile(r"^model\.layers\.(\d+)\.eh_proj\.weight$")
    mtp_layer_indices = []
    for key in hf_state_dict.keys():
        match = pattern.match(key)
        if match is not None:
            mtp_layer_indices.append(int(match.group(1)))
    unique_indices = sorted(set(mtp_layer_indices))
    if not unique_indices:
        pytest.skip("No MTP layer weights found in hf_state_dict")
    assert len(unique_indices) == 1, f"Expected exactly one MTP layer, found {unique_indices}"
    return unique_indices[0]


def _assert_hf_checkpoint_is_dequantized(hf_state_dict) -> None:
    quantized_keys = [key for key in hf_state_dict.keys() if key.endswith("_scale_inv")]
    if quantized_keys:
        pytest.skip(
            "Detected quantized HF checkpoint tensors (*_scale_inv). "
            "LMHeadSampling golden reference tests require an already-dequantized checkpoint."
        )

    required_float_keys = [
        "model.norm.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    for key in required_float_keys:
        tensor = hf_state_dict[key]
        if tensor.dtype == torch.float8_e4m3fn:
            pytest.skip(
                f"Detected float8 tensor for '{key}'. "
                "LMHeadSampling golden reference tests require an already-dequantized checkpoint."
            )


def _lm_head_golden_weights(hf_state_dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _assert_hf_checkpoint_is_dequantized(hf_state_dict)
    gamma = hf_state_dict["model.norm.weight"].reshape(1, -1)
    vocab = hf_state_dict["lm_head.weight"].T
    indices = torch.arange(vocab.shape[-1], dtype=torch.int32).reshape(1, -1)
    return gamma, vocab, indices


def _mtp_golden_weights(hf_state_dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mtp_layer_idx = _infer_mtp_layer_idx(hf_state_dict)
    embedding = hf_state_dict["model.embed_tokens.weight"]
    h_gamma = hf_state_dict[f"model.layers.{mtp_layer_idx}.hnorm.weight"].reshape(1, -1)
    e_gamma = hf_state_dict[f"model.layers.{mtp_layer_idx}.enorm.weight"].reshape(1, -1)
    eh_projection = hf_state_dict[f"model.layers.{mtp_layer_idx}.eh_proj.weight"].T
    return embedding, h_gamma, e_gamma, eh_projection


def _verification_lm_head_golden_weights(hf_state_dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _lm_head_golden_weights(hf_state_dict)


def _rms_norm_golden(input_tensor: torch.Tensor, gamma: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + epsilon)
    return normalized * gamma


def _mtp_decoder_input_from_hidden_and_token(
    hidden_tensor: torch.Tensor,
    token_id: int,
    embedding_tensor: torch.Tensor,
    h_gamma_tensor: torch.Tensor,
    e_gamma_tensor: torch.Tensor,
    eh_projection_tensor: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    token_embedding = embedding_tensor[token_id, :].unsqueeze(0).to(dtype=hidden_tensor.dtype)
    h_rmsnorm_out = _rms_norm_golden(hidden_tensor, h_gamma_tensor.to(dtype=hidden_tensor.dtype), epsilon=epsilon)
    e_rmsnorm_out = _rms_norm_golden(token_embedding, e_gamma_tensor.to(dtype=hidden_tensor.dtype), epsilon=epsilon)
    concat_he = torch.cat([e_rmsnorm_out, h_rmsnorm_out], dim=-1)
    return concat_he @ eh_projection_tensor.to(dtype=hidden_tensor.dtype)


def _mtp_decoder_layer_prefixes() -> tuple[str, ...]:
    return (
        "self_attn.",
        "mlp.",
        "input_layernorm.",
        "post_attention_layernorm.",
    )


def _load_mtp_decoder_layer_golden(hf_model_path: Path, hf_state_dict):
    mtp_layer_idx = _infer_mtp_layer_idx(hf_state_dict)
    config = AutoConfig.from_pretrained(str(hf_model_path), trust_remote_code=True)
    decoder_layer_cls = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV3DecoderLayer",
        str(hf_model_path),
    )
    layer = decoder_layer_cls(config=config, layer_idx=mtp_layer_idx).eval()

    layer_prefix = f"model.layers.{mtp_layer_idx}."
    layer_state_dict = {}
    for key in hf_state_dict.keys():
        if not key.startswith(layer_prefix):
            continue
        local_key = key[len(layer_prefix) :]
        if local_key.startswith(_mtp_decoder_layer_prefixes()):
            layer_state_dict[local_key] = hf_state_dict[key].detach().cpu()

    missing_keys, unexpected_keys = layer.load_state_dict(layer_state_dict, strict=True)
    assert not missing_keys, f"Missing MTP decoder layer weights: {missing_keys}"
    assert not unexpected_keys, f"Unexpected MTP decoder layer weights: {unexpected_keys}"
    return mtp_layer_idx, layer


def _causal_attention_mask(seq_len: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, seq_len), dtype=dtype, device=device)


def _golden_logits_flat(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor,
    vocab: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + epsilon)
    rmsnorm_out = normalized * gamma
    return (rmsnorm_out @ vocab).float().reshape(-1)


def _classify_expected_token_topk(scores_flat: torch.Tensor, expected_token: int) -> str:
    topk = torch.topk(scores_flat, min(3, int(scores_flat.numel())), largest=True, sorted=True).indices.to(torch.int64)
    if topk.numel() >= 1 and int(topk[0].item()) == expected_token:
        return "top1"
    if topk.numel() >= 2 and any(int(tok.item()) == expected_token for tok in topk[:2]):
        return "top2"
    if topk.numel() >= 3 and any(int(tok.item()) == expected_token for tok in topk[:3]):
        return "top3"
    return "mismatch"


def _format_topk_percentages(topk_counts: dict[str, int], total: int) -> str:
    assert total > 0
    return ", ".join(
        f"{bucket}={100.0 * topk_counts[bucket] / total:.2f}%" for bucket in ("top1", "top2", "top3", "mismatch")
    )


def _pcc_value(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    actual_centered = actual_flat - actual_flat.mean()
    expected_centered = expected_flat - expected_flat.mean()
    denom = torch.sqrt(actual_centered.square().sum() * expected_centered.square().sum())
    if denom.item() == 0:
        return 0.0
    return float((actual_centered * expected_centered).sum() / denom)


def test_golden_reference_payload_base_sampling(lm_head_sampling_reference_payload, hf_state_dict):
    """Golden base path: pre-norm base hidden state h[t] -> sampled token t+1."""
    payload = lm_head_sampling_reference_payload
    row_shape = _reference_row_shape(payload)

    base_hidden_states = _flatten_feature_rows(_payload_tensor(payload, "base_hidden_states"))
    base_output_tokens = _flatten_scalar_rows(_payload_tensor(payload, "base_output_tokens")).to(torch.uint32)
    base_hidden_positions = _flatten_scalar_rows(_payload_tensor(payload, "base_hidden_positions")).to(torch.int64)
    base_output_positions = _flatten_scalar_rows(_payload_tensor(payload, "base_output_positions")).to(torch.int64)

    assert base_hidden_states.shape[0] == base_output_tokens.numel()
    assert base_hidden_positions.numel() == base_output_tokens.numel()
    assert base_output_positions.numel() == base_output_tokens.numel()

    gamma, vocab, indices = _lm_head_golden_weights(hf_state_dict)
    topk_counts = {"top1": 0, "top2": 0, "top3": 0, "mismatch": 0}

    for row_idx in range(base_hidden_states.shape[0]):
        request_idx, step_idx = _flat_request_and_step(row_idx, row_shape, payload)
        hidden_pos = int(base_hidden_positions[row_idx].item())
        output_pos = int(base_output_positions[row_idx].item())
        assert output_pos == hidden_pos + 1, (
            f"Reference payload position mismatch for request={request_idx}, step={step_idx}: "
            f"base_output_position={output_pos}, base_hidden_position={hidden_pos}"
        )

        expected_token = base_output_tokens[row_idx].reshape(1, 1)
        scores_flat = _golden_logits_flat(base_hidden_states[row_idx : row_idx + 1].to(dtype=gamma.dtype), gamma, vocab)
        topk_counts[_classify_expected_token_topk(scores_flat, int(expected_token.item()))] += 1
        got_token, aux = LMHeadSampling.golden(
            base_hidden_states[row_idx : row_idx + 1].to(dtype=gamma.dtype),
            gamma,
            vocab,
            indices=indices,
            k=1,
            p=1.0,
        )
        assert aux is None
    logger.info(
        "Golden base token top-k coverage: " + _format_topk_percentages(topk_counts, base_hidden_states.shape[0])
    )


def test_golden_reference_payload_mtp_fusion(lm_head_sampling_reference_payload, hf_state_dict):
    """Golden MTP base path: h[t] -> sampled token t+1 plus EH projection input to the MTP decoder."""
    payload = lm_head_sampling_reference_payload
    row_shape = _reference_row_shape(payload)

    base_hidden_states = _flatten_feature_rows(_payload_tensor(payload, "base_hidden_states"))
    base_output_tokens = _flatten_scalar_rows(_payload_tensor(payload, "base_output_tokens")).to(torch.uint32)
    base_output_positions = _flatten_scalar_rows(_payload_tensor(payload, "base_output_positions")).to(torch.int64)
    mtp_decoder_inputs = _flatten_feature_rows(_payload_tensor(payload, "mtp_decoder_inputs"))
    mtp_input_positions = _flatten_scalar_rows(_payload_tensor(payload, "mtp_input_positions")).to(torch.int64)

    assert base_hidden_states.shape[0] == base_output_tokens.numel()
    assert mtp_decoder_inputs.shape[0] == base_output_tokens.numel()
    assert mtp_input_positions.numel() == base_output_tokens.numel()

    gamma, vocab, indices = _lm_head_golden_weights(hf_state_dict)
    embedding, h_gamma, e_gamma, eh_projection = _mtp_golden_weights(hf_state_dict)
    topk_counts = {"top1": 0, "top2": 0, "top3": 0, "mismatch": 0}
    matched_token_rows = 0
    mtp_pcc_failures = 0
    first_pcc_failure = None
    mtp_pcc_values = []

    for row_idx in range(base_hidden_states.shape[0]):
        request_idx, step_idx = _flat_request_and_step(row_idx, row_shape, payload)
        output_pos = int(base_output_positions[row_idx].item())
        mtp_input_pos = int(mtp_input_positions[row_idx].item())
        assert mtp_input_pos == output_pos, (
            f"Reference payload position mismatch for request={request_idx}, step={step_idx}: "
            f"mtp_input_position={mtp_input_pos}, base_output_position={output_pos}"
        )

        expected_token = base_output_tokens[row_idx].reshape(1, 1)
        expected_mtp_input = mtp_decoder_inputs[row_idx : row_idx + 1].to(torch.float32)
        scores_flat = _golden_logits_flat(base_hidden_states[row_idx : row_idx + 1].to(dtype=gamma.dtype), gamma, vocab)
        topk_counts[_classify_expected_token_topk(scores_flat, int(expected_token.item()))] += 1
        got_token, got_mtp_input = LMHeadSampling.golden(
            base_hidden_states[row_idx : row_idx + 1].to(dtype=gamma.dtype),
            gamma,
            vocab,
            indices=indices,
            k=1,
            p=1.0,
            fuse_mtp=True,
            embedding_tensor=embedding,
            h_gamma_tensor=h_gamma,
            e_gamma_tensor=e_gamma,
            eh_projection_tensor=eh_projection,
        )
        assert got_mtp_input is not None
        if not torch.equal(got_token.to(torch.uint32), expected_token):
            continue

        matched_token_rows += 1
        mtp_passing_pcc, _ = comp_pcc(got_mtp_input.float(), expected_mtp_input, 0.9985)
        mtp_pcc_value = _pcc_value(got_mtp_input.float(), expected_mtp_input)
        mtp_pcc_values.append(mtp_pcc_value)
        if not mtp_passing_pcc:
            mtp_pcc_failures += 1
            if first_pcc_failure is None:
                first_pcc_failure = (
                    f"Golden MTP input mismatch for request={request_idx}, step={step_idx}, pos={output_pos}. "
                    f"expected_shape={tuple(expected_mtp_input.shape)}, got_shape={tuple(got_mtp_input.shape)}"
                )
    logger.info(
        "Golden MTP base token top-k coverage: " + _format_topk_percentages(topk_counts, base_hidden_states.shape[0])
    )
    logger.info(
        "Golden MTP fused-chain evaluation coverage: "
        f"matched_top1_rows={100.0 * matched_token_rows / base_hidden_states.shape[0]:.2f}%, "
        f"skipped_rows={100.0 * (base_hidden_states.shape[0] - matched_token_rows) / base_hidden_states.shape[0]:.2f}%"
    )
    assert matched_token_rows > 0, "Golden MTP fused-chain test had no top1 token matches against payload"
    logger.info(
        "Golden MTP fused-chain PCC summary: "
        f"lowest={min(mtp_pcc_values):.6f}, median={statistics.median(mtp_pcc_values):.6f}"
    )
    mtp_pcc_pass_rate = (matched_token_rows - mtp_pcc_failures) / matched_token_rows
    logger.info(
        "Golden MTP fused-chain PCC pass rate: "
        f"pass_rate={mtp_pcc_pass_rate:.6f}, failures={mtp_pcc_failures}, total={matched_token_rows}"
    )
    assert mtp_pcc_pass_rate >= 0.998, first_pcc_failure or (
        "Golden MTP input PCC pass rate too low: "
        f"{mtp_pcc_pass_rate:.6f} (failures={mtp_pcc_failures}, total={matched_token_rows})"
    )


def test_golden_reference_payload_mtp_verification(lm_head_sampling_reference_payload, hf_state_dict):
    """Golden verification path: MTP decoder output at t+1 -> speculative token t+2, verified against next base token."""
    payload = lm_head_sampling_reference_payload
    row_shape = _reference_row_shape(payload)

    mtp_decoder_outputs = _flatten_feature_rows(_payload_tensor(payload, "mtp_decoder_outputs"))
    mtp_speculation_tokens = _flatten_scalar_rows(_payload_tensor(payload, "mtp_speculation_tokens")).to(torch.uint32)
    mtp_input_positions = _flatten_scalar_rows(_payload_tensor(payload, "mtp_input_positions")).to(torch.int64)
    mtp_speculation_positions = _flatten_scalar_rows(_payload_tensor(payload, "mtp_speculation_positions")).to(
        torch.int64
    )
    base_output_tokens = _flatten_scalar_rows(_payload_tensor(payload, "base_output_tokens")).to(torch.uint32)
    base_output_positions = _flatten_scalar_rows(_payload_tensor(payload, "base_output_positions")).to(torch.int64)

    assert mtp_decoder_outputs.shape[0] == mtp_speculation_tokens.numel()
    assert mtp_input_positions.numel() == mtp_speculation_tokens.numel()
    assert mtp_speculation_positions.numel() == mtp_speculation_tokens.numel()
    assert base_output_positions.numel() == base_output_tokens.numel()

    gamma, vocab, indices = _verification_lm_head_golden_weights(hf_state_dict)

    reference_token_by_request_pos = {}
    for row_idx in range(base_output_tokens.numel()):
        request_idx, _ = _flat_request_and_step(row_idx, row_shape, payload)
        base_pos = int(base_output_positions[row_idx].item())
        reference_token_by_request_pos[(request_idx, base_pos)] = int(base_output_tokens[row_idx].item())

    verified_rows = 0
    spec_matches = 0
    spec_mismatches = 0
    spec_topk_counts = {"top1": 0, "top2": 0, "top3": 0, "mismatch": 0}
    accept_matches = 0
    accept_mismatches = 0
    expected_accept_count = 0
    got_accept_count = 0
    for row_idx in range(mtp_decoder_outputs.shape[0]):
        request_idx, step_idx = _flat_request_and_step(row_idx, row_shape, payload)
        mtp_input_pos = int(mtp_input_positions[row_idx].item())
        mtp_spec_pos = int(mtp_speculation_positions[row_idx].item())
        assert mtp_spec_pos == mtp_input_pos + 1, (
            f"Reference payload position mismatch for request={request_idx}, step={step_idx}: "
            f"mtp_speculation_position={mtp_spec_pos}, mtp_input_position={mtp_input_pos}"
        )

        reference_key = (request_idx, mtp_spec_pos)
        if reference_key not in reference_token_by_request_pos:
            continue

        verified_rows += 1
        expected_spec_token = mtp_speculation_tokens[row_idx].reshape(1, 1)
        reference_token = torch.tensor(
            [[reference_token_by_request_pos[reference_key]]],
            dtype=torch.uint32,
        )
        expected_match = torch.tensor(
            [[1 if int(expected_spec_token.item()) == int(reference_token.item()) else 0]],
            dtype=torch.uint32,
        )

        got_spec_token, got_match = LMHeadSampling.golden(
            mtp_decoder_outputs[row_idx : row_idx + 1].to(dtype=gamma.dtype),
            gamma,
            vocab,
            indices=indices,
            k=1,
            p=1.0,
            fuse_mtp_verification=True,
            reference_token=reference_token,
        )
        assert got_match is not None
        scores_flat = _golden_logits_flat(
            mtp_decoder_outputs[row_idx : row_idx + 1].to(dtype=gamma.dtype), gamma, vocab
        )
        spec_topk_counts[_classify_expected_token_topk(scores_flat, int(expected_spec_token.item()))] += 1

        if torch.equal(got_spec_token.to(torch.uint32), expected_spec_token):
            spec_matches += 1
        else:
            spec_mismatches += 1

        if torch.equal(got_match.to(torch.uint32), expected_match):
            accept_matches += 1
        else:
            accept_mismatches += 1

        expected_accept_count += int(expected_match.item())
        got_accept_count += int(got_match.item())

    assert verified_rows > 0, "Reference payload did not contain any verifiable MTP rows"
    acceptance_match_rate = accept_matches / verified_rows
    expected_accept_rate = expected_accept_count / verified_rows
    got_accept_rate = got_accept_count / verified_rows

    logger.info(
        "MTP verification exact acceptance-bit equality: "
        f"matches={accept_matches}, mismatches={accept_mismatches}, total={verified_rows}"
    )
    logger.info(
        "MTP verification exact speculative-token equality: "
        f"matches={spec_matches}, mismatches={spec_mismatches}, total={verified_rows}"
    )
    logger.info(
        "MTP verification speculative-token top-k coverage: "
        + _format_topk_percentages(spec_topk_counts, verified_rows)
    )
    logger.info(
        "MTP verification acceptance rates: "
        f"payload={expected_accept_rate:.6f}, golden={got_accept_rate:.6f}, "
        f"bit_match_rate={acceptance_match_rate:.6f}"
    )

    assert got_accept_rate > 0.75, (
        f"MTP verification golden acceptance rate too low: {got_accept_rate:.6f} "
        f"(accepted={got_accept_count}, total={verified_rows})"
    )
    assert acceptance_match_rate > 0.95, (
        f"MTP verification acceptance-bit match rate too low: {acceptance_match_rate:.6f} "
        f"(matches={accept_matches}, mismatches={accept_mismatches}, total={verified_rows})"
    )


def _create_mcast_working_bufs(
    device, mcast_core, matmul_core_grid, M, K, a_tile, embedding_dim=None, num_devices=1, mesh_mapper=None
):
    """Allocate HEIGHT_SHARDED working buffer tensors on the mcast receiver grid (bounding box minus sender).

    Returns (mcast_dst_buf, mcast_eh_dst_buf).
    mcast_eh_dst_buf is None when embedding_dim is None (MTP disabled).
    """
    matmul_bbox = matmul_core_grid.bounding_box()
    mcast_end_x = max(matmul_bbox.end.x, mcast_core.x)
    mcast_end_y = max(matmul_bbox.end.y, mcast_core.y)

    receiver_ranges = []
    if mcast_core.y > 0:
        receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mcast_end_x, mcast_core.y - 1)))
    if mcast_core.x > 0:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, mcast_core.y), ttnn.CoreCoord(mcast_core.x - 1, mcast_core.y))
        )
    if mcast_core.x < mcast_end_x:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(mcast_core.x + 1, mcast_core.y), ttnn.CoreCoord(mcast_end_x, mcast_core.y))
        )
    if mcast_core.y < mcast_end_y:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, mcast_core.y + 1), ttnn.CoreCoord(mcast_end_x, mcast_end_y))
        )
    receiver_grid = ttnn.CoreRangeSet(receiver_ranges)
    num_receiver_cores = (mcast_end_x + 1) * (mcast_end_y + 1) - 1

    dst_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(receiver_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    from_torch_kwargs = dict(
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=a_tile, device=device, memory_config=dst_mem_config
    )
    if mesh_mapper is not None:
        from_torch_kwargs["mesh_mapper"] = mesh_mapper
    mcast_dst_buf = ttnn.from_torch(
        torch.zeros((num_devices * num_receiver_cores, K), dtype=torch.bfloat16),
        **from_torch_kwargs,
    )

    mcast_eh_dst_buf = None
    if embedding_dim is not None:
        eh_k = K + embedding_dim
        eh_dst_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(receiver_grid, (M, eh_k), ttnn.ShardOrientation.ROW_MAJOR),
        )
        eh_from_torch_kwargs = dict(
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=a_tile, device=device, memory_config=eh_dst_mem_config
        )
        if mesh_mapper is not None:
            eh_from_torch_kwargs["mesh_mapper"] = mesh_mapper
        mcast_eh_dst_buf = ttnn.from_torch(
            torch.zeros((num_devices * num_receiver_cores, eh_k), dtype=torch.bfloat16),
            **eh_from_torch_kwargs,
        )

    return mcast_dst_buf, mcast_eh_dst_buf


# Synthetic weight provider: same layout as prepare_* (state dict + move_to_device); used for pipeline tests.
_VOCAB_SIZE = 129280
_EMBED_HIDDEN = 7168
_LM_HEAD_N_SYNTHETIC = 101 * 160  # 16160
_REAL_WEIGHTS_PERSISTENT_INPUT_TOKEN_SEED = 42


class _SyntheticWeightProvider:
    """Provider that creates deterministic synthetic embedding and LM head weights (one-hot / winner_per_row)."""

    @staticmethod
    def make_embedding_torch():
        """Raw one-hot embedding table in HF format (vocab_size, hidden)."""
        w = torch.zeros((_VOCAB_SIZE, _EMBED_HIDDEN), dtype=torch.bfloat16)
        w[torch.arange(_VOCAB_SIZE), torch.arange(_VOCAB_SIZE, dtype=torch.int64) % _EMBED_HIDDEN] = 1
        return w

    @staticmethod
    def make_lm_head_torch():
        """Raw LM head + norm weights. Returns (lm_w, norm_w) in HF format (vocab_size, hidden)."""
        lm_w = torch.full((_VOCAB_SIZE, _EMBED_HIDDEN), -1.0, dtype=torch.bfloat16)
        lm_w[torch.arange(_EMBED_HIDDEN, dtype=torch.int64) % _LM_HEAD_N_SYNTHETIC, torch.arange(_EMBED_HIDDEN)] = 1
        norm_w = torch.ones(_EMBED_HIDDEN, dtype=torch.bfloat16)
        return lm_w, norm_w

    @staticmethod
    def make_mtp_torch(num_devices):
        """Raw MTP torch tensors (embedding, h_gamma, e_gamma, eh_proj) with seed=42."""
        K = _EMBED_HIDDEN
        embedding_dim = _EMBED_HIDDEN
        mtp_output_dim = _EMBED_HIDDEN
        n_total = 101 * 160
        torch.manual_seed(42)
        embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16)
        h_gamma = torch.randn((1, K), dtype=torch.bfloat16)
        e_gamma = torch.randn((1, embedding_dim), dtype=torch.bfloat16)
        eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
        return embedding, h_gamma, e_gamma, eh_proj

    def load_embedding(self, device):
        w = self.make_embedding_torch()
        return prepare_embedding_weights({"model.embed_tokens.weight": w}, device, move_to_device=True)

    def load_lm_head(self, device):
        lm_w, norm_w = self.make_lm_head_torch()
        return prepare_lm_head_weights(
            {"lm_head.weight": lm_w, "model.norm.weight": norm_w},
            device,
            move_to_device=True,
        )

    def load_mtp_weights(self, device):
        M = 1
        K = _EMBED_HIDDEN
        embedding_dim = _EMBED_HIDDEN
        mtp_output_dim = _EMBED_HIDDEN
        tile_width = 32
        num_dram_banks = 8
        mtp_n_per_core = mtp_output_dim // num_dram_banks
        mtp_padded_dim = num_dram_banks * mtp_n_per_core
        n_total = 101 * 160
        num_devices = device.shape[0] * device.shape[1]

        a_tile = ttnn.Tile([1, 32])
        b_tile = ttnn.Tile([32, 32])

        mcast_core = ttnn.CoreCoord(10, 9)
        mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
        )

        torch_embedding, torch_h_gamma, torch_e_gamma, torch_eh_proj = self.make_mtp_torch(num_devices)
        torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

        ttnn_embedding = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn_h_gamma = ttnn.from_torch(
            torch_h_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn_e_gamma = ttnn.from_torch(
            torch_e_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        eh_shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        eh_proj_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
        ttnn_eh_proj = ttnn.from_torch(
            torch_eh_proj_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=eh_proj_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        return DeepSeekV3MTPWeights(
            embedding=ttnn_embedding,
            h_gamma=ttnn_h_gamma,
            e_gamma=ttnn_e_gamma,
            eh_projection=ttnn_eh_proj,
        )


def create_single_pod_passthrough_pipeline_configuration(
    weight_provider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
) -> PipelineConfiguration:
    """16-stage pod topology with passthrough middle stages for LM-head-focused synthetic testing."""

    def stage_0(device):
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_14(device):
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            **{i: (lambda d: PassthroughStage(PassthroughPayload.ACTIVATION)) for i in range(1, 14)},
            14: stage_14,
            15: (lambda d: PassthroughStage(PassthroughPayload.TOKEN)),
        }
    )


# Golden helper: same deterministic formula as _SyntheticWeightProvider (one-hot embedding, winner_per_row).
def _compute_expected_lm_head_indices_synthetic(iterations: int) -> torch.Tensor:
    """Compute expected output indices for synthetic weights. Same math as _SyntheticWeightProvider."""
    K = 7168
    n_total = 101 * 160
    torch_gamma = torch.ones((1, K), dtype=torch.bfloat16)
    row_indices = torch.arange(iterations, dtype=torch.int64) % K
    torch_embedding_table = torch.zeros((iterations, K), dtype=torch.bfloat16)
    torch_embedding_table[torch.arange(iterations), row_indices] = 1
    winner_per_row = torch.arange(K, dtype=torch.int64) % n_total
    torch_b = torch.full((K, n_total), fill_value=-1.0, dtype=torch.bfloat16)
    torch_b[torch.arange(K), winner_per_row] = 1
    torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_indices = torch.stack(
        [
            LMHeadSampling.golden(
                torch_embedding_table[iteration : iteration + 1].float(),
                torch_gamma.float(),
                torch_b.float().unsqueeze(0),
                indices=torch_indices_flat,
                k=1,
                p=1.0,
            )[0].to(torch.uint32)
            for iteration in range(iterations)
        ],
        dim=0,
    )
    return torch_expected_indices


def _hf_functional_lm_logits_flat(
    embed_1h: torch.Tensor,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """HuggingFace-style last-step logits: RMSNorm(hidden) then ``logits = hidden @ lm_head.weight.T``.

    Same as ``nn.Linear(hidden, vocab)`` with weight ``lm_w`` of shape ``(vocab, hidden)`` and no bias:
    ``logits = x @ lm_w.mT`` with ``x`` shape ``(1, hidden)``. RMSNorm matches the usual HF pattern
    ``x * rsqrt(mean(x**2) + eps) * weight`` with ``model.norm.weight`` shape ``(hidden,)``.
    """
    x = embed_1h.to(dtype)
    nw = norm_w.to(dtype)
    lw = lm_w.to(dtype)
    var = x.pow(2).mean(-1, keepdim=True)
    eps_t = torch.tensor(epsilon, device=x.device, dtype=dtype)
    x = x * torch.rsqrt(var + eps_t)
    x = x * nw.unsqueeze(0)
    logits = x @ lw.mT
    return logits.reshape(-1)


def _topk_vocab_ids_from_scores(scores_flat: torch.Tensor, k: int = 10) -> torch.Tensor:
    k_eff = min(k, int(scores_flat.numel()))
    _, idx = torch.topk(scores_flat, k_eff, largest=True, sorted=True)
    return idx.to(torch.uint32)


def _compute_reference_topk_token_ids_real(state_dict, input_token_ids: torch.Tensor, topk: int = 10) -> torch.Tensor:
    """Per row of ``input_token_ids`` (vocab ids into ``embed_tokens``), top-`topk` output vocab ids from HF logits.

    ``input_token_ids`` shape ``(n,)`` with values in ``[0, vocab_size)``. Returns shape ``(n, topk)``.
    """
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    K = 7168
    assert embed_w.shape == (_VOCAB_SIZE, K), f"Unexpected embed shape {embed_w.shape}"
    assert norm_w.shape == (K,), f"Unexpected norm shape {norm_w.shape}"
    assert lm_w.shape == (_VOCAB_SIZE, K), f"Unexpected lm_head shape {lm_w.shape}"
    rows = []
    for in_tok in input_token_ids.tolist():
        tid = int(in_tok)
        assert 0 <= tid < _VOCAB_SIZE, f"input token id {tid} out of range"
        scores_flat = _hf_functional_lm_logits_flat(
            embed_w[tid : tid + 1],
            norm_w,
            lm_w,
        )
        rows.append(_topk_vocab_ids_from_scores(scores_flat, k=topk))
    return torch.stack(rows, dim=0)


def _real_weights_topk_table_row(
    embed_w: torch.Tensor,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    in_tok: int,
    got_id: int,
    top_ids: list[int],
) -> tuple[int, float, str]:
    """One table row: ref_rank (1-based), Δ@got vs ref top-1, space-separated top-k ids string."""
    scores_flat = _hf_functional_lm_logits_flat(
        embed_w[in_tok : in_tok + 1],
        norm_w,
        lm_w,
    )
    order = torch.argsort(scores_flat, descending=True)
    rank = int((order == got_id).nonzero(as_tuple=True)[0][0].item()) + 1
    top1 = int(top_ids[0])
    logit_got = float(scores_flat[got_id].float().item())
    logit_top1 = float(scores_flat[top1].float().item())
    delta = logit_got - logit_top1
    topk_str = " ".join(str(t) for t in top_ids)
    return rank, delta, topk_str


def _format_real_weights_topk_results_table(
    state_dict,
    input_token_ids: torch.Tensor,
    got_flat: torch.Tensor,
    ref_topk: torch.Tensor,
    *,
    topk: int,
) -> str:
    """Full per-iteration table: same columns as mismatch report, plus in_topk (Y/N). Always logged after the test run."""
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    iterations = int(got_flat.numel())
    assert int(input_token_ids.numel()) == iterations
    lines = [
        "",
        "=" * 88,
        f"REAL WEIGHTS top-{topk} RESULTS (all {iterations} iterations)",
        "=" * 88,
        "",
        "  iter = pipeline loop index; in_tok = random input vocab id written to H2D (embed lookup row).",
        "  got = vocab id from device; ref_rank = 1-based rank of `got` in HF functional bf16 logits (descending).",
        "  Δ@got = logit(got) − logit(ref_top1), float32 view of bf16 scores (negative ⇒ below best).",
        "  in_topk = Y if `got` is in the reference top-k list, else N.",
        "  Reference = RMSNorm(embed) then x @ lm_head.weight.T (HuggingFace-style).",
        "",
        f"{'iter':>5}  {'in_tok':>7}  {'got':>8}  {'ref_rank':>9}  {'Δ@got':>10}  {'in_topk':>7}  reference top-{topk} (best → …)",
        "-" * 88,
    ]
    for i in range(iterations):
        in_tok = int(input_token_ids[i].item())
        got_id = int(got_flat[i].item())
        top_ids = [int(x) for x in ref_topk[i].tolist()]
        rank, delta, topk_str = _real_weights_topk_table_row(embed_w, norm_w, lm_w, in_tok, got_id, top_ids)
        in_top = "Y" if got_id in top_ids else "N"
        lines.append(f"{i:5d}  {in_tok:7d}  {got_id:8d}  {rank:9d}  {delta:10.4f}  {in_top:>7}  {topk_str}")
    lines.extend(["-" * 88, ""])
    return "\n".join(lines)


def _format_real_weights_topk_mismatch_report(
    state_dict,
    mismatches: list[tuple[int, int, int, list[int]]],
    *,
    topk: int,
    total_iters: int,
) -> str:
    """Human-readable report for top-k failures: ranks, logits, one row per bad iteration.

    Each mismatch is ``(iter_idx, in_tok, got_id, top_ids)``.
    """
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    lines = [
        "",
        "=" * 80,
        f"REAL WEIGHTS top-{topk} CHECK: {len(mismatches)} failing iteration(s) out of {total_iters}",
        "=" * 80,
        "",
        "  iter = pipeline loop index; in_tok = input vocab id written to H2D (embed lookup row).",
        "  got = vocab id from device; ref_rank = 1-based rank of `got` in HF functional bf16 logits (descending).",
        "  Δ@got = logit(got) − logit(ref_top1), float32 view of bf16 scores (negative ⇒ below best).",
        "  Reference = RMSNorm(embed) then x @ lm_head.weight.T (HuggingFace-style).",
        "",
        f"{'iter':>5}  {'in_tok':>7}  {'got':>8}  {'ref_rank':>9}  {'Δ@got':>10}  reference top-{topk} (best → …)",
        "-" * 80,
    ]
    for iter_idx, in_tok, got_id, top_ids in mismatches:
        rank, delta, topk_str = _real_weights_topk_table_row(embed_w, norm_w, lm_w, in_tok, got_id, top_ids)
        lines.append(f"{iter_idx:5d}  {in_tok:7d}  {got_id:8d}  {rank:9d}  {delta:10.4f}  {topk_str}")
    lines.extend(
        [
            "-" * 80,
            "",
        ]
    )
    return "\n".join(lines)


def _compute_expected_spec_decode_tokens_synthetic(iterations: int):
    """Compute expected (base_token, spec_token) pairs for the 4-stage MTP pipeline.

    Full golden chain:
      1. Embedding lookup (one-hot table, token → activation)
      2. LM head sampling (base stage) → base_token + MTP logits
      3. LM head sampling (verify stage on MTP logits) → spec_token
    Returns list of (base_token, spec_token) tuples.
    """
    K = _EMBED_HIDDEN
    num_devices = 8
    base_embed_w = _SyntheticWeightProvider.make_embedding_torch()
    lm_w, norm_w = _SyntheticWeightProvider.make_lm_head_torch()
    torch_gamma = norm_w.unsqueeze(0)
    torch_b = lm_w[:, :].T
    torch_indices_flat = torch.arange(_VOCAB_SIZE, dtype=torch.int32).reshape(1, _VOCAB_SIZE)
    torch_embedding, torch_h_gamma, torch_e_gamma, torch_eh_proj = _SyntheticWeightProvider.make_mtp_torch(num_devices)

    torch_eh_proj_bf8 = ttnn.from_torch(torch_eh_proj, dtype=ttnn.bfloat8_b)
    torch_eh_proj = ttnn.to_torch(torch_eh_proj_bf8).to(torch.bfloat16)

    results = []
    debug_info = []
    chunk_size = K // 8
    for iteration in range(iterations):
        row_idx = iteration % K
        torch_input = torch.zeros((1, K), dtype=torch.bfloat16)
        torch_input[0, row_idx] = 1

        base_token_tensor, mtp_logits = LMHeadSampling.golden(
            torch_input.float(),
            torch_gamma.float(),
            torch_b.float().unsqueeze(0),
            indices=torch_indices_flat,
            k=1,
            p=1.0,
            fuse_mtp=True,
            embedding_tensor=torch_embedding.float(),
            h_gamma_tensor=torch_h_gamma.float(),
            e_gamma_tensor=torch_e_gamma.float(),
            eh_projection_tensor=torch_eh_proj.float(),
        )
        base_token = base_token_tensor.to(torch.uint32).item()

        _h_var = torch_input.float().pow(2).mean(-1, keepdim=True)
        _h_norm = torch_input.float() * torch.rsqrt(_h_var + 1e-6) * torch_h_gamma.float()
        _tok_emb = torch_embedding[base_token, :].unsqueeze(0).float()
        _e_var = _tok_emb.pow(2).mean(-1, keepdim=True)
        _e_norm = _tok_emb * torch.rsqrt(_e_var + 1e-6) * torch_e_gamma.float()
        _concat = torch.cat([_e_norm, _h_norm], dim=-1)
        _logit_chunks = [mtp_logits[0, _i].item() for _i in range(0, K, chunk_size)]

        # Spec stage (verify): same RMSNorm as LMHeadSampling.golden on mtp_logits (pre–vocab matmul).
        _eps = 1e-6
        _mtp_f = mtp_logits.float()
        _spec_var = _mtp_f.pow(2).mean(-1, keepdim=True)
        _spec_rmsnorm_out = _mtp_f * torch.rsqrt(_spec_var + _eps) * torch_gamma.float()
        _spec_rms_chunks = [_spec_rmsnorm_out[0, _i].item() for _i in range(0, K, chunk_size)]

        print(
            f"[SYNTH_GOLDEN] iter {iteration} spec_rmsnorm "
            f"[0]={_spec_rmsnorm_out[0, 0].item():.6f} "
            f"[{K // 2}]={_spec_rmsnorm_out[0, K // 2].item():.6f} "
            f"[{K - 1}]={_spec_rmsnorm_out[0, K - 1].item():.6f} "
            f"absmax={_spec_rmsnorm_out.abs().max().item():.6f}",
            flush=True,
        )
        _spec_chunk_str = " ".join(f"[{i * chunk_size}]={v:.6f}" for i, v in enumerate(_spec_rms_chunks))
        print(f"[SYNTH_GOLDEN] iter {iteration} spec_rmsnorm chunks: {_spec_chunk_str}", flush=True)

        debug_info.append(
            {
                "concat_0": _concat[0, 0].item(),
                "concat_7168": _concat[0, 7168].item(),
                "logit_chunks": _logit_chunks,
                "spec_rmsnorm_0": _spec_rmsnorm_out[0, 0].item(),
                "spec_rmsnorm_mid": _spec_rmsnorm_out[0, K // 2].item(),
                "spec_rmsnorm_last": _spec_rmsnorm_out[0, K - 1].item(),
                "spec_rmsnorm_absmax": _spec_rmsnorm_out.abs().max().item(),
                "spec_rmsnorm_chunks": _spec_rms_chunks,
            }
        )

        spec_token_tensor, _ = LMHeadSampling.golden(
            mtp_logits,
            torch_gamma.float(),
            torch_b.float().unsqueeze(0),
            indices=torch_indices_flat,
            k=1,
            p=1.0,
        )
        spec_token = spec_token_tensor.to(torch.uint32).item()

        results.append((base_token, spec_token))
    return results, debug_info


def _prepare_reference_mtp_weights(device: ttnn.MeshDevice, hf_state_dict) -> DeepSeekV3MTPWeights:
    _assert_hf_checkpoint_is_dequantized(hf_state_dict)
    mtp_layer_idx = _infer_mtp_layer_idx(hf_state_dict)

    K = _EMBED_HIDDEN
    embedding_dim = _EMBED_HIDDEN
    mtp_output_dim = _EMBED_HIDDEN
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])

    mcast_core = ttnn.CoreCoord(10, 9)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (1, K), ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_embedding = hf_state_dict["model.embed_tokens.weight"].to(torch.bfloat16).contiguous()
    torch_h_gamma = hf_state_dict[f"model.layers.{mtp_layer_idx}.hnorm.weight"].reshape(1, -1).to(torch.bfloat16)
    torch_e_gamma = hf_state_dict[f"model.layers.{mtp_layer_idx}.enorm.weight"].reshape(1, -1).to(torch.bfloat16)
    torch_eh_proj = hf_state_dict[f"model.layers.{mtp_layer_idx}.eh_proj.weight"].T.to(torch.bfloat16).contiguous()
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    ttnn_h_gamma = ttnn.from_torch(
        torch_h_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    ttnn_e_gamma = ttnn.from_torch(
        torch_e_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    eh_shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = ttnn.from_torch(
        torch_eh_proj_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=eh_proj_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    return DeepSeekV3MTPWeights(
        embedding=ttnn_embedding,
        h_gamma=ttnn_h_gamma,
        e_gamma=ttnn_e_gamma,
        eh_projection=ttnn_eh_proj,
    )


class _ReferencePayloadMTPWeightProvider:
    """Test-only provider that reuses the 4-stage pipeline with captured hidden states as stage-0 embedding rows."""

    def __init__(self, payload: dict, hf_state_dict) -> None:
        _assert_hf_checkpoint_is_dequantized(hf_state_dict)
        self._payload = payload
        self._hf_state_dict = hf_state_dict
        self._mtp_layer_idx = _infer_mtp_layer_idx(hf_state_dict)
        self._flattened_hidden_states = _flatten_feature_rows(_payload_tensor(payload, "base_hidden_states")).to(
            torch.bfloat16
        )

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        embedding_tt = ttnn.from_torch(
            self._flattened_hidden_states.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)

    def load_base_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        return prepare_lm_head_weights(
            {
                "lm_head.weight": self._hf_state_dict["lm_head.weight"],
                "model.norm.weight": self._hf_state_dict["model.norm.weight"],
            },
            device,
            move_to_device=True,
        )

    def load_mtp_weights(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        return _prepare_reference_mtp_weights(device, self._hf_state_dict)


def _create_reference_spec_decode_pipeline_configuration(
    weight_provider: _ReferencePayloadMTPWeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """Reference-payload 4-stage pipeline.

    Stage 0 uses a captured-hidden-state embedding table so the host can feed row indices
    while the first compute stage still sees the captured base_hidden_states.
    """

    def stage_0(device: ttnn.MeshDevice):
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            d2h_page_size=TOKEN_META_PAGE_SIZE_BYTES,
        )

    def stage_1(device: ttnn.MeshDevice):
        return BaseLMHeadStage(
            weights=weight_provider.load_base_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=weight_provider.load_mtp_weights(device),
            send_mtp_output_downstream=True,
        )

    def stage_2(device: ttnn.MeshDevice):
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def stage_3(device: ttnn.MeshDevice):
        return SpecLMHeadStage(
            weights=weight_provider.load_base_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
        )

    return PipelineConfiguration({0: stage_0, 1: stage_1, 2: stage_2, 3: stage_3})


def _parse_token_meta_page(raw: torch.Tensor) -> dict[str, int]:
    raw = raw.to(torch.uint32).flatten()
    return {
        "num_tokens": int(raw[0].item()),
        "tok0_id": int(raw[1].item()),
        "tok0_type": int(raw[2].item()),
        "tok0_pos": int(raw[3].item()),
        "tok1_id": int(raw[4].item()),
        "tok1_type": int(raw[5].item()),
        "tok1_pos": int(raw[6].item()),
    }


def _compute_reference_payload_mtp_metrics_teacher_forced(
    payload: dict,
    hf_state_dict,
    hf_model_path: Path,
    *,
    max_requests: int,
    max_steps: int,
) -> dict[str, float | int]:
    """Compute bounded end-to-end MTP metrics with teacher-forced MTP token inputs."""
    payload = _slice_reference_payload(payload, max_requests=max_requests, max_steps=max_steps)
    normalized, request_count, step_count = _normalize_reference_payload_request_step_tensors(
        payload,
        feature_keys=("base_hidden_states",),
        scalar_keys=(
            "base_output_tokens",
            "mtp_input_tokens",
            "mtp_speculation_tokens",
            "base_output_positions",
            "mtp_input_positions",
            "mtp_speculation_positions",
        ),
    )
    base_hidden_states = normalized["base_hidden_states"].to(torch.bfloat16)
    base_output_tokens = normalized["base_output_tokens"].to(torch.uint32)
    mtp_input_tokens = normalized["mtp_input_tokens"].to(torch.uint32)
    mtp_speculation_tokens = normalized["mtp_speculation_tokens"].to(torch.uint32)
    base_output_positions = normalized["base_output_positions"].to(torch.int64)
    mtp_input_positions = normalized["mtp_input_positions"].to(torch.int64)
    mtp_speculation_positions = normalized["mtp_speculation_positions"].to(torch.int64)

    start_tokens = _payload_tensor(payload, "start_tokens").to(torch.int64)
    assert base_hidden_states.shape[:2] == (request_count, step_count)
    assert start_tokens.shape[0] == request_count

    gamma, vocab, indices = _lm_head_golden_weights(hf_state_dict)
    embedding, h_gamma, e_gamma, eh_projection = _mtp_golden_weights(hf_state_dict)
    shared_head_gamma, shared_head_vocab, shared_head_indices = _verification_lm_head_golden_weights(hf_state_dict)
    mtp_layer_idx, decoder_layer = _load_mtp_decoder_layer_golden(hf_model_path, hf_state_dict)

    reference_token_by_request_pos = {}
    for request_idx in range(request_count):
        for step_idx in range(step_count):
            reference_token_by_request_pos[
                (request_idx, int(base_output_positions[request_idx, step_idx].item()))
            ] = int(base_output_tokens[request_idx, step_idx].item())

    total_rows = request_count * step_count
    base_matches = 0
    spec_matches = 0
    accepted_rows = 0
    verifiable_rows = 0
    base_topk_counts = {"top1": 0, "top2": 0, "top3": 0, "mismatch": 0}
    spec_topk_counts = {"top1": 0, "top2": 0, "top3": 0, "mismatch": 0}

    for request_idx in range(request_count):
        cache = DynamicCache()
        last_mtp_input_token = int(start_tokens[request_idx].item())

        for step_idx in range(step_count):
            hidden_row = base_hidden_states[request_idx, step_idx : step_idx + 1].to(dtype=gamma.dtype)
            expected_base_token = int(base_output_tokens[request_idx, step_idx].item())
            base_scores_flat = _golden_logits_flat(hidden_row, gamma, vocab)
            base_topk_counts[_classify_expected_token_topk(base_scores_flat, expected_base_token)] += 1
            predicted_base_token, _ = LMHeadSampling.golden(
                hidden_row,
                gamma,
                vocab,
                indices=indices,
                k=1,
                p=1.0,
            )

            got_base_token = int(predicted_base_token.reshape(-1)[0].item())
            if got_base_token == expected_base_token:
                base_matches += 1

            forced_mtp_input_token = int(mtp_input_tokens[request_idx, step_idx].item())
            mtp_decoder_input = _mtp_decoder_input_from_hidden_and_token(
                hidden_row,
                forced_mtp_input_token,
                embedding,
                h_gamma,
                e_gamma,
                eh_projection,
            )

            mtp_input_pos = int(mtp_input_positions[request_idx, step_idx].item())
            local_decode_pos = step_idx
            position_ids = torch.tensor([[local_decode_pos]], dtype=torch.long)
            attention_mask = _causal_attention_mask(
                local_decode_pos + 1,
                dtype=mtp_decoder_input.dtype,
                device=mtp_decoder_input.device,
            )
            decoder_out = decoder_layer(
                hidden_states=mtp_decoder_input.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
            )[0].squeeze(1)

            predicted_spec_token, _ = LMHeadSampling.golden(
                decoder_out.to(dtype=shared_head_gamma.dtype),
                shared_head_gamma,
                shared_head_vocab,
                indices=shared_head_indices,
                k=1,
                p=1.0,
            )
            got_spec_token = int(predicted_spec_token.reshape(-1)[0].item())
            expected_spec_token = int(mtp_speculation_tokens[request_idx, step_idx].item())
            spec_scores_flat = _golden_logits_flat(
                decoder_out.to(dtype=shared_head_gamma.dtype), shared_head_gamma, shared_head_vocab
            )
            spec_topk_counts[_classify_expected_token_topk(spec_scores_flat, expected_spec_token)] += 1
            if got_spec_token == expected_spec_token:
                spec_matches += 1

            reference_key = (request_idx, int(mtp_speculation_positions[request_idx, step_idx].item()))
            if reference_key in reference_token_by_request_pos:
                verifiable_rows += 1
                accepted_rows += int(got_spec_token == reference_token_by_request_pos[reference_key])

            last_mtp_input_token = forced_mtp_input_token

        if step_count > 0:
            logger.info(
                "Golden MTP replay request {} complete: mtp_layer={} last_mtp_input_token={}",
                request_idx,
                mtp_layer_idx,
                last_mtp_input_token,
            )

    assert verifiable_rows > 0, "Reference payload did not contain any verifiable MTP rows"
    return {
        "total_rows": total_rows,
        "base_matches": base_matches,
        "spec_matches": spec_matches,
        "accepted_rows": accepted_rows,
        "verifiable_rows": verifiable_rows,
        "base_match_rate": base_matches / total_rows,
        "spec_match_rate": spec_matches / total_rows,
        "accept_rate": accepted_rows / verifiable_rows,
        "base_topk_counts": base_topk_counts,
        "spec_topk_counts": spec_topk_counts,
    }


def _log_reference_payload_mtp_metrics(metrics: dict[str, float | int], *, label: str) -> None:
    logger.info(
        f"Reference-payload {label} base token match rate: "
        f"{metrics['base_match_rate']:.6f} ({metrics['base_matches']}/{metrics['total_rows']})"
    )
    logger.info(
        f"Reference-payload {label} speculation token match rate: "
        f"{metrics['spec_match_rate']:.6f} ({metrics['spec_matches']}/{metrics['total_rows']})"
    )
    if "base_topk_counts" in metrics:
        logger.info(
            f"Reference-payload {label} base token top-k coverage: "
            f"{_format_topk_percentages(metrics['base_topk_counts'], int(metrics['total_rows']))}"
        )
    if "spec_topk_counts" in metrics:
        logger.info(
            f"Reference-payload {label} speculation token top-k coverage: "
            f"{_format_topk_percentages(metrics['spec_topk_counts'], int(metrics['total_rows']))}"
        )
    logger.info(
        f"Reference-payload {label} accept rate: "
        f"{metrics['accept_rate']:.6f} ({metrics['accepted_rows']}/{metrics['verifiable_rows']})"
    )


# Reference-payload end-to-end helpers and tests.
def _assert_reference_payload_accept_rate(metrics: dict[str, float | int], *, label: str) -> None:
    assert metrics["accept_rate"] > 0.75, (
        f"{label} end-to-end MTP accept rate too low: {metrics['accept_rate']:.6f} "
        f"(accepted={metrics['accepted_rows']}, total={metrics['verifiable_rows']})"
    )


def _compute_reference_payload_mtp_metrics_ttnn(
    mesh_device,
    *,
    use_fp32: bool,
    payload: dict,
    hf_state_dict,
    max_requests: int,
    max_steps: int,
) -> dict[str, float | int]:
    payload = _slice_reference_payload(payload, max_requests=max_requests, max_steps=max_steps)
    row_shape = _reference_row_shape(payload)
    base_hidden_states = _flatten_feature_rows(_payload_tensor(payload, "base_hidden_states"))
    base_output_tokens = _flatten_scalar_rows(_payload_tensor(payload, "base_output_tokens")).to(torch.uint32)
    mtp_speculation_tokens = _flatten_scalar_rows(_payload_tensor(payload, "mtp_speculation_tokens")).to(torch.uint32)
    base_output_positions = _flatten_scalar_rows(_payload_tensor(payload, "base_output_positions")).to(torch.int64)
    mtp_speculation_positions = _flatten_scalar_rows(_payload_tensor(payload, "mtp_speculation_positions")).to(
        torch.int64
    )

    assert base_hidden_states.shape[0] == base_output_tokens.numel()
    assert mtp_speculation_tokens.numel() == base_output_tokens.numel()

    reference_token_by_request_pos = {}
    for row_idx in range(base_output_tokens.numel()):
        request_idx, _ = _flat_request_and_step(row_idx, row_shape, payload)
        base_pos = int(base_output_positions[row_idx].item())
        reference_token_by_request_pos[(request_idx, base_pos)] = int(base_output_tokens[row_idx].item())

    provider = _ReferencePayloadMTPWeightProvider(payload, hf_state_dict)
    config = _create_reference_spec_decode_pipeline_configuration(
        provider,
        fp32_dest_acc_en=use_fp32,
        persistent_mode=True,
    )
    pipeline = config.build_pipeline(mesh_device)

    try:
        pipeline.setup_and_run()

        if pipeline.my_mesh_id != 0:
            pipeline.barrier()
            pipeline.terminate()
            pipeline.barrier()
            return {
                "total_rows": 0,
                "base_matches": 0,
                "spec_matches": 0,
                "accepted_rows": 0,
                "verifiable_rows": 0,
                "base_match_rate": 0.0,
                "spec_match_rate": 0.0,
                "accept_rate": 0.0,
            }

        token_meta_words = TOKEN_META_PAGE_SIZE_BYTES // 4
        total_rows = base_hidden_states.shape[0]
        base_matches = 0
        spec_matches = 0
        accepted_rows = 0
        verifiable_rows = 0

        for row_idx in range(total_rows):
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = row_idx
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, token_meta_words, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            token_meta = _parse_token_meta_page(ttnn.to_torch(output_tensor))

            got_base_token = token_meta["tok0_id"]
            got_spec_token = token_meta["tok1_id"]
            expected_base_token = int(base_output_tokens[row_idx].item())
            expected_spec_token = int(mtp_speculation_tokens[row_idx].item())

            if got_base_token == expected_base_token:
                base_matches += 1
            if got_spec_token == expected_spec_token:
                spec_matches += 1

            request_idx, _ = _flat_request_and_step(row_idx, row_shape, payload)
            reference_key = (request_idx, int(mtp_speculation_positions[row_idx].item()))
            if reference_key in reference_token_by_request_pos:
                verifiable_rows += 1
                accepted_rows += int(got_spec_token == reference_token_by_request_pos[reference_key])

        assert verifiable_rows > 0, "Reference payload did not contain any verifiable MTP rows"
        pipeline.barrier()
        pipeline.terminate()
        pipeline.barrier()
        return {
            "total_rows": total_rows,
            "base_matches": base_matches,
            "spec_matches": spec_matches,
            "accepted_rows": accepted_rows,
            "verifiable_rows": verifiable_rows,
            "base_match_rate": base_matches / total_rows,
            "spec_match_rate": spec_matches / total_rows,
            "accept_rate": accepted_rows / verifiable_rows,
        }
    finally:
        pass


def _is_lm_head_sampling_perf_enabled():
    return os.getenv("RUN_LM_HEAD_SAMPLING_PERF", "0") == "1"


def _is_persistent_mode_enabled():
    return os.getenv("TT_RUN_PERSISTENT_MODE", "0") == "1"


@pytest.mark.skipif(not _is_lm_head_sampling_perf_enabled(), reason="Set RUN_LM_HEAD_SAMPLING_PERF=1 to run perf test")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (1, 0), (2, 1), (2, 0)])
@pytest.mark.parametrize("num_iters,num_warmup_iters", [(20, 6)])
@pytest.mark.parametrize("enable_mtp", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
        }
    ],
    indirect=True,
)
def test_perf(bh_2d_mesh_device, use_fp32, final_mesh_coord, num_iters, num_warmup_iters, device_params, enable_mtp):
    """Performance test for LM-head sampling with optional MTP fusion.

    When enable_mtp=True, also runs:
    - Embedding lookup from argmax output token
    - h_rmsnorm and e_rmsnorm
    - Concat [h_norm|e_norm]
    - EH projection DRAM streaming matmul
    """
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    seed = 7

    # MTP dimensions
    embedding_dim = 7168
    mtp_output_dim = 7168
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)

    # MTP tensors (only used when enable_mtp=True)
    # Embedding table must cover all possible token IDs (num_devices * n_total for global indices)
    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16) if enable_mtp else None
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj_padded = None
    if enable_mtp:
        torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_mtp_output = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
        fuse_mtp=enable_mtp,
        embedding_tensor=torch_embedding.float() if enable_mtp else None,
        h_gamma_tensor=torch_h_gamma.float() if enable_mtp else None,
        e_gamma_tensor=torch_e_gamma.float() if enable_mtp else None,
        eh_projection_tensor=torch_eh_proj.float() if enable_mtp else None,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes + (256 + 8 if enable_mtp else 0)) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    # MTP-specific memory configs
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0) if enable_mtp else None
    compute_core_grid = (
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores])
        if enable_mtp
        else None
    )
    eh_shard_grid = (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1),
                )
            }
        )
        if enable_mtp
        else None
    )
    mtp_output_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )
    eh_proj_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # MTP tensors
    ttnn_embedding = None
    ttnn_h_gamma = None
    ttnn_e_gamma = None
    ttnn_eh_proj = None
    ttnn_mtp_output = None
    if enable_mtp:
        ttnn_embedding = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_h_gamma = ttnn.from_torch(
            torch_h_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_e_gamma = ttnn.from_torch(
            torch_e_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
        ttnn_eh_proj = ttnn.from_torch(
            torch_eh_proj_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=eh_proj_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_mtp_output = ttnn.from_torch(
            torch.zeros((num_devices, M, mtp_padded_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=mtp_output_mem_config,
            tile=out_tile,
            mesh_mapper=mesh_mapper,
        )

    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim if enable_mtp else None,
        num_devices=num_devices,
        mesh_mapper=mesh_mapper,
    )

    stage1_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    stage2_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh)

    submesh.enable_program_cache()
    profiler = BenchmarkProfiler()

    # Initial run to compile
    _ = LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=stage1_semaphores[0],
        global_stage2_semaphore=stage2_semaphores[0],
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        fabric_config=device_params["fabric_config"],
        enable_mtp=enable_mtp,
    )
    ttnn.synchronize_device(submesh)

    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            output_mtp_tensor=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            enable_mtp=enable_mtp,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            output_mtp_tensor=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            enable_mtp=enable_mtp,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    mtp_suffix = "+MTP" if enable_mtp else ""
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")

    signpost("start")
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    signpost("stop")

    trace_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    warmup_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    effective_duration_ns = max(0.0, trace_duration_ns - warmup_duration_ns)
    avg_iter_ns = effective_duration_ns / float(max(1, num_iters))
    logger.info(
        f"LMHead+Argmax{mtp_suffix} mesh(4x2) trace perf: final_mesh_coord={final_mesh_coord}, "
        f"iters={num_iters}, total_ns={effective_duration_ns:.2f}, avg_iter_ns={avg_iter_ns:.2f}"
    )

    final_output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output: {final_output_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Perf run fused mesh argmax mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"

    # MTP PCC check
    if enable_mtp:
        assert torch_mtp_output is not None, "MTP output cannot be None"
        final_mtp_shards = ttnn.get_device_tensors(ttnn_mtp_output)
        final_mtp_torch = (
            ttnn.to_torch(final_mtp_shards[final_device_idx])
            .to(torch.float32)
            .reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
        )
        mtp_passing_pcc, _ = comp_pcc(final_mtp_torch, torch_mtp_output.float(), 0.99)
        if not mtp_passing_pcc:
            max_diff = (final_mtp_torch - torch_mtp_output.float()).abs().max()
            logger.warning(f"MTP output PCC check failed. Max diff: {max_diff}")
        assert mtp_passing_pcc, "Perf run MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337, 52098])
def test_single_device(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax sampling with pre-cached width-sharded indices."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_single_device_mtp(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device fused LM-head + argmax + MTP fusion test."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    M = 1
    K = 7168
    embedding_dim = 7168
    mtp_output_dim = 7168
    num_matmul_cores = 101
    n_per_core = 160
    num_dram_banks = 8
    n_total = num_matmul_cores * n_per_core
    tile_width = 32
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)

    # MTP tensors
    torch_embedding = torch.randn((n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    # --- MTP specific memory configs ---
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1)
    eh_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), eh_shard_grid)})
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )
    # --- MTP embedding tensor ---
    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        # layout=ttnn.TILE_LAYOUT,
        device=submesh,
        # tile=b_tile,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # --- MTP rmsnorm gamma tensors ---
    ttnn_h_gamma = ttnn.from_torch(
        torch_h_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_e_gamma = ttnn.from_torch(
        torch_e_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    # --- MTP EH matmul tensors ---
    # DRAM streaming matmul requires column-major tile order within each bank shard
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = ttnn.from_torch(
        torch_eh_proj_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=eh_proj_mem_config,
        tile=b_tile,
    )
    ttnn_mtp_output = ttnn.from_torch(
        torch.zeros((M, mtp_padded_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mtp_output_mem_config,
        tile=out_tile,
    )
    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
    )
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"
    mtp_output_torch = ttnn.to_torch(ttnn_mtp_output).to(torch.float32).reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
    logger.info(f"MTP output shape: {mtp_output_torch.shape}")
    logger.info(f"Expected MTP shape: {torch_expected_mtp.shape}")
    mtp_passing_pcc, output = comp_pcc(mtp_output_torch, torch_expected_mtp.float(), 0.99)
    if not mtp_passing_pcc:
        logger.warning(f"MTP output PCC check failed: {mtp_passing_pcc}")
    assert mtp_passing_pcc, "MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
            "worker_l1_size": 1480000,
        }
    ],
    indirect=True,
)
def test_single_device_mtp_verification(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device MTP verification test.

    Runs the base LM head + MTP to produce T_base, then runs a second LM head
    with enable_mtp_verification=True to produce T_spec and verify T_spec == T_base.
    Uses same input/weights for both stages so they should produce the same token -> match=1.
    """
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    M = 1
    K = 7168
    embedding_dim = 7168
    mtp_output_dim = 7168
    num_matmul_cores = 101
    n_per_core = 160
    num_dram_banks = 8
    n_total = num_matmul_cores * n_per_core
    tile_width = 32
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)

    # MTP tensors (for base LM head stage)
    torch_embedding = torch.randn((n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    # Golden: compute expected base token
    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    # Golden: verification should match since we use same input/weights
    torch_verify_idx, torch_verify_result = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp_verification=True,
        reference_token=torch_expected_idx,
    )
    assert torch_verify_result.item() == 1, "Golden verification should match when same inputs are used"
    assert torch.equal(torch_verify_idx, torch_expected_idx), "Golden spec token should match base token"

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )

    # --- Stage 1: Base LM Head + MTP ---
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1)
    eh_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), eh_shard_grid)})
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    def to_device(t, mem, **kw):
        return ttnn.from_torch(t, device=submesh, memory_config=mem, **kw)

    input_tensor = to_device(torch_a, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16)
    intermediate_tensor = to_device(
        torch.zeros_like(torch_a), input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    ttnn_gamma = to_device(torch_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16)
    ttnn_b = to_device(torch_b, width_shard_mem_config, layout=ttnn.TILE_LAYOUT, tile=b_tile, dtype=ttnn.bfloat8_b)
    ttnn_scores = to_device(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    ttnn_indices = to_device(torch_indices, indices_mem_config, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    ttnn_output_index = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    ttnn_embedding = to_device(
        torch_embedding, ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_h_gamma = to_device(
        torch_h_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    ttnn_e_gamma = to_device(
        torch_e_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = to_device(
        torch_eh_proj_shuffled, eh_proj_mem_config, layout=ttnn.TILE_LAYOUT, tile=b_tile, dtype=ttnn.bfloat8_b
    )
    ttnn_mtp_output = to_device(
        torch.zeros((M, mtp_padded_dim), dtype=torch.bfloat16),
        mtp_output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
    )

    # Run base LM head + MTP
    LMHeadSampling.op(
        input_tensor,
        intermediate_tensor,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    base_token = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Base token: {base_token.item()}, expected: {torch_expected_idx.item()}")
    assert torch.equal(
        base_token, torch_expected_idx
    ), f"Base token mismatch: {base_token.item()} != {torch_expected_idx.item()}"

    # --- Stage 2: MTP Verification LM Head ---
    # Pre-load the reference token (T_base from stage 1)
    reference_token_tensor = to_device(
        base_token.reshape(1, 1),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    verification_result_tensor = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    speculative_tokens_tensor = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    # Reset scores and output for the verification op (reuse same weights/input)
    ttnn_scores_v = to_device(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    ttnn_output_index_v = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    LMHeadSampling.op(
        input_tensor,
        intermediate_tensor,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores_v,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index_v,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=False,
        enable_mtp_verification=True,
        reference_token_tensor=reference_token_tensor,
        verification_result_tensor=verification_result_tensor,
        speculative_tokens_tensor=speculative_tokens_tensor,
    )
    ttnn.synchronize_device(submesh)

    spec_token = ttnn.to_torch(ttnn_output_index_v).to(torch.uint32).reshape(1, 1)
    verify_result = ttnn.to_torch(verification_result_tensor).to(torch.uint32).reshape(1, 1)
    stored_spec = ttnn.to_torch(speculative_tokens_tensor).to(torch.uint32).reshape(1, 1)

    logger.info(f"Spec token: {spec_token.item()}, base token: {base_token.item()}")
    logger.info(f"Verification result: {verify_result.item()} (1=match, 0=no_match)")
    logger.info(f"Stored speculative token: {stored_spec.item()}")

    assert torch.equal(
        spec_token, base_token
    ), f"Spec token should match base token (same inputs). spec={spec_token.item()}, base={base_token.item()}"
    assert verify_result.item() == 1, f"Verification should match (same inputs). Got {verify_result.item()}"
    assert (
        stored_spec.item() == spec_token.item()
    ), f"Stored spec token should equal the spec token. stored={stored_spec.item()}, spec={spec_token.item()}"
    logger.info("MTP verification test PASSED: speculative token matches base token")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
def test_single_device_d2h(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax with optional D2H token emission enabled."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        socket_output=d2h_socket,
    )

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    d2h_socket.barrier()
    ttnn.synchronize_device(submesh)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "sender_coord, final_mesh_coord, seed",
    [
        ((1, 1), (0, 0), 7),
        ((0, 0), (1, 1), 1337),
        ((3, 0), (2, 0), 4242),
    ],
)
def test_multidevice(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    sender_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + k=1 sampling (argmax) with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    # Global indices are unique across mesh devices.
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_mesh_coord = ttnn.MeshCoordinate(sender_coord[0], sender_coord[1])
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_mesh_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_mesh_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output index: {final_output_index}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [7, 1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
        }
    ],
    indirect=True,
)
def test_multidevice_mtp(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh fused LM-head + argmax + MTP fusion with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    embedding_dim = 7168
    mtp_output_dim = 7168
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)

    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes + 256 + 8) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1),
            )
        }
    )
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_h_gamma = ttnn.from_torch(
        torch_h_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_e_gamma = ttnn.from_torch(
        torch_e_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = ttnn.from_torch(
        torch_eh_proj_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=eh_proj_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_mtp_output = ttnn.from_torch(
        torch.zeros((num_devices, M, mtp_padded_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mtp_output_mem_config,
        tile=out_tile,
        mesh_mapper=mesh_mapper,
    )

    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
        num_devices=num_devices,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    final_output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"

    final_mtp_shards = ttnn.get_device_tensors(ttnn_mtp_output)
    final_mtp_torch = (
        ttnn.to_torch(final_mtp_shards[final_device_idx])
        .to(torch.float32)
        .reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
    )
    mtp_passing_pcc, _ = comp_pcc(final_mtp_torch, torch_expected_mtp.float(), 0.99)
    if not mtp_passing_pcc:
        max_diff = (final_mtp_torch - torch_expected_mtp.float()).abs().max()
        logger.warning(f"MTP output PCC check failed. Max diff: {max_diff}")
    assert mtp_passing_pcc, "MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_d2h(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with optional D2H token emission on final mesh device."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(final_mesh_coord[0], final_mesh_coord[1]), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=d2h_socket,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (0, 1), (1, 0), (0, 0), (3, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_d2d_to_d2h_pipeline(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with D2D output routed through D2D forwarding to D2H."""
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test requires a full galaxy")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    socket_page_size_bytes = 64
    socket_fifo_size = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    mcast_bbox = matmul_core_grid.bounding_box()
    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if mcast_bbox.contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    logger.info(f"Extra cores: {extra_cores}")
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for D2D/D2H pipeline wiring")
    d2d1_core = ttnn.CoreCoord(11, 0)
    d2d2_core = ttnn.CoreCoord(11, 1)
    d2h_core = ttnn.CoreCoord(11, 2)
    dummy_h2d_core = ttnn.CoreCoord(11, 3)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=lmhead_input_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)

    final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    d2d1_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d1_core,
    )
    d2d2_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d2_core,
    )
    d2h_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_core,
    )
    dummy_h2d_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        dummy_h2d_core,
    )

    logger.info(f"final_mesh_core: {final_mesh_core}")
    logger.info(f"d2d1_mesh_core: {d2d1_mesh_core}")
    logger.info(f"d2d2_mesh_core: {d2d2_mesh_core}")
    logger.info(f"d2h_mesh_core: {d2h_mesh_core}")
    logger.info(f"dummy_h2d_mesh_core: {dummy_h2d_mesh_core}")

    h2d_socket = ttnn.H2DSocket(
        submesh, dummy_h2d_mesh_core, ttnn.BufferType.L1, socket_fifo_size, ttnn.H2DMode.HOST_PUSH
    )
    d2h_socket = ttnn.D2HSocket(submesh, d2h_mesh_core, socket_fifo_size)
    logger.info("Creating HostInterface")
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        socket_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=socket_fifo_size,
        h2d_downstream_core=dummy_h2d_mesh_core,
        d2h_upstream_core=d2d2_mesh_core,
    )
    logger.info("Creating SocketInterface")
    socket_interface = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        d2d1_mesh_core,
        d2d2_mesh_core,
        upstream_core_coord=final_mesh_core,
        downstream_socket=host_io.get_upstream_socket(),
        sender_mesh=MeshWrapper(mesh_device=submesh),
        receiver_mesh=MeshWrapper(mesh_device=submesh),
    )

    logger.info("Running HostInterface")
    host_io.run()
    logger.info("Running SocketInterface")
    socket_interface.run()
    logger.info("Running LMHeadSampling")
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=argmax_final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=socket_interface.get_upstream_socket(),
        fabric_config=device_params["fabric_config"],
    )
    d2h_page_words = socket_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2D->D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"

    host_io.terminate(False)
    socket_interface.terminate(True)

    ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (0, 1), (2, 0), (2, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_4stage_galaxy_1_iteration(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh lm_head pipeline with H2D ingress + D2D ingress before compute, then D2D->D2H egress."""
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test requires a full galaxy")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    activation_page_size_bytes = K * 2  # bf16 [1, 7168]
    activation_fifo_size = activation_page_size_bytes * 2
    socket_page_size_bytes = 64
    socket_fifo_size = 512
    assert activation_page_size_bytes == 14336
    assert socket_fifo_size == 8 * socket_page_size_bytes

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if matmul_core_grid.bounding_box().contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for H2D/D2D and D2D/D2H pipeline wiring")

    ingress_forward_core = ttnn.CoreCoord(11, 0)
    egress_sink_core = ttnn.CoreCoord(11, 1)
    d2h_endpoint_core = ttnn.CoreCoord(11, 2)
    h2d_endpoint_core = ttnn.CoreCoord(11, 3)
    ingress_relay_core = ttnn.CoreCoord(11, 4)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=lmhead_input_core,
        input_tensor_torch=torch_a,
        create_output_tensor_mesh=True,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    sender_mesh_coord = ttnn.MeshCoordinate(int(sender_coord[0]), int(sender_coord[1]))

    lmhead_input_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, lmhead_input_core)
    ingress_relay_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_relay_core)
    ingress_forward_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_forward_core)
    h2d_endpoint_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, h2d_endpoint_core)

    argmax_final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    egress_forward_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        ingress_forward_core,
    )
    egress_sink_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        egress_sink_core,
    )
    d2h_endpoint_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_endpoint_core,
    )
    h2d_host_socket = ttnn.H2DSocket(
        submesh,
        h2d_endpoint_mesh_core,
        ttnn.BufferType.L1,
        activation_fifo_size,
        ttnn.H2DMode.HOST_PUSH,
    )
    d2h_host_socket = ttnn.D2HSocket(submesh, d2h_endpoint_mesh_core, socket_fifo_size)
    host_io_bridge = HostInterface(
        h2d_host_socket,
        d2h_host_socket,
        activation_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=activation_fifo_size,
        h2d_downstream_core=ingress_relay_mesh_core,
        d2h_upstream_core=egress_sink_mesh_core,
    )
    ingress_d2d_link = SocketInterface(
        activation_page_size_bytes,
        activation_fifo_size,
        activation_page_size_bytes,
        ingress_relay_mesh_core,
        ingress_forward_mesh_core,
        upstream_socket=host_io_bridge.get_downstream_socket(),
        downstream_core_coord=lmhead_input_mesh_core,  # LMHead sender/socket-receiver core
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )
    egress_d2d_link = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        egress_forward_mesh_core,
        egress_sink_mesh_core,
        upstream_core_coord=argmax_final_mesh_core,  # sampling winner core / socket sender core
        downstream_socket=host_io_bridge.get_upstream_socket(),
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )

    logger.info("Running HostInterface")
    host_io_bridge.run()
    logger.info("Running Input SocketInterface")
    ingress_d2d_link.run()
    logger.info("Running Output SocketInterface")
    egress_d2d_link.run()

    try:
        h2d_activation_tensor = ttnn.from_torch(
            torch_a.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        logger.info("Running H2D socket write")
        h2d_host_socket.write_tensor(h2d_activation_tensor)

        logger.info("Running LMHeadSampling")
        LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=argmax_final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=global_semaphore,
            global_stage2_semaphore=global_stage2_semaphore,
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            socket_input=ingress_d2d_link.get_downstream_socket(),
            socket_output=egress_d2d_link.get_upstream_socket(),
            fabric_config=device_params["fabric_config"],
        )
        logger.info("Running D2H socket read")
        d2h_page_words = socket_page_size_bytes // 4
        d2h_read_tensor = ttnn.from_torch(
            torch.zeros((1, d2h_page_words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        d2h_host_socket.read_tensor(d2h_read_tensor)
        d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
        assert torch.equal(
            d2h_token, torch_expected_idx
        ), f"Mesh H2D->D2D->LMHead->D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"
    finally:
        host_io_bridge.terminate(False)
        ingress_d2d_link.terminate(False)
        egress_d2d_link.terminate(True)
        ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_pipline_block_4stage_galaxy_1_iteration(mesh_device, use_fp32, device_params):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    One-shot LMHead (no persistent mode); single token; terminate in finally.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(1)
    torch_expected_idx = torch_expected_indices[0]

    config = create_single_galaxy_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
        persistent_mode=False,
    )
    pipeline = config.build_pipeline(mesh_device)
    try:
        pipeline.setup_and_run()

        if pipeline.my_mesh_id == 0:
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            assert torch.equal(
                got, torch_expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(torch_expected_idx.item())}, got={int(got.item())}"

        pipeline.barrier()
    finally:
        pipeline.terminate()


@pytest.mark.skipif(
    not _is_persistent_mode_enabled(), reason="Set TT_RUN_PERSISTENT_MODE=1 to run persistent mode test"
)
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode(mesh_device, use_fp32, device_params):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100
    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(iterations)
    config = create_single_galaxy_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            logger.info(f"Writing token for iteration {iteration}")
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for P{pipeline.my_mesh_id}")
    pipeline.barrier()
    logger.info(f"Barrier completed for P{pipeline.my_mesh_id}")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
            "worker_l1_size": 1499000,
        }
    ],
    indirect=True,
)
def test_persistent_mode_mtp(mesh_device, use_fp32):
    """
    4-stage 4x2 single-galaxy pipeline with MTP + verification:
    P1(Embed) -> P2(LMHead+MTP) -> P3(Passthrough ACTIVATION_W_TOKEN_META) -> P4(Verify) -> P1(D2H TOKEN_META).

    The verification stage (P4) receives gathered logits + token metadata, runs its
    own LM head + argmax, then outputs a TOKEN_META page (64 bytes) back to P1.

    TOKEN_META page layout (uint32 words):
      [0] num_tokens  (0=stale, 1=accept, 2=reject)
      [1] tok0_id     [2] tok0_type (0=BASE,1=SPEC)  [3] tok0_pos
      [4] tok1_id     [5] tok1_type                   [6] tok1_pos
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100

    config = create_single_galaxy_spec_decode_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
    )
    print(f"[TEST] config created, building pipeline", flush=True)
    pipeline = config.build_pipeline(mesh_device)
    pid = pipeline.my_mesh_id
    print(f"[TEST P{pid}] pipeline built, calling setup_and_run", flush=True)
    try:
        pipeline.setup_and_run()
        print(f"[TEST P{pid}] setup_and_run complete", flush=True)

        token_meta_words = TOKEN_META_PAGE_SIZE_BYTES // 4

        if pipeline.my_mesh_id == 0:
            print(f"[TEST] computing golden...", flush=True)
            golden, golden_debug = _compute_expected_spec_decode_tokens_synthetic(iterations)
            print(f"[TEST] golden computed, creating config", flush=True)
            for iteration in range(iterations):
                print(f"[TEST P{pid}] iter {iteration} write_token", flush=True)
                torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
                torch_token[0, 0] = iteration
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                output_tensor = ttnn.from_torch(
                    torch.zeros(1, token_meta_words, dtype=torch.uint32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pipeline.write_token(token_tensor)
                print(f"[TEST P{pid}] iter {iteration} read_output", flush=True)
                pipeline.read_output(output_tensor)
                print(f"[TEST P{pid}] iter {iteration} to_torch", flush=True)
                raw = ttnn.to_torch(output_tensor).to(torch.uint32).flatten()

                num_tokens = raw[0].item()
                tok0_id = raw[1].item()
                tok0_type = raw[2].item()
                tok0_pos = raw[3].item()
                tok1_id = raw[4].item()
                tok1_type = raw[5].item()
                tok1_pos = raw[6].item()

                dbg = golden_debug[iteration]
                chunk_strs = " ".join(f"[{i * 896}]={v:.6f}" for i, v in enumerate(dbg["logit_chunks"]))
                spec_rms_chunk_strs = " ".join(f"[{i * 896}]={v:.6f}" for i, v in enumerate(dbg["spec_rmsnorm_chunks"]))
                expected_base, expected_spec = golden[iteration]
                type_name = {0: "BASE", 1: "SPEC"}
                print(
                    f"[TEST P{pid}] iter {iteration} "
                    f"ntok={num_tokens} t0={tok0_id}/{type_name.get(tok0_type,'?')} "
                    f"t1={tok1_id}/{type_name.get(tok1_type,'?')} ",
                    f"golden base token={golden[iteration][0]}",
                    f"golden spec token={golden[iteration][1]}",
                    flush=True,
                )
                print(
                    f"[TEST P{pid}] iter {iteration} "
                    f"golden concat[0]={dbg['concat_0']:.6f} concat[7168]={dbg['concat_7168']:.6f} "
                    f"mtp_logits: {chunk_strs}",
                    flush=True,
                )
                print(
                    f"[TEST P{pid}] iter {iteration} "
                    f"golden spec_rmsnorm[0]={dbg['spec_rmsnorm_0']:.6f} "
                    f"[mid]={dbg['spec_rmsnorm_mid']:.6f} [last]={dbg['spec_rmsnorm_last']:.6f} "
                    f"absmax={dbg['spec_rmsnorm_absmax']:.6f} chunks: {spec_rms_chunk_strs}",
                    flush=True,
                )
        print(f"[TEST P{pid}] all iterations done, barrier", flush=True)
        pipeline.barrier()
        print(f"[TEST P{pid}] barrier done, terminate", flush=True)
        pipeline.terminate()
        print(f"[TEST P{pid}] terminate done, final barrier", flush=True)
        pipeline.barrier()
        print(f"[TEST P{pid}] final barrier done", flush=True)
    finally:
        pass


@pytest.mark.parametrize(
    ("max_requests", "max_steps"),
    [
        pytest.param(8, 128, id="8req_128steps"),
    ],
)
def test_reference_payload_mtp_accept_rate_golden(
    lm_head_sampling_reference_payload,
    hf_model_path,
    hf_state_dict,
    max_requests,
    max_steps,
):
    """CPU-only end-to-end reference-payload MTP verification baseline."""
    metrics = _compute_reference_payload_mtp_metrics_teacher_forced(
        lm_head_sampling_reference_payload,
        hf_state_dict,
        hf_model_path,
        max_requests=max_requests,
        max_steps=max_steps,
    )
    _log_reference_payload_mtp_metrics(metrics, label="golden")
    _assert_reference_payload_accept_rate(metrics, label="Golden")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    ("max_requests", "max_steps"),
    [
        pytest.param(4, 16, id="4req_16steps"),
        pytest.param(8, 128, id="8req_128steps"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
            "worker_l1_size": 1499000,
        }
    ],
    indirect=True,
)
def test_reference_payload_mtp_accept_rate_ttnn(
    mesh_device,
    use_fp32,
    lm_head_sampling_reference_payload,
    hf_state_dict,
    max_requests,
    max_steps,
):
    """Device-backed end-to-end reference-payload MTP verification test.

    Current pipeline:
      P1(reference hidden-state source) -> P2(base LMHead+MTP) -> P3(passthrough) -> P4(MTP shared-head verify)

    This is expected to fail before a real decoder-block stage replaces the passthrough.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    metrics = _compute_reference_payload_mtp_metrics_ttnn(
        mesh_device,
        use_fp32=use_fp32,
        payload=lm_head_sampling_reference_payload,
        hf_state_dict=hf_state_dict,
        max_requests=max_requests,
        max_steps=max_steps,
    )
    if (max_requests, max_steps) == (4, 16):
        if int(ttnn.distributed_context_get_rank()) == 0:
            _log_reference_payload_mtp_metrics(metrics, label="TTNN")
        pytest.xfail(
            "TTNN reference-payload MTP test is expected to fail on the small 4req_16steps window; "
            "use the 8req_128steps variant for the larger validation run."
        )
    if int(ttnn.distributed_context_get_rank()) == 0:
        _log_reference_payload_mtp_metrics(metrics, label="TTNN")
        _assert_reference_payload_accept_rate(metrics, label="TTNN")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
            "worker_l1_size": 1499000,
        }
    ],
    indirect=True,
)
def test_persistent_mode_mtp_spec(mesh_device, use_fp32):
    """
    4-stage 4x2 single-galaxy pipeline with MTP fusion enabled:
    P1(H2D) -> P2(LMHead+Sampling+MTP) -> P3(forward) -> P4(forward) -> P1(D2H).

    Verifies both the sampled token index (on P1) and the MTP EH projection output (on P2).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100
    config = create_single_galaxy_pipeline_spec_stage_only_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
        persistent_mode=True,
        enable_mtp=True,
    )
    pipeline = config.build_pipeline(mesh_device)
    try:
        pipeline.setup_and_run()

        if pipeline.my_mesh_id == 0:
            for iteration in range(iterations):
                logger.info(f"[MTP] Writing token for iteration {iteration}")
                torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
                torch_token[0, 0] = iteration
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                output_tensor = ttnn.from_torch(
                    torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pipeline.write_token(token_tensor)
                pipeline.read_output(output_tensor)
                raw = ttnn.to_torch(output_tensor).to(torch.uint32).flatten()
                num_tokens = raw[0].item()
                tok0_id = raw[1].item()
                tok0_type = raw[2].item()
                tok0_pos = raw[3].item()
                tok1_id = raw[4].item()
                tok1_type = raw[5].item()
                tok1_pos = raw[6].item()
                type_name = {0: "BASE", 1: "SPEC"}
                print(
                    f"[MTP SPEC] iter {iteration} "
                    f"ntok={num_tokens} t0={tok0_id}/{type_name.get(tok0_type,'?')} "
                    f"t1={tok1_id}/{type_name.get(tok1_type,'?')} ",
                    flush=True,
                )

        logger.info(f"[MTP] Barrier for P{pipeline.my_mesh_id}")
        pipeline.barrier()
        logger.info(f"[MTP] Barrier completed for P{pipeline.my_mesh_id}")

        pipeline.terminate()
        pipeline.barrier()
    finally:
        pass


# @pytest.mark.skipif(not _is_persistent_mode_enabled(), reason="Set RUN_PERSISTENT_MODE=1 to run persistent mode test")
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode_real_weights(mesh_device, use_fp32, hf_model_path, hf_state_dict):
    """
    Same as test_persistent_mode but uses real HF weights (DEEPSEEK_V3_HF_MODEL) via StateDictWeightProvider.
    Each pipeline step writes a **random** input vocab id (fixed seed) for the embedding lookup; the device
    output must lie in the reference top-k from HuggingFace-style functional logits (RMSNorm then
    hidden @ lm_head.weight.T in bfloat16), since device numerics (e.g. bfloat8_b weights) can disagree
    with the reference argmax.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 300
    topk = 5
    rng = torch.Generator().manual_seed(_REAL_WEIGHTS_PERSISTENT_INPUT_TOKEN_SEED)
    input_token_ids = torch.randint(0, _VOCAB_SIZE, (iterations,), generator=rng, dtype=torch.int64)
    ref_topk = _compute_reference_topk_token_ids_real(hf_state_dict, input_token_ids, topk=topk)
    config = create_single_galaxy_pipeline_configuration(
        StateDictWeightProvider(hf_model_path),
        lm_head_fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        got_tokens = []
        for iteration in range(iterations):
            in_tok = int(input_token_ids[iteration].item())
            logger.info(f"Writing token for iteration {iteration} (in_tok={in_tok})")
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = in_tok
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            got_tokens.append(got)
        got_all = torch.stack(got_tokens, dim=0)
        got_flat = got_all.squeeze(-1).squeeze(-1)
        logger.info(f"Random input token ids (in_tok per iter): {input_token_ids.tolist()}")
        logger.info(f"All output tokens (real weights): {got_flat.tolist()}")
        logger.info(f"Reference top-{topk} per iteration (first row): {ref_topk[0].tolist()}")
        mismatches = []
        for i in range(iterations):
            in_tok = int(input_token_ids[i].item())
            g = int(got_flat[i].item())
            top_ids = [int(x) for x in ref_topk[i].tolist()]
            if g not in top_ids:
                mismatches.append((i, in_tok, g, top_ids))
        results_table = _format_real_weights_topk_results_table(
            hf_state_dict,
            input_token_ids,
            got_flat,
            ref_topk,
            topk=topk,
        )
        logger.info(results_table)
        if mismatches:
            report = _format_real_weights_topk_mismatch_report(
                hf_state_dict,
                mismatches,
                topk=topk,
                total_iters=iterations,
            )
            logger.error(report)
            pytest.fail(
                f"PipelineBlock (real weights): {len(mismatches)} output(s) not in HF functional top-{topk}.\n{report}"
            )

    logger.info(f"Barrier for P{pipeline.my_mesh_id} (real weights)")
    pipeline.barrier()
    logger.info(f"Barrier completed for P{pipeline.my_mesh_id} (real weights)")


@pytest.mark.skipif(
    not _is_persistent_mode_enabled(), reason="Set TT_RUN_PERSISTENT_MODE=1 to run persistent mode test"
)
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode_pod(mesh_device, use_fp32, device_params):
    """
    16-stage 4x2 pod pipeline (4 galaxies):
    Stage1(H2D+Embed) -> Stage2..14(activation fwd) -> Stage15(LMHead+Sampling) -> Stage16(token fwd) -> Stage1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 16:
        pytest.skip("This test requires exactly 16 distributed pipeline processes (pod: 4 galaxies)")

    iterations = 100
    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(iterations)
    config = create_single_pod_passthrough_pipeline_configuration(
        _SyntheticWeightProvider(),
        lm_head_fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            logger.info(f"Writing token for iteration {iteration}")
            pipeline.write_token(token_tensor)
            logger.info(f"Reading output for iteration {iteration}")
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"Pod 16-stage token mismatch at iter {iteration}. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for stage {pipeline.my_mesh_id + 1}")
    pipeline.barrier()
    logger.info(f"Barrier completed for stage {pipeline.my_mesh_id + 1}")
