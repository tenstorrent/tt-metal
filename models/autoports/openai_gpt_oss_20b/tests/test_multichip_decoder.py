# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tracy import signpost

import models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_fused_decoder import _functional_helpers_for_layer
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import _state_tensor
from models.autoports.openai_gpt_oss_20b.tt.multichip_decoder import (
    DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE,
    DECODE_COLLECTIVE_RS_AG_PAD64,
    EP_DEGREE,
    EXPERT_STRATEGY_EP,
    EXPERT_STRATEGY_TP,
    TARGET_MESH_SHAPE,
    TP_DEGREE,
    MultichipConfig,
    MultichipDecoder,
    _validate_qkv_geometry,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.ccl import CCLManager

SLIDING_LAYER = 12
FULL_LAYER = 13
ARTIFACT_DIR = Path("models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs")
OPTIMIZED_ARTIFACT_DIR = Path("models/autoports/openai_gpt_oss_20b/doc/optimized_multichip_decoder/logs")


def _replicated(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _host_replicated(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _device_torch(tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]


def _replicated_host(tensor) -> torch.Tensor:
    shards = _device_torch(tensor)
    assert len(shards) == TP_DEGREE
    for rank, shard in enumerate(shards[1:], start=1):
        assert torch.equal(shards[0], shard), f"replicated tensor differs on rank {rank}"
    return shards[0]


def _head_sharded_host(tensor) -> torch.Tensor:
    return torch.cat(_device_torch(tensor), dim=1)


def _local_cache_host(cache, physical_block: int, logical_length: int) -> torch.Tensor:
    rank_caches = _device_torch(cache)
    logical = [rank_cache[physical_block, :, :logical_length, :].unsqueeze(0) for rank_cache in rank_caches]
    return torch.cat(logical, dim=1)


def _logical_cache_host(
    cache,
    physical_block_ids: list[int],
    logical_length: int,
) -> torch.Tensor:
    rank_caches = _device_torch(cache)
    rank_logical = []
    for rank_cache in rank_caches:
        pages = [rank_cache[physical_block] for physical_block in physical_block_ids]
        rank_logical.append(torch.cat(pages, dim=1)[:, :logical_length, :].unsqueeze(0))
    return torch.cat(rank_logical, dim=1)


def _assert_pcc(reference: torch.Tensor, actual: torch.Tensor, threshold: float, label: str):
    passed, message = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{label}: {message}")
    assert passed, f"{label}: {message}"


def _decoder(
    state,
    config,
    mesh_device,
    *,
    layer_idx: int,
    max_cache_len: int,
    multichip_config: MultichipConfig | None = None,
    optimization_config: OptimizationConfig | None = None,
):
    return MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        multichip_config=multichip_config,
        optimization_config=optimization_config,
    )


def _reference_path(layer_idx: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_synthetic_layer{layer_idx}.pt"


def _real_reference_path(layer_idx: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_real_layer{layer_idx}.pt"


def _boundary_reference_path(layer_idx: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_real_boundary_layer{layer_idx}.pt"


def _precision_reference_path(layer_idx: int, seq_len: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_real_precision_layer{layer_idx}_seq{seq_len}.pt"


def _context_reference_path(layer_idx: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_context_endpoint_layer{layer_idx}.pt"


def _context_manual_reference_path(layer_idx: int) -> Path:
    return ARTIFACT_DIR / f"current_optimized_manual_context_endpoint_layer{layer_idx}.pt"


def _perf_reference_path(layer_idx: int, seq_len: int = 128) -> Path:
    return ARTIFACT_DIR / f"current_optimized_perf_layer{layer_idx}_seq{seq_len}.json"


def _perf_result_path(layer_idx: int, seq_len: int = 128) -> Path:
    override = os.environ.get("MULTICHIP_PERF_RESULT_PATH")
    if override:
        return Path(override.format(layer_idx=layer_idx, seq_len=seq_len))
    return ARTIFACT_DIR / f"multichip_perf_layer{layer_idx}_seq{seq_len}.json"


def _prefill_routing(decoder, attention_output, seq_len: int):
    sparse_l1_chain = decoder.experts is not None and decoder.optimization_config.expert_input_l1 and seq_len == 1
    norm_memory_config = decoder.advisor_norm_memory_config if sparse_l1_chain else ttnn.DRAM_MEMORY_CONFIG
    attention_output = ttnn.to_memory_config(attention_output, norm_memory_config)
    normalized = ttnn.rms_norm(
        attention_output,
        epsilon=decoder.rms_norm_eps,
        weight=(
            decoder.advisor_norm_weights["post_attention_norm"] if sparse_l1_chain else decoder.post_attention_norm
        ),
        memory_config=norm_memory_config,
        program_config=decoder.advisor_norm_program_config if sparse_l1_chain else None,
        compute_kernel_config=decoder.compute_kernel_config,
    )
    return decoder._route(normalized, seq_len)


def _routing_top4(routing: torch.Tensor) -> torch.Tensor:
    top4 = torch.topk(routing.float(), 4, dim=1, largest=True, sorted=False).indices
    return torch.sort(top4, dim=1).values


def _assert_top4_agreement(
    reference: torch.Tensor,
    actual: torch.Tensor,
    label: str,
    threshold: float = 0.95,
):
    reference_top4 = _routing_top4(reference)
    actual_top4 = _routing_top4(actual)
    exact_token_matches = torch.all(reference_top4 == actual_top4, dim=1)
    agreement = exact_token_matches.float().mean().item()
    print(f"{label} top-4 selected-expert agreement: {agreement:.8f}")
    assert agreement >= threshold, f"{label}: top-4 agreement {agreement:.8f} < {threshold}"


def _precision_optimization_variant() -> OptimizationConfig | None:
    variant = os.environ.get("MULTICHIP_PRECISION_VARIANT", "selected")
    if variant == "selected":
        return None
    if variant == "legacy":
        return OptimizationConfig().with_changes(
            use_manual_prefill_attention=False,
            prefill_expert_math_fidelity="lofi",
            full_prefill_expert_math_fidelity="lofi",
        )
    if variant == "manual_only":
        return OptimizationConfig().with_changes(
            use_manual_prefill_attention=True,
            prefill_expert_math_fidelity="lofi",
            full_prefill_expert_math_fidelity="lofi",
        )
    raise ValueError(f"unknown MULTICHIP_PRECISION_VARIANT={variant!r}")


def _multichip_candidate_config_from_env() -> MultichipConfig:
    """Expose off-by-default topology candidates to correctness/perf tests."""

    defaults = MultichipConfig()

    def optional_pair(name: str):
        value = os.environ.get(name)
        return tuple(int(part) for part in value.split("x")) if value else None

    def optional_int(name: str):
        value = os.environ.get(name)
        return int(value) if value else None

    return MultichipConfig(
        decode_collective=os.environ.get(
            "MULTICHIP_DECODE_COLLECTIVE_AB",
            defaults.decode_collective,
        ),
        expert_strategy=os.environ.get(
            "MULTICHIP_EXPERT_STRATEGY",
            EXPERT_STRATEGY_EP,
        ),
        use_fused_o_projection_rs=os.environ.get("MULTICHIP_FUSED_O_RS", "0") == "1",
        use_fused_o_projection_ag=os.environ.get("MULTICHIP_FUSED_O_AG", "0") == "1",
        fused_o_ag_pad_hidden=os.environ.get("MULTICHIP_FUSED_O_AG_PAD_HIDDEN", "0") == "1",
        fused_ag_matmul_payload_dtype=os.environ.get(
            "MULTICHIP_FUSED_AG_PAYLOAD_DTYPE",
            "bfloat16",
        ),
        qkv_input_cores=int(os.environ.get("MULTICHIP_QKV_INPUT_CORES", str(defaults.qkv_input_cores))),
        qkv_in0_block_w=int(os.environ.get("MULTICHIP_QKV_IN0_BLOCK_W", str(defaults.qkv_in0_block_w))),
        qkv_output_tiles_per_core=int(
            os.environ.get("MULTICHIP_QKV_OUTPUT_TILES_PER_CORE", str(defaults.qkv_output_tiles_per_core))
        ),
        qkv_out_subblock_w=int(os.environ.get("MULTICHIP_QKV_OUT_SUBBLOCK_W", str(defaults.qkv_out_subblock_w))),
        attention_weight_dtype=os.environ.get(
            "MULTICHIP_ATTENTION_WEIGHT_DTYPE",
            defaults.attention_weight_dtype,
        ),
        decode_attention_weight_dtype=os.environ.get(
            "MULTICHIP_DECODE_ATTENTION_WEIGHT_DTYPE",
            (
                os.environ["MULTICHIP_ATTENTION_WEIGHT_DTYPE"]
                if "MULTICHIP_ATTENTION_WEIGHT_DTYPE" in os.environ
                else defaults.decode_attention_weight_dtype
            ),
        ),
        attention_math_fidelity=os.environ.get(
            "MULTICHIP_ATTENTION_MATH_FIDELITY",
            defaults.attention_math_fidelity,
        ),
        long_decode_attention_math_fidelity=os.environ.get(
            "MULTICHIP_LONG_DECODE_ATTENTION_MATH_FIDELITY",
            defaults.long_decode_attention_math_fidelity,
        ),
        expert_weight_dtype=os.environ.get("MULTICHIP_EXPERT_WEIGHT_DTYPE", "bfloat8_b"),
        expert_gate_up_weight_dtype=os.environ.get("MULTICHIP_EXPERT_GATE_UP_WEIGHT_DTYPE", "selected"),
        expert_down_weight_dtype=os.environ.get("MULTICHIP_EXPERT_DOWN_WEIGHT_DTYPE", "selected"),
        expert_activation_dtype=os.environ.get("MULTICHIP_EXPERT_ACTIVATION_DTYPE", "bfloat16"),
        expert_math_fidelity=os.environ.get("MULTICHIP_EXPERT_MATH_FIDELITY", defaults.expert_math_fidelity),
        decode_ccl_dtype=os.environ.get("MULTICHIP_DECODE_CCL_DTYPE", defaults.decode_ccl_dtype),
        prefill_expert_cores=optional_pair("MULTICHIP_PREFILL_EXPERT_CORES") or defaults.prefill_expert_cores,
        expert_in0_block_w=(optional_int("MULTICHIP_PREFILL_EXPERT_IN0_BLOCK_W") or defaults.expert_in0_block_w),
        prefill_expert_subblock_w=(
            optional_int("MULTICHIP_PREFILL_EXPERT_SUBBLOCK_W") or defaults.prefill_expert_subblock_w
        ),
        decode_gate_up_cores=optional_pair("MULTICHIP_DECODE_GATE_UP_CORES") or defaults.decode_gate_up_cores,
        decode_down_cores=optional_pair("MULTICHIP_DECODE_DOWN_CORES") or defaults.decode_down_cores,
        decode_gate_up_in0_block_w=(
            optional_int("MULTICHIP_DECODE_GATE_UP_IN0_BLOCK_W") or defaults.decode_gate_up_in0_block_w
        ),
        decode_down_in0_block_w=(optional_int("MULTICHIP_DECODE_DOWN_IN0_BLOCK_W") or defaults.decode_down_in0_block_w),
        decode_gate_up_subblock_w=(
            optional_int("MULTICHIP_DECODE_GATE_UP_SUBBLOCK_W") or defaults.decode_gate_up_subblock_w
        ),
        decode_down_subblock_w=(optional_int("MULTICHIP_DECODE_DOWN_SUBBLOCK_W") or defaults.decode_down_subblock_w),
        prefill_expert_output_l1=os.environ.get(
            "MULTICHIP_PREFILL_EXPERT_OUTPUT_L1",
            "1" if defaults.prefill_expert_output_l1 else "0",
        )
        == "1",
        prefill_expert_output_l1_max_seq=int(
            os.environ.get(
                "MULTICHIP_PREFILL_EXPERT_OUTPUT_L1_MAX_SEQ",
                str(defaults.prefill_expert_output_l1_max_seq),
            )
        ),
        decode_expert_output_l1=os.environ.get(
            "MULTICHIP_DECODE_EXPERT_OUTPUT_L1",
            "1" if defaults.decode_expert_output_l1 else "0",
        )
        == "1",
        active_prefill_chunk_size=int(
            os.environ.get("MULTICHIP_ACTIVE_PREFILL_CHUNK_SIZE", str(defaults.active_prefill_chunk_size))
        ),
        use_packed_sparse_gate_up=os.environ.get("MULTICHIP_PACKED_SPARSE_GATE_UP", "0") == "1",
        use_dram_sharded_decode_attention=os.environ.get("MULTICHIP_DRAM_SHARDED_DECODE_ATTENTION", "0") == "1",
        dram_attention_core_limit=int(os.environ.get("MULTICHIP_DRAM_ATTENTION_CORE_LIMIT", "90")),
    )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_current_single_chip_optimized_reference(mesh_device, device_params, layer_idx):
    """Capture the authoritative TTNN comparison in an isolated 1x1 run."""

    del device_params
    seq_len = 17
    cache_len = 128
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._synthetic_state(config)
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=cache_len,
        )
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(9000 + layer_idx)
        hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        token = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        prefill = decoder.prefill_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
        )
        decode_input = functional_test._tt_tensor(token, mesh_device)
        attention_output = decoder._decode_attention(
            decode_input,
            key_cache,
            value_cache,
            seq_len,
        )
        norm_input = ttnn.to_memory_config(attention_output, decoder.advisor_norm_memory_config)
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=decoder.rms_norm_eps,
            weight=decoder.advisor_norm_weights["post_attention_norm"],
            memory_config=decoder.advisor_norm_memory_config,
            program_config=decoder.advisor_norm_program_config,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        routing = decoder._route(normalized, 1)
        decode = decoder._optimized_moe_forward(attention_output, 1)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer_idx": layer_idx,
                "seq_len": seq_len,
                "prefill": functional_test._to_host(prefill),
                "attention_output": functional_test._to_host(attention_output),
                "routing": functional_test._to_host(routing),
                "decode": functional_test._to_host(decode),
                "key_cache": functional_test._to_host(key_cache),
                "value_cache": functional_test._to_host(value_cache),
            },
            _reference_path(layer_idx),
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@pytest.mark.parametrize("seq_len", [128, 129, 2048])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_real_weight_current_single_chip_precision_reference(
    mesh_device,
    device_params,
    layer_idx,
    seq_len,
):
    """Capture precision-sensitive prefill and following-decode boundaries."""

    del device_params
    max_cache_len = seq_len + 1
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=max_cache_len,
        )
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(61000 + layer_idx * 10000 + seq_len)
        hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        token = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        attention_output = decoder._prefill_attention(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
            seq_len,
        )
        prefill_routing = _prefill_routing(decoder, attention_output, seq_len)
        prefill = decoder._optimized_moe_forward(attention_output, seq_len)
        prefill_key = functional_test._to_host(key_cache)[:, :, :seq_len, :]
        prefill_value = functional_test._to_host(value_cache)[:, :, :seq_len, :]

        decode_attention = decoder._decode_attention(
            functional_test._tt_tensor(token, mesh_device),
            key_cache,
            value_cache,
            seq_len,
        )
        decode_routing = _prefill_routing(decoder, decode_attention, 1)
        decode = decoder._optimized_moe_forward(decode_attention, 1)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer_idx": layer_idx,
                "seq_len": seq_len,
                "attention_output": functional_test._to_host(attention_output),
                "prefill_routing": functional_test._to_host(prefill_routing),
                "prefill": functional_test._to_host(prefill),
                "prefill_key": prefill_key,
                "prefill_value": prefill_value,
                "decode_attention": functional_test._to_host(decode_attention),
                "decode_routing": functional_test._to_host(decode_routing),
                "decode": functional_test._to_host(decode),
                "appended_key": functional_test._to_host(key_cache)[:, :, seq_len : seq_len + 1, :],
                "appended_value": functional_test._to_host(value_cache)[:, :, seq_len : seq_len + 1, :],
            },
            _precision_reference_path(layer_idx, seq_len),
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_real_weight_current_single_chip_optimized_reference(mesh_device, device_params, layer_idx):
    """Capture real-weight prefill/decode from the current optimized baseline."""

    del device_params
    seq_len = 17
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=128,
        )
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(17000 + layer_idx)
        hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        token = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        prefill = decoder.prefill_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
        )
        attention_output = decoder._decode_attention(
            functional_test._tt_tensor(token, mesh_device),
            key_cache,
            value_cache,
            seq_len,
        )
        norm_input = ttnn.to_memory_config(attention_output, decoder.advisor_norm_memory_config)
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=decoder.rms_norm_eps,
            weight=decoder.advisor_norm_weights["post_attention_norm"],
            memory_config=decoder.advisor_norm_memory_config,
            program_config=decoder.advisor_norm_program_config,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        routing = decoder._route(normalized, 1)
        decode = decoder._optimized_moe_forward(attention_output, 1)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer_idx": layer_idx,
                "seq_len": seq_len,
                "prefill": functional_test._to_host(prefill),
                "attention_output": functional_test._to_host(attention_output),
                "routing": functional_test._to_host(routing),
                "decode": functional_test._to_host(decode),
                "key_cache": functional_test._to_host(key_cache),
                "value_cache": functional_test._to_host(value_cache),
            },
            _real_reference_path(layer_idx),
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_real_weight_current_single_chip_boundary_reference(mesh_device, device_params, layer_idx):
    """Capture the current optimized decoder across the sliding-window boundary."""

    del device_params
    prefill_len = 127
    decode_positions = list(range(prefill_len, prefill_len + 5))
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=256,
        )
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(41000 + layer_idx)
        hidden = torch.randn(
            (1, 1, prefill_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        tokens = [
            torch.randn(
                (1, 1, 1, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            )
            for _ in decode_positions
        ]
        prefill = decoder.prefill_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
        )
        prefill_key_cache = functional_test._to_host(key_cache)[:, :, :prefill_len, :]
        prefill_value_cache = functional_test._to_host(value_cache)[:, :, :prefill_len, :]
        attention_outputs = []
        routing_outputs = []
        decode = []
        for position, token in zip(decode_positions, tokens):
            attention = decoder._decode_attention(
                functional_test._tt_tensor(token, mesh_device),
                key_cache,
                value_cache,
                position,
            )
            norm_input = ttnn.to_memory_config(attention, decoder.advisor_norm_memory_config)
            normalized = ttnn.rms_norm(
                norm_input,
                epsilon=decoder.rms_norm_eps,
                weight=decoder.advisor_norm_weights["post_attention_norm"],
                memory_config=decoder.advisor_norm_memory_config,
                program_config=decoder.advisor_norm_program_config,
                compute_kernel_config=decoder.compute_kernel_config,
            )
            routing = decoder._route(normalized, 1)
            output = decoder._optimized_moe_forward(attention, 1)
            attention_outputs.append(functional_test._to_host(attention))
            routing_outputs.append(functional_test._to_host(routing))
            decode.append(functional_test._to_host(output))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer_idx": layer_idx,
                "prefill_len": prefill_len,
                "decode_positions": decode_positions,
                "prefill": functional_test._to_host(prefill),
                "prefill_key_cache": prefill_key_cache,
                "prefill_value_cache": prefill_value_cache,
                "attention": attention_outputs,
                "routing": routing_outputs,
                "decode": decode,
            },
            _boundary_reference_path(layer_idx),
        )


def test_multichip_runtime_contract_and_fallback_audit():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TP_DEGREE == 4
    assert EP_DEGREE == 4
    config = MultichipConfig()
    assert config.kv_cache_dtype == "bfloat8_b"
    assert config.expert_weight_dtype == "bfloat8_b"
    assert config.expert_activation_dtype == "bfloat16"
    assert config.decode_collective == DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE
    assert config.expert_strategy == EXPERT_STRATEGY_EP
    assert not config.use_fused_o_projection_ag
    assert not config.fused_o_ag_pad_hidden
    assert config.fused_ag_matmul_payload_dtype == "bfloat16"
    assert config.attention_weight_dtype == "bfloat16"
    assert config.decode_attention_weight_dtype == "bfloat8_b"
    assert config.attention_math_fidelity == "hifi2"
    assert config.decode_gate_up_cores == (9, 10)
    assert config.decode_down_cores == (9, 10)
    assert config.prefill_expert_output_l1
    assert config.prefill_expert_output_l1_max_seq == 32

    assert MultichipDecoder.prefill_forward is not OptimizedDecoder.prefill_forward
    assert MultichipDecoder.decode_forward is not OptimizedDecoder.decode_forward
    runtime_methods = (
        MultichipDecoder._all_reduce,
        MultichipDecoder._project_o_and_reduce,
        MultichipDecoder._manual_prefill_attention,
        MultichipDecoder._prefill_attention,
        MultichipDecoder._logical_cache_range,
        MultichipDecoder._logical_cache_prefix,
        MultichipDecoder._manual_paged_decode_attention,
        MultichipDecoder._decode_norm,
        MultichipDecoder._decode_attention,
        MultichipDecoder._ep_active_expert_chunk,
        MultichipDecoder._active_prefill_moe,
        MultichipDecoder._multichip_moe_forward,
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
    )
    forbidden = (
        "torch.",
        "from_torch",
        "to_torch",
        ".cpu(",
        "super().prefill_forward",
        "super().decode_forward",
        "FunctionalDecoder.prefill_forward",
        "OptimizedDecoder.prefill_forward",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"

    expert_source = inspect.getsource(MultichipDecoder._ep_active_expert_chunk)
    # The source contains the off-default packed gate/up branch plus the
    # selected split gate, up, and down calls.  Because the branches are
    # mutually exclusive and packing defaults off, the runtime still issues
    # exactly three sparse matmuls and never executes dense all-expert work.
    assert expert_source.count("ttnn.sparse_matmul(") == 4
    assert not config.use_packed_sparse_gate_up
    assert "if self.multichip_config.use_packed_sparse_gate_up:" in expert_source
    assert "nnz=None" in expert_source
    assert "ttnn.mesh_partition(" in expert_source


def test_multichip_geometry_validation(expect_error):
    assert _validate_qkv_geometry(
        MultichipConfig(),
        k_tiles=90,
        n_tiles=40,
        grid_x=11,
        grid_y=10,
    ) == (9, 20, 2)
    assert _validate_qkv_geometry(
        MultichipConfig(
            qkv_input_cores=23,
            qkv_in0_block_w=2,
            qkv_output_tiles_per_core=1,
            qkv_out_subblock_w=1,
        ),
        k_tiles=90,
        n_tiles=40,
        grid_x=11,
        grid_y=10,
    ) == (4, 40, 4)
    with expect_error(ValueError, "must divide K tiles"):
        _validate_qkv_geometry(
            MultichipConfig(qkv_input_cores=11, qkv_in0_block_w=10),
            k_tiles=90,
            n_tiles=40,
            grid_x=11,
            grid_y=10,
        )


def test_blackhole_fused_matmul_reduce_scatter_is_source_rejected():
    """Record the upstream prefill-only gate; decode is exercised separately."""

    source_path = Path("models/demos/gpt_oss/tt/attention/operations.py")
    source = source_path.read_text()
    assert "#46181" in source
    assert 'if "blackhole" in ttnn.get_arch_name():' in source
    assert "return False" in source
    artifact = {
        "candidate": "minimal_matmul_strided_reduce_scatter_async",
        "target_arch": "blackhole",
        "verdict": "source-rejected",
        "reason": (
            "#46181: M_tiles=32 races on Blackhole; reduce-scatter can read "
            "matmul output before completion and produce nondeterministic garbage"
        ),
        "scope": "upstream prefill M_tiles=32 only; not used to reject decode M_tiles=1",
        "source": str(source_path),
        "production_gate": 'if "blackhole" in ttnn.get_arch_name(): return False',
    }
    OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_ARTIFACT_DIR / "candidate_fused_mm_rs_blackhole_rejection.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FUSED_MM_RS_REJECTION {json.dumps(artifact, sort_keys=True)} artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_decode_fused_o_projection_reduce_scatter_exact_model_shape(
    mesh_device,
    device_params,
    layer_idx,
):
    """Adapt fused MM+RS to the exact M1/K1024/N2944 TP4 O projection."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")

    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        output = _state_tensor(state, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        padded_weight = F.pad(output.T, (0, 64))
        generator = torch.Generator().manual_seed(73000 + layer_idx)
        attended = torch.randn(
            (1, 1, 1, config.num_attention_heads * config.head_dim),
            generator=generator,
            dtype=torch.bfloat16,
        )

    attended_device = ttnn.from_torch(
        attended,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weight_device = ttnn.from_torch(
        padded_weight,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mm_config = ttnn.MinimalMatmulConfig(
        M_block_size=1,
        K_block_size=8,
        N_block_size=1,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
    )
    mm_out, scattered = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        attended_device,
        weight_device,
        3,
        ccl.get_rs_ping_pong_semaphore(),
        ttnn.CoreCoord(0, 2),
        compute_kernel_config=compute,
        num_links=1,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        config=mm_config,
        barrier_semaphore=ccl.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=1,
        num_workers_per_link=2,
    )
    ttnn.synchronize_device(mesh_device)

    attended_chunks = torch.chunk(attended.float(), TP_DEGREE, dim=3)
    weight_chunks = torch.chunk(padded_weight.float(), TP_DEGREE, dim=0)
    partials = [
        torch.matmul(local_input, local_weight) for local_input, local_weight in zip(attended_chunks, weight_chunks)
    ]
    reduced = torch.stack(partials).sum(dim=0)
    golden_scattered = torch.chunk(reduced, TP_DEGREE, dim=3)
    scattered_pcc = []
    for rank, (actual, expected) in enumerate(zip(_device_torch(scattered), golden_scattered)):
        _assert_pcc(expected, actual, 0.99, f"layer{layer_idx} fused MM+RS rank{rank}")
        scattered_pcc.append(
            float(torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1])
        )

    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=1,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Ring,
        multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=ccl.get_barrier_semaphore(),
    )
    ttnn.synchronize_device(mesh_device)
    gathered_host = _replicated_host(gathered)[..., : config.hidden_size]
    _assert_pcc(reduced[..., : config.hidden_size], gathered_host, 0.99, f"layer{layer_idx} fused MM+RS+AG")
    artifact = {
        "candidate": "decode_minimal_matmul_strided_reduce_scatter_async",
        "layer_idx": layer_idx,
        "mesh": list(TARGET_MESH_SHAPE),
        "logical_shapes": {
            "attended_global": list(attended.shape),
            "attended_per_rank": [1, 1, 1, 1024],
            "weight_per_rank": [1024, 2944],
            "matmul_per_rank": [1, 1, 1, 2944],
            "reduce_scatter_per_rank": [1, 1, 1, 736],
            "public_output": [1, 1, 1, 2880],
        },
        "dtype": "bfloat8_b",
        "math_fidelity": "hifi2",
        "program": {
            "M_block_size": 1,
            "K_block_size": 8,
            "N_block_size": 1,
            "subblock_h": 1,
            "subblock_w": 1,
            "grid": [4, 2],
            "rs_core_grid_offset": [0, 2],
            "chunk_width_in_mm_blocks": 1,
            "num_workers_per_link": 2,
        },
        "scattered_pcc": scattered_pcc,
        "verdict": "correct",
    }
    OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_ARTIFACT_DIR / f"candidate_fused_mm_rs_decode_layer{layer_idx}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FUSED_MM_RS_DECODE_RESULT {json.dumps(artifact, sort_keys=True)} artifact={path}")
    mm_out.deallocate(True)
    scattered.deallocate(True)
    gathered.deallocate(True)


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_decode_collective_rs_ag_pad64_matches_selected_all_reduce(
    mesh_device,
    device_params,
    layer_idx,
):
    """Compare the real padded RS+AG decode path with the final default."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")
    seq_len = 17
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        selected = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(),
        )
        candidate = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=replace(
                MultichipConfig(),
                decode_collective=DECODE_COLLECTIVE_RS_AG_PAD64,
            ),
        )
        selected_key, selected_value = selected.create_kv_cache()
        candidate_key, candidate_value = candidate.create_kv_cache()
        selected_page_table = selected.create_page_table([1, 0])
        candidate_page_table = candidate.create_page_table([1, 0])
        generator = torch.Generator().manual_seed(71000 + layer_idx)
        prefill_hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        decode_hidden = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        selected.prefill_forward(
            _replicated(prefill_hidden, mesh_device),
            key_cache=selected_key,
            value_cache=selected_value,
            page_table=selected_page_table,
        )
        candidate.prefill_forward(
            _replicated(prefill_hidden, mesh_device),
            key_cache=candidate_key,
            value_cache=candidate_value,
            page_table=candidate_page_table,
        )
        selected_output = selected.decode_forward(
            _replicated(decode_hidden, mesh_device),
            key_cache=selected_key,
            value_cache=selected_value,
            page_table=selected_page_table,
            cache_position=seq_len,
            cache_position_tensor=selected.create_position_tensor(seq_len),
        )
        candidate_output = candidate.decode_forward(
            _replicated(decode_hidden, mesh_device),
            key_cache=candidate_key,
            value_cache=candidate_value,
            page_table=candidate_page_table,
            cache_position=seq_len,
            cache_position_tensor=candidate.create_position_tensor(seq_len),
        )
        selected_host = _replicated_host(selected_output)
        candidate_host = _replicated_host(candidate_output)
        _assert_pcc(
            selected_host,
            candidate_host,
            0.999,
            f"layer{layer_idx} padded RS+AG versus selected AR",
        )
        pcc = float(
            torch.corrcoef(
                torch.stack(
                    [
                        selected_host.float().flatten(),
                        candidate_host.float().flatten(),
                    ]
                )
            )[0, 1]
        )
        artifact = {
            "candidate": DECODE_COLLECTIVE_RS_AG_PAD64,
            "selected_config": asdict(MultichipConfig()),
            "candidate_config": asdict(
                replace(
                    MultichipConfig(),
                    decode_collective=DECODE_COLLECTIVE_RS_AG_PAD64,
                )
            ),
            "layer_idx": layer_idx,
            "mesh": list(TARGET_MESH_SHAPE),
            "payload_dtype": "bfloat16",
            "selected_input_shape": [1, 1, 1, 2880],
            "selected_input_bytes_per_rank": 5760,
            "padded_input_shape": [1, 1, 1, 2944],
            "padded_input_bytes_per_rank": 5888,
            "reduce_scatter_output_shape": [1, 1, 1, 736],
            "reduce_scatter_output_bytes_per_rank": 1472,
            "persistent_resources": "CCLManager ping-pong semaphores",
            "output_pcc": pcc,
        }
        OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = OPTIMIZED_ARTIFACT_DIR / f"candidate_rs_ag_pad64_layer{layer_idx}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"RS_AG_PAD64_RESULT {json.dumps(artifact, sort_keys=True)} artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_fused_o_projection_all_gather_selected_family_candidate(
    mesh_device,
    device_params,
    layer_idx,
):
    """A/B natural H/4=720 attended-AG/local-O against local-O/ring-AR."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")
    trace_replays = int(os.environ.get("MULTICHIP_TOPOLOGY_TRACE_REPLAYS", "100"))
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(
                decode_collective=DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE,
                use_fused_o_projection_ag=True,
                fused_o_ag_pad_hidden=False,
                fused_ag_matmul_payload_dtype="bfloat8_b",
                attention_weight_dtype="bfloat8_b",
                attention_math_fidelity="hifi2",
                long_decode_attention_math_fidelity="hifi2",
                decode_gate_up_cores=(9, 10),
                decode_gate_up_in0_block_w=45,
                decode_gate_up_subblock_w=1,
                decode_down_cores=(9, 10),
                decode_down_in0_block_w=90,
                decode_down_subblock_w=1,
            ),
        )
    generator = torch.Generator().manual_seed(8181 + layer_idx)
    attended = torch.randn(
        (1, 1, 1, config.num_attention_heads * config.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_attended = ttnn.from_torch(
        attended,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def selected_boundary():
        partial = ttnn.linear(
            tt_attended,
            decoder.output_weight,
            bias=decoder.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=decoder.tp_o_output_config,
            program_config=decoder.tp_o_program_config,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
        return decoder._all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def candidate_boundary():
        return decoder._project_o_and_reduce(tt_attended, is_decode=True)

    selected = selected_boundary()
    candidate = candidate_boundary()
    ttnn.synchronize_device(mesh_device)
    selected_host = _replicated_host(selected)
    candidate_host = _replicated_host(candidate)
    _assert_pcc(
        selected_host,
        candidate_host,
        0.999,
        f"layer{layer_idx} fused attended-AG natural720",
    )
    expected = (
        torch.matmul(
            attended.float(),
            _state_tensor(state, layer_idx, "self_attn.o_proj.weight").T.float(),
        )
        + _state_tensor(state, layer_idx, "self_attn.o_proj.bias").float()
    )
    _assert_pcc(
        expected,
        candidate_host,
        0.999,
        f"layer{layer_idx} fused attended-AG natural720 versus torch",
    )
    output_pcc = float(
        torch.corrcoef(torch.stack([selected_host.float().flatten(), candidate_host.float().flatten()]))[0, 1]
    )
    selected_ms = _time_trace_replay(
        mesh_device,
        selected_boundary,
        trace_replays,
        f"GPT_OSS_O_AR_LAYER{layer_idx}",
    )
    candidate_ms = _time_trace_replay(
        mesh_device,
        candidate_boundary,
        trace_replays,
        f"GPT_OSS_FUSED_O_AG_LAYER{layer_idx}",
    )
    artifact = {
        "candidate": "fused_attended_ag_local_o_natural720_bfloat8_b",
        "layer_idx": layer_idx,
        "mesh": list(TARGET_MESH_SHAPE),
        "attended_local_shape": [1, 1, 1, 1024],
        "attended_gather_shape": [1, 1, 1, 4096],
        "local_output_shape": [1, 1, 1, 720],
        "logical_output_shape": [1, 1, 1, 2880],
        "payload_dtype": "bfloat8_b",
        "persistent_buffer": "attended all-gather [1,1,1,4096]",
        "output_pcc": output_pcc,
        "trace_replays": trace_replays,
        "selected_local_o_ar_ms": selected_ms,
        "candidate_fused_ag_local_o_ms": candidate_ms,
        "selected_over_candidate_speedup": selected_ms / candidate_ms,
    }
    OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_ARTIFACT_DIR / f"candidate_fused_o_ag_selected_layer{layer_idx}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FUSED_O_AG_RESULT {json.dumps(artifact, sort_keys=True)} artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_tp4_gate_selected_experts_match_ep4_candidate(
    mesh_device,
    device_params,
    layer_idx,
):
    """Compare gate-selected TP4 intermediate shards with EP4 whole experts."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")
    trace_replays = int(os.environ.get("MULTICHIP_TOPOLOGY_TRACE_REPLAYS", "100"))
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        ep_decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(expert_strategy=EXPERT_STRATEGY_EP),
        )
        tp_decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(expert_strategy=EXPERT_STRATEGY_TP),
        )
    generator = torch.Generator().manual_seed(8282 + layer_idx)
    hidden = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    ep_hidden = _replicated(hidden, mesh_device)
    tp_hidden = _replicated(hidden, mesh_device)
    ep_norm = ep_decoder._decode_norm(ep_hidden, weight_name="post_attention_norm")
    ep_norm = ttnn.to_memory_config(ep_norm, ttnn.DRAM_MEMORY_CONFIG)
    tp_norm = tp_decoder._decode_norm(tp_hidden, weight_name="post_attention_norm")
    tp_norm = ttnn.to_memory_config(tp_norm, ttnn.DRAM_MEMORY_CONFIG)
    ep_routing = ep_decoder._route(ep_norm, 1)
    tp_routing = tp_decoder._route(tp_norm, 1)
    _assert_top4_agreement(
        _replicated_host(ep_routing),
        _replicated_host(tp_routing),
        f"layer{layer_idx} TP4 versus EP4 router",
        threshold=1.0,
    )

    def ep_boundary():
        norm = ep_decoder._decode_norm(ep_hidden, weight_name="post_attention_norm")
        norm = ttnn.to_memory_config(norm, ttnn.DRAM_MEMORY_CONFIG)
        routing = ep_decoder._route(norm, 1)
        partial = ep_decoder._ep_active_expert_chunk(norm, routing, is_decode=True)
        expert_output = ep_decoder._ring_all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.add(
            ep_hidden,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def tp_boundary():
        norm = tp_decoder._decode_norm(tp_hidden, weight_name="post_attention_norm")
        norm = ttnn.to_memory_config(norm, ttnn.DRAM_MEMORY_CONFIG)
        routing = tp_decoder._route(norm, 1)
        partial = tp_decoder._ep_active_expert_chunk(norm, routing, is_decode=True)
        expert_output = tp_decoder._ring_all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.add(
            tp_hidden,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ep_output = ep_boundary()
    tp_output = tp_boundary()
    ttnn.synchronize_device(mesh_device)
    ep_host = _replicated_host(ep_output)
    tp_host = _replicated_host(tp_output)
    _assert_pcc(
        ep_host,
        tp_host,
        0.99,
        f"layer{layer_idx} TP4 gate-selected experts versus EP4",
    )
    output_pcc = float(torch.corrcoef(torch.stack([ep_host.float().flatten(), tp_host.float().flatten()]))[0, 1])
    ep_ms = _time_trace_replay(
        mesh_device,
        ep_boundary,
        trace_replays,
        f"GPT_OSS_EP4_EXPERTS_LAYER{layer_idx}",
    )
    tp_ms = _time_trace_replay(
        mesh_device,
        tp_boundary,
        trace_replays,
        f"GPT_OSS_TP4_EXPERTS_LAYER{layer_idx}",
    )
    artifact = {
        "candidate": "tp4_gate_selected_experts",
        "selected": "ep4_gate_selected_experts",
        "layer_idx": layer_idx,
        "mesh": list(TARGET_MESH_SHAPE),
        "selected_ep_weight_shapes_per_rank": {
            "gate_up": [1, 8, 2880, 2880],
            "down": [1, 8, 2880, 2880],
        },
        "candidate_tp_weight_shapes_per_rank": {
            "gate_up": [1, 32, 2880, 720],
            "down": [1, 32, 720, 2880],
        },
        "gate_selected_experts_per_token": tp_decoder.experts_per_token,
        "output_pcc": output_pcc,
        "router_top4_exact": True,
        "trace_replays": trace_replays,
        "ep4_boundary_ms": ep_ms,
        "tp4_boundary_ms": tp_ms,
        "ep4_over_tp4_speedup": ep_ms / tp_ms,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"candidate_tp4_experts_layer{layer_idx}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"TP4_EXPERT_RESULT {json.dumps(artifact, sort_keys=True)} artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_fused_o_rs_deferred_through_post_attention_moe(
    mesh_device,
    device_params,
    layer_idx,
):
    """Carry fused O+RS through post-attention norm/router to the sparse boundary."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")
    hidden_size = 2880
    padded_hidden_size = 2944
    local_hidden_size = padded_hidden_size // TP_DEGREE
    trace_replays = int(os.environ.get("MULTICHIP_TOPOLOGY_TRACE_REPLAYS", "100"))

    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(
                decode_collective=DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE,
                use_fused_o_projection_rs=True,
            ),
        )

    post_attention_norm = _state_tensor(
        state,
        layer_idx,
        "post_attention_layernorm.weight",
    ).to(torch.bfloat16)
    candidate_norm_weight = ttnn.from_torch(
        F.pad(post_attention_norm, (0, 64)).reshape(1, 1, 1, padded_hidden_size),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_weight = _state_tensor(state, layer_idx, "mlp.router.weight").T.to(torch.bfloat16)
    candidate_router_weight = ttnn.from_torch(
        F.pad(router_weight, (0, 0, 0, 64)),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_bias = _state_tensor(state, layer_idx, "mlp.router.bias").float()
    candidate_router_bias = ttnn.from_torch(
        torch.stack([router_bias] + [torch.zeros_like(router_bias) for _ in range(TP_DEGREE - 1)]),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    stats_gather = {"buffer": None}
    normalized_gather = ttnn.zeros(
        [1, 1, 1, padded_hidden_size],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    residual_gather = ttnn.zeros(
        [1, 1, 1, padded_hidden_size],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    generator = torch.Generator().manual_seed(73700 + layer_idx)
    hidden = torch.randn(
        (1, 1, 1, hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    attended = torch.randn(
        (1, 1, 1, config.num_attention_heads * config.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_hidden = _replicated(hidden, mesh_device)
    baseline_attended_input = ttnn.from_torch(
        attended,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    candidate_attended_input = ttnn.from_torch(
        attended,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def baseline_boundary():
        attended_local = ttnn.to_memory_config(
            baseline_attended_input,
            ttnn.L1_MEMORY_CONFIG,
        )
        partial = ttnn.linear(
            attended_local,
            decoder.decode_output_weight,
            bias=decoder.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=decoder.tp_o_output_config,
            program_config=decoder.tp_o_program_config,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        projected = decoder._minimal_all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attention_residual = ttnn.add(
            tt_hidden,
            projected,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized = decoder._decode_norm(
            attention_residual,
            weight_name="post_attention_norm",
        )
        normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
        routing = decoder._route(normalized, 1)
        normalized_for_expert = ttnn.clone(
            normalized,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        partial_expert = decoder._ep_active_expert_chunk(
            normalized_for_expert,
            routing,
            is_decode=True,
        )
        expert_output = decoder._minimal_all_reduce(
            partial_expert,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.add(
            attention_residual,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return attention_residual, normalized, routing, output

    def candidate_boundary():
        attended_local = ttnn.clone(
            candidate_attended_input,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        projected_local = decoder._fused_o_projection_reduce_scatter(
            attended_local,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        projected_local = ttnn.to_memory_config(
            projected_local,
            ttnn.L1_MEMORY_CONFIG,
        )
        padded_hidden = ttnn.pad(
            tt_hidden,
            [(0, 0), (0, 0), (0, 0), (0, 64)],
            value=0.0,
        )
        hidden_local = ttnn.mesh_partition(
            padded_hidden,
            dim=3,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attention_residual_local = ttnn.add(
            hidden_local,
            projected_local,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        stats = ttnn.rms_norm_pre_all_gather(
            attention_residual_local,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        if stats_gather["buffer"] is None:
            stats_shape = list(stats.shape)
            stats_shape[-1] *= TP_DEGREE
            stats_gather["buffer"] = ttnn.zeros(
                stats_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        gathered_stats = ttnn.experimental.all_gather_async(
            stats,
            persistent_output_buffer=stats_gather["buffer"],
            dim=3,
            multi_device_global_semaphore=decoder.ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=decoder.ccl_manager.num_links,
            topology=decoder.ccl_manager.topology,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gathered_stats = ttnn.mul(
            gathered_stats,
            padded_hidden_size / hidden_size,
        )
        normalized_local = ttnn.rms_norm_post_all_gather(
            attention_residual_local,
            gathered_stats,
            epsilon=decoder.rms_norm_eps,
            weight=candidate_norm_weight,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        router_input = ttnn.typecast(
            ttnn.reshape(normalized_local, [1, local_hidden_size]),
            ttnn.float32,
        )
        router_logits = ttnn.linear(
            router_input,
            candidate_router_weight,
            bias=candidate_router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        router_logits = decoder._ring_all_reduce(
            router_logits,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(
            router_logits,
            decoder.experts_per_token,
            1,
            True,
            True,
        )
        top_values = ttnn.softmax(top_values, 1, numeric_stable=True)
        routing = ttnn.scatter(
            ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_values,
        )

        normalized_padded = ttnn.experimental.all_gather_async(
            normalized_local,
            persistent_output_buffer=normalized_gather,
            dim=3,
            multi_device_global_semaphore=decoder.ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=decoder.ccl_manager.num_links,
            topology=decoder.ccl_manager.topology,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized = ttnn.slice(
            normalized_padded,
            [0, 0, 0, 0],
            [1, 1, 1, hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized_for_expert = ttnn.clone(
            normalized,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        partial_expert = decoder._ep_active_expert_chunk(
            normalized_for_expert,
            routing,
            is_decode=True,
        )
        expert_output = decoder._minimal_all_reduce(
            partial_expert,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        attention_residual_padded = ttnn.experimental.all_gather_async(
            attention_residual_local,
            persistent_output_buffer=residual_gather,
            dim=3,
            multi_device_global_semaphore=decoder.ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=decoder.ccl_manager.num_links,
            topology=decoder.ccl_manager.topology,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_residual = ttnn.slice(
            attention_residual_padded,
            [0, 0, 0, 0],
            [1, 1, 1, hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.add(
            attention_residual,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return attention_residual_local, normalized_local, routing, output

    baseline_residual, baseline_normalized, baseline_routing, baseline_output = baseline_boundary()
    candidate_residual, candidate_normalized, candidate_routing, candidate_output = candidate_boundary()
    ttnn.synchronize_device(mesh_device)
    baseline_residual_local = ttnn.mesh_partition(
        ttnn.pad(
            baseline_residual,
            [(0, 0), (0, 0), (0, 0), (0, 64)],
            value=0.0,
        ),
        dim=3,
        cluster_axis=decoder.mesh_config.tp_axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    baseline_normalized_local = ttnn.mesh_partition(
        ttnn.pad(
            baseline_normalized,
            [(0, 0), (0, 0), (0, 0), (0, 64)],
            value=0.0,
        ),
        dim=3,
        cluster_axis=decoder.mesh_config.tp_axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    pcc_values = {"attention_residual": [], "post_attention_norm": []}
    for name, reference_mesh, actual_mesh in (
        ("attention_residual", baseline_residual_local, candidate_residual),
        ("post_attention_norm", baseline_normalized_local, candidate_normalized),
    ):
        for rank, (reference, actual) in enumerate(zip(_device_torch(reference_mesh), _device_torch(actual_mesh))):
            _assert_pcc(
                reference,
                actual,
                0.99,
                f"layer{layer_idx} deferred fused O+RS {name} rank{rank}",
            )
            pcc_values[name].append(
                float(torch.corrcoef(torch.stack([reference.float().flatten(), actual.float().flatten()]))[0, 1])
            )
    _assert_top4_agreement(
        _replicated_host(baseline_routing),
        _replicated_host(candidate_routing),
        f"layer{layer_idx} deferred fused O+RS router",
        threshold=1.0,
    )
    baseline_output_host = _replicated_host(baseline_output)
    candidate_output_host = _replicated_host(candidate_output)
    _assert_pcc(
        baseline_output_host,
        candidate_output_host,
        0.99,
        f"layer{layer_idx} deferred fused O+RS final output",
    )
    output_pcc = float(
        torch.corrcoef(
            torch.stack(
                [
                    baseline_output_host.float().flatten(),
                    candidate_output_host.float().flatten(),
                ]
            )
        )[0, 1]
    )

    baseline_ms = _time_trace_replay(
        mesh_device,
        baseline_boundary,
        trace_replays,
        f"GPT_OSS_FUSED_O_RS_DEFERRED_BASELINE_LAYER{layer_idx}",
    )
    candidate_ms = _time_trace_replay(
        mesh_device,
        candidate_boundary,
        trace_replays,
        f"GPT_OSS_FUSED_O_RS_DEFERRED_CANDIDATE_LAYER{layer_idx}",
    )
    artifact = {
        "candidate": "fused_o_rs_deferred_through_post_attention_moe",
        "layer_idx": layer_idx,
        "mesh": list(TARGET_MESH_SHAPE),
        "logical_hidden_size": hidden_size,
        "padded_hidden_size": padded_hidden_size,
        "local_hidden_size": local_hidden_size,
        "pcc": {name: min(values) for name, values in pcc_values.items()} | {"final_output": output_pcc},
        "router_top4_exact": True,
        "trace_replays": trace_replays,
        "baseline_replicated_full_boundary_ms": baseline_ms,
        "candidate_deferred_gather_full_boundary_ms": candidate_ms,
        "baseline_over_candidate_speedup": baseline_ms / candidate_ms,
        "candidate_collective_sequence": [
            "fused O matmul + reduce-scatter",
            "RMS statistics all-gather",
            "row-sharded router all-reduce",
            "normalized activation all-gather at sparse gate/up boundary",
            "expert output minimal all-reduce",
            "residual all-gather at final residual add",
        ],
        "contract": (
            "Fused O projection H2944 RS -> local736 attention residual -> "
            "distributed post-attention RMSNorm -> row-sharded router; gather "
            "normalized activations only when the current sparse gate/up H2880 "
            "contract requires them, and gather the residual at the layer output."
        ),
        "rejection_rule": (
            "The family is retained only if this complete layer boundary beats "
            "the selected replicated O-AR + MoE boundary for both layer kinds."
        ),
    }
    OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_ARTIFACT_DIR / f"candidate_fused_o_rs_deferred_layer{layer_idx}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"FUSED_O_RS_DEFERRED_RESULT {json.dumps(artifact, sort_keys=True)} " f"artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_carried_ep_residual_to_distributed_norm_router_and_qkv(
    mesh_device,
    device_params,
    layer_idx,
):
    """Carry a padded EP partial through distributed norm into real consumers."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_TOPOLOGY_CANDIDATES") != "1":
        pytest.skip("set RUN_MULTICHIP_TOPOLOGY_CANDIDATES=1 for topology candidates")
    payload_dtype = ttnn.bfloat16
    hidden_size = 2880
    padded_hidden_size = 2944
    local_hidden_size = padded_hidden_size // TP_DEGREE
    trace_replays = int(os.environ.get("MULTICHIP_TOPOLOGY_TRACE_REPLAYS", "100"))

    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=MultichipConfig(
                decode_collective=DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE,
            ),
        )
    candidate_ccl = CCLManager(
        mesh_device,
        num_links=decoder.multichip_config.num_links,
        topology=ttnn.Topology.Ring,
    )

    # The current real-weight fixture is layer-scoped. Decoder layers share
    # this exact input-norm/router/QKV geometry, so this layer's real weights
    # model the next stack consumer without introducing synthetic tensors.
    q = _state_tensor(state, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
    k = _state_tensor(state, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
    v = _state_tensor(state, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
    q_bias = _state_tensor(state, layer_idx, "self_attn.q_proj.bias").to(torch.bfloat16)
    k_bias = _state_tensor(state, layer_idx, "self_attn.k_proj.bias").to(torch.bfloat16)
    v_bias = _state_tensor(state, layer_idx, "self_attn.v_proj.bias").to(torch.bfloat16)
    packed_rank_weights = []
    packed_rank_biases = []
    for rank in range(TP_DEGREE):
        packed_rank_weights.append(
            torch.cat(
                (
                    torch.chunk(q, TP_DEGREE, dim=0)[rank].T,
                    torch.chunk(k, TP_DEGREE, dim=0)[rank].T,
                    torch.chunk(v, TP_DEGREE, dim=0)[rank].T,
                ),
                dim=-1,
            )
        )
        packed_rank_biases.append(
            torch.cat(
                (
                    torch.chunk(q_bias, TP_DEGREE, dim=0)[rank],
                    torch.chunk(k_bias, TP_DEGREE, dim=0)[rank],
                    torch.chunk(v_bias, TP_DEGREE, dim=0)[rank],
                ),
                dim=-1,
            )
        )
    packed_qkv = torch.cat(packed_rank_weights, dim=-1)
    padded_qkv = torch.nn.functional.pad(packed_qkv, (0, 0, 0, 64))
    packed_bias = torch.cat(packed_rank_biases, dim=-1).reshape(1, 1, -1)
    candidate_qkv_weight = ttnn.from_torch(
        padded_qkv,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    candidate_qkv_bias = ttnn.from_torch(
        packed_bias,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    next_norm = _state_tensor(state, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
    candidate_norm_weight = ttnn.from_torch(
        torch.nn.functional.pad(next_norm, (0, 64)).reshape(1, 1, 1, padded_hidden_size),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_weight = _state_tensor(state, layer_idx, "mlp.router.weight").T.to(torch.bfloat16)
    candidate_router_weight = ttnn.from_torch(
        torch.nn.functional.pad(router_weight, (0, 0, 0, 64)),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    router_bias = _state_tensor(state, layer_idx, "mlp.router.bias").float()
    rank_selective_router_bias = torch.stack(
        [router_bias] + [torch.zeros_like(router_bias) for _ in range(TP_DEGREE - 1)]
    )
    candidate_router_bias = ttnn.from_torch(
        rank_selective_router_bias,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    rs_intermediate = ttnn.zeros(
        [1, 1, 1, padded_hidden_size],
        dtype=payload_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    rs_output = ttnn.zeros(
        [1, 1, 1, local_hidden_size],
        dtype=payload_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    qkv_gather = ttnn.zeros(
        [1, 1, 1, padded_hidden_size],
        dtype=payload_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    stats_gather = {"buffer": None}
    generator = torch.Generator().manual_seed(7171 + layer_idx)
    hidden = torch.randn(
        (1, 1, 1, hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_hidden = _replicated(hidden, mesh_device)

    def baseline_boundary():
        normalized = decoder._decode_norm(tt_hidden, weight_name="post_attention_norm")
        normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
        routing = decoder._route(normalized, 1)
        partial = decoder._ep_active_expert_chunk(normalized, routing, is_decode=True)
        expert_output = decoder._all_reduce(partial, memory_config=ttnn.L1_MEMORY_CONFIG)
        next_hidden = ttnn.add(
            tt_hidden,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        next_normalized = decoder._decode_norm(next_hidden, weight_name="input_norm")
        next_normalized = ttnn.to_memory_config(next_normalized, ttnn.DRAM_MEMORY_CONFIG)
        next_routing = decoder._route(next_normalized, 1)
        qkv_input = ttnn.to_memory_config(next_normalized, decoder.tp_qkv_input_config)
        qkv = ttnn.linear(
            qkv_input,
            decoder.qkv_weight,
            bias=decoder.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=decoder.tp_qkv_output_config,
            program_config=decoder.tp_qkv_program_config,
            compute_kernel_config=decoder.decode_compute_kernel_config,
        )
        return next_hidden, next_normalized, next_routing, qkv

    def candidate_boundary():
        normalized = decoder._decode_norm(tt_hidden, weight_name="post_attention_norm")
        normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
        routing = decoder._route(normalized, 1)
        partial = decoder._ep_active_expert_chunk(normalized, routing, is_decode=True)
        partial = ttnn.pad(partial, [(0, 0), (0, 0), (0, 0), (0, 64)], value=0.0)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            partial,
            persistent_output_buffers=[rs_intermediate, rs_output],
            dim=3,
            multi_device_global_semaphore=candidate_ccl.get_rs_ping_pong_semaphore(),
            num_links=candidate_ccl.num_links,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=candidate_ccl.topology,
            cluster_axis=decoder.mesh_config.tp_axis,
        )
        padded_hidden = ttnn.pad(tt_hidden, [(0, 0), (0, 0), (0, 0), (0, 64)], value=0.0)
        hidden_shard = ttnn.mesh_partition(
            padded_hidden,
            dim=3,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        residual = ttnn.add(
            hidden_shard,
            scattered,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        stats = ttnn.rms_norm_pre_all_gather(
            residual,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        if stats_gather["buffer"] is None:
            stats_shape = list(stats.shape)
            stats_shape[-1] *= TP_DEGREE
            stats_gather["buffer"] = ttnn.zeros(
                stats_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        gathered_stats = ttnn.experimental.all_gather_async(
            stats,
            persistent_output_buffer=stats_gather["buffer"],
            dim=3,
            multi_device_global_semaphore=candidate_ccl.get_ag_ping_pong_semaphore(),
            num_links=candidate_ccl.num_links,
            topology=candidate_ccl.topology,
            cluster_axis=decoder.mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gathered_stats = ttnn.mul(gathered_stats, padded_hidden_size / hidden_size)
        normalized_local = ttnn.rms_norm_post_all_gather(
            residual,
            gathered_stats,
            epsilon=decoder.rms_norm_eps,
            weight=candidate_norm_weight,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        router_input = ttnn.typecast(
            ttnn.reshape(normalized_local, [1, local_hidden_size]),
            ttnn.float32,
        )
        router_logits = ttnn.linear(
            router_input,
            candidate_router_weight,
            bias=candidate_router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        router_logits = decoder._ring_all_reduce(
            router_logits,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(
            router_logits,
            decoder.experts_per_token,
            1,
            True,
            True,
        )
        top_values = ttnn.softmax(top_values, 1, numeric_stable=True)
        next_routing = ttnn.scatter(
            ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_values,
        )

        qkv = ttnn.experimental.all_gather_minimal_matmul_async(
            normalized_local,
            candidate_qkv_weight,
            bias_tensor=candidate_qkv_bias,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=23,
                N_block_size=2,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
            ),
            multi_device_global_semaphore=candidate_ccl.get_ag_ping_pong_semaphore(),
            topology=candidate_ccl.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=decoder.decode_compute_kernel_config,
            persistent_output_buffer=qkv_gather,
            num_links=candidate_ccl.num_links,
            cluster_axis=decoder.mesh_config.tp_axis,
            force_transpose=True,
            num_workers_per_link=4,
            num_buffers_per_channel=2,
        )[0]
        return residual, normalized_local, next_routing, qkv

    baseline_hidden, baseline_normalized, baseline_routing, baseline_qkv = baseline_boundary()
    candidate_hidden, candidate_normalized, candidate_routing, candidate_qkv = candidate_boundary()
    ttnn.synchronize_device(mesh_device)
    baseline_hidden_local = ttnn.mesh_partition(
        ttnn.pad(baseline_hidden, [(0, 0), (0, 0), (0, 0), (0, 64)], value=0.0),
        dim=3,
        cluster_axis=decoder.mesh_config.tp_axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    baseline_normalized_local = ttnn.mesh_partition(
        ttnn.pad(baseline_normalized, [(0, 0), (0, 0), (0, 0), (0, 64)], value=0.0),
        dim=3,
        cluster_axis=decoder.mesh_config.tp_axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    pcc_values = {"residual": [], "normalized": [], "router": [], "packed_qkv": []}
    comparisons = (
        ("residual", baseline_hidden_local, candidate_hidden),
        ("normalized", baseline_normalized_local, candidate_normalized),
        ("router", baseline_routing, candidate_routing),
        ("packed_qkv", baseline_qkv, candidate_qkv),
    )
    for name, reference_mesh, actual_mesh in comparisons:
        reference_shards = _device_torch(reference_mesh)
        actual_shards = _device_torch(actual_mesh)
        for rank, (reference, actual) in enumerate(zip(reference_shards, actual_shards)):
            _assert_pcc(
                reference,
                actual,
                0.999,
                f"layer{layer_idx} carried {name} rank{rank}",
            )
            pcc = torch.corrcoef(torch.stack([reference.float().flatten(), actual.float().flatten()]))[0, 1]
            pcc_values[name].append(float(pcc))
    _assert_top4_agreement(
        _replicated_host(baseline_routing),
        _replicated_host(candidate_routing),
        f"layer{layer_idx} carried router",
        threshold=1.0,
    )

    baseline_ms = _time_trace_replay(
        mesh_device,
        baseline_boundary,
        trace_replays,
        f"GPT_OSS_CARRIED_BASELINE_LAYER{layer_idx}",
    )
    candidate_ms = _time_trace_replay(
        mesh_device,
        candidate_boundary,
        trace_replays,
        f"GPT_OSS_CARRIED_CANDIDATE_LAYER{layer_idx}",
    )
    artifact = {
        "candidate": "carried_ep_residual_distributed_norm_router_qkv",
        "layer_idx": layer_idx,
        "mesh": list(TARGET_MESH_SHAPE),
        "payload_dtype": "bfloat16",
        "logical_hidden_size": hidden_size,
        "padded_hidden_size": padded_hidden_size,
        "local_hidden_size": local_hidden_size,
        "rms_stats_padding_correction": padded_hidden_size / hidden_size,
        "qkv_local_width": packed_qkv.shape[-1] // TP_DEGREE,
        "persistent_buffers": [
            "reduce_scatter_intermediate",
            "reduce_scatter_output",
            "stats_all_gather",
            "qkv_all_gather",
        ],
        "pcc": {name: min(values) for name, values in pcc_values.items()},
        "router_top4_exact": True,
        "trace_replays": trace_replays,
        "baseline_replicated_boundary_ms": baseline_ms,
        "candidate_carried_boundary_ms": candidate_ms,
        "baseline_over_candidate_speedup": baseline_ms / candidate_ms,
        "contract": (
            "EP partial H2880 -> pad2944 -> persistent RS -> local736 residual -> "
            "distributed RMSNorm -> row-sharded router + AR and persistent fused AG+real packed QKV"
        ),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"candidate_carried_ep_residual_layer{layer_idx}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"CARRIED_EP_RESULT {json.dumps(artifact, sort_keys=True)} artifact={path}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_synthetic_non_aligned_prefill_decode_and_local_paged_cache(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    seq_len = 17
    cache_len = 128
    if not _reference_path(layer_idx).is_file():
        pytest.fail(
            f"missing {_reference_path(layer_idx)}; run "
            "test_capture_current_single_chip_optimized_reference in an isolated 1x1 process first"
        )
    optimized_reference = torch.load(_reference_path(layer_idx), weights_only=True)
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._synthetic_state(config)
        reference_layer = functional_test._hf_layer(state, config)
        decoder = _decoder(state, config, mesh_device, layer_idx=layer_idx, max_cache_len=cache_len)
        assert decoder.local_num_heads == 16
        assert decoder.local_num_kv_heads == 2
        assert decoder.local_num_experts == 8
        assert decoder.attention_window == (128 if layer_idx == SLIDING_LAYER else None)

        key_cache, value_cache = decoder.create_kv_cache()
        page_table = decoder.create_page_table([1, 0])
        generator = torch.Generator().manual_seed(9000 + layer_idx)
        hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_prefill, reference_key, reference_value, reference_state = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
        )
        actual_prefill = decoder.prefill_forward(
            _replicated(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        _assert_pcc(
            reference_prefill,
            _replicated_host(actual_prefill),
            0.99,
            f"layer{layer_idx} synthetic prefill S={seq_len}",
        )
        _assert_pcc(
            optimized_reference["prefill"],
            _replicated_host(actual_prefill),
            0.99,
            f"layer{layer_idx} prefill versus optimized TTNN",
        )
        _assert_pcc(
            reference_key,
            _local_cache_host(key_cache, physical_block=1, logical_length=seq_len),
            0.99,
            f"layer{layer_idx} local paged K cache",
        )
        _assert_pcc(
            reference_value,
            _local_cache_host(value_cache, physical_block=1, logical_length=seq_len),
            0.99,
            f"layer{layer_idx} local paged V cache",
        )

        token = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_decode, decode_key, decode_value, _ = functional_test._reference_layer(
            reference_layer,
            token,
            config,
            start_pos=seq_len,
            cache=reference_state,
        )
        position = decoder.create_position_tensor(seq_len)
        decode_input = _replicated(token, mesh_device)
        attention_output = decoder._decode_attention(
            decode_input,
            key_cache,
            value_cache,
            page_table,
            seq_len,
            position,
        )
        _assert_pcc(
            optimized_reference["attention_output"],
            _replicated_host(attention_output),
            0.99,
            f"layer{layer_idx} decode attention versus optimized TTNN",
        )
        normalized = decoder._decode_norm(attention_output, weight_name="post_attention_norm")
        normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
        routing = decoder._route(normalized, 1)
        _assert_pcc(
            optimized_reference["routing"],
            _replicated_host(routing),
            0.99,
            f"layer{layer_idx} routing versus optimized TTNN",
        )
        actual_decode = decoder._multichip_moe_forward(attention_output, 1)
        _assert_pcc(
            optimized_reference["decode"],
            _replicated_host(actual_decode),
            0.99,
            f"layer{layer_idx} decode versus optimized TTNN position={seq_len}",
        )
        _assert_pcc(
            decode_key[:, :, -1:, :],
            _local_cache_host(key_cache, physical_block=1, logical_length=seq_len + 1)[:, :, -1:, :],
            0.99,
            f"layer{layer_idx} appended K cache",
        )
        _assert_pcc(
            decode_value[:, :, -1:, :],
            _local_cache_host(value_cache, physical_block=1, logical_length=seq_len + 1)[:, :, -1:, :],
            0.99,
            f"layer{layer_idx} appended V cache",
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_synthetic_non_aligned_prefill_crosses_page_and_work_chunk(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    seq_len = 129
    cache_len = 192
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._synthetic_state(config)
        reference_layer = functional_test._hf_layer(state, config)
        decoder = _decoder(state, config, mesh_device, layer_idx=layer_idx, max_cache_len=cache_len)
        physical_blocks = list(reversed(range(decoder.num_cache_blocks)))
        key_cache, value_cache = decoder.create_kv_cache()
        page_table = decoder.create_page_table(physical_blocks)
        generator = torch.Generator().manual_seed(29000 + layer_idx)
        hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_prefill, reference_key, reference_value, _ = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
        )
        actual_prefill = decoder.prefill_forward(
            _replicated(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        assert tuple(actual_prefill.shape)[-2:] == (seq_len, config.hidden_size)
        _assert_pcc(
            reference_prefill,
            _replicated_host(actual_prefill),
            0.99,
            f"layer{layer_idx} non-aligned page-crossing prefill S={seq_len}",
        )
        logical_key = _logical_cache_host(key_cache, physical_blocks, seq_len)
        logical_value = _logical_cache_host(value_cache, physical_blocks, seq_len)
        assert logical_key.shape[-2] == logical_value.shape[-2] == seq_len
        reference_cache_len = reference_key.shape[-2]
        _assert_pcc(
            reference_key,
            logical_key[:, :, -reference_cache_len:, :],
            0.99,
            f"layer{layer_idx} non-aligned page-crossing K cache",
        )
        _assert_pcc(
            reference_value,
            logical_value[:, :, -reference_cache_len:, :],
            0.99,
            f"layer{layer_idx} non-aligned page-crossing V cache",
        )


@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_stacked_decoder_device_layout_contract(mesh_device, device_params):
    del device_params
    seq_len = 17
    max_cache_len = 128
    with _functional_helpers_for_layer(SLIDING_LAYER):
        config = functional_test._config()
        sliding_state = functional_test._synthetic_state(config)
    with _functional_helpers_for_layer(FULL_LAYER):
        full_state = functional_test._synthetic_state(config)
    state = {**sliding_state, **full_state}
    sliding = _decoder(
        state,
        config,
        mesh_device,
        layer_idx=SLIDING_LAYER,
        max_cache_len=max_cache_len,
    )
    full = _decoder(
        state,
        config,
        mesh_device,
        layer_idx=FULL_LAYER,
        max_cache_len=max_cache_len,
    )
    sliding_key, sliding_value = sliding.create_kv_cache()
    full_key, full_value = full.create_kv_cache()
    sliding_page_table = sliding.create_page_table([1, 0])
    full_page_table = full.create_page_table([0, 1])
    generator = torch.Generator().manual_seed(31000)
    hidden = torch.randn(
        (1, 1, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    sliding_prefill = sliding.prefill_forward(
        _replicated(hidden, mesh_device),
        key_cache=sliding_key,
        value_cache=sliding_value,
        page_table=sliding_page_table,
    )
    full_prefill = full.prefill_forward(
        sliding_prefill,
        key_cache=full_key,
        value_cache=full_value,
        page_table=full_page_table,
    )
    assert tuple(sliding_prefill.shape) == tuple(full_prefill.shape) == (1, 1, seq_len, config.hidden_size)
    assert sliding_prefill.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert full_prefill.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    _replicated_host(sliding_prefill)
    _replicated_host(full_prefill)

    token = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    sliding_position = sliding.create_position_tensor(seq_len)
    full_position = full.create_position_tensor(seq_len)
    sliding_decode = sliding.decode_forward(
        _replicated(token, mesh_device),
        key_cache=sliding_key,
        value_cache=sliding_value,
        page_table=sliding_page_table,
        cache_position=seq_len,
        cache_position_tensor=sliding_position,
    )
    full_decode = full.decode_forward(
        sliding_decode,
        key_cache=full_key,
        value_cache=full_value,
        page_table=full_page_table,
        cache_position=seq_len,
        cache_position_tensor=full_position,
    )
    assert tuple(sliding_decode.shape) == tuple(full_decode.shape) == (1, 1, 1, config.hidden_size)
    assert sliding_decode.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert full_decode.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    _replicated_host(sliding_decode)
    _replicated_host(full_decode)


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_current_single_chip_full_context_endpoint_reference(
    mesh_device,
    device_params,
    layer_idx,
):
    """Capture the strongest feasible current-optimized endpoint control."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_CONTEXT") != "1":
        pytest.skip("set RUN_MULTICHIP_CONTEXT=1 for the 131072-token cache gate")
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        supported_context = config.max_position_embeddings
        assert supported_context == 131072
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=supported_context,
        )
        key_cache, value_cache = decoder.create_kv_cache()
        hidden = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=torch.Generator().manual_seed(51000 + layer_idx),
            dtype=torch.bfloat16,
        )
        last_position = supported_context - 1
        output = decoder.decode_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=last_position,
        )
        ttnn.synchronize_device(mesh_device)
        output_host = ttnn.to_torch(output)
        key_host = ttnn.to_torch(key_cache)[0, :, last_position].clone()
        value_host = ttnn.to_torch(value_cache)[0, :, last_position].clone()
        assert torch.isfinite(output_host.float()).all()
        assert torch.isfinite(key_host.float()).all()
        assert torch.isfinite(value_host.float()).all()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "output": output_host.clone(),
                "key_row": key_host,
                "value_row": value_host,
                "hidden": hidden,
                "last_position": last_position,
            },
            _context_reference_path(layer_idx),
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_current_single_chip_manual_full_context_endpoint_reference(
    mesh_device,
    device_params,
    layer_idx,
):
    """Capture exact FP32 attention and downstream endpoint boundaries."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_CONTEXT") != "1":
        pytest.skip("set RUN_MULTICHIP_CONTEXT=1 for the 131072-token cache gate")
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        supported_context = config.max_position_embeddings
        assert supported_context == 131072
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=supported_context,
            optimization_config=OptimizationConfig().with_changes(
                use_manual_long_decode_attention=True,
            ),
        )
        key_cache, value_cache = decoder.create_kv_cache()
        hidden = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=torch.Generator().manual_seed(51000 + layer_idx),
            dtype=torch.bfloat16,
        )
        last_position = supported_context - 1
        attention_output = decoder._decode_attention(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
            last_position,
        )
        routing = _prefill_routing(decoder, attention_output, 1)
        output = decoder._optimized_moe_forward(attention_output, 1)
        ttnn.synchronize_device(mesh_device)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "attention_output": functional_test._to_host(attention_output),
                "routing": functional_test._to_host(routing),
                "output": functional_test._to_host(output),
                "key_row": functional_test._to_host(key_cache)[0, :, last_position].clone(),
                "value_row": functional_test._to_host(value_cache)[0, :, last_position].clone(),
                "hidden": hidden,
                "last_position": last_position,
            },
            _context_manual_reference_path(layer_idx),
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_full_context_cache_allocation_and_last_page_update(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    if os.environ.get("RUN_MULTICHIP_CONTEXT") != "1":
        pytest.skip("set RUN_MULTICHIP_CONTEXT=1 for the 131072-token cache gate")
    reference_path = _context_reference_path(layer_idx)
    if not reference_path.is_file():
        pytest.fail(
            f"missing {reference_path}; run "
            "test_capture_current_single_chip_full_context_endpoint_reference "
            "in an isolated 1x1 process first"
        )
    reference = torch.load(reference_path, weights_only=True)
    manual_reference_path = _context_manual_reference_path(layer_idx)
    if not manual_reference_path.is_file():
        pytest.fail(
            f"missing {manual_reference_path}; run "
            "test_capture_current_single_chip_manual_full_context_endpoint_reference "
            "in an isolated 1x1 process first"
        )
    manual_reference = torch.load(manual_reference_path, weights_only=True)
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
    supported_context = config.max_position_embeddings
    assert supported_context == 131072
    decoder = _decoder(
        state,
        config,
        mesh_device,
        layer_idx=layer_idx,
        max_cache_len=supported_context,
    )
    reverse_blocks = list(reversed(range(decoder.num_cache_blocks)))
    page_table = decoder.create_page_table(reverse_blocks)
    key_cache, value_cache = decoder.create_kv_cache()
    expected = (
        supported_context // decoder.multichip_config.page_block_size,
        config.num_key_value_heads // TP_DEGREE,
        decoder.multichip_config.page_block_size,
        config.head_dim,
    )
    assert tuple(key_cache.shape) == tuple(value_cache.shape) == expected
    hidden = reference["hidden"]
    last_position = int(reference["last_position"])
    attention_output = decoder._decode_attention(
        _replicated(hidden, mesh_device),
        key_cache,
        value_cache,
        page_table,
        last_position,
        decoder.create_position_tensor(last_position),
    )
    normalized = decoder._decode_norm(attention_output, weight_name="post_attention_norm")
    normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
    routing = decoder._route(normalized, 1)
    output = decoder._multichip_moe_forward(attention_output, 1)
    assert tuple(output.shape) == (1, 1, 1, config.hidden_size)
    _assert_pcc(
        manual_reference["attention_output"],
        _replicated_host(attention_output),
        0.99,
        f"real layer{layer_idx} advertised-context endpoint exact attention",
    )
    _assert_pcc(
        manual_reference["routing"],
        _replicated_host(routing),
        0.99,
        f"real layer{layer_idx} advertised-context endpoint routing",
    )
    _assert_top4_agreement(
        manual_reference["routing"],
        _replicated_host(routing),
        f"real layer{layer_idx} advertised-context endpoint routing",
        threshold=1.0,
    )
    output_host = _replicated_host(output)
    assert torch.isfinite(output_host.float()).all()
    _assert_pcc(
        manual_reference["output"],
        output_host,
        0.99,
        f"real layer{layer_idx} advertised-context endpoint exact output",
    )
    _assert_pcc(
        reference["output"],
        output_host,
        0.99,
        f"real layer{layer_idx} advertised-context endpoint output",
    )
    physical_page = reverse_blocks[last_position // decoder.multichip_config.page_block_size]
    last_offset = last_position % decoder.multichip_config.page_block_size
    key_rows = []
    value_rows = []
    for rank, (local_key, local_value) in enumerate(zip(_device_torch(key_cache), _device_torch(value_cache))):
        key_row = local_key[physical_page, :, last_offset]
        value_row = local_value[physical_page, :, last_offset]
        assert torch.isfinite(key_row.float()).all()
        assert torch.isfinite(value_row.float()).all()
        assert torch.count_nonzero(key_row).item() > 0, f"rank {rank} K not written at advertised endpoint"
        assert torch.count_nonzero(value_row).item() > 0, f"rank {rank} V not written at advertised endpoint"
        key_rows.append(key_row)
        value_rows.append(value_row)
    _assert_pcc(
        reference["key_row"],
        torch.cat(key_rows, dim=0),
        0.999,
        f"real layer{layer_idx} advertised-context endpoint K row",
    )
    _assert_pcc(
        reference["value_row"],
        torch.cat(value_rows, dim=0),
        0.999,
        f"real layer{layer_idx} advertised-context endpoint V row",
    )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_real_weight_prefill_decode_matches_current_optimized(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    seq_len = 17
    path = _real_reference_path(layer_idx)
    if not path.is_file():
        pytest.fail(
            f"missing {path}; run test_capture_real_weight_current_single_chip_optimized_reference "
            "in an isolated 1x1 process first"
        )
    reference = torch.load(path, weights_only=True)
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=128,
            multichip_config=_multichip_candidate_config_from_env(),
        )
        key_cache, value_cache = decoder.create_kv_cache()
        page_table = decoder.create_page_table([1, 0])
        generator = torch.Generator().manual_seed(17000 + layer_idx)
        hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        token = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        prefill = decoder.prefill_forward(
            _replicated(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        _assert_pcc(
            reference["prefill"],
            _replicated_host(prefill),
            0.99,
            f"real layer{layer_idx} prefill versus optimized TTNN",
        )
        _assert_pcc(
            reference["key_cache"][:, :, :seq_len, :],
            _local_cache_host(key_cache, physical_block=1, logical_length=seq_len),
            0.99,
            f"real layer{layer_idx} prefill K cache",
        )
        _assert_pcc(
            reference["value_cache"][:, :, :seq_len, :],
            _local_cache_host(value_cache, physical_block=1, logical_length=seq_len),
            0.99,
            f"real layer{layer_idx} prefill V cache",
        )

        position = decoder.create_position_tensor(seq_len)
        attention_output = decoder._decode_attention(
            _replicated(token, mesh_device),
            key_cache,
            value_cache,
            page_table,
            seq_len,
            position,
        )
        _assert_pcc(
            reference["attention_output"],
            _replicated_host(attention_output),
            0.99,
            f"real layer{layer_idx} decode attention",
        )
        normalized = decoder._decode_norm(attention_output, weight_name="post_attention_norm")
        normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
        routing = decoder._route(normalized, 1)
        routing_host = _replicated_host(routing)
        assert torch.count_nonzero(routing_host, dim=1).tolist() == [decoder.experts_per_token]
        _assert_pcc(
            reference["routing"],
            routing_host,
            0.99,
            f"real layer{layer_idx} routing",
        )
        decode = decoder._multichip_moe_forward(attention_output, 1)
        _assert_pcc(
            reference["decode"],
            _replicated_host(decode),
            0.99,
            f"real layer{layer_idx} decode",
        )
        _assert_pcc(
            reference["key_cache"][:, :, seq_len : seq_len + 1, :],
            _local_cache_host(key_cache, physical_block=1, logical_length=seq_len + 1)[:, :, -1:, :],
            0.99,
            f"real layer{layer_idx} appended K",
        )
        _assert_pcc(
            reference["value_cache"][:, :, seq_len : seq_len + 1, :],
            _local_cache_host(value_cache, physical_block=1, logical_length=seq_len + 1)[:, :, -1:, :],
            0.99,
            f"real layer{layer_idx} appended V",
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@pytest.mark.parametrize("seq_len", [128, 129, 2048])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_real_weight_precision_lengths_match_current_optimized(
    mesh_device,
    device_params,
    layer_idx,
    seq_len,
):
    """Qualify aligned, non-aligned, and bounded-long real-weight prefill."""

    del device_params
    path = _precision_reference_path(layer_idx, seq_len)
    if not path.is_file():
        pytest.fail(
            f"missing {path}; run test_capture_real_weight_current_single_chip_precision_reference "
            "in an isolated 1x1 process first"
        )
    reference = torch.load(path, weights_only=True)
    max_cache_len = seq_len + 1
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=max_cache_len,
            multichip_config=_multichip_candidate_config_from_env(),
            optimization_config=_precision_optimization_variant(),
        )
        physical_blocks = list(reversed(range(decoder.num_cache_blocks)))
        page_table = decoder.create_page_table(physical_blocks)
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(61000 + layer_idx * 10000 + seq_len)
        hidden = torch.randn(
            (1, 1, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        token = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        attention_output = decoder._prefill_attention(
            _replicated(hidden, mesh_device),
            key_cache,
            value_cache,
            page_table,
            seq_len,
        )
        _assert_pcc(
            reference["attention_output"],
            _replicated_host(attention_output),
            0.99,
            f"real layer{layer_idx} attention residual S={seq_len}",
        )
        prefill_routing = _prefill_routing(decoder, attention_output, seq_len)
        prefill_routing_host = _replicated_host(prefill_routing)
        _assert_pcc(
            reference["prefill_routing"],
            prefill_routing_host,
            0.99,
            f"real layer{layer_idx} prefill routing S={seq_len}",
        )
        _assert_top4_agreement(
            reference["prefill_routing"],
            prefill_routing_host,
            f"real layer{layer_idx} prefill routing S={seq_len}",
        )
        prefill = decoder._multichip_moe_forward(attention_output, seq_len)
        _assert_pcc(
            reference["prefill"],
            _replicated_host(prefill),
            0.99,
            f"real layer{layer_idx} final prefill S={seq_len}",
        )
        logical_key = _logical_cache_host(key_cache, physical_blocks, seq_len)
        logical_value = _logical_cache_host(value_cache, physical_blocks, seq_len)
        _assert_pcc(
            reference["prefill_key"],
            logical_key,
            0.99,
            f"real layer{layer_idx} logical paged K S={seq_len}",
        )
        _assert_pcc(
            reference["prefill_value"],
            logical_value,
            0.99,
            f"real layer{layer_idx} logical paged V S={seq_len}",
        )

        decode_attention = decoder._decode_attention(
            _replicated(token, mesh_device),
            key_cache,
            value_cache,
            page_table,
            seq_len,
            decoder.create_position_tensor(seq_len),
        )
        _assert_pcc(
            reference["decode_attention"],
            _replicated_host(decode_attention),
            0.99,
            f"real layer{layer_idx} following decode attention position={seq_len}",
        )
        decode_routing = _prefill_routing(decoder, decode_attention, 1)
        decode_routing_host = _replicated_host(decode_routing)
        _assert_pcc(
            reference["decode_routing"],
            decode_routing_host,
            0.99,
            f"real layer{layer_idx} following decode routing position={seq_len}",
        )
        _assert_top4_agreement(
            reference["decode_routing"],
            decode_routing_host,
            f"real layer{layer_idx} following decode routing position={seq_len}",
        )
        decode = decoder._multichip_moe_forward(decode_attention, 1)
        _assert_pcc(
            reference["decode"],
            _replicated_host(decode),
            0.99,
            f"real layer{layer_idx} following decode position={seq_len}",
        )
        appended_key = _logical_cache_host(key_cache, physical_blocks, seq_len + 1)[:, :, -1:, :]
        appended_value = _logical_cache_host(value_cache, physical_blocks, seq_len + 1)[:, :, -1:, :]
        _assert_pcc(
            reference["appended_key"],
            appended_key,
            0.99,
            f"real layer{layer_idx} appended logical K position={seq_len}",
        )
        _assert_pcc(
            reference["appended_value"],
            appended_value,
            0.99,
            f"real layer{layer_idx} appended logical V position={seq_len}",
        )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_real_weight_boundary_positions_match_current_optimized(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    path = _boundary_reference_path(layer_idx)
    if not path.is_file():
        pytest.fail(
            f"missing {path}; run test_capture_real_weight_current_single_chip_boundary_reference "
            "in an isolated 1x1 process first"
        )
    reference = torch.load(path, weights_only=True)
    prefill_len = reference["prefill_len"]
    decode_positions = reference["decode_positions"]
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(state, config, mesh_device, layer_idx=layer_idx, max_cache_len=256)
        key_cache, value_cache = decoder.create_kv_cache()
        physical_blocks = list(reversed(range(decoder.num_cache_blocks)))
        page_table = decoder.create_page_table(physical_blocks)
        generator = torch.Generator().manual_seed(41000 + layer_idx)
        hidden = torch.randn(
            (1, 1, prefill_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        tokens = [
            torch.randn(
                (1, 1, 1, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            )
            for _ in decode_positions
        ]
        prefill = decoder.prefill_forward(
            _replicated(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        _assert_pcc(
            reference["prefill"],
            _replicated_host(prefill),
            0.99,
            f"real layer{layer_idx} boundary prefill S={prefill_len}",
        )
        _assert_pcc(
            reference["prefill_key_cache"],
            _head_sharded_host(decoder._logical_cache_prefix(key_cache, page_table, prefill_len)),
            0.99,
            f"real layer{layer_idx} boundary logical K gather",
        )
        _assert_pcc(
            reference["prefill_value_cache"],
            _head_sharded_host(decoder._logical_cache_prefix(value_cache, page_table, prefill_len)),
            0.99,
            f"real layer{layer_idx} boundary logical V gather",
        )
        for position, token, expected_attention, expected_routing, expected in zip(
            decode_positions,
            tokens,
            reference["attention"],
            reference["routing"],
            reference["decode"],
        ):
            attention = decoder._decode_attention(
                _replicated(token, mesh_device),
                key_cache,
                value_cache,
                page_table,
                position,
                decoder.create_position_tensor(position),
            )
            _assert_pcc(
                expected_attention,
                _replicated_host(attention),
                0.99,
                f"real layer{layer_idx} boundary attention position={position}",
            )
            normalized = decoder._decode_norm(attention, weight_name="post_attention_norm")
            normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
            routing = decoder._route(normalized, 1)
            _assert_pcc(
                expected_routing,
                _replicated_host(routing),
                0.99,
                f"real layer{layer_idx} boundary routing position={position}",
            )
            output = decoder._multichip_moe_forward(attention, 1)
            _assert_pcc(
                expected,
                _replicated_host(output),
                0.99,
                f"real layer{layer_idx} boundary decode position={position}",
            )


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_warmed_trace_replay_mutates_hidden_position_and_local_cache(
    mesh_device,
    device_params,
    layer_idx,
):
    del device_params
    prefill_len = 17
    max_cache_len = 256
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=max_cache_len,
            multichip_config=_multichip_candidate_config_from_env(),
        )
        reverse_blocks = list(reversed(range(decoder.num_cache_blocks)))
        eager_page_table = decoder.create_page_table(reverse_blocks)
        trace_page_table = decoder.create_page_table(reverse_blocks)
        eager_key, eager_value = decoder.create_kv_cache()
        trace_key, trace_value = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(24000 + layer_idx)
        prefill_hidden = torch.randn(
            (1, 1, prefill_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        prefill_device = _replicated(prefill_hidden, mesh_device)
        decoder.prefill_forward(
            prefill_device,
            key_cache=eager_key,
            value_cache=eager_value,
            page_table=eager_page_table,
        )
        decoder.prefill_forward(
            prefill_device,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
        )

        cases = []
        for position in range(prefill_len, prefill_len + 3):
            hidden = torch.randn(
                (1, 1, 1, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            )
            cases.append(
                {
                    "position": position,
                    "hidden": hidden,
                    "eager_hidden": _replicated(hidden, mesh_device),
                    "eager_position": decoder.create_position_tensor(position),
                    "host_hidden": _host_replicated(hidden, mesh_device),
                    "host_position": _host_replicated(
                        torch.tensor([position], dtype=torch.int32),
                        mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                }
            )

        eager_outputs = []
        for case in cases:
            eager = decoder.decode_forward(
                case["eager_hidden"],
                key_cache=eager_key,
                value_cache=eager_value,
                page_table=eager_page_table,
                cache_position=case["position"],
                cache_position_tensor=case["eager_position"],
            )
            ttnn.synchronize_device(mesh_device)
            eager_outputs.append(_replicated_host(eager).clone())

        trace_hidden = _replicated(cases[0]["hidden"], mesh_device)
        trace_position = decoder.create_position_tensor(prefill_len)
        warm = decoder.decode_forward(
            trace_hidden,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
            cache_position=prefill_len,
            cache_position_tensor=trace_position,
        )
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = decoder.decode_forward(
            trace_hidden,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
            cache_position=prefill_len,
            cache_position_tensor=trace_position,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            for case, eager_output in zip(cases, eager_outputs):
                ttnn.copy_host_to_device_tensor(case["host_hidden"], trace_hidden)
                ttnn.copy_host_to_device_tensor(case["host_position"], trace_position)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                actual = _replicated_host(trace_output)
                _assert_pcc(
                    eager_output,
                    actual,
                    0.999,
                    f"real layer{layer_idx} trace mutable position={case['position']}",
                )

            deterministic = _replicated_host(trace_output).clone()
            for replay in range(5):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                assert torch.equal(
                    deterministic,
                    _replicated_host(trace_output),
                ), f"layer{layer_idx} trace replay {replay} changed"
        finally:
            ttnn.release_trace(mesh_device, trace_id)

        last_position = cases[-1]["position"]
        logical_block = last_position // decoder.multichip_config.page_block_size
        physical_block = reverse_blocks[logical_block]
        for name, eager_cache, trace_cache in (
            ("key", eager_key, trace_key),
            ("value", eager_value, trace_value),
        ):
            for rank, (eager_local, trace_local) in enumerate(
                zip(_device_torch(eager_cache), _device_torch(trace_cache))
            ):
                assert torch.equal(
                    eager_local[physical_block],
                    trace_local[physical_block],
                ), f"layer{layer_idx} rank{rank} traced {name} page differs from eager"


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_warmed_long_position_trace_replay_matches_eager(
    mesh_device,
    device_params,
    layer_idx,
):
    """Page-banked manual attention replays mutable positions and cache writes."""

    del device_params
    prefill_len = 128
    max_cache_len = 256
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=max_cache_len,
            multichip_config=_multichip_candidate_config_from_env(),
        )
        reverse_blocks = list(reversed(range(decoder.num_cache_blocks)))
        eager_page_table = decoder.create_page_table(reverse_blocks)
        trace_page_table = decoder.create_page_table(reverse_blocks)
        eager_key, eager_value = decoder.create_kv_cache()
        trace_key, trace_value = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(52000 + layer_idx)
        prefill_hidden = torch.randn(
            (1, 1, prefill_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        prefill_device = _replicated(prefill_hidden, mesh_device)
        decoder.prefill_forward(
            prefill_device,
            key_cache=eager_key,
            value_cache=eager_value,
            page_table=eager_page_table,
        )
        decoder.prefill_forward(
            prefill_device,
            key_cache=trace_key,
            value_cache=trace_value,
            page_table=trace_page_table,
        )

        bank_positions = ((128, 129, 130, 131, 191), (192, 193))
        cases = {}
        eager_outputs = {}
        for position in (position for bank in bank_positions for position in bank):
            hidden = torch.randn(
                (1, 1, 1, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            )
            cases[position] = {
                "hidden": hidden,
                "host_hidden": _host_replicated(hidden, mesh_device),
                "host_position": _host_replicated(
                    torch.tensor([position], dtype=torch.int32),
                    mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            }
            eager = decoder.decode_forward(
                _replicated(hidden, mesh_device),
                key_cache=eager_key,
                value_cache=eager_value,
                page_table=eager_page_table,
                cache_position=position,
                cache_position_tensor=decoder.create_position_tensor(position),
            )
            ttnn.synchronize_device(mesh_device)
            eager_outputs[position] = _replicated_host(eager).clone()

        replay_count = 0
        elapsed = 0.0
        for positions in bank_positions:
            capture_position = positions[0]
            trace_hidden = _replicated(cases[capture_position]["hidden"], mesh_device)
            trace_position = decoder.create_position_tensor(capture_position)
            warm = decoder.decode_forward(
                trace_hidden,
                key_cache=trace_key,
                value_cache=trace_value,
                page_table=trace_page_table,
                cache_position=capture_position,
                cache_position_tensor=trace_position,
            )
            ttnn.synchronize_device(mesh_device)
            warm.deallocate(True)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            trace_output = decoder.decode_forward(
                trace_hidden,
                key_cache=trace_key,
                value_cache=trace_value,
                page_table=trace_page_table,
                cache_position=capture_position,
                cache_position_tensor=trace_position,
            )
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            try:
                start = time.perf_counter()
                for position in positions:
                    ttnn.copy_host_to_device_tensor(cases[position]["host_hidden"], trace_hidden)
                    ttnn.copy_host_to_device_tensor(cases[position]["host_position"], trace_position)
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    actual = _replicated_host(trace_output)
                    _assert_pcc(
                        eager_outputs[position],
                        actual,
                        0.999,
                        f"real layer{layer_idx} trace-bank decode position={position}",
                    )
                    physical_block = reverse_blocks[position // decoder.multichip_config.page_block_size]
                    offset = position % decoder.multichip_config.page_block_size
                    for name, eager_cache, trace_cache in (
                        ("key", eager_key, trace_key),
                        ("value", eager_value, trace_value),
                    ):
                        for rank, (eager_local, trace_local) in enumerate(
                            zip(_device_torch(eager_cache), _device_torch(trace_cache))
                        ):
                            assert torch.equal(
                                eager_local[physical_block, :, offset],
                                trace_local[physical_block, :, offset],
                            ), f"layer{layer_idx} rank{rank} traced {name} row differs at position={position}"
                elapsed += time.perf_counter() - start
                replay_count += len(positions)

                deterministic = _replicated_host(trace_output).clone()
                for replay in range(5):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    assert torch.equal(
                        deterministic,
                        _replicated_host(trace_output),
                    ), f"layer{layer_idx} trace-bank replay {replay} changed"
            finally:
                ttnn.release_trace(mesh_device, trace_id)

        print(
            f"layer{layer_idx} sequential trace-bank replay including input refresh, "
            f"sync, and validation readback: {elapsed * 1000.0 / replay_count:.6f} ms"
        )


def _time_single_chip_prefill(decoder, mesh_device, hidden, repeats: int, layer_idx: int):
    key_cache, value_cache = decoder.create_kv_cache()
    device_hidden = functional_test._tt_tensor(hidden, mesh_device)
    decoder.prefill_forward(device_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header=f"GPT_OSS_SINGLE_PREFILL_LAYER{layer_idx}")
    start = time.perf_counter()
    for _ in range(repeats):
        decoder.prefill_forward(device_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    signpost(header=f"GPT_OSS_SINGLE_PREFILL_LAYER{layer_idx}_END")
    return elapsed_ms, key_cache, value_cache


def _time_multichip_prefill(decoder, mesh_device, hidden, page_table, repeats: int, layer_idx: int):
    key_cache, value_cache = decoder.create_kv_cache()
    device_hidden = _replicated(hidden, mesh_device)
    decoder.prefill_forward(
        device_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    signpost(header=f"GPT_OSS_MULTICHIP_PREFILL_LAYER{layer_idx}")
    start = time.perf_counter()
    for _ in range(repeats):
        decoder.prefill_forward(
            device_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    signpost(header=f"GPT_OSS_MULTICHIP_PREFILL_LAYER{layer_idx}_END")
    return elapsed_ms, key_cache, value_cache


def _time_trace_replay(mesh_device, capture, trace_replays: int, label: str):
    capture()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    capture()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost(header=label)
        start = time.perf_counter()
        for _ in range(trace_replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / trace_replays
        signpost(header=f"{label}_END")
        return elapsed_ms
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([(1, 1)])
def test_capture_current_single_chip_optimized_perf(mesh_device, device_params, layer_idx):
    del device_params
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to capture warmed 1x1 timing")
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", "128"))
    prefill_repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "20"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "500"))
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_cache_len=max(256, seq_len + 1),
        )
    generator = torch.Generator().manual_seed(61000 + layer_idx)
    prefill_hidden = torch.randn(
        (1, 1, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    decode_hidden = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    prefill_ms, key_cache, value_cache = _time_single_chip_prefill(
        decoder,
        mesh_device,
        prefill_hidden,
        prefill_repeats,
        layer_idx,
    )
    device_decode = functional_test._tt_tensor(decode_hidden, mesh_device)
    decode_ms = _time_trace_replay(
        mesh_device,
        lambda: decoder.decode_forward(
            device_decode,
            key_cache,
            value_cache,
            current_pos=seq_len,
        ),
        trace_replays,
        f"GPT_OSS_SINGLE_DECODE_LAYER{layer_idx}",
    )
    result = {
        "baseline": "current OptimizedDecoder",
        "layer_idx": layer_idx,
        "layer_kind": "sliding" if layer_idx == SLIDING_LAYER else "full",
        "mesh": [1, 1],
        "seq_len": seq_len,
        "prefill_ms": prefill_ms,
        "traced_decode_ms": decode_ms,
        "prefill_repeats": prefill_repeats,
        "trace_replays": trace_replays,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _perf_reference_path(layer_idx, seq_len).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"SINGLE_CHIP_PERF {json.dumps(result, sort_keys=True)}")


@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_multichip_decoder_perf(mesh_device, device_params, layer_idx):
    del device_params
    if os.environ.get("RUN_MULTICHIP_DECODER_PERF") != "1":
        pytest.skip("set RUN_MULTICHIP_DECODER_PERF=1 to capture warmed 1x4 timing")
    seq_len = int(os.environ.get("MULTICHIP_DECODER_PERF_SEQ", "128"))
    prefill_repeats = int(os.environ.get("MULTICHIP_DECODER_PREFILL_REPEATS", "20"))
    trace_replays = int(os.environ.get("MULTICHIP_DECODER_TRACE_REPLAYS", "500"))
    reference_path = _perf_reference_path(layer_idx, seq_len)
    if not reference_path.is_file():
        pytest.fail(f"missing isolated 1x1 perf reference {reference_path}")
    baseline = json.loads(reference_path.read_text())
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=max(256, seq_len + 1),
            multichip_config=_multichip_candidate_config_from_env(),
        )
    page_table = decoder.create_page_table(list(reversed(range(decoder.num_cache_blocks))))
    generator = torch.Generator().manual_seed(61000 + layer_idx)
    prefill_hidden = torch.randn(
        (1, 1, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    decode_hidden = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    prefill_ms, key_cache, value_cache = _time_multichip_prefill(
        decoder,
        mesh_device,
        prefill_hidden,
        page_table,
        prefill_repeats,
        layer_idx,
    )
    device_decode = _replicated(decode_hidden, mesh_device)
    position = decoder.create_position_tensor(seq_len)
    decode_ms = _time_trace_replay(
        mesh_device,
        lambda: decoder.decode_forward(
            device_decode,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            cache_position=seq_len,
            cache_position_tensor=position,
        ),
        trace_replays,
        f"GPT_OSS_MULTICHIP_DECODE_LAYER{layer_idx}",
    )
    candidate_config = decoder.multichip_config
    result = {
        "baseline": baseline,
        "implementation": (f"TP4 attention + {candidate_config.expert_strategy.upper()}4 gate-selected active experts"),
        "decode_collective": candidate_config.decode_collective,
        "expert_strategy": candidate_config.expert_strategy,
        "fused_o_projection_rs": candidate_config.use_fused_o_projection_rs,
        "fused_o_projection_ag": candidate_config.use_fused_o_projection_ag,
        "fused_o_ag_pad_hidden": candidate_config.fused_o_ag_pad_hidden,
        "fused_ag_matmul_payload_dtype": candidate_config.fused_ag_matmul_payload_dtype,
        "attention_weight_dtype": candidate_config.attention_weight_dtype,
        "decode_attention_weight_dtype": candidate_config.decode_attention_weight_dtype,
        "short_decode_attention_weight_dtype": candidate_config.decode_attention_weight_dtype,
        "long_decode_attention_weight_dtype": candidate_config.attention_weight_dtype,
        "attention_math_fidelity": candidate_config.attention_math_fidelity,
        "long_decode_attention_math_fidelity": candidate_config.long_decode_attention_math_fidelity,
        "expert_weight_dtype": candidate_config.expert_weight_dtype,
        "expert_gate_up_weight_dtype": candidate_config.expert_gate_up_weight_dtype,
        "expert_down_weight_dtype": candidate_config.expert_down_weight_dtype,
        "expert_activation_dtype": candidate_config.expert_activation_dtype,
        "expert_math_fidelity": candidate_config.expert_math_fidelity,
        "decode_ccl_dtype": candidate_config.decode_ccl_dtype,
        "prefill_expert_cores": candidate_config.prefill_expert_cores,
        "prefill_expert_in0_block_w": candidate_config.expert_in0_block_w,
        "prefill_expert_subblock_w": candidate_config.prefill_expert_subblock_w,
        "prefill_expert_output_l1": candidate_config.prefill_expert_output_l1,
        "prefill_expert_output_l1_max_seq": candidate_config.prefill_expert_output_l1_max_seq,
        "decode_expert_output_l1": candidate_config.decode_expert_output_l1,
        "decode_gate_up_cores": candidate_config.decode_gate_up_cores,
        "decode_gate_up_in0_block_w": candidate_config.decode_gate_up_in0_block_w,
        "decode_gate_up_subblock_w": candidate_config.decode_gate_up_subblock_w,
        "decode_down_cores": candidate_config.decode_down_cores,
        "decode_down_in0_block_w": candidate_config.decode_down_in0_block_w,
        "decode_down_subblock_w": candidate_config.decode_down_subblock_w,
        "use_dram_sharded_decode_attention": candidate_config.use_dram_sharded_decode_attention,
        "dram_attention_core_limit": candidate_config.dram_attention_core_limit,
        "layer_idx": layer_idx,
        "layer_kind": "sliding" if layer_idx == SLIDING_LAYER else "full",
        "mesh": list(TARGET_MESH_SHAPE),
        "seq_len": seq_len,
        "multichip_prefill_ms": prefill_ms,
        "multichip_traced_decode_ms": decode_ms,
        "prefill_speedup": baseline["prefill_ms"] / prefill_ms,
        "prefill_efficiency": baseline["prefill_ms"] / prefill_ms / TP_DEGREE,
        "decode_speedup": baseline["traced_decode_ms"] / decode_ms,
        "decode_efficiency": baseline["traced_decode_ms"] / decode_ms / TP_DEGREE,
        "prefill_repeats": prefill_repeats,
        "trace_replays": trace_replays,
    }
    _perf_result_path(layer_idx, seq_len).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"MULTICHIP_PERF {json.dumps(result, sort_keys=True)}")


@pytest.mark.parametrize("seq_len", [17, 128])
@pytest.mark.parametrize("layer_idx", [SLIDING_LAYER, FULL_LAYER])
@parametrize_mesh_with_fabric([TARGET_MESH_SHAPE])
def test_profile_ep_activity_accounting(mesh_device, device_params, layer_idx, seq_len):
    """Capture rank-local EP activity for the exact profiler inputs."""

    del device_params
    if os.environ.get("RUN_MULTICHIP_EP_ACCOUNTING") != "1":
        pytest.skip("set RUN_MULTICHIP_EP_ACCOUNTING=1 to capture profiler EP activity")
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(
            state,
            config,
            mesh_device,
            layer_idx=layer_idx,
            max_cache_len=max(256, seq_len + 1),
        )
    page_table = decoder.create_page_table(list(reversed(range(decoder.num_cache_blocks))))
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(61000 + layer_idx)
    prefill_hidden = torch.randn(
        (1, 1, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    decode_hidden = torch.randn(
        (1, 1, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    attention_output = decoder._prefill_attention(
        _replicated(prefill_hidden, mesh_device),
        key_cache,
        value_cache,
        page_table,
        seq_len,
    )
    prefill_routing = _replicated_host(_prefill_routing(decoder, attention_output, seq_len))
    position = decoder.create_position_tensor(seq_len)
    decode_attention = decoder._decode_attention(
        _replicated(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        page_table,
        seq_len,
        position,
    )
    decode_normalized = decoder._decode_norm(decode_attention, weight_name="post_attention_norm")
    decode_normalized = ttnn.to_memory_config(decode_normalized, ttnn.DRAM_MEMORY_CONFIG)
    decode_routing = _replicated_host(decoder._route(decode_normalized, 1))

    def summarize(routing: torch.Tensor):
        selected = routing.reshape(-1, decoder.num_experts).ne(0)
        ranks = []
        for rank in range(EP_DEGREE):
            local = selected[:, rank * decoder.local_num_experts : (rank + 1) * decoder.local_num_experts]
            per_token = local.sum(dim=1)
            histogram = torch.bincount(per_token, minlength=decoder.experts_per_token + 1)
            ranks.append(
                {
                    "rank": rank,
                    "per_token_max": int(per_token.max().item()),
                    "per_token_histogram": {
                        str(count): int(tokens) for count, tokens in enumerate(histogram.tolist()) if tokens
                    },
                    "unique_active_experts_in_batch_group": int(local.any(dim=0).sum().item()),
                }
            )
        return ranks

    result = {
        "mesh": list(TARGET_MESH_SHAPE),
        "expert_parallel_degree": EP_DEGREE,
        "global_experts": decoder.num_experts,
        "local_experts_per_rank": decoder.local_num_experts,
        "global_selected_experts_per_token": decoder.experts_per_token,
        "layer_idx": layer_idx,
        "layer_kind": "sliding" if layer_idx == SLIDING_LAYER else "full",
        "seq_len": seq_len,
        "input_seed": 61000 + layer_idx,
        "prefill_rank_activity": summarize(prefill_routing),
        "decode_rank_activity": summarize(decode_routing),
    }
    OPTIMIZED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_ARTIFACT_DIR / f"profile_ep_activity_layer{layer_idx}_seq{seq_len}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"EP_ACTIVITY {json.dumps(result, sort_keys=True)} artifact={path}")
