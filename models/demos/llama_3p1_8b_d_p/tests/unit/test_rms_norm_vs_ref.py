# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for plain Llama-3.1 RMSNorm."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _device_addresses
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics as _metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import RMSNorm

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP = MESH_SHAPE[0]
TP = MESH_SHAPE[1]
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
EPSILON = Llama31_8BConfig.RMS_NORM_EPS
SCOPED_DFB_BYTES_PER_CORE = 1_146_880
REAL_WEIGHT_NAMES = (
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
)


def _load_selected_checkpoint_weights():
    return read_raw_weights(HF_MODEL, REAL_WEIGHT_NAMES)


def _synthetic_gamma():
    return torch.linspace(0.35, 1.65, HIDDEN_SIZE, dtype=torch.float32)


def _reference_rms_norm(host_input, weight):
    rounded_weight = weight.to(torch.bfloat16).float()
    reference = LlamaRMSNorm(HIDDEN_SIZE, eps=EPSILON).float()
    with torch.no_grad():
        reference.weight.copy_(rounded_weight)
    return reference(host_input.to(torch.bfloat16).float())


def _run_case(mesh_device, norm, weight, host_input, *, label):
    rounded_input = host_input.to(torch.bfloat16)
    expected = _reference_rms_norm(rounded_input, weight)
    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None))
    tt_input = ttnn.from_torch(
        rounded_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=input_mapper,
    )

    input_addresses = _device_addresses(tt_input)
    before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tt_input)]
    tt_output = norm(tt_input)
    ttnn.synchronize_device(mesh_device)

    assert tuple(tt_output.shape) == (1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config() == ttnn.DRAM_MEMORY_CONFIG

    output_shards = ttnn.get_device_tensors(tt_output)
    input_shards = ttnn.get_device_tensors(tt_input)
    assert len(output_shards) == SP * TP
    assert len(input_shards) == SP * TP
    errors = []
    for sp_coord in range(SP):
        seq_start = sp_coord * LOCAL_SEQUENCE
        expected_shard = expected[:, :, seq_start : seq_start + LOCAL_SEQUENCE, :]
        input_shard_expected = rounded_input[:, :, seq_start : seq_start + LOCAL_SEQUENCE, :]
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual = ttnn.to_torch(output_shards[device_idx])[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE]
            after = ttnn.to_torch(input_shards[device_idx])
            assert torch.equal(before[device_idx], after), f"{label} device={device_idx}: input was modified"
            assert torch.equal(
                after[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE], input_shard_expected
            ), f"{label} device={device_idx}: wrong SP rows or TP replication"
            assert torch.isfinite(actual).all(), f"{label} device={device_idx}: nonfinite output"

            if torch.count_nonzero(expected_shard) == 0:
                assert torch.count_nonzero(actual) == 0, f"{label} device={device_idx}: zero input was not exact zero"
                errors.append((device_idx, 1.0, 0.0))
            else:
                pcc, nl2 = _metrics(expected_shard, actual)
                errors.append((device_idx, pcc, nl2))
                assert pcc >= 0.9999, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
                assert nl2 <= 0.01, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"

    logger.info(
        f"RMSNorm {label}: min_PCC={min(item[1] for item in errors):.7f}, "
        f"max_NL2={max(item[2] for item in errors):.7f} over {len(errors)} devices; "
        f"input_addresses={[hex(address) for address in input_addresses]}"
    )
    tt_output.deallocate(True)
    tt_input.deallocate(True)
    return input_addresses


# Run plain RMSNorm against the independent Transformers implementation on all 32 chips; the
# synthetic and checkpoint cases catch mean subtraction, epsilon loss, wrong gamma, stale cached
# addresses, TP feature sharding, SP row replication errors, and destructive residual input reuse.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_plain_rms_norm_matches_transformers_on_every_chip_and_reuses_program(mesh_device, expect_error):
    torch.manual_seed(20260915)
    synthetic_weight = _synthetic_gamma()
    checkpoint_weights = _load_selected_checkpoint_weights()
    norms = {
        "synthetic": RMSNorm(mesh_device, synthetic_weight),
        **{name: RMSNorm(mesh_device, weight) for name, weight in checkpoint_weights.items()},
    }

    assert tuple(norms["synthetic"].weight.shape) == (1, 1, 128, 32)
    assert norms["synthetic"].weight.dtype == ttnn.bfloat16
    assert norms["synthetic"].weight.layout == ttnn.ROW_MAJOR_LAYOUT
    assert norms["synthetic"].weight.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    weight_addresses = {name: _device_addresses(norm.weight) for name, norm in norms.items()}
    assert len({addresses[0] for addresses in weight_addresses.values()}) == len(weight_addresses)
    logger.info(
        "RMSNorm gamma addresses: "
        + ", ".join(f"{name}={hex(addresses[0])}" for name, addresses in weight_addresses.items())
    )

    l1_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    logger.info(
        f"RMSNorm live L1 before launch: total_bytes_per_bank={l1_before.total_bytes_per_bank}, "
        f"largest_contiguous_bytes_free_per_bank={l1_before.largest_contiguous_bytes_free_per_bank}, "
        f"scoped_named_DFB_bytes_per_core={SCOPED_DFB_BYTES_PER_CORE}"
    )
    assert l1_before.total_bytes_per_bank >= SCOPED_DFB_BYTES_PER_CORE
    assert l1_before.largest_contiguous_bytes_free_per_bank >= SCOPED_DFB_BYTES_PER_CORE

    normal_input = torch.randn(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)
    cases = [
        ("synthetic-normal", norms["synthetic"], synthetic_weight, normal_input),
        ("synthetic-zero", norms["synthetic"], synthetic_weight, torch.zeros_like(normal_input)),
        (
            "synthetic-constant-1.75",
            norms["synthetic"],
            synthetic_weight,
            torch.full_like(normal_input, 1.75),
        ),
        (
            "synthetic-tiny-1e-4",
            norms["synthetic"],
            synthetic_weight,
            torch.randn_like(normal_input) * 1e-4,
        ),
        *[(name, norms[name], checkpoint_weights[name], torch.randn_like(normal_input)) for name in REAL_WEIGHT_NAMES],
        ("synthetic-normal-return", norms["synthetic"], synthetic_weight, normal_input),
    ]

    mesh_device.enable_program_cache()
    cached_entries = None
    input_address_history = []
    address_guard = None
    for case_idx, (label, norm, weight, host_input) in enumerate(cases):
        input_address_history.append(_run_case(mesh_device, norm, weight, host_input, label=label))
        entries = mesh_device.num_program_cache_entries()
        if case_idx == 0:
            cached_entries = entries
            assert cached_entries > 0
            address_guard = ttnn.from_torch(
                torch.zeros_like(normal_input, dtype=torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
            )
        else:
            assert entries == cached_entries, f"{label}: cached program count changed {cached_entries} -> {entries}"

    assert any(
        current != input_address_history[0] for current in input_address_history[1:]
    ), "allocator did not exercise a changed input address"
    logger.info(
        f"RMSNorm cached-program reuse: entries={cached_entries} across {len(cases)} calls, "
        f"{len(set(input_address_history))} distinct all-chip input address tuples"
    )
    assert address_guard is not None
    address_guard.deallocate(True)

    # Reject host inputs before RMSNorm can combine them with device-resident gamma.
    host_input = torch.zeros(1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE, dtype=torch.bfloat16)
    with expect_error(ValueError, "device ttnn.Tensor"):
        norms["synthetic"](host_input)
    host_tensor = ttnn.from_torch(host_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(ValueError, "device ttnn.Tensor"):
        norms["synthetic"](host_tensor)

    with expect_error(ValueError, "one-dimensional"):
        RMSNorm(mesh_device, torch.ones(1, HIDDEN_SIZE))
    with expect_error(ValueError, "width 4096"):
        RMSNorm(mesh_device, torch.ones(HIDDEN_SIZE - 1))
    with expect_error(ValueError, "finite and positive"):
        RMSNorm(mesh_device, torch.ones(HIDDEN_SIZE), eps=0.0)
    with expect_error(ValueError, "finite and positive"):
        RMSNorm(mesh_device, torch.ones(HIDDEN_SIZE), eps=float("nan"))
