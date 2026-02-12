# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tests.pytest_utils import (
    build_expanded_test_ids,
    expand_test_cases_with_position_ids_ranges,
)
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_paged,
    torch_cache_from_transformers_single_layer,
)

PCC_REQUIRED = 0.99
PCC_REQUIRED_KVPE = 0.999
TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)


def get_cache_on_host(tt_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """
    Get the KVPE cache on the host from the TTNN cache.

    Args:
        tt_cache (ttnn.Tensor): The TTNN cache tensor.
        mesh_device (ttnn.MeshDevice): The mesh device to get the cache from.
        row_idx (int | None): The row index to get the cache from. If None, gets the entire cache.
    Returns:
        torch.Tensor: The cache tensor on the host.
    """
    return ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def generate_reference_io(
    model_path: Path,
    module_path: str | None,
    hf_config: PretrainedConfig,
    layer_idx: int,
    seq_len: int,
    batch_size: int,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    decode_position_id: int | None = None,
    use_real_weights: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate reference input/output for testing.

    Args:
        decode_position_id: Configuration for position_ids generation (only used in decode mode):
            - None: Generate random position_ids in range [0, max_seq_len - 1)
            - int: Use this specific position for all batches
    """
    reference_model = DeepseekV3Attention(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
    if use_real_weights:
        if module_path is None:
            state_dict = add_inv_scale_to_state_dict(
                reference_model.state_dict(),
                block_shape=hf_config.quantization_config["weight_block_size"],
            )
        else:
            state_dict = sub_state_dict(state_dict, module_path + ".")
            dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
            reference_model.load_state_dict(dequantized_state_dict)
    else:
        random_state_dict = {}
        for name, tensor in reference_model.state_dict().items():
            if torch.is_floating_point(tensor):
                # Keep weights small to reduce quantization error for random-weight tests.
                random_state_dict[name] = torch.randn_like(tensor) * 0.02
            else:
                random_state_dict[name] = torch.zeros_like(tensor)
        state_dict = add_inv_scale_to_state_dict(
            random_state_dict,
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
        # Ensure no NaNs/Infs to satisfy MLA2D weight replication checks
        float8_dtypes = tuple(
            dt
            for dt in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e5m2fnuz", None),
            )
            if dt is not None
        )
        for name, tensor in state_dict.items():
            if torch.is_floating_point(tensor):
                if float8_dtypes and tensor.dtype in float8_dtypes:
                    cleaned = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    state_dict[name] = cleaned.to(tensor.dtype)
                else:
                    state_dict[name] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
        reference_model.load_state_dict(dequantized_state_dict)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        # Handle decode_position_ids for decode mode
        if decode_position_id is None:
            # Generate random position_ids
            position_ids = position_ids_or_seq_lens = torch.randint(
                0, hf_config.max_seq_len - 1, (batch_size,), dtype=torch.long
            )
        else:
            # Must be an int, use that value for all batches
            if not isinstance(decode_position_id, int):
                raise ValueError(f"decode_position_id must be int or None, got {type(decode_position_id)}")
            if not (0 <= decode_position_id < hf_config.max_seq_len):
                raise ValueError(
                    f"decode_position_id must be in [0, {hf_config.max_seq_len - 1}], got {decode_position_id}"
                )
            position_ids = position_ids_or_seq_lens = torch.ones(batch_size, dtype=torch.long) * decode_position_id
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, zeroed_cache=True
    )
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED):
    passing, pcc = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(f"PCC: {pcc}")
    return passing


def check_cache_matches(tt_cache, reference_cache, lora_rank, pcc_required=PCC_REQUIRED_KVPE):
    # Check correct shapes
    dims_correct = True
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        dims_correct = False
    if len(reference_cache.shape) != 4:
        logger.error(
            f"Expected reference_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {reference_cache.shape=}"
        )
        dims_correct = False
    if not dims_correct:
        return False

    if tt_cache.shape != reference_cache.shape:
        logger.error(f"Cache shape mismatch: {tt_cache.shape=} vs {reference_cache.shape=}")
        return False

    if torch.all(tt_cache == 0) != torch.all(reference_cache == 0):
        if torch.all(tt_cache == 0):
            logger.error("TTNN cache is all zeros, but reference cache is not.")
        else:
            logger.error("Reference cache is all zeros, but TTNN cache is not.")
        return False

    # Check PCC
    tt_cache_kv = tt_cache[..., :lora_rank]
    tt_cache_pe = tt_cache[..., lora_rank:]

    ref_cache_kv = reference_cache[..., :lora_rank]  # [bsz, _, _, head_dim]
    ref_cache_pe = reference_cache[..., lora_rank:]  # [bsz, _, _, rope_head_dim]

    kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required)
    pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required)

    logger.info(f"Cache KV PCC: {kv_pcc_message}")
    logger.info(f"Cache PE PCC: {pe_pcc_message}")
    return kv_passing and pe_passing


def check_cache_unchanged(tt_cache, exclusion_area: tuple[slice, slice, slice, slice]):
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        return False

    tt_cache = tt_cache.clone()
    tt_cache[exclusion_area] = 0  # Zero out the area we don't want to check
    all_zeros = True
    for user_id in range(tt_cache.shape[0]):
        if not torch.all(tt_cache[user_id] == 0):
            logger.error(
                f"Cache for user {user_id % USERS_PER_ROW} row {user_id // USERS_PER_ROW} not empty outside exclusion area"
            )
            all_zeros = False

    return all_zeros


def run_test_forward_pass_mla1d(
    layer_idx,
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    state_dict,
    decode_position_ids: int | None = None,
    trace_mode: bool = False,
    use_real_weights: bool = True,
):
    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    # Get reference IO
    logger.info("Setting up reference IO")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
        decode_position_ids,
        use_real_weights,
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, (1, mesh_device.shape[1]), paged_config, user_id
    )

    # Set up model config
    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        MLA1D,
        hf_config_short,
        (state_dict,) * mesh_device.shape[0],
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(MLA1D, mode, hf_config_short, mesh_device)
    model_state = MLA1D.create_state(
        hf_config_short, paged_config, mesh_device, ccl, (paged_input_cache,) * mesh_device.shape[0]
    )
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_device.shape),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    tt_page_table = MLA1D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    tt_rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)

    # Forward pass
    logger.info("Running TTNN forward pass")

    cur_row_idx = 0 if trace_mode else torch.randint(0, mesh_device.shape[0], ()).item()

    def run_forward() -> ttnn.Tensor:
        if mode == "prefill":
            return MLA1D.forward_prefill(tt_input, user_id, cur_row_idx, run_config, tt_rope_tensors, tt_page_table)
        return MLA1D.forward_decode(
            tt_input, position_ids_tensor, cur_row_idx, run_config, tt_rope_tensors, tt_page_table
        )

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        tt_output_torch = ttnn.to_torch(
            tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
        )[cur_row_idx]

        tt_cache = torch_cache_from_paged(
            get_cache_on_host(run_config["kvpe_cache"], mesh_device), torch_page_table, mesh_device.get_num_devices()
        )
        if mode == "prefill":
            batch_id = user_id + cur_row_idx * USERS_PER_ROW
            assert (
                check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
                and check_cache_matches(
                    tt_cache[batch_id : batch_id + 1, :, :seq_len],
                    output_cache,
                    hf_config_short.kv_lora_rank,
                    pcc_required=PCC_REQUIRED_KVPE,
                )
                and check_cache_unchanged(
                    tt_cache, (slice(batch_id, batch_id + 1), slice(None), slice(None, seq_len), slice(None))
                )
            ), f"MLA output for prefill {seq_len=} {user_id=} {cur_row_idx=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"
        else:
            assert (
                check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
                and check_cache_matches(
                    tt_cache[torch.arange(batch_size) + cur_row_idx * USERS_PER_ROW, :, position_ids, :].unsqueeze(2),
                    output_cache[:, :, -1:, :],
                    hf_config_short.kv_lora_rank,
                    pcc_required=PCC_REQUIRED_KVPE,
                )
                and check_cache_unchanged(
                    tt_cache,
                    (
                        slice(cur_row_idx * USERS_PER_ROW, (cur_row_idx + 1) * USERS_PER_ROW),
                        slice(None),
                        slice(None),
                        slice(None),
                    ),
                )
            ), f"MLA output for decode {batch_size=} {position_ids=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"

    if trace_mode:
        tt_output = run_forward()
        ttnn.synchronize_device(mesh_device)
        check_outputs(tt_output)
        ttnn.deallocate(tt_output)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_forward()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        check_outputs(trace_output)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_forward()
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                check_outputs(tt_output)
            ttnn.deallocate(tt_output)


def run_test_forward_pass_mla2d(
    layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    state_dict,
    decode_position_ids: int | None = None,
    trace_mode: bool = False,
    use_real_weights: bool = True,
):
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    # Get reference IO
    logger.info("Setting up reference IO")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
        decode_position_ids,
        use_real_weights,
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW * mesh_device.shape[0], ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        MLA2D,
        hf_config_short,
        (state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(MLA2D, mode, hf_config_short, mesh_device)
    model_state = MLA2D.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_cache)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    tt_page_table = MLA2D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    tt_rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)

    # Forward pass
    logger.info("Running TTNN forward pass")

    def run_forward() -> ttnn.Tensor:
        if mode == "prefill":
            return MLA2D.forward_prefill(tt_input, user_id, run_config, tt_rope_tensors, tt_page_table)
        return MLA2D.forward_decode(tt_input, position_ids_tensor, run_config, tt_rope_tensors, tt_page_table)

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        tt_output_torch = ttnn.to_torch(
            tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
        ).reshape(
            -1, seq_len, hf_config_short.hidden_size
        )  # Concatenate all batches together

        tt_cache = torch_cache_from_paged(
            get_cache_on_host(run_config["mla1d"]["kvpe_cache"], mesh_device),
            torch_page_table,
            mesh_device.get_num_devices(),
        )
        if mode == "prefill":
            assert (
                check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
                and check_cache_matches(
                    tt_cache[user_id : user_id + 1, :, :seq_len],
                    output_cache,
                    hf_config_short.kv_lora_rank,
                    pcc_required=PCC_REQUIRED_KVPE,
                )
                and check_cache_unchanged(
                    tt_cache, (slice(user_id, user_id + 1), slice(None), slice(None, seq_len), slice(None))
                )
            ), f"MLA output for prefill {seq_len=} {user_id=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"
        else:
            assert check_output_matches(
                tt_output_torch, reference_output, pcc_required=PCC_REQUIRED
            ) and check_cache_matches(
                tt_cache[torch.arange(batch_size), :, position_ids, :].unsqueeze(2),
                output_cache[:, :, -1:, :],
                hf_config_short.kv_lora_rank,
                pcc_required=PCC_REQUIRED_KVPE,
            ), f"MLA output for decode {batch_size=} {position_ids=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"

    if trace_mode:
        tt_output = run_forward()
        ttnn.synchronize_device(mesh_device)
        check_outputs(tt_output)
        ttnn.deallocate(tt_output)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_forward()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        check_outputs(trace_output)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_forward()
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                check_outputs(tt_output)
            ttnn.deallocate(tt_output)


# Base test cases - ranges will be expanded into individual test cases
# see documentation for expand_test_cases_with_position_ids_ranges for more details
BASE_TEST_CASES = [
    # mode, seq_len, batch_size_per_row, decode_position_ids
    ("decode", 1, USERS_PER_ROW, None),
    # ("decode", 1, USERS_PER_ROW, (4096, 8192, 32)), # Example.
] + [
    ("prefill", seq_len, 1, None)
    if seq_len == 128
    else pytest.param(
        "prefill",
        seq_len,
        1,
        None,
        marks=pytest.mark.skipif(
            CI_ACTIVE,
            reason=(
                f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
            ),
        ),
    )
    for seq_len in PREFILL_SEQ_LENS
]  # decode_position_ids is not applicable for prefill

# Expand ranges into individual position_ids for pytest
EXPANDED_TEST_CASES = expand_test_cases_with_position_ids_ranges(BASE_TEST_CASES)
EXPANDED_TEST_IDS = build_expanded_test_ids(EXPANDED_TEST_CASES)


@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    EXPANDED_TEST_CASES,
    ids=EXPANDED_TEST_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 10485760,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, marks=_CI_SKIP_MARK, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
@pytest.mark.parametrize(
    "module_path",
    [None, "model.layers.0.self_attn"],
)
@pytest.mark.parametrize(
    "test_closure",
    [
        pytest.param(run_test_forward_pass_mla2d, marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"])),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    trace_mode,
    use_real_weights,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    # Hardcoded arguments; can later change them to test arguments if needed
    layer_idx = 0

    # Only use decode_position_ids for decode mode
    if mode != "decode":
        decode_position_ids = None
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")

    test_closure(
        layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        ccl,
        model_path,
        module_path,
        force_recalculate_weight_config,
        state_dict,
        decode_position_ids,
        trace_mode,
        use_real_weights,
    )


if __name__ == "__main__":
    pytest.main([__file__])
