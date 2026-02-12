# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tests.pytest_utils import (
    build_expanded_test_ids,
    expand_test_cases_with_position_ids_ranges,
)
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers_single_layer,
)

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)
pytestmark = pytest.mark.timeout(1200)


def generate_reference_io(
    use_real_weights: bool,
    model_path: Path,
    module_path: str | None,
    hf_config: PretrainedConfig,
    layer_idx: int,
    seq_len: int,
    batch_size: int,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    decode_position_id: int | None = None,
):
    reference_model = DeepseekV3DecoderLayer(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
    if use_real_weights and module_path is not None:
        state_dict = sub_state_dict(state_dict, module_path + ".")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
    else:
        # This needs to be disabled as deterministic way to quantize weights is not supported
        torch.use_deterministic_algorithms(False)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        if decode_position_id is None:
            position_ids = position_ids_or_seq_lens = torch.randint(
                0, hf_config.max_seq_len - 1, (batch_size,), dtype=torch.long
            )
        else:
            if not isinstance(decode_position_id, int):
                raise ValueError(f"decode_position_id must be int or None, got {type(decode_position_id)}")
            if not (0 <= decode_position_id < hf_config.max_seq_len):
                raise ValueError(
                    f"decode_position_id must be in [0, {hf_config.max_seq_len - 1}], got {decode_position_id}"
                )
            position_ids = position_ids_or_seq_lens = torch.ones(batch_size, dtype=torch.long) * decode_position_id
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, False
    )
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def run_test_forward_pass_decoder1d(
    DecoderBlockClass: type[DecoderBlockBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size,
    trace_mode,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    state_dict,
    use_real_weights,
    decode_position_ids: int | None = None,
):
    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    state_dict, position_ids, torch_input, reference_output, input_cache, _ = generate_reference_io(
        use_real_weights,
        model_path,
        module_path,
        hf_config_short,
        reference_layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
        decode_position_ids,
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
        DecoderBlockClass,
        hf_config_short,
        (state_dict,) * mesh_device.shape[0],
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device)
    model_state = DecoderBlockClass.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        is_padding_layer=(False,) * mesh_device.shape[0],
        mla_caches=(paged_input_cache,) * mesh_device.shape[0],
    )
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
    rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")
    cur_row_idx = torch.randint(0, mesh_device.shape[0], ()).item()

    def run_forward() -> ttnn.Tensor:
        if mode == "prefill":
            return DecoderBlockClass.forward_prefill(
                tt_input, user_id, cur_row_idx, run_config, rope_tensors, tt_page_table
            )
        return DecoderBlockClass.forward_decode(
            tt_input, position_ids_tensor, cur_row_idx, run_config, rope_tensors, tt_page_table
        )

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        )[cur_row_idx]
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.9899)

    if trace_mode:
        # Iteration 0: eager compile run (not traced)
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

    ttnn.deallocate(tt_input)
    if position_ids_tensor is not None:
        ttnn.deallocate(position_ids_tensor)
    ttnn.deallocate(tt_page_table)
    rope_values = rope_tensors.values() if isinstance(rope_tensors, dict) else rope_tensors
    for rope_tensor in rope_values:
        if isinstance(rope_tensor, ttnn.Tensor):
            ttnn.deallocate(rope_tensor)


def run_test_forward_pass_decoder2d(
    DecoderBlockClass: type[DecoderBlock2DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    trace_mode,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    state_dict,
    use_real_weights,
    decode_position_ids: int | None = None,
):
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    state_dict, position_ids, torch_input, reference_output, input_cache, _ = generate_reference_io(
        use_real_weights,
        model_path,
        module_path,
        hf_config_short,
        reference_layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
        decode_position_ids,
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW * mesh_device.shape[0], ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        DecoderBlockClass,
        hf_config_short,
        (state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device)
    model_state = DecoderBlockClass.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        mla_cache=paged_input_cache,
    )
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")

    def run_forward() -> ttnn.Tensor:
        if mode == "prefill":
            return DecoderBlockClass.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_table)
        return DecoderBlockClass.forward_decode(tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_table)

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        )
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.9899)

    if trace_mode:
        # Iteration 0: eager compile run (not traced)
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

    ttnn.deallocate(tt_input)
    if position_ids_tensor is not None:
        ttnn.deallocate(position_ids_tensor)
    ttnn.deallocate(tt_page_table)
    rope_values = rope_tensors.values() if isinstance(rope_tensors, dict) else rope_tensors
    for rope_tensor in rope_values:
        if isinstance(rope_tensor, ttnn.Tensor):
            ttnn.deallocate(rope_tensor)


# Base test cases - ranges will be expanded into individual test cases
# see documentation for expand_test_cases_with_position_ids_ranges for more details
BASE_TEST_CASES = [
    # mode, seq_len, batch_size_per_row, decode_position_ids
    ("decode", 1, USERS_PER_ROW, None),
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
]
EXPANDED_TEST_CASES = expand_test_cases_with_position_ids_ranges(BASE_TEST_CASES)
EXPANDED_TEST_IDS = build_expanded_test_ids(EXPANDED_TEST_CASES)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
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
    "DecoderBlockClass, module_path, reference_layer_idx, test_closure",
    [
        pytest.param(
            DecoderBlock2D,
            None,
            0,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            MoEDecoderBlock2D,
            None,
            3,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            DecoderBlock2D,
            "model.layers.0",
            0,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            MoEDecoderBlock2D,
            "model.layers.3",
            3,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
    ],
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    EXPANDED_TEST_CASES,
    ids=EXPANDED_TEST_IDS,
)
def test_forward_pass(
    DecoderBlockClass: type[DecoderBlock2DBase],
    trace_mode,
    use_real_weights,
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    if getattr(hf_config_short, "_attn_implementation", None) is None:
        hf_config_short._attn_implementation = "eager"
    if CI_ACTIVE and not (DecoderBlockClass in (DecoderBlock2D, MoEDecoderBlock2D) and module_path is not None):
        pytest.skip("CI runs only DecoderBlock2D/MoEDecoderBlock2D with module_path set.")
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")
    if use_real_weights and module_path is None:
        pytest.skip("Real weights require module_path to be set.")

    if mode != "decode":
        decode_position_ids = None

    test_closure(
        DecoderBlockClass,
        module_path,
        reference_layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        trace_mode,
        hf_config_short,
        cache_path,
        mesh_device,
        model_path,
        ccl,
        force_recalculate_weight_config,
        state_dict,
        use_real_weights,
        decode_position_ids,
    )


if __name__ == "__main__":
    pytest.main([__file__])
