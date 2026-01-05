# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_1d import DecoderBlock1D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_1d_base import DecoderBlock1DBase
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_1d import MoEDecoderBlock1D
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


def generate_reference_io(
    model_path: Path,
    module_path: str | None,
    hf_config: PretrainedConfig,
    layer_idx: int,
    seq_len: int,
    batch_size: int,
    mode: str,
    state_dict: dict[str, torch.Tensor],
):
    logger.info(
        f"[DEBUG] generate_reference_io: layer_idx={layer_idx}, seq_len={seq_len}, batch_size={batch_size}, mode={mode}"
    )
    reference_model = DeepseekV3DecoderLayer(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
    logger.info(f"[DEBUG] Reference model created")
    if module_path is not None:
        logger.info(f"[DEBUG] Loading state dict from module_path={module_path}")
        state_dict = sub_state_dict(state_dict, module_path + ".")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
        logger.info(f"[DEBUG] State dict loaded")
    else:
        # This needs to be disabled as deterministic way to quantize weights is not supported
        logger.info(f"[DEBUG] Creating state dict with inv_scale")
        torch.use_deterministic_algorithms(False)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
        logger.info(f"[DEBUG] State dict with inv_scale created")

    logger.info(f"[DEBUG] Creating random input tensor...")
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
        logger.info(f"[DEBUG] Prefill mode: position_ids_or_seq_lens={position_ids_or_seq_lens}")
    else:
        position_ids = position_ids_or_seq_lens = torch.randint(0, hf_config.max_seq_len - 1, (batch_size,))
        logger.info(f"[DEBUG] Decode mode: position_ids={position_ids}")
    logger.info(f"[DEBUG] Running reference with attention...")
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, False
    )
    logger.info(f"[DEBUG] Reference output shape: {reference_output.shape}")
    logger.info(f"[DEBUG] Converting caches...")
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    logger.info(f"[DEBUG] Caches converted")
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        logger.info(f"[DEBUG] Decode mode: permuted tensors")
    logger.info(f"[DEBUG] generate_reference_io completed")
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def run_test_forward_pass_decoder1d(
    DecoderBlockClass: type[DecoderBlock1DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    state_dict,
):
    logger.info(
        f"[DEBUG] Starting run_test_forward_pass_decoder1d: mode={mode}, seq_len={seq_len}, batch_size={batch_size}, DecoderBlockClass={DecoderBlockClass.__name__}"
    )
    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    logger.info(f"[DEBUG] Generating reference IO...")
    state_dict, position_ids, torch_input, reference_output, input_cache, _ = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        reference_layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
    )
    logger.info(
        f"[DEBUG] Reference IO generated successfully. Input shape: {torch_input.shape}, Output shape: {reference_output.shape}"
    )

    # Set up page config
    logger.info("Setting up model configs")
    logger.info(f"[DEBUG] Setting up page config for mode={mode}")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    logger.info(f"[DEBUG] user_id={user_id}, mesh_device.shape={mesh_device.shape}")
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    logger.info(f"[DEBUG] paged_config created: {paged_config}")
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, (1, mesh_device.shape[1]), paged_config, user_id
    )
    logger.info(f"[DEBUG] Paged cache created successfully")

    # Set up model config
    logger.info(f"[DEBUG] Getting weight config...")
    weight_config = get_test_weight_config(
        DecoderBlockClass,
        hf_config_short,
        (state_dict,) * mesh_device.shape[0],
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    logger.info(f"[DEBUG] Weight config obtained")
    logger.info(f"[DEBUG] Getting model config...")
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device)
    logger.info(f"[DEBUG] Model config obtained")
    logger.info(f"[DEBUG] Creating model state...")
    model_state = DecoderBlockClass.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        is_padding_layer=(False,) * mesh_device.shape[0],
        mla_caches=(paged_input_cache,) * mesh_device.shape[0],
    )
    logger.info(f"[DEBUG] Model state created")
    logger.info(f"[DEBUG] Creating shared state...")
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device)
    logger.info(f"[DEBUG] Shared state created")
    logger.info(f"[DEBUG] Creating run config...")
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)
    logger.info(f"[DEBUG] Run config created")

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    logger.info(f"[DEBUG] Converting torch input to ttnn tensor...")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"[DEBUG] tt_input created successfully")

    logger.info(f"[DEBUG] Creating position_ids_tensor (mode={mode})...")
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
    logger.info(f"[DEBUG] position_ids_tensor created: {position_ids_tensor is not None}")

    logger.info(f"[DEBUG] Creating page table...")
    tt_page_table = MLA1D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    logger.info(f"[DEBUG] Page table created")
    logger.info(f"[DEBUG] Getting rope tensors...")
    rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)
    logger.info(f"[DEBUG] Rope tensors obtained")
    logger.info(f"[DEBUG] Recomputing paged_config...")
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    logger.info(f"[DEBUG] Paged config recomputed")

    # Forward pass
    logger.info("Running TTNN forward pass")
    logger.info(f"[DEBUG] About to start forward pass: mode={mode}")

    cur_row_idx = torch.randint(0, mesh_device.shape[0], ()).item()
    logger.info(f"[DEBUG] Selected cur_row_idx={cur_row_idx}")

    if mode == "prefill":
        logger.info(f"[DEBUG] CALLING forward_prefill with user_id={user_id}, cur_row_idx={cur_row_idx}")
        logger.info(
            f"[DEBUG] Input shapes - tt_input: {tt_input}, rope_tensors: {rope_tensors}, tt_page_table: {tt_page_table}"
        )
        try:
            tt_output = DecoderBlockClass.forward_prefill(
                tt_input, user_id, cur_row_idx, run_config, rope_tensors, tt_page_table
            )
            logger.info(f"[DEBUG] forward_prefill COMPLETED successfully")
        except Exception as e:
            logger.error(f"[DEBUG] forward_prefill FAILED with exception: {type(e).__name__}: {e}")
            raise
    else:
        logger.info(f"[DEBUG] CALLING forward_decode with cur_row_idx={cur_row_idx}")
        try:
            tt_output = DecoderBlockClass.forward_decode(
                tt_input, position_ids_tensor, cur_row_idx, run_config, rope_tensors, tt_page_table
            )
            logger.info(f"[DEBUG] forward_decode COMPLETED successfully")
        except Exception as e:
            logger.error(f"[DEBUG] forward_decode FAILED with exception: {type(e).__name__}: {e}")
            raise

    logger.info(f"[DEBUG] Converting output to torch...")
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    )[cur_row_idx]
    logger.info(f"[DEBUG] Output converted. Shape: {tt_output_torch.shape}")

    # Check output PCC
    logger.info(f"[DEBUG] Checking output PCC...")
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.9899)
    logger.info(f"[DEBUG] PCC check passed!")


def run_test_forward_pass_decoder2d(
    DecoderBlockClass: type[DecoderBlock2DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    state_dict,
):
    logger.info(
        f"[DEBUG] Starting run_test_forward_pass_decoder2d: mode={mode}, seq_len={seq_len}, batch_size_per_row={batch_size_per_row}, DecoderBlockClass={DecoderBlockClass.__name__}"
    )
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    logger.info(f"[DEBUG] Generating reference IO...")
    state_dict, position_ids, torch_input, reference_output, input_cache, _ = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        reference_layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
    )
    logger.info(
        f"[DEBUG] Reference IO generated successfully. Input shape: {torch_input.shape}, Output shape: {reference_output.shape}"
    )

    # Set up page config
    logger.info("Setting up model configs")
    logger.info(f"[DEBUG] Setting up page config for mode={mode}")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW * mesh_device.shape[0], ()).item()
    logger.info(f"[DEBUG] user_id={user_id}, mesh_device.shape={mesh_device.shape}")
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    logger.info(f"[DEBUG] paged_config created: {paged_config}")
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )
    logger.info(f"[DEBUG] Paged cache created successfully")

    # Set up model config
    logger.info(f"[DEBUG] Getting weight config...")
    weight_config = get_test_weight_config(
        DecoderBlockClass,
        hf_config_short,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    logger.info(f"[DEBUG] Weight config obtained")
    logger.info(f"[DEBUG] Getting model config...")
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device)
    logger.info(f"[DEBUG] Model config obtained")
    logger.info(f"[DEBUG] Creating model state...")
    model_state = DecoderBlockClass.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        mla_cache=paged_input_cache,
    )
    logger.info(f"[DEBUG] Model state created")
    logger.info(f"[DEBUG] Creating shared state...")
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device)
    logger.info(f"[DEBUG] Shared state created")
    logger.info(f"[DEBUG] Creating run config...")
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)
    logger.info(f"[DEBUG] Run config created")

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    logger.info(f"[DEBUG] Converting torch input to ttnn tensor...")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"[DEBUG] tt_input created successfully")

    logger.info(f"[DEBUG] Creating position_ids_tensor (mode={mode})...")
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
    logger.info(f"[DEBUG] position_ids_tensor created: {position_ids_tensor is not None}")

    logger.info(f"[DEBUG] Creating page table...")
    tt_page_table = MLA2D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    logger.info(f"[DEBUG] Page table created")
    logger.info(f"[DEBUG] Getting rope tensors...")
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    logger.info(f"[DEBUG] Rope tensors obtained")
    logger.info(f"[DEBUG] Recomputing paged_config...")
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    logger.info(f"[DEBUG] Paged config recomputed")

    # Forward pass
    logger.info("Running TTNN forward pass")
    logger.info(f"[DEBUG] About to start forward pass: mode={mode}")

    if mode == "prefill":
        logger.info(f"[DEBUG] CALLING forward_prefill with user_id={user_id}")
        logger.info(
            f"[DEBUG] Input shapes - tt_input: {tt_input}, rope_tensors: {rope_tensors}, tt_page_table: {tt_page_table}"
        )
        try:
            tt_output = DecoderBlockClass.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_table)
            logger.info(f"[DEBUG] forward_prefill COMPLETED successfully")
        except Exception as e:
            logger.error(f"[DEBUG] forward_prefill FAILED with exception: {type(e).__name__}: {e}")
            raise
    else:
        logger.info(f"[DEBUG] CALLING forward_decode")
        try:
            tt_output = DecoderBlockClass.forward_decode(
                tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_table
            )
            logger.info(f"[DEBUG] forward_decode COMPLETED successfully")
        except Exception as e:
            logger.error(f"[DEBUG] forward_decode FAILED with exception: {type(e).__name__}: {e}")
            raise

    logger.info(f"[DEBUG] Converting output to torch...")
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )
    logger.info(f"[DEBUG] Output converted. Shape: {tt_output_torch.shape}")

    # Check output PCC
    logger.info(f"[DEBUG] Checking output PCC...")
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.9899)
    logger.info(f"[DEBUG] PCC check passed!")


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "DecoderBlockClass, module_path, reference_layer_idx, test_closure",
    [
        (DecoderBlock1D, None, 0, run_test_forward_pass_decoder1d),
        (MoEDecoderBlock1D, None, 3, run_test_forward_pass_decoder1d),
        (DecoderBlock1D, "model.layers.0", 0, run_test_forward_pass_decoder1d),
        (MoEDecoderBlock1D, "model.layers.3", 3, run_test_forward_pass_decoder1d),
        (DecoderBlock2D, None, 0, run_test_forward_pass_decoder2d),
        (MoEDecoderBlock2D, None, 3, run_test_forward_pass_decoder2d),
        (DecoderBlock2D, "model.layers.0", 0, run_test_forward_pass_decoder2d),
        (MoEDecoderBlock2D, "model.layers.3", 3, run_test_forward_pass_decoder2d),
    ],
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row",
    [
        # ("decode", 1, 32),
        ("prefill", 8192, 1),
        # ("prefill", 2048, 1),  # Test chunking # TODO: Uncomment once MLA prefill works
    ],
)
def test_forward_pass(
    DecoderBlockClass: type[DecoderBlock1DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
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
    logger.info(f"[DEBUG] ========== test_forward_pass START ==========")
    logger.info(
        f"[DEBUG] DecoderBlockClass={DecoderBlockClass.__name__}, module_path={module_path}, reference_layer_idx={reference_layer_idx}"
    )
    logger.info(f"[DEBUG] mode={mode}, seq_len={seq_len}, batch_size_per_row={batch_size_per_row}")
    try:
        test_closure(
            DecoderBlockClass,
            module_path,
            reference_layer_idx,
            mode,
            seq_len,
            batch_size_per_row,
            hf_config_short,
            cache_path,
            mesh_device,
            model_path,
            ccl,
            force_recalculate_weight_config,
            state_dict,
        )
        logger.info(f"[DEBUG] ========== test_forward_pass COMPLETED SUCCESSFULLY ==========")
    except Exception as e:
        logger.error(f"[DEBUG] ========== test_forward_pass FAILED ==========")
        logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
        logger.error(f"[DEBUG] Exception message: {e}")
        import traceback

        logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    pytest.main([__file__])
