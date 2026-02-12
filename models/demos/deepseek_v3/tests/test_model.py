# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN, build_test_cases_and_ids
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_caches_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers,
)


def generate_reference_io(
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    batch_size: int,
    hf_config: PretrainedConfig,
    model_path: str,
    state_dict: dict[str, torch.Tensor],
    decode_position_id: int | None = None,
):
    """Generate reference input and output for the given mode using either real or random weights."""
    # This needs to be disabled as deterministic way to quantize weights is not supported
    torch.use_deterministic_algorithms(False)

    if use_real_weights:
        torch.use_deterministic_algorithms(False)

        state_dict = sub_state_dict(state_dict, "", hf_config.num_hidden_layers)

        logger.info(f"Creating reference model")
        # Create model on meta device (no weight initialization or memory allocation)
        with torch.device("meta"):
            reference_model = DeepseekV3ForCausalLM(hf_config).eval()

        # Move to target device without allocating memory for parameters
        reference_model = reference_model.to_empty(device=torch.device("cpu"))

        logger.info(f"Loading state dict into reference model")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
        reference_model = reference_model.to(torch.bfloat16)
    else:
        logger.info("Creating reference model with random weights")
        reference_model = DeepseekV3ForCausalLM(hf_config).eval().to(torch.bfloat16)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.to(torch.bfloat16).state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )

    torch_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
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

    def extract_output_and_cache(model_output):
        if isinstance(model_output, tuple):
            # HF outputs can be tuples with cache at different indices.
            cache_idx = 2 if len(model_output) == 3 else 1
            return model_output[0], model_output[cache_idx]
        if hasattr(model_output, "logits"):
            return model_output.logits, model_output.past_key_values
        if hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state, model_output.past_key_values
        raise AttributeError(f"Model output has neither 'last_hidden_state' nor 'logits': {type(model_output)}")

    if mode == "decode" and torch.any(position_ids_or_seq_lens != 0).item():
        # this is the non-zero position_ids case and we are prefilling the cache for the decode step
        # this reference cache is used by TT model also for the non-zero position_ids case
        logger.info("Running reference prefill for decode cache for non-zero position_ids case")
        prefill_len = int(position_ids_or_seq_lens.max().item())
        assert torch.all(
            position_ids_or_seq_lens <= prefill_len
        ).item(), "position_ids must not exceed prefill_len used to build the attention mask"
        prefill_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, prefill_len), dtype=torch.long)
        prefill_seq_lens = torch.full((batch_size,), prefill_len, dtype=torch.long)

        _, _, prefill_cache = run_reference_with_attention(
            reference_model, prefill_input, prefill_seq_lens, None, hf_config, "prefill", False
        )

        torch_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, 1), dtype=torch.long)
        position_ids = position_ids_or_seq_lens
        position_ids_2d = position_ids.unsqueeze(1)
        mask = torch.full((batch_size, 1, 1, prefill_len + 1), float("-inf"), dtype=torch.bfloat16)
        # Vectorized construction of the attention mask using broadcasting.
        seq_positions = torch.arange(prefill_len + 1, device=position_ids.device)
        valid_positions = seq_positions.unsqueeze(0) < position_ids.unsqueeze(1)
        mask = mask.masked_fill(valid_positions.reshape(batch_size, 1, 1, prefill_len + 1), 0.0)
        mask[:, :, :, -1] = 0.0

        with torch.no_grad():
            model_output_raw = reference_model(
                torch_input,
                attention_mask=mask,
                position_ids=position_ids_2d,
                output_attentions=False,
                use_cache=True,
                past_key_values=prefill_cache,
            )
        reference_output, output_cache = extract_output_and_cache(model_output_raw)
        input_cache = prefill_cache
    else:
        reference_output, input_cache, output_cache = run_reference_with_attention(
            reference_model, torch_input, position_ids_or_seq_lens, None, hf_config, mode, False
        )
    logger.info(f"Reference model output shape: {reference_output.shape}")
    input_cache = torch_cache_from_transformers(input_cache)
    output_cache = torch_cache_from_transformers(output_cache)

    if mode == "decode":
        torch_input = torch_input.transpose(1, 0)  # [seq_len, batch_size]
        reference_output = reference_output.transpose(1, 0)  # [seq_len, batch_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def run_test_forward_pass_dpmodel(
    use_real_weights,
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
    decode_position_ids: int | None = None,
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
        use_real_weights, mode, seq_len, batch_size, hf_config_short, model_path, state_dict, decode_position_ids
    )

    # Set up page config
    logger.info("Setting up model configs")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, dp_factor)
    paged_input_caches, torch_page_tables = paged_caches_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        RowBatchedModel, hf_config_short, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(RowBatchedModel, mode, hf_config_short, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
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

    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
        for torch_page_table in torch_page_tables
    )
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables
        )

    ttnn.synchronize_device(mesh_device)

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )
    assert (
        tt_output_torch.shape[-1] == hf_config_short.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config_short.vocab_size}"

    # Check output PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.97)


TEST_CASES, TEST_IDS = build_test_cases_and_ids(
    USERS_PER_ROW,
    DEFAULT_PREFILL_SEQ_LEN,  # default prefill sequence length to test
)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_real_weights",
    [True],  # Test only with real weights for now
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_forward_pass(
    use_real_weights,
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
    set_deterministic_env,
    state_dict,
):
    # Set fewer number oflayers to speed test and avoid OOM
    hf_config_short.num_hidden_layers = 5

    # Only use decode_position_ids for decode mode
    if mode != "decode":
        decode_position_ids = None

    run_test_forward_pass_dpmodel(
        use_real_weights,
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
        decode_position_ids,
    )


if __name__ == "__main__":
    pytest.main([__file__])
