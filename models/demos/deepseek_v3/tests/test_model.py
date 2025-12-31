# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
import time

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
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
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        # position_ids = torch.randint(0, hf_config.max_seq_len - 1, (batch_size,))
        position_ids = position_ids_or_seq_lens = torch.zeros(
            (batch_size,), dtype=torch.long
        )  # TODO: investigate the PCC issue with real weights

    logger.info(
        f"Running reference model with torch_input shape: {torch_input.shape} and position_ids shape: {position_ids_or_seq_lens.shape}"
    )
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
):
    debug_timings = os.getenv("DEEPSEEK_V3_TEST_DEBUG_TIMINGS", "0") == "1"
    debug_sync = os.getenv("DEEPSEEK_V3_TEST_DEBUG_SYNC", "0") == "1"

    def _log_step(msg: str) -> float:
        logger.info(msg)
        return time.perf_counter()

    def _log_done(msg: str, start_t: float | None = None) -> None:
        if start_t is None or not debug_timings:
            logger.info(msg)
        else:
            logger.info(f"{msg} (dt={time.perf_counter() - start_t:.3f}s)")

    t_total = _log_step(
        f"[test_model] BEGIN run_test_forward_pass_dpmodel(mode={mode}, seq_len={seq_len}, batch_size_per_row={batch_size_per_row}, use_real_weights={use_real_weights})"
    )

    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    # Get reference IO
    t0 = _log_step("[test_model] Setting up reference IO (generate_reference_io)")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        use_real_weights, mode, seq_len, batch_size, hf_config_short, model_path, state_dict
    )
    _log_done(
        f"[test_model] Reference IO ready: torch_input={tuple(torch_input.shape)}, reference_output={tuple(reference_output.shape)}",
        t0,
    )

    # Set up page config
    t0 = _log_step("[test_model] Setting up model configs (paged caches + weight/model configs + run_config)")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, dp_factor)
    paged_input_caches, torch_page_tables = paged_caches_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )
    _log_done("[test_model] Paged caches + page tables prepared", t0)

    # Set up model config
    t1 = _log_step("[test_model] Building weight_config + model_config + states")
    weight_config = get_test_weight_config(
        RowBatchedModel, hf_config_short, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(RowBatchedModel, mode, hf_config_short, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)
    _log_done("[test_model] run_config ready", t1)

    # Set up ttnn inputs
    t0 = _log_step("[test_model] Setting up model inputs (ttnn.from_torch + rope + page tables)")

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

    t1 = _log_step("[test_model] Creating TT page tables")
    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
        for torch_page_table in torch_page_tables
    )
    _log_done(f"[test_model] TT page tables ready (count={len(tt_page_tables)})", t1)

    t1 = _log_step("[test_model] Creating RoPE tensors")
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    _log_done("[test_model] RoPE tensors ready", t1)

    _log_done("[test_model] Model inputs ready", t0)

    # Forward pass
    t0 = _log_step("[test_model] Running TTNN forward pass (RowBatchedModel.forward_*)")
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables
        )
    _log_done(
        f"[test_model] TTNN forward returned: tt_output shape={tt_output.shape}, dtype={tt_output.dtype}",
        t0,
    )

    if debug_sync:
        t1 = _log_step("[test_model] DEBUG_SYNC=1 -> ttnn.synchronize_device(mesh_device) after forward")
        ttnn.synchronize_device(mesh_device)
        _log_done("[test_model] Device synchronized after forward", t1)

    t0 = _log_step("[test_model] Converting tt_output -> torch (ttnn.to_torch)")
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )
    _log_done(
        f"[test_model] ttnn.to_torch done: tt_output_torch={tuple(tt_output_torch.shape)}, dtype={tt_output_torch.dtype}",
        t0,
    )

    debug_pcc_breakdown = 1
    if debug_pcc_breakdown:
        # This is intentionally lightweight: compute PCC on small slices to identify if a single mesh
        # row/col shard is corrupt (e.g., bad row) vs the whole tensor.
        tt_full = tt_output_torch.squeeze()
        ref_full = reference_output.squeeze()
        logger.info(
            f"[test_model] DEBUG_PCC_BREAKDOWN=1: tt_full.shape={tuple(tt_full.shape)}, ref_full.shape={tuple(ref_full.shape)}"
        )

        if tt_full.ndim == 2 and ref_full.ndim == 2 and tt_full.shape == ref_full.shape:
            mesh_rows, mesh_cols = mesh_device.shape
            seq_len_full, vocab_full = tt_full.shape

            seq_sample = int(os.getenv("DEEPSEEK_V3_TEST_DEBUG_SEQ_SAMPLE", "256"))
            vocab_sample = int(os.getenv("DEEPSEEK_V3_TEST_DEBUG_VOCAB_SAMPLE", "2048"))
            seq_sample = max(1, min(seq_sample, seq_len_full))
            vocab_sample = max(1, min(vocab_sample, vocab_full))

            # Per-row (sequence-parallel) breakdown: compare each row chunk over a small vocab prefix.
            if seq_len_full % mesh_rows == 0:
                seq_chunk = seq_len_full // mesh_rows
                for r in range(mesh_rows):
                    s0, s1 = r * seq_chunk, (r + 1) * seq_chunk
                    tt_slice = tt_full[s0:s1, :vocab_sample]
                    ref_slice = ref_full[s0:s1, :vocab_sample]
                    _, pcc_val = comp_pcc(ref_slice, tt_slice, pcc=0.97)
                    tt_max_abs = tt_slice.abs().max().item()
                    tt_finite = torch.isfinite(tt_slice).float().mean().item()
                    logger.info(
                        f"[test_model] DEBUG_PCC_BREAKDOWN seq_shard[{r}] seq={s0}:{s1} vocab=0:{vocab_sample} "
                        f"pcc={pcc_val:.6f} finite={tt_finite:.6f} max_abs={tt_max_abs:.3e}"
                    )
            else:
                logger.warning(
                    f"[test_model] DEBUG_PCC_BREAKDOWN: seq_len_full={seq_len_full} not divisible by mesh_rows={mesh_rows}; skipping seq shard PCCs"
                )

            # Per-col (vocab-sharded) breakdown: compare each vocab shard over a small seq prefix.
            if vocab_full % mesh_cols == 0:
                vocab_chunk = vocab_full // mesh_cols
                for c in range(mesh_cols):
                    v0, v1 = c * vocab_chunk, (c + 1) * vocab_chunk
                    tt_slice = tt_full[:seq_sample, v0:v1]
                    ref_slice = ref_full[:seq_sample, v0:v1]
                    _, pcc_val = comp_pcc(ref_slice, tt_slice, pcc=0.97)
                    tt_max_abs = tt_slice.abs().max().item()
                    tt_finite = torch.isfinite(tt_slice).float().mean().item()
                    logger.info(
                        f"[test_model] DEBUG_PCC_BREAKDOWN vocab_shard[{c}] seq=0:{seq_sample} vocab={v0}:{v1} "
                        f"pcc={pcc_val:.6f} finite={tt_finite:.6f} max_abs={tt_max_abs:.3e}"
                    )
            else:
                logger.warning(
                    f"[test_model] DEBUG_PCC_BREAKDOWN: vocab_full={vocab_full} not divisible by mesh_cols={mesh_cols}; skipping vocab shard PCCs"
                )
        else:
            logger.warning("[test_model] DEBUG_PCC_BREAKDOWN: unexpected tensor ranks/shapes; skipping")

    assert (
        tt_output_torch.shape[-1] == hf_config_short.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config_short.vocab_size}"

    # Check output PCC
    t0 = _log_step("[test_model] Running PCC check (assert_hidden_dim_pcc)")
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    _log_done("[test_model] PCC check passed", t0)
    _log_done("[test_model] END run_test_forward_pass_dpmodel", t_total)


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
    "mode, seq_len, batch_size_per_row",
    [
        # ("decode", 1, 32),
        # ("prefill", 128, 1),
        # test all prefill from 512 to 128k
        # ("prefill", 512, 1),
        # ("prefill", 1024, 1),
        # ("prefill", 2048, 1),
        # ("prefill", 4096, 1),
        ("prefill", 8192, 1),
        # ("prefill", 16384, 1),
        # ("prefill", 32768, 1),
        # ("prefill", 65536, 1),
        # ("prefill", 131072, 1),
    ],
)
def test_forward_pass(
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
    set_deterministic_env,
    state_dict,
):
    # Set less layers and shorter max length for the sake of testing
    hf_config_short.num_hidden_layers = 8

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
    )


if __name__ == "__main__":
    pytest.main([__file__])
