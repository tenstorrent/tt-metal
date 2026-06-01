# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import os
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.reference.qwen import Attention
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL

PAGED_ATTENTION_DEBUG_FLAG = os.getenv("QWEN_DEBUG_PAGED_ATTENTION", "0") == "1"
PAGED_ATTENTION_CASES = (PAGED_ATTENTION_DEBUG_FLAG,)
PAGED_ATTENTION_IDS = ("paged_attention" if PAGED_ATTENTION_DEBUG_FLAG else "default_attention",)


def _decode_input_memcfg(model_args, force_replicated_input):
    """Match activation sharding to QKV matmul mode (dram=column, ring=prefetcher NOC1 grid)."""
    if force_replicated_input:
        return ttnn.DRAM_MEMORY_CONFIG
    qkv_mm = os.getenv("QWEN_DEVICE_QKV_MATMUL", "host_fused").lower()
    if qkv_mm in ("dram", "multicast", "per_device", "host_fused"):
        return model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"]
    return model_args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"]


def _tt_attention_output_to_torch(tt_tensor, mesh_device, cluster_shape, batch_size, hidden_dim):
    """
    Read decode output to [B,1,dim]. Host WO / host-ref paths replicate full [1,1,B,dim] on every chip.
    """
    try:
        shards = ttnn.get_device_tensors(tt_tensor)
        if shards:
            t = ttnn.to_torch(shards[0]).float()
            if t.dim() == 4 and t.shape[-1] >= hidden_dim:
                return t[0, 0, :batch_size, :hidden_dim].unsqueeze(1)
    except Exception as exc:
        logger.warning(f"[qwen-attn-test] replicated readback failed: {exc}")

    cols = int(cluster_shape[1])
    try:
        shards = ttnn.get_device_tensors(tt_tensor)
        if len(shards) >= cols:
            col_parts = [ttnn.to_torch(shards[c]).float() for c in range(cols)]
            merged = torch.cat(col_parts, dim=-1)
            if merged.dim() == 4 and merged.shape[-1] >= hidden_dim:
                return merged[0, 0, :batch_size, :hidden_dim].unsqueeze(1)
    except Exception as exc:
        logger.warning(f"[qwen-attn-test] column-shard assembly failed: {exc}")

    tt_out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
    )
    logger.info(f"[qwen-attn-test] tt_out shape after concat={tuple(tt_out.shape)}")
    if tt_out.dim() == 4 and tt_out.shape[1] > 1:
        row_diff = float((tt_out[:, 0] - tt_out[:, 1]).abs().max().item())
        logger.info(f"[qwen-attn-test] mesh row0 vs row1 max_abs_diff={row_diff:.6e}")
        tt_out = tt_out[:, :1]
    return tt_out[0, 0, :batch_size, :hidden_dim].unsqueeze(1)


def _decode_pos_tensor(pos, batch_size, mesh_device, cluster_shape):
    current_pos = torch.tensor([pos for _ in range(batch_size)])
    return ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if batch_size > 1 else (None, None),
            mesh_shape=cluster_shape,
        ),
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    PAGED_ATTENTION_CASES,
    ids=PAGED_ATTENTION_IDS,
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_qwen_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    allow_bad_pcc = os.getenv("QWEN_ALLOW_BAD_PCC", "0") == "1"
    # WQKV is 2D-sharded (dim//4 per device); replicated full dim breaks matmul (5120 vs 1280).
    force_replicated_input = os.getenv("QWEN_ATTN_FORCE_REPLICATED_INPUT", "0") == "1"
    logger.info(f"QWEN_DEBUG_PAGED_ATTENTION={int(paged_attention)}")
    logger.info(f"QWEN_ALLOW_BAD_PCC={int(allow_bad_pcc)}")
    logger.info(f"QWEN_ATTN_FORCE_REPLICATED_INPUT={int(force_replicated_input)}")

    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seq_len)
    decode_input_memcfg = _decode_input_memcfg(model_args, force_replicated_input)
    # This unit test should run without runtime prefetcher.
    model_args.use_prefetcher = False
    model_args.n_layers = 1  # For the unit test, just run a sigle layer

    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 Model Loaded")

    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)
    logger.info(f"Reference Model Loaded with QK norm support")

    seq_len = 1

    generation_start_pos = int(os.getenv("QWEN_ATTN_TEST_START_POS", "0"))
    generation_length = int(os.getenv("QWEN_DECODE_DEBUG_STEPS", "1"))
    all_tests_pass = True
    logger.info(f"QWEN_ATTN_TEST_START_POS={generation_start_pos}")
    if paged_attention and os.getenv("QWEN_USE_SIMPLE_DECODE_SDPA", "0") == "1":
        logger.warning(
            "QWEN_USE_SIMPLE_DECODE_SDPA=1 is incompatible with paged KV; "
            "host SDPA ignores page_table and returns zeros. Use fused paged SDPA (env=0)."
        )

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    prefetcher_setup = None
    worker_sub_device_id = None
    if model_args.use_prefetcher:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=2,
            n_layers=1,
            is_qwen=True,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id

    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, is_qwen=True)

    tt_model = TtLlamaAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )
    use_host_ref_decode = os.getenv("QWEN_HOST_DECODE_FALLBACK", "0") == "1"
    qkv_stage_debug = os.getenv("QWEN_QKV_STAGE_DEBUG", "0") == "1"
    if use_host_ref_decode:
        logger.info(
            "QWEN_HOST_DECODE_FALLBACK=1: decode uses torch reference on gathered input "
            "(host SDPA+WO still run only when fallback=0)"
        )
    if qkv_stage_debug:
        logger.info("QWEN_QKV_STAGE_DEBUG=1: will PCC-check TT Q/K after rope vs torch reference")

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)
    if use_host_ref_decode or qkv_stage_debug:
        tt_model.host_reference_attn = reference_model
        tt_model.host_freqs_cis = freqs_cis

    # Prime reference + TT paged KV for [0, generation_start_pos) with the same dummy
    # inputs so decode at start_pos compares matching cache histories (not ref-only).
    prime_tt_kv = os.getenv("QWEN_ATTN_PRIME_TT_KV", "1") == "1"
    if generation_start_pos > 0:
        prime_dummies = []
        with torch.no_grad():
            for pos in range(generation_start_pos):
                dummy = torch.randn(batch_size, seq_len, model_args.dim) * 0.05
                prime_dummies.append(dummy)
                freqs_i = freqs_cis[pos, :].unsqueeze(0)
                reference_model(dummy, pos, freqs_i, mask=None)
        logger.info(f"Primed reference KV cache for positions [0, {generation_start_pos})")

        if prime_tt_kv:
            logger.info(
                f"Priming TT paged KV with {generation_start_pos} decode steps "
                f"(slow on first compile; set QWEN_ATTN_TEST_START_POS=0 for fast PCC)"
            )
            for pos, dummy in enumerate(prime_dummies):
                tt_dummy = dummy.clone()
                attention_input = model_args.prepare_residual_tensor_decode(
                    tt_dummy,
                    decode_input_memcfg,
                    force_replicated=force_replicated_input,
                )
                pos_tensor = _decode_pos_tensor(pos, batch_size, mesh_device, model_args.cluster_shape)
                rot_mats = rope_setup.get_rot_mats(torch.tensor([pos for _ in range(batch_size)]))
                tt_model(
                    attention_input,
                    pos_tensor,
                    rot_mats=rot_mats,
                    mode="decode",
                    page_table=page_table_tt,
                )
            logger.info(f"Primed TT paged KV cache for positions [0, {generation_start_pos})")
        else:
            logger.warning("QWEN_ATTN_PRIME_TT_KV=0: TT paged cache is empty at start_pos>0; PCC will be invalid.")

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = _decode_pos_tensor(generation_start_pos, batch_size, mesh_device, model_args.cluster_shape)
    if model_args.use_prefetcher:
        # Explicitly allocate global CB to avoid memory fragmentation
        prefetcher_setup.create_global_cb()

    for i in range(generation_length):
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = torch.randn(batch_size, seq_len, model_args.dim) * 0.05

        tt_attention_input = pt_attention_input.clone()
        if use_host_ref_decode or qkv_stage_debug:
            tt_model.host_input_golden = pt_attention_input

        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            decode_input_memcfg,
            force_replicated=force_replicated_input,
        )

        # No-prefetch decode uses the non-fused rotary op, which consumes the
        # tiled decode rotary tables.
        rot_mats = rope_setup.get_rot_mats(current_pos)

        if model_args.use_prefetcher:
            ttnn.dram_prefetcher(
                prefetcher_setup.get_input_tensors(),
                num_layers=1,
                global_cb=prefetcher_setup.global_circular_buffer,
            )
            mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        logger.info("Starting attention computation")
        logger.info(
            f"[qwen-attn-test] step={i} pos={int(current_pos[0])} "
            f"input_rms={float(pt_attention_input.pow(2).mean().sqrt().item()):.6e}"
        )

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        tt_output_torch = _tt_attention_output_to_torch(
            tt_out, mesh_device, model_args.cluster_shape, batch_size, model_args.dim
        )

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)

        ref_rms = float(reference_output.pow(2).mean().sqrt().item())
        tt_rms = float(tt_output_torch.pow(2).mean().sqrt().item())
        logger.info(f"[qwen-attn-test] output_rms ref={ref_rms:.6e} tt={tt_rms:.6e}")

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
        abs_diff = (reference_output - tt_output_torch).abs()
        tt_nan = int(torch.isnan(tt_output_torch).sum().item())
        tt_inf = int(torch.isinf(tt_output_torch).sum().item())
        ref_nan = int(torch.isnan(reference_output).sum().item())
        ref_inf = int(torch.isinf(reference_output).sum().item())

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        logger.info(
            f"[qwen-decode-debug] step={i} pos={int(current_pos[0])} "
            f"mean_abs_diff={abs_diff.mean().item():.6e} max_abs_diff={abs_diff.max().item():.6e}"
        )
        logger.info(
            f"[qwen-decode-debug] step={i} pos={int(current_pos[0])} "
            f"tt_nan={tt_nan} tt_inf={tt_inf} ref_nan={ref_nan} ref_inf={ref_inf}"
        )
        if passing:
            logger.info(f"[pos={current_pos[0]}] Qwen_Attention Passed!")
        else:
            if allow_bad_pcc:
                logger.warning(
                    f"[pos={current_pos[0]}] Qwen_Attention PCC Failed but tolerated " f"(QWEN_ALLOW_BAD_PCC=1)."
                )
            else:
                logger.warning(f"[pos={current_pos[0]}] Qwen_Attention Failed!")
                all_tests_pass = False

        # Increment position
        next_pos = generation_start_pos + i + 1
        current_pos = torch.tensor([next_pos for _ in range(batch_size)])
        current_pos_tensor = _decode_pos_tensor(next_pos, batch_size, mesh_device, model_args.cluster_shape)
    tt_ccl.close()
    if all_tests_pass:
        logger.info("Qwen Attention output Passed!")
    else:
        logger.warning("Qwen Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
