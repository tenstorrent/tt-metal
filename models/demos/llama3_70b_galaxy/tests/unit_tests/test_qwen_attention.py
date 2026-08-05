# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Single parametrized test. Wormhole runs the original (main) prefetcher path; Blackhole Galaxy runs
# the no-prefetcher bring-up path. The architecture is detected once at import so the pytest
# parameters (fabric config, paged attention) and the in-body setup select the right path.
import torch
import pytest
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
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    IS_BLACKHOLE as _IS_BLACKHOLE,
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)


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


def _tt_attention_output_to_torch(tt_tensor, mesh_device, cluster_shape, batch_size, hidden_dim):
    """
    Read decode output to [B,1,dim]. The no-prefetch path replicates full [1,1,B,dim] on every chip.
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
    if tt_out.dim() == 4 and tt_out.shape[1] > 1:
        tt_out = tt_out[:, :1]
    return tt_out[0, 0, :batch_size, :hidden_dim].unsqueeze(1)


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
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
    (False,) if _IS_BLACKHOLE else (True,),
    ids=("default_attention",) if _IS_BLACKHOLE else ("paged_attention",),
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

    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seq_len)
    if _IS_BLACKHOLE:
        # Blackhole bring-up runs the unit test without the runtime prefetcher.
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

    generation_start_pos = 0 if _IS_BLACKHOLE else 127
    generation_length = 1
    all_tests_pass = True

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

    if _IS_BLACKHOLE:
        # Keep the default device context when prefetcher is disabled; installing a custom subdevice
        # manager here can conflict with kernel core groups.
        prefetcher_setup = None
        worker_sub_device_id = None
    else:
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

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = _decode_pos_tensor(generation_start_pos, batch_size, mesh_device, model_args.cluster_shape)
    if not _IS_BLACKHOLE:
        # Explicitly allocate global CB to avoid memory fragmentation
        prefetcher_setup.create_global_cb()

    # Blackhole (no prefetcher) uses the non-ring activation sharding and the non-fused rotary op.
    input_memcfg = model_args.model_config[
        "SHARDED_ATTN_INPUT_MEMCFG" if _IS_BLACKHOLE else "SHARDED_ATTN_INPUT_RING_MEMCFG"
    ]

    for i in range(generation_length):
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = torch.randn(batch_size, seq_len, model_args.dim) * 0.05

        tt_attention_input = pt_attention_input.clone()

        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            input_memcfg,
            force_replicated=False,
        )

        # Get cos/sin matrices for the current position of each user
        if _IS_BLACKHOLE:
            rot_mats = rope_setup.get_rot_mats(current_pos)
        else:
            rot_mats = rope_setup.get_rm_rot_mats(current_pos)

        if not _IS_BLACKHOLE:
            ttnn.dram_prefetcher(
                prefetcher_setup.get_input_tensors(),
                num_layers=1,
                global_cb=prefetcher_setup.global_circular_buffer,
            )
            mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        logger.info("Starting attention computation")

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        if _IS_BLACKHOLE:
            tt_output_torch = _tt_attention_output_to_torch(
                tt_out, mesh_device, model_args.cluster_shape, batch_size, model_args.dim
            )
        else:
            # multi-device attention module returns replicated output
            tt_out = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
            )
            tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos[0]}] Qwen_Attention Passed!")
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
