# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Single file, single command. Wormhole runs the original (main) code path; Blackhole runs the
# Qwen3 Blackhole-Galaxy bring-up code path. The two are exposed as two tests, each skipped on the
# other architecture, so `pytest test_qwen_decoder.py` runs the right one on each platform.
import os
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.reference.qwen import TransformerBlock
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


def _is_blackhole_galaxy():
    # Optional explicit override (set to "blackhole"/"bh" or "wormhole"/"wh").
    forced = os.environ.get("QWEN_TEST_FORCE_ARCH", "").lower()
    if forced in ("blackhole", "bh"):
        return True
    if forced in ("wormhole", "wormhole_b0", "wh"):
        return False
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            return True
        if cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG):
            return False
    except Exception:
        pass
    arch = os.environ.get("ARCH_NAME", "")
    if not arch:
        try:
            arch = ttnn.get_arch_name()
        except Exception:
            arch = ""
    return "blackhole" in arch.lower()


_IS_BLACKHOLE = _is_blackhole_galaxy()


# The 8x4 Blackhole Galaxy decode path runs all column-axis (cluster_axis=1) and row-axis
# (cluster_axis=0) collectives on device. That requires a 2D-torus fabric: FABRIC_1D / FABRIC_1D_RING
# only build routing for a single line, so the cross-column collective throws `IndexError: map::at`.
_QWEN_DECODER_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_XY


@torch.no_grad()
@pytest.mark.skipif(_IS_BLACKHOLE, reason="Wormhole-only path; Blackhole runs test_llama_decoder_inference_bh.")
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
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
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
def test_llama_decoder_inference_wh(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
        is_qwen=True,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, is_qwen=True)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 127
    generation_length = 10
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
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

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=model_args.n_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    seqlen = 1

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if batch_size > 1 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rm_rot_mats(current_pos)
        tt_pf = prefetcher_setup.get_input_tensors()
        ttnn.dram_prefetcher(
            tt_pf,
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT model
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_output_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Llama Decoder Block Passed!")
        else:
            logger.warning("Llama Decoder Block Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    tt_ccl.close()

    if all_tests_pass:
        logger.info(f"All {generation_length} Llama decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Llama decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"


@torch.no_grad()
@pytest.mark.skipif(not _IS_BLACKHOLE, reason="Blackhole-only path; Wormhole runs test_llama_decoder_inference_wh.")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _QWEN_DECODER_FABRIC_CONFIG,
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
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
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
def test_llama_decoder_inference_bh(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    allow_bad_pcc = os.getenv("QWEN_ALLOW_BAD_PCC", "0") == "1"

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    # This unit test runs the Blackhole no-prefetch path (matches test_qwen_attention.py).
    model_args.use_prefetcher = False
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    prefetcher_setup = None
    worker_sub_device_id = None
    if model_args.use_prefetcher:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=5,
            n_layers=model_args.n_layers,
            is_qwen=True,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id

    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, is_qwen=True)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = int(os.getenv("QWEN_DECODER_START_POS", "127"))
    generation_length = int(os.getenv("QWEN_DECODER_STEPS", "10"))
    all_tests_pass = True
    logger.info(
        f"QWEN_DECODER_START_POS={generation_start_pos} QWEN_DECODER_STEPS={generation_length} "
        f"QWEN_ALLOW_BAD_PCC={int(allow_bad_pcc)} use_prefetcher={model_args.use_prefetcher}"
    )

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
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

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=model_args.n_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    seqlen = 1

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if batch_size > 1 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    # Explicitly allocate global CB to avoid memory fragmentation
    if model_args.use_prefetcher:
        prefetcher_setup.create_global_cb()

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rm_rot_mats(current_pos)
        if model_args.use_prefetcher:
            tt_pf = prefetcher_setup.get_input_tensors()
            ttnn.dram_prefetcher(
                tt_pf,
                num_layers=1,
                global_cb=prefetcher_setup.global_circular_buffer,
            )
            mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT model
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_output_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if os.getenv("QWEN_DECODER_STAGE_DEBUG", "0") == "1" and i == 0:
            # Reproduce the reference block's stagewise intermediates and PCC-compare against
            # the TT decode captures to localize where the full-block PCC degrades.
            ref_an = reference_model.attention_norm(pt_decode_input)
            ref_at = reference_model.attention(ref_an, current_pos[0], freqs_cis_i, mask=None)
            ref_h = pt_decode_input + ref_at
            ref_fin = reference_model.ffn_norm(ref_h)
            ref_ffn = reference_model.feed_forward(ref_fin)
            stage_refs = {
                "attn_out": ref_at,
                "h_residual": ref_h,
                "ff_in": ref_fin,
                "ff_out": ref_ffn,
            }
            for name, ref_t in stage_refs.items():
                tt_t = tt_model.stage_debug_tensors.get(name, None)
                if tt_t is None:
                    logger.warning(f"[decoder-stage] {name}: TT capture missing")
                    continue
                if isinstance(tt_t, str):
                    logger.warning(f"[decoder-stage] {name}: {tt_t}")
                    continue
                ref_v = ref_t.reshape(-1, 1, model_args.dim).float()
                s_pass, s_msg = comp_pcc(ref_v, tt_t)
                logger.info(f"[decoder-stage] {name}: {s_msg} pass={s_pass}")

        if passing:
            logger.info("Llama Decoder Block Passed!")
        else:
            logger.warning("Llama Decoder Block Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    tt_ccl.close()

    if all_tests_pass:
        logger.info(f"All {generation_length} Llama decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Llama decode Failed!")
        if allow_bad_pcc:
            logger.warning("QWEN_ALLOW_BAD_PCC=1: not failing the test despite low PCC (bring-up mode).")
        else:
            assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
