# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Single file, single command. Wormhole runs the original (main) code path; Blackhole runs the
# Qwen3 Blackhole-Galaxy bring-up code path. The two are exposed as two tests, each skipped on the
# other architecture, so `pytest test_qwen_mlp.py` runs the right one on each platform.
import os
import torch
import torch.nn.functional as F
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.model_config import ModelArgs
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.reference.qwen import FeedForward


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


@torch.no_grad()
@pytest.mark.skipif(_IS_BLACKHOLE, reason="Wormhole-only path; Blackhole runs test_qwen_mlp_inference_bh.")
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
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
def test_qwen_mlp_inference_wh(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Load reference model

    # Note that the Llama3 tests use a reference Llama model, here we call MLP from tt_transformers
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    first_layer_prefix = model_args_ref.get_state_dict_prefix("MLP", 0)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict_ref.items() if (k.startswith(first_layer_prefix))
    }

    # reference_model = model_args_ref.reference_mlp()
    reference_model = FeedForward(
        dim=5120,
        hidden_dim=25600,
        multiple_of=1,
        ffn_dim_multiplier=None,
    )
    reference_model.load_state_dict(partial_state_dict_ref)

    logger.info(f"Reference Model Loaded")

    # Load Qwen3 model
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    logger.info(f"Qwen3 Model Loaded")

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=3,
        n_layers=1,
        is_qwen=True,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, is_qwen=True)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    prev_pcc = None

    logger.info("Run Qwen_MLP_PF")
    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()
    for i in range(20):
        ttnn.dram_prefetcher(
            prefetcher_setup.get_input_tensors(),
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3),
                mesh_shape=model_args.cluster_shape,
            ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
            dtype=ttnn.bfloat8_b,
            memory_config=model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        logger.info("Run Qwen_MLP")
        tt_output = tt_model(tt_input, mode)
        logger.info(f"tt_output shape: {tt_output.shape}")

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
        logger.info("Qwen MLP Done")

        tt_output_torch = tt_output_torch[:, :1, :, : model_args.dim]

        ref_input = torch_input[:, :, :, : model_args.dim]
        reference_output = reference_model(ref_input)[:, :, :1, :]

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if prev_pcc is not None:
            assert prev_pcc == pcc_message, f"PCC changed from {prev_pcc} to {pcc_message} during inference."
        prev_pcc = pcc_message

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_MLP Passed!")
    else:
        logger.warning("Qwen_MLP Failed!")
    tt_ccl.close()
    assert passing, f"Qwen MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."


@torch.no_grad()
@pytest.mark.skipif(not _IS_BLACKHOLE, reason="Blackhole-only path; Wormhole runs test_qwen_mlp_inference_wh.")
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # Match decode unit-test configuration.
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        }
    ],
    indirect=True,
)
def test_qwen_mlp_inference_bh(max_seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode"

    # Load Qwen3 model
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=max_seq_len)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    logger.info(f"Qwen3 Model Loaded")

    # Build reference MLP from the same Qwen state dict used by TT path.
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
    partial_state_dict_ref = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=model_args.hidden_dim,
        multiple_of=1,
        ffn_dim_multiplier=None,
    )
    reference_model.load_state_dict(partial_state_dict_ref)
    logger.info("Reference Model Loaded")

    model_config = model_args.get_model_config()
    use_prefetcher = False
    model_config["USE_PREFETCHER"] = False
    enable_stage_debug = os.getenv("QWEN_MLP_STAGE_DEBUG", "0") == "1"

    if use_prefetcher:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=3,
            n_layers=1,
            is_qwen=True,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        tt_ccl_worker_sub_device_id = prefetcher_setup.worker_sub_device_id
    else:
        # Keep default device context when prefetcher is disabled.
        # Installing a custom subdevice manager here can conflict with kernel core groups.
        prefetcher_setup = None
        tt_ccl_worker_sub_device_id = None

    tt_ccl = TT_CCL(mesh_device, model_args, tt_ccl_worker_sub_device_id, is_qwen=True)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    seq_len = 1

    logger.info("Run Qwen_MLP_PF")
    # Explicitly allocate global CB to avoid memory fragmentation
    if use_prefetcher:
        prefetcher_setup.create_global_cb()
    for i in range(20):
        pt_decode_input = (torch.rand(batch_size, seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        if use_prefetcher:
            ttnn.dram_prefetcher(
                prefetcher_setup.get_input_tensors(),
                num_layers=1,
                global_cb=prefetcher_setup.global_circular_buffer,
            )
            mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        tt_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            # Use the decode MLP input memcfg (decoder residual memcfg is attention-oriented).
            model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"],
        )

        logger.info("Run Qwen_MLP")
        if i == 0 and enable_stage_debug:
            tt_output, tt_intermediates = tt_model(tt_input, mode, return_intermediates=True)
        else:
            tt_output = tt_model(tt_input, mode)
            tt_intermediates = None
        logger.info(f"tt_output shape: {tt_output.shape}")
        local_shard_width = tt_output.shape[-1]
        expected_local_width = model_args.dim // model_args.cluster_shape[1]
        logger.info(f"tt local shard width: {local_shard_width}, expected per-TP shard width: {expected_local_width}")
        assert (
            local_shard_width == expected_local_width
        ), f"Unexpected local shard width: got {local_shard_width}, expected {expected_local_width}"

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)
        logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
        logger.info("Qwen MLP Done")

        reference_output = reference_model(pt_decode_input)
        logger.info(f"reference_output shape: {reference_output.shape}")
        assert (
            reference_output.shape == tt_output_torch.shape
        ), f"Shape mismatch: reference {reference_output.shape} vs tt {tt_output_torch.shape}"

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if i == 0 and tt_intermediates is not None:

            def to_global_torch(tt_tensor):
                tt_torch = ttnn.to_torch(
                    tt_tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
                    ),
                )
                return tt_torch[:, 0:1, : model_args.max_batch_size, :].view(-1, 1, tt_torch.shape[-1])

            ref_ff1 = reference_model.w1(pt_decode_input)
            ref_activation = F.silu(ref_ff1) * reference_model.w3(pt_decode_input)
            ref_ff2 = reference_model.w2(ref_activation)

            tt_ff1_torch = to_global_torch(tt_intermediates["ff1_reduced"])
            tt_activation_torch = to_global_torch(tt_intermediates["activation"])
            tt_ff2_input_torch = to_global_torch(tt_intermediates["ff2_input"])
            tt_ff2_input_alt_torch = ttnn.to_torch(
                tt_intermediates["ff2_input"],
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
            )
            tt_ff2_input_alt_torch = tt_ff2_input_alt_torch[:, 0:1, : model_args.max_batch_size, :].view(
                -1, 1, tt_ff2_input_alt_torch.shape[-1]
            )
            tt_ff2_pre_torch = ttnn.to_torch(
                tt_intermediates["ff2_pre_allreduce"],
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
            )
            tt_ff2_torch = to_global_torch(tt_intermediates["ff2_output"])

            def compare_overlap(name, ref_tensor, tt_tensor, search_ref_slices=False):
                shared_width = min(ref_tensor.shape[-1], tt_tensor.shape[-1])
                tt_slice = tt_tensor[..., :shared_width]
                if search_ref_slices and ref_tensor.shape[-1] >= tt_slice.shape[-1]:
                    slice_width = tt_slice.shape[-1]
                    n_slices = ref_tensor.shape[-1] // slice_width
                    best_pcc = None
                    best_idx = -1
                    best_pass = False
                    for idx in range(n_slices):
                        ref_slice = ref_tensor[..., idx * slice_width : (idx + 1) * slice_width]
                        stage_passing, stage_pcc = comp_pcc(ref_slice, tt_slice, 0.99)
                        pcc_val = float(str(stage_pcc).split()[-1]) if isinstance(stage_pcc, str) else float(stage_pcc)
                        if best_pcc is None or pcc_val > best_pcc:
                            best_pcc = pcc_val
                            best_idx = idx
                            best_pass = stage_passing
                    logger.info(
                        f"{name} compare: ref_shape={tuple(ref_tensor.shape)} tt_shape={tuple(tt_tensor.shape)} "
                        f"slice_width={slice_width} best_ref_slice_idx={best_idx} pass={best_pass} best_pcc={best_pcc}"
                    )
                else:
                    ref_slice = ref_tensor[..., :shared_width]
                    stage_passing, stage_pcc = comp_pcc(ref_slice, tt_slice, 0.99)
                    logger.info(
                        f"{name} compare: ref_shape={tuple(ref_tensor.shape)} tt_shape={tuple(tt_tensor.shape)} "
                        f"shared_width={shared_width} pass={stage_passing} pcc={stage_pcc}"
                    )

            compare_overlap("FF1", ref_ff1, tt_ff1_torch)
            compare_overlap("Activation", ref_activation, tt_activation_torch)
            compare_overlap("FF2 input", ref_activation, tt_ff2_input_torch, search_ref_slices=True)
            compare_overlap("FF2 input alt-mesh", ref_activation, tt_ff2_input_alt_torch, search_ref_slices=True)
            compare_overlap("FF2", ref_ff2, tt_ff2_torch)

            # FF2 split diagnostics: pre-allreduce (matmul output) vs post-allreduce.
            logger.info(f"FF2 pre-allreduce raw shape: {tuple(tt_ff2_pre_torch.shape)}")
            pre_rows = tt_ff2_pre_torch[
                :,
                : model_args.cluster_shape[0],
                : model_args.max_batch_size,
                : model_args.dim,
            ]
            pre_row0 = pre_rows[:, 0:1, :, :].view(-1, 1, model_args.dim)
            pre_summed = pre_rows.sum(dim=1, keepdim=True).view(-1, 1, model_args.dim)
            compare_overlap("FF2 pre-allreduce row0", ref_ff2, pre_row0)
            compare_overlap("FF2 pre-allreduce summed_rows", ref_ff2, pre_summed)
            ff2_stage_passing, ff2_stage_pcc = comp_pcc(pre_summed, tt_ff2_torch, 0.99)
            logger.info(f"FF2 allreduce consistency: pre_summed_vs_post pass={ff2_stage_passing} pcc={ff2_stage_pcc}")

            # Free intermediate buffers after debug stage comparison.
            ttnn.deallocate(tt_intermediates["ff1_reduced"])
            ttnn.deallocate(tt_intermediates["ff3_reduced"])
            ttnn.deallocate(tt_intermediates["activation"])
            ttnn.deallocate(tt_intermediates["ff2_input"])
            ttnn.deallocate(tt_intermediates["ff2_pre_allreduce"])

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_MLP Passed!")
    else:
        logger.warning("Qwen_MLP Failed!")
    tt_ccl.close()
    assert passing, f"Qwen MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
