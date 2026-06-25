# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
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
# Blackhole prefill runs column-axis (cluster_axis=1) collectives on device, which require a 2D-torus
# fabric; Wormhole uses the original 1D ring path.
_PREFILL_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_XY if _IS_BLACKHOLE else ttnn.FabricConfig.FABRIC_1D_RING


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": _PREFILL_FABRIC_CONFIG}],
    indirect=True,
)
def test_qwen_mlp_inference_prefill(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
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

    reference_model = FeedForward(
        dim=5120,
        hidden_dim=25600,
        multiple_of=1,
        ffn_dim_multiplier=None,
    )
    reference_model.load_state_dict(partial_state_dict_ref)

    logger.info(f"Reference Model Loaded")

    # Load Qwen model
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 1
    model_args.use_prefetcher = False
    state_dict = model_args.load_state_dict()

    logger.info(f"Qwen Model Loaded")

    # Mirror llama_model.setup_prefill for the no-prefetcher (Blackhole) path: do not load the
    # narrow prefetcher worker sub-device, so prefill ops run on the full compute grid. With the
    # prefetcher enabled (Wormhole) we still build it so the worker sub-device matches production.
    if model_args.use_prefetcher:
        prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id
    else:
        prefetcher_setup = None
        worker_sub_device_id = None
    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, mode="prefill", is_qwen=True)

    model_args.WEIGHTS_DTYPE = dtype
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

    logger.info("Run Qwen_MLP_PF")
    for i in range(3):
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3),
                mesh_shape=model_args.cluster_shape,
            ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        logger.info("Run Qwen_MLP")
        tt_output = tt_model.forward_prefill(tt_input, batch_size)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        logger.info("Qwen MLP Done")

        tt_output_torch = tt_output_torch[:, :1, :, : model_args.dim]

        ref_input = torch_input[:, :1, :, : model_args.dim]
        reference_output = reference_model(ref_input)

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_MLP Prefill Passed!")
    else:
        logger.warning("Qwen_MLP Prefill Failed!")
    tt_ccl.close()
    assert passing, f"Qwen MLP prefill output does not meet PCC requirement {pcc_required}: {pcc_message}."
