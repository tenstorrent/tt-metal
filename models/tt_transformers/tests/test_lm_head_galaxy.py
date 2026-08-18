# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

MESH_SHAPE = (8, 4)
NUM_PRECEDING_AXIS1_ALL_GATHERS = 7


class TracedTTCCL(TT_CCL):
    """Record ping-pong slots selected by a focused CCL regression."""

    def __init__(self, mesh_device):
        super().__init__(mesh_device)
        self.semaphore_trace = []

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        self.semaphore_trace.append(("barrier", cluster_axis, self.barrier_semaphore_idx[semaphore_index]))
        return super().get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        self.semaphore_trace.append(("ag", cluster_axis, self.ag_semaphores_idx[semaphore_index]))
        return super().get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        self.semaphore_trace.append(("rs", cluster_axis, self.rs_semaphores_idx[semaphore_index]))
        return super().get_and_cycle_rs_semaphore_handles(cluster_axis)


def prime_token_zero_axis1_semaphore_parity(tt_ccl):
    """Match the seven completed axis-1 AGs before token-0 reaches LMHead.

    The one-layer Llama-3.2-1B decode prefix performs axis-1 all-gathers for
    attention norm, QKV all-reduce, attention output, FF norm, W1, W3, and
    final output norm. Completed collectives leave the double-buffer selectors
    at slot 1 without changing the LMHead reduce-scatter selector.
    """

    for _ in range(NUM_PRECEDING_AXIS1_ALL_GATHERS):
        tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1)
        tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1)

    assert tt_ccl.ag_semaphores_idx[1] == 1
    assert tt_ccl.rs_semaphores_idx[1] == 0
    assert tt_ccl.barrier_semaphore_idx[1] == 1
    tt_ccl.semaphore_trace.clear()


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_llama32_1b_decode_lm_head_composite_all_reduce_galaxy(mesh_device, reset_seeds, ensure_gc):
    model_args = ModelArgs(
        mesh_device,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=256,
        cache_hf=True,
        optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
    )
    if not model_args.is_galaxy or model_args.base_model_name != "Llama-3.2-1B":
        pytest.skip("This regression requires Llama-3.2-1B on an 8x4 Galaxy mesh")

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    output_weight = state_dict[f"{state_dict_prefix}output.weight"]

    reference_model = model_args.reference_lm_head()
    reference_model.load_state_dict({"weight": output_weight})

    tt_ccl = TracedTTCCL(mesh_device)
    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        max_columns_per_device=model_args.max_columns_per_device_lm_head,
        prefetcher=None,
    )

    # Use the production decode helper so the one valid batch row is padded to
    # 32 with zeros, width-fractured over mesh columns, and placed in the exact
    # L1 sharding delivered by final norm to the no-prefetch LMHead.
    torch_input = torch.randn(1, 1, model_args.dim, dtype=torch.bfloat16)
    reference_output = reference_model(torch_input)
    tt_input = model_args.prepare_residual_tensor_decode(
        torch_input,
        model_args.get_lm_head_input_mem_config(Mode.DECODE, None),
    )

    prime_token_zero_axis1_semaphore_parity(tt_ccl)
    tt_output = tt_model(tt_input)

    # LMHead's composite all-reduce must consume RS0/barrier1 followed by
    # AG1/barrier0. This is the token-0 ordering that the full model exercises.
    assert tt_ccl.semaphore_trace == [
        ("rs", 1, 0),
        ("barrier", 1, 1),
        ("ag", 1, 1),
        ("barrier", 1, 0),
    ]

    # Both calls are intentional: enqueue-only coverage would miss the hang
    # observed when the full model reads the asynchronously produced logits.
    ttnn.synchronize_device(mesh_device)
    tt_output_torch = (
        ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, MESH_SHAPE, dims=(3, 1)),
        )
        .permute(2, 1, 0, 3)
        .squeeze(2)[:1, 0:1, : model_args.vocab_size]
    )

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"Galaxy LMHead output does not meet PCC requirement {pcc_required}: {pcc_message}"

    tt_output.deallocate(True)
