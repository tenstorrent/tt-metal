# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Isolated accuracy test for the Qwen3 LM head on Galaxy.

Runs LMHead.forward in decode mode against a torch reference. On the BH prefetcher path this
exercises the ring (gather_in0) lm_head matmul + BH gather-reduce path used by the full model in
decode — the component the decoder-layer unit test never touches.
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.common.utility_functions import (
    comp_pcc,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)


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
def test_qwen_lm_head_inference(mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    seq_len = 32  # decode M: batch 32 fused into M

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, max_seq_len=256, dummy_weights=False)
    model_args.n_layers = 1
    use_prefetcher = model_args.use_prefetcher
    logger.info(f"use_prefetcher={use_prefetcher}")

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    torch_weight = state_dict[f"{state_dict_prefix}output.weight"]  # [vocab, dim]

    if not use_prefetcher:
        prefetcher_setup = None
        worker_sub_device_id = None
    else:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=0,
            n_layers=model_args.n_layers,
            is_qwen=True,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id

    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, is_qwen=True)

    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        tt_ccl=tt_ccl,
        prefetcher_setup=prefetcher_setup,
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    reference_output = torch_input.float() @ torch_weight.float().t()  # [1, 1, seq, vocab]

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=model_args.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run Qwen LM Head (decode)")
    if hasattr(tt_ccl, "tt_lm_head_buffer") and tt_ccl.tt_lm_head_buffer is not None:
        tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(tt_ccl.tt_lm_head_buffer, tt_ccl.lm_head_buffer_mem_cfg)

    # Repeat to catch sporadic corruption (the full model shows intermittent junk logits in
    # specific ring slices, not a static offset bug).
    n_iters = 10
    pcc_required = 0.99
    worst_pcc = 1.0
    all_pass = True
    for it in range(n_iters):
        tt_outputs = tt_model(tt_input, worker_sub_device_id, mode="decode")
        tt_outputs_torch = [
            ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
            )
            for tt_output in tt_outputs
        ]
        for t in tt_outputs:
            ttnn.deallocate(t)
        tt_output_torch = torch.concat(tt_outputs_torch, dim=-1)
        tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.vocab_size]

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
        pcc_val = float(str(pcc_message).split(",")[-1].strip().split(" ")[-1]) if passing else 0.0
        logger.info(f"iter {it}: PCC: {pcc_message}")
        if not passing:
            all_pass = False
            # Localize: per-vocab-range PCC (832 = per-ring-core slice width on one device column)
            vocab = model_args.vocab_size
            for c in range(0, vocab, 19456):
                lo, hi = c, min(c + 19456, vocab)
                ok, msg = comp_pcc(reference_output[..., lo:hi], tt_output_torch[..., lo:hi], pcc_required)
                if not ok:
                    logger.warning(f"iter {it} vocab[{lo}:{hi}] PCC: {msg}")

    tt_ccl.close()

    assert all_pass, f"Qwen LM head PCC below {pcc_required} in at least one iteration"
