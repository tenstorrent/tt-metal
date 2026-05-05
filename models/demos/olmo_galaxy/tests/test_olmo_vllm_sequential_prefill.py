# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.olmo_galaxy.demo.text_olmo_demo import create_olmo_tt_model
from models.demos.olmo_galaxy.tt.generator_vllm import OLMo3ForCausalLM

DEVICE_PARAMS = [
    {
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        "trace_region_size": 184915840,
        "num_command_queues": 1,
        "worker_l1_size": 1345000,
        "fabric_config": True,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
]


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_olmo_vllm_sequential_b1_prefill_two_users(mesh_device, device_params, reset_seeds, ensure_gc):
    """Exercise the vLLM OLMo wrapper's sequential B1 prefill path.

    Two users are below the batched-prefill threshold, so `prefill_forward_text`
    runs them one at a time and drains/resets OLMo CCL state between users.
    """
    page_params = {"page_block_size": 64, "page_max_num_blocks": 4096}
    model_args, tt_model, page_table, kv_cache = create_olmo_tt_model(
        mesh_device=mesh_device,
        max_batch_size=32,
        max_seq_len=256,
        num_layers=1,
        page_params=page_params,
        use_paged_kv_cache=True,
    )
    generator = OLMo3ForCausalLM(tt_model, model_args, mesh_device)
    generator.prefill_traces_warmup = True

    tokens = torch.tensor(
        [
            [1, 439, 318, 262, 3139, 30, 2],
            [1, 2437, 389, 345, 1804, 30, 2],
        ],
        dtype=torch.long,
    )
    prompt_lens = torch.tensor([tokens.shape[1], tokens.shape[1]], dtype=torch.long)

    next_tokens = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=True,
        sampling_params=None,
        empty_slots=[0, 1],
    )

    assert next_tokens.shape == (2,)
    assert torch.isfinite(next_tokens.float()).all()
