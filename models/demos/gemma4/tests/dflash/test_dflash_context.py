# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 2 validation: extract the DFlash drafter's "context" from a REAL Gemma4-31B
target forward pass on real T3K hardware, using the real weights loaded in
test_dflash_weights.py, and PCC-compare against the real torch reference
(dump_torch_context.py's saved output, computed in the separate transformers==
5.15.0 venv the reference requires -- see reference/dflash/README.md).

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_context.py -k 1x8 -s
"""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.context import compute_context
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights
from models.tt_transformers.tt.common import PagedAttentionConfig

TORCH_REF_PATH = (
    "/tmp/claude-1002/-home-user-proj-sdk/4b2381cd-910e-4293-86a0-feecee22c8e9/scratchpad/torch_context_ref.pt"
)
MAX_SEQ_LEN = 128


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL", "google/gemma-4-31B-it")


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_context_extraction_t3k(mesh_device, device_params, model_path):
    from models.common.utility_functions import comp_pcc

    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    input_ids = ref["input_ids"]  # [1, seq_len]
    torch_context = ref["context"]  # [1, seq_len, hidden_size]
    target_layer_ids = list(ref["target_layer_ids"])

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    assert target_layer_ids == list(config.target_layer_ids), "torch ref / config target_layer_ids mismatch"
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))

    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)

    batch_size = 1
    page_params = {"page_block_size": 64, "page_max_num_blocks": MAX_SEQ_LEN // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks"]
    )

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,  # need all 60 -- deepest tap is layer 57
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )

    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        batch_size, paged_attention_config.max_num_blocks
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    # Pad to the FULL MAX_SEQ_LEN bucket (matching demo/text_demo.py's convention).
    seq_len = input_ids.shape[-1]
    padded_len = MAX_SEQ_LEN
    input_ids_padded = torch.nn.functional.pad(input_ids.squeeze(0), (0, padded_len - seq_len), value=0)

    embeds, _, _, _, _, _ = model.prepare_inputs_prefill(input_ids_padded.unsqueeze(0), page_table=page_table_tt)

    tapped = {}

    def _probe(layer_idx, hidden_states):
        if layer_idx in target_layer_ids:
            # to_memory_config (not clone): hidden_states may be width-sharded in L1 at this
            # point, and ttnn.clone requires input/output to share the same sharded/interleaved
            # layout -- to_memory_config handles the resharding into plain interleaved DRAM.
            tapped[layer_idx] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

    model.layer_probe = _probe
    try:
        logits = model.ttnn_prefill_forward(
            embeds,
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            get_last_token=-1,
            input_ids_torch=input_ids_padded.unsqueeze(0),
            embeds_torch=None,
        )
        ttnn.deallocate(logits)
    finally:
        model.layer_probe = None

    assert set(tapped.keys()) == set(target_layer_ids), f"missing taps: {set(target_layer_ids) - set(tapped.keys())}"
    tapped_ordered = [tapped[i] for i in target_layer_ids]

    context_tt = compute_context(weights, tapped_ordered)
    context_back = ttnn.to_torch(ttnn.get_device_tensors(context_tt)[0]).float()
    context_back = context_back.reshape(1, padded_len, -1)[:, :seq_len, :]

    passing, pcc = comp_pcc(torch_context, context_back, pcc=0.97)
    print(f"context extraction PCC: {pcc}")
    assert passing, f"context PCC {pcc} below threshold"
    print("[PASSED] real Gemma4-31B target -> 6-layer tap -> fc/hidden_norm context matches torch reference")
