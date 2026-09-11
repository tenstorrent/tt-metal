# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validates dflash_generate(..., use_trace=True) -- the Metal-traced steady-state loop
(generate.py::_traced_steady_state) -- against the same 2-iteration torch reference
test_dflash_generate.py uses. A single dflash_generate call requesting exactly the
reference's tokens exercises iteration 0 (always eager, to get real steady-state inputs
to compile/capture against) + trace capture + replay for every iteration after that.

Confirms the traced path produces the EXACT SAME result as the eager path (both match
the torch reference bit-for-bit) -- i.e. tracing is a pure performance change, not a
behavior change. See docs/dflash_design.md for the real T3K speed numbers this enables.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_generate_traced.py -k 1x8 -s
"""

import os

import pytest
import torch

from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.generate import dflash_generate
from models.demos.gemma4.tt.dflash.lm_head import load_gemma4_lm_head_weight
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights
from models.tt_transformers.tt.common import PagedAttentionConfig

TORCH_REF_PATH = (
    "/tmp/claude-1002/-home-user-proj-sdk/4b2381cd-910e-4293-86a0-feecee22c8e9/scratchpad/torch_multi_iter_ref.pt"
)
MAX_SEQ_LEN = 128


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL", "google/gemma-4-31B-it")


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_generate_traced_t3k(mesh_device, device_params, model_path):
    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    ctx_len = ref["ctx_len"]
    input_ids = ref["input_ids"]
    ref_new_tokens = ref["final_output_ids"][0, ctx_len:].tolist()
    max_new_tokens = len(ref_new_tokens)

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)
    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)
    lm_head_weight = load_gemma4_lm_head_weight(mesh_device, mesh_config)

    page_params = {"page_block_size": 64, "page_max_num_blocks": MAX_SEQ_LEN // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks"]
    )
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    input_ids_padded = torch.nn.functional.pad(input_ids.squeeze(0), (0, MAX_SEQ_LEN - ctx_len), value=0)

    output_ids, acceptance_lengths = dflash_generate(
        model,
        mesh_device,
        weights,
        lm_head_weight,
        config,
        mesh_config,
        ccl_manager,
        tt_kv_cache,
        page_table,
        input_ids_padded,
        ctx_len,
        max_new_tokens,
        use_trace=True,
    )

    print(f"ttnn generated (traced):  {output_ids}")
    print(f"torch reference:          {ref_new_tokens}")
    print(f"acceptance_lengths: {acceptance_lengths}")

    assert output_ids == ref_new_tokens, f"traced generation diverged: {output_ids} vs {ref_new_tokens}"
    print("[PASSED] traced dflash_generate matches the torch reference across 2 real iterations")
