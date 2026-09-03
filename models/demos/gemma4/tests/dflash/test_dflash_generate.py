# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validates the full, self-contained multi-iteration dflash_generate loop (tt/dflash/
generate.py) against a 2-iteration torch reference (dump_torch_multi_iter.py) -- unlike
Steps 1-6, NOTHING here is fed in from the torch dump except the initial prompt: noise
embeddings, RoPE cos/sin, and the sliding-window context are all built by
dflash_generate itself from tokens it generates on real hardware.

This is the mechanism the demo (models/demos/gemma4/demo/dflash_demo.py) uses for actual
generation, so this test is the end-to-end correctness gate for that demo's "with DFlash"
numbers.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_generate.py -k 1x8 -s
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
def test_dflash_generate_t3k(mesh_device, device_params, model_path):
    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    ctx_len = ref["ctx_len"]
    input_ids = ref["input_ids"]
    # Reference's first two iterations produced 3 + 2 = 5 tokens total (see
    # dump_torch_multi_iter.py's printed "produced" values) -- request exactly that many
    # so this run's committed sequence is directly comparable to the static reference.
    ref_new_tokens = ref["final_output_ids"][0, ctx_len:].tolist()
    max_new_tokens = len(ref_new_tokens)

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    ref_layer_configs = [(bool(c[0]), int(c[1]) if c[1] is not None else None) for c in ref["layer_configs"]]
    assert tuple(ref_layer_configs) == config.layer_configs, (
        f"config.layer_configs {config.layer_configs} doesn't match the checkpoint's real "
        f"module attributes {ref_layer_configs}"
    )
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
    )

    print(f"ttnn generated:  {output_ids}")
    print(f"torch reference: {ref_new_tokens}")
    print(f"acceptance_lengths: {acceptance_lengths}")

    assert output_ids == ref_new_tokens, f"multi-iteration generation diverged: {output_ids} vs {ref_new_tokens}"
    print("[PASSED] self-contained dflash_generate matches the torch reference across 2 real iterations")
