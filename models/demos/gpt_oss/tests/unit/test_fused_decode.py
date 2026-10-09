# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused decode path (tt/fused_decode) against the HuggingFace reference on Blackhole 1x4.

A 2-layer model (layer 0 sliding-window, layer 1 full attention) with random weights decodes a short sequence one
token at a time through the fused decode path: decode inputs op, layer boundaries (fabric all-reduce + residual add +
RMSNorm), streamed QKV / o_proj / router / routed experts, and the streamed LM head with folded logits. Every step's
logits, unfolded on the host by Model.process_output_decode, are compared with the reference logits at that position.

Run on a QuietBox 2 (Blackhole 1x4):
    HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_fused_decode.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import TestFactory, parametrize_mesh_with_fabric

NUM_LAYERS = 2
DECODE_STEPS = 8
PAGE_BLOCK = 64
PAGE_BLOCKS = 64  # 4096-token KV cache, as the demo (decode SDPA reads keys in chunks of up to 256 tokens)
PCC_THRESHOLD = 0.95


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="~45 s on a QuietBox 2; the CI gpt-oss-20b unit job has a 5-minute budget. Run locally on Blackhole 1x4.",
)
@parametrize_mesh_with_fabric([(1, 4)])
def test_fused_decode_matches_reference(mesh_device, device_params, reset_seeds):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    if not is_blackhole():
        pytest.skip("The fused decode path runs on Blackhole only")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    config.num_hidden_layers = NUM_LAYERS
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"

    reference = GptOssForCausalLM(config).eval()
    state_dict = convert_hf_qkv_to_meta_format(reference.state_dict(), config.head_dim)

    mesh_shape = tuple(mesh_device.shape)
    page_table = torch.arange(PAGE_BLOCKS, dtype=torch.int32).reshape(1, PAGE_BLOCKS)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device)),
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=PAGE_BLOCK, max_num_blocks=PAGE_BLOCKS),
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=1,
    )
    if not model.fused_decode:
        pytest.skip(f"The fused decode path is not supported on this device / mesh {mesh_shape}")

    input_ids = torch.randint(0, config.vocab_size, (1, DECODE_STEPS))
    with torch.no_grad():
        reference_logits = reference(
            input_ids=input_ids,
            position_ids=torch.arange(DECODE_STEPS).unsqueeze(0),
            use_cache=False,
        ).logits[0]

    for pos in range(DECODE_STEPS):
        tokens, current_pos, rot_idxs, tt_page_table = model.prepare_inputs_decode(
            input_ids[0, pos : pos + 1], torch.tensor([pos]), page_table=page_table
        )
        tt_logits, _ = model.ttnn_decode_forward(
            tokens=tokens, current_pos=current_pos, rot_mat_idxs=rot_idxs, page_table=tt_page_table
        )
        logits = model.process_output_decode(tt_logits.cpu(), B=1).reshape(-1)
        assert logits.shape[-1] == config.vocab_size
        passing, pcc = comp_pcc(reference_logits[pos], logits.float(), PCC_THRESHOLD)
        logger.info(f"position {pos}: logits PCC {pcc}")
        assert passing, f"position {pos}: logits PCC {pcc} < {PCC_THRESHOLD}"
