# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched decode path (tt/batched_decode.py with the dense decode MoE) against the HuggingFace reference on Blackhole.

A 2-layer model (layer 0 sliding-window, layer 1 full attention) with random weights decodes 32 users at once, each
user its own random token sequence, one token per step. Every user's logits at every step are compared with the
reference logits at that position.

Meshes: 1x4 at TP=4 (QuietBox 2), 1x2 at TP=2 (one P300 board), and data parallel on a QuietBox 2: the 1x4 mesh split
into two 1x2 submeshes (one per board), one model on each, both decoding their own 32 users.

Run on a QuietBox 2 (Blackhole 1x4) or one P300 board (1x2):
    HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_batched_decode.py
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
USERS = 32
PAGE_BLOCK = 64
BLOCKS_PER_USER = 4  # 256 tokens per user: decode SDPA reads keys in chunks of up to 256 tokens
# Random weights make the logits nearly flat, so quantization alone (BFP4 experts) spreads a user's logits PCC over
# about 0.87-0.99. The TP=4 batched decode main already ships and the TP=2 one give the same spread (lowest user 0.868 /
# 0.873, every position's mean 0.965 or more). A wrong row, expert or shard drops a user far below these bars.
MIN_USER_PCC = 0.85
MIN_MEAN_PCC = 0.95


def _reference(config):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    config.num_hidden_layers = NUM_LAYERS
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference = GptOssForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (USERS, DECODE_STEPS))
    with torch.no_grad():
        logits = reference(
            input_ids=input_ids,
            position_ids=torch.arange(DECODE_STEPS).unsqueeze(0).expand(USERS, -1),
            use_cache=False,
        ).logits
    return reference, input_ids, logits


def _batched_model(mesh_device, config, state_dict):
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    mesh_shape = tuple(mesh_device.shape)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device)),
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=PAGE_BLOCK, max_num_blocks=USERS * BLOCKS_PER_USER),
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=USERS,
    )
    if not all(layer.batched_decode for layer in model.layers):
        pytest.skip(f"The batched decode path is not supported on this device / mesh {mesh_shape}")
    # The batched decode keeps its own copies of the attention projections; prefill still needs the originals.
    for layer in model.layers:
        weights = layer.self_attn.weights
        assert weights.wqkv.is_allocated() and weights.o_proj.is_allocated(), "batched decode freed a prefill weight"
    return model


def _decode_and_compare(models, input_ids, reference_logits, vocab_size):
    """Decode DECODE_STEPS tokens on every model (each its own USERS users, the same sequences) and compare each step's
    logits with the reference. All models' steps are enqueued before any output is read."""
    page_table = torch.arange(USERS * BLOCKS_PER_USER, dtype=torch.int32).reshape(USERS, BLOCKS_PER_USER)
    pccs = torch.zeros(len(models), DECODE_STEPS, USERS, dtype=torch.float64)
    for pos in range(DECODE_STEPS):
        outputs = []
        for model in models:
            tokens, current_pos, rot_idxs, tt_page_table = model.prepare_inputs_decode(
                input_ids[:, pos], torch.full((USERS,), pos), page_table=page_table
            )
            tt_logits, _ = model.ttnn_decode_forward(
                tokens=tokens, current_pos=current_pos, rot_mat_idxs=rot_idxs, page_table=tt_page_table
            )
            outputs.append(tt_logits)
        for i, (model, tt_logits) in enumerate(zip(models, outputs)):
            logits = model.process_output_decode(tt_logits.cpu(), B=USERS).reshape(USERS, -1)[:, :vocab_size].float()
            for user in range(USERS):
                pccs[i, pos, user] = comp_pcc(reference_logits[user, pos], logits[user], MIN_USER_PCC)[1]
    for i in range(len(models)):
        logger.info(
            f"model {i}: logits PCC per position (min / mean over {USERS} users): "
            + ", ".join(f"{p.min():.4f}/{p.mean():.4f}" for p in pccs[i])
        )
    worst = (pccs == pccs.min()).nonzero()[0].tolist()
    assert (
        pccs.min() >= MIN_USER_PCC
    ), f"model {worst[0]}, position {worst[1]}, user {worst[2]}: logits PCC {pccs.min():.4f}"
    assert pccs.mean(dim=-1).min() >= MIN_MEAN_PCC, f"mean logits PCC over users {pccs.mean(dim=-1).min():.4f}"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="A 2-layer model per mesh shape; run locally on Blackhole 1x2 or 1x4.",
)
@parametrize_mesh_with_fabric([(1, 2), (1, 4)])
def test_batched_decode_matches_reference(mesh_device, device_params, reset_seeds):
    if not is_blackhole():
        pytest.skip("The batched decode path runs on Blackhole only")
    config = TestFactory.setup_test(mesh_device, use_real_weights=False)["config"]
    reference, input_ids, reference_logits = _reference(config)
    state_dict = convert_hf_qkv_to_meta_format(reference.state_dict(), config.head_dim)
    model = _batched_model(mesh_device, config, state_dict)
    _decode_and_compare([model], input_ids, reference_logits, config.vocab_size)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Two 2-layer models on a QuietBox 2; run locally on Blackhole 1x4.",
)
@parametrize_mesh_with_fabric([(1, 4)])
def test_data_parallel_batched_decode_matches_reference(mesh_device, device_params, reset_seeds):
    """Data parallel on a QuietBox 2: one TP=2 model per P300 board (1x2 submeshes of the 1x4 mesh), as the vLLM
    integration builds them for --data-parallel-size 2."""
    if not is_blackhole():
        pytest.skip("The batched decode path runs on Blackhole only")
    # No device work on the parent mesh: its command queue would stay claimed by the parent while the submeshes close.
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, 2))
    config = TestFactory.setup_test(submeshes[0], use_real_weights=False)["config"]
    reference, input_ids, reference_logits = _reference(config)
    state_dict = convert_hf_qkv_to_meta_format(reference.state_dict(), config.head_dim)
    models = [_batched_model(submesh, config, state_dict) for submesh in submeshes]
    _decode_and_compare(models, input_ids, reference_logits, config.vocab_size)
