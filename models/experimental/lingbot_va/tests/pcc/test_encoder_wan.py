# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: HuggingFace UMT5EncoderModel vs TT UMT5Encoder."""

import pytest
import torch
import ttnn
from transformers import UMT5EncoderModel

from models.common.metrics import compute_pcc
from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/text_encoder"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
SEQ_LEN = 512


@pytest.fixture(scope="module")
def hf_model():
    model = UMT5EncoderModel.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    model.eval()
    return model


@pytest.mark.parametrize(
    "mesh_device",
    [mesh_shape_request_param()],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_umt5_encoder_comparison(mesh_device, hf_model):
    text_weights = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    torch.manual_seed(42)

    input_ids = torch.randint(
        low=0,
        high=hf_model.config.vocab_size,
        size=(BATCH_SIZE, SEQ_LEN),
        dtype=torch.long,
    )
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    with torch.no_grad():
        text_out = hf_model(input_ids=input_ids, attention_mask=attention_mask)
    text_embed = text_out.last_hidden_state.float()

    encoder_parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    umt5_config = UMT5Config(
        vocab_size=hf_model.config.vocab_size,
        embed_dim=hf_model.config.d_model,
        ff_dim=hf_model.config.d_ff,
        kv_dim=hf_model.config.d_kv,
        num_heads=hf_model.config.num_heads,
        num_hidden_layers=hf_model.config.num_layers,
        max_prompt_length=SEQ_LEN,
        layer_norm_eps=hf_model.config.layer_norm_epsilon,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
    )

    tt_encoder = TTUMT5Encoder(
        config=umt5_config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=encoder_parallel_config,
    )
    tt_encoder.load_torch_state_dict(text_weights)

    tt_input = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, device=mesh_device)

    tt_out = tt_encoder(tt_input, attention_mask=tt_mask)
    tt_out = tt_out[-1]
    tt_embed = ttnn.to_torch(tt_out).float()

    while tt_embed.dim() > 3:
        tt_embed = tt_embed.squeeze(0)

    assert tt_embed.shape == text_embed.shape, f"Shape mismatch: HF={text_embed.shape}, TT={tt_embed.shape}"

    pcc = compute_pcc(text_embed, tt_embed)
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
