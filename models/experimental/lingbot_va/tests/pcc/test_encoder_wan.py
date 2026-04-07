# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: HuggingFace UMT5EncoderModel vs TT UMT5Encoder.

Multi-device setup follows ``models/tt_dit/tests/encoders/umt5/test_umt5.py``:

- ``device_params`` with ``fabric_config=FABRIC_1D`` (``line_params``) for fabric-backed dispatch.
- ``num_links`` passed into ``CCLManager`` (default ``1``).
- ``parallel_config_and_ccl_manager``: ``tensor_parallel.factor = mesh_device.shape[1]``,
  ``mesh_axis=1`` (column TP).

Mesh shape from ``pcc_mesh_shape_request_param()`` (full system mesh). TT paths use fixture
``work_mesh_device`` (``(1,1)`` submesh when ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH`` is set).

**Wall time:** ``pytestmark = pytest.mark.timeout(600)``. Shorten sequence with
``LINGBOT_VA_UMT5_PCC_SEQ_LEN`` (e.g. ``128``).
"""

import gc
import os
from pathlib import Path

import pytest
import torch
import ttnn
from transformers import UMT5EncoderModel

from models.experimental.lingbot_va.tests.download_pretrained_weights import setup_checkpoint_root_for_tests
from models.experimental.lingbot_va.tests.mesh_utils import (
    mesh_num_devices,
    pcc_mesh_shape_request_param,
    umt5_encoder_hidden_states_to_torch,
    umt5_mesh_mapper_for_text_inputs,
    umt5_pad_input_ids_and_mask,
    umt5_post_encoder_hidden_states,
)

setup_checkpoint_root_for_tests()
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params

os.environ.setdefault("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "0")

pytestmark = pytest.mark.timeout(600)

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/text_encoder"


def _text_encoder_checkpoint_dir() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", os.getcwd())).resolve() / CHECKPOINT_PATH


MIN_PCC = 0.99
MAX_RELATIVE_RMSE = 0.15
BATCH_SIZE = 1
SEQ_LEN = int(os.environ.get("LINGBOT_VA_UMT5_PCC_SEQ_LEN", "512"))


@pytest.fixture
def parallel_config_and_ccl_manager(work_mesh_device, num_links, topology):
    """Same TP/CCL construction as ``test_umt5.parallel_config_and_ccl_manager``."""
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=work_mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(
        mesh_device=work_mesh_device,
        num_links=num_links,
        topology=topology,
    )
    return parallel_config, ccl_manager


@pytest.fixture(scope="module")
def hf_model():
    ckpt = _text_encoder_checkpoint_dir()
    if not ckpt.is_dir():
        pytest.skip(f"Lingbot-VA text encoder checkpoint not found: {ckpt}")
    model = UMT5EncoderModel.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    model.eval()
    return model


@pytest.mark.usefixtures("reset_seeds")
@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params", "topology"),
    [
        pytest.param(
            pcc_mesh_shape_request_param(),
            1,
            line_params,
            ttnn.Topology.Linear,
            id="lingbot_umt5_encoder_pcc",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_umt5_encoder_comparison(
    work_mesh_device,
    num_links,
    topology,
    parallel_config_and_ccl_manager,
    hf_model,
):
    assert num_links >= 1
    assert topology == ttnn.Topology.Linear
    encoder_parallel_config, ccl_manager = parallel_config_and_ccl_manager
    tp_factor = encoder_parallel_config.tensor_parallel.factor

    if mesh_num_devices(work_mesh_device) > 1 and tp_factor > 1 and hf_model.config.num_heads % tp_factor != 0:
        pytest.skip(
            f"HF num_heads={hf_model.config.num_heads} not divisible by encoder TP factor {tp_factor} "
            f"for mesh {work_mesh_device.shape}"
        )

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
    del text_out
    gc.collect()

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
        mesh_device=work_mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=encoder_parallel_config,
    )
    tt_encoder.load_torch_state_dict(text_weights)

    mesh_mapper = umt5_mesh_mapper_for_text_inputs(work_mesh_device, encoder_parallel_config)
    input_ids_pad, attention_mask_pad, pad_n = umt5_pad_input_ids_and_mask(
        input_ids, attention_mask, work_mesh_device, encoder_parallel_config
    )

    tt_input = ttnn.from_torch(
        input_ids_pad,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=work_mesh_device,
        mesh_mapper=mesh_mapper,
    )
    tt_mask = ttnn.from_torch(
        attention_mask_pad,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=work_mesh_device,
        mesh_mapper=mesh_mapper,
    )

    ttnn.synchronize_device(work_mesh_device)

    tt_hidden = tt_encoder(tt_input, attention_mask=tt_mask)
    tt_last = tt_hidden[-1]
    tt_out = umt5_post_encoder_hidden_states(ccl_manager, tt_last, tt_mask, work_mesh_device, encoder_parallel_config)
    ttnn.synchronize_device(work_mesh_device)
    tt_embed = umt5_encoder_hidden_states_to_torch(tt_out).float()

    if pad_n:
        tt_embed = tt_embed[:BATCH_SIZE]

    while tt_embed.dim() > 3:
        tt_embed = tt_embed.squeeze(0)

    assert tt_embed.shape == text_embed.shape, f"Shape mismatch: HF={text_embed.shape}, TT={tt_embed.shape}"

    assert_quality(text_embed, tt_embed, pcc=MIN_PCC, relative_rmse=MAX_RELATIVE_RMSE)
