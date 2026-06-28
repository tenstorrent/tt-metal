# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Full Ideogram 4.0 transformer (embeddings + block stack + final layer) vs the
# OFFICIAL reference Ideogram4Transformer. Learnable transforms run on device;
# the parameter-free scaffolding (MRoPE cos/sin, the sinusoidal time embedding,
# indicator masks/index) is precomputed on host and fed in, mirroring the model's
# documented host/device split. Asserts PCC >= 0.99 on the velocity prediction.
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4Transformer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....reference.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

IMAGE_POSITION_OFFSET = 65536


def _build_model_inputs(config, batch_size, llm_len, image_len):
    """[llm tokens | image tokens] packed single-stream batch for the full model."""
    seq_len = llm_len + image_len
    llm_features = torch.randn(batch_size, seq_len, config.llm_features_dim)
    x = torch.randn(batch_size, seq_len, config.in_channels)
    t = torch.rand(batch_size)  # flow-matching time in [0, 1]
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # single segment

    indicator = torch.full((batch_size, seq_len), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :llm_len] = LLM_TOKEN_INDICATOR

    # position_ids (B, L, 3) = (t, h, w): llm tokens linear on all axes, image on an offset grid.
    position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long)
    lpos = torch.arange(llm_len)
    position_ids[:, :llm_len, 0] = lpos
    position_ids[:, :llm_len, 1] = lpos
    position_ids[:, :llm_len, 2] = lpos
    grid_h = int(round(image_len**0.5))
    while image_len % grid_h != 0:
        grid_h -= 1
    grid_w = image_len // grid_h
    hh = torch.arange(grid_h).repeat_interleave(grid_w)
    ww = torch.arange(grid_w).repeat(grid_h)
    position_ids[:, llm_len:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, llm_len:, 1] = IMAGE_POSITION_OFFSET + hh
    position_ids[:, llm_len:, 2] = IMAGE_POSITION_OFFSET + ww

    return llm_features, x, t, position_ids, segment_ids, indicator


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
# TP=1/SP=1 uses no CCL, so no fabric is required (and it sidesteps flaky fabric init).
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("num_layers", [2, 34], ids=["layers2", "layers34"])
@pytest.mark.parametrize(
    ("batch_size", "llm_len", "image_len"),
    [pytest.param(1, 64, 256, id="llm64_img256")],
)
def test_transformer_model(
    *,
    mesh_device: ttnn.MeshDevice,
    num_layers: int,
    batch_size: int,
    llm_len: int,
    image_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    seq_len = llm_len + image_len

    config = modeling_ideogram4.Ideogram4Config(num_layers=num_layers)

    torch_model = modeling_ideogram4.Ideogram4Transformer(config).to(dtype=torch_dtype)
    torch_model.eval()

    llm_features, x, t, position_ids, segment_ids, indicator = _build_model_inputs(
        config, batch_size, llm_len, image_len
    )
    llm_features = llm_features.to(torch_dtype)
    x = x.to(torch_dtype)

    with torch.no_grad():
        torch_out = torch_model(
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )

    # ---- tt model ----
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    tt_model = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=None,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # ---- host-precomputed scaffolding ----
    cos, sin = torch_model.rotary_emb(position_ids)  # (B, L, head_dim)
    t_sin = Ideogram4Transformer.sinusoidal_embedding(t, config.emb_dim)  # (B, emb_dim)
    llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.float32).unsqueeze(-1)
    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)

    tt_out = tt_model(
        x=bf16_tensor(x, device=mesh_device),
        llm_features=bf16_tensor(llm_features, device=mesh_device),
        t_sin=bf16_tensor(t_sin.unsqueeze(1), device=mesh_device),
        cos=bf16_tensor(cos.unsqueeze(1), device=mesh_device),
        sin=bf16_tensor(sin.unsqueeze(1), device=mesh_device),
        image_indicator_index=tensor.from_torch(image_idx, device=mesh_device, dtype=ttnn.uint32),
        llm_token_mask=bf16_tensor(llm_token_mask, device=mesh_device),
        output_image_mask=bf16_tensor(output_image_mask, device=mesh_device),
        attn_mask=None,
        spatial_sequence_length=seq_len,
    )
    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[None, None, None])

    logger.info(f"ideogram4 model: layers={num_layers} B={batch_size} llm={llm_len} img={image_len} seq={seq_len}")
    # Only OUTPUT_IMAGE positions carry a meaningful velocity (the reference zeros the
    # image-latent input at llm positions; their output is unused conditioning noise).
    image_mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]  # (L,), same across batch
    assert_quality(torch_out[:, image_mask], tt_out_torch[:, image_mask], pcc=0.99)
