# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL-8B text encoder for Ideogram 4.0: qwen25vl port + per-head QK-RMSNorm,
# no qkv bias, and the 13-layer feature tap. HF reference is built from config
# (no 8B pull); "real" overlays the shipped Ideogram fp8 text_encoder weights.

import os

import pytest
import torch
import transformers
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from ....reference.ideogram4.dequant import dequant_fp8_state_dict
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from .common import capture_layer_outputs, encoder_from_hf_config, hf_rope_params

REPO = "Qwen/Qwen3-VL-8B-Instruct"
FP8 = os.environ.get("IDEOGRAM4_WEIGHTS")
_NEEDS_WEIGHTS = pytest.mark.skipif(not FP8, reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)")


def _reference_lm(weights: str):
    cfg = transformers.AutoConfig.from_pretrained(REPO)
    hf = transformers.AutoModel.from_config(cfg).to(torch.bfloat16)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    if weights == "real":
        sd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
        sd = {k[len("language_model.") :]: v for k, v in sd.items() if k.startswith("language_model.")}
        incompat = lm.load_state_dict(sd, strict=False)
        # empty missing/unexpected proves the weights landed; else both sides share the random init and PCC is vacuous
        assert not incompat.missing_keys and not incompat.unexpected_keys, (
            f"real Qwen3-VL load key mismatch: missing={incompat.missing_keys[:5]} "
            f"unexpected={incompat.unexpected_keys[:5]}"
        )
    return lm.eval()


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [
        pytest.param((2, 4), (2, 4), 1, id="tp4_fsdp2"),
        pytest.param((2, 4), (2, 4), 0, id="tp2_fsdp4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize(
    "weights",
    [pytest.param("random", id="random"), pytest.param("real", id="real", marks=_NEEDS_WEIGHTS)],
)
@pytest.mark.parametrize("masked", [pytest.param(False, id="nomask"), pytest.param(True, id="masked")])
@pytest.mark.parametrize("seq_len", [128])
def test_qwen3vl_text_encoder(
    *, mesh_device: ttnn.MeshDevice, submesh_shape, tp_axis, weights: str, masked: bool, seq_len: int
) -> None:
    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    lm = _reference_lm(weights)
    cfg = lm.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    rope_theta, mrope_section = hf_rope_params(cfg)

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with capture_layer_outputs(lm, QWEN3_VL_ACTIVATION_LAYERS) as caps:
        with torch.no_grad():
            lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    golden = [caps[i].float() for i in QWEN3_VL_ACTIVATION_LAYERS]

    enc = encoder_from_hf_config(
        cfg,
        activation_layers=QWEN3_VL_ACTIVATION_LAYERS,
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
        is_fsdp=True,
    )
    enc.load_torch_state_dict(lm.state_dict())

    cos, sin = create_rope_tensors(1, seq_len, None, head_dim, rope_theta, mrope_section)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh)
    attn_mask = tensor.from_torch(torch.ones(1, seq_len, dtype=torch.bool), device=submesh) if masked else None
    tt_caps = enc.forward(
        tt_ids,
        attention_mask=attn_mask,
        pos_embeds=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
    )

    pcc = 0.99 if weights == "real" else 0.98  # random weights over 36 layers accumulate more bf16 error
    for layer_idx, g, tt_t in zip(QWEN3_VL_ACTIVATION_LAYERS, golden, tt_caps):
        logger.info(f"qwen3vl [{weights}] TP={tp_factor} layer {layer_idx}:")
        assert_quality(g, tensor.to_torch(tt_t, mesh_axes=[None, None, None]), pcc=pcc)
