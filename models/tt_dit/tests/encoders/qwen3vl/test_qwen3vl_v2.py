# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Text-only inference through `Qwen3VlEncoder` against transformers, on the released 8B weights."""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch
import transformers
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlEncoder
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.reference.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from models.tt_dit.reference.ideogram4.dequant import dequant_fp8_state_dict
from models.tt_dit.tests.encoders.qwen3vl.common import capture_layer_outputs
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

CHECKPOINT = "Qwen/Qwen3-VL-8B-Instruct"
FP8 = os.environ.get("IDEOGRAM4_WEIGHTS")
_NEEDS_WEIGHTS = pytest.mark.skipif(not FP8, reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)")


def test_state_conversion_is_text_only() -> None:
    """The vision tower is dropped at load time, so text-only is the only thing that can run."""
    state_dict = {
        "model.visual.patch_embed.proj.weight": torch.zeros(1),
        "model.visual.blocks.0.attn.qkv.weight": torch.zeros(1),
        "model.language_model.embed_tokens.weight": torch.zeros(1),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.zeros(1),
        "model.language_model.norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }

    converted = Qwen3VlEncoder.convert_state(state_dict)

    assert set(converted) == {
        "token_embedding.weight",
        "layers.0.attn.q_proj.weight",
        "layers.0.attn.q_norm.weight",
        "final_norm.weight",
        "final_linear.weight",
    }

    # The Ideogram 4 checkpoint holds the language model without the `model.` prefix.
    unprefixed = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    assert set(Qwen3VlEncoder.convert_state(unprefixed)) == set(converted) - {"final_linear.weight"}


@pytest.mark.parametrize(
    ("mesh_device", "tp", "fsdp"),
    [
        pytest.param((1, 4), (4, 1), None, id="1x4"),
        pytest.param((1, 8), (8, 1), None, id="1x8"),
        pytest.param((2, 4), (4, 1), (2, 0), id="2x4_tp4_fsdp2"),
        pytest.param((2, 4), (2, 0), (4, 1), id="2x4_tp2_fsdp4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}, id="line")],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        pytest.param(False, id="unmasked"),
    ],
)
def test_text_only_forward(
    *, mesh_device: ttnn.MeshDevice, tp: tuple[int, int], fsdp: tuple[int, int] | None, masked: bool
) -> None:
    """A prompt of token ids alone reproduces the reference hidden states, with no image input."""
    torch.manual_seed(0)

    batch_size = 1
    sequence_length = 512

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig.from_tuples(tp=tp, sp=None, fsdp=fsdp)

    torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16)
    text_config = torch_model.config.text_config

    model = Qwen3VlEncoder(
        Qwen3VlEncoder.config_from_hf(torch_model.config),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    model.load_torch_state_dict(Qwen3VlEncoder.convert_state(torch_model.state_dict()))

    tokens = torch.randint(0, text_config.vocab_size, [batch_size, sequence_length])
    lengths = torch.randint(sequence_length // 4, 3 * sequence_length // 4, [batch_size])
    mask = torch.arange(sequence_length).flip([0]) < lengths.unsqueeze(1) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        mask=tt_mask,
        skip_final_linear=True,
        output_hidden_states=True,
    )
    tt_hidden_states_torch = [tensor.to_torch(t) for t in tt_hidden_states]

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.model.forward(
            input_ids=tokens,
            attention_mask=mask if mask is not None else torch.ones_like(tokens),
            output_hidden_states=True,
        )
    assert not isinstance(out, tuple)
    # The embeddings and every layer's output; the last entry is normalized like ours.
    hidden_states = [*out.hidden_states[:-1], out.last_hidden_state]

    if mask is not None:
        # Masked positions at the start of the sequence hold undefined values from a softmax over
        # all -inf, so compare only the real tokens.
        _, _, d = hidden_states[0].shape
        hidden_states = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in hidden_states]
        tt_hidden_states_torch = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in tt_hidden_states_torch]

    assert len(hidden_states) == len(tt_hidden_states_torch)

    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x, tt_x, pcc=0.99, relative_rmse=0.15)


def _reference_lm(weights: str):
    cfg = transformers.AutoConfig.from_pretrained(CHECKPOINT)
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


# Qwen3-VL-8B text encoder for Ideogram 4.0 and its 13-layer feature tap. HF reference is built from
# config (no 8B pull); "real" overlays the shipped Ideogram fp8 text_encoder weights.
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
    fsdp_axis = 1 - tp_axis

    lm = _reference_lm(weights)
    cfg = lm.config

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with capture_layer_outputs(lm, QWEN3_VL_ACTIVATION_LAYERS) as caps:
        with torch.no_grad():
            lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    golden = [caps[i].float() for i in QWEN3_VL_ACTIVATION_LAYERS]

    # As the Ideogram 4 pipeline builds it: no head, and no final norm since the taps are raw layer outputs.
    enc = Qwen3VlEncoder(
        dataclasses.replace(Qwen3VlEncoder.config_from_hf(cfg), final_norm=False, final_linear=False),
        device=submesh,
        parallel_config=EncoderParallelConfig.from_tuples(
            tp=(tp_factor, tp_axis), sp=None, fsdp=(tuple(submesh.shape)[fsdp_axis], fsdp_axis)
        ),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
    )
    enc.load_torch_state_dict(
        Qwen3VlEncoder.convert_state(
            {f"model.language_model.{k}": v for k, v in lm.state_dict().items() if k != "norm.weight"}
        )
    )

    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh)
    attn_mask = tensor.from_torch(torch.ones(1, seq_len, dtype=torch.bool), device=submesh) if masked else None
    tt_caps = enc.forward(tt_ids, mask=attn_mask, output_hidden_states=[i + 1 for i in QWEN3_VL_ACTIVATION_LAYERS])

    pcc = 0.99 if weights == "real" else 0.98  # random weights over 36 layers accumulate more bf16 error
    for layer_idx, g, tt_t in zip(QWEN3_VL_ACTIVATION_LAYERS, golden, tt_caps):
        logger.info(f"qwen3vl [{weights}] TP={tp_factor} layer {layer_idx}:")
        assert_quality(g, tensor.to_torch(tt_t), pcc=pcc)
