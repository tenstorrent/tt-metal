# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""`TransformerEncoder` against tiny random-weight `transformers` models.

The per-model encoder tests load real checkpoints and run a multi-billion-parameter CPU reference,
which makes them the wrong place to validate a change to the shared base. This covers the base's
code paths in seconds: the three attention variants it supports, tensor and sequence parallelism,
masked and unmasked prefill with implicit and explicit positions, and teacher-forced decode,
masked and traced.
"""

from __future__ import annotations

import pytest
import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.smollm3.model_smollm3 import STATE_CONVERSION as SMOLLM3_STATE_CONVERSION
from models.tt_dit.encoders.transformer import StateConversion, TransformerEncoder, TransformerEncoderConfig
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params, mesh_device_config_to_string

VOCAB_SIZE = 512
EMBED_SIZE = 128
FF_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = EMBED_SIZE // NUM_HEADS
NORM_EPS = 1e-6
ROPE_THETA = 10000.0
PROMPT_LENGTH = 56
NEW_TOKENS = 8
MAX_LENGTH = PROMPT_LENGTH + NEW_TOKENS  # a tile of 32 on each of two sequence-parallel shards
BATCH_SIZE = 2
PADDING = 24  # left padding of the second row when masked

# (transformers config, transformers model, what the base must enable for it, extra config kwargs)
ARCHITECTURES = {
    "llama": (transformers.LlamaConfig, transformers.LlamaForCausalLM, {}, {}),
    "qwen2": (transformers.Qwen2Config, transformers.Qwen2ForCausalLM, {"attn_qkv_bias": True}, {}),
    "qwen3": (transformers.Qwen3Config, transformers.Qwen3ForCausalLM, {"attn_qk_norm": True}, {"head_dim": HEAD_SIZE}),
}
# The Llama family that SmolLM3 belongs to, plus the q/k norms of Qwen3.
STATE_CONVERSION = StateConversion(
    rename=[
        *(SMOLLM3_STATE_CONVERSION.rename or []),
        (r"^model\.layers\.([0-9]+)\.self_attn\.([qk])_norm", r"layers.\1.attn.\2_norm"),
    ],
)

MESH = pytest.mark.parametrize(
    ("mesh_device", "sp_axis"),
    [
        pytest.param((1, 1), None, id="1x1"),
        pytest.param((1, 2), None, id="1x2"),
        pytest.param((2, 2), 0, id="2x2sp"),
    ],
    indirect=["mesh_device"],
)
DECODE_MESH = pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2)], ids=mesh_device_config_to_string, indirect=True)
DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params", [{**line_params, "trace_region_size": 16_000_000}], ids=["line"], indirect=True
)
ARCHITECTURE = pytest.mark.parametrize("architecture", list(ARCHITECTURES))
MASKED = pytest.mark.parametrize("masked", [pytest.param(False, id="unmasked"), pytest.param(True, id="masked")])


def _reference_model(architecture: str) -> transformers.PreTrainedModel:
    config_cls, model_cls, _, extra = ARCHITECTURES[architecture]
    config = config_cls(
        vocab_size=VOCAB_SIZE,
        hidden_size=EMBED_SIZE,
        intermediate_size=FF_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        rms_norm_eps=NORM_EPS,
        rope_theta=ROPE_THETA,
        tie_word_embeddings=False,
        max_position_embeddings=256,
        **extra,
    )
    torch.manual_seed(0)
    return model_cls(config).eval()


def _encoder(
    architecture: str, reference: transformers.PreTrainedModel, mesh_device: ttnn.MeshDevice, *, sp_axis: int | None
) -> TransformerEncoder:
    """The encoder of `reference`, tensor-parallel over mesh axis 1 and sequence-parallel over `sp_axis`."""
    _, _, flags, _ = ARCHITECTURES[architecture]
    config = TransformerEncoderConfig(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        ff_size=FF_SIZE,
        head_size=HEAD_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        norm_eps=NORM_EPS,
        rope_config=RopeConfig(theta=ROPE_THETA),
        # `attn_qkv_bias` (qwen2) and `attn_qk_norm` (qwen3) come from the per-architecture flags.
        **{"attn_qkv_bias": False, "attn_out_bias": False, **flags},
    )
    tp_axis = _tp_axis(mesh_device)
    if tp_axis is None and sp_axis is None:
        parallel_config = ccl_manager = None
    else:
        parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=(
                ParallelFactor(factor=mesh_device.shape[sp_axis], mesh_axis=sp_axis) if sp_axis is not None else None
            ),
        )
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    model = TransformerEncoder(config, device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)
    model.load_torch_state_dict(STATE_CONVERSION.convert(reference.state_dict()))
    return model


def _tp_axis(mesh_device: ttnn.MeshDevice) -> int | None:
    return 1 if mesh_device.shape[1] > 1 else None


def _prompt(*, masked: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random tokens of shape (BATCH_SIZE, MAX_LENGTH), the second row left-padded when `masked`."""
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB_SIZE, [BATCH_SIZE, MAX_LENGTH])
    if not masked:
        return tokens, None
    lengths = torch.tensor([MAX_LENGTH, MAX_LENGTH - PADDING])
    return tokens, torch.arange(MAX_LENGTH).flip([0]) < lengths.unsqueeze(1)


def _real_rows(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Drops padded positions, whose values are undefined on both sides."""
    if mask is None:
        return x
    return x.masked_select(mask.unsqueeze(-1).bool()).view(-1, x.shape[-1])


@MESH
@DEVICE_PARAMS
@ARCHITECTURE
@MASKED
def test_prefill(*, mesh_device: ttnn.MeshDevice, sp_axis: int | None, architecture: str, masked: bool) -> None:
    """Every layer's input and the logits of one forward, against the reference.

    The masked run also passes the plain-range positions explicitly, which must not change anything.
    """
    reference = _reference_model(architecture)
    model = _encoder(architecture, reference, mesh_device, sp_axis=sp_axis)
    tokens, mask = _prompt(masked=masked)

    with torch.no_grad():
        out = reference.forward(
            input_ids=tokens,
            attention_mask=mask if mask is not None else torch.ones_like(tokens),
            output_hidden_states=True,
        )
    assert not isinstance(out, tuple)
    hidden_states = list(out.hidden_states or [])

    positions = torch.arange(MAX_LENGTH).expand(BATCH_SIZE, -1).float() if masked else None
    tt_out = model.forward(
        tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32, mesh_axes=[None, sp_axis]),
        mask=tensor.from_torch(mask, device=mesh_device) if mask is not None else None,
        positions=tensor.from_torch(positions, device=mesh_device, dtype=ttnn.float32) if masked else None,
        output_hidden_states=True,
    )
    assert isinstance(tt_out, list)
    *tt_hidden_states, tt_logits = tt_out
    assert len(tt_hidden_states) == len(hidden_states)

    for x, tt_x in zip(hidden_states, tt_hidden_states, strict=True):
        assert_quality(
            _real_rows(x, mask),
            _real_rows(tensor.to_torch(tt_x, mesh_axes=[None, sp_axis, None]), mask),
            pcc=0.999,
            relative_rmse=0.05,
        )
    assert_quality(
        _real_rows(out.logits, mask),
        _real_rows(tensor.to_torch(tt_logits, mesh_axes=[None, sp_axis, _tp_axis(mesh_device)]), mask),
        pcc=0.999,
        relative_rmse=0.05,
    )


@DECODE_MESH
@DEVICE_PARAMS
@ARCHITECTURE
@MASKED
@pytest.mark.parametrize("traced", [pytest.param(False, id="untraced"), pytest.param(True, id="traced")])
def test_generate(*, mesh_device: ttnn.MeshDevice, architecture: str, masked: bool, traced: bool) -> None:
    """Teacher-forced decode against the reference's logits at the same positions.

    `guide` feeds the reference tokens back in, so every step's logits line up with the reference's.
    """
    reference = _reference_model(architecture)
    model = _encoder(architecture, reference, mesh_device, sp_axis=None)
    tokens, mask = _prompt(masked=masked)

    with torch.no_grad():
        ref_logits = reference.forward(
            input_ids=tokens, attention_mask=mask if mask is not None else torch.ones_like(tokens)
        ).logits

    generation = model.generate(
        tokens[:, :PROMPT_LENGTH],
        mask=mask[:, :PROMPT_LENGTH] if mask is not None else None,
        max_length=MAX_LENGTH,
        eos_tokens=None,
        guide=tokens,
        return_logits=True,
        traced=traced,
    )

    assert torch.equal(generation.tokens, tokens)
    assert generation.logits is not None
    assert_quality(
        ref_logits[:, PROMPT_LENGTH - 1 : MAX_LENGTH - 1],
        generation.logits,
        pcc=0.999,
        relative_rmse=0.05,
    )
