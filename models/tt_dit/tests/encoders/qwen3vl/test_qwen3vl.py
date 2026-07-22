# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC test for the Qwen3-VL text tower (KREA-2 text encoder) ported to tt_dit.

Builds a REDUCED-layer HF reference (random init, eval mode) with the KREA-2
text_config, loads its state_dict into the tt encoder, and compares hidden states
(final + a couple of tapped intermediate layers) with assert_quality.
"""

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality

# KREA-2 text_config (krea/Krea-2-Turbo, text_encoder/config.json -> text_config)
KREA2_TEXT_CONFIG = dict(
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    attention_bias=False,
    tie_word_embeddings=True,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 5000000,
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
    },
)

# KREA-2 pipeline taps these indices into the HF hidden_states tuple.
KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def _build_hf_text_model(num_hidden_layers: int):
    cfg = transformers.Qwen3VLTextConfig(
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=4096,
        **KREA2_TEXT_CONFIG,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    model = Qwen3VLTextModel(cfg)
    model.eval()
    return model, cfg


@pytest.mark.parametrize(
    ("mesh_device", "num_hidden_layers", "sequence_length"),
    [
        pytest.param((1, 1), 4, 128, id="1x1_L4_s128"),
        pytest.param((1, 1), 8, 128, id="1x1_L8_s128"),
    ],
    indirect=["mesh_device"],
)
def test_qwen3vl_text_encoder(*, mesh_device: ttnn.MeshDevice, num_hidden_layers: int, sequence_length: int) -> None:
    torch.manual_seed(0)

    batch_size = 1
    tp_axis = 1
    tp_factor = mesh_device.shape[tp_axis]

    if tp_factor > 1:
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        )
    else:
        ccl_manager = None
        parallel_config = None

    hf_model, cfg = _build_hf_text_model(num_hidden_layers)

    model = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_parameters["rope_theta"],
        mrope_section=cfg.rope_parameters["mrope_section"],
        mrope_interleaved=cfg.rope_parameters["mrope_interleaved"],
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    model.load_torch_state_dict(hf_model.state_dict())

    tokens = torch.randint(0, cfg.vocab_size, [batch_size, sequence_length])
    attention_mask = None  # unmasked / fully causal path (is_causal=True)

    cos, sin = model.create_rope_tensors(batch_size, sequence_length, attention_mask)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_cos = tensor.from_torch(cos, device=mesh_device)
    tt_sin = tensor.from_torch(sin, device=mesh_device)

    logger.info("running ttnn model...")
    # Returns [emb, layer0, ..., layer_{L-1}, final_norm]; total L+2 entries.
    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=None,
        pos_embeds=(tt_cos, tt_sin),
    )
    tt_hidden_torch = [tensor.to_torch(h) for h in tt_hidden_states]

    logger.info("running torch model...")
    with torch.no_grad():
        out = hf_model.forward(tokens, output_hidden_states=True)
    hf_hidden = out.hidden_states  # tuple length L+1 (index 0 = emb, index k = pre-norm layer k out)
    hf_last = out.last_hidden_state  # final-norm output

    # tt_hidden layout: [0..L] mirror hf_hidden[0..L]; tt_hidden[L+1] mirrors hf_last.
    assert len(tt_hidden_torch) == len(hf_hidden) + 1

    pccs = []
    # Compare embedding + each intermediate (pre-norm) layer output.
    for i in range(len(hf_hidden)):
        ref = hf_hidden[i]
        got = tt_hidden_torch[i]
        try:
            assert_quality(ref, got, pcc=0.98)
            status = "ok"
        except Exception as err:  # noqa: BLE001
            status = f"FAIL ({err})"
        # recompute raw pcc for the report
        import torch as _t

        a = ref.float().flatten()
        b = got.float().flatten()
        pcc = _t.corrcoef(_t.stack([a, b]))[0, 1].item()
        pccs.append((f"hidden[{i}]", pcc, status))

    # final-norm output
    a = hf_last.float().flatten()
    b = tt_hidden_torch[-1].float().flatten()
    import torch as _t

    final_pcc = _t.corrcoef(_t.stack([a, b]))[0, 1].item()

    logger.info("=== Qwen3-VL text encoder PCC report (num_layers={}) ===", num_hidden_layers)
    for name, pcc, status in pccs:
        logger.info("  {}: pcc={:.5f} {}", name, pcc, status)
    logger.info("  final_norm(last_hidden_state): pcc={:.5f}", final_pcc)

    # Assert final-norm output and every intermediate tap meet the target.
    assert_quality(hf_last, tt_hidden_torch[-1], pcc=0.98)
    for i in range(1, len(hf_hidden)):
        assert_quality(hf_hidden[i], tt_hidden_torch[i], pcc=0.98)
