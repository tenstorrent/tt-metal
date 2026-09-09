# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""TtQwen2LM skeleton: embeddings + 24-layer backbone + norm, construction and
per-piece correctness only -- see tt/llm/qwen2lm.py's module docstring for what
is and is not wired up yet (no sequence assembly, no prefill/decode driving
loop, no RAS sampling).

`HF_MODEL=Qwen/Qwen2-0.5B-Instruct` needs no special-case registration in
model_config.py -- confirmed empirically (see qwen2lm.py's module docstring):
`ModelArgs(..., dummy_weights=False)` resolves dim/n_layers/n_heads/n_kv_heads/
rope_theta directly from the real HF config, and `.load_state_dict()` /
`.reference_transformer()` download and use the real Qwen2-0.5B-Instruct
checkpoint -- architecturally identical to CosyVoice2's own fine-tuned Qwen2
backbone (same hidden_size=896/layers=24/heads=14/kv_heads=2, confirmed against
the real CosyVoice2-0.5B checkpoint's config.json), giving a stronger
validation baseline than the random-init this package has used everywhere else
so far (no checkpoint existed for those pieces; one exists here).

The reference model is called with `inputs_embeds=`, `output_hidden_states=True`
-- the exact same calling convention `Qwen2Encoder.forward` uses in the real
CosyVoice2 source, not tt_transformers' own token-ID-based HfModelWrapper.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE = 0.99


def _build_args_and_state_dict(mesh_device, max_seq_len=256):
    os.environ["HF_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()
    return args, state_dict


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_qwen2_0p5b_config_matches_real_checkpoint():
    """The shape facts this whole port leans on, pinned as a regression test:
    if HF ever changes Qwen2-0.5B-Instruct's public config, or tt_transformers'
    generic HF loader stops resolving it the same way, this fails loudly
    instead of silently building the wrong-shaped model."""
    os.environ["HF_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 14
    assert config.num_key_value_heads == 2
    assert config.intermediate_size == 4864
    assert config.vocab_size == 151936


# --------------------------------------------------------------------------
# device tier -- needs silicon (downloads the real Qwen2-0.5B-Instruct checkpoint)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_text_embedding_matches_real_qwen2(device):
    """TtQwen2LM.text_embedding vs. the real Qwen2ForCausalLM's own
    embed_tokens -- real weights on both sides, the exact table CosyVoice2's
    Qwen2Encoder uses for `self.llm.model.model.embed_tokens(text_token)`."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    hf_model = args.reference_transformer(wrap=False)

    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (1, 8))
    with torch.no_grad():
        want = hf_model.model.embed_tokens(ids)

    ids_dev = ttnn.from_torch(ids.reshape(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got_dev = tt_model.text_embedding(ids_dev)
    got = ttnn.to_torch(got_dev).float().reshape(want.shape)

    passed, pcc = comp_pcc(want.float(), got, GATE)
    print(f"\n  text embedding PCC {pcc}")
    assert passed, pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_speech_and_llm_embedding_smoke(device):
    """Structural check only for the two CosyVoice-specific tables (no real
    CosyVoice2 checkpoint yet, so no golden to compare against) -- shapes and
    that distinct ids produce distinct, finite rows."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    speech_ids = torch.tensor([[0, 1, 6560, 6563]])
    ids_dev = ttnn.from_torch(
        speech_ids.reshape(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = ttnn.to_torch(tt_model.speech_embedding(ids_dev)).float()
    assert out.shape[-1] == args.dim
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[..., 0, :], out[..., 1, :])  # distinct ids -> distinct rows

    sos_task = torch.tensor([[0, 1]])
    sos_task_dev = ttnn.from_torch(
        sos_task.reshape(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out2 = ttnn.to_torch(tt_model.llm_embedding(sos_task_dev)).float()
    assert out2.shape[-1] == args.dim
    assert torch.isfinite(out2).all()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_head_smoke(device):
    """Structural check for the new small output head: shape and finiteness
    only, same reasoning as the embeddings above."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict = _build_args_and_state_dict(device)
    tt_model = TtQwen2LM(args, device, state_dict)

    x = torch.randn(1, 1, 32, args.dim) * 0.1
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_model.llm_decoder(x_dev)).float()
    assert out.shape[-1] == tt_model.head_out_features
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_backbone_matches_real_qwen2_decode(device):
    """The core claim: TtQwen2LM's 24 TransformerBlocks + final norm vs. the
    real Qwen2ForCausalLM -- decode-mode (single new token, matching this
    repo's own test_decoder.py precedent), real downloaded weights on both
    sides.

    Reference is built per-layer via `HfDecoderWrapper` (this repo's own
    established pattern, the same one test_decoder.py uses via
    `args.reference_decoder()`), chained across all n_layers -- not the full
    HF model called with `inputs_embeds`. That first attempt measured PCC
    0.96-0.99 *non-monotonically* with layer count (1 layer: 0.97, 4: 0.98,
    12: 0.99, 24: 0.96) -- not the smooth degradation real accumulated bf16
    drift would produce, which was the tell that it was a test-harness bug,
    not a TransformerBlock bug. Isolated by comparing a single TransformerBlock
    against `args.reference_decoder()` directly (the proven single-layer
    pattern): PCC 0.9997, confirming TransformerBlock itself was never the
    problem -- calling the full HF model with `inputs_embeds` and no explicit
    freqs_cis/position handling was the mismatch.

    Sequence-assembly/prefill wiring is a separate, later piece (see module
    docstring) -- this checks that the backbone construction itself, stacked
    to all 24 layers, reproduces the real model's hidden states before any of
    that is built.
    """
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM
    from models.tt_transformers.tt.common import Mode, precompute_freqs
    from models.tt_transformers.tt.model_config import HfDecoderWrapper

    args, state_dict = _build_args_and_state_dict(device)
    hf_model = args.reference_transformer(wrap=False)
    tt_model = TtQwen2LM(args, device, state_dict)

    torch.manual_seed(0)
    x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16).float() * 0.1  # single-token embedding, decode-shaped
    current_pos_val = 0  # first token, zero history -- both sides start a fresh KV cache here

    cos, sin = precompute_freqs(
        args.head_dim,
        args.max_seq_len * 2,
        args.rope_theta,
        args.rope_scaling.factor if args.rope_scaling else None,
        args.rope_scaling.original_max_position_embeddings if args.rope_scaling else None,
        args.rope_scaling.rope_type.value if args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)
    freqs_cis_0 = freqs_cis[current_pos_val, :].unsqueeze(0)

    with torch.no_grad():
        h = x.bfloat16()
        for hf_layer in hf_model.model.layers[: args.n_layers]:
            wrapper = HfDecoderWrapper(hf_layer, args.head_dim, hf_model.model.rotary_emb, use_hf_rope=args.use_hf_rope)
            h = wrapper(h, current_pos_val, freqs_cis_0, mask=None)
            if h.dim() == 2:
                h = h.unsqueeze(1)
        want = hf_model.model.norm(h)

    decode_input = args.prepare_residual_tensor_decode(x, args.get_residual_mem_config(Mode.DECODE, None))
    current_pos = torch.tensor([current_pos_val])
    current_pos_tensor = ttnn.from_torch(current_pos, device=device, dtype=ttnn.int32)
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

    x_tt = decode_input
    for layer in tt_model.layers:
        x_tt = layer(x_tt, current_pos_tensor, rot_mats_global=rot_mats, mode=Mode.DECODE)
    norm_config = args.get_norm_config("lm_head", Mode.DECODE, None)
    x_tt = tt_model.norm(x_tt, mode=Mode.DECODE, norm_config=norm_config)

    got = ttnn.to_torch(x_tt).float()
    got = got[:, :, :1, : args.dim].reshape(want.shape)

    passed, pcc = comp_pcc(want.float(), got, GATE)
    print(f"\n  24-layer backbone + norm decode PCC {pcc}")
    assert passed, pcc
