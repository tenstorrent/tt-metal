# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Real CosyVoice2-0.5B checkpoint weights (`llm.pt`, from
`FunAudioLLM/CosyVoice2-0.5B`) loaded into `TtQwen2LM` -- both the 24-layer
Qwen2 backbone AND the CosyVoice-specific tables (`speech_embedding`,
`llm_embedding`, `llm_decoder`), which `test_qwen2lm.py`'s existing tests
leave random-init (no checkpoint existed when that file was written). See
`tt/checkpoint.py`'s `build_local_qwen2_checkpoint_dir` for how the backbone
is loaded (reuses `ModelArgs`' existing, already-validated HF-loading path
rather than a new low-level loader) and `TtQwen2LM.__init__`'s
`cosyvoice_state_dict` param for the CosyVoice-specific tables (direct
1:1 key match, no remapping needed).

Fourth and last module in this bring-up's checkpoint-loading order (HiFT
vocoder, F0 predictor, flow decoder, done -- then LLM backbone, here),
highest risk by design: this is the one module already confirmed, by direct
tensor diff against public `Qwen/Qwen2-0.5B-Instruct`, to be a genuinely
different fine-tuned checkpoint (same architecture and shapes, `allclose`
false everywhere, cosine similarity 0.17-0.99) -- `test_qwen2lm.py`'s
existing tests, which use the public checkpoint, were always going to be a
weaker check than this for correctness, even though they were a stronger
check than fully-random init.

Uses `GATE_DECODE = 0.94`, not `GATE = 0.99`, for the same reason
`test_qwen2lm.py`'s own multistep-decode test does -- see that file's module
docstring ("GATE vs. GATE_DECODE"): this is real multi-step decode at full
(24-layer) depth with a bf16 KV cache, and 0.94 is tt_transformers' own
field-tested bar for exactly that scenario, not a loosened threshold invented
here.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE = 0.99
GATE_DECODE = 0.94  # see module docstring


def _build_real_args_and_state_dict(mesh_device, tmp_path, max_seq_len=256):
    from models.demos.audio.cosyvoice2.tt.checkpoint import build_local_qwen2_checkpoint_dir, load_checkpoint_file
    from models.tt_transformers.tt.model_config import ModelArgs

    llm_sd = load_checkpoint_file("llm.pt")
    local_dir = build_local_qwen2_checkpoint_dir(llm_sd, str(tmp_path / "qwen2_ckpt"))
    os.environ["HF_MODEL"] = local_dir

    args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()
    return args, state_dict, llm_sd


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_cosyvoice_table_shapes_match_real_checkpoint():
    """`speech_embedding`/`llm_embedding`/`llm_decoder` shapes in real `llm.pt`
    match `head_out_features = speech_token_size + 3 = 6564` (this package's
    existing assumption), checked directly against real trained tensors, not
    just the architecture comment."""
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

    llm_sd = load_checkpoint_file("llm.pt")
    assert llm_sd["speech_embedding.weight"].shape == (6564, 896)
    assert llm_sd["llm_embedding.weight"].shape == (2, 896)
    assert llm_sd["llm_decoder.weight"].shape == (6564, 896)
    assert llm_sd["llm_decoder.bias"].shape == (6564,)


def test_real_backbone_differs_from_public_qwen2_instruct():
    """The finding this bring-up's checkpoint-loading order was built around,
    checked fresh here rather than only in a prior session's now-lost
    analysis: real `llm.pt`'s embedded Qwen2 backbone (`llm.model.model.*`)
    is architecture-identical to but numerically different from public
    `Qwen/Qwen2-0.5B-Instruct` -- same shape, `allclose` false, moderate
    cosine similarity (not ~1.0, not ~0.0 -- a genuine fine-tune, not a
    random/unrelated model)."""
    from transformers import AutoModelForCausalLM

    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

    llm_sd = load_checkpoint_file("llm.pt")
    public = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype=torch.float32)
    public_sd = public.state_dict()

    a = llm_sd["llm.model.model.embed_tokens.weight"].float()
    b = public_sd["model.embed_tokens.weight"].float()
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-6)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    assert 0.1 < cos < 0.999, cos  # neither identical nor unrelated


# --------------------------------------------------------------------------
# device tier -- needs silicon (downloads llm.pt, 2.02 GB, and public
# Qwen/Qwen2-0.5B-Instruct is NOT needed here, unlike test_qwen2lm.py)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_backbone_matches_real_checkpoint_multistep_decode(device, tmp_path):
    """The real-checkpoint counterpart of `test_qwen2lm.py`'s
    `test_device_backbone_matches_real_qwen2_multistep_decode` -- SAME
    populated-KV-cache, 6-sequential-steps setup, just built from real
    `llm.pt` weights (via a local HF-format checkpoint dir) instead of public
    `Qwen/Qwen2-0.5B-Instruct`."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM
    from models.tt_transformers.tt.common import Mode, precompute_freqs
    from models.tt_transformers.tt.model_config import HfDecoderWrapper

    args, state_dict, llm_sd = _build_real_args_and_state_dict(device, tmp_path)
    hf_model = args.reference_transformer(wrap=False)
    tt_model = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd)

    torch.manual_seed(99)
    n_steps = 6
    tokens = [torch.randn(1, 1, args.dim, dtype=torch.bfloat16).float() * 0.1 for _ in range(n_steps)]

    cos, sin = precompute_freqs(
        args.head_dim,
        args.max_seq_len * 2,
        args.rope_theta,
        args.rope_scaling.factor if args.rope_scaling else None,
        args.rope_scaling.original_max_position_embeddings if args.rope_scaling else None,
        args.rope_scaling.rope_type.value if args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    wrappers = [
        HfDecoderWrapper(layer, args.head_dim, hf_model.model.rotary_emb, use_hf_rope=args.use_hf_rope)
        for layer in hf_model.model.layers[: args.n_layers]
    ]
    want = None
    with torch.no_grad():
        for pos in range(n_steps):
            h = tokens[pos].bfloat16()
            freqs_i = freqs_cis[pos, :].unsqueeze(0)
            for wrapper in wrappers:
                h = wrapper(h, pos, freqs_i, mask=None)
                if h.dim() == 2:
                    h = h.unsqueeze(1)
            if pos == n_steps - 1:
                want = hf_model.model.norm(h)

    norm_config = args.get_norm_config("lm_head", Mode.DECODE, None)
    got = None
    for pos in range(n_steps):
        x = tokens[pos]
        decode_input = args.prepare_residual_tensor_decode(x, args.get_residual_mem_config(Mode.DECODE, None))
        current_pos = torch.tensor([pos])
        current_pos_tensor = ttnn.from_torch(current_pos, device=device, dtype=ttnn.int32)
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
        x_tt = decode_input
        for layer in tt_model.layers:
            x_tt = layer(x_tt, current_pos_tensor, rot_mats_global=rot_mats, mode=Mode.DECODE)
        if pos == n_steps - 1:
            x_tt = tt_model.norm(x_tt, mode=Mode.DECODE, norm_config=norm_config)
            got = ttnn.to_torch(x_tt).float()
            got = got[:, :, :1, : args.dim].reshape(want.shape)

    passed, pcc = comp_pcc(want.float(), got, GATE_DECODE)
    print(f"\n  real-checkpoint 24-layer backbone + norm decode (position {n_steps - 1}) PCC {pcc}")
    assert passed, pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_device_generate_smoke_real_checkpoint(device, tmp_path):
    """`generate()`'s real control-flow loop (prefill then several sequential
    decode steps, same cache continuing across all of them) with the real
    CosyVoice2 backbone AND real speech_embedding/llm_embedding/llm_decoder --
    the real-checkpoint counterpart of `test_qwen2lm_generate.py`'s
    `test_device_generate_smoke`. Still no golden speech-token sequence to
    check identity against (that needs a real prompt pipeline this bring-up
    hasn't wired up yet), so this checks the loop holds together end to end
    with every real weight this module has, not token-level correctness."""
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    args, state_dict, llm_sd = _build_real_args_and_state_dict(device, tmp_path)
    tt_model = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd)

    torch.manual_seed(3)
    text_ids = torch.randint(0, args.vocab_size, (1, 4))

    for sampler in ("greedy", "ras", "ras_device"):
        out = tt_model.generate(text_ids, max_tokens=4, sampler=sampler, seed=0)
        assert len(out) <= 4
        assert all(0 <= t < tt_model.head_out_features for t in out)
