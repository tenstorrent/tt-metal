# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Greedy decode with KV cache: hybrid + LM head vs HF ``DynamicCache`` loop.

Device SDPA + greedy **token ids** are not asserted against HF (bf16 SDPA vs eager can change argmax);
see ``test_causal_lm_incremental_teacher_forced_device_sdpa_logits_match_hf`` for SDPA parity on logits
with a fixed token stream.
"""

import pytest
import torch

from models.tt_transformers.tt.mistral_small_4.causal_lm_incremental import (
    causal_lm_greedy_generate_hf_reference_torch,
    causal_lm_greedy_generate_hybrid_bf16,
)


def _mini_mistral4_causal_lm_all_dense():
    """Two dense layers — avoids uninitialized MoE experts skewing greedy logits."""
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=2,
        max_position_embeddings=4096,
        rope_interleave=False,
        rope_parameters={
            "type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_causal_lm_greedy_generate_hybrid_matches_hf_reference(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    torch.manual_seed(47)
    cfg = _mini_mistral4_causal_lm_all_dense()
    cfg._attn_implementation = "eager"
    model = Mistral4ForCausalLM(cfg).to(torch.bfloat16).eval()

    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 5), dtype=torch.long)
    max_new_tokens = 4

    expected = causal_lm_greedy_generate_hf_reference_torch(prompt_ids, model, max_new_tokens, attention_mask_2d=None)
    out = causal_lm_greedy_generate_hybrid_bf16(
        mesh_device,
        prompt_ids,
        model,
        max_new_tokens,
        attention_mask_2d=None,
        use_device_sdpa_attention=False,
    )
    assert torch.equal(out, expected), f"hybrid {out} != hf {expected}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_causal_lm_incremental_teacher_forced_device_sdpa_logits_match_hf(mesh_device, reset_seeds):
    """
    Device SDPA greedy ids can diverge from HF (bf16 SDPA vs eager → different argmax).
    Replay the **HF greedy** token stream and check logits match at prefill + each decode step.

    Expect **modest PCC** (~0.5) vs HF on full vocabulary logits — same situation as other device-SDPA
    causal-LM parity tests (LM head amplifies attention drift).
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    from models.common.utility_functions import comp_allclose, comp_pcc
    from models.tt_transformers.tt.mistral_small_4.causal_lm_incremental import (
        causal_lm_greedy_generate_hf_reference_torch,
        causal_lm_incremental_step_logits_hybrid_bf16,
    )

    torch.manual_seed(48)
    cfg = _mini_mistral4_causal_lm_all_dense()
    cfg._attn_implementation = "eager"
    model = Mistral4ForCausalLM(cfg).to(torch.bfloat16).eval()

    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)
    max_new_tokens = 3
    p_len = int(prompt_ids.shape[1])

    seq_hf = causal_lm_greedy_generate_hf_reference_torch(prompt_ids, model, max_new_tokens, attention_mask_2d=None)

    past_tt = None
    logits_tt, past_tt = causal_lm_incremental_step_logits_hybrid_bf16(
        mesh_device,
        prompt_ids,
        model,
        past_key_values=past_tt,
        attention_mask_2d=None,
        use_device_sdpa_attention=True,
    )

    past_hf = DynamicCache(config=model.config)
    with torch.no_grad():
        out_hf = model(input_ids=prompt_ids, past_key_values=past_hf, use_cache=True)
    logits_hf = out_hf.logits

    # Stacked layers + ``lm_head`` + device SDPA vs HF eager yields modest logits PCC on Blackhole
    # (same order as other ``mistral_small_4`` full-logits checks; ~0.52 observed).
    ok, msg = comp_pcc(logits_hf, logits_tt, pcc=0.50)
    assert ok, msg
    close, amsg = comp_allclose(logits_hf, logits_tt, rtol=2.5, atol=1.2)
    assert close, amsg

    for i in range(max_new_tokens):
        inp = seq_hf[:, p_len + i : p_len + i + 1]
        logits_tt, past_tt = causal_lm_incremental_step_logits_hybrid_bf16(
            mesh_device,
            inp,
            model,
            past_key_values=past_tt,
            attention_mask_2d=None,
            use_device_sdpa_attention=True,
        )
        with torch.no_grad():
            out_hf = model(input_ids=inp, past_key_values=past_hf, use_cache=True)
        logits_hf = out_hf.logits

        ok_i, msg_i = comp_pcc(logits_hf, logits_tt, pcc=0.50)
        assert ok_i, f"decode step {i}: {msg_i}"
        close_i, amsg_i = comp_allclose(logits_hf, logits_tt, rtol=2.5, atol=1.2)
        assert close_i, f"decode step {i}: {amsg_i}"
