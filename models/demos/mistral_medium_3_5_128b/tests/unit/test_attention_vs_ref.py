# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite row 4: the whole attention block vs the torch reference.

Recipe row ``gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`` — "QKV proj -> head split -> QK-norm
-> RoPE -> causal SDPA -> o_proj. Same random weights both sides, shared cos/sin so the test
measures attention and not the RoPE constants." QK-norm is absent from this model, so the chain
here is QKV proj -> head split -> RoPE -> causal SDPA -> o_proj.

"Shared cos/sin" is enforced structurally rather than by passing the reference's tensors in: the
device builds its cos/sin from ``reference.modeling.yarn_inv_freq``, the single YaRN
implementation in the package. ``test_rope_mats_vs_ref`` below measures that the device's
Meta-interleaved matrices are the permutation of the reference's HF half-split ones, so a failure
in the RoPE constants shows up there and not as a mystery inside attention.

The block is run in both layouts (see ``tt/attention/prefill.py``):

* ``sequence_parallel=False`` — sequence replicated across the SP rows, plain causal SDPA. No SP
  collective, so a failure is in the attention math itself.
* ``sequence_parallel=True`` — the production layout, gathered SP attention. A failure here with
  the replicated case passing is a collective or a sharding-convention fault.

Full width: 96 Q heads / 8 KV heads / head_dim 128 / hidden 12288, i.e. 24 Q and 2 KV heads per
chip at TP=4.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    MistralAttention,
    MistralYarnRotaryEmbedding,
    causal_mask,
    hf_to_meta,
)
from models.demos.mistral_medium_3_5_128b.tests.device_utils import (
    assert_pcc,
    from_mesh_replicated,
    from_mesh_sp,
    replicate,
    to_mesh,
)
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.mistral_medium_3_5_128b.tt.rope import build_rope_mats

SEQ = 2048


def attention_config(cfg, *, max_seq_len=SEQ, sequence_parallel=False):
    return AttentionConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq_len=max_seq_len,
        rms_norm_eps=cfg.rms_norm_eps,
        sequence_parallel=sequence_parallel,
    )


def reference_attention(cfg, seed=2):
    """A reference attention block with deterministic random weights, plus its HF state dict."""
    torch.manual_seed(seed)
    ref = MistralAttention(cfg, REF_DTYPE).eval()
    state_dict = {
        f"{name}.weight": getattr(ref, name).weight.detach() for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    return ref, state_dict


def reference_cos_sin(cfg, start, end):
    positions = torch.arange(start, end, dtype=torch.int64)[None]
    return MistralYarnRotaryEmbedding(cfg, REF_DTYPE)(positions)


def test_rope_mats_vs_ref(galaxy, mesh_config, cfg):
    """The device's Meta-interleaved cos/sin are the permutation of the reference's HF ones.

    Both sides include the YaRN ``attention_scaling``. Isolating this means an attention PCC
    failure can be attributed to attention rather than to the rotary constants — which matters
    here because this config takes the ``truncate=True`` YaRN branch, the opposite of the borrowed
    gpt-oss source.
    """
    cos_hf, sin_hf = reference_cos_sin(cfg, 0, SEQ)  # [1, SEQ, head_dim]
    expect_cos = hf_to_meta(cos_hf)[None]  # [1, 1, SEQ, head_dim]
    expect_sin = hf_to_meta(sin_hf)[None]

    cos_tt, sin_tt = build_rope_mats(galaxy, cfg, 0, SEQ, mesh_config=mesh_config, sequence_parallel=True)
    assert_pcc("rope_cos", expect_cos, from_mesh_sp(galaxy, cos_tt))
    assert_pcc("rope_sin", expect_sin, from_mesh_sp(galaxy, sin_tt))


@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["replicated", "sp"])
def test_attention_vs_ref(galaxy, mesh_config, ccl, cfg, sequence_parallel):
    """One attention block, full width, against the torch reference."""
    ref, state_dict = reference_attention(cfg)
    torch.manual_seed(3)
    x = (torch.randn(1, SEQ, cfg.hidden_size) * 0.1).to(REF_DTYPE)

    cos, sin = reference_cos_sin(cfg, 0, SEQ)
    with torch.no_grad():
        ref_out, _, _ = ref(x, cos, sin, causal_mask(SEQ, SEQ))
    ref_out = ref_out.unsqueeze(0)  # [1, 1, SEQ, hidden]

    attn = Attention(
        galaxy,
        attention_config(cfg, sequence_parallel=sequence_parallel),
        state_dict,
        ccl,
        mesh_config,
        ProgramConfig(),
        layer_idx=0,
    )
    rope_mats = build_rope_mats(galaxy, cfg, 0, SEQ, mesh_config=mesh_config, sequence_parallel=sequence_parallel)

    x4 = x.unsqueeze(0)
    if sequence_parallel:
        tt_out = attn(to_mesh(galaxy, x4, dims=[-2, None]), rope_mats)
        out = from_mesh_sp(galaxy, tt_out)
    else:
        tt_out = attn(replicate(galaxy, x4), rope_mats)
        out = from_mesh_replicated(galaxy, tt_out)

    assert_pcc(f"attention[{'sp' if sequence_parallel else 'replicated'}]", ref_out, out)
