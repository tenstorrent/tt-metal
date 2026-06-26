# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.tt.denoise_forward import denoise_attention_forward


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeAttention:
    def __init__(self):
        self.kwargs = None

    def __call__(self, hidden_states, **kwargs):
        self.kwargs = kwargs
        return hidden_states


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeAttention()


class _FakeModel:
    def __init__(self):
        self.mesh_device = object()
        self.layers = [_FakeLayer()]

    def _get_rope_mats(self, layer_idx, seq_len):
        return ("cos", "sin")


def test_denoise_attention_defaults_to_maskless_noncausal_prefix_kv():
    model = _FakeModel()
    prompt_kv = (_FakeTensor([1, 1, 64, 16]), _FakeTensor([1, 1, 64, 16]))
    canvas_hidden = _FakeTensor([1, 1, 256, 32])

    out = denoise_attention_forward(
        model,
        layer_idx=0,
        prompt_kv=prompt_kv,
        canvas_hidden=canvas_hidden,
    )

    assert out is canvas_hidden
    kwargs = model.layers[0].self_attn.kwargs
    assert kwargs["is_decode"] is False
    assert kwargs["is_causal"] is False
    assert kwargs["kv_phase"] is KVCachePhase.DENOISE_READONLY
    assert kwargs["attn_mask"] is None
    assert kwargs["prefix_kv"] is prompt_kv
    assert kwargs["q_rope_offset"] == 64
