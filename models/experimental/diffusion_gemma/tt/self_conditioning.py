# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device (ttnn) self-conditioning gated MLP — the net-new diffusion weight module (#47461 loader / #47463 runtime).

Device mirror of ``reference/self_conditioning.py`` (the pure-torch oracle) and of
transformers ``DiffusionGemmaSelfConditioning``::

    forward = post_norm(inputs_embeds + down(gelu_tanh(gate(pre_norm(signal))) * up(pre_norm(signal))))

- ``pre_norm``  — scaled RMSNorm (`model.decoder.self_conditioning.pre_norm.weight`)
- ``post_norm`` — **scaleless** RMSNorm (no checkpoint weight; absent by design)
- gate/up/down  — bias-free linears, gemma4 GeGLU pattern (`tt/shared_mlp.py`)

RMSNorm uses the weight **directly** (NOT the Gemma2/3 ``1+weight`` convention) —
matches both `ttnn.rms_norm` and the reference, so weights load verbatim.

The module is small (2816→2112→2816) and is kept replicated for the current QB2
integration path. Weights come from a ``weight_mapping.remap_state_dict``
self-conditioning sub-dict (short keys ``{pre_norm,gate_proj,up_proj,down_proj}.weight``).
Validated on QB2 vs the reference oracle by ``tests/test_device_self_conditioning.py`` and
as part of the mesh denoise logits wrapper in ``tests/test_device_bidirectional_attention_integration.py``.
"""

from __future__ import annotations

import ttnn

from models.experimental.diffusion_gemma.weight_mapping import expected_self_conditioning_shapes


def _config_value(config, name: str):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def validate_self_conditioning_state(state_dict, *, hidden_size: int, intermediate_size: int) -> None:
    """Validate remapped self-conditioning weights before moving them to device."""
    expected = expected_self_conditioning_shapes(hidden_size, intermediate_size)
    missing = sorted(set(expected) - set(state_dict))
    if missing:
        raise ValueError(f"missing self-conditioning weights: {missing}")
    for key, shape in expected.items():
        if tuple(state_dict[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(state_dict[key].shape)}, expected {shape}")


def build_self_conditioning(
    device,
    state_dict,
    *,
    config=None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    eps: float | None = None,
    dtype=ttnn.bfloat16,
    module_cls=None,
):
    """Build ``TtSelfConditioning`` from remapped checkpoint weights and config."""
    if config is not None:
        hidden_size = hidden_size if hidden_size is not None else _config_value(config, "hidden_size")
        intermediate_size = (
            intermediate_size if intermediate_size is not None else _config_value(config, "intermediate_size")
        )
        eps = eps if eps is not None else _config_value(config, "rms_norm_eps")
    if hidden_size is None or intermediate_size is None:
        raise ValueError("hidden_size and intermediate_size are required")
    eps = 1e-6 if eps is None else eps
    validate_self_conditioning_state(state_dict, hidden_size=hidden_size, intermediate_size=intermediate_size)
    cls = TtSelfConditioning if module_cls is None else module_cls
    return cls(
        device,
        state_dict,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        eps=eps,
        dtype=dtype,
    )


class TtSelfConditioning:
    def __init__(
        self,
        device,
        state_dict,
        *,
        hidden_size,
        intermediate_size,
        eps=1e-6,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.eps = eps
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # scaled pre_norm weight: ROW_MAJOR, bf16, [1,1,hidden/32,32] (gemma4 RMSNorm layout).
        pre_w = state_dict["pre_norm.weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        self.pre_norm_weight = ttnn.as_tensor(
            pre_w,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # post_norm is scaleless — no checkpoint weight, no tensor built.

        # gate/up/down linears: HF [out,in] -> [1,1,in,out], TILE, DRAM.
        def _lin(key):
            w = state_dict[key].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = _lin("gate_proj.weight")
        self.up_proj = _lin("up_proj.weight")
        self.down_proj = _lin("down_proj.weight")

    def forward(self, inputs_embeds_tt, signal_tt):
        """``inputs_embeds_tt`` / ``signal_tt``: ``[1,1,L,hidden]`` TILE_LAYOUT.

        Zero signal -> ``post_norm(inputs_embeds)`` (NOT inputs_embeds), matching the
        decoder: it always post-normalizes its input embeddings.
        """
        normed = ttnn.rms_norm(signal_tt, weight=self.pre_norm_weight, epsilon=self.eps)

        gate = ttnn.linear(normed, self.gate_proj)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)  # gelu_pytorch_tanh
        up = ttnn.linear(normed, self.up_proj)
        normed.deallocate(True)

        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        sc = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        summed = ttnn.add(inputs_embeds_tt, sc)
        sc.deallocate(True)
        out = ttnn.rms_norm(summed, epsilon=self.eps)  # scaleless post_norm
        summed.deallocate(True)
        return out

    def soft_embedding(self, prev_logits_tt, embedding_weight_tt, *, compute_kernel_config=None):
        """Probability-weighted token embedding from prev-step logits — the decoder's
        soft-embedding step (modeling: ``softmax(logits, dim=-1) @ embed_tokens.weight``).

        ``prev_logits_tt`` ``[1,1,L,vocab]`` (TILE), ``embedding_weight_tt`` the tied
        table ``[1,1,vocab,hidden]`` (TILE). Returns the signal ``[1,1,L,hidden]``.
        For the production vocab (262144) drive ``softmax`` with an fp32
        ``compute_kernel_config`` (bf16 over a 262k-wide reduction is lossy — see the
        bfp8 entropy drift); a moderate vocab is fine in bf16.
        """
        if compute_kernel_config is not None:
            probs = ttnn.softmax(
                prev_logits_tt, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config
            )
        else:
            probs = ttnn.softmax(prev_logits_tt, dim=-1)
        signal = ttnn.matmul(probs, embedding_weight_tt)  # [1,1,L,vocab] @ [1,1,vocab,hidden] -> [1,1,L,hidden]
        probs.deallocate(True)
        # canonical: * embed_scale = hidden_size**0.5 (the tied embedding's scale). The pre_norm eps
        # floor does NOT absorb this at the tiny soft-RMS of a 262k-vocab softmax, so it is load-bearing.
        scaled = ttnn.multiply(signal, float(self.hidden_size) ** 0.5)
        signal.deallocate(True)
        return scaled

    def condition(self, inputs_embeds_tt, prev_logits_tt, embedding_weight_tt, *, compute_kernel_config=None):
        """Full self-conditioning step: soft-embed prev logits, then apply the module
        (mirrors the reference ``SelfConditioning.condition`` / decoder forward).

        ``prev_logits_tt is None`` (first step / encoder pass) -> zero signal, so the
        result is ``post_norm(inputs_embeds)``.
        """
        if prev_logits_tt is None:
            signal = ttnn.mul(inputs_embeds_tt, 0.0)  # zeros, same shape/layout/dtype
        else:
            signal = self.soft_embedding(
                prev_logits_tt, embedding_weight_tt, compute_kernel_config=compute_kernel_config
            )
        out = self.forward(inputs_embeds_tt, signal)
        signal.deallocate(True)
        return out
