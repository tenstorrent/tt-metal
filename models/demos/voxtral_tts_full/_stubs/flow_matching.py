# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TT-NN port of `VoxtralFlowMatching` (Voxtral-TTS Block 2): hidden -> 37 audio codes.

Reference: `voxtral_flow_ref.reference_frame` = `semantic_code` ++ `decode_frame`.

    h [B, 3072]
      -> semantic_codebook_output -> mask -> argmax                = semantic code [B, 1]
      -> 7 Euler steps of a 3-layer bidirectional transformer with
         classifier-free guidance                                  = acoustic floats [B, 36]
      -> clamp(-1, 1) -> rescale to 21 FSQ levels -> round         = acoustic codes  [B, 36]
      = audio_codes [B, 37], with the +N_AUDIO_SPECIAL offset applied

The transformer sees a sequence of exactly THREE tokens -- input_projection(x_t),
time_projection(t_emb), llm_projection(h) -- and the velocity is read off position 0 only.
Attention is bidirectional and unmasked: no RoPE, no causal mask. `rope_theta` in params.json
is inert for this module.

CFG IS ONE BATCH-2 FORWARD, not two graphs: the conditional and unconditional halves are
stacked on the batch axis and combined afterwards as `alpha*v_cond + (1-alpha)*v_uncond`. The
unconditional half zeroes the LLM conditioning, and llm_projection is bias-free, so its third
token is exactly zero -- staged once at build rather than recomputed.

WHAT IS PRECOMPUTED, AND WHY IT IS EXACT. The step count, the timestep grid and the time
embedding are all fixed by the model: 7 Euler steps over `linspace(0, 1, 8)`, so t_i = i/7 and
dt = 1/7, and `time_embedding(t)` depends on nothing but t. The middle token is therefore the
same tensor on every call, and `time_projection` of it likewise -- both are staged at build,
which keeps the per-step work to the parts that actually vary. `time_embedding.inv_freq` is
absent from the released checkpoint and recomputed by the reference's own formula; the same
formula is used here.

X_0 IS AN INITIAL CONDITION, NOT A WEIGHT. Real inference draws it from a Gaussian. A host
tensor cannot be marshalled inside the forward without putting host work on the compute path,
so this port integrates from a staged zero unless the caller hands it a tensor ALREADY on
device, which it will then use. The PCC test passes the matching zero (see the harness note in
tests/pcc/conftest.py), so the two integrate the same trajectory.

CFG_ALPHA. The reference's default is 1.2 and `VoxtralFlowMatching.forward` only forwards a
cfg_alpha it was actually given, so 1.2 -- not config.json's `flow_cfg_alpha: 1.3` -- is what
this block runs unless a caller overrides it. `build(cfg_alpha=...)` follows the same default.

PRECISION IS A KNOB HERE BECAUSE THE OUTPUT IS QUANTISED. `build(dtype=...)` selects the staged
weight/activation precision; bfloat16 is the default and what the per-component PCC test runs.
The e2e pipeline builds this block with `ttnn.float32`, and the reason is specific to Block 2:
its 36 acoustic floats are ROUNDED onto 21 FSQ levels, so accuracy is only useful up to the
nearest rounding boundary and useless past it. Measured on the reference's own trajectories, a
few of the 36 dimensions per frame land within 0.005 of a boundary (in the scaled 0..20 units),
i.e. within 5e-4 of x -- and bfloat16 carries ~4e-3 of relative error, so those dimensions flip
a code on arithmetic noise alone. A flipped code is not a small error downstream: it shifts a
latent by a full 1/20th of its range, and the waveform PCC collapses rather than degrades. fp32
puts three orders of margin between the arithmetic and the boundary. The block is 390M
parameters, so the extra ~0.8 GB is affordable where it would not be for the 3.4B backbone.
"""
from __future__ import annotations

import math

import torch
import ttnn

# One compute config for the whole model port; see `_stubs/attention.py` for the measurement.
# fp32 destination accumulation matters most for `ttnn.rms_norm`, which without it returns an
# output ~2% SHORT of the reference -- and this block runs its 3 layers 7 times per frame, so
# a per-layer scale error is applied 21 times before the codes are read out.
from models.demos.voxtral_tts_full._stubs.attention import (
    linear,
    matmul,
    softmax,
    stage_weight,
)
from models.demos.voxtral_tts_full._stubs.attention import rms_norm as _rms_norm

# From voxtral_common_ref / the checkpoint's params.json.
_DIM = 3072
_N_LAYERS = 3
_N_HEADS = 32
_N_KV_HEADS = 8
_HEAD_DIM = 128
_NORM_EPS = 1e-5
_TIME_THETA = 10000.0
_N_STEPS = 7
_CFG_ALPHA = 1.2
_N_ACOUSTIC = 36
_ACOUSTIC_LEVELS = 21
_N_AUDIO_SPECIAL = 2
_EMPTY_AUDIO_ID = 0
_END_AUDIO_ID = 1
_SEMANTIC_CODEBOOK_SIZE = 8192

# Forbidden semantic codes are removed with an additive bias rather than by writing -inf into
# the logits, which keeps the masking on device. -1e9 underflows exp()/loses argmax exactly the
# way -inf does.
_MASK_FILL = -1e9


class TtVoxtralFlowMatching:
    def __init__(self, device, tensors, cfg_alpha, n_steps):
        self.device = device
        self.__dict__.update(tensors)
        self.cfg_alpha = cfg_alpha
        self.n_steps = n_steps

    @classmethod
    def build(
        cls,
        device,
        torch_module,
        cfg_alpha: float = _CFG_ALPHA,
        n_steps: int = _N_STEPS,
        batch: int = 1,
        dtype=ttnn.bfloat16,
    ):
        w = {k: v.detach().float() for k, v in torch_module._as_dict().items()}
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

        def stage(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous().to(torch_dtype), dtype=dtype, layout=layout, device=device)

        # Matmul operands go through `stage_weight`, which at ttnn.float32 stages the hi/lo
        # bfloat16 pair (see `_stubs/attention.py`): the FPU keeps only ~11 mantissa bits of a
        # native fp32 operand, and this block's output is ROUNDED, so the bits it drops are the
        # ones that decide a code. Norm vectors, masks and the staged tokens are elementwise
        # only -- bit-exact in fp32 -- so they stay plain tensors.
        mm_weight = lambda t: stage_weight(device, t, dtype)  # noqa: E731

        def layer(prefix):
            return {
                "attn_norm": stage(w[prefix + "attention_norm.weight"]),
                "wq": mm_weight(w[prefix + "attention.wq.weight"].t()),
                "wk": mm_weight(w[prefix + "attention.wk.weight"].t()),
                "wv": mm_weight(w[prefix + "attention.wv.weight"].t()),
                "wo": mm_weight(w[prefix + "attention.wo.weight"].t()),
                "ffn_norm": stage(w[prefix + "ffn_norm.weight"]),
                "w1": mm_weight(w[prefix + "feed_forward.w1.weight"].t()),
                "w2": mm_weight(w[prefix + "feed_forward.w2.weight"].t()),
                "w3": mm_weight(w[prefix + "feed_forward.w3.weight"].t()),
            }

        # `semantic_code`: [EMPTY_AUDIO] is forbidden ([END_AUDIO] is allowed -- that is how
        # generation stops) and so is everything past the real codebook.
        n_semantic_out = w["semantic_codebook_output.weight"].shape[0]
        mask = torch.zeros(1, n_semantic_out)
        mask[:, _EMPTY_AUDIO_ID] = _MASK_FILL
        mask[:, _N_AUDIO_SPECIAL + _SEMANTIC_CODEBOOK_SIZE :] = _MASK_FILL

        # `time_embedding`: cat(cos, sin) -- that order -- of t * inv_freq, then projected.
        # Fixed per step, so both are folded into one staged token per step.
        inv_freq = w.get("time_embedding.inv_freq")
        if inv_freq is None:
            half = _DIM // 2
            inv_freq = torch.exp(-math.log(_TIME_THETA) * torch.arange(half).float() / half)
        steps = torch.linspace(0, 1, n_steps + 1)
        time_tokens = []
        for i in range(n_steps):
            angles = steps[i].reshape(1, 1) @ inv_freq.reshape(1, -1)
            half_dim = int(angles.shape[-1])
            embedded = torch.zeros(1, 2 * half_dim, dtype=angles.dtype)
            embedded[0, :half_dim] = angles.cos().reshape(-1)
            embedded[0, half_dim:] = angles.sin().reshape(-1)
            projected = embedded @ w["time_projection.weight"].t()
            time_tokens.append(stage(projected.reshape(1, 1, _DIM).repeat(2 * batch, 1, 1)))

        tensors = {
            "w_semantic": mm_weight(w["semantic_codebook_output.weight"].t()),
            "semantic_mask": stage(mask),
            "w_input": mm_weight(w["input_projection.weight"].t()),
            "w_llm": mm_weight(w["llm_projection.weight"].t()),
            "w_acoustic": mm_weight(w["acoustic_codebook_output.weight"].t()),
            "final_norm": stage(w["norm.weight"]),
            "layers": [layer(f"layers.{i}.") for i in range(_N_LAYERS)],
            "time_tokens": time_tokens,
            # llm_projection is bias-free, so the unconditional token is exactly zero.
            "uncond_token": stage(torch.zeros(batch, 1, _DIM)),
            "x_start": stage(torch.zeros(batch, _N_ACOUSTIC)),
            "batch": batch,
            "dtype": dtype,
        }
        return cls(device, tensors, float(cfg_alpha), int(n_steps))

    _linear = staticmethod(linear)

    @staticmethod
    def _norm(x, weight):
        return _rms_norm(x, weight, _NORM_EPS)

    def _block(self, x, layer):
        seq_len = int(x.shape[-2])
        batch = int(x.shape[0])

        h = self._norm(x, layer["attn_norm"])

        def heads(t, n):
            return ttnn.permute(ttnn.reshape(t, (batch, seq_len, n, _HEAD_DIM)), (0, 2, 1, 3))

        q = heads(self._linear(h, layer["wq"]), _N_HEADS)
        k = heads(self._linear(h, layer["wk"]), _N_KV_HEADS)
        v = heads(self._linear(h, layer["wv"]), _N_KV_HEADS)

        # GQA by folding each KV group's query heads into the sequence axis (see the attention
        # port); bidirectional and unmasked, so there is no bias term to add.
        n_rep = _N_HEADS // _N_KV_HEADS
        qg = ttnn.reshape(q, (batch, _N_KV_HEADS, n_rep * seq_len, _HEAD_DIM))
        scores = matmul(qg, ttnn.permute(k, (0, 1, 3, 2)))
        # The softmax is the model's composed one (see `_stubs/attention.py`): the fused op is
        # 2.8x looser even at HiFi4, and on its DEFAULT config the rows do not sum to 1 at all.
        # This block integrates 7 Euler steps into a value that is then ROUNDED onto 21 FSQ
        # levels, so a biased attention mass is a bias on the code that comes out.
        probs = softmax(ttnn.mul(scores, _HEAD_DIM**-0.5), dim=-1)
        attn = matmul(probs, v)
        attn = ttnn.reshape(attn, (batch, _N_HEADS, seq_len, _HEAD_DIM))
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (batch, seq_len, _N_HEADS * _HEAD_DIM))

        x = ttnn.add(x, self._linear(attn, layer["wo"]))
        h = self._norm(x, layer["ffn_norm"])
        gated = ttnn.mul(ttnn.silu(self._linear(h, layer["w1"])), self._linear(h, layer["w3"]))
        return ttnn.add(x, self._linear(gated, layer["w2"]))

    def _predict_velocity(self, x_pair, llm_token_pair, time_token):
        """[2B, 36] -> [2B, 36]: assemble the 3-token sequence and read position 0."""
        batch = int(x_pair.shape[0])
        current = ttnn.reshape(self._linear(x_pair, self.w_input), (batch, 1, _DIM))
        seq = ttnn.concat([current, time_token, llm_token_pair], dim=1)
        for layer in self.layers:
            seq = self._block(seq, layer)
        final = self._norm(seq, self.final_norm)
        position0 = ttnn.reshape(ttnn.slice(final, [0, 0, 0], [batch, 1, _DIM]), (batch, _DIM))
        return self._linear(position0, self.w_acoustic)

    def __call__(self, llm_hidden, cfg_alpha=None, n_steps=None, x_0=None, **kwargs):
        batch = self.batch

        # The caller's activation may arrive at the residual stream's precision; the staged
        # weights fix this block's. Convert rather than refuse, so a bf16 backbone can drive an
        # fp32 flow block (which is the configuration the e2e gate runs -- see the docstring).
        if llm_hidden.dtype != self.dtype:
            llm_hidden = ttnn.typecast(llm_hidden, self.dtype)

        # --- semantic code: greedy argmax over the masked logits -------------------------
        logits = ttnn.add(self._linear(llm_hidden, self.w_semantic), self.semantic_mask)
        semantic = ttnn.reshape(ttnn.argmax(logits, dim=-1), (batch, 1))

        # --- acoustic codes: Euler-integrate the guided velocity field --------------------
        llm_token = ttnn.reshape(self._linear(llm_hidden, self.w_llm), (batch, 1, _DIM))
        llm_token_pair = ttnn.concat([llm_token, self.uncond_token], dim=0)

        x = x_0 if isinstance(x_0, ttnn.Tensor) else self.x_start
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        alpha = self.cfg_alpha
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            velocities = self._predict_velocity(
                ttnn.concat([x, x], dim=0), llm_token_pair, self.time_tokens[step]
            )
            conditional = ttnn.slice(velocities, [0, 0], [batch, _N_ACOUSTIC])
            unconditional = ttnn.slice(velocities, [batch, 0], [2 * batch, _N_ACOUSTIC])
            guided = ttnn.add(ttnn.mul(conditional, alpha), ttnn.mul(unconditional, 1.0 - alpha))
            x = ttnn.add(x, ttnn.mul(guided, dt))

        # --- FSQ quantisation ------------------------------------------------------------
        scaled = ttnn.mul(ttnn.add(ttnn.clamp(x, -1.0, 1.0), 1.0), (_ACOUSTIC_LEVELS - 1) / 2.0)
        codes = ttnn.round(scaled)

        # A frame whose semantic code is [END_AUDIO] is never decoded -- its acoustic slots
        # are [EMPTY_AUDIO], which is 0, so zeroing them is the whole of that rule. The
        # comparison is safe in bf16: 1 is exact, and no other code rounds onto it.
        is_end = ttnn.eq(ttnn.typecast(semantic, self.dtype), float(_END_AUDIO_ID))
        codes = ttnn.mul(codes, ttnn.add(ttnn.neg(is_end), 1.0))

        codes = ttnn.typecast(ttnn.add(codes, float(_N_AUDIO_SPECIAL)), ttnn.uint32)
        return ttnn.concat(
            [
                ttnn.to_layout(semantic, ttnn.ROW_MAJOR_LAYOUT),
                ttnn.to_layout(codes, ttnn.ROW_MAJOR_LAYOUT),
            ],
            dim=-1,
        )


def build(device, torch_module=None, **kwargs):
    return TtVoxtralFlowMatching.build(device, torch_module, **kwargs)


def flow_matching(device, torch_module=None, **kwargs):
    return TtVoxtralFlowMatching.build(device, torch_module, **kwargs)
