# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral-TTS flow-matching acoustic transformer (BLOCK 2, 390M).

Mirrors reference/voxtral_flow_ref.py op-for-op. Per generated frame:

    h [B,3072] --semantic head--> mask --> argmax                  = semantic code   [B,1]
             \\--7 x Euler step of a 3-layer bidirectional TF -->  = acoustic floats [B,36]
                                        clamp/scale/round (FSQ)    = acoustic codes  [B,36]
    = audio_codes [B,37], offset by N_AUDIO_SPECIAL, ready for Block 1's embed_frame

WHAT MAKES THIS BLOCK UNUSUAL:
  * The sequence is exactly THREE tokens -- [input_proj(x_t), time_proj(t_emb), llm_proj(h)] --
    and the velocity is read off POSITION 0. The other two exist only so attention can mix time
    and LLM conditioning into position 0.
  * Attention is BIDIRECTIONAL and unmasked: no RoPE, no causal mask, despite the GQA 32/8 head
    layout. `rope_theta` in params.json is inert here. Adding RoPE would be silently wrong.
  * CFG is batched, not doubled: cond and uncond (zeroed h) go through as one 2B forward, so a
    step is a batch-2 graph rather than two graphs.
  * Every one of the 7 steps is THE SAME SHAPE, which is what makes the whole solver a natural
    single-trace target. Upstream CUDA-graph the ODE and report 47% lower latency / 2.5x RTF.
    Not traced yet -- see STATUS.md; note the Block 3 tracing null result does NOT transfer,
    because these ops are tiny where Block 3's were device-bound.

HOST vs DEVICE. The 7 Euler steps and the 3-layer transformer run on device. Three things stay on
host, deliberately:
  * the semantic argmax -- a [B,8320] masked argmax whose result is an INDEX used to look up an
    embedding on host anyway (same reasoning as the codec's semantic gather).
  * the FSQ quantise -- clamp/scale/round on [B,36]; 36 values per frame is not worth a dispatch.
  * time_embedding -- [B,64] of sin/cos from a fixed inv_freq table, once per step.
The velocity field, which is 99% of the arithmetic, never leaves the device.
"""

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_flow_ref import (
    _fsq_quantize,
    load_flow_state,
    time_embedding,
)
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    DEFAULT_CKPT,
    EMPTY_AUDIO_ID,
    END_AUDIO_ID,
    FM_HEAD_DIM,
    FM_INPUT_DIM,
    FM_N_HEADS,
    FM_N_KV_HEADS,
    FM_N_LAYERS,
    FM_NORM_EPS,
    N_ACOUSTIC_CODEBOOK,
    N_AUDIO_SPECIAL,
    SEMANTIC_CODEBOOK_SIZE,
)

CFG_ALPHA = 1.2
N_DECODING_STEPS = 7
SCALE = FM_HEAD_DIM**-0.5

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.float32


class TtVoxtralFlow:
    """Block 2 on device. __call__(h) -> audio_codes torch [B,37] int64."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE):
        self.device = device
        self.dtype = dtype
        w = load_flow_state(ckpt_path)
        self.inv_freq = w["time_embedding.inv_freq"]          # host: time_embedding
        self.semantic_host = w["semantic_codebook_output.weight"].float()  # host: masked argmax

        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                       device=device)
        vec = lambda t: dev(t.reshape(1, 1, -1))
        lin = lambda t: dev(t.t())   # torch Linear [out,in] -> ttnn.linear wants [in,out]

        self.proj = {k: lin(w[f"{k}.weight"]) for k in
                     ("input_projection", "time_projection", "llm_projection",
                      "acoustic_codebook_output")}
        self.norm = vec(w["norm.weight"])
        self.layers = []
        for i in range(FM_N_LAYERS):
            p = f"layers.{i}."
            self.layers.append({
                "an": vec(w[p + "attention_norm.weight"]),
                "fn": vec(w[p + "ffn_norm.weight"]),
                "wq": lin(w[p + "attention.wq.weight"]),
                "wk": lin(w[p + "attention.wk.weight"]),
                "wv": lin(w[p + "attention.wv.weight"]),
                "wo": lin(w[p + "attention.wo.weight"]),
                "w1": lin(w[p + "feed_forward.w1.weight"]),
                "w2": lin(w[p + "feed_forward.w2.weight"]),
                "w3": lin(w[p + "feed_forward.w3.weight"]),
            })

    # ----------------------------------------------------------------------------------
    # One bidirectional block over the 3-token sequence
    # ----------------------------------------------------------------------------------
    def _block(self, x, w, B):
        """x [1,B*3,3072] -> same. Pre-norm, GQA 32/8, unmasked attention, SwiGLU."""
        h = ttnn.rms_norm(x, weight=w["an"], epsilon=FM_NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG)
        q = ttnn.linear(h, w["wq"], compute_kernel_config=COMPUTE_CONFIG)
        k = ttnn.linear(h, w["wk"], compute_kernel_config=COMPUTE_CONFIG)
        v = ttnn.linear(h, w["wv"], compute_kernel_config=COMPUTE_CONFIG)
        # Heads via reshape+permute rather than the fused nlp_create_qkv_heads used in Block 3:
        # the sequence is 3 tokens, so there is no data-movement cost worth fusing, and the fused
        # op needs a batch layout this does not have.
        hq = lambda t, n: ttnn.permute(ttnn.reshape(t, [B, 3, n, FM_HEAD_DIM]), (0, 2, 1, 3))
        qh, kh, vh = hq(q, FM_N_HEADS), hq(k, FM_N_KV_HEADS), hq(v, FM_N_KV_HEADS)
        # GQA: repeat k/v to the query head count. NO mask and NO RoPE -- bidirectional by design.
        rep = FM_N_HEADS // FM_N_KV_HEADS
        kr = ttnn.repeat_interleave(kh, rep, dim=1)
        vr = ttnn.repeat_interleave(vh, rep, dim=1)
        s = ttnn.matmul(qh, ttnn.transpose(kr, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        s = ttnn.multiply(s, SCALE)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vr, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [B, 3, FM_N_HEADS * FM_HEAD_DIM])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG))
        h = ttnn.rms_norm(x, weight=w["fn"], epsilon=FM_NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG)
        g = ttnn.silu(ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG))
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG))
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG))

    def _predict_velocity(self, x_t, llm_h, t_emb):
        """torch [B,36], [B,3072], [B,64] -> velocity torch [B,36]. Position 0 only."""
        B = x_t.shape[0]
        T = lambda t: ttnn.from_torch(t.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
                                     device=self.device)
        p0 = ttnn.linear(T(x_t), self.proj["input_projection"], compute_kernel_config=COMPUTE_CONFIG)
        p1 = ttnn.linear(T(t_emb), self.proj["time_projection"], compute_kernel_config=COMPUTE_CONFIG)
        p2 = ttnn.linear(T(llm_h), self.proj["llm_projection"], compute_kernel_config=COMPUTE_CONFIG)
        # [B,3072] x3 -> [B,3,3072]: the 3-token sequence, in the reference's order.
        seq = ttnn.concat([ttnn.reshape(p, [B, 1, FM_INPUT_DIM]) for p in (p0, p1, p2)], dim=1)
        for w in self.layers:
            seq = self._block(seq, w, B)
        seq = ttnn.rms_norm(seq, weight=self.norm, epsilon=FM_NORM_EPS,
                            compute_kernel_config=COMPUTE_CONFIG)
        pos0 = ttnn.slice(seq, [0, 0, 0], [B, 1, FM_INPUT_DIM])
        v = ttnn.linear(pos0, self.proj["acoustic_codebook_output"],
                        compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.to_torch(v).float().reshape(B, N_ACOUSTIC_CODEBOOK)

    # ----------------------------------------------------------------------------------
    # Semantic code (host) and the Euler solve (device velocity)
    # ----------------------------------------------------------------------------------
    def semantic_code(self, llm_hidden):
        """h [B,3072] -> [B,1]. Masked greedy argmax; kept on host, see the module docstring.
        [EMPTY_AUDIO] is forbidden; [END_AUDIO] is ALLOWED because that is how generation stops."""
        logits = (llm_hidden.float() @ self.semantic_host.t())
        logits[:, EMPTY_AUDIO_ID] = -float("inf")
        logits[:, N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE:] = -float("inf")
        return logits.argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def decode_frame(self, sem_code, llm_hidden, cfg_alpha=CFG_ALPHA,
                     n_steps=N_DECODING_STEPS, x_0=None):
        """[B,1], [B,3072] -> acoustic codes [B,36] int64, offset applied."""
        B = sem_code.shape[0]
        should = (sem_code != END_AUDIO_ID).reshape(B)
        x = torch.randn(B, N_ACOUSTIC_CODEBOOK) if x_0 is None else x_0.clone()
        ts = torch.linspace(0, 1, n_steps + 1)
        zero_h = torch.zeros_like(llm_hidden)
        for i in range(n_steps):
            t, dt = ts[i], ts[i + 1] - ts[i]
            t_emb = time_embedding(t.view(1, 1).repeat(B, 1), self.inv_freq)
            # cond+uncond as ONE 2B forward, matching the reference exactly.
            v_all = self._predict_velocity(
                torch.cat([x, x], 0),
                torch.cat([llm_hidden, zero_h], 0),
                torch.cat([t_emb, t_emb], 0),
            )
            v = cfg_alpha * v_all[:B] + (1 - cfg_alpha) * v_all[B:]
            x = x + v * dt
        codes = _fsq_quantize(x)          # host: clamp/scale/round on [B,36]
        codes[~should] = EMPTY_AUDIO_ID
        return codes + N_AUDIO_SPECIAL

    @torch.no_grad()
    def __call__(self, llm_hidden, **kw):
        """h [B,3072] -> audio_codes [B,37] int64 (semantic ++ acoustic)."""
        sem = self.semantic_code(llm_hidden)
        return torch.cat([sem, self.decode_frame(sem, llm_hidden, **kw)], dim=1)


def main():
    """Compare against the CPU reference. The output is INTEGER codes, so equality is exact."""
    from models.experimental.voxtral_tts.reference import voxtral_flow_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        gen = TtVoxtralFlow(dev)
        w = ref.load_flow_state()
        h, x_0 = ref.make_synthetic_inputs(batch=2, seed=0)

        # 1) one velocity evaluation -- the unit a trace would capture
        t_emb = ref.time_embedding(torch.tensor(0.375).view(1, 1).repeat(2, 1),
                                   w["time_embedding.inv_freq"])
        exp_v = ref.predict_velocity(x_0, h, t_emb, w)
        got_v = gen._predict_velocity(x_0, h, t_emb)
        print(f"  [velocity      ] PCC {pcc(got_v, exp_v):.8f}  maxabs {(got_v-exp_v).abs().max():.3e}")

        # 2) semantic code -- must match EXACTLY, it is an index
        exp_s, got_s = ref.semantic_code(h, w), gen.semantic_code(h)
        print(f"  [semantic code ] exact match: {bool((exp_s==got_s).all())}  {exp_s.flatten().tolist()}")

        # 3) full frame, deterministic x_0 -- 37 INTEGER codes, so exact or not
        exp_f = ref.reference_frame(h, w, x_0=x_0)
        got_f = gen(h, x_0=x_0)
        n_diff = int((exp_f != got_f).sum())
        print(f"  [full frame    ] {'IDENTICAL' if n_diff==0 else f'{n_diff} of {exp_f.numel()} codes differ'}")
        if n_diff:
            print(f"      ref  {exp_f[0, :10].tolist()}")
            print(f"      got  {got_f[0, :10].tolist()}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
