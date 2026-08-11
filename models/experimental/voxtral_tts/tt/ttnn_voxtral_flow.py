# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral-TTS flow-matching acoustic transformer (BLOCK 2, 390M).

See NOTES.md [flow-01].
"""

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_flow_ref import (
    _fsq_quantize,
    load_flow_state,
    time_embedding,
)

# Block 2 has Block 1's dims exactly, so the decode matmul program configs are shared rather
# than duplicated -- NOTES.md [flow-23]. gpt does not import flow, so this cannot cycle.
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import DECODE_PRG, sharded_norm
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
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
# Fused q++k++v width, GQA-aware. (The sub-widths were the hand-rolled split's slice
# offsets until 6.45 replaced it with the fused op; they now only build _QKV_WIDTH.)
_Q_WIDTH = FM_N_HEADS * FM_HEAD_DIM
_KV_WIDTH = FM_N_KV_HEADS * FM_HEAD_DIM
_QKV_WIDTH = _Q_WIDTH + 2 * _KV_WIDTH

# NOTES.md [flow-02] -- EVERY INTERMEDIATE INSIDE `_block` LIVES IN L1, not DRAM...
_L1 = ttnn.L1_MEMORY_CONFIG

# NOTES.md [flow-03] -- Math fidelity for the VELOCITY NETWORK...
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
# NOTES.md [flow-04] -- Activation dtype. Every ttnn op here inherits its input's...
DTYPE = ttnn.bfloat16

# NOTES.md [flow-05] -- MATMUL weight storage, independent of the activation...
WEIGHT_DTYPE = ttnn.bfloat8_b

# NOTES.md [flow-06] -- The SEMANTIC head is the one thing here that is not...
SEMANTIC_DTYPE = ttnn.float32

# The norm IS width-sharded -- _norm below calls Block 1's shared `sharded_norm`. NOTES.md
# [gpt-28] / STATUS.md 6.67 is the current answer; [flow-07] is the superseded N150 record.


class TtVoxtralFlow:
    """Block 2 on device. __call__(h) -> audio_codes torch [B,37] int64."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT):
        self.device = device
        self.dtype = DTYPE
        w = load_flow_state(ckpt_path)
        self.inv_freq = w["time_embedding.inv_freq"]          # host: time_embedding
        self._sched = {}                     # (batch, n_steps) -> (time tokens, step widths)
        self._cfgbuf = {}                    # batch -> reused [2B,3072] cond++uncond host buffer

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                         device=device)
        # RMSNorm gammas stay at the ACTIVATION dtype: 3072 values is no bandwidth, and a per-block
        # shared exponent is a poor fit for a 1-D scale vector. Only matmuls take WEIGHT_DTYPE.
        vec = lambda t: up(t.reshape(1, 1, -1), DTYPE)
        lin = lambda t: up(t.t(), WEIGHT_DTYPE)  # torch [out,in] -> ttnn.linear wants [in,out]

        # NOTES.md [flow-08] -- SEMANTIC HEAD, ON DEVICE AND IN FP32...
        self.semantic_dev = up(w["semantic_codebook_output.weight"].float().t(), SEMANTIC_DTYPE)
        _vocab = w["semantic_codebook_output.weight"].shape[0]
        _mask = torch.zeros(1, 1, _vocab)
        _mask[:, :, EMPTY_AUDIO_ID] = -1e9
        _mask[:, :, N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE:] = -1e9
        # HOST, not device -- the mask add and the reduce both moved off chip, NOTES.md [flow-08a].
        self.semantic_mask_host = _mask.reshape(-1).float()

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
                # NOTES.md [flow-09] -- q, k and v fused into ONE weight -> one matmul instead of...
                "wqkv": lin(torch.cat([w[p + "attention.wq.weight"] * SCALE,
                                       w[p + "attention.wk.weight"],
                                       w[p + "attention.wv.weight"]], dim=0)),
                "wo": lin(w[p + "attention.wo.weight"]),
                "w1": lin(w[p + "feed_forward.w1.weight"]),
                "w2": lin(w[p + "feed_forward.w2.weight"]),
                "w3": lin(w[p + "feed_forward.w3.weight"]),
            })

    # ----------------------------------------------------------------------------------
    # One bidirectional block over the 3-token sequence
    # ----------------------------------------------------------------------------------
    def _norm(self, x, gamma):
        """RMSNorm, width-sharded on the decode path -- NOTES.md [flow-07], [gpt-28]."""
        return sharded_norm(x, gamma, FM_NORM_EPS, _L1)

    def _block(self, x, w, B):
        """x [1,B*3,3072] -> same. Pre-norm, GQA 32/8, unmasked attention, SwiGLU."""
        h = self._norm(x, w["an"])
        # q, k and v share one weight and one matmul -- see NOTES.md [flow-09] for the fusion.
        # NOTES.md [flow-23] -- same dims as Block 1, one tile of rows, so its configs apply
        qkv = ttnn.linear(h, w["wqkv"], program_config=DECODE_PRG["wqkv"],
                          compute_kernel_config=COMPUTE_CONFIG)
        # NOTES.md [flow-10] -- FUSED head split. 6.31 hand-rolled this into 9 ops and won
        # 1.233 ms/frame on the N150; on Blackhole a small op costs 3.4x more (67.7 us against
        # ~20), so trading 9 ops for 1 wins 3.836 ms/frame here at IDENTICAL accuracy. 6.45.
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [B, 1, 3, _QKV_WIDTH]), num_heads=FM_N_HEADS,
            num_kv_heads=FM_N_KV_HEADS, transpose_k_heads=False, memory_config=_L1)
        # NOTES.md [flow-11] -- sdpa for the interior: 4 ops -> 1, worth 2.555 ms/frame. It handles
        # GQA natively, so the row fold and REP are unnecessary. scale=1.0 IS MANDATORY -- SCALE is
        # folded into wqkv's q rows ([flow-09]), and the default would apply 1/sqrt(d) twice.
        a = ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=COMPUTE_CONFIG)
        # back to folded rows so wo and the MLP get the single-weight-read layout too
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        # NOTES.md [flow-22] -- in place. Only safe because the residual stream is born in L1
        # in _trunk: add_ writes where x already lives, so a DRAM x would silently undo
        # [flow-02]'s L1 residency and move a code. STATUS.md 6.48.
        x = ttnn.add_(x, ttnn.linear(a, w["wo"], program_config=DECODE_PRG["wo"],
                                     compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1))
        h = self._norm(x, w["fn"])
        # NOTES.md [flow-12] -- SiLU fuses only via the program config, never activation="silu"
        g = ttnn.linear(h, w["w1"], program_config=DECODE_PRG["w1"],
                        compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1)
        u = ttnn.multiply_(g, ttnn.linear(h, w["w3"], program_config=DECODE_PRG["w3"],
                                          compute_kernel_config=COMPUTE_CONFIG,
                                          memory_config=_L1))
        return ttnn.add_(x, ttnn.linear(u, w["w2"], program_config=DECODE_PRG["w2"],
                                        compute_kernel_config=COMPUTE_CONFIG,
                                        memory_config=_L1))

    def _up(self, t, dtype=None):
        return ttnn.from_torch(t.contiguous(), dtype=dtype or self.dtype,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    def _trunk(self, p0, p1, p2, B):
        """three [B,1,3072] projections -> velocity [B,1,36]. The 3-token sequence, reference order.

        B is the CFG-doubled batch; the caller supplies [B,1,3072]. See NOTES.md [flow-19].
        """
        # memory_config is load-bearing, not tidiness: it puts the RESIDUAL STREAM in L1 so
        # _block's add_ inherits it. Drop it and in-place silently reverts [flow-02]. [flow-22].
        seq = ttnn.concat([p0, p1, p2], dim=1, memory_config=_L1)
        # NOTES.md [flow-13] -- FOLD THE CFG BATCH INTO ROWS -- worth 2.23x...
        seq = ttnn.reshape(seq, [1, B * 3, FM_INPUT_DIM])
        for w in self.layers:
            seq = self._block(seq, w, B)
        seq = self._norm(seq, self.norm)
        # NOTES.md [flow-14] -- PROJECT FIRST, THEN NARROW -- worth 1.09 ms/frame...
        out = ttnn.linear(seq, self.proj["acoustic_codebook_output"],
                          compute_kernel_config=COMPUTE_CONFIG)
        out = ttnn.reshape(out, [B, 3, N_ACOUSTIC_CODEBOOK])
        return ttnn.slice(out, [0, 0, 0], [B, 1, N_ACOUSTIC_CODEBOOK])

    def _cfg_input(self, B, llm_hidden):
        """-> [2B, 3072] = llm_hidden (cond) over zeros (uncond), in a buffer reused per batch.

        See NOTES.md [flow-15].
        """
        buf = self._cfgbuf.get(B)
        if buf is None:
            buf = self._cfgbuf[B] = torch.zeros(2 * B, FM_INPUT_DIM)
        buf[:B] = llm_hidden            # bottom half stays zero by construction
        return buf

    def _schedule(self, B, n_steps):
        """-> (time-conditioning tokens on device, per-step dt). Built once per (batch, n_steps).

        See NOTES.md [flow-16].
        """
        key = (B, n_steps)
        if key not in self._sched:
            ts = torch.linspace(0, 1, n_steps + 1)
            self._sched[key] = (
                # already [B,1,3072]; constant for the model's life, so reshaped once here.
                [ttnn.reshape(
                    ttnn.linear(self._up(time_embedding(ts[i].view(1, 1).repeat(B, 1),
                                                        self.inv_freq)),
                                self.proj["time_projection"],
                                compute_kernel_config=COMPUTE_CONFIG),
                    [B, 1, FM_INPUT_DIM])
                 for i in range(n_steps)],
                [float(ts[i + 1] - ts[i]) for i in range(n_steps)],
            )
        return self._sched[key]

    def _predict_velocity(self, x_t, llm_h, t_emb):
        """torch [B,36], [B,3072], [B,64] -> velocity torch [B,36]. Position 0 only.

        Kept as a torch-in/torch-out entry point for the reference comparison in main(); the Euler
        solve in decode_frame does NOT go through it, because it keeps everything on device."""
        B = x_t.shape[0]
        p0 = ttnn.linear(self._up(x_t), self.proj["input_projection"],
                         compute_kernel_config=COMPUTE_CONFIG)
        p1 = ttnn.linear(self._up(t_emb), self.proj["time_projection"],
                         compute_kernel_config=COMPUTE_CONFIG)
        p2 = ttnn.linear(self._up(llm_h), self.proj["llm_projection"],
                         compute_kernel_config=COMPUTE_CONFIG)
        # _trunk wants [B,1,3072]; these three arrive 2D because the inputs here are 2D torch
        v = self._trunk(*(ttnn.reshape(p, [B, 1, FM_INPUT_DIM]) for p in (p0, p1, p2)), B)
        return ttnn.to_torch(v).float().reshape(B, N_ACOUSTIC_CODEBOOK)

    # ----------------------------------------------------------------------------------
    # Semantic code (host) and the Euler solve (device velocity)
    # ----------------------------------------------------------------------------------
    def semantic_code(self, llm_hidden):
        """h [B,3072] -> [B,1]. Greedy argmax: the fp32 matmul on device, the mask and the reduce
        on the HOST. See NOTES.md [flow-08a].
        """
        B = llm_hidden.shape[0]
        h = ttnn.from_torch(llm_hidden.reshape(1, B, -1).float().contiguous(),
                            dtype=SEMANTIC_DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        logits = ttnn.to_torch(ttnn.linear(h, self.semantic_dev,
                                           compute_kernel_config=COMPUTE_CONFIG)).float()
        return (logits.reshape(B, -1) + self.semantic_mask_host).argmax(-1).reshape(B, 1).long()

    def _solve(self, x, h, B, n_steps, cfg_alpha):
        """(x0 fp32 [B,1,36], cond++uncond [2B,3072]) -> x fp32 [B,1,36]. PURE DEVICE GRAPH.

        See NOTES.md [flow-17].
        """
        B2 = 2 * B
        # NOTES.md [flow-21] -- project AND reshape p2 once per frame, not once per step...
        p2 = ttnn.reshape(ttnn.linear(h, self.proj["llm_projection"],
                                      compute_kernel_config=COMPUTE_CONFIG),
                          [B2, 1, FM_INPUT_DIM])
        p1s, dts = self._schedule(B2, n_steps)
        for i, dt in enumerate(dts):
            # cond+uncond as ONE 2B forward, matching the reference exactly.
            x2 = ttnn.concat([x, x], dim=0)
            p0 = ttnn.linear(ttnn.typecast(x2, self.dtype), self.proj["input_projection"],
                             compute_kernel_config=COMPUTE_CONFIG)
            v = ttnn.typecast(self._trunk(p0, p1s[i], p2, B2), ttnn.float32)
            v_cond = ttnn.slice(v, [0, 0, 0], [B, 1, N_ACOUSTIC_CODEBOOK])
            v_unc = ttnn.slice(v, [B, 0, 0], [B2, 1, N_ACOUSTIC_CODEBOOK])
            v_cfg = ttnn.add(ttnn.multiply(v_cond, cfg_alpha),
                             ttnn.multiply(v_unc, 1.0 - cfg_alpha))
            x = ttnn.add(x, ttnn.multiply(v_cfg, dt))
        return x

    @torch.no_grad()
    def decode_frame(self, sem_code, llm_hidden, cfg_alpha=CFG_ALPHA,
                     n_steps=N_DECODING_STEPS, x_0=None):
        """[B,1], [B,3072] -> acoustic codes [B,36] int64, offset applied.

        See NOTES.md [flow-18].
        """
        B = sem_code.shape[0]
        should = (sem_code != END_AUDIO_ID).reshape(B)
        x0 = torch.randn(B, N_ACOUSTIC_CODEBOOK) if x_0 is None else x_0
        h_host = self._cfg_input(B, llm_hidden)

        x = self._solve(self._up(x0.reshape(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32),
                        self._up(h_host), B, n_steps, cfg_alpha)

        codes = _fsq_quantize(ttnn.to_torch(x).float().reshape(B, N_ACOUSTIC_CODEBOOK))
        codes[~should] = EMPTY_AUDIO_ID
        return codes + N_AUDIO_SPECIAL

    @torch.no_grad()
    def __call__(self, llm_hidden, **kw):
        """h [B,3072] -> audio_codes [B,37] int64 (semantic ++ acoustic)."""
        sem = self.semantic_code(llm_hidden)
        return torch.cat([sem, self.decode_frame(sem, llm_hidden, **kw)], dim=1)
