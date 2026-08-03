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
  * Every one of the 7 steps is THE SAME SHAPE, so the whole solver captures as one device trace
    (`_trace`). It is implemented and correct but OFF, because it measured worth ~6% -- see below.

THIS BLOCK IS THE BOTTLENECK, measured -- ~100 ms of a ~110 ms frame, against ~48 ms for Block 1's
whole 3.4B decode step. Parameter count says the opposite and parameter count is the wrong proxy.

IT IS WEIGHT-READ BOUND, and the fix was structural. A batched matmul re-reads the whole weight per
batch element, so CFG's batch-2 forward was doubling every weight read; `_trunk` therefore folds the
batch into ROWS, which is free because 6 rows still fit one 32-row tile. Worth 2.23x, bit-identical.
Neither of the two things that look like the bottleneck is one: 4x less arithmetic (LoFi) buys 2.7%,
and host dispatch is a flat ~6.5 ms. Full measurements and the rejected alternatives -- sdpa, bfp4,
lower fidelity -- are in STATUS.md's Block 2 section. Read it before optimizing here.

WHERE THE TIME GOES, per frame, steady state on one N150 (Block 2 is 42.5 ms of a 77.6 ms frame,
so it is now the LARGER half and the natural next target):

    _solve -- 7 Euler steps          ~35 ms    7 SEQUENTIAL velocity evaluations. Each is a
                                               3-layer transformer over 3 tokens, batch-2 CFG
                                               folded to 6 rows, so every matmul does 32 tile
                                               rows of work for 6 useful ones.
    semantic_code                      3.1 ms  [B,8320] masked argmax, host
    host tail (FSQ quantise etc)       0.7 ms
    ------------------------------------------
    Block 2 total                     42.5 ms

The structural problem is the SEQUENCE: 7 steps that each depend on the previous, so none of the
usual batching tricks apply within a frame. What HAS been tried and rejected: lower math fidelity
(HiFi2/LoFi save ~4 ms and cost 10-20x the integer-code errors -- see COMPUTE_CONFIG), and a
device trace (correct, bit-identical, but ~6 ms/frame SLOWER on this part -- see USE_TRACE).
BFP8 weights ARE now on and were worth 1.23x. An untried idea worth a look: the 3-token sequence
wastes 26 of 32 tile rows, and STATUS.md notes CONCURRENT REQUESTS could fill them -- throughput,
not latency.

HOST vs DEVICE. The whole Euler solve -- the 3-layer transformer, the CFG combine and the state
update -- runs on device, with nothing left in the loop that a trace could not capture. Two things
stay on host, deliberately, and both are once per frame rather than once per step:
  * the semantic argmax -- a [B,8320] masked argmax whose result is an INDEX used to look up an
    embedding on host anyway (same reasoning as the codec's semantic gather).
  * the FSQ quantise -- clamp/scale/round on [B,36]; 36 values per frame is not worth a dispatch.
`time_embedding` is also host code, but it is no longer per step or even per frame: the solver
schedule is fixed, so `_schedule()` builds its projections once and caches them on device.
"""

import os
import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_flow_ref import (
    _fsq_quantize,
    load_flow_state,
    time_embedding,
)
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

# Math fidelity for the VELOCITY NETWORK. HiFi4 + fp32 destination accumulation is the most
# expensive setting on the part, and Block 2's solve is 48.5 ms of the 94 ms frame, so this is the
# first lever worth pulling -- STATUS.md 7 names it and it had never been measured.
#
# This does NOT touch the solver arithmetic. `x` and the CFG combine stay fp32 in decode_frame,
# deliberately: `x` accumulates n_steps increments so its error compounds, and the CFG combine is
# a difference of two nearly-equal vectors where the small difference IS the signal (fp32 2.4e-7
# vs bf16 7.0e-3, ~29,000x). Lowering fidelity here must never be allowed to reach those.
#
# VOXTRAL_FLOW_FIDELITY=hifi4|hifi2|lofi, VOXTRAL_FLOW_FP32ACC=0|1.
_FIDELITY = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3,
             "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}
_FID = _FIDELITY[os.environ.get("VOXTRAL_FLOW_FIDELITY", "hifi4")]
_FP32ACC = os.environ.get("VOXTRAL_FLOW_FP32ACC", "1") != "0"
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=_FID, math_approx_mode=False, fp32_dest_acc_en=_FP32ACC,
    packer_l1_acc=True,
)
# Activation dtype. Every ttnn op here inherits its input's dtype, so this one constant sets all
# activations; matmul accumulation stays fp32 via fp32_dest_acc_en above. It does NOT touch the
# solver arithmetic, which is fp32 on purpose -- see decode_frame, that exemption is load-bearing.
# Measured at 1.42x end-to-end for no quality cost (STATUS.md 3.2). To A/B it, pass dtype= or
# replace the instance: rebinding this constant does nothing, because __init__'s `dtype=DTYPE`
# default binds at definition time.
DTYPE = ttnn.bfloat16

# Capture the whole n_steps solve as one device trace. WORKS IN THE PIPELINE NOW, and is still
# OFF -- because it measures SLOWER here, not because it is broken.
#
# It used to be unusable end to end: correct standalone, but uncorrelated audio (PCC 0.002 against
# the untraced build) and not even deterministic. Cause: allocating device buffers while a trace
# is live corrupts it, and Block 1's decode allocated every frame. Two things fixed it --
# ttnn_voxtral_gpt.USE_TRACE makes Block 1's step a replay that allocates nothing per frame, and
# ttnn_voxtral_pipeline._prepare_traces warms up EVERY traced block before capturing ANY of them.
# Capturing lazily is what bit us: block B's warm-up allocated after block A was captured, and A's
# next replay hung. With the ordering fixed, eager and dual-traced frames are BIT-IDENTICAL --
# 0 differing codes of 2849 across three cases.
#
# THE MEASUREMENT, which is why it stays off. On a 466-frame clip: 96.2 ms/frame dual-traced
# against 90.0 eager, so tracing both blocks COSTS ~6 ms/frame. Block 1's trace needs one pinned
# decode window (~4 ms/step of extra attention) and is itself 0.7 ms slower at equal window, and
# Block 2's trace does not recover that. The old "~6.5 ms/frame" estimate for this trace was
# measured as a host/device gap, not as an end-to-end saving. Decode on this N150 is device-bound,
# not host-dispatch bound -- the ign/voxtral_p150_qb2 branch reports the opposite, but that is
# Blackhole.
#
# Kept because it is correct, cheap to re-enable, and the picture may invert on other silicon.
# Enable BOTH or neither: this one alone still corrupts. Read trap #1 before touching capture.
USE_TRACE = os.environ.get("VOXTRAL_FLOW_TRACE", "0") == "1"
TRACE_REGION_SIZE = 64 * 1024 * 1024

# MATMUL weight storage, independent of the activation dtype (None = same as DTYPE).
#
# BFP8 IS THE DEFAULT NOW, and this reverses an earlier conclusion. STATUS.md says bfp8 "costs
# accuracy for less than the free fix delivers" -- but that compared bfp8 WITHOUT the batch fold
# against bf16 WITH it. Measured with the fold in place, on top of it:
#
#     config                    velocity PCC   codes differ   ms/frame
#     HiFi4 fp32acc bf16 W       0.9999949        3/222         52.46
#     HiFi4 fp32acc BFP8 W       0.9999845        4/222         42.57   <- default
#     HiFi2 no-fp32acc bf16 W    0.9998382       35/222         48.72
#     LoFi  no-fp32acc bf16 W    0.9992816       62/222         48.59
#
# 1.23x for one extra differing code in 222. Note what the same table says about FIDELITY: HiFi2
# and LoFi save ~4 ms and cost 10-20x the code errors, so the expensive math setting stays. Codes
# are what reach the audio, which is why the gate is "codes differing from the reference frame"
# and not PCC alone (STATUS.md 3.2 used the same pairing to adopt bf16).
WEIGHT_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}.get(
    os.environ.get("VOXTRAL_FLOW_WEIGHTS", "bfp8"), None)


class TtVoxtralFlow:
    """Block 2 on device. __call__(h) -> audio_codes torch [B,37] int64."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE, weight_dtype=None):
        self.device = device
        self.dtype = dtype
        weight_dtype = weight_dtype or WEIGHT_DTYPE or dtype
        w = load_flow_state(ckpt_path)
        self.inv_freq = w["time_embedding.inv_freq"]          # host: time_embedding
        self.semantic_host = w["semantic_codebook_output.weight"].float()  # host: masked argmax
        self._sched = {}                     # (batch, n_steps) -> (time tokens, step widths)
        self._cfgbuf = {}                    # batch -> reused [2B,3072] cond++uncond host buffer
        self._traces = {}                    # (batch, n_steps, cfg_alpha) -> (tid, x_in, h_in, out)
        self._warm = {}                      # same key -> (x_in, h_in) before capture

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                         device=device)
        # RMSNorm gammas stay at the ACTIVATION dtype: 3072 values is no bandwidth, and a per-block
        # shared exponent is a poor fit for a 1-D scale vector. Only matmuls take weight_dtype.
        vec = lambda t: up(t.reshape(1, 1, -1), dtype)
        lin = lambda t: up(t.t(), weight_dtype)   # torch [out,in] -> ttnn.linear wants [in,out]

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
                # k and v fused into one weight -> one matmul, one weight stream. torch stores
                # Linear as [out, in], so concatenate along dim 0 before the transpose in `lin`.
                "wkv": lin(torch.cat([w[p + "attention.wk.weight"],
                                      w[p + "attention.wv.weight"]], dim=0)),
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
        # k and v share one weight and one matmul; nlp_create_qkv_heads then splits them AND builds
        # the head layout in a single op -- 3 linears + 6 reshape/permutes become 2 linears + 1 op.
        q = ttnn.linear(h, w["wq"], compute_kernel_config=COMPUTE_CONFIG)
        kv = ttnn.linear(h, w["wkv"], compute_kernel_config=COMPUTE_CONFIG)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(q, [B, 1, 3, FM_N_HEADS * FM_HEAD_DIM]),
            ttnn.reshape(kv, [B, 1, 3, 2 * FM_N_KV_HEADS * FM_HEAD_DIM]),
            num_heads=FM_N_HEADS, num_kv_heads=FM_N_KV_HEADS,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # GQA by hand: repeat k/v to the query head count. NO mask and NO RoPE -- bidirectional by
        # design. ttnn's sdpa would fuse these 7 ops into 1, but was measured and rejected on
        # accuracy (STATUS.md); note its is_causal defaults to True, which here is silently wrong.
        rep = FM_N_HEADS // FM_N_KV_HEADS
        kr = ttnn.repeat_interleave(kh, rep, dim=1)
        vr = ttnn.repeat_interleave(vh, rep, dim=1)
        s = ttnn.matmul(qh, ttnn.transpose(kr, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        s = ttnn.multiply(s, SCALE)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vr, compute_kernel_config=COMPUTE_CONFIG)
        # back to folded rows so wo and the MLP get the single-weight-read layout too
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG))
        h = ttnn.rms_norm(x, weight=w["fn"], epsilon=FM_NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG)
        g = ttnn.silu(ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG))
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG))
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG))

    def _up(self, t, dtype=None):
        return ttnn.from_torch(t.contiguous(), dtype=dtype or self.dtype,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    def _trunk(self, p0, p1, p2, B):
        """three [B,*,3072] projections -> velocity [B,1,36]. The 3-token sequence, reference order.

        B here is the CFG-doubled batch: decode_frame passes 2*batch."""
        seq = ttnn.concat([ttnn.reshape(p, [B, 1, FM_INPUT_DIM]) for p in (p0, p1, p2)], dim=1)
        # FOLD THE CFG BATCH INTO ROWS -- worth 2.23x, bit-identical. A batched matmul re-reads the
        # whole weight per batch element, so batch-2 doubled every weight read; 6 rows still fit one
        # 32-row tile, so folding is free. A linear applies per row independently. Attention is the
        # only thing that needs the batch separated again. Numbers in STATUS.md.
        seq = ttnn.reshape(seq, [1, B * 3, FM_INPUT_DIM])
        for w in self.layers:
            seq = self._block(seq, w, B)
        seq = ttnn.rms_norm(seq, weight=self.norm, epsilon=FM_NORM_EPS,
                            compute_kernel_config=COMPUTE_CONFIG)
        seq = ttnn.reshape(seq, [B, 3, FM_INPUT_DIM])
        pos0 = ttnn.slice(seq, [0, 0, 0], [B, 1, FM_INPUT_DIM])
        return ttnn.linear(pos0, self.proj["acoustic_codebook_output"],
                           compute_kernel_config=COMPUTE_CONFIG)

    def _cfg_input(self, B, llm_hidden):
        """-> [2B, 3072] = llm_hidden (cond) over zeros (uncond), in a buffer reused per batch.

        CFG's unconditional half is a ZEROED hidden state, so the bottom half is zeros on every
        frame forever -- only the top half is ever written, so the zeros cannot drift. Rebuilding
        it with `cat([h, zeros_like(h)])` allocated twice per frame, ~12x per second of audio.
        """
        buf = self._cfgbuf.get(B)
        if buf is None:
            buf = self._cfgbuf[B] = torch.zeros(2 * B, FM_INPUT_DIM)
        buf[:B] = llm_hidden            # bottom half stays zero by construction
        return buf

    def _schedule(self, B, n_steps):
        """-> (time-conditioning tokens on device, per-step dt). Built once per (batch, n_steps).

        The solver schedule `linspace(0, 1, n_steps+1)` never changes, so BOTH halves of it are
        constants: `time_projection(time_embedding(t))` does not depend on the prompt, the frame or
        x (it used to be recomputed every step -- a host sin/cos plus a 3072x3072 matmul each), and
        neither do the step widths. They are derived and cached together here so a change to the
        schedule cannot move the tokens without moving the dt values with them."""
        key = (B, n_steps)
        if key not in self._sched:
            ts = torch.linspace(0, 1, n_steps + 1)
            self._sched[key] = (
                [ttnn.linear(self._up(time_embedding(ts[i].view(1, 1).repeat(B, 1), self.inv_freq)),
                             self.proj["time_projection"], compute_kernel_config=COMPUTE_CONFIG)
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
        v = self._trunk(p0, p1, p2, B)
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

    def _solve(self, x, h, B, n_steps, cfg_alpha):
        """(x0 fp32 [B,1,36], cond++uncond [2B,3072]) -> x fp32 [B,1,36]. PURE DEVICE GRAPH.

        No host arithmetic, no allocation from torch, nothing shape-dependent on the data -- which
        is what makes it capturable as a trace. Keep it that way: one host op in here silently
        makes the whole solve untraceable again."""
        B2 = 2 * B
        # llm conditioning is constant across the solve, so project it ONCE rather than per step
        # (it was n_steps identical 3072x3072 matmuls).
        p2 = ttnn.linear(h, self.proj["llm_projection"], compute_kernel_config=COMPUTE_CONFIG)
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

    def warmup_trace(self, B, n_steps, cfg_alpha):
        """Allocate this trace's persistent inputs and run one eager solve. ALLOCATES.

        Split from `capture_trace` because allocating a device buffer while ANY trace is live
        corrupts it: with Block 1 traced too, every warm-up must finish before the first capture.
        `ttnn_voxtral_pipeline._prepare_traces` owns that ordering.
        """
        key = (B, n_steps, float(cfg_alpha))
        if key in self._traces or key in self._warm:
            return
        x_in = self._up(torch.zeros(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
        h_in = self._up(torch.zeros(2 * B, FM_INPUT_DIM))
        self._schedule(2 * B, n_steps)                    # constants, outside the capture
        self._solve(x_in, h_in, B, n_steps, cfg_alpha)    # compile kernels, outside the capture
        ttnn.synchronize_device(self.device)
        self._warm[key] = (x_in, h_in)

    def capture_trace(self, B, n_steps, cfg_alpha):
        """-> (trace_id, x_in, h_in, out). Captured once per (batch, n_steps, cfg_alpha).

        cfg_alpha and n_steps are baked into the captured graph as constants, hence the key:
        calling with a different guidance strength captures a second trace rather than silently
        replaying the first one's alpha.
        """
        key = (B, n_steps, float(cfg_alpha))
        if key in self._traces:
            return self._traces[key]
        self.warmup_trace(B, n_steps, cfg_alpha)
        x_in, h_in = self._warm[key]
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            out = self._solve(x_in, h_in, B, n_steps, cfg_alpha)
        finally:
            # MUST run even if _solve raises: an exception escaping an open capture leaves the
            # device wedged for the rest of the process (STATUS.md trap #1).
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._traces[key] = (tid, x_in, h_in, out)
        return self._traces[key]

    def release_traces(self):
        """Drop every captured trace, so the caller may allocate again (e.g. the next prefill)."""
        for tid, *_ in self._traces.values():
            ttnn.release_trace(self.device, tid)
        self._traces, self._warm = {}, {}

    @torch.no_grad()
    def decode_frame(self, sem_code, llm_hidden, cfg_alpha=CFG_ALPHA,
                     n_steps=N_DECODING_STEPS, x_0=None):
        """[B,1], [B,3072] -> acoustic codes [B,36] int64, offset applied.

        THE WHOLE SOLVE STAYS ON DEVICE. It used to round-trip per step -- upload x/t/h, download
        the velocity, then do the CFG combine and Euler update in torch -- which cost n_steps host
        round-trips per frame and, more importantly, made the loop untraceable, since a device trace
        cannot contain host arithmetic.

        PRECISION IS NOT UNIFORM HERE, deliberately. The velocity network runs at self.dtype (bf16
        by default) but the solver state does NOT:
          * `x` is fp32 and stays fp32 for its whole life. It accumulates n_steps increments, so any
            error in it COMPOUNDS -- unlike a per-step rounding error in the velocity, which does
            not.
          * the CFG combine is fp32. `cfg_alpha*v_cond + (1-cfg_alpha)*v_uncond` with alpha=1.2 is a
            difference of two nearly-equal vectors, and the small difference is the entire point of
            CFG. Measured on device: the combine is accurate to 2.4e-7 in fp32 but only 7.0e-3 from
            bf16 inputs -- ~29,000x worse. Doing this in bf16 would be a real quality bug that PCC
            on the velocity would not reveal.
        Hence the explicit typecasts: down to self.dtype only to enter the network, straight back up
        to fp32 on the way out. The cast entering `input_projection` is also load-bearing for SPEED:
        ttnn allows an fp32 activation against a bf16 weight and returns fp32, which would silently
        promote the entire trunk to fp32 and forfeit the 1.55x.
        """
        B = sem_code.shape[0]
        should = (sem_code != END_AUDIO_ID).reshape(B)
        x0 = torch.randn(B, N_ACOUSTIC_CODEBOOK) if x_0 is None else x_0
        h_host = self._cfg_input(B, llm_hidden)

        if USE_TRACE:
            tid, x_in, h_in, out = self.capture_trace(B, n_steps, cfg_alpha)
            # Refresh the captured inputs IN PLACE -- the trace holds their addresses, so a fresh
            # tensor here would be silently ignored and the previous frame's data replayed.
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(x0.reshape(B, 1, N_ACOUSTIC_CODEBOOK).contiguous(),
                                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), x_in)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(h_host.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT),
                h_in)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            x = out
        else:
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


def main():
    """Compare against the CPU reference. The output is INTEGER codes, so equality is exact."""
    from models.experimental.voxtral_tts.reference import voxtral_flow_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    dev = ttnn.open_device(device_id=0, l1_small_size=65536,
                           trace_region_size=TRACE_REGION_SIZE)
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
