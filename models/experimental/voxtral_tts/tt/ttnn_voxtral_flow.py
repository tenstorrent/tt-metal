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
  * Every one of the 7 steps is THE SAME SHAPE. That made the whole solver capturable as one
    device trace; it was correct and bit-identical but ~6 ms/frame SLOWER on this N150, so the
    capture machinery was removed rather than left as a dormant second path.

IT IS NOT WEIGHT-READ BOUND. It WAS, which is why the CFG batch fold in `_trunk` was worth 2.23x
(a batched matmul re-reads the whole weight per batch element, so batch-2 doubled every read; 6
rows still fit one 32-row tile, so folding is free). But do the arithmetic on where it stands now:
the velocity net is 349M params (3 layers x 116.4M) at BFP8, so 7 steps stream ~2.6 GB, and at the
194 GB/s Block 1 demonstrably reaches that is a 13.4 ms floor. The solve measured 35 ms -- 74 GB/s,
38% of the ceiling.

THE GAP IS PER-KERNEL COST -- and note the precise wording, because the obvious reading of it is
wrong and was tested. A step is ~88 ops and only 18 carry weights; the rest are reshapes,
transposes, softmaxes, slices and typecasts on tiny tensors. It is tempting to conclude "delete
small ops", and that DOES NOT WORK: fusing the CFG combine from 5 ops to 3 measured 1.001x, and
`inplace=True` on the norm measured 0.997x. Tracing, which removes host dispatch entirely, is
+0.16 ms. The cost is device-side and per KERNEL, and these kernels are already at the floor.

WHAT ACTUALLY WORKS IS FEWER, BIGGER KERNELS. Every win here has that shape: the CFG row fold
(2.23x) and the GQA row fold (1.40x) both made matmuls bigger by folding work into unused tile
rows, and the qkv fusion (0.96 ms) merged a 2048-wide matmul that was costing the same as a
4096-wide one into a single larger call. Judge a proposal on whether it makes kernels BIGGER, not
on how many ops it deletes.

WHERE THE TIME GOES, per frame, steady state on one N150 (~23 ms of a ~51 ms frame; Block 1's
26.6 ms is the larger half):

    _solve -- 7 Euler steps          ~21 ms    7 SEQUENTIAL velocity evaluations, each a 3-layer
                                               transformer over 3 tokens, CFG batch-2 folded to 6
                                               rows. Floor is 13.4 ms (2.6 GB at 194 GB/s), so
                                               there is still ~1.6x of per-kernel cost in here.
                                               The biggest single non-matmul line is
                                               nlp_create_qkv_heads at ~97 us x21 -- see _block,
                                               where its floor is measured and shown to be fixed.
    semantic_code                      1.25 ms [B,8320] masked argmax, now ON DEVICE in fp32.
                                               Was 2.74 ms of real host CPU. fp32 is mandatory --
                                               it produces an INDEX; see semantic_dev.
    host tail (FSQ quantise etc)       0.7 ms
    ------------------------------------------
    Block 2 total                     ~23 ms   (42.5 before the row fold, sharded norm, qkv
                                               fusion, device semantic head and L1 interior)

The structural problem is the SEQUENCE: 7 steps that each depend on the previous, so none of the
usual batching tricks apply within a frame.

TRIED AND REJECTED, so they do not get retried: lower math fidelity (HiFi2/LoFi save ~4 ms for
10-20x the integer-code errors -- see COMPUTE_CONFIG); sdpa for the attention interior (1.147x, but
codes differing from the fp32 reference go 7/288 -> 21/288 over 8 draws -- rejected once before
BFP8 and again after, same answer); the CFG-combine and inplace-norm micro-fusions above; and a
device trace, re-measured after all of the above at +0.16 ms (1.006x) with bit-identical codes.
BFP8 weights ARE on and were worth 1.23x.

STILL OPEN, and both are structural rather than op-level:
  * FEWER EULER STEPS. 7 -> 5 removes 28% of the solve outright. This changes what the model
    produces, so it needs a listening pass, not a metric.
  * CONCURRENT REQUESTS. The 3-token sequence wastes 26 of 32 tile rows and nothing within one
    utterance can fill them, since the steps are sequential and the frames autoregressive. This is
    throughput, not latency -- it will not move RTF for a single utterance.

That trace result is the load-bearing evidence for the per-kernel framing at the top: tracing
removes host command submission almost entirely, and removing it changes nothing. STATUS.md 6.6
has the table.

HOST vs DEVICE. The whole Euler solve -- the 3-layer transformer, the CFG combine and the state
update -- runs on device, with nothing left in the loop that a trace could not capture. Two things
stay on host, deliberately, and both are once per frame rather than once per step:
  * the semantic argmax -- a [B,8320] masked argmax whose result is an INDEX used to look up an
    embedding on host anyway (same reasoning as the codec's semantic gather).
  * the FSQ quantise -- clamp/scale/round on [B,36]; 36 values per frame is not worth a dispatch.
`time_embedding` is also host code, but it is no longer per step or even per frame: the solver
schedule is fixed, so `_schedule()` builds its projections once and caches them on device.
"""

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
# Query heads per KV head (GQA 32/8). See the row fold in _block.
REP = FM_N_HEADS // FM_N_KV_HEADS
# Width of the fused q++k++v projection, GQA-aware: 32 q heads + 2 x 8 kv heads.
_QKV_WIDTH = (FM_N_HEADS + 2 * FM_N_KV_HEADS) * FM_HEAD_DIM

# EVERY INTERMEDIATE INSIDE `_block` LIVES IN L1, not DRAM. This is the same finding as the
# L1-resident q/k/v (see _block): at this block's shapes, WHERE a tensor lives matters as much as
# how big the kernel is. Nothing here exceeds ~110 KB ([1,6,9216] bf16), and each value is consumed
# within a few ops of being produced, so a DRAM round trip per intermediate is pure latency.
#
# Measured cumulatively, 8 draws, all at IDENTICAL accuracy (9/288 differing codes throughout):
#
#     q/k/v L1 only (was)                      24.18 ms
#     + attention interior (scores, scaled, av) 23.85    1.014x
#     + MLP intermediates (g, w3_out, u)        23.22    1.041x
#     + residual stream                         23.04    1.049x   <- shipped
#
# The one candidate that does NOT pay is the `_norm` output: 0.999x alone, so it stays DRAM. That
# is worth knowing -- it is not "L1 everywhere is better", it is specifically the values with a
# consumer close behind. And note the LIMIT: a width-SHARDED activation into a DRAM-weight matmul
# is SLOWER (8.94 vs 5.32 ms per 26 norm+linear pairs). Interleaved-L1 is the useful middle.
_L1 = ttnn.L1_MEMORY_CONFIG

# Math fidelity for the VELOCITY NETWORK. HiFi4 + fp32 destination accumulation is the most
# expensive setting available and it STAYS: lowering it is a bad trade at this block's shapes.
#
#     config                    velocity PCC   codes differ   ms/frame
#     HiFi4 fp32acc BFP8 W       0.9999845        4/222         42.57   <- shipped
#     HiFi2 no-fp32acc bf16 W    0.9998382       35/222         48.72
#     LoFi  no-fp32acc bf16 W    0.9992816       62/222         48.59
#
# ~4 ms for 10-20x the integer-code errors. Codes are what reach the audio, which is why the gate
# is "codes differing from the reference frame" and not PCC alone.
#
# This does NOT touch the solver arithmetic. `x` and the CFG combine stay fp32 in decode_frame,
# deliberately: `x` accumulates n_steps increments so its error compounds, and the CFG combine is
# a difference of two nearly-equal vectors where the small difference IS the signal (fp32 2.4e-7
# vs bf16 7.0e-3, ~29,000x).
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
# Activation dtype. Every ttnn op here inherits its input's dtype, so this one constant sets all
# activations; matmul accumulation stays fp32 via fp32_dest_acc_en above. Measured at 1.42x
# end-to-end for no quality cost (STATUS.md 3.2).
DTYPE = ttnn.bfloat16

# MATMUL weight storage, independent of the activation dtype. BFP8 is worth 1.23x here for ONE
# extra differing code in 222 (see the table above). That reverses an earlier conclusion in
# STATUS.md, which compared bfp8 WITHOUT the batch fold against bf16 WITH it; measured on top of
# the fold, bfp8 wins.
WEIGHT_DTYPE = ttnn.bfloat8_b

# The SEMANTIC head is the one thing here that is not bf16/BFP8, and it is not a free
# choice -- see semantic_dev in __init__. It emits an INDEX, so a rounding difference is
# not a small error, it is a different code.
SEMANTIC_DTYPE = ttnn.float32

# RMSNORM, WIDTH-SHARDED. Same finding as Block 1's _NORM_SHARD, and it matters more here: 7 norms
# per Euler step x 7 steps = 49 calls per frame. Interleaved, each costs ~115 us on a [1,6,3072]
# tensor -- latency, not arithmetic, since one core reduces the row with a DRAM round trip either
# side. Spread over 8 cores the norm+linear pair measures 1.46x (5.32 vs 7.78 ms per 26).
#
# fp32 accumulation is UNCHANGED, so this is not the rejected "lower the fidelity" trade above.
# The row count is pinned by the program config, which is fine: every norm in the solve sees the
# same [1, B*3, 3072], tile-padded to 32 rows.
_NORM_GRID_X = 8
_NORM_SHARD = ttnn.create_sharded_memory_config(
    shape=(1, 1, 32, FM_INPUT_DIM), core_grid=ttnn.CoreGrid(y=1, x=_NORM_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH)
_NORM_PRG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(_NORM_GRID_X, 1), subblock_w=4, block_h=1,
    block_w=FM_INPUT_DIM // _NORM_GRID_X // 32, inplace=False)


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

        # SEMANTIC HEAD, ON DEVICE AND IN FP32. This used to be a host matmul -- [1,3072] @
        # [3072,8320] plus a mask and an argmax -- and it measured 2.74 ms per frame of real CPU
        # time, ~4% of the frame. On device it is 1.25 ms, so 1.49 ms/frame.
        #
        # FP32 IS NOT NEGOTIABLE HERE, unlike everywhere else in this module. The result is an
        # INDEX, not a value: two close logits ranked the other way round change the semantic code
        # outright. Measured over 64 hidden states, bf16 weights pick a DIFFERENT index on 4 of
        # them; fp32 matches the host answer on all 64. bf16 would be 1.04 ms -- 0.2 ms more, for
        # a wrong primary code on ~6% of frames.
        #
        # The mask is additive and prebuilt rather than the host version's two -inf assignments:
        # -1e9 underflows exp() to zero the same way, and one add is cheaper than two writes.
        # [EMPTY_AUDIO] is forbidden; [END_AUDIO] is ALLOWED, since that is how generation stops.
        self.semantic_dev = up(w["semantic_codebook_output.weight"].float().t(), SEMANTIC_DTYPE)
        _vocab = w["semantic_codebook_output.weight"].shape[0]
        _mask = torch.zeros(1, 1, _vocab)
        _mask[:, :, EMPTY_AUDIO_ID] = -1e9
        _mask[:, :, N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE:] = -1e9
        self.semantic_mask = up(_mask, SEMANTIC_DTYPE)

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
                # q, k and v fused into ONE weight -> one matmul instead of three. torch stores
                # Linear as [out, in], so concatenate along dim 0 before the transpose in `lin`.
                #
                # Fusing q in as well (it used to be its own matmul alongside a fused kv) is worth
                # 1.449x on this pair -- 2.13 against 3.09 ms per frame -- and it is the same
                # arithmetic: a linear computes each output column independently, and 4096 is a
                # multiple of the 32-wide tile, so the BFP8 blocks are unchanged too.
                #
                # WHY IT WINS, because it says where the next one is: `wkv` alone is only 2048
                # wide and measured 73 us, the SAME as the 4096-wide `wq`. There is a fixed cost
                # of roughly 40-50 us per matmul launch at these shapes, and width past ~4096 is
                # nearly free. Fusing pays exactly when one of the pair is too narrow to earn its
                # launch. It does NOT pay otherwise: w1+w3 are 9216 each, and merging them into
                # 18432 measured 0.998x -- no gain -- and 0.951x once the output split is charged.
                "wqkv": lin(torch.cat([w[p + "attention.wq.weight"],
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
        """RMSNorm, width-sharded -- see _NORM_SHARD. Both memory_config moves are required: the
        op will not write interleaved from a sharded input, and handing the sharded result to the
        next matmul directly measured slower than converting back."""
        h = ttnn.rms_norm(ttnn.to_memory_config(x, _NORM_SHARD), weight=gamma, epsilon=FM_NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG, program_config=_NORM_PRG,
                          memory_config=_NORM_SHARD)
        return ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)

    def _block(self, x, w, B):
        """x [1,B*3,3072] -> same. Pre-norm, GQA 32/8, unmasked attention, SwiGLU."""
        h = self._norm(x, w["an"])
        # q, k and v share one weight and one matmul; nlp_create_qkv_heads splits all three AND
        # builds the head layout in a single op -- given no `input_kv` it reads one fused tensor.
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
        # THIS OP HAS A ~97 us FLOOR AND IT IS THE MOST EXPENSIVE NON-MATMUL LINE IN THE BLOCK --
        # 2.7 ms/frame over 21 calls, more than the wqkv matmul that feeds it. The floor is fixed
        # cost, not data movement: the same call on S=32 (10.7x the data) also measures 97 us. So
        # there is nothing to win by feeding it less or by laying the input out differently, and
        # both restructurings that avoid it are WORSE -- hand-rolled slice+reshape+permute is
        # 158 us, and riding the CFG pair on the sequence dim is 259 us. The two sibling ttnn ops
        # (create_qkv_heads, transformer.split_query_key_value_and_split_heads) reject GQA shapes.
        #
        # WHAT DOES WORK IS THE OUTPUT MEMORY CONFIG, and for a reason the op-level numbers hide.
        # Isolated, an L1 output saves only ~7 us on the op itself. In the real block it is worth
        # 2.5 ms/frame, because q/k/v then stay in L1 for the four ops that consume them:
        #
        #     output   transpose_k   ms/frame   codes != fp32 ref (8 draws)
        #     DRAM     False           26.75         7/288          <- was
        #     DRAM     True            26.72         7/288
        #     L1       False           24.28         9/288
        #     L1       True            24.17         9/288          <- shipped, 1.106x
        #
        # So L1 carries all of the speed AND all of the cost: 2 extra differing codes in 288.
        # `transpose_k_heads=True` is free -- it emits k already transposed for the scores matmul,
        # deleting our own transpose op -- but worth almost nothing on its own (1.001x); it is on
        # because one fewer op is one fewer thing to read.
        #
        # NOT bit-exact, and the chain is: the three tensors are bit-identical either way (verified
        # with torch.equal), but an L1-resident operand makes the downstream matmul pick a different
        # program config, hence a different accumulation order. Velocity PCC 0.99998522 ->
        # 0.99998164. Gated on codes and the 15-case run, not on PCC.
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [B, 1, 3, _QKV_WIDTH]),
            num_heads=FM_N_HEADS, num_kv_heads=FM_N_KV_HEADS,
            transpose_k_heads=True, memory_config=_L1)
        # GQA BY ROW FOLD, NOT BY REPEAT -- the same lesson as the CFG fold above, and worth 1.40x
        # on this block. Mathematically the same attention: on device it gives the same velocity
        # PCC and the same 3-of-74 code diff, and in fp32 on host the two agree to 6e-07, which is
        # reduction order, not a different computation. Query head j reads kv head j//4, so reshaping
        # q from [B,32,3,d] to [B,8,12,d] stacks those 4 heads' 3 tokens as 12 ROWS against a
        # single kv head. Heads are contiguous in dim 1, so the grouping lands exactly on the GQA
        # mapping and the inverse reshape puts them back. Equivalent because a row of the score
        # matrix only ever interacts with itself: softmax is over the last dim (the 3 keys).
        #
        # The win is the same one the CFG fold gets: `repeat_interleave` made the two attention
        # matmuls BATCH-32, and a batched matmul costs per batch element. This makes them batch-8
        # with 12 rows each, and rows inside a 32-row tile are free. It also deletes the two
        # repeat_interleave ops, which were materialising k/v 4x.
        #
        # NO mask and NO RoPE here -- bidirectional by design. sdpa would fuse the interior into
        # ONE op and measures faster still (1.147x), but it triples the differing codes (7/288 ->
        # 21/288 over 8 draws); see STATUS.md 6.8.
        s = ttnn.matmul(ttnn.reshape(qh, [B, FM_N_KV_HEADS, REP * 3, FM_HEAD_DIM]),
                        kh, compute_kernel_config=COMPUTE_CONFIG,   # kh already transposed
                        memory_config=_L1)
        s = ttnn.multiply(s, SCALE, memory_config=_L1)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vh, compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1)
        # back to folded rows so wo and the MLP get the single-weight-read layout too
        a = ttnn.reshape(a, [B, FM_N_HEADS, 3, FM_HEAD_DIM])
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG,
                                    memory_config=_L1), memory_config=_L1)
        h = self._norm(x, w["fn"])
        g = ttnn.silu(ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG,
                                  memory_config=_L1), memory_config=_L1)
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG,
                                         memory_config=_L1), memory_config=_L1)
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG,
                                       memory_config=_L1), memory_config=_L1)

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
        seq = self._norm(seq, self.norm)
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
        """h [B,3072] -> [B,1]. Masked greedy argmax, on device in fp32 -- see semantic_dev."""
        B = llm_hidden.shape[0]
        h = ttnn.from_torch(llm_hidden.reshape(1, B, -1).float().contiguous(),
                            dtype=SEMANTIC_DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        logits = ttnn.add(ttnn.linear(h, self.semantic_dev, compute_kernel_config=COMPUTE_CONFIG),
                          self.semantic_mask)
        return ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(B, 1).long()

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
