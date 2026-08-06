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
# Width of the fused q++k++v projection, GQA-aware: 32 q heads + 2 x 8 kv heads. The q and kv
# sub-widths are the slice offsets the hand-rolled head split cuts on -- see NOTES.md [flow-10].
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

# NOTES.md [flow-07] -- RMSNORM, WIDTH-SHARDED. Same finding as Block 1's...
# Grid and blocking: swept, see NOTES.md [gpt-04] for the Block 1 version of the same experiment.
# More cores is monotonically faster end to end here too -- 8x1/8x2/8x4 measure 24.628/24.572/24.465
# ms/frame -- and `subblock_w` is INERT: 4 and 1 measure byte-identical at 8x1. 32 cores is the most
# that divides evenly (3072/32 = 96 tiles; 96/64 is not an integer).
_NORM_GRID_X, _NORM_GRID_Y, _NORM_SUBBLOCK_W = 8, 4, 1
_NORM_CORES = _NORM_GRID_X * _NORM_GRID_Y
_NORM_SHARD = ttnn.create_sharded_memory_config(
    shape=(1, 1, 32, FM_INPUT_DIM), core_grid=ttnn.CoreGrid(y=_NORM_GRID_Y, x=_NORM_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH)
_NORM_PRG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(_NORM_GRID_X, _NORM_GRID_Y),
    subblock_w=_NORM_SUBBLOCK_W, block_h=1,
    block_w=FM_INPUT_DIM // _NORM_CORES // 32, inplace=False)


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
        # q, k and v share one weight and one matmul -- see NOTES.md [flow-09] for the fusion.
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
        # NOTES.md [flow-10] -- HAND-ROLLED HEAD SPLIT, worth 1.233 ms/frame over...
        _cut = lambda a, b: ttnn.slice(qkv, [0, 0, a], [1, B * 3, b], memory_config=_L1)
        qh = ttnn.permute(ttnn.reshape(_cut(0, _Q_WIDTH), [B, 3, FM_N_HEADS, FM_HEAD_DIM]),
                          (0, 2, 1, 3), memory_config=_L1)
        # k comes out already transposed for the scores matmul, as transpose_k_heads=True used to do
        kh = ttnn.permute(ttnn.reshape(_cut(_Q_WIDTH, _Q_WIDTH + _KV_WIDTH),
                                       [B, 3, FM_N_KV_HEADS, FM_HEAD_DIM]),
                          (0, 2, 3, 1), memory_config=_L1)
        vh = ttnn.permute(ttnn.reshape(_cut(_Q_WIDTH + _KV_WIDTH, _QKV_WIDTH),
                                       [B, 3, FM_N_KV_HEADS, FM_HEAD_DIM]),
                          (0, 2, 1, 3), memory_config=_L1)
        # NOTES.md [flow-11] -- GQA BY ROW FOLD, NOT BY REPEAT -- the same lesson as the...
        s = ttnn.matmul(ttnn.reshape(qh, [B, FM_N_KV_HEADS, REP * 3, FM_HEAD_DIM]),
                        kh, compute_kernel_config=COMPUTE_CONFIG,   # kh already transposed
                        memory_config=_L1)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vh, compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1)
        # back to folded rows so wo and the MLP get the single-weight-read layout too
        a = ttnn.reshape(a, [B, FM_N_HEADS, 3, FM_HEAD_DIM])
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG,
                                    memory_config=_L1), memory_config=_L1)
        h = self._norm(x, w["fn"])
        # NOTES.md [flow-12] -- SiLU rides along on the w1 matmul instead of being its...
        g = ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1,
                        activation="silu")
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG,
                                         memory_config=_L1), memory_config=_L1)
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG,
                                       memory_config=_L1), memory_config=_L1)

    def _up(self, t, dtype=None):
        return ttnn.from_torch(t.contiguous(), dtype=dtype or self.dtype,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    def _trunk(self, p0, p1, p2, B):
        """three [B,1,3072] projections -> velocity [B,1,36]. The 3-token sequence, reference order.

        B here is the CFG-doubled batch: decode_frame passes 2*batch.

        CALLER SUPPLIES [B,1,3072], not [B,3072] -- see NOTES.md [flow-19]. The reshapes used to live
        here and cost 10.1 us a step for nothing: p0 arrives in that shape already, and p1 and p2
        change far less often than this runs."""
        seq = ttnn.concat([p0, p1, p2], dim=1)
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
                # already [B,1,3072]: _trunk wants that shape and these are constant for the life of
                # the model, so the reshape is paid once here instead of n_steps times per frame.
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
        """h [B,3072] -> [B,1]. Masked greedy argmax, on device in fp32 -- see semantic_dev."""
        B = llm_hidden.shape[0]
        h = ttnn.from_torch(llm_hidden.reshape(1, B, -1).float().contiguous(),
                            dtype=SEMANTIC_DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        logits = ttnn.add(ttnn.linear(h, self.semantic_dev, compute_kernel_config=COMPUTE_CONFIG),
                          self.semantic_mask)
        return ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(B, 1).long()

    def _solve(self, x, h, B, n_steps, cfg_alpha):
        """(x0 fp32 [B,1,36], cond++uncond [2B,3072]) -> x fp32 [B,1,36]. PURE DEVICE GRAPH.

        See NOTES.md [flow-17].
        """
        B2 = 2 * B
        # llm conditioning is constant across the solve, so project it ONCE rather than per step
        # (it was n_steps identical 3072x3072 matmuls). Reshaped once here for the same reason:
        # _trunk wants [B,1,3072] and p2 changes once a frame, not once a step -- NOTES.md [flow-19].
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
