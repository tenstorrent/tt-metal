# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral-TTS flow-matching acoustic transformer (FLOW MODEL, 390M).

Mirrors reference/voxtral_flow_ref.py op for op. Per frame: h [B,3072] -> semantic code [B,1]
(device fp32 matmul, host mask + argmax) and 36 acoustic codes from an Euler solve of a 3-layer
bidirectional transformer over a 3-token sequence, CFG batched to 2B.

    pytest models/experimental/voxtral_tts/tests/pcc/test_flow_pcc.py   # on device
"""

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_flow_ref import (
    _fsq_quantize,
    load_flow_state,
    time_embedding,
)

# Same dims as the backbone and at most one tile of rows, so its decode matmul program configs apply
# unchanged (gpt does not import flow, so no cycle).
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import decode_grid, decode_program_configs, sharded_norm
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    CFG_ALPHA,
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
    N_DECODING_STEPS,
    SEMANTIC_CODEBOOK_SIZE,
)

SCALE = FM_HEAD_DIM**-0.5
# Fused q++k++v width. The sub-widths are _split_heads' slice offsets, all tile-aligned.
_Q_WIDTH = FM_N_HEADS * FM_HEAD_DIM
_KV_WIDTH = FM_N_KV_HEADS * FM_HEAD_DIM
_QKV_WIDTH = _Q_WIDTH + 2 * _KV_WIDTH

# Every intermediate inside _block lives in L1: all are small and consumed within a few ops.
_L1 = ttnn.L1_MEMORY_CONFIG


def _split_heads(qkv, B):
    """[1,B*3,6144] -> q [B,32,3,128], k/v [B,8,3,128], by slice/reshape/permute rather than the
    fused `nlp_create_qkv_heads`, which is slower at this shape."""
    t = ttnn.reshape(qkv, [B, 1, 3, _QKV_WIDTH])

    def take(lo, hi, nh):
        # memory_config=_L1 on both is load-bearing: without it q/k/v land in DRAM.
        s = ttnn.slice(t, [0, 0, 0, lo], [B, 1, 3, hi], memory_config=_L1)
        return ttnn.permute(ttnn.reshape(s, [B, 3, nh, FM_HEAD_DIM]), (0, 2, 1, 3), memory_config=_L1)

    return (
        take(0, _Q_WIDTH, FM_N_HEADS),
        take(_Q_WIDTH, _Q_WIDTH + _KV_WIDTH, FM_N_KV_HEADS),
        take(_Q_WIDTH + _KV_WIDTH, _QKV_WIDTH, FM_N_KV_HEADS),
    )


# HiFi4 + fp32 accumulation for the velocity network: lower fidelity is both slower and less accurate here.
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
# Activation dtype; every op inherits it from its input. Accumulation stays fp32 (COMPUTE_CONFIG).
DTYPE = ttnn.bfloat16

# Matmul weight storage, independent of the activation dtype.
WEIGHT_DTYPE = ttnn.bfloat8_b

# The semantic head emits an index, so it stays fp32: a rounding difference there is a different code.
SEMANTIC_DTYPE = ttnn.float32


class TtVoxtralFlow:
    """The flow model on device. __call__(h) -> audio_codes torch [B,37] int64."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT):
        self.device = device
        self.decode_prg = decode_program_configs(decode_grid(device.compute_with_storage_grid_size()))
        self.dtype = DTYPE
        w = load_flow_state(ckpt_path)
        self.inv_freq = w["time_embedding.inv_freq"]  # host: time_embedding
        self._sched = {}  # (batch, n_steps) -> (time tokens, step widths)
        self._cfgbuf = {}  # batch -> reused [2B,3072] cond++uncond host buffer

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
        # RMSNorm gammas stay at the activation dtype, having no bandwidth to save; only matmul weights
        # take WEIGHT_DTYPE.
        vec = lambda t: up(t.reshape(1, 1, -1), DTYPE)
        lin = lambda t: up(t.t(), WEIGHT_DTYPE)  # torch [out,in] -> ttnn.linear wants [in,out]

        # Semantic head: device matmul in fp32.
        self.semantic_dev = up(w["semantic_codebook_output.weight"].float().t(), SEMANTIC_DTYPE)
        _vocab = w["semantic_codebook_output.weight"].shape[0]
        _mask = torch.zeros(1, 1, _vocab)
        _mask[:, :, EMPTY_AUDIO_ID] = -1e9
        _mask[:, :, N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE :] = -1e9
        # The mask add and the argmax run on the host, where they are cheaper than on device.
        self.semantic_mask_host = _mask.reshape(-1).float()

        self.proj = {
            k: lin(w[f"{k}.weight"])
            for k in ("input_projection", "time_projection", "llm_projection", "acoustic_codebook_output")
        }
        self.norm = vec(w["norm.weight"])
        self.layers = []
        for i in range(FM_N_LAYERS):
            p = f"layers.{i}."
            self.layers.append(
                {
                    "an": vec(w[p + "attention_norm.weight"]),
                    "fn": vec(w[p + "ffn_norm.weight"]),
                    # q, k, v fused into one weight, SCALE folded into the q rows.
                    "wqkv": lin(
                        torch.cat(
                            [
                                w[p + "attention.wq.weight"] * SCALE,
                                w[p + "attention.wk.weight"],
                                w[p + "attention.wv.weight"],
                            ],
                            dim=0,
                        )
                    ),
                    "wo": lin(w[p + "attention.wo.weight"]),
                    "w1": lin(w[p + "feed_forward.w1.weight"]),
                    "w2": lin(w[p + "feed_forward.w2.weight"]),
                    "w3": lin(w[p + "feed_forward.w3.weight"]),
                }
            )

    # ----------------------------------------------------------------------------------
    # One bidirectional block over the 3-token sequence
    # ----------------------------------------------------------------------------------
    def _norm(self, x, gamma):
        """RMSNorm, width-sharded (the backbone's `sharded_norm`)."""
        return sharded_norm(x, gamma, FM_NORM_EPS, _L1)

    def _block(self, x, w, B):
        """x [1,B*3,3072] -> same. Pre-norm, GQA 32/8, unmasked attention, SwiGLU."""
        h = self._norm(x, w["an"])
        # q, k and v in one matmul, on the backbone's program config.
        qkv = ttnn.linear(h, w["wqkv"], program_config=self.decode_prg["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
        # hand-rolled head split
        qh, kh, vh = _split_heads(qkv, B)
        # sdpa handles GQA natively. scale=1.0 is mandatory: SCALE is already folded into wqkv's
        # q rows.
        a = ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=COMPUTE_CONFIG
        )
        # back to folded rows so wo and the MLP get the single-weight-read layout too
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        # In place; safe only because _trunk puts the residual stream in L1.
        x = ttnn.add_(
            x,
            ttnn.linear(
                a,
                w["wo"],
                program_config=self.decode_prg["wo"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=_L1,
            ),
        )
        h = self._norm(x, w["fn"])
        # SiLU is fused by the w1 program config, not by an activation kwarg.
        g = ttnn.linear(
            h, w["w1"], program_config=self.decode_prg["w1"], compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1
        )
        u = ttnn.multiply_(
            g,
            ttnn.linear(
                h,
                w["w3"],
                program_config=self.decode_prg["w3"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=_L1,
            ),
        )
        return ttnn.add_(
            x,
            ttnn.linear(
                u,
                w["w2"],
                program_config=self.decode_prg["w2"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=_L1,
            ),
        )

    def _up(self, t, dtype=None):
        return ttnn.from_torch(t.contiguous(), dtype=dtype or self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _trunk(self, p0, p1, p2, B):
        """three [B,1,3072] projections -> velocity [B,1,36]. The 3-token sequence, reference order.

        B is the CFG-doubled batch.
        """
        # memory_config is load-bearing: add_ writes wherever x lives, so this puts the residual stream
        # in L1 for _block.
        seq = ttnn.concat([p0, p1, p2], dim=1, memory_config=_L1)
        # Fold the CFG batch into rows, so every matmul reads its weight once, not once per batch.
        seq = ttnn.reshape(seq, [1, B * 3, FM_INPUT_DIM])
        for w in self.layers:
            seq = self._block(seq, w, B)
        seq = self._norm(seq, self.norm)
        # Project all rows first, then narrow to position 0: both moves are then 36 wide, not 3072.
        out = ttnn.linear(seq, self.proj["acoustic_codebook_output"], compute_kernel_config=COMPUTE_CONFIG)
        out = ttnn.reshape(out, [B, 3, N_ACOUSTIC_CODEBOOK])
        return ttnn.slice(out, [0, 0, 0], [B, 1, N_ACOUSTIC_CODEBOOK])

    def _cfg_input(self, B, llm_hidden):
        """-> [2B, 3072] = llm_hidden (cond) over zeros (uncond), in a buffer reused per batch."""
        buf = self._cfgbuf.get(B)
        if buf is None:
            buf = self._cfgbuf[B] = torch.zeros(2 * B, FM_INPUT_DIM)
        buf[:B] = llm_hidden  # bottom half stays zero by construction
        return buf

    def _schedule(self, B, n_steps):
        """-> (time-conditioning tokens on device, per-step dt). Built once per (batch, n_steps): the
        schedule and its time tokens depend on nothing else."""
        key = (B, n_steps)
        if key not in self._sched:
            ts = torch.linspace(0, 1, n_steps + 1)
            self._sched[key] = (
                # already [B,1,3072]; constant for the model's life, so reshaped once here.
                [
                    ttnn.reshape(
                        ttnn.linear(
                            self._up(time_embedding(ts[i].view(1, 1).repeat(B, 1), self.inv_freq)),
                            self.proj["time_projection"],
                            compute_kernel_config=COMPUTE_CONFIG,
                        ),
                        [B, 1, FM_INPUT_DIM],
                    )
                    for i in range(n_steps)
                ],
                [float(ts[i + 1] - ts[i]) for i in range(n_steps)],
            )
        return self._sched[key]

    def _predict_velocity(self, x_t, llm_h, t_emb):
        """torch [B,36], [B,3072], [B,3072] -> velocity torch [B,36]. Position 0 only.

        Torch-in/torch-out entry point for the PCC tests; the solve in decode_frame bypasses it."""
        B = x_t.shape[0]
        p0 = ttnn.linear(self._up(x_t), self.proj["input_projection"], compute_kernel_config=COMPUTE_CONFIG)
        p1 = ttnn.linear(self._up(t_emb), self.proj["time_projection"], compute_kernel_config=COMPUTE_CONFIG)
        p2 = ttnn.linear(self._up(llm_h), self.proj["llm_projection"], compute_kernel_config=COMPUTE_CONFIG)
        # _trunk wants [B,1,3072]; these three arrive 2D because the inputs here are 2D torch
        v = self._trunk(*(ttnn.reshape(p, [B, 1, FM_INPUT_DIM]) for p in (p0, p1, p2)), B)
        return ttnn.to_torch(v).float().reshape(B, N_ACOUSTIC_CODEBOOK)

    # ----------------------------------------------------------------------------------
    # Semantic code (host) and the Euler solve (device velocity)
    # ----------------------------------------------------------------------------------
    def semantic_code(self, llm_hidden):
        """h [B,3072] -> [B,1]. Greedy argmax: the fp32 matmul on device, the mask and the reduce
        on the host.
        """
        B = llm_hidden.shape[0]
        h = ttnn.from_torch(
            llm_hidden.reshape(1, B, -1).float().contiguous(),
            dtype=SEMANTIC_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        logits = ttnn.to_torch(ttnn.linear(h, self.semantic_dev, compute_kernel_config=COMPUTE_CONFIG)).float()
        return (logits.reshape(B, -1) + self.semantic_mask_host).argmax(-1).reshape(B, 1).long()

    def _solve(self, x, h, B, n_steps, cfg_alpha):
        """(x0 fp32 [B,1,36], cond++uncond [2B,3072]) -> x fp32 [B,1,36]. PURE DEVICE GRAPH.

        No host ops in here, so it stays traceable.
        """
        B2 = 2 * B
        # the llm conditioning is constant across the solve: project and reshape it once per frame
        p2 = ttnn.reshape(
            ttnn.linear(h, self.proj["llm_projection"], compute_kernel_config=COMPUTE_CONFIG), [B2, 1, FM_INPUT_DIM]
        )
        p1s, dts = self._schedule(B2, n_steps)
        for i, dt in enumerate(dts):
            # cond+uncond as ONE 2B forward, matching the reference exactly.
            x2 = ttnn.concat([x, x], dim=0)
            p0 = ttnn.linear(
                ttnn.typecast(x2, self.dtype), self.proj["input_projection"], compute_kernel_config=COMPUTE_CONFIG
            )
            v = ttnn.typecast(self._trunk(p0, p1s[i], p2, B2), ttnn.float32)
            v_cond = ttnn.slice(v, [0, 0, 0], [B, 1, N_ACOUSTIC_CODEBOOK])
            v_unc = ttnn.slice(v, [B, 0, 0], [B2, 1, N_ACOUSTIC_CODEBOOK])
            v_cfg = ttnn.add(ttnn.multiply(v_cond, cfg_alpha), ttnn.multiply(v_unc, 1.0 - cfg_alpha))
            x = ttnn.add(x, ttnn.multiply(v_cfg, dt))
        return x

    @torch.no_grad()
    def decode_frame(self, sem_code, llm_hidden, cfg_alpha=CFG_ALPHA, n_steps=N_DECODING_STEPS, x_0=None):
        """[B,1], [B,3072] -> acoustic codes [B,36] int64, offset applied.

        Solver state and the CFG combine stay fp32.
        """
        B = sem_code.shape[0]
        should = (sem_code != END_AUDIO_ID).reshape(B)
        x0 = torch.randn(B, N_ACOUSTIC_CODEBOOK) if x_0 is None else x_0
        h_host = self._cfg_input(B, llm_hidden)

        x = self._solve(
            self._up(x0.reshape(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32), self._up(h_host), B, n_steps, cfg_alpha
        )

        codes = _fsq_quantize(ttnn.to_torch(x).float().reshape(B, N_ACOUSTIC_CODEBOOK))
        codes[~should] = EMPTY_AUDIO_ID
        return codes + N_AUDIO_SPECIAL

    @torch.no_grad()
    def __call__(self, llm_hidden, **kw):
        """h [B,3072] -> audio_codes [B,37] int64 (semantic ++ acoustic)."""
        sem = self.semantic_code(llm_hidden)
        return torch.cat([sem, self.decode_frame(sem, llm_hidden, **kw)], dim=1)
