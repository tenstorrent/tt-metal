# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral-TTS flow-matching acoustic transformer (FLOW MODEL, 390M).

Mirrors reference/voxtral_flow_ref.py op for op. Per frame: h [B,3072] -> semantic code [B,1]
(device fp32 matmul, host mask + argmax) and 36 acoustic codes from an Euler solve of a 3-layer
bidirectional transformer over a 3-token sequence, CFG batched to 2B.

    pytest models/experimental/voxtral_tts/tests/pcc/test_flow_pcc.py   # on device
"""

import os

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
# Matmul fidelity for the token-major (B > 1) blocks only, for the fidelity experiment: at 192
# folded rows the matmuls are compute-bound, so HiFi2 would halve their time; the fp32-reference
# gate in tests/perf/test_flow_tm.py decides whether it is acceptable. Default HiFi4 (= COMPUTE_CONFIG).
_FID = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}
COMPUTE_CONFIG_TM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=_FID.get(os.environ.get("VOXTRAL_FLOW_FIDELITY", "hifi4").lower(), ttnn.MathFidelity.HiFi4),
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
        self._grid = decode_grid(device.compute_with_storage_grid_size())
        self.decode_prg = decode_program_configs(self._grid)
        self._prg_by_tiles = {1: self.decode_prg}  # folded-row tiles -> matmul program configs
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

    def _prg(self, rows):
        """Matmul program configs for `rows` folded rows: the backbone's decode configs for one tile
        (B=1: 6 rows), per_core_M = tiles otherwise (B users: 2*B*3 rows). Built once per tile count."""
        m = -(-int(rows) // 32)
        prg = self._prg_by_tiles.get(m)
        if prg is None:
            if m > 1 and os.environ.get("VOXTRAL_FLOW_PRG", "mcast1d") == "default":
                # ttnn's own matmul choice for multi-tile rows; for the B>1 timing comparison.
                prg = {k: None for k in decode_program_configs(self._grid)}
            else:
                prg = decode_program_configs(self._grid, m_tiles=m)
            self._prg_by_tiles[m] = prg
        return prg

    def _block(self, x, w, B):
        """x [1,B*3,3072] -> same. Pre-norm, GQA 32/8, unmasked attention, SwiGLU."""
        prg = self._prg(B * 3)
        h = self._norm(x, w["an"])
        # q, k and v in one matmul, on the backbone's program config.
        qkv = ttnn.linear(h, w["wqkv"], program_config=prg["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
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
                program_config=prg["wo"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=_L1,
            ),
        )
        h = self._norm(x, w["fn"])
        # SiLU is fused by the w1 program config, not by an activation kwarg.
        g = ttnn.linear(h, w["w1"], program_config=prg["w1"], compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1)
        u = ttnn.multiply_(
            g,
            ttnn.linear(
                h,
                w["w3"],
                program_config=prg["w3"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=_L1,
            ),
        )
        return ttnn.add_(
            x,
            ttnn.linear(
                u,
                w["w2"],
                program_config=prg["w2"],
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

        No host ops in here, so it stays traceable. B > 1 takes the token-major solve (`_solve_tm`:
        1.9x faster at B=32 and closer to the fp32 reference at every B measured) unless
        VOXTRAL_FLOW_TM=0; B = 1 keeps this path, bit for bit.
        """
        if B > 1 and os.environ.get("VOXTRAL_FLOW_TM", "1") != "0":  # default for B > 1; see _solve_tm
            return self._solve_tm(x, h, B, n_steps, cfg_alpha)
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

    # ----------------------------------------------------------------------------------
    # Token-major solve for B > 1 users.
    #
    # The 3-token sequence per CFG row pads to 32 rows per head in tile layout, so at 2B = 64 rows
    # the batch-major `_block` spends most of its time moving padded head tensors (measured: 17.9 of
    # 35.7 ms per frame in the blocks at B=32, plus 8.2 ms of padded reshapes in the per-step glue,
    # against 1.5 ms of attention arithmetic). Here the folded rows are ordered token-major
    # (row = token * 2B + cfg_row), so the sequence is a plain concat of three [1, 2B, 3072]
    # tensors, attention runs ONCE over all 3*2B rows with a block-diagonal mask through the fused
    # head ops the backbone prefill uses, and the solver state stays [1, B, 36]. Same math as
    # `_block`/`_trunk`; only the row order and the op choice differ.
    # ----------------------------------------------------------------------------------
    def _tm_rows(self, B2):
        """Folded rows 3*B2, padded up to a tile multiple (sdpa is wrong on unaligned rows)."""
        rows = 3 * B2
        return -(-rows // 32) * 32

    def _tm_mask(self, rows_pad):
        """[1, 1, rows_pad, rows_pad] additive mask: 0 where two rows belong to the same CFG row
        (row r of the real 3*B2 belongs to CFG row r % B2), -1e9 elsewhere; padding rows attend
        only themselves. Built once per (padded row count, B2): two batch sizes can pad to the same row count
        (B2=8 and B2=10 both give 32) but need different block-diagonal masks."""
        B2 = self._tm_B2
        key = ("mask", rows_pad, B2)
        m = self._sched.get(key)
        if m is None:
            rows = 3 * B2
            r = torch.arange(rows_pad)
            owner = torch.where(r < rows, r % B2, B2 + r)  # padding rows get unique owners
            same = owner.reshape(-1, 1) == owner.reshape(1, -1)
            m = ttnn.from_torch(
                torch.where(same, 0.0, -1e9).reshape(1, 1, rows_pad, rows_pad).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._sched[key] = m
        return m

    def _tm_sdpa_prg(self):
        key = ("sdpa_tm",)
        p = self._sched.get(key)
        if p is None:
            g = self.device.compute_with_storage_grid_size()
            p = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(min(8, g.x), min(8, g.y)),
                q_chunk_size=32,
                k_chunk_size=32,
                exp_approx_mode=False,
            )
            self._sched[key] = p
        return p

    def _schedule_tm(self, B2, n_steps):
        """The time tokens of `_schedule`, built directly as [1, B2, 3072] rows (a reshape of the
        padded [B2, 1, 3072] tensors is NOT exact once B2 spans more than one tile)."""
        key = ("tm", B2, n_steps)
        if key not in self._sched:
            ts = torch.linspace(0, 1, n_steps + 1)
            toks = []
            for i in range(n_steps):
                emb = time_embedding(ts[i].view(1, 1).repeat(B2, 1), self.inv_freq)  # [B2, D]
                p = ttnn.linear(
                    self._up(emb.reshape(1, B2, -1)), self.proj["time_projection"], compute_kernel_config=COMPUTE_CONFIG
                )
                toks.append(p)  # [1, B2, 3072]
            self._sched[key] = (toks, [float(ts[i + 1] - ts[i]) for i in range(n_steps)])
        return self._sched[key]

    @staticmethod
    def _as_rows(t, rows, width):
        """[rows, 1, width] (padded tiles) or [rows, width] -> [1, rows, width], exactly: through a
        row-major view when the source is tiled with a padded middle dim."""
        if tuple(t.shape) == (1, rows, width):
            return t
        if len(t.shape) == 2:
            return ttnn.reshape(t, [1, rows, width])
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(rm, [1, rows, width]), ttnn.TILE_LAYOUT)

    def _block_tm(self, x, w, B2):
        """x [1, rows_pad, 3072] token-major -> same. Pre-norm, GQA 32/8 masked to each CFG row's
        own 3 tokens, SwiGLU. rows_pad = 3*B2 rounded up to a tile multiple."""
        rows = self._tm_rows(B2)
        self._tm_B2 = B2
        prg = self._prg(rows)
        h = self._norm(x, w["an"])
        qkv = ttnn.linear(h, w["wqkv"], program_config=prg["wqkv"], compute_kernel_config=COMPUTE_CONFIG_TM)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [1, 1, rows, _QKV_WIDTH]),
            num_heads=FM_N_HEADS,
            num_kv_heads=FM_N_KV_HEADS,
            transpose_k_heads=False,
            memory_config=_L1,
        )
        # Explicit 32-row chunks with exact exp: the default sdpa config at these shapes reads
        # PCC 0.9999 / max|diff| 0.08 against torch, this one 1.0000 / 0.03 (bf16 noise).
        a = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=self._tm_mask(rows),
            is_causal=False,
            scale=1.0,
            program_config=self._tm_sdpa_prg(),
            compute_kernel_config=COMPUTE_CONFIG,
        )
        a = ttnn.reshape(ttnn.experimental.nlp_concat_heads(a, memory_config=_L1), [1, rows, FM_N_HEADS * FM_HEAD_DIM])
        x = ttnn.add_(
            x,
            ttnn.linear(
                a, w["wo"], program_config=prg["wo"], compute_kernel_config=COMPUTE_CONFIG_TM, memory_config=_L1
            ),
        )
        h = self._norm(x, w["fn"])
        g = ttnn.linear(
            h, w["w1"], program_config=prg["w1"], compute_kernel_config=COMPUTE_CONFIG_TM, memory_config=_L1
        )
        u = ttnn.multiply_(
            g,
            ttnn.linear(
                h, w["w3"], program_config=prg["w3"], compute_kernel_config=COMPUTE_CONFIG_TM, memory_config=_L1
            ),
        )
        return ttnn.add_(
            x,
            ttnn.linear(
                u, w["w2"], program_config=prg["w2"], compute_kernel_config=COMPUTE_CONFIG_TM, memory_config=_L1
            ),
        )

    def _trunk_tm(self, p0, p1, p2, B2):
        """three [1, B2, 3072] projections -> velocity [1, B2, 36] (token 0's rows)."""
        seq = ttnn.concat([p0, p1, p2], dim=1, memory_config=_L1)  # [1, 3*B2, 3072], token-major
        rows_pad = self._tm_rows(B2)
        if rows_pad != 3 * B2:
            # keyed by B2 too: B2=8 and B2=10 both pad to 32 rows but need 8 and 2 pad rows (as _tm_mask)
            key = ("pad", rows_pad, B2)
            pad = self._sched.get(key)
            if pad is None:
                pad = self._sched[key] = self._up(torch.zeros(1, rows_pad - 3 * B2, FM_INPUT_DIM))
            seq = ttnn.concat([seq, pad], dim=1, memory_config=_L1)
        self._tm_B2 = B2
        for w in self.layers:
            seq = self._block_tm(seq, w, B2)
        seq = self._norm(seq, self.norm)
        out = ttnn.linear(seq, self.proj["acoustic_codebook_output"], compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.slice(out, [0, 0, 0], [1, B2, N_ACOUSTIC_CODEBOOK])

    def _solve_tm(self, x, h, B, n_steps, cfg_alpha):
        """(x0 fp32 [B,1,36] or [1,B,36], cond++uncond [2B,...,3072]) -> x fp32 [1, B, 36]. Pure device
        graph, token-major rows. Callers reshape the result to [B, 36] on the host."""
        B2 = 2 * B
        x = self._as_rows(x, B, N_ACOUSTIC_CODEBOOK)
        h = self._as_rows(h, B2, FM_INPUT_DIM)
        p2 = ttnn.linear(h, self.proj["llm_projection"], compute_kernel_config=COMPUTE_CONFIG)  # [1, B2, 3072]
        p1s, dts = self._schedule_tm(B2, n_steps)
        for i, dt in enumerate(dts):
            x2 = ttnn.concat([x, x], dim=1)  # [1, B2, 36]: cond rows then uncond rows
            p0 = ttnn.linear(
                ttnn.typecast(x2, self.dtype), self.proj["input_projection"], compute_kernel_config=COMPUTE_CONFIG
            )
            v = ttnn.typecast(self._trunk_tm(p0, p1s[i], p2, B2), ttnn.float32)  # [1, B2, 36]
            v_cond = ttnn.slice(v, [0, 0, 0], [1, B, N_ACOUSTIC_CODEBOOK])
            v_unc = ttnn.slice(v, [0, B, 0], [1, B2, N_ACOUSTIC_CODEBOOK])
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
