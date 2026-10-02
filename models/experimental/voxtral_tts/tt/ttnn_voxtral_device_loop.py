# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase B: the per-frame sampling glue on device, so a decode step is one trace replay.

Between two frames the batched pipeline used to do, on the host: read the semantic logits and the
flow output, argmax, FSQ-round, decide which rows stopped, look up and sum the 37 next-frame
embeddings, advance the positions, pick the next noise row and copy everything back. `DeviceFrameLoop`
does all of that as device ops appended to the traced frame graph, carrying the loop state in
persistent device tensors:

  stopped [1,B,1] f32   1.0 once a row emitted [END_AUDIO] or reached its cap
  pos     [1,B,1] f32   the row's next KV slot (frozen once stopped)
  tB      [1,B,1] f32   index of the frame being produced (same value in every row)
  t1      [1,1,1] f32   the same, as the index for the noise lookup
  cache   [B,1,F,64] f32  the codes of every produced frame (38 columns used), written with paged_update_cache
  caps    [1,B,1] f32   per-row frame cap (input per batch)
  noise   3 x [F, B*36] bf16 row-major  the fp32 noise split into three bf16 pieces that sum back exactly

Integer-valued quantities live in fp32 (exact below 2^24) because the elementwise ops are richest
there; they are typecast to uint32/int32 only at the consumers (argmax output, embedding indices,
cache index, the backbone's position tensors). The next input embedding is sum_k table[code_k +
offset_k]: three bf16 lookups (the fp32 table split into exact bf16 pieces) and one matmul with a 0/1 selection
matrix [B, B*64] (each row a tile-aligned block) do the gather-and-sum with fp32 accumulation (three bf16 pieces reproduce the fp32
table exactly).

The host reads the stopped mask every `check_every` frames (one tiny readback) to end the batch early,
and the codes cache once at the end.
"""

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    END_AUDIO_ID,
    N_ACOUSTIC_CODEBOOK,
    N_AUDIO_SPECIAL,
    NUM_CODEBOOKS,
    codebook_offsets,
)

TILE = 32
CODES_W = 64  # 37 codes (semantic as two columns) padded to two tiles
SEM_SPLIT = 128
F32, BF16, U32, I32 = ttnn.float32, ttnn.bfloat16, ttnn.uint32, ttnn.int32
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def split_bf16(x, pieces):
    """fp32 -> `pieces` bf16 tensors whose sum reproduces x (2 pieces: 16 mantissa bits, 3: exact)."""
    out, rest = [], x.float()
    for _ in range(pieces):
        p = rest.to(torch.bfloat16)
        out.append(p)
        rest = rest - p.float()
    return out


def _shard_grid(B):
    """B cores as one rectangle of an 8-wide grid, for the paged_update_cache input."""
    if B <= 8:
        return ttnn.CoreGrid(x=B, y=1)
    assert B % 8 == 0, f"device loop supports B <= 8 or a multiple of 8, got {B}"
    return ttnn.CoreGrid(x=8, y=B // 8)


class DeviceFrameLoop:
    def __init__(self, device, B, max_frames, audio_embeddings, semantic_mask, xin_dtype, check_every=8):
        """audio_embeddings: torch fp32 [V, 3072] (the backbone's flat table); semantic_mask: torch
        fp32 [vocab] (-1e9 on the ids the greedy pick may not choose); xin_dtype: dtype of the
        backbone's input buffer."""
        self.dev, self.B = device, int(B)
        self.F = -(-int(max_frames) // TILE) * TILE
        self.xin_dtype = xin_dtype
        self.check_every = int(check_every)
        dv = lambda t, d, layout=TL: ttnn.from_torch(t.contiguous(), dtype=d, layout=layout, device=device)
        B = self.B
        # ---- constants ----
        self.mask = dv(semantic_mask.reshape(1, 1, -1).float(), F32)  # [1,1,vocab]
        self.offs = dv(codebook_offsets().float().reshape(1, 1, NUM_CODEBOOKS), F32)  # [1,1,37]
        self.tabs = [dv(t, BF16, RM) for t in split_bf16(audio_embeddings, 3)]  # [V,3072] row-major bf16 pieces
        # Each row's 37 codes get their own tile-aligned block of K_ROW columns. Packed at stride 37,
        # a row's terms straddle tile boundaries at a row-dependent offset, the fp32 partial sums group
        # differently, and identical codes in two rows give embeddings that differ in the last bit
        # (only rows b = 0 mod 8 agree), so the same request decodes differently in different slots.
        self.K_ROW = -(-NUM_CODEBOOKS // TILE) * TILE  # 64
        self.N = B * self.K_ROW
        sel = torch.zeros(1, B, self.N)
        for b in range(B):
            sel[0, b, b * self.K_ROW : b * self.K_ROW + NUM_CODEBOOKS] = 1.0
        self.sel = dv(sel, BF16)  # [1,B,B*K_ROW] 0/1 (exact in bf16)
        self.zeros_pad_rm = dv(torch.zeros(1, B, CODES_W - NUM_CODEBOOKS - 1), F32, RM)
        self.cache_in_cfg = ttnn.create_sharded_memory_config(
            shape=(TILE, CODES_W),
            core_grid=_shard_grid(B),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # ---- per-batch inputs ----
        self.caps = dv(torch.zeros(1, B, 1), F32)
        self.noise = [dv(torch.zeros(self.F, B * N_ACOUSTIC_CODEBOOK), BF16, RM) for _ in range(3)]
        # ---- loop state ----
        self.stopped = dv(torch.zeros(1, B, 1), F32)
        self.pos = dv(torch.zeros(1, B, 1), F32)
        self.tB = dv(torch.zeros(1, B, 1), F32)
        self.t1 = dv(torch.zeros(1, 1, 1), F32)
        self.cache = dv(torch.zeros(B, 1, self.F, CODES_W), F32)

    # ------------------------------------------------------------------
    # per batch (host -> persistent buffers, no allocation)
    # ------------------------------------------------------------------
    def seed0(self, buf, lens, caps, x0_all):
        """lens [B] prompt lengths | caps [B] frame caps | x0_all [F,B,36] fp32 noise (row t is the
        noise of frame t). State for producing frame 0 through sample_and_advance: nothing stopped,
        positions one slot back (the sampler advances live rows), frame index 0, frame 0's noise in
        the pipeline's x0 buffer."""
        B = self.B
        F = x0_all.shape[0]
        assert F <= self.F, f"{F} frames > device loop capacity {self.F}"
        col = lambda t: t.reshape(1, B, 1).float().contiguous()
        put = lambda host, dst, layout=TL: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=dst.dtype, layout=layout), dst
        )
        put(torch.zeros(1, B, 1), self.stopped)
        put(col(lens.float() - 1.0), self.pos)
        put(col(caps.float()), self.caps)
        put(torch.zeros(1, B, 1), self.tB)
        put(torch.zeros(1, 1, 1), self.t1)
        table = torch.zeros(self.F, B * N_ACOUSTIC_CODEBOOK)
        table[:F] = x0_all.reshape(F, -1).float()
        for piece, dst in zip(split_bf16(table, 3), self.noise):
            put(piece, dst, RM)
        put(x0_all[0].reshape(1, B, N_ACOUSTIC_CODEBOOK).float().contiguous(), buf["x0"])

    def read_stopped(self):
        return ttnn.to_torch(self.stopped).reshape(-1) > 0.5

    def read_codes(self):
        """-> torch int64 [B, F, 37]: the codes of frame t at [:, t]."""
        rec = ttnn.to_torch(self.cache)[:, 0, :, : NUM_CODEBOOKS + 1].round().long()
        sem = rec[:, :, 0] * SEM_SPLIT + rec[:, :, 1]
        return torch.cat([sem.unsqueeze(-1), rec[:, :, 2:]], dim=-1)

    # ------------------------------------------------------------------
    # inside the traced graph
    # ------------------------------------------------------------------
    def sample_and_advance(self, lg, xr, buf):
        """lg [1,B,vocab] f32 semantic logits, xr [1,B,36] (or [B,1,36]) f32 flow output -> writes
        the codes of this frame into the cache, the next frame's inputs into `buf` and the loop state
        into itself. Returns nothing the host needs."""
        B = self.B
        Bpad = -(-B // TILE) * TILE
        # --- semantic code: masked argmax on device (row-major input: multi-core path) ---
        sem_u = ttnn.argmax(ttnn.to_layout(ttnn.add(lg, self.mask), RM), dim=-1, keepdim=True)  # [1,B,1] u32
        sem_f_rm = ttnn.typecast(sem_u, F32)
        sem_f = ttnn.to_layout(sem_f_rm, TL)
        # --- acoustic codes: FSQ, exact integers 0..20 ---
        if tuple(xr.shape) != (1, B, N_ACOUSTIC_CODEBOOK):
            xr = ttnn.reshape(xr, [1, B, N_ACOUSTIC_CODEBOOK])
        ac = ttnn.round(
            ttnn.multiply(ttnn.add(ttnn.clamp(xr, min=-1.0, max=1.0), 1.0), (ACOUSTIC_CODEBOOK_SIZE - 1) / 2.0)
        )
        # --- stop mask: END now, or at the cap, or already stopped ---
        end = ttnn.eq(sem_f, float(END_AUDIO_ID))
        capped = ttnn.ge(self.tB, self.caps)
        stopped = ttnn.maximum(ttnn.maximum(self.stopped, end), capped)  # [1,B,1]
        live = ttnn.rsub(stopped, 1.0)  # 1 - stopped
        ac_codes = ttnn.add(ttnn.multiply(ac, live), float(N_AUDIO_SPECIAL))  # stopped rows: EMPTY + offset
        ac_codes_rm = ttnn.to_layout(ac_codes, RM)
        # --- this frame's codes into the cache at index tB ---
        # The record. A tile copy (paged_update_cache) goes through the 19-bit source registers, so an
        # integer above 2048 is not stored exactly; the semantic code (< 8320) is stored as quotient and
        # remainder by 128 (both < 128) and reassembled by read_codes().
        sem_hi = ttnn.floor(ttnn.multiply(sem_f, 1.0 / SEM_SPLIT))
        sem_lo = ttnn.subtract(sem_f, ttnn.multiply(sem_hi, float(SEM_SPLIT)))
        codes_rm = ttnn.concat(
            [ttnn.to_layout(sem_hi, RM), ttnn.to_layout(sem_lo, RM), ac_codes_rm, self.zeros_pad_rm], dim=-1
        )  # [1,B,64] f32 RM: [hi, lo, 36 acoustic, 26 zeros]
        cache_in = ttnn.pad(ttnn.reshape(codes_rm, [1, B, 1, CODES_W]), [(0, 0), (0, 0), (0, TILE - 1), (0, 0)], 0.0)
        cache_in = ttnn.to_memory_config(ttnn.to_layout(cache_in, TL), self.cache_in_cfg)
        t_i32 = ttnn.reshape(ttnn.to_layout(ttnn.typecast(self.tB, I32), RM), [B])
        # fp32 accumulate: without it the tile copy packs through a 16-bit register and truncates
        # the integer codes above 256 to bf16 spacing (3247 -> 3232).
        ttnn.experimental.paged_update_cache(
            self.cache, cache_in, update_idxs_tensor=t_i32, compute_kernel_config=_HIFI4
        )
        # --- next input embedding: sum_k table[code_k + offset_k] ---
        idx = ttnn.add(ttnn.to_layout(ttnn.concat([sem_f_rm, ac_codes_rm], dim=-1), TL), self.offs)  # [1,B,37] f32
        idx = ttnn.pad(
            ttnn.to_layout(ttnn.typecast(idx, U32), RM), [(0, 0), (0, 0), (0, self.K_ROW - NUM_CODEBOOKS)], 0
        )
        idx = ttnn.reshape(idx, [1, self.N])  # row b's codes at b*K_ROW.., index 0 in the pad (sel is 0 there)
        x_new = None
        for tab in self.tabs:
            e = ttnn.embedding(idx, tab, layout=TL)  # [1,B*K_ROW,3072] bf16
            part = ttnn.matmul(self.sel, e, dtype=F32, compute_kernel_config=_HIFI4)  # [1,B,3072] f32
            x_new = part if x_new is None else ttnn.add(x_new, part)
        ttnn.copy(ttnn.typecast(x_new, self.xin_dtype), buf["xin"])
        # --- positions: live rows advance one slot ---
        pos = ttnn.add(self.pos, live)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos, I32), RM), [B]), buf["pos_i32"])
        pos_u = ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos, U32), RM), [1, B])
        if Bpad > B:
            pos_u = ttnn.pad(pos_u, [(0, 0), (0, Bpad - B)], 0)
        ttnn.copy(pos_u, buf["pos_u32"])
        # --- frame counters and the next frame's noise ---
        tB = ttnn.add(self.tB, 1.0)
        t1 = ttnn.add(self.t1, 1.0)
        t_idx = ttnn.reshape(ttnn.to_layout(ttnn.typecast(t1, U32), RM), [1, 1])
        x0 = None
        for tab in self.noise:
            piece = ttnn.embedding(t_idx, tab)  # [1,1,B*36] bf16 RM
            piece = ttnn.typecast(ttnn.to_layout(ttnn.reshape(piece, [1, B, N_ACOUSTIC_CODEBOOK]), TL), F32)
            x0 = piece if x0 is None else ttnn.add(x0, piece)
        ttnn.copy(x0, buf["x0"])
        # --- loop state ---
        ttnn.copy(stopped, self.stopped)
        ttnn.copy(pos, self.pos)
        ttnn.copy(tB, self.tB)
        ttnn.copy(t1, self.t1)
