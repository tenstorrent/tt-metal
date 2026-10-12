# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash vision encoder (DeepSeek-ViT + aligner) in ttnn, one device (tt-blaze vit_plan.md §4 V1).

model.py's tower (``inference/vision.py``), op for op:

    x = patch_embed(patches.flatten(1))                    Linear(588 -> 1024) with bias
    32 x  x = x + wo(SDPA(rope(q), rope(k), v))            wqkv Linear(1024 -> 3072) with bias, q/k/v CONTIGUOUS; 16 heads of 64
          x = x + w2(silu(gate) * up)                      w1 Linear(1024 -> 5632) no bias: gate = rows 0:2816, up = 2816:5632
    x = norm(x)                                            RMSNorm eps 1e-6, fp32 weight / statistics
    aligner: pad the [n_h, n_w, 1024] grid to multiples of 3 with ZEROS, F.unfold(3, stride 3) (channel-major 9216 =
             c * 9 + kh * 3 + kw), w2(gelu(w1(.))) with biases, exact GELU -> [n_llm_h * n_llm_w, 5120]

* 2D RoPE is rotate-half over the full head_dim 64: pair (i, i + 32) turns by ``h * inv_freq[i]`` (i < 16) or
  ``w * inv_freq[i - 16]`` (16 <= i < 32), so cos / sin of width 64 = cat(cos32, cos32) feed ``rotary_embedding``.
* Patches are padded to a multiple of 128 rows (the SDPA chunk); PatchEmbed's bias makes the pad rows non-zero, so the
  SDPA runs WINDOWED over ``[0, N, Np]``: real rows attend only to real rows. The pad rows are dropped before the aligner.
* The aligner's 3x3 unfold runs on the host (one [N, 1024] read-back; V1 correctness first), its two GEMMs on device.

Nothing in this module talks to the prefill: ``V41Vision.encode(patches, n_vit_h, n_vit_w)`` returns the rows that
model.py's ``merge_image_embeddings`` writes into the IMAGE slots of the span.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

import ttnn

PATCH = 14
DIM = 1024
HEADS = 16
HEAD_DIM = 64
INTER = 2816
RATIO = 3
EPS = 1e-6
# SDPA chunks. The flash kernel keeps its running output / max in bf16 CBs (sdpa_program_factory.cpp im_df), so every k chunk
# is one bf16 rescale-and-add: the error grows with N / k_chunk (DS41F-0045, T4's 8649 patches vs fp32, aligner rows per-row
# mean: k 128 0.99674 / 439 ms, k 512 0.99796 / 394 ms, k 1024 (q 64) 0.99817 / 440 ms, k 2048 (q 32) fails to build; the fp32
# math path 0.99927 / 7962 ms)
Q_CHUNK = int(os.environ.get("V41_VIT_Q_CHUNK", "128"))
K_CHUNK = int(os.environ.get("V41_VIT_K_CHUNK", "512"))
ROW_PAD = max(Q_CHUNK, K_CHUNK)  # the padded patch count is a multiple of both chunks
# V41_VIT_ATTN=math: explicit attention per block of query rows -- S = q k^T in fp32, the pad keys masked, an fp32 softmax,
# then P v -- no running bf16 rescale; slower, the precision reference for the flash path
ATTN_MODE = os.environ.get("V41_VIT_ATTN", "flash")
MATH_ROWS = int(os.environ.get("V41_VIT_MATH_ROWS", "1024"))
K_PAD = 608  # 588 = 3 * 14 * 14 -> 19 tiles


def vision_cos_sin(n_h: int, n_w: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """model.py ``get_vision_cos_sin`` widened to the full head_dim: [n_h * n_w, 64] fp32 cos / sin."""
    dim = HEAD_DIM // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1)  # [N, 32]
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def aligner_unfold(x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
    """The aligner's input [L, 9216] from the ViT output [n_h * n_w, 1024], exactly as model.py's Aligner (host)."""
    x = x.reshape(n_h, n_w, -1).permute(2, 0, 1)
    x = F.pad(x, (0, -n_w % RATIO, 0, -n_h % RATIO))
    return F.unfold(x.unsqueeze(0).float(), RATIO, stride=RATIO).squeeze(0).transpose(0, 1).to(x.dtype)


def load_vision_state(ckpt) -> dict[str, torch.Tensor]:
    """Every ``vision.*`` / ``aligner.*`` / ``image_*`` tensor of the checkpoint (all bf16, no scales)."""
    return {n: ckpt.raw(n) for n in ckpt.weight_map if n.startswith(("vision.", "aligner.", "image_"))}


class V41Vision:
    def __init__(
        self,
        device,
        state: dict[str, torch.Tensor],
        *,
        n_layers: int = 32,
        weight_dtype=ttnn.bfloat16,
        mlp_fp32: bool = False,
    ):
        self.device, self.n_layers, self.mlp_fp32 = device, n_layers, mlp_fp32
        # on a multi-chip mesh (the prefill runner's, DS41F-0049: a (1, 1) submesh of it hung the runner) every tensor is
        # replicated and the host reads chip 0's copy
        self.n_dev = int(device.get_num_devices()) if hasattr(device, "get_num_devices") else 1
        self.ck_hifi4 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def mat(w: torch.Tensor, pad_k: int | None = None):  # torch Linear [out, in] -> ttnn [in, out]
            t = w.t().contiguous()
            if pad_k is not None and pad_k > t.shape[0]:
                t = F.pad(t, (0, 0, 0, pad_k - t.shape[0]))
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def vec(b: torch.Tensor):
            return ttnn.from_torch(
                b.reshape(1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def norm_w(w: torch.Tensor):  # DS41F-0041: a ROW_MAJOR [1, 1, W/32, 32] weight (exact on every row)
            return ttnn.from_torch(
                w.float().reshape(1, 1, -1, 32).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        p = "vision."
        self.pe_w = mat(state[p + "patch_embed.proj.weight"], pad_k=K_PAD)
        self.pe_b = vec(state[p + "patch_embed.proj.bias"])
        self.blocks = []
        for i in range(n_layers):
            b = f"{p}blocks.{i}."
            self.blocks.append(
                dict(
                    n1=norm_w(state[b + "norm1.weight"]),
                    n2=norm_w(state[b + "norm2.weight"]),
                    wqkv=mat(state[b + "attn.wqkv.weight"]),
                    bqkv=vec(state[b + "attn.wqkv.bias"]),
                    wo=mat(state[b + "attn.wo.weight"]),
                    bo=vec(state[b + "attn.wo.bias"]),
                    w1=mat(state[b + "mlp.w1.weight"]),
                    w2=mat(state[b + "mlp.w2.weight"]),
                )
            )
        self.norm = norm_w(state[p + "norm.weight"])
        self.al_w1, self.al_b1 = mat(state["aligner.w1.weight"]), vec(state["aligner.w1.bias"])
        self.al_w2, self.al_b2 = mat(state["aligner.w2.weight"]), vec(state["aligner.w2.bias"])
        self.delims = {k: state[k].clone() for k in ("image_start", "image_end", "image_newline") if k in state}
        self.scale = 1.0 / math.sqrt(HEAD_DIM)
        self._rope: dict = {}

    # ── pieces ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _rope_tables(self, n_h: int, n_w: int, n_pad: int):
        key = (n_h, n_w, n_pad)
        if key not in self._rope:
            cos, sin = vision_cos_sin(n_h, n_w)
            cos, sin = (F.pad(t, (0, 0, 0, n_pad - t.shape[0])) for t in (cos, sin))
            self._rope[key] = tuple(
                ttnn.from_torch(
                    t.reshape(1, 1, n_pad, HEAD_DIM).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for t in (cos, sin)
            )
        return self._rope[key]

    def _math_attention(self, q, k, v, n_pad: int):
        """softmax(q k^T * scale + key_mask) v with an fp32 score / softmax, per block of MATH_ROWS query rows."""
        kt = ttnn.transpose(k, -2, -1)
        outs = []
        for r0 in range(0, n_pad, MATH_ROWS):
            r1 = min(n_pad, r0 + MATH_ROWS)
            qc = ttnn.slice(q, [0, 0, r0, 0], [1, HEADS, r1, HEAD_DIM])
            sc = ttnn.matmul(qc, kt, dtype=ttnn.float32, compute_kernel_config=self.ck_hifi4)
            ttnn.deallocate(qc)
            sc = ttnn.add(ttnn.multiply(sc, self.scale), self._key_mask, dtype=ttnn.float32)
            p = ttnn.softmax(sc, dim=-1, numeric_stable=True, compute_kernel_config=self.ck_hifi4)
            ttnn.deallocate(sc)
            outs.append(ttnn.matmul(p, v, dtype=ttnn.bfloat16, compute_kernel_config=self.ck_hifi4))
            ttnn.deallocate(p)
        ttnn.deallocate(kt)
        o = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        return o

    def _windows(self, n: int, n_pad: int):
        """The SDPA windows [0, n, n_pad] (real rows attend real rows only) and, for the math path, the key mask."""
        if ATTN_MODE == "math":
            m = torch.zeros(1, 1, 1, n_pad)
            m[..., n:] = float("-inf")
            self._key_mask = ttnn.from_torch(
                m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )  # row-broadcast
        bounds = [0, n] + ([n_pad] if n_pad > n else [])
        return ttnn.from_torch(
            torch.tensor(bounds, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=EPS, weight=w, compute_kernel_config=self.ck_hifi4)

    def _lin(self, x, w, b=None):
        return ttnn.linear(
            x,
            w,
            bias=b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi4,
        )

    def block(self, x, i: int, cos, sin, cu, taps: dict | None = None):
        """x [1, 1, Np, 1024] -> the next block's input (model.py ``Block.forward``). ``taps``: receives the attention
        sublayer output ``attn`` and the MLP output ``mlp`` (device tensors, before their residual adds)."""
        w = self.blocks[i]
        h = self._rms(x, w["n1"])
        qkv = self._lin(h, w["wqkv"], w["bqkv"])
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=HEADS, num_kv_heads=HEADS, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q2 = ttnn.experimental.rotary_embedding(q, cos, sin, compute_kernel_config=self.ck_hifi4)
        k2 = ttnn.experimental.rotary_embedding(k, cos, sin, compute_kernel_config=self.ck_hifi4)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        n_pad = x.shape[-2]
        if ATTN_MODE == "math":
            o = self._math_attention(q2, k2, v, n_pad)
        else:
            o = ttnn.transformer.scaled_dot_product_attention(
                q2,
                k2,
                v,
                is_causal=False,
                scale=self.scale,
                cu_window_seqlens=cu,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                    q_chunk_size=min(Q_CHUNK, n_pad),
                    k_chunk_size=min(K_CHUNK, n_pad),
                    exp_approx_mode=False,
                ),
                compute_kernel_config=self.ck_sdpa,
            )
        for t in (q2, k2, v):
            ttnn.deallocate(t)
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        a = self._lin(o2, w["wo"], w["bo"])
        ttnn.deallocate(o2)
        x = ttnn.add(x, a)
        if taps is not None:
            taps["attn"] = a
        else:
            ttnn.deallocate(a)
        h = self._rms(x, w["n2"])
        # mlp_fp32 (experiment, default off): gate / up and silu(gate) * up in fp32. From block 12 a few hidden units drive the
        # massive channels (308 / 53, +-1520) to ~0 on ordinary rows; fp32 here did NOT recover those rows (T1 block 12 per-row
        # min 0.62 -> 0.76, block 20 0.73 -> 0.25 vs fp32; the aligner rows unchanged at 0.96-0.97), so the error enters before
        # the MLP (tt-blaze DS41F-0045)
        gu = ttnn.linear(
            h,
            w["w1"],
            dtype=ttnn.float32 if self.mlp_fp32 else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi4,
        )
        ttnn.deallocate(h)
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, n_pad, INTER])
        up = ttnn.slice(gu, [0, 0, 0, INTER], [1, 1, n_pad, 2 * INTER])
        ttnn.deallocate(gu)
        if self.mlp_fp32:
            sg = ttnn.silu(gate)
            m = ttnn.mul(sg, up, dtype=ttnn.float32)
            ttnn.deallocate(sg)
        else:
            m = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        y = self._lin(m, w["w2"])
        ttnn.deallocate(m)
        out = ttnn.add(x, y)
        ttnn.deallocate(x)
        if taps is not None:
            taps["mlp"] = y
        else:
            ttnn.deallocate(y)
        return out

    def block_forced(self, x_in: torch.Tensor, i: int, n_h: int, n_w: int) -> dict:
        """Block ``i`` alone on a host input [N, 1024] (teacher forcing from a reference's block i-1 output): the host
        tensors ``attn`` and ``mlp`` (sublayer outputs before their residual adds) and ``out`` [N, 1024]."""
        n = n_h * n_w
        n_pad = -(-n // ROW_PAD) * ROW_PAD
        x = ttnn.from_torch(
            F.pad(x_in.to(torch.bfloat16), (0, 0, 0, n_pad - n)).reshape(1, 1, n_pad, DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos, sin = self._rope_tables(n_h, n_w, n_pad)
        cu = self._windows(n, n_pad)
        taps: dict = {}
        out = self.block(x, i, cos, sin, cu, taps=taps)
        res = {k: self._host(t).reshape(n_pad, DIM)[:n].float() for k, t in taps.items()}
        res["out"] = self._host(out).reshape(n_pad, DIM)[:n].float()
        for t in list(taps.values()) + [out]:
            ttnn.deallocate(t)
        return res

    # ── public ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _host(self, t) -> torch.Tensor:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if self.n_dev > 1 else ttnn.to_torch(t)

    def vit(self, patches: torch.Tensor, n_h: int, n_w: int, *, capture: list | None = None) -> torch.Tensor:
        """model.py ``ViT.forward``: patches [N, 3, 14, 14] (or [N, 588]) -> the final-norm output [N, 1024] bf16 (host).
        ``capture``: a list that receives every block's output [N, 1024] (host, for the per-block PCC)."""
        n = n_h * n_w
        assert patches.shape[0] == n, (patches.shape, n_h, n_w)
        n_pad = -(-n // ROW_PAD) * ROW_PAD
        x = patches.reshape(n, -1).to(torch.bfloat16)
        x = F.pad(x, (0, K_PAD - x.shape[1], 0, n_pad - n)).reshape(1, 1, n_pad, K_PAD)
        x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = self._lin(x, self.pe_w, self.pe_b)
        cos, sin = self._rope_tables(n_h, n_w, n_pad)
        cu = self._windows(n, n_pad)
        for i in range(self.n_layers):
            x = self.block(x, i, cos, sin, cu)
            if capture is not None:
                capture.append(self._host(x).reshape(n_pad, DIM)[:n].clone())
        y = self._rms(x, self.norm)
        ttnn.deallocate(x)
        out = self._host(y).reshape(n_pad, DIM)[:n].to(torch.bfloat16)
        ttnn.deallocate(y)
        return out

    def aligner(self, vit_out: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """model.py ``Aligner.forward``: [n_h * n_w, 1024] -> [n_llm_h * n_llm_w, 5120] bf16 (host)."""
        u = aligner_unfold(vit_out, n_h, n_w)  # [L, 9216]
        L = u.shape[0]
        l_pad = -(-L // 32) * 32
        t = ttnn.from_torch(
            F.pad(u, (0, 0, 0, l_pad - L)).reshape(1, 1, l_pad, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = self._lin(t, self.al_w1, self.al_b1)
        ttnn.deallocate(t)
        g = ttnn.gelu(h, fast_and_approximate_mode=False)
        ttnn.deallocate(h)
        y = self._lin(g, self.al_w2, self.al_b2)
        ttnn.deallocate(g)
        out = self._host(y).reshape(l_pad, -1)[:L].to(torch.bfloat16)
        ttnn.deallocate(y)
        return out

    def encode(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """model.py ``Transformer.encode_image``: the aligner rows for one image's IMAGE slots, in reading order."""
        return self.aligner(self.vit(patches, n_h, n_w), n_h, n_w)
