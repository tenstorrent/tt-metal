"""TTNN flow-matching DiT (MiniMaxMusic3Transformer1DModel): 36 pre-LN blocks, dim 2048, 32 heads x 64, partial
RoPE (first 32 dims, rotate-half), SwiGLU FF 8192 with biases, a timestep token prepended.

Layout: the sequence lives on rows of a [B, 1, Lp, C] tile tensor. Row 0 is the timestep token, rows 1..L the
latent frames, rows L+1..Lp-1 padding (Lp = multiple of 128); pad KEYS are masked in attention, pad rows' outputs
are discarded. preprocess_conv / postprocess_conv are 1x1 convs = linears over the channel dim. The host prepares
`xin = [latents | zeros | condition]` per row and the Fourier features of t; everything else runs on device.
One trace per Lp (the two window lengths of a song: 689 and the last, shorter one)."""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, Optional

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.config import DiTConfig
from models.autoports.minimaxai_minimax_music3.reference.dit import rotary_tables
from models.autoports.minimaxai_minimax_music3.tt.weights import cache_root

PAD_TO = 128
MASK_VALUE = -1e9


def padded_len(seq_len_with_temb: int) -> int:
    return int(math.ceil(seq_len_with_temb / PAD_TO) * PAD_TO)


def _dtype_tag(dt) -> str:
    return str(dt).replace("DataType.", "").lower()


class TTDiT:
    def __init__(
        self,
        mesh_device,
        state: Dict[str, torch.Tensor],
        cfg: DiTConfig,
        *,
        weights_dtype=ttnn.bfloat16,
        use_trace: bool = True,
        log: Callable[[str], None] = print,
        num_layers: Optional[int] = None,
    ):
        self.mesh, self.cfg, self.log, self.use_trace = mesh_device, cfg, log, use_trace
        assert mesh_device.get_num_devices() == 1, "TTDiT runs on one chip"
        self.n_layers = num_layers or cfg.num_layers
        self.dim, self.heads, self.head_dim = cfg.inner_dim, cfg.num_attention_heads, cfg.attention_head_dim
        cache = cache_root() / "dit" / _dtype_tag(weights_dtype)
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        def dev(t: torch.Tensor, name: str, dtype=weights_dtype):
            return ttnn.as_tensor(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=str(cache / name),
            )

        def vec(t: torch.Tensor, name: str):
            return dev(t.reshape(1, 1, 1, -1).float(), name, dtype=ttnn.bfloat16)

        w = state
        self.time_proj_w = w["time_proj.weight"].float().reshape(-1)  # [128] (host Fourier features)
        self.w_t1, self.b_t1 = dev(w["time_embed.linear_1.weight"].T, "t1_w"), vec(
            w["time_embed.linear_1.bias"], "t1_b"
        )
        self.w_t2, self.b_t2 = dev(w["time_embed.linear_2.weight"].T, "t2_w"), vec(
            w["time_embed.linear_2.bias"], "t2_b"
        )
        self.w_pre = dev(w["preprocess_conv.weight"].squeeze(-1).T, "pre_w")  # [2304, 2304]
        self.w_in = dev(w["proj_in.weight"].T, "in_w")  # [2304, 2048]
        self.w_out = dev(w["proj_out.weight"].T, "out_w")  # [2048, 128]
        self.w_post = dev(w["postprocess_conv.weight"].squeeze(-1).T, "post_w")  # [128, 128]
        self.blocks = []
        ff = cfg.ff_inner_dim
        for i in range(self.n_layers):
            p = f"transformer_blocks.{i}."
            qkv = torch.cat([w[p + "attn.to_q.weight"], w[p + "attn.to_k.weight"], w[p + "attn.to_v.weight"]], dim=0).T
            ffw, ffb = w[p + "ff_in.weight"], w[p + "ff_in.bias"]
            self.blocks.append(
                {
                    "n1_w": vec(w[p + "norm1.weight"], f"b{i}_n1w"),
                    "n1_b": vec(w[p + "norm1.bias"], f"b{i}_n1b"),
                    "wqkv": dev(qkv, f"b{i}_wqkv"),
                    "wo": dev(w[p + "attn.to_out.0.weight"].T, f"b{i}_wo"),
                    "n2_w": vec(w[p + "norm2.weight"], f"b{i}_n2w"),
                    "n2_b": vec(w[p + "norm2.bias"], f"b{i}_n2b"),
                    "w_a": dev(ffw[:ff].T, f"b{i}_wa"),
                    "b_a": vec(ffb[:ff], f"b{i}_ba"),  # gate_states (first half)
                    "w_g": dev(ffw[ff:].T, f"b{i}_wg"),
                    "b_g": vec(ffb[ff:], f"b{i}_bg"),  # gate (second half, SiLU)
                    "w_o": dev(w[p + "ff_out.weight"].T, f"b{i}_wffo"),
                    "b_o": vec(w[p + "ff_out.bias"], f"b{i}_bffo"),
                }
            )
        # rotate-half within the first 32-dim tile of every head; the ROPE op applies trans_mat per 32-wide tile
        T = torch.zeros(1, 1, 32, 32)
        for j in range(16):
            T[0, 0, j + 16, j] = -1.0
            T[0, 0, j, j + 16] = 1.0
        self.trans_mat = dev(T, "rope_trans", dtype=ttnn.bfloat16)
        arch = mesh_device.arch()
        fid = os.environ.get("MUSIC3_DIT_FIDELITY", "hifi4" if weights_dtype == ttnn.bfloat16 else "hifi2")
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4 if fid == "hifi4" else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=(fid == "hifi4"),
            packer_l1_acc=True,
        )
        self.fidelity = fid
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.grid = mesh_device.compute_with_storage_grid_size()
        self._shapes: Dict[int, dict] = {}  # per Lp: persistent inputs, rope tables, mask, sel, trace, outputs
        self.log(
            f"DiT weights on device in {time.time() - t0:.1f}s ({self.n_layers} layers, {_dtype_tag(weights_dtype)})"
        )

    # ------------------------------------------------------------------ per-length state
    def _state(self, L: int) -> dict:
        """L = number of latent frames (sequence has L+1 rows incl. the timestep token)."""
        Lp = padded_len(L + 1)
        if Lp in self._shapes and self._shapes[Lp]["L"] == L:
            return self._shapes[Lp]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        cos, sin = rotary_tables(Lp, self.cfg.rotary_dim)  # [Lp, 32] each (rotate-half pairs i, i+16)
        cos_full = torch.cat([cos, torch.ones(Lp, self.head_dim - self.cfg.rotary_dim)], dim=-1).reshape(
            1, 1, Lp, self.head_dim
        )
        sin_full = torch.cat([sin, torch.zeros(Lp, self.head_dim - self.cfg.rotary_dim)], dim=-1).reshape(
            1, 1, Lp, self.head_dim
        )
        mask = torch.zeros(1, 1, Lp, Lp)
        mask[..., L + 1 :] = MASK_VALUE  # pad keys
        sel = torch.zeros(1, 1, Lp, 32)
        sel[0, 0, 0, 0] = 1.0  # places the timestep token at row 0
        mk = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        st = {
            "L": L,
            "Lp": Lp,
            "cos": mk(cos_full),
            "sin": mk(sin_full),
            "mask": mk(mask),
            "sel": mk(sel),
            "xin": mk(torch.zeros(2, 1, Lp, self.cfg.concat_channels)),
            "tf": mk(torch.zeros(2, 1, 32, self.cfg.fourier_embedding_dim)),
            "trace": None,
            "out": None,
        }
        # SDPA chunking: Lp is a multiple of 128
        st["sdpa_pc"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.grid.x, self.grid.y),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._shapes[Lp] = st
        return st

    # ------------------------------------------------------------------ graph
    def _forward(self, st: dict):
        xin, tf = st["xin"], st["tf"]
        # timestep token: Fourier features -> linear_1 -> SiLU -> linear_2 -> placed at row 0 via a one-hot selector matmul
        t = ttnn.linear(
            tf, self.w_t1, bias=self.b_t1, activation="silu", compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16
        )
        t = ttnn.linear(
            t, self.w_t2, bias=self.b_t2, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16
        )  # [2,1,32,2048], row 0 valid
        temb_rows = ttnn.matmul(st["sel"], t, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)  # [2,1,Lp,2048]
        # preprocess 1x1 conv (+ residual) and input projection
        x = ttnn.add(ttnn.linear(xin, self.w_pre, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16), xin)
        x = ttnn.linear(x, self.w_in, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)
        x = ttnn.add(x, temb_rows)
        for B in self.blocks:
            h = ttnn.layer_norm(x, epsilon=1e-5, weight=B["n1_w"], bias=B["n1_b"])
            qkv = ttnn.linear(h, B["wqkv"], compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.heads, num_kv_heads=self.heads, transpose_k_heads=False
            )
            q = ttnn.experimental.rotary_embedding_llama(q, st["cos"], st["sin"], self.trans_mat, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_llama(k, st["cos"], st["sin"], self.trans_mat, is_decode_mode=False)
            a = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=st["mask"],
                is_causal=False,
                program_config=st["sdpa_pc"],
                compute_kernel_config=self.ck_sdpa,
            )
            a = ttnn.experimental.nlp_concat_heads(a)
            x = ttnn.add(x, ttnn.linear(a, B["wo"], compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16))
            h = ttnn.layer_norm(x, epsilon=1e-5, weight=B["n2_w"], bias=B["n2_b"])
            a_ = ttnn.linear(h, B["w_a"], bias=B["b_a"], compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)
            g_ = ttnn.linear(
                h, B["w_g"], bias=B["b_g"], activation="silu", compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16
            )
            x = ttnn.add(
                x,
                ttnn.linear(
                    ttnn.multiply(a_, g_),
                    B["w_o"],
                    bias=B["b_o"],
                    compute_kernel_config=self.ck_mm,
                    dtype=ttnn.bfloat16,
                ),
            )
        y = ttnn.linear(x, self.w_out, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)  # [2,1,Lp,128]
        y = ttnn.add(ttnn.linear(y, self.w_post, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16), y)
        return y

    def _ensure_trace(self, st: dict):
        if st["out"] is None:
            st["out"] = self._forward(st)  # compile pass
            ttnn.synchronize_device(self.mesh)
            if self.use_trace:
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                st["out"] = self._forward(st)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                ttnn.synchronize_device(self.mesh)
                st["trace"] = tid
                self.log(f"DiT: trace captured for Lp={st['Lp']} (L={st['L']})")

    # ------------------------------------------------------------------ public forward (reference signature)
    @torch.inference_mode()
    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """latents [B<=2, 128, L], timestep [B], condition [B, L, 2048] -> velocity [B, 128, L] (float32, host)."""
        B, C, L = latents.shape
        assert B <= 2 and C == self.cfg.in_channels, latents.shape
        st = self._state(L)
        Lp = st["Lp"]
        xin = torch.zeros(2, 1, Lp, self.cfg.concat_channels, dtype=torch.bfloat16)
        xin[:B, 0, 1 : L + 1, :C] = latents.transpose(1, 2).to(torch.bfloat16)
        xin[:B, 0, 1 : L + 1, 2 * C :] = condition.to(torch.bfloat16)
        angles = 2.0 * math.pi * timestep.float().reshape(B, 1) * self.time_proj_w.reshape(1, -1)
        feats = torch.cat((angles.cos(), angles.sin()), dim=-1)  # [B, 256]
        tf = torch.zeros(2, 1, 32, self.cfg.fourier_embedding_dim, dtype=torch.bfloat16)
        tf[:B, 0, 0] = feats.to(torch.bfloat16)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(xin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), st["xin"]
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(tf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), st["tf"]
        )
        self._ensure_trace(st)
        if st["trace"] is not None:
            ttnn.execute_trace(self.mesh, st["trace"], cq_id=0, blocking=True)
            out = st["out"]
        else:
            out = self._forward(st)
        y = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[:B, 0, 1 : L + 1, :]  # [B, L, 128]
        return y.transpose(1, 2).contiguous()

    __call__ = forward

    def release(self):
        for st in self._shapes.values():
            if st["trace"] is not None:
                ttnn.release_trace(self.mesh, st["trace"])
                st["trace"] = None
