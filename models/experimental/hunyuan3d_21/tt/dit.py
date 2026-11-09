# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hunyuan3D-2.1 shape DiT (HunYuanDiTPlain: 21 blocks, width 2048, 16 heads, MoE in the last 6) on one device.

Token layout on device: rows [0, 4096) are the latent tokens, row 4096 the timestep token, rows (4096, 4608) padding.
Upstream prepends the timestep token, but the model has no positional embedding and every op is per token or
attention over all tokens, so its position only matters where the final layer drops it; keeping it at the tile-aligned
row 4096 avoids any non-aligned concat. Self-attention keeps padding out with SDPA windows [0, 4097, 4608]; the 1370
DINOv2 context tokens are padded to 1536 and masked. CFG runs as batch 2 (cond, uncond). The timestep MLP runs on the
host (one vector per call). MoE: the top-2 experts per row are combined with their softmax weights (gate in fp32);
each expert runs only on the rows routed to it (Precision.routed_moe), or, as the reference path, all 8 experts on
every row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.hunyuan3d_21.reference import dit_host as host
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor

N_TOK = 4608  # 4096 latents + timestep token, padded to 9 x 512 (SDPA key chunk 512 in the windowed kernel)
T_ROW = host.LATENTS
N_REAL = host.LATENTS + 1
CTX_PAD = 1536  # 1370 DINOv2 tokens padded to 3 x 512
MASK_VALUE = -1e9


def _hifi4(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@dataclass(frozen=True)
class Precision:
    """Numerics switches, defaults from what the TRELLIS.2 flow DiTs needed to sit at the CPU bf16 floor."""

    fp32_residual: bool = True  # residual stream x (and the skip values) in fp32
    fp32_qk: bool = True  # q/k RMSNorm in fp32, one cast to bf16 before SDPA
    fp32_input: bool = True  # latents enter as bf16 hi + lo against a stacked [W; W]
    fp32_output: bool = True  # final linear writes fp32
    fp32_gate: bool = True  # MoE gate logits and softmax in fp32 (fewer top-k flips)
    # top-2 on the host from the device gate scores (exact torch.topk): the on-device threshold (max, then max of the
    # rest) keeps 3 experts whenever two scores round equal, which put block 15 at 3.3x (fp32) / 7.4x (bf16) the CPU
    # bf16 floor at step 15
    host_topk: bool = True
    sdpa_k_chunk: int = 512
    # MoE: run each expert only on the rows routed to it (gathered with ttnn.embedding, row count rounded up to
    # `moe_bucket` so the programs are reused across calls) instead of all 8 experts on every row; needs host_topk.
    # Same G1 error as dense (0.765x vs 0.761x the bf16 floor), 0.956 vs 1.150 s per call.
    routed_moe: bool = True
    moe_bucket: int = 256
    # fidelity of the block matmuls (attention projections, MLPs, experts, skip linears); the gate, the input
    # embedder and the final linear stay HiFi4. HiFi2 is 0.86 s per call but its G1 error is 1.31x the bf16 floor
    # (limit 1.5x); LoFi fails G1 (8.2x)
    mm_fidelity: str = "HiFi4"


def _bf16(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.typecast(x, ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x


def _add(x: ttnn.Tensor, branch: ttnn.Tensor) -> ttnn.Tensor:
    if x.dtype != branch.dtype:
        branch = ttnn.typecast(branch, x.dtype)
    return ttnn.add(x, branch)


class HeadRMSNorm(Module):
    """nn.RMSNorm(head_dim) applied per head: gamma [head_dim] shared by all heads."""

    def __init__(self, *, mesh_device, dtype) -> None:
        super().__init__()
        self.weight = Parameter(total_shape=[1, 1, 1, host.HEAD_DIM], device=mesh_device, dtype=dtype)
        self.cfg = _hifi4(mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].reshape(1, 1, 1, -1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.multiply(
            ttnn.rms_norm(x, epsilon=host.QK_NORM_EPS, compute_kernel_config=self.cfg), self.weight.data
        )


class Attention(Module):
    def __init__(self, *, ctx_dim: int | None, mesh_device, precision: Precision) -> None:
        super().__init__()
        self.is_self = ctx_dim is None
        d = host.HIDDEN
        mm = _hifi4(mesh_device)
        self.to_q = Linear(d, d, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_k = Linear(ctx_dim or d, d, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_v = Linear(ctx_dim or d, d, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.out_proj = Linear(d, d, bias=True, mesh_device=mesh_device, compute_kernel_config=mm)
        qk_dtype = ttnn.float32 if precision.fp32_qk else ttnn.bfloat16
        self.q_norm = HeadRMSNorm(mesh_device=mesh_device, dtype=qk_dtype)
        self.k_norm = HeadRMSNorm(mesh_device=mesh_device, dtype=qk_dtype)
        self.fp32_qk = precision.fp32_qk
        grid = mesh_device.compute_with_storage_grid_size()
        k = precision.sdpa_k_chunk
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=k, exp_approx_mode=False
        )
        self.sdpa_kernel = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def _heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        try:
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                x, num_heads=host.HEADS, num_kv_heads=0, transpose_k_heads=False
            )
        except RuntimeError:
            print(f"HEADS_FAIL self={self.is_self} shape={x.shape} dtype={x.dtype}", flush=True)
            raise
        return out

    def _qk(self, x: ttnn.Tensor, norm: HeadRMSNorm) -> ttnn.Tensor:
        x = self._heads(x)
        if self.fp32_qk:
            x = ttnn.typecast(x, ttnn.float32)
        return _bf16(norm(x))

    def forward(
        self, x: ttnn.Tensor, windows: ttnn.Tensor, context: ttnn.Tensor | None = None, mask=None
    ) -> ttnn.Tensor:
        kv_in = x if self.is_self else context
        q = self._qk(self.to_q(x), self.q_norm)
        k = self._qk(self.to_k(kv_in), self.k_norm)
        v = self._heads(self.to_v(kv_in))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=None if self.is_self else mask,
            cu_window_seqlens=windows if self.is_self else None,
            program_config=self.sdpa_cfg,
            compute_kernel_config=self.sdpa_kernel,
        )
        # concatenate_heads returns [B, S, D]; keep the [B, 1, S, D] layout (a 3-D result would broadcast against the
        # residual into [B, B, S, D])
        o = ttnn.transformer.concatenate_heads(out)
        return self.out_proj(ttnn.reshape(o, (o.shape[0], 1, o.shape[-2], o.shape[-1])))


class FeedForward(Module):
    """fc1 -> exact GELU -> fc2 (the dense MLP, the shared expert and each routed expert have this shape)."""

    def __init__(self, *, mesh_device) -> None:
        super().__init__()
        mm = _hifi4(mesh_device)
        self.fc1 = Linear(
            host.HIDDEN, 4 * host.HIDDEN, activation_fn="gelu", mesh_device=mesh_device, compute_kernel_config=mm
        )
        self.fc2 = Linear(4 * host.HIDDEN, host.HIDDEN, mesh_device=mesh_device, compute_kernel_config=mm)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # diffusers FeedForward: net.0.proj (GELU module's Linear) and net.2
        for src, dst in (("net.0.proj", "fc1"), ("net.2", "fc2")):
            for p in ("weight", "bias"):
                if f"{src}.{p}" in state:
                    state[f"{dst}.{p}"] = state.pop(f"{src}.{p}")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.fc2(self.fc1(x))


class MoE(Module):
    """MoEBlock: shared expert + sum over the top-2 of 8 experts weighted by their (unnormalised) softmax scores."""

    def __init__(self, *, mesh_device, precision: Precision) -> None:
        super().__init__()
        self.experts = ModuleList(FeedForward(mesh_device=mesh_device) for _ in range(host.NUM_EXPERTS))
        self.shared_experts = FeedForward(mesh_device=mesh_device)
        self.fp32_gate = precision.fp32_gate
        self.host_topk = precision.host_topk
        self.routed = precision.routed_moe
        self.bucket = precision.moe_bucket
        assert not self.routed or self.host_topk, "routed MoE needs the host top-k"
        self.mesh_device = mesh_device
        gdt = ttnn.float32 if precision.fp32_gate else ttnn.bfloat16
        # 8 gate rows padded to 32 output columns; the padding gets MASK_VALUE so softmax gives it 0
        self.gate = Linear(
            host.HIDDEN, 32, bias=True, dtype=gdt, mesh_device=mesh_device, compute_kernel_config=_hifi4(mesh_device)
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "gate.weight" in state:
            w = state.pop("gate.weight")
            state["gate.weight"] = torch.cat([w, torch.zeros(32 - w.shape[0], w.shape[1])], 0)
            state["gate.bias"] = torch.cat([torch.zeros(w.shape[0]), torch.full((32 - w.shape[0],), MASK_VALUE)])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        xin = ttnn.typecast(x, ttnn.float32) if self.fp32_gate else x
        scores = ttnn.softmax(self.gate(xin, dtype=ttnn.float32 if self.fp32_gate else None), dim=-1)
        top_i = None
        if self.host_topk:
            s = ttnn.to_torch(scores).float()
            top_v, top_i = s[..., : host.NUM_EXPERTS].topk(host.MOE_TOP_K, dim=-1)
            wt = torch.zeros_like(s).scatter_(-1, top_i, top_v)
            w = ttnn.from_torch(wt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        else:
            # top-2 by threshold: the largest score, then the largest of the rest; keep scores >= the second
            m1 = ttnn.max(scores, dim=-1, keepdim=True)
            rest = ttnn.where(ttnn.ge(scores, m1), -1.0, scores)
            m2 = ttnn.max(rest, dim=-1, keepdim=True)
            w = ttnn.multiply(scores, ttnn.ge(scores, m2))
        xb = _bf16(x)
        y = self.shared_experts(xb)
        y = ttnn.typecast(y, ttnn.float32)
        if self.routed:
            return self._routed(xb, y, top_i, w)
        for e, expert in enumerate(self.experts):
            we = ttnn.slice(w, [0, 0, 0, e], [w.shape[0], w.shape[1], w.shape[2], e + 1])
            y = ttnn.add(y, ttnn.multiply(ttnn.typecast(expert(xb), ttnn.float32), we))
        return y

    def _routed(self, xb: ttnn.Tensor, y: ttnn.Tensor, top_i: torch.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """y + sum_e w_e * expert_e(x), each expert on its routed rows only. Padding rows (>= N_REAL) are not routed.
        Gather in: rows of x by index (embedding over x as a row-major table). Scatter back: for every row, the
        expert output row that holds it, or a padding row of the expert output (weight 0 there)."""
        dev = self.mesh_device
        b, _, n, d = xb.shape
        rows = b * n
        table = ttnn.to_layout(ttnn.reshape(xb, (1, 1, rows, d)), ttnn.ROW_MAJOR_LAYOUT)
        sel = top_i[:, 0, :N_REAL]  # [b, N_REAL, k]
        flat = (torch.arange(b)[:, None] * n + torch.arange(N_REAL)[None]).reshape(-1)
        sel = sel.reshape(-1, sel.shape[-1])
        for e, expert in enumerate(self.experts):
            mine = flat[(sel == e).any(-1)]
            cap = -(-(mine.numel() + 1) // self.bucket) * self.bucket  # at least one padding row
            ids = torch.zeros(1, cap, dtype=torch.int32)
            ids[0, : mine.numel()] = mine
            xg = ttnn.embedding(
                ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),
                table,
                layout=ttnn.TILE_LAYOUT,
            )
            ye = expert(ttnn.reshape(xg, (1, 1, cap, d)))
            back = torch.full((1, rows), cap - 1, dtype=torch.int32)
            back[0, mine] = torch.arange(mine.numel(), dtype=torch.int32)
            ye = ttnn.to_layout(ye, ttnn.ROW_MAJOR_LAYOUT)
            yb = ttnn.embedding(
                ttnn.from_torch(back, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),
                ye,
                layout=ttnn.TILE_LAYOUT,
            )
            yb = ttnn.reshape(yb, (b, 1, n, d))
            we = ttnn.slice(w, [0, 0, 0, e], [w.shape[0], w.shape[1], w.shape[2], e + 1])
            y = ttnn.add(y, ttnn.multiply(ttnn.typecast(yb, ttnn.float32), we))
        return y


class Block(Module):
    def __init__(self, layer: int, *, mesh_device, precision: Precision) -> None:
        super().__init__()
        d = host.HIDDEN
        self.norm1 = LayerNorm(d, norm_eps=host.NORM_EPS, mesh_device=mesh_device)
        self.norm2 = LayerNorm(d, norm_eps=host.NORM_EPS, mesh_device=mesh_device)
        self.norm3 = LayerNorm(d, norm_eps=host.NORM_EPS, mesh_device=mesh_device)
        self.attn1 = Attention(ctx_dim=None, mesh_device=mesh_device, precision=precision)
        self.attn2 = Attention(ctx_dim=host.CONTEXT_DIM, mesh_device=mesh_device, precision=precision)
        self.use_skip = host.has_skip(layer)
        if self.use_skip:
            mm = _hifi4(mesh_device)
            # skip_linear(cat[skip, x]) split into the skip half and the x half: no concat on device
            self.skip_linear_a = Linear(d, d, bias=True, mesh_device=mesh_device, compute_kernel_config=mm)
            self.skip_linear_b = Linear(d, d, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
            self.skip_norm = LayerNorm(d, norm_eps=host.NORM_EPS, mesh_device=mesh_device)
        self.use_moe = host.is_moe_layer(layer)
        if self.use_moe:
            self.moe = MoE(mesh_device=mesh_device, precision=precision)
        else:
            self.mlp = FeedForward(mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "skip_linear.weight" in state:
            w = state.pop("skip_linear.weight")
            state["skip_linear_a.weight"], state["skip_linear_b.weight"] = w[:, : host.HIDDEN], w[:, host.HIDDEN :]
            state["skip_linear_a.bias"] = state.pop("skip_linear.bias")

    def forward(self, x, context, mask, windows, skip=None):
        if self.use_skip:
            s = self.skip_linear_a(_bf16(skip), dtype=ttnn.float32)
            s = ttnn.add(s, self.skip_linear_b(_bf16(x), dtype=ttnn.float32))
            x = self.skip_norm(s)
            x = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        x = _add(x, self.attn1(_bf16(self.norm1(x)), windows))
        x = _add(x, self.attn2(_bf16(self.norm2(x)), windows, context, mask))
        h = _bf16(self.norm3(x))
        return _add(x, self.moe(h) if self.use_moe else self.mlp(h))


class TtHunyuanDiT(Module):
    def __init__(self, *, mesh_device, precision: Precision = Precision()) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.precision = precision
        mm = _hifi4(mesh_device)
        k_in = host.IN_CHANNELS * (2 if precision.fp32_input else 1)
        self.x_embedder = Linear(k_in, host.HIDDEN, mesh_device=mesh_device, compute_kernel_config=mm)
        self.blocks = ModuleList(Block(i, mesh_device=mesh_device, precision=precision) for i in range(host.DEPTH))
        self.final_norm = LayerNorm(host.HIDDEN, norm_eps=host.NORM_EPS, mesh_device=mesh_device)
        self.final_linear = Linear(host.HIDDEN, host.IN_CHANNELS, mesh_device=mesh_device, compute_kernel_config=mm)
        if precision.mm_fidelity != "HiFi4":
            self._set_block_fidelity(getattr(ttnn.MathFidelity, precision.mm_fidelity))
        self._host: dict[str, torch.Tensor] = {}
        self._consts: dict[int, tuple] = {}
        self.collect: list | None = None  # debug: per-block outputs copied to the host when a list

    def _set_block_fidelity(self, fidelity) -> None:
        cfg = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def walk(m: Module) -> None:
            for name, child in m.named_children():
                if isinstance(child, Linear) and name != "gate":
                    child.compute_config = cfg
                else:
                    walk(child)

        walk(self.blocks)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state.update(host.fix_qkv_layout({k: v for k, v in state.items() if ".attn" in k}))
        self._host = {k: state.pop(k) for k in list(state) if k.startswith("t_embedder.")}
        # dense MLP keys (blocks.i.mlp.fc1/fc2) already match; MoE keys (blocks.i.moe.experts.e.net.*, shared_experts,
        # gate) are renamed by FeedForward and MoE
        if "x_embedder.weight" in state and self.precision.fp32_input:
            w = state["x_embedder.weight"]
            state["x_embedder.weight"] = torch.cat([w, w], dim=1)
        if "final_layer.norm_final.weight" in state:
            for p in ("weight", "bias"):
                state[f"final_norm.{p}"] = state.pop(f"final_layer.norm_final.{p}")
                state[f"final_linear.{p}"] = state.pop(f"final_layer.linear.{p}")
        self._x_bias = state["x_embedder.bias"].clone()

    def consts(self, batch: int):
        if batch not in self._consts:
            dev = self.mesh_device
            mask = torch.zeros(batch, 1, N_TOK, CTX_PAD)
            mask[..., host.CONTEXT_LEN :] = MASK_VALUE
            onehot = torch.zeros(batch, 1, N_TOK, 32)
            onehot[:, :, T_ROW, 0] = 1.0  # hi half of the timestep vector
            onehot[:, :, T_ROW, 1] = 1.0  # lo half
            windows = ttnn.from_torch(
                torch.tensor([0, N_REAL, N_TOK], dtype=torch.int32),
                device=dev,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            self._consts[batch] = (bf16_tensor(mask, device=dev), bf16_tensor(onehot, device=dev), windows)
        return self._consts[batch]

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Torch in, torch out, like HunYuanDiTPlain.forward: x [B, 4096, 64], t [B] in [0, 1], cond [B, 1370, 1024]
        -> velocity [B, 4096, 64] (fp32)."""
        dev = self.mesh_device
        b = x.shape[0]
        mask, onehot, windows = self.consts(b)
        tok = torch.zeros(b, 1, N_TOK, host.IN_CHANNELS)
        tok[:, 0, : host.LATENTS] = x.float()
        if self.precision.fp32_input:
            hi = tok.bfloat16().float()
            tok = torch.cat([hi, tok - hi], dim=-1)
        h = self.x_embedder(bf16_tensor(tok, device=dev), dtype=ttnn.float32)
        # timestep token at row T_ROW: its input row is zero, so it holds the bias; add (c - bias) as bf16 hi + lo
        c = host.timestep_vector(self._host, t) - self._x_bias
        c_hi = c.bfloat16().float()
        cm = torch.zeros(b, 1, 32, host.HIDDEN)
        cm[:, 0, 0], cm[:, 0, 1] = c_hi, c - c_hi
        h = ttnn.add(
            h, ttnn.matmul(onehot, bf16_tensor(cm, device=dev), dtype=ttnn.float32, compute_kernel_config=_hifi4(dev))
        )
        if not self.precision.fp32_residual:
            h = _bf16(h)
        ctx = torch.zeros(b, 1, CTX_PAD, host.CONTEXT_DIM)
        ctx[:, 0, : cond.shape[1]] = cond.float()
        ctx = bf16_tensor(ctx, device=dev)
        skips = []
        for i, block in enumerate(self.blocks):
            skip = skips.pop() if block.use_skip else None
            h = block(h, ctx, mask, windows, skip)
            if self.collect is not None:
                self.collect.append(ttnn.to_torch(h).float()[:, 0, :N_REAL])
            if i < host.DEPTH // 2:
                skips.append(h)
        h = self.final_norm(h)
        out = self.final_linear(_bf16(h), dtype=ttnn.float32 if self.precision.fp32_output else None)
        out = ttnn.to_torch(out).float()
        return out[:, 0, : host.LATENTS, :]
