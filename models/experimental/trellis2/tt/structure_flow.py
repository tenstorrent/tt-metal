# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 sparse-structure flow DiT (dense 16^3 grid, 4096 tokens) on one device.

Mirrors `reference/structure_flow_host.functional_forward` op for op. The timestep MLP and shared adaLN run on the
host (one vector per call); the input layer, 30 blocks, final norm and output layer run on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.trellis2.reference import structure_flow_host as host
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor


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
    """Numerics switches. All on by default: with tt_dit's Wan settings (all off) the per-call error was ~10x the CPU
    bf16 floor and the 12-step sample missed G1 (last call 0.960); with all on, a single pass matches that floor."""

    sdpa_hifi4_fp32_acc: bool = True  # SDPA at HiFi4 with fp32 dest accumulation (off: HiFi2, bf16 acc)
    linear_hifi4: bool = True  # every Linear at HiFi4 (off: HiFi2; both accumulate in fp32)
    fp32_residual: bool = True  # keep the residual stream x in fp32 between blocks (off: bf16)
    # out_layer writes fp32 (off: bf16; fails G2 free-running because CFG 7.5 amplifies the rounding)
    fp32_output: bool = True
    # input_layer and out_layer at HiFi4 (off: tt_dit's default fidelity; the texture sample drifts 6.8x
    # the CPU bf16 floor over 12 steps)
    io_hifi4: bool = True
    # out_layer reads the final norm output in fp32 (off: cast to bf16 first)
    fp32_out_input: bool = True
    # q/k per-head RMSNorm, gamma and RoPE in fp32 with fp32 tables, one cast to bf16 before SDPA, as TRELLIS.2's
    # bf16 model does (off: every op rounds to bf16, cos/sin tables in bf16)
    fp32_qk: bool = True
    # x_t reaches the input layer with ~16 mantissa bits, as TRELLIS.2's bf16 model keeps it fp32: uploaded as bf16
    # hi and lo halves against a stacked [W; W] weight (an fp32 activation overflows L1 at K=32). Off: one bf16 copy.
    fp32_input: bool = False


def hi_lo_split(x: torch.Tensor) -> torch.Tensor:
    """[..., C] fp32 -> [..., 2C]: bf16-exact hi and the remainder lo, so that W @ hi + W @ lo carries ~16 bits."""
    hi = x.bfloat16().float()
    return torch.cat([hi, x - hi], dim=-1)


def _norm_in(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.typecast(x, ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x


def _residual_add(x: ttnn.Tensor, branch: ttnn.Tensor) -> ttnn.Tensor:
    if x.dtype != branch.dtype:
        branch = ttnn.typecast(branch, x.dtype)
    return ttnn.add(x, branch)


@dataclass
class DeviceTables:
    """Per-request constants shared by every block."""

    cos: ttnn.Tensor  # [1, H, N, E]
    sin: ttnn.Tensor  # [1, H, N, E]
    rot: ttnn.Tensor  # [E, E]
    cross_mask: ttnn.Tensor  # [1, 1, N, Lp]
    self_windows: ttnn.Tensor | None = None  # SDPA cu_window_seqlens [0, n_real, N] (uint32); None: no padding


class MultiHeadRMSNorm(Module):
    """TRELLIS.2 MultiHeadRMSNorm: F.normalize(x) * gamma * sqrt(E) == rms_norm(x) * gamma, gamma per head."""

    def __init__(
        self, num_heads: int, head_dim: int, *, mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType = ttnn.bfloat16
    ) -> None:
        super().__init__()
        self.gamma = Parameter(total_shape=[1, num_heads, 1, head_dim], device=mesh_device, dtype=dtype)
        self.compute_kernel_config = _hifi4(mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "gamma" in state:
            state["gamma"] = state["gamma"].unsqueeze(0).unsqueeze(2)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.rms_norm(x, epsilon=host.QK_NORM_EPS, compute_kernel_config=self.compute_kernel_config)
        return ttnn.multiply(x, self.gamma.data)


class Attention(Module):
    def __init__(
        self, dim: int, num_heads: int, *, ctx_dim: int | None, mesh_device: ttnn.MeshDevice, precision: Precision
    ) -> None:
        super().__init__()
        self.is_self = ctx_dim is None
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        mm = _hifi4(mesh_device) if precision.linear_hifi4 else None
        self.to_q = Linear(dim, dim, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_k = Linear(ctx_dim or dim, dim, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_v = Linear(ctx_dim or dim, dim, mesh_device=mesh_device, compute_kernel_config=mm)
        self.to_out = Linear(dim, dim, mesh_device=mesh_device, compute_kernel_config=mm)
        self.fp32_qk = precision.fp32_qk
        qk_dtype = ttnn.float32 if self.fp32_qk else ttnn.bfloat16
        self.q_rms_norm = MultiHeadRMSNorm(num_heads, self.head_dim, mesh_device=mesh_device, dtype=qk_dtype)
        self.k_rms_norm = MultiHeadRMSNorm(num_heads, self.head_dim, mesh_device=mesh_device, dtype=qk_dtype)

        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if precision.sdpa_hifi4_fp32_acc else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=precision.sdpa_hifi4_fp32_acc,
        )
        self.rope_compute_kernel_config = _hifi4(mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "to_qkv.weight" in state:
            for name, w, b in zip(
                ("to_q", "to_k", "to_v"), state.pop("to_qkv.weight").chunk(3), state.pop("to_qkv.bias").chunk(3)
            ):
                state[f"{name}.weight"], state[f"{name}.bias"] = w, b
        if "to_kv.weight" in state:
            for name, w, b in zip(
                ("to_k", "to_v"), state.pop("to_kv.weight").chunk(2), state.pop("to_kv.bias").chunk(2)
            ):
                state[f"{name}.weight"], state[f"{name}.bias"] = w, b

    def _heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=self.num_heads, num_kv_heads=0, transpose_k_heads=False
        )
        return out

    def _rope(self, x: ttnn.Tensor, tables: DeviceTables) -> ttnn.Tensor:
        rotated = ttnn.matmul(x, tables.rot, compute_kernel_config=self.rope_compute_kernel_config)
        return ttnn.add(ttnn.multiply(x, tables.cos), ttnn.multiply(rotated, tables.sin))

    def _qk(self, x: ttnn.Tensor, norm: MultiHeadRMSNorm, tables: DeviceTables) -> ttnn.Tensor:
        x = self._heads(x)
        if self.fp32_qk:
            x = ttnn.typecast(x, ttnn.float32)
        x = norm(x)
        if self.is_self:
            x = self._rope(x, tables)
        return _norm_in(x)

    def forward(self, x: ttnn.Tensor, tables: DeviceTables, context: ttnn.Tensor | None = None) -> ttnn.Tensor:
        kv_in = x if self.is_self else context
        q = self._qk(self.to_q(x), self.q_rms_norm, tables)
        k = self._qk(self.to_k(kv_in), self.k_rms_norm, tables)
        v = self._heads(self.to_v(kv_in))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=None if self.is_self else tables.cross_mask,
            cu_window_seqlens=tables.self_windows if self.is_self else None,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(out), 0)
        return self.to_out(out)


class FeedForward(Module):
    def __init__(self, dim: int, hidden: int, *, mesh_device: ttnn.MeshDevice, precision: Precision) -> None:
        super().__init__()
        mm = _hifi4(mesh_device) if precision.linear_hifi4 else None
        self.ff1 = Linear(dim, hidden, activation_fn="gelu_tanh", mesh_device=mesh_device, compute_kernel_config=mm)
        self.ff2 = Linear(hidden, dim, mesh_device=mesh_device, compute_kernel_config=mm)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for src, dst in (("mlp.0", "ff1"), ("mlp.2", "ff2")):
            for p in ("weight", "bias"):
                if f"{src}.{p}" in state:
                    state[f"{dst}.{p}"] = state.pop(f"{src}.{p}")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.ff2(self.ff1(x))


class Block(Module):
    """ModulatedTransformerCrossBlock with share_mod: adaLN self-attn, affine-LN cross-attn (no gate), adaLN MLP."""

    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision) -> None:
        super().__init__()
        d = host.MODEL_CHANNELS
        self.norm1 = LayerNorm(d, norm_eps=host.BLOCK_NORM_EPS, norm_elementwise_affine=False, mesh_device=mesh_device)
        self.norm2 = LayerNorm(d, norm_eps=host.BLOCK_NORM_EPS, norm_elementwise_affine=True, mesh_device=mesh_device)
        self.norm3 = LayerNorm(d, norm_eps=host.BLOCK_NORM_EPS, norm_elementwise_affine=False, mesh_device=mesh_device)
        self.self_attn = Attention(d, host.NUM_HEADS, ctx_dim=None, mesh_device=mesh_device, precision=precision)
        self.cross_attn = Attention(
            d, host.NUM_HEADS, ctx_dim=host.COND_CHANNELS, mesh_device=mesh_device, precision=precision
        )
        self.mlp = FeedForward(d, host.MLP_CHANNELS, mesh_device=mesh_device, precision=precision)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # the per-block modulation table is folded into the host-side adaLN vectors
        state.pop("modulation", None)

    def forward(self, x: ttnn.Tensor, mod: list[ttnn.Tensor], context: ttnn.Tensor, tables: DeviceTables):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        h = ttnn.add(ttnn.multiply(_norm_in(self.norm1(x)), scale_msa), shift_msa)
        x = _residual_add(x, ttnn.multiply(self.self_attn(h, tables), gate_msa))
        x = _residual_add(x, self.cross_attn(_norm_in(self.norm2(x)), tables, context))
        h = ttnn.add(ttnn.multiply(_norm_in(self.norm3(x)), scale_mlp), shift_mlp)
        return _residual_add(x, ttnn.multiply(self.mlp(h), gate_mlp))


class TtStructureFlow(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision = Precision()) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.precision = precision
        d = host.MODEL_CHANNELS
        io = _hifi4(mesh_device) if precision.io_hifi4 else None
        k_in = host.CHANNEL_PAD * (2 if precision.fp32_input else 1)
        self.input_layer = Linear(k_in, d, mesh_device=mesh_device, compute_kernel_config=io)
        self.blocks = ModuleList(Block(mesh_device=mesh_device, precision=precision) for _ in range(host.NUM_BLOCKS))
        self.final_norm = LayerNorm(
            d, norm_eps=host.FINAL_NORM_EPS, norm_elementwise_affine=False, mesh_device=mesh_device
        )
        self.out_layer = Linear(d, host.CHANNEL_PAD, mesh_device=mesh_device, compute_kernel_config=io)
        self._host_state: dict[str, torch.Tensor] = {}
        self._tables: dict[int, DeviceTables] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pad = host.CHANNEL_PAD - host.IN_CHANNELS
        self._host_state = {k: v for k, v in state.items() if k.startswith(host.HOST_KEYS)}
        self._host_state.update({k: v for k, v in state.items() if k.endswith(".modulation")})
        for k in list(state):
            if k.startswith(host.HOST_KEYS):
                del state[k]
        if "input_layer.weight" in state:
            w = torch.nn.functional.pad(state["input_layer.weight"], (0, pad))
            state["input_layer.weight"] = torch.cat([w, w], dim=1) if self.precision.fp32_input else w
        if "out_layer.weight" in state:
            state["out_layer.weight"] = torch.nn.functional.pad(state["out_layer.weight"], (0, 0, 0, pad))
            state["out_layer.bias"] = torch.nn.functional.pad(state["out_layer.bias"], (0, pad))

    def tables(self, n_kv: int) -> DeviceTables:
        if n_kv not in self._tables:
            cos, sin = host.rope_tables(host.grid_coords())
            expand = lambda t: t.reshape(1, 1, *t.shape).expand(1, host.NUM_HEADS, *t.shape).contiguous()
            dev = self.mesh_device
            rope_tensor = float32_tensor if self.precision.fp32_qk else bf16_tensor
            self._tables[n_kv] = DeviceTables(
                cos=rope_tensor(expand(cos), device=dev),
                sin=rope_tensor(expand(sin), device=dev),
                rot=rope_tensor(host.pair_rotation(), device=dev),
                cross_mask=bf16_tensor(host.cross_mask(host.RESOLUTION**3, n_kv), device=dev),
            )
        return self._tables[n_kv]

    def modulation(self, t: torch.Tensor) -> list[list[ttnn.Tensor]]:
        m = host.modulation(self._host_state, t)
        parts = (m.shift_msa, m.scale_msa, m.gate_msa, m.shift_mlp, m.scale_mlp, m.gate_mlp)
        return [
            [bf16_tensor(p[i].reshape(1, 1, 1, -1), device=self.mesh_device) for p in parts]
            for i in range(host.NUM_BLOCKS)
        ]

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Torch in, torch out: x [1, 8, 16, 16, 16], t [1] (x1000), cond [1, L, 1024] -> [1, 8, 16, 16, 16]."""
        dev = self.mesh_device
        tables = self.tables(cond.shape[1])
        mods = self.modulation(t)
        context = bf16_tensor(host.pad_context(cond.float()), device=dev)
        tokens = host.x_to_tokens(x.float())
        if self.precision.fp32_input:
            tokens = hi_lo_split(tokens)
        h = self.input_layer(bf16_tensor(tokens, device=dev))
        if self.precision.fp32_residual:
            h = ttnn.typecast(h, ttnn.float32)
        for block, mod in zip(self.blocks, mods):
            h = block(h, mod, context, tables)
        h = self.final_norm(h)
        h = self.out_layer(
            h if self.precision.fp32_out_input else _norm_in(h),
            dtype=ttnn.float32 if self.precision.fp32_output else None,
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()
        return host.tokens_to_x(out)
