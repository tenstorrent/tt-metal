# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ViT-L/16 image encoder (TRELLIS.2's condition model) on one device.

Mirrors `reference/dinov3_host.functional_forward` op for op. The host unfolds the image into 16x16 patches and uploads
them once; on device a Linear embeds them and a constant table adds the CLS and register tokens. Tokens are padded to a
multiple of 512 and the padding is its own SDPA window (as in the SLat DiTs), so the attention can use large key
chunks. RoPE (rotate-half form, 2D patch centres) runs in fp32 over all tokens with identity rows on the prefix and the
padding. Numerics follow the structure DiT's `Precision()` defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.trellis2.reference import dinov3_host as host
from models.experimental.trellis2.tt.structure_flow import (
    Precision,
    _hifi4,
    _norm_in,
    _residual_add,
    sdpa_program_config,
)
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor


@dataclass
class DinoTables:
    """Per image size: RoPE tables [1, H, n_pad, 64], the rotate-half matrix, the token windows [0, n, n_pad]."""

    cos: ttnn.Tensor
    sin: ttnn.Tensor
    rot: ttnn.Tensor
    windows: ttnn.Tensor
    n: int
    n_pad: int


class LayerScale(Module):
    def __init__(self, dim: int, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.lambda1 = Parameter(total_shape=[1, dim], device=mesh_device, dtype=ttnn.float32)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "lambda1" in state:
            state["lambda1"] = state["lambda1"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.multiply(x, self.lambda1.data)


class Attention(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision) -> None:
        super().__init__()
        d = host.HIDDEN
        mm = _hifi4(mesh_device) if precision.linear_hifi4 else None
        self.q_proj = Linear(d, d, mesh_device=mesh_device, compute_kernel_config=mm)
        self.k_proj = Linear(d, d, bias=False, mesh_device=mesh_device, compute_kernel_config=mm)
        self.v_proj = Linear(d, d, mesh_device=mesh_device, compute_kernel_config=mm)
        self.o_proj = Linear(d, d, mesh_device=mesh_device, compute_kernel_config=mm)
        self.fp32_qk = precision.fp32_qk
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_large_k_chunk = precision.sdpa_large_k_chunk
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if precision.sdpa_hifi4_fp32_acc else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=precision.sdpa_hifi4_fp32_acc,
        )
        self.rope_compute_kernel_config = _hifi4(mesh_device)

    def _heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=host.NUM_HEADS, num_kv_heads=0, transpose_k_heads=False
        )
        return out

    def _rope(self, x: ttnn.Tensor, tables: DinoTables) -> ttnn.Tensor:
        x = self._heads(x)
        if self.fp32_qk:
            x = ttnn.typecast(x, ttnn.float32)
        rotated = ttnn.matmul(x, tables.rot, compute_kernel_config=self.rope_compute_kernel_config)
        return _norm_in(ttnn.add(ttnn.multiply(x, tables.cos), ttnn.multiply(rotated, tables.sin)))

    def forward(self, x: ttnn.Tensor, tables: DinoTables) -> ttnn.Tensor:
        q = self._rope(self.q_proj(x), tables)
        k = self._rope(self.k_proj(x), tables)
        v = self._heads(self.v_proj(x))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            cu_window_seqlens=tables.windows,
            program_config=sdpa_program_config(
                self.grid, tables.n_pad, windowed=True, large_k_chunk=self.sdpa_large_k_chunk
            ),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        return self.o_proj(ttnn.unsqueeze(ttnn.transformer.concatenate_heads(out), 0))


class MLP(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision) -> None:
        super().__init__()
        mm = _hifi4(mesh_device) if precision.linear_hifi4 else None
        self.up_proj = Linear(
            host.HIDDEN, host.MLP, activation_fn="gelu", mesh_device=mesh_device, compute_kernel_config=mm
        )
        self.down_proj = Linear(host.MLP, host.HIDDEN, mesh_device=mesh_device, compute_kernel_config=mm)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.down_proj(self.up_proj(x))


class Layer(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision) -> None:
        super().__init__()
        d = host.HIDDEN
        self.norm1 = LayerNorm(d, norm_eps=host.NORM_EPS, norm_elementwise_affine=True, mesh_device=mesh_device)
        self.attention = Attention(mesh_device=mesh_device, precision=precision)
        self.layer_scale1 = LayerScale(d, mesh_device=mesh_device)
        self.norm2 = LayerNorm(d, norm_eps=host.NORM_EPS, norm_elementwise_affine=True, mesh_device=mesh_device)
        self.mlp = MLP(mesh_device=mesh_device, precision=precision)
        self.layer_scale2 = LayerScale(d, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor, tables: DinoTables) -> ttnn.Tensor:
        x = _residual_add(x, self.layer_scale1(self.attention(_norm_in(self.norm1(x)), tables)))
        return _residual_add(x, self.layer_scale2(self.mlp(_norm_in(self.norm2(x)))))


class TtDinoV3(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, precision: Precision = Precision()) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.precision = precision
        mm = _hifi4(mesh_device) if precision.io_hifi4 else None
        self.patch_embed = Linear(
            3 * host.PATCH * host.PATCH, host.HIDDEN, mesh_device=mesh_device, compute_kernel_config=mm
        )
        self.layer = ModuleList(Layer(mesh_device=mesh_device, precision=precision) for _ in range(host.NUM_LAYERS))
        self.final_norm = LayerNorm(
            host.HIDDEN, norm_eps=host.NORM_EPS, norm_elementwise_affine=False, mesh_device=mesh_device
        )
        self._prefix_rows: torch.Tensor | None = None
        self._tables: dict[int, DinoTables] = {}
        self._prefix: dict[int, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        w = state.pop("embeddings.patch_embeddings.weight", None)
        b = state.pop("embeddings.patch_embeddings.bias", None)
        if w is not None:
            state["patch_embed.weight"] = w.reshape(w.shape[0], -1)
            state["patch_embed.bias"] = b
            # prefix rows enter the patch Linear as zeros, so the table holds the token minus the Linear's bias
            prefix = torch.cat([state["embeddings.cls_token"][0], state["embeddings.register_tokens"][0]])
            self._prefix_rows = prefix.float() - b.float()
        for k in [k for k in state if k.startswith("embeddings.") or k.startswith("norm.")]:
            del state[k]

    def tables(self, size: int) -> DinoTables:
        if size not in self._tables:
            n = host.num_tokens(size)
            n_pad = host.padded_tokens(n)
            cos, sin = host.rope_tables(size, n_pad)
            expand = lambda t: t.reshape(1, 1, *t.shape).expand(1, host.NUM_HEADS, *t.shape).contiguous()
            rope_tensor = float32_tensor if self.precision.fp32_qk else bf16_tensor
            dev = self.mesh_device
            self._tables[size] = DinoTables(
                cos=rope_tensor(expand(cos), device=dev),
                sin=rope_tensor(expand(sin), device=dev),
                rot=rope_tensor(host.rotate_half_matrix(), device=dev),
                windows=ttnn.from_torch(
                    torch.tensor([0, n, n_pad], dtype=torch.int32),
                    device=dev,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                ),
                n=n,
                n_pad=n_pad,
            )
            table = torch.zeros(n_pad, host.HIDDEN)
            table[: host.NUM_PREFIX] = self._prefix_rows
            self._prefix[size] = float32_tensor(table[None, None], device=dev)
        return self._tables[size]

    def forward_tokens(self, patches: ttnn.Tensor, size: int) -> ttnn.Tensor:
        """Device in, device out: patches [1, 1, n_pad, 768] (zero rows for prefix and padding) -> [1, 1, n_pad, 1024]
        fp32; rows past the real tokens are don't-care."""
        tables = self.tables(size)
        h = ttnn.add(ttnn.typecast(self.patch_embed(patches), ttnn.float32), self._prefix[size])
        if not self.precision.fp32_residual:
            h = ttnn.typecast(h, ttnn.bfloat16)
        for layer in self.layer:
            h = layer(h, tables)
        return self.final_norm(h)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Torch in, torch out: pixels [1, 3, S, S] (preprocessed) -> cond [1, N, 1024]."""
        size = pixels.shape[-1]
        tables = self.tables(size)
        rows = torch.zeros(tables.n_pad, 3 * host.PATCH * host.PATCH)
        rows[host.NUM_PREFIX : tables.n] = host.patchify(pixels.float())
        out = self.forward_tokens(bf16_tensor(rows[None, None], device=self.mesh_device), size)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, :, : tables.n]
