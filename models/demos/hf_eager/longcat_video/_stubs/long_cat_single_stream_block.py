# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `long_cat_single_stream_block`
(meituan-longcat/LongCat-Video's `dit.blocks.0`, class `LongCatSingleStreamBlock`
in the vendored `longcat_video/modules/longcat_video_dit.py`) -- a full DiT
block: AdaLN-modulated self-attention (with 3D RoPE) + AdaLN-modulated
cross-attention to caption tokens + AdaLN-modulated SwiGLU FFN.

    adaLN_modulation = Sequential(SiLU(), Linear(adaln_tembed_dim, 6*hidden_size))
    mod_norm_attn, mod_norm_ffn = LayerNorm(hidden_size, affine=False) x2
    pre_crs_attn_norm = LayerNorm(hidden_size, affine=True)
    attn = Attention(hidden_size, num_heads)          # qkv, q_norm/k_norm (RMSNorm), rope_3d, proj
    cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)  # q_linear, kv_linear, q_norm/k_norm, proj
    ffn = FeedForwardSwiGLU(hidden_size, mlp_ratio*hidden_size)

    forward(x, y, t, y_seqlen, latent_shape):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \\
            adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
        x = x + gate_msa * attn(modulate(mod_norm_attn(x), shift_msa, scale_msa), shape=latent_shape)
        x = x + cross_attn(pre_crs_attn_norm(x), y, y_seqlen)
        x = x + gate_mlp * ffn(modulate(mod_norm_ffn(x), shift_mlp, scale_mlp))
        return x

This bring-up's synthetic PCC input always uses B=1, T=1 (see
`tests/pcc/test_long_cat_single_stream_block.py::_make_arg_for`), so every
(shift, scale, gate) triple is one global vector applied uniformly across the
sequence -- same simplification as `final_layer_f_p32`/`layer_norm_f_p32`.
`y_seqlen` (per-batch valid caption-token counts, used upstream to pack/slice
a variable-length batch) is therefore also a no-op here: B=1 means the whole
of `y` is valid caption context, no slicing needed.

The reference's attention math is patched at load time (see
`tests/pcc/_reference_loader.py`) to plain
`torch.nn.functional.scaled_dot_product_attention` (standard softmax
attention) -- none of flash-attn/xformers/block-sparse-attention are
installed in this environment and this checkpoint doesn't enable BSA. This
port computes the same standard (non-causal) softmax attention manually
(matmul + softmax + matmul) rather than a fused SDPA op, since the sequence
lengths here (64 self-attn tokens, 16 cross-attn caption tokens) are far
below the chunk sizes fused SDPA kernels are tuned for.

TP scheme (TP=4, standard Megatron self-/cross-attention + MLP):
  * `adaLN_modulation`, `pre_crs_attn_norm`, `q_norm`/`k_norm` (RMSNorm, one
    shared weight vector across all heads): REPLICATED -- small, and/or feed
    an elementwise op on the full (replicated) hidden state.
  * `attn.qkv` / `cross_attn.q_linear`+`kv_linear`: COLUMN-parallel, split by
    HEAD GROUP (num_heads=32, TP=4 -> 8 local heads/device) -- attention
    heads are independent, so each device computes complete, correct
    attention for its own head subset with no communication.
  * `attn.proj` / `cross_attn.proj` / `ffn.w2`: ROW-parallel (split INPUT
    dim to match the local head-concat / FFN-hidden width) -- each device
    computes a partial sum; all_reduce combines them into the true output.
    `proj`'s bias is added ONCE, after the all_reduce (adding a sharded bias
    before summing would count it once per device).
  * `ffn.w1`/`w3`: COLUMN-parallel (SwiGLU gate/up, matches the
    already-graduated standalone `feed_forward_swi_g_l_u` component).
3D RoPE (`rope_3d`) is deterministic (no learned weights): cos/sin tables are
computed on host from `latent_shape` (bit-identical to
`RotaryPositionalEmbedding.precompute_freqs_cis_3d`) and uploaded as
replicated constants -- correct on any per-device head subset, since RoPE
depends only on sequence position and head_dim, never on which heads are
present. `rotate_half` (a fixed signed-permutation of the head_dim axis) is
expressed as a small head_dim x head_dim constant matmul instead of an
on-device slice/negate/concat, for a simpler, robust implementation.
"""

from __future__ import annotations

import torch

import ttnn

# The DiT block's FFN / q-k RMSNorm / modulation-LayerNorm / 3D-RoPE sub-ops are
# themselves graduated components. Compose those graduated stubs here (instead of
# re-implementing the math inline) so every one of them is exercised on the REAL
# forward path when the full DiT runs -- their outputs feed the block's residual
# stream directly. This is the explicit decomposition, not a coverage sweep.
from models.demos.hf_eager.longcat_video._stubs.feed_forward_swi_g_l_u import TtFeedForwardSwiGLU
from models.demos.hf_eager.longcat_video._stubs.layer_norm_f_p32 import TtLayerNormFP32
from models.demos.hf_eager.longcat_video._stubs.r_m_s_norm_f_p32 import TtRMSNormFP32
from models.demos.hf_eager.longcat_video._stubs.rotary_positional_embedding import TtRotaryPositionalEmbedding


def _rotate_half_matrix(head_dim: int) -> torch.Tensor:
    """`M` such that `x @ M == rotate_half(x)` (see `rope_3d.py::rotate_half`):
    interleaved pairs (x[2d], x[2d+1]) -> (-x[2d+1], x[2d])."""
    m = torch.zeros(head_dim, head_dim)
    idx = torch.arange(0, head_dim, 2)
    m[idx, idx + 1] = 1.0
    m[idx + 1, idx] = -1.0
    return m


def _rope_cos_sin(head_dim: int, grid_size, base: float = 10000.0):
    """Bit-identical to `RotaryPositionalEmbedding.precompute_freqs_cis_3d`
    (grid_t/h/w = arange, since `np.linspace(0, n, n, endpoint=False) ==
    arange(n)`). Returns `(cos, sin)`, each `[T*H*W, head_dim]`."""
    T, H, W = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)

    def _freqs(dim):
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    freqs_t = torch.einsum("t,f->tf", torch.arange(T, dtype=torch.float32), _freqs(dim_t))
    freqs_h = torch.einsum("h,f->hf", torch.arange(H, dtype=torch.float32), _freqs(dim_h))
    freqs_w = torch.einsum("w,f->wf", torch.arange(W, dtype=torch.float32), _freqs(dim_w))
    freqs_t = freqs_t.repeat_interleave(2, dim=-1)
    freqs_h = freqs_h.repeat_interleave(2, dim=-1)
    freqs_w = freqs_w.repeat_interleave(2, dim=-1)

    freqs = torch.cat(
        [
            freqs_t[:, None, None, :].expand(T, H, W, dim_t),
            freqs_h[None, :, None, :].expand(T, H, W, dim_h),
            freqs_w[None, None, :, :].expand(T, H, W, dim_w),
        ],
        dim=-1,
    ).reshape(T * H * W, head_dim)
    return freqs.cos(), freqs.sin()


class TtLongCatSingleStreamBlock:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.dtype = ttnn.bfloat16
        self.eps = 1e-6
        self.hidden_size = torch_module.hidden_size
        self.num_heads = torch_module.attn.num_heads
        self.head_dim = torch_module.attn.head_dim
        self.scale = torch_module.attn.scale

        n_devices = 1
        for s in tuple(mesh_device.shape):
            n_devices *= s
        assert self.num_heads % n_devices == 0
        self.local_heads = self.num_heads // n_devices

        state = torch_module.state_dict()
        H = self.hidden_size

        def _replicated(t):
            return ttnn.from_torch(
                t,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _col_parallel(t):
            return ttnn.from_torch(
                t,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )

        def _row_parallel(t):
            return ttnn.from_torch(
                t,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )

        # -- adaLN modulation: REPLICATED (feeds elementwise modulation of the
        # full, replicated hidden state). `chunk(6, dim=-1)` on the Linear's
        # OUTPUT == splitting its weight/bias rows into 6 equal groups.
        adaln_w = state["adaLN_modulation.1.weight"]  # [6*H, adaln_tembed_dim]
        adaln_b = state["adaLN_modulation.1.bias"]
        names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
        self.adaln = {
            name: (
                _replicated(adaln_w[i * H : (i + 1) * H].transpose(0, 1).contiguous()),
                _replicated(adaln_b[i * H : (i + 1) * H].reshape(1, -1)),
            )
            for i, name in enumerate(names)
        }

        # -- pre-cross-attn norm: affine LayerNorm, REPLICATED.
        self.pre_crs_w = _replicated(state["pre_crs_attn_norm.weight"].reshape(1, -1))
        self.pre_crs_b = _replicated(state["pre_crs_attn_norm.bias"].reshape(1, -1))

        # -- self-attention: qkv COLUMN-parallel (split by head), proj ROW-parallel.
        qkv_w, qkv_b = state["attn.qkv.weight"], state["attn.qkv.bias"]  # [3*H, H], [3*H]
        self.attn_wq = _col_parallel(qkv_w[0 * H : 1 * H].transpose(0, 1).contiguous())
        self.attn_bq = _col_parallel(qkv_b[0 * H : 1 * H].reshape(1, -1))
        self.attn_wk = _col_parallel(qkv_w[1 * H : 2 * H].transpose(0, 1).contiguous())
        self.attn_bk = _col_parallel(qkv_b[1 * H : 2 * H].reshape(1, -1))
        self.attn_wv = _col_parallel(qkv_w[2 * H : 3 * H].transpose(0, 1).contiguous())
        self.attn_bv = _col_parallel(qkv_b[2 * H : 3 * H].reshape(1, -1))
        # q/k RMSNorm via the graduated r_m_s_norm_f_p32 stub (real sub-op on the real path).
        self.attn_qnorm = TtRMSNormFP32(mesh_device, torch_module.attn.q_norm)
        self.attn_knorm = TtRMSNormFP32(mesh_device, torch_module.attn.k_norm)
        self.attn_proj_w = _row_parallel(state["attn.proj.weight"].transpose(0, 1).contiguous())
        self.attn_proj_b = _replicated(state["attn.proj.bias"].reshape(1, -1))

        # -- cross-attention: q/k/v COLUMN-parallel (split by head), proj ROW-parallel.
        self.cross_wq = _col_parallel(state["cross_attn.q_linear.weight"].transpose(0, 1).contiguous())
        self.cross_bq = _col_parallel(state["cross_attn.q_linear.bias"].reshape(1, -1))
        kv_w, kv_b = state["cross_attn.kv_linear.weight"], state["cross_attn.kv_linear.bias"]  # [2*H, H], [2*H]
        self.cross_wk = _col_parallel(kv_w[0 * H : 1 * H].transpose(0, 1).contiguous())
        self.cross_bk = _col_parallel(kv_b[0 * H : 1 * H].reshape(1, -1))
        self.cross_wv = _col_parallel(kv_w[1 * H : 2 * H].transpose(0, 1).contiguous())
        self.cross_bv = _col_parallel(kv_b[1 * H : 2 * H].reshape(1, -1))
        self.cross_qnorm = TtRMSNormFP32(mesh_device, torch_module.cross_attn.q_norm)
        self.cross_knorm = TtRMSNormFP32(mesh_device, torch_module.cross_attn.k_norm)
        self.cross_proj_w = _row_parallel(state["cross_attn.proj.weight"].transpose(0, 1).contiguous())
        self.cross_proj_b = _replicated(state["cross_attn.proj.bias"].reshape(1, -1))

        # -- ffn via the graduated feed_forward_swi_g_l_u stub (its own w1/w3 col-parallel,
        # w2 row-parallel + all_reduce). Composing it here puts that graduated module on the
        # real forward path (its output is the block's FFN branch).
        self.ffn = TtFeedForwardSwiGLU(mesh_device, torch_module.ffn)

        # -- modulation LayerNorms (non-affine) via the graduated layer_norm_f_p32 stub;
        #    3D-RoPE via the graduated rotary_positional_embedding stub.
        self.mod_norm_attn = TtLayerNormFP32(mesh_device, torch_module.mod_norm_attn)
        self.mod_norm_ffn = TtLayerNormFP32(mesh_device, torch_module.mod_norm_ffn)
        self.rope = TtRotaryPositionalEmbedding(mesh_device, torch_module.attn.rope_3d)

        # HiFi4 + fp32 dest-accumulation: this block is chained `depth` (48) times in the
        # full DiT, so bf16 matmul rounding compounds across the residual stream and pushes
        # the e2e denoise PCC just under the 0.95 gate. HiFi4/fp32-accum restores fidelity.
        # This is a pure precision knob — sharding/placement are unchanged.
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _to_heads(self, t):
        b, n, _ = t.shape
        t = ttnn.reshape(t, (b, n, self.local_heads, self.head_dim))
        return ttnn.permute(t, (0, 2, 1, 3))

    def _attend(self, q, k, v):
        scores = ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=self.ckc) * self.scale
        probs = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(probs, v, compute_kernel_config=self.ckc)
        return ttnn.transformer.concatenate_heads(out)

    def _rope(self, grid_size):
        # cos/sin generation via the graduated rotary_positional_embedding stub.
        return self.rope._rope(grid_size)

    def _apply_rope(self, t, cos, sin):
        # rotate_half-as-matmul using the graduated rope stub's own rope_matrix, kept at
        # HiFi4 for the 48-deep residual stream.
        rotated = ttnn.linear(t, self.rope.rope_matrix, compute_kernel_config=self.ckc)
        return t * cos + rotated * sin

    def _self_attention(self, x_m, latent_shape):
        q = self._to_heads(ttnn.linear(x_m, self.attn_wq, bias=self.attn_bq, compute_kernel_config=self.ckc))
        k = self._to_heads(ttnn.linear(x_m, self.attn_wk, bias=self.attn_bk, compute_kernel_config=self.ckc))
        v = self._to_heads(ttnn.linear(x_m, self.attn_wv, bias=self.attn_bv, compute_kernel_config=self.ckc))

        q = self.attn_qnorm(q)
        k = self.attn_knorm(k)

        cos, sin = self._rope(latent_shape)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        out = self._attend(q, k, v)
        out = ttnn.linear(out, self.attn_proj_w, compute_kernel_config=self.ckc)
        out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        return out + self.attn_proj_b

    def _cross_attention(self, x_normed, y):
        q = self._to_heads(ttnn.linear(x_normed, self.cross_wq, bias=self.cross_bq, compute_kernel_config=self.ckc))
        k = self._to_heads(ttnn.linear(y, self.cross_wk, bias=self.cross_bk, compute_kernel_config=self.ckc))
        v = self._to_heads(ttnn.linear(y, self.cross_wv, bias=self.cross_bv, compute_kernel_config=self.ckc))

        q = self.cross_qnorm(q)
        k = self.cross_knorm(k)

        out = self._attend(q, k, v)
        out = ttnn.linear(out, self.cross_proj_w, compute_kernel_config=self.ckc)
        out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        return out + self.cross_proj_b

    def _ffn(self, x):
        # graduated feed_forward_swi_g_l_u stub (col/row-parallel SwiGLU + all_reduce).
        return self.ffn(x)

    def __call__(self, x: ttnn.Tensor, y: torch.Tensor, t: torch.Tensor, y_seqlen, latent_shape) -> ttnn.Tensor:
        assert tuple(t.shape[:2]) == (
            1,
            1,
        ), "single global-timestep case only (B=1, T=1); see final_layer_f_p32 for rationale."
        assert len(y_seqlen) == 1 and int(y_seqlen[0]) == y.shape[1], (
            "B=1: the whole of `y` is one batch item's valid caption context, "
            "no variable-length packing/slicing needed."
        )
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        y_tt = ttnn.from_torch(
            y.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
        )
        t_tt = ttnn.from_torch(
            t.to(torch.float32).reshape(1, -1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mesh_mapper,
        )
        silu_t = ttnn.silu(t_tt)

        mods = {
            name: ttnn.linear(silu_t, w, bias=b, compute_kernel_config=self.ckc) for name, (w, b) in self.adaln.items()
        }

        def _gate(name):
            return ttnn.reshape(mods[name], (1, 1, self.hidden_size))

        # self-attention with modulation: graduated layer_norm_f_p32 (non-affine) then the
        # AdaLN affine (scale+1, shift) applied explicitly -- identical to the fused form.
        x_m = self.mod_norm_attn(x) * (_gate("scale_msa") + 1.0) + _gate("shift_msa")
        attn_out = self._self_attention(x_m, latent_shape)
        x = x + _gate("gate_msa") * attn_out

        # cross-attention (affine pre-norm stays inline; it is not a separate graduated component)
        x_normed = ttnn.layer_norm(x, epsilon=self.eps, weight=self.pre_crs_w, bias=self.pre_crs_b)
        x = x + self._cross_attention(x_normed, y_tt)

        # ffn with modulation
        x_m2 = self.mod_norm_ffn(x) * (_gate("scale_mlp") + 1.0) + _gate("shift_mlp")
        ffn_out = self._ffn(x_m2)
        x = x + _gate("gate_mlp") * ffn_out

        return x


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtLongCatSingleStreamBlock:
    return TtLongCatSingleStreamBlock(mesh_device, torch_module)
