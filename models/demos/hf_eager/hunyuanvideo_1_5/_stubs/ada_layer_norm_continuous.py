# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_continuous` of tencent/HunyuanVideo-1.5.

Reference submodule: `norm_out`, a diffusers `AdaLayerNormContinuous`
(embedding_dim == conditioning_embedding_dim == inner_dim, norm_type
"layer_norm", elementwise_affine=False, eps=1e-6):

    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))  # Linear(C, 2C)
    scale, shift = torch.chunk(emb, 2, dim=1)                          # each (B, C)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]     # LayerNorm(C), no affine

Inputs at test time:
    x                     : (B, L, C)  — primary tensor (arrives as a ttnn tensor)
    conditioning_embedding: (B, C)     — per-batch vector (arrives as a torch tensor)

Native ttnn strategy
--------------------
`torch.chunk(linear(...), 2, dim=1)` splits the 2C linear output into the
`scale` and `shift` halves. Column j of the output is `silu · W[j] + b[j]`, so
the first C columns depend only on rows 0:C of the weight and the last C on rows
C:2C. We therefore split the host weight into two (C, C) matmuls at build time
(`w_scale`, `w_shift` + their bias halves) — mathematically identical to the
chunk, but avoids slicing a device activation.

    silu  = ttnn.silu(conditioning_embedding)
    scale = silu @ w_scale + b_scale        # (B, C)
    shift = silu @ w_shift + b_shift        # (B, C)
    norm  = ttnn.layer_norm(x, eps)         # no weight/bias (elementwise_affine=False)
    out   = norm * (1 + scale)[:, None, :] + shift[:, None, :]

All device math runs in float32 with a HiFi4 compute config for a clean PCC.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the AdaLayerNormContinuous weights and return a native ttnn forward."""

    m = torch_module
    lin = m.linear  # Linear(conditioning_dim, 2 * embedding_dim)
    norm = m.norm  # LayerNorm(embedding_dim, elementwise_affine=False)

    out_features = int(lin.out_features)
    embedding_dim = out_features // 2  # C
    eps = float(getattr(norm, "eps", 1e-6))

    # LayerNorm is constructed with elementwise_affine=False here, so it carries
    # no weight/bias; guard anyway in case a variant enables affine.
    norm_weight = getattr(norm, "weight", None)
    norm_bias = getattr(norm, "bias", None)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    w = lin.weight.detach()  # (2C, conditioning_dim)
    b = lin.bias.detach() if lin.bias is not None else None

    # chunk(emb, 2, dim=1): first C output cols <- weight rows 0:C, last C <- rows C:2C.
    w_scale = f32(w[:embedding_dim, :].t())  # (conditioning_dim, C)
    w_shift = f32(w[embedding_dim:, :].t())  # (conditioning_dim, C)
    if b is not None:
        b_scale = f32(b[:embedding_dim].reshape(1, embedding_dim))
        b_shift = f32(b[embedding_dim:].reshape(1, embedding_dim))
    else:
        b_scale = b_shift = None

    ttnn_norm_w = f32(norm_weight.detach().reshape(1, 1, embedding_dim)) if norm_weight is not None else None
    ttnn_norm_b = f32(norm_bias.detach().reshape(1, 1, embedding_dim)) if norm_bias is not None else None

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _wln(x, eps):
        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
        # ragged tile-row). Spread it over a width-sharded row of cores with a
        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
        # on the ragged logical row). Same lever proven on the transformer block.
        B, L, Cx = (int(d) for d in x.shape)
        # TT pads each logical dimension independently; derive the flattened
        # shard height from physical storage rather than rounded logical B*L.
        padded_m = 1
        padded_shape = list(x.padded_shape)
        for d in padded_shape[:-1]:
            padded_m *= int(d)
        Mt = padded_m // 32
        Nt = int(padded_shape[-1]) // 32
        gx = 8
        while gx > 1 and Nt % gx != 0:
            gx -= 1
        # The sharded LayerNorm kernel allocates multiple static circular
        # buffers beyond the tensor shard. At 13f, block_h=25 measured 2.35 MiB
        # of CBs and exceeded Blackhole's 1.5 MiB L1 despite a 400 KiB shard.
        # Retain this dispatch optimization only for its tuned tiny shapes.
        shard_bytes = (Mt * 32) * ((Nt // gx) * 32) * 2
        if Mt > 8 or 6 * shard_bytes > 1_300_000:
            return ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)
        shard_shape = [Mt * 32, (Nt // gx) * 32]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
        xs = ttnn.to_memory_config(x, mem)
        bw = Nt // gx
        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
        while sw > 1 and bw % sw != 0:
            sw -= 1
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, 1),
            subblock_w=sw,
            block_h=Mt,
            block_w=bw,
            inplace=False,
        )
        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

    def forward(x, conditioning_embedding=None, *args, **kwargs):
        if conditioning_embedding is None:
            if args:
                conditioning_embedding = args[0]
            else:
                raise TypeError("ada_layer_norm_continuous forward needs `conditioning_embedding`")

        x = _to_f32_device(x)
        cond = _to_f32_device(conditioning_embedding)

        # emb = linear(silu(conditioning_embedding)), split into scale / shift.
        s = ttnn.silu(cond)  # (B, C_cond)
        # Fold bias into the matmul epilogue (launch-bound path: fewer op launches).
        if b_scale is not None:
            scale = ttnn.linear(s, w_scale, bias=b_scale, compute_kernel_config=compute_config)
            shift = ttnn.linear(s, w_shift, bias=b_shift, compute_kernel_config=compute_config)
        else:
            scale = ttnn.matmul(s, w_scale, compute_kernel_config=compute_config)
            shift = ttnn.matmul(s, w_shift, compute_kernel_config=compute_config)

        # (B, C) -> (B, 1, C) so it broadcasts over x's sequence dim.
        b_dim = int(scale.shape[0])
        scale = ttnn.reshape(scale, (b_dim, 1, embedding_dim))
        shift = ttnn.reshape(shift, (b_dim, 1, embedding_dim))

        if ttnn_norm_w is None and ttnn_norm_b is None:
            norm = _wln(x, eps)  # non-affine: width-sharded (shard knob)
        else:
            norm = ttnn.layer_norm(
                x, epsilon=eps, weight=ttnn_norm_w, bias=ttnn_norm_b, compute_kernel_config=compute_config
            )

        # norm*(1+scale)+shift fused into one ternary launch (was add(mul(norm,·),shift)).
        out = ttnn.addcmul(shift, norm, ttnn.add(scale, 1.0))
        return out

    return forward


def ada_layer_norm_continuous(*args, **kwargs):
    raise RuntimeError(
        "ada_layer_norm_continuous requires build(device, torch_module) to bind the "
        "linear/norm weights; the bare callable has no parameters."
    )
