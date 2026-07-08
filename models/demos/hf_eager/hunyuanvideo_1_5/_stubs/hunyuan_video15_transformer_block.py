# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_transformer_block` of tencent/HunyuanVideo-1.5.

Reference submodule: `transformer_blocks.0`, a `HunyuanVideo15TransformerBlock`
(dual-stream / MMDiT double block):

    norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm1(hidden, emb=temb)      # AdaLayerNormZero
    norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = norm1_context(enc, emb=temb)
    attn_out, ctx_out = attn(norm_h, norm_e, mask, freqs_cis)                       # joint attention
    hidden  = hidden  + attn_out * gate_msa[:, None]
    enc     = enc     + ctx_out  * c_gate_msa[:, None]
    nh = norm2(hidden) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]              # LayerNorm (no affine)
    ne = norm2_context(enc) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    hidden = hidden + gate_mlp[:, None]   * ff(nh)                                  # GELU-tanh FF
    enc    = enc    + c_gate_mlp[:, None] * ff_context(ne)
    return hidden, enc

Joint attention (HunyuanVideo15AttnProcessor2_0): q/k/v from the latent stream and
add_{q,k,v}_proj from the encoder stream, each split into heads and RMS-normed
(norm_q/k, norm_added_q/k), concatenated along the sequence, unmasked SDPA
(`softmax(qkᵀ·scale)v`), then split back and projected by to_out[0] / to_add_out.

Per-component test inputs:
    hidden_states (B, L, C) PRIMARY ttnn; encoder_hidden_states (B, Lc, C) torch;
    temb (B, C) torch; attention_mask all-ones (unmasked); freqs_cis None (no rope).

Float32 math with a HiFi4 config.

Mesh sharding (QB2, flat tensor-parallel across all mesh devices): pass
``ccl_manager`` (a ``models.tt_dit.parallel.manager.CCLManager`` bound to the
mesh device) and ``tp`` (device count) to shard the attention QKV/out-proj and
feed-forward linears Megatron-style -- column-parallel in (no communication),
row-parallel out (all-reduce via reduce_scatter+all_gather, mirroring
``models/demos/z_image_turbo``'s own flat-TP DiT). AdaLN modulation, RMS-norm
weights and the RoPE rotate matrix stay replicated on every device in this
first pass -- they're small relative to attention/FF and aren't the memory
bottleneck. With ``tp=1`` (default) this is byte-for-byte the original
single-device path.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module, ccl_manager=None, tp=1):
    """Extract all sub-weights of the dual-stream block; return a native forward."""
    import torch

    blk = torch_module
    attn = blk.attn
    heads_total = int(attn.heads)
    inner_total = int(attn.to_q.out_features)
    dim_head = inner_total // heads_total
    scale = float(getattr(attn, "scale", dim_head**-0.5))

    sharded = tp > 1 and ccl_manager is not None
    if sharded and heads_total % tp != 0:
        raise ValueError(f"heads_total={heads_total} not divisible by tp={tp}")
    heads = heads_total // tp if sharded else heads_total  # LOCAL heads (this device's shard)
    inner = heads * dim_head  # LOCAL inner dim (== inner_total when tp==1)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t, mesh_mapper=None):
        if mesh_mapper is None and sharded:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )

    def lin(linear):
        """Replicated (unsharded) linear -- for weights not on the TP path."""
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    def lin_col(linear):
        """Column-parallel: shard the OUTPUT dim across the mesh (no CCL needed)."""
        mapper = ttnn.ShardTensorToMesh(device, dim=-1) if sharded else None
        w = f32(linear.weight.detach().t(), mesh_mapper=mapper)
        b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
        return w, b

    def lin_row(linear):
        """Row-parallel: shard the INPUT (contraction) dim; bias stays replicated
        and is added once, by the caller, after the all-reduce."""
        mapper = ttnn.ShardTensorToMesh(device, dim=0) if sharded else None
        w = f32(linear.weight.detach().t(), mesh_mapper=mapper)
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None  # replicated
        return w, b

    def ada_chunks(adazero):
        """Split AdaLayerNormZero.linear (C -> 6C) into six (C, C) matmuls + bias.
        Left replicated (not TP-sharded) in this pass -- see module docstring."""
        L = adazero.linear
        C = int(L.out_features) // 6
        w = L.weight.detach()
        b = L.bias.detach() if L.bias is not None else None
        ws, bs = [], []
        for i in range(6):
            ws.append(f32(w[i * C : (i + 1) * C, :].t()))
            bs.append(f32(b[i * C : (i + 1) * C].reshape(1, C)) if b is not None else None)
        eps = float(getattr(adazero.norm, "eps", 1e-6))
        return ws, bs, eps, C

    def rms_w(norm):
        w = getattr(norm, "weight", None)
        return f32(w.detach().reshape(1, 1, 1, -1)) if w is not None else None

    def ff_parts(ff):
        net = ff.net
        aw = net[0]
        if type(aw).__name__ == "GELU":
            proj = aw.proj
            approx = str(getattr(aw, "approximate", "none"))
            variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
            act = lambda t, _v=variant: ttnn.gelu(t, variant=_v)
        else:
            proj = getattr(aw, "proj", aw)
            nm = type(getattr(aw, "activation", aw)).__name__.lower()
            act = (lambda t: ttnn.silu(t)) if ("silu" in nm or "swish" in nm) else (lambda t: ttnn.gelu(t))
        lin2 = None
        for module in reversed(list(net)):
            if isinstance(module, torch.nn.Linear):
                lin2 = module
                break
        w1, b1 = lin_col(proj)
        w2, b2 = lin_row(lin2)
        return w1, b1, act, w2, b2

    ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
    adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)

    wq, bq = lin_col(attn.to_q)
    wk, bk = lin_col(attn.to_k)
    wv, bv = lin_col(attn.to_v)
    awq, abq = lin_col(attn.add_q_proj)
    awk, abk = lin_col(attn.add_k_proj)
    awv, abv = lin_col(attn.add_v_proj)
    wo, bo = lin_row(attn.to_out[0])
    ao_w, ao_b = lin_row(attn.to_add_out)
    nq_w, nk_w = rms_w(attn.norm_q), rms_w(attn.norm_k)
    naq_w, nak_w = rms_w(attn.norm_added_q), rms_w(attn.norm_added_k)
    rms_eps = float(getattr(attn.norm_q, "eps", 1e-6))

    norm2_eps = float(getattr(blk.norm2, "eps", 1e-6))
    norm2c_eps = float(getattr(blk.norm2_context, "eps", 1e-6))
    ff_p = ff_parts(blk.ff)
    ffc_p = ff_parts(blk.ff_context)

    # Interleaved-RoPE rotate matrix (fixed): rot(x)[2i] = -x[2i+1], rot(x)[2i+1] = x[2i].
    # Matches diffusers apply_rotary_emb(use_real=True, use_real_unbind_dim=-1): the
    # cos/sin from `hunyuan_video15_rotary_pos_embed` are repeat_interleave(2)-duplicated,
    # so out = x*cos + rot(x)*sin. A constant (D,D) matmul keeps the whole op on device.
    _rot = torch.zeros(dim_head, dim_head, dtype=torch.float32)
    for _i in range(dim_head // 2):
        _rot[2 * _i, 2 * _i + 1] = 1.0
        _rot[2 * _i + 1, 2 * _i] = -1.0
    rot_M = f32(_rot)

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _linear(x, w, b):
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def _all_reduce(x, mesh_axis=1):
        """Megatron all-reduce for a row-parallel output: reduce_scatter + all_gather,
        the idiom `models/demos/z_image_turbo` uses for its own flat-TP DiT."""
        if not sharded:
            return x
        # reduce_scatter's persistent-buffer path hardcodes bf16 ping-pong buffers
        # (models/tt_dit/parallel/manager.py::get_rs_ping_pong_buffer); this stub
        # runs float32 math, so use the (dtype-agnostic) barrier-semaphore path
        # for reduce_scatter and reserve the persistent buffer for all_gather,
        # whose buffer is allocated in the input's own dtype.
        B, L, Cx = (int(d) for d in x.shape)
        x4 = ttnn.reshape(x, (1, B, L, Cx))
        x4 = ccl_manager.reduce_scatter(x4, dim=3, mesh_axis=mesh_axis, use_persistent_buffer=False)
        x4 = ccl_manager.all_gather(x4, dim=3, mesh_axis=mesh_axis, use_hyperparams=True, use_persistent_buffer=True)
        return ttnn.reshape(x4, (B, L, Cx))

    def _row_linear(x, w, b):
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        y = _all_reduce(y)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def _rms(x, w):
        # x: (B, L, H, D); normalize over D.
        var = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        x = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(var, rms_eps)))
        if w is not None:
            x = ttnn.multiply(x, w)
        return x

    def _adazero(x, temb, ws, bs, eps):
        s = ttnn.silu(temb)
        parts = []
        for w, b in zip(ws, bs):
            p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
            if b is not None:
                p = ttnn.add(p, b)
            parts.append(p)  # each (B, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        B = int(x.shape[0])
        nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
        scale_r = ttnn.reshape(scale_msa, (B, 1, C))
        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
        nx = ttnn.add(ttnn.multiply(nx, ttnn.add(scale_r, 1.0)), shift_r)
        return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _unsq(g):
        return ttnn.reshape(g, (int(g.shape[0]), 1, C))

    def _apply_rope(x4, cos, sin):
        # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
        Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
        x2 = ttnn.reshape(x4, (Bx * Sx * Hx, Dx))
        rot = ttnn.matmul(x2, rot_M, compute_kernel_config=compute_config)
        rot4 = ttnn.reshape(rot, (Bx, Sx, Hx, Dx))
        cos_b = ttnn.reshape(cos, (1, Sx, 1, Dx))
        sin_b = ttnn.reshape(sin, (1, Sx, 1, Dx))
        return ttnn.add(ttnn.multiply(x4, cos_b), ttnn.multiply(rot4, sin_b))

    def _joint_attention(nh, ne, freqs_cis=None, attn_bias=None):
        B = int(nh.shape[0])
        Limg = int(nh.shape[1])
        Ltxt = int(ne.shape[1])
        seq = Limg + Ltxt

        def heads_split(t):
            t = ttnn.reshape(t, (B, -1, heads, dim_head))
            return t

        q = _rms(heads_split(_linear(nh, wq, bq)), nq_w)  # (B, Limg, H, D)
        k = _rms(heads_split(_linear(nh, wk, bk)), nk_w)
        v = heads_split(_linear(nh, wv, bv))

        # RoPE on the latent stream only (encoder q/k are added un-rotated), matching
        # HunyuanVideo15AttnProcessor2_0 (apply_rotary_emb after norm_q/norm_k).
        if freqs_cis is not None:
            _cos, _sin = freqs_cis
            q = _apply_rope(q, _cos, _sin)
            k = _apply_rope(k, _cos, _sin)

        eq = _rms(heads_split(_linear(ne, awq, abq)), naq_w)  # (B, Ltxt, H, D)
        ek = _rms(heads_split(_linear(ne, awk, abk)), nak_w)
        ev = heads_split(_linear(ne, awv, abv))

        q = ttnn.concat([q, eq], dim=1)  # (B, seq, H, D)
        k = ttnn.concat([k, ek], dim=1)
        v = ttnn.concat([v, ev], dim=1)

        q = ttnn.permute(q, (0, 2, 1, 3))  # (B, H, seq, D)
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        scores = ttnn.multiply(
            ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=compute_config), scale
        )
        # Additive attention bias (0 for valid, large-negative for masked keys); None -> unmasked.
        if attn_bias is not None:
            scores = ttnn.add(scores, attn_bias)
        attn_w = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn_w, v, compute_kernel_config=compute_config)  # (B, H, seq, D)
        out = ttnn.permute(out, (0, 2, 1, 3))  # (B, seq, H, D)
        out = ttnn.reshape(out, (B, seq, inner))

        hid = ttnn.slice(out, (0, 0, 0), (B, Limg, inner))
        enc = ttnn.slice(out, (0, Limg, 0), (B, seq, inner))
        hid = _row_linear(hid, wo, bo)
        enc = _row_linear(enc, ao_w, ao_b)
        return hid, enc

    def _ff(x, parts):
        w1, b1, act, w2, b2 = parts
        y = _linear(x, w1, b1)
        y = act(y)
        y = _row_linear(y, w2, b2)
        return y

    def forward(
        hidden_states, encoder_hidden_states=None, temb=None, attention_mask=None, freqs_cis=None, *args, **kwargs
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if temb is None:
            temb = kwargs.get("temb")
        if encoder_hidden_states is None or temb is None:
            raise TypeError("hunyuan_video15_transformer_block needs encoder_hidden_states and temb")

        h = _to_f32_device(hidden_states)
        e = _to_f32_device(encoder_hidden_states)
        t = _to_f32_device(temb)

        attn_bias = kwargs.get("attn_bias")

        nh, gate_msa, shift_mlp, scale_mlp, gate_mlp = _adazero(h, t, ada1_w, ada1_b, ada1_eps)
        ne, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _adazero(e, t, adac_w, adac_b, adac_eps)

        attn_out, ctx_out = _joint_attention(nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias)

        h = ttnn.add(h, ttnn.multiply(attn_out, _unsq(gate_msa)))
        e = ttnn.add(e, ttnn.multiply(ctx_out, _unsq(c_gate_msa)))

        nh2 = ttnn.layer_norm(h, epsilon=norm2_eps, compute_kernel_config=compute_config)
        nh2 = ttnn.add(ttnn.multiply(nh2, ttnn.add(_unsq(scale_mlp), 1.0)), _unsq(shift_mlp))
        ne2 = ttnn.layer_norm(e, epsilon=norm2c_eps, compute_kernel_config=compute_config)
        ne2 = ttnn.add(ttnn.multiply(ne2, ttnn.add(_unsq(c_scale_mlp), 1.0)), _unsq(c_shift_mlp))

        h = ttnn.add(h, ttnn.multiply(_unsq(gate_mlp), _ff(nh2, ff_p)))
        e = ttnn.add(e, ttnn.multiply(_unsq(c_gate_mlp), _ff(ne2, ffc_p)))
        return h, e

    return forward


def hunyuan_video15_transformer_block(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_transformer_block requires build(device, torch_module) to bind the "
        "block weights; the bare callable has no parameters."
    )
