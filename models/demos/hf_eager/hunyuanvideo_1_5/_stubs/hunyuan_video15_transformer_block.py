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


def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis=0):
    """Extract all sub-weights of the dual-stream block; return a native forward.

    Parallelism (all optional, default = single-device):
      tp (head/tensor parallel, mesh axis `tp_axis`): Megatron column/row-parallel
        split of the 16 attention heads + FF, all-reduced on tp_axis. Activations
        replicated across tp_axis.
      sp (sequence/context parallel, mesh axis `sp_axis`, sp>1): the LATENT stream
        is sharded along its sequence across sp_axis; the joint attention uses the
        ring flash kernel (ttnn.transformer.ring_joint_scaled_dot_product_attention)
        so each sp rank does 1/sp of the O(seq^2) attention. The prompt stream stays
        replicated across sp_axis. This is the lever for high-frame latency (the
        attention dominates at 121f, seq~49k). Requires a 2D mesh; weights shard on
        tp_axis and replicate on sp_axis (ShardTensor2dMesh)."""
    import os

    import torch

    # bf16 fast path (HY_DIT_BF16=1): load weights bf16 + run the block's matmuls/
    # norms/SDPA in bf16 (HiFi2, fp32 accumulate). The block casts activations to
    # bf16 on entry and back to fp32 on exit, so it stays fp32-in/fp32-out (drop-in
    # for the fp32 glue/sub-stubs). ~2-4x faster matmuls + ~2x less weight/activation
    # DRAM. Default OFF = original fp32 behavior. The joint SDPA is bf16 either way.
    _bf16 = os.environ.get("HY_DIT_BF16", "0") == "1"
    wdt = ttnn.bfloat16 if _bf16 else ttnn.float32

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

    # Sequence/context parallel (sp>1): 2D mesh, latent seq sharded on sp_axis.
    seq_parallel = sp > 1 and ccl_manager is not None
    mesh_shape = tuple(device.shape) if seq_parallel else None

    def _mapper(shard_dim):
        """Weight mesh-mapper. 2D (seq-parallel): shard `shard_dim` on tp_axis,
        replicate on sp_axis. 1D (head-TP only): plain shard on tp mesh. None:
        single device. shard_dim=None -> fully replicated."""
        if seq_parallel:
            dims = [None, None]
            dims[tp_axis] = shard_dim  # None on sp_axis => replicate the sequence axis
            return ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=dims)
        if sharded and shard_dim is not None:
            return ttnn.ShardTensorToMesh(device, dim=shard_dim)
        if sharded:
            return ttnn.ReplicateTensorToMesh(device)
        return None

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2 if _bf16 else ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # SDPA-specific config: drop fp32 dest-accum in the attention softmax/exp core.
    # The fp32 dest-accum halves SFPU/packer throughput there (tt_dit's own SDPA uses
    # fp32_dest_acc_en=False); the attention tolerates bf16 accumulate. PCC-gated.
    sdpa_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2 if _bf16 else ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # Joint (dual-stream) flash attention: replaces the old explicit
    # matmul(q,kT)+softmax+matmul(.,v) sequence, which materializes the full
    # (seq x seq) score matrix in DRAM -- at real video resolutions (>~60
    # frames) that matrix alone exceeds one chip's DRAM (measured on QB2:
    # ~40GB at 121 frames vs a ~34GB chip). ttnn.transformer.
    # joint_scaled_dot_product_attention is a FlashAttention-2-style kernel
    # purpose-built for this exact "two Q/K/V streams, concatenated rear-wise"
    # shape and never materializes that matrix. Chunk sizes are a starting
    # point, not tuned; adjust if the op rejects them for a given seq length.
    _grid = device.compute_with_storage_grid_size()
    if seq_parallel:
        # Ring SDPA needs cores for the K/V all-gather: reserve the last core row
        # for CCL, run SDPA on the rest (matches tt_dit blocks/attention.py).
        sdpa_worker_grid = (_grid.x, _grid.y - 1)
        ccl_core_grid_offset = (0, _grid.y - 1)
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_worker_grid,
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
    else:
        ccl_core_grid_offset = None
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_grid.x, _grid.y),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

    def f32(t, mesh_mapper=None):
        if mesh_mapper is None and (sharded or seq_parallel):
            mesh_mapper = _mapper(None)  # fully replicated (2D-aware)
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=wdt,
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
        """Column-parallel: shard the OUTPUT dim across tp_axis (no CCL needed)."""
        mapper = _mapper(-1)
        w = f32(linear.weight.detach().t(), mesh_mapper=mapper)
        b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
        return w, b

    def lin_col_qkv(linears):
        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
        parallelism correct, interleave the output columns per tp-device group as
        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
        own local heads of q, k and v contiguously; the forward then slices the
        fused output at the LOCAL inner dim."""
        g = tp if sharded else 1
        hloc = heads_total // g

        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)

        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
        if all(m.bias is not None for m in linears):
            bcat = torch.cat(
                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
                dim=1,
            ).reshape(1, -1)
            b = f32(bcat, mesh_mapper=_mapper(-1))
        else:
            b = None
        return w, b

    def lin_row(linear):
        """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
        replicated and is added once, by the caller, after the all-reduce."""
        mapper = _mapper(0)
        w = f32(linear.weight.detach().t(), mesh_mapper=mapper)
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None  # replicated
        return w, b

    def ada_chunks(adazero):
        """AdaLayerNormZero.linear (C -> 6C): keep it as ONE fused matmul and slice
        the six modulation params from the output. On the launch-bound path this is
        1 matmul launch instead of 6 (the split M=32 modulation matmuls were ~24 of
        the dominant launches). Left replicated (not TP-sharded) -- see docstring."""
        L = adazero.linear
        C = int(L.out_features) // 6
        w = f32(L.weight.detach().t())  # (Cin, 6C)
        # Bake the modulation "+1" into the bias of the two SCALE params (order:
        # shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp -> idx 1 & 4), so
        # the matmul emits (1+scale) directly and the runtime add(scale,1.0) is gone
        # (correct for any batch; the bias is per-feature). Downstream modulation then
        # collapses to a single fused addcmul(shift, norm, scale).
        bias = L.bias.detach().clone() if L.bias is not None else torch.zeros(6 * C)
        bias = bias.reshape(6, C)
        bias[1] += 1.0
        bias[4] += 1.0
        b = f32(bias.reshape(1, -1))  # (1, 6C)
        eps = float(getattr(adazero.norm, "eps", 1e-6))
        return w, b, eps, C

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

    w_qkv, b_qkv = lin_col_qkv([attn.to_q, attn.to_k, attn.to_v])
    w_aqkv, b_aqkv = lin_col_qkv([attn.add_q_proj, attn.add_k_proj, attn.add_v_proj])
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

    def _to_wdt_device(t):
        """Bring an input onto device in the block's working dtype (wdt): fp32 in the
        default path, bf16 in the HY_DIT_BF16 fast path."""
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != wdt:
                t = ttnn.typecast(t, wdt)
            return t
        return ttnn.from_torch(t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)

    def _linear(x, w, b):
        # Fuse bias into the matmul epilogue (ttnn.linear) instead of a separate
        # ttnn.add: the standalone add is its own dispatch-bound op launch, and
        # the block issues one per QKV/FF projection, so folding them removes
        # that many launches from the profiled 2-layer forward.
        if b is not None:
            return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config)
        return ttnn.matmul(x, w, compute_kernel_config=compute_config)

    def _all_reduce(x, mesh_axis=None):
        """Megatron all-reduce for a row-parallel output: reduce_scatter + all_gather,
        the idiom `models/demos/z_image_turbo` uses for its own flat-TP DiT. Reduces
        across the TENSOR-parallel axis (tp_axis); the sequence-parallel axis is left
        alone (its shards stay sharded)."""
        if mesh_axis is None:
            mesh_axis = tp_axis
        if not sharded:
            return x
        # reduce_scatter's persistent-buffer path hardcodes bf16 ping-pong buffers
        # (models/tt_dit/parallel/manager.py::get_rs_ping_pong_buffer). In the fp32
        # path use the (dtype-agnostic) barrier-semaphore path; in the bf16 fast
        # path (_bf16) the activations ARE bf16 and match that buffer, so use it to
        # skip the barrier-semaphore overhead on the 4 all-reduces/block. all_gather
        # always uses its own dtype-matched persistent buffer.
        B, L, Cx = (int(d) for d in x.shape)
        x4 = ttnn.reshape(x, (1, B, L, Cx))
        x4 = ccl_manager.reduce_scatter(x4, dim=3, mesh_axis=mesh_axis, use_persistent_buffer=_bf16)
        x4 = ccl_manager.all_gather(x4, dim=3, mesh_axis=mesh_axis, use_hyperparams=True, use_persistent_buffer=True)
        return ttnn.reshape(x4, (B, L, Cx))

    def _row_linear(x, w, b):
        # Single-device (tp=1): the all-reduce is a no-op, so fold the bias into the
        # matmul epilogue (ttnn.linear) instead of a standalone dispatch-bound add.
        # Removes one add launch per to_out / to_add_out / FF-down call (8/forward).
        if not sharded:
            if b is not None:
                return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config)
            return ttnn.matmul(x, w, compute_kernel_config=compute_config)
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        y = _all_reduce(y)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def _rms(x, w):
        # x: (B, L, H, D); normalize over D. Fused kernel (verified PCC~1.0 vs
        # the prior manual mean/multiply/rsqrt/multiply sequence it replaces).
        return ttnn.rms_norm(x, epsilon=rms_eps, weight=w, compute_kernel_config=compute_config)

    def _adazero(x, s, w, b, eps):
        # `s` is the PRE-computed silu(temb): temb is identical for the hidden and
        # context streams (both _adazero calls share it), so silu is hoisted to the
        # caller and computed ONCE per block instead of twice (one fewer dispatch-
        # bound launch on the launch-bound path).
        # ONE fused (C -> 6C) matmul (bias in epilogue), then slice the 6 params.
        if b is not None:
            p = ttnn.linear(s, w, bias=b, compute_kernel_config=compute_config)
        else:
            p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
        Bp = int(p.shape[0])
        # Reshape the fused output to 3D ONCE, so every sliced param is already
        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
        # single op as the 2D slice was.
        p = ttnn.reshape(p, (Bp, 1, 6 * C))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
        )
        nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
        # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
        return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _rope_bcast(cos, sin, Sx, Dx, dtype):
        # Broadcast-reshape (and dtype-match) the (S,D) freqs to (1,S,1,D) ONCE per
        # block: rope runs for BOTH q and k with identical cos/sin, so hoisting this
        # out of _apply_rope removes 2 reshapes (+ up to 2 typecasts) per block on the
        # launch-bound path.
        cos_b = ttnn.reshape(cos, (1, Sx, 1, Dx))
        sin_b = ttnn.reshape(sin, (1, Sx, 1, Dx))
        if cos_b.dtype != dtype:  # bf16 fast path: match the fp32 freqs to activations
            cos_b = ttnn.typecast(cos_b, dtype)
            sin_b = ttnn.typecast(sin_b, dtype)
        return cos_b, sin_b

    def _apply_rope(x4, cos_b, sin_b):
        # x4: (B, S, H, D); cos_b/sin_b: pre-broadcast (1,S,1,D). out = x*cos + rot(x)*sin.
        Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
        x2 = ttnn.reshape(x4, (Bx * Sx * Hx, Dx))
        rot = ttnn.matmul(x2, rot_M, compute_kernel_config=compute_config)
        rot4 = ttnn.reshape(rot, (Bx, Sx, Hx, Dx))
        # x*cos + rot*sin fused: mul + addcmul (2 ops) instead of mul + mul + add (3).
        return ttnn.addcmul(ttnn.multiply(x4, cos_b), rot4, sin_b)

    def _joint_attention(nh, ne, freqs_cis=None, attn_bias=None, logical_n=None):
        """Joint (dual-stream) attention via the fused flash-attention-style
        ttnn.transformer.joint_scaled_dot_product_attention kernel instead of
        an explicit matmul(q,kT)+softmax+matmul(.,v) sequence. The naive form
        materializes the full (seq x seq) score matrix in DRAM -- at real
        video resolutions (>~60 frames) that alone exceeds one chip's DRAM
        (measured on QB2: ~40GB at 121 frames vs a ~34GB chip). The fused
        kernel never materializes it.

        `nh` (latent/"original") and `ne` (text/"joint") are attended jointly
        by internally concatenating them rear-wise (joint_strategy="rear",
        matching this model's [latent, text] token order) then splitting the
        output back into the two streams -- the same math the old code did
        by hand via ttnn.concat/ttnn.slice, just fused.

        `attn_bias` is IGNORED: neither joint_scaled_dot_product_attention nor
        its ring/sequence-parallel sibling accepts any mask/bias parameter
        (checked against the op's C++ struct definitions, not just the
        Python docstring). Callers must not rely on per-key masking here --
        see tt/pipeline.py's `_reorder_concat` (excludes t2v's always-invalid
        image tokens outright rather than masking them) and the note on
        dropped per-row text-padding masking there."""
        B = int(nh.shape[0])
        Limg = int(nh.shape[1])
        Ltxt = int(ne.shape[1])

        def heads_split(t):
            t = ttnn.reshape(t, (B, -1, heads, dim_head))
            return t

        # Fused QKV: one matmul launch, slice the three streams at the LOCAL inner
        # dim (each device holds [q|k|v] for its local heads).
        qkv = _linear(nh, w_qkv, b_qkv)  # (B, Limg, 3*inner) local
        q = _rms(heads_split(ttnn.slice(qkv, (0, 0, 0), (B, Limg, inner))), nq_w)  # (B, Limg, H, D)
        k = _rms(heads_split(ttnn.slice(qkv, (0, 0, inner), (B, Limg, 2 * inner))), nk_w)
        v = heads_split(ttnn.slice(qkv, (0, 0, 2 * inner), (B, Limg, 3 * inner)))

        # RoPE on the latent stream only (encoder q/k are added un-rotated), matching
        # HunyuanVideo15AttnProcessor2_0 (apply_rotary_emb after norm_q/norm_k).
        if freqs_cis is not None:
            _cos, _sin = freqs_cis
            cos_b, sin_b = _rope_bcast(_cos, _sin, int(q.shape[1]), int(q.shape[3]), q.dtype)
            q = _apply_rope(q, cos_b, sin_b)
            k = _apply_rope(k, cos_b, sin_b)

        eqkv = _linear(ne, w_aqkv, b_aqkv)  # (B, Ltxt, 3*inner) local
        eq = _rms(heads_split(ttnn.slice(eqkv, (0, 0, 0), (B, Ltxt, inner))), naq_w)  # (B, Ltxt, H, D)
        ek = _rms(heads_split(ttnn.slice(eqkv, (0, 0, inner), (B, Ltxt, 2 * inner))), nak_w)
        ev = heads_split(ttnn.slice(eqkv, (0, 0, 2 * inner), (B, Ltxt, 3 * inner)))

        q = ttnn.permute(q, (0, 2, 1, 3))  # (B, H, Limg, D)
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        eq = ttnn.permute(eq, (0, 2, 1, 3))  # (B, H, Ltxt, D)
        ek = ttnn.permute(ek, (0, 2, 1, 3))
        ev = ttnn.permute(ev, (0, 2, 1, 3))

        # `scale` is NOT passed: the op binds it as a `.noconvert()` C++ float, so
        # a Python double (every Python float) is rejected with a TypeError -- which
        # is why every other caller in the repo (tt_dit SD3.5/Mochi/etc.) omits it.
        # The kernel's internal default is `1/sqrt(head_dim)` (joint_sdpa_device_
        # operation.cpp), identical to this model's `scale` (== dim_head**-0.5). The
        # assert makes a future checkpoint carrying a non-standard `attn.scale` fail
        # loudly here instead of silently getting the default.
        assert abs(scale - dim_head**-0.5) < 1e-9, (
            f"joint SDPA can only use its built-in 1/sqrt(head_dim) scale "
            f"({dim_head**-0.5}), but this module's scale is {scale}"
        )
        # The joint SDPA kernel only accepts bf16/bf8 q/k/v (asserted in
        # joint_sdpa_device_operation.cpp); the old explicit matmul path took
        # fp32. Cast the six streams to bf16 for the kernel and cast the two
        # outputs back to the original dtype. In the real 54-layer run
        # everything is already bf16 (coerce_bf16), so this is a no-op there;
        # in the fp32 PCC self-test it limits precision loss to bf16 input
        # rounding (the kernel still accumulates in fp32 via compute_config's
        # fp32_dest_acc_en).
        attn_dtype = q.dtype
        if attn_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            q, k, v = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
            eq, ek, ev = (ttnn.typecast(t, ttnn.bfloat16) for t in (eq, ek, ev))
        if seq_parallel:
            # Sequence-parallel: q/k/v are seq-sharded on sp_axis ([B, H_local,
            # seq/sp, D]); the ring kernel all-gathers K/V across sp_axis so each
            # rank's local Q attends the full sequence, doing 1/sp of the work.
            # `logical_n` is the UNPADDED full spatial seq (the pipeline pads to a
            # multiple of sp before sharding). The prompt (eq/ek/ev) is replicated
            # across sp_axis; its output comes back replicated.
            hid_out, enc_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                eq,
                ek,
                ev,
                persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(k.shape, 2, sp_axis),
                persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(v.shape, 2, sp_axis),
                joint_strategy="rear",
                logical_n=logical_n,
                program_config=sdpa_program_config,
                compute_kernel_config=sdpa_compute_config,
                dim=2,
                multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
                num_links=ccl_manager.num_links,
                cluster_axis=sp_axis,
                mesh_device=device,
                topology=ccl_manager.topology,
                subdevice_id=ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=ccl_core_grid_offset,
            )
        else:
            hid_out, enc_out = ttnn.transformer.joint_scaled_dot_product_attention(
                q,
                k,
                v,
                eq,
                ek,
                ev,
                joint_strategy="rear",
                program_config=sdpa_program_config,
                compute_kernel_config=sdpa_compute_config,
            )
        if attn_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            hid_out = ttnn.typecast(hid_out, attn_dtype)
            enc_out = ttnn.typecast(enc_out, attn_dtype)
        hid_out = ttnn.permute(hid_out, (0, 2, 1, 3))  # (B, Limg, H, D)
        enc_out = ttnn.permute(enc_out, (0, 2, 1, 3))  # (B, Ltxt, H, D)
        hid_out = ttnn.reshape(hid_out, (B, Limg, inner))
        enc_out = ttnn.reshape(enc_out, (B, Ltxt, inner))

        hid = _row_linear(hid_out, wo, bo)
        enc = _row_linear(enc_out, ao_w, ao_b)
        return hid, enc

    def _ff(x, parts):
        w1, b1, act, w2, b2 = parts
        y = _linear(x, w1, b1)
        y = act(y)
        y = _row_linear(y, w2, b2)
        return y

    def _wln(x, eps):
        # grid knob: LN input is a single ragged tile-row (M_tiles=1). Build a
        # tile-PADDED width-sharded L1 spec by hand (create_sharded_memory_config
        # derives height from the ragged logical L and fails tile-alignment) so the
        # width dim spreads over a row of gx cores instead of the default tiny grid.
        B, L, Cx = (int(d) for d in x.shape)
        # TT pads each logical dimension independently. For example, logical
        # [2, 14, 2048] has physical [2, 32, 2048], so rounding B*L to one tile
        # produces shard height 32 for storage whose flattened height is 64.
        # Size the shard/program from physical storage while preserving the
        # tensor's logical shape metadata.
        padded_m = 1
        padded_shape = list(x.padded_shape)
        for d in padded_shape[:-1]:
            padded_m *= int(d)
        Mt = padded_m // 32
        Nt = int(padded_shape[-1]) // 32
        gx = 8
        while gx > 1 and Nt % gx != 0:
            gx -= 1
        # L1-fit guard: width-sharding the whole [M, C/gx] block into each core's L1
        # OOMs at real video frame counts (M~16k tokens @ 121f -> ~16MB/core >> L1).
        # It's only a win at the tiny profiled M; fall back to the stock interleaved
        # LN when the per-core shard won't fit L1.
        if (Mt * 32) * ((Nt // gx) * 32) * 2 > 700_000:
            return ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)
        shard_shape = [padded_m, (Nt // gx) * 32]
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

    def forward(
        hidden_states, encoder_hidden_states=None, temb=None, attention_mask=None, freqs_cis=None, *args, **kwargs
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if temb is None:
            temb = kwargs.get("temb")
        if encoder_hidden_states is None or temb is None:
            raise TypeError("hunyuan_video15_transformer_block needs encoder_hidden_states and temb")

        h = _to_wdt_device(hidden_states)
        e = _to_wdt_device(encoder_hidden_states)
        t = _to_wdt_device(temb)

        attn_bias = kwargs.get("attn_bias")

        s_t = ttnn.silu(t)  # shared by both streams' AdaLN modulation (compute once)
        nh, gate_msa, shift_mlp, scale_mlp, gate_mlp = _adazero(h, s_t, ada1_w, ada1_b, ada1_eps)
        ne, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _adazero(e, s_t, adac_w, adac_b, adac_eps)

        attn_out, ctx_out = _joint_attention(
            nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias, logical_n=kwargs.get("logical_n")
        )

        # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
        # Modulation params come out of _adazero already (B,1,C) — no _unsq reshape.
        h = ttnn.addcmul(h, attn_out, gate_msa)
        e = ttnn.addcmul(e, ctx_out, c_gate_msa)

        # norm2 modulation fused to addcmul(shift, norm, scale); scale_mlp already
        # carries (1+scale) (+1 baked into the AdaLN bias in ada_chunks).
        nh2 = _wln(h, norm2_eps)
        nh2 = ttnn.addcmul(shift_mlp, nh2, scale_mlp)
        ne2 = _wln(e, norm2c_eps)
        ne2 = ttnn.addcmul(c_shift_mlp, ne2, c_scale_mlp)

        h = ttnn.addcmul(h, gate_mlp, _ff(nh2, ff_p))
        e = ttnn.addcmul(e, c_gate_mlp, _ff(ne2, ffc_p))
        if wdt != ttnn.float32:  # keep the block fp32-in/fp32-out for the fp32 glue
            h = ttnn.typecast(h, ttnn.float32)
            e = ttnn.typecast(e, ttnn.float32)
        return h, e

    return forward


def hunyuan_video15_transformer_block(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_transformer_block requires build(device, torch_module) to bind the "
        "block weights; the bare callable has no parameters."
    )
