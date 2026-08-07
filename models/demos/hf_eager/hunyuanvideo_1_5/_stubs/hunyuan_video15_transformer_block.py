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

from collections import namedtuple

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

# Row-parallel bias in its two prepared forms. ``full`` is the replicated
# ``(1, C)`` bias added to the all-gathered activation; ``local`` is the same
# bias fractured over the TP axis, added to the reduce-scatter output instead.
_RowBias = namedtuple("_RowBias", ["full", "local"])


def _enabled(value):
    """Parse an opt-in/out environment value without silently accepting typos."""
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _layer_norm_shard_fits(padded_m, local_width, bytes_per_element):
    """Conservatively account for LayerNorm's static input/output/reduction CBs."""
    shard_bytes = padded_m * local_width * bytes_per_element
    return 6 * shard_bytes <= 1_300_000


def _select_sdpa_chunks(*, sequence_parallel, sp, tp, blackhole, environ):
    """Return validated SDPA chunks while retaining the measured Hunyuan default.

    ``wan_bh_sp8tp4`` exposes Wan2.2's mature Blackhole SP8xTP4 setting for an
    explicit A/B. It is deliberately not the default: Hunyuan has a different
    joint-token length and reserves a different CCL core strip.
    """
    chunks = (128, 512) if sequence_parallel else (128, 128)
    preset = environ.get("HY_DIT_SDPA_PRESET", "hunyuan").strip().lower()
    if preset == "wan_bh_sp8tp4":
        if not (sequence_parallel and blackhole and sp == 8 and tp == 4):
            raise ValueError("wan_bh_sp8tp4 requires Blackhole sequence-parallel SP=8, TP=4")
        chunks = (288, 512)
    elif preset != "hunyuan":
        raise ValueError(f"unknown HY_DIT_SDPA_PRESET={preset!r}")

    q_chunk = int(environ.get("HY_DIT_SDPA_Q_CHUNK", chunks[0]))
    k_chunk = int(environ.get("HY_DIT_SDPA_K_CHUNK", chunks[1]))
    if q_chunk <= 0 or k_chunk <= 0 or q_chunk % 32 or k_chunk % 32:
        raise ValueError(f"SDPA chunks must be positive tile multiples, got q={q_chunk}, k={k_chunk}")
    return q_chunk, k_chunk


def _select_collective_overlap(*, sharded, tp, blackhole, bf16, topology, environ):
    """Validate the Hunyuan latent-stream fused MM+RS opt-in.

    The fused kernel's contract is tensor-parallel only; SP does not alter its
    local row-linear shapes.  Production targets SP8xTP4, while the TP4-only
    component test exercises the same local contract on four devices.

    ``ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`` asserts
    ``topology == Ring`` in its device operation, so the fused kernel cannot
    run on the Galaxy's physical ``FABRIC_1D`` Linear path.  Reject that here,
    at build time, rather than letting a device-side TT_FATAL abort a
    generation part-way through the 54-block stack.
    """
    enabled = _enabled(environ.get("HY_DIT_MMRS_OVERLAP", "0"))
    if not enabled:
        return False
    if not sharded or tp != 4 or not blackhole or not bf16:
        raise ValueError("HY_DIT_MMRS_OVERLAP requires Blackhole TP=4 with HY_DIT_BF16=1")
    if topology != ttnn.Topology.Ring:
        raise ValueError(
            "HY_DIT_MMRS_OVERLAP needs a Ring CCL topology: "
            "minimal_matmul_strided_reduce_scatter_async only supports Ring, and the "
            f"Hunyuan Galaxy path builds its CCLManager with {topology!r}. "
            "Leave HY_DIT_MMRS_OVERLAP=0 on FABRIC_1D."
        )
    return True


def _row_bias_shard_width(width, tp):
    """Per-device width of a TP-fractured row-parallel bias.

    The fractured bias is added to a reduce-scatter output, so its width has to
    divide evenly across the TP axis and stay tile aligned.
    """
    if tp <= 1:
        raise ValueError(f"a fractured row bias needs tp>1, got tp={tp}")
    if width % tp or (width // tp) % 32:
        raise ValueError(
            f"HY_DIT_RS_DOMAIN_BIAS needs a tile-aligned per-device bias width, got width={width}, tp={tp}"
        )
    return width // tp


def _select_rs_domain_bias(*, sharded, tp, environ):
    """Validate the opt-in scattered-domain row-parallel bias.

    A row-parallel projection currently evaluates ``all_gather(reduce_scatter(
    partials)) + bias`` on the full replicated width.  With this enabled the
    bias is added to the reduce-scatter output instead, so the all-gather
    replicates an already-biased shard.  Every output element still evaluates
    ``reduce(partials) + bias`` from the same operands in the same dtype, and
    an all-gather is a pure copy, so the result is bit-identical -- the bias add
    just runs on a tensor ``tp`` times narrower.
    """
    enabled = _enabled(environ.get("HY_DIT_RS_DOMAIN_BIAS", "0"))
    if not enabled:
        return False
    if not sharded or tp <= 1:
        raise ValueError("HY_DIT_RS_DOMAIN_BIAS requires a tensor-parallel mesh (tp>1)")
    return True


def _run_dual_stream_projection_schedule(overlap, hidden_start, context_complete, hidden_finish):
    """Run a dependency-safe dual-stream row-projection schedule.

    In overlap mode ``hidden_start`` emits the reduce-scattered output of fused
    MMRS and the independent context projection is enqueued before the latent
    all-gather/bias finish.

    This ordering buys no concurrency on its own.  Every model program is
    enqueued on command queue 0 (queue 1 carries only host-to-device resident
    copies), and programs on one queue run to completion in order, so a
    standalone collective never overlaps a later standalone matmul.  The only
    overlap here is inside the fused device program, which streams completed
    matmul blocks into its reduce-scatter.  The schedule is kept because it
    expresses the dependency structure and keeps the legacy path a single
    call, not because reordering hides a collective.
    """
    hidden = hidden_start()
    context = context_complete()
    if overlap:
        hidden = hidden_finish(hidden)
    return hidden, context


def build(
    device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis=0, weight_cache_prefix=None
):
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

    from models.common.utility_functions import is_blackhole
    from models.tt_dit.utils.matmul import get_fused_mmrs_config, get_matmul_config, get_matmul_core_grid

    # bf16 fast path (HY_DIT_BF16=1): load weights bf16 + run the block's matmuls/
    # norms/SDPA in bf16 (HiFi2, fp32 accumulate). The block casts activations to
    # bf16 on entry and back to fp32 on exit, so it stays fp32-in/fp32-out (drop-in
    # for the fp32 glue/sub-stubs). ~2-4x faster matmuls + ~2x less weight/activation
    # DRAM. Default OFF = original fp32 behavior. The joint SDPA is bf16 either way.
    _bf16 = os.environ.get("HY_DIT_BF16", "0") == "1"
    # Reuse tt_dit's fused projection+split primitive (the same path used by
    # WanAttention/ColParallelLinear). It emits q/k/v directly and removes the
    # six post-matmul slices per dual-stream block. Keep the old linear+slices
    # path available for unsupported runtime/shape combinations.
    _qkv_split = _enabled(os.environ.get("HY_DIT_QKV_SPLIT", "0"))
    # Fuse the post-attention head merge. The legacy permute+reshape pair routes
    # through a (B, L, H, D) intermediate; with TP=4 the LOCAL head count is 4,
    # which lands in the second-to-last position and tile-pads to 32, so every
    # element moves with 8x its necessary bytes. A 13f device profile measured
    # the head-layout reshapes + permutes at 27.6% of total device kernel time,
    # with ReshapeViewDeviceOperation the single most expensive op in the model.
    _fused_heads = _enabled(os.environ.get("HY_DIT_FUSED_HEADS", "1"))
    # Same idea on the attention INPUT side, where the profile put ~18% of device
    # time (vs ~9% for the output merge). nlp_create_qkv_heads emits (B, H, S, D)
    # from the fused projection, so heads_split's padded intermediate, the six
    # permutes, and the RoPE collapse through the padded axis all disappear.
    _fused_qkv_heads = _enabled(os.environ.get("HY_DIT_FUSED_QKV_HEADS", "1"))
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
    _mmrs_overlap = _select_collective_overlap(
        sharded=sharded,
        tp=tp,
        blackhole=is_blackhole(),
        bf16=_bf16,
        topology=ccl_manager.topology if ccl_manager is not None else None,
        environ=os.environ,
    )
    _rs_domain_bias = _select_rs_domain_bias(sharded=sharded, tp=tp, environ=os.environ)

    # Prepared-weight cache directory. Everything that changes a weight's dtype,
    # shape or shard placement goes in the tag, so a cache entry can only be
    # reused by an identical configuration. Sequential per-block indices are then
    # a safe key even though the number of f32() calls varies with the flags.
    _wcache_dir = None
    _wcount = [0]
    # Every device weight this block allocates, so the caller can free them once
    # the DiT is dead. In a one-shot generation the last denoise step is the last
    # use of these: VAE decode follows and never touches them, and the resident
    # DiT otherwise holds ~99% of DRAM, which is what blocks the H/W-sharded VAE.
    _weights = []
    if _enabled(os.environ.get("HY_DIT_WEIGHT_CACHE", "0")):
        _root = (
            os.environ.get("HY_DIT_WEIGHT_CACHE_DIR")
            or os.environ.get("TT_DIT_CACHE_DIR")
            or "~/.cache/tt-dit"
        )
        _mesh_tag = "x".join(str(d) for d in tuple(device.shape)) if seq_parallel or sharded else "1"
        _wcache_dir = os.path.join(
            os.path.expanduser(_root),
            "hunyuanvideo15_dit",
            f"mesh{_mesh_tag}_tp{tp}_sp{sp}_ax{tp_axis}{sp_axis}"
            f"_{'bf16' if _bf16 else 'fp32'}_rsb{int(_rs_domain_bias)}",
        )
        os.makedirs(_wcache_dir, exist_ok=True)

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
    q_chunk_size, k_chunk_size = _select_sdpa_chunks(
        sequence_parallel=seq_parallel,
        sp=sp,
        tp=tp,
        blackhole=is_blackhole(),
        environ=os.environ,
    )
    if seq_parallel:
        # Ring SDPA needs cores for the K/V all-gather: reserve the last core row
        # for CCL, run SDPA on the rest (matches tt_dit blocks/attention.py).
        sdpa_worker_grid = (_grid.x, _grid.y - 1)
        ccl_core_grid_offset = (0, _grid.y - 1)
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_worker_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
    else:
        ccl_core_grid_offset = None
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_grid.x, _grid.y),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

    def f32(t, mesh_mapper=None):
        if mesh_mapper is None and (sharded or seq_parallel):
            mesh_mapper = _mapper(None)  # fully replicated (2D-aware)

        # Prepared-weight cache. `ttnn.from_torch` with a mesh mapper costs far
        # more than reloading an already-prepared tensor -- measured 5-10x on
        # the real 8x4 mesh (289 ms -> 34 ms for 143 MB of block weights), and
        # the cost is host-side dtype conversion plus tilisation, not transfer.
        # `DumpTensorMode.LOCAL` is what makes this correct AND compact: it
        # persists each device's own shard and restores the same placement, so
        # the round trip is bit-exact and the cache is 1.00x the logical size
        # (a plain dump of `from_device(t)` silently serialises ONE shard and
        # reloads it replicated -- wrong weights on 31 of 32 devices).
        # Weights are created in deterministic order per block, so a sequential
        # index is a sufficient key; everything that changes weight layout is in
        # the directory tag instead.
        path = None
        if _wcache_dir is not None and weight_cache_prefix is not None:
            path = os.path.join(_wcache_dir, f"{weight_cache_prefix}.w{_wcount[0]}.tensorbin")
            _wcount[0] += 1
            if os.path.exists(path):
                cached = ttnn.load_tensor(path, device=device)
                # Register cache hits too: on a warm cache EVERY weight takes
                # this path, so skipping it leaves free_weights() with nothing
                # to release.
                _weights.append(cached)
                return cached

        out = ttnn.from_torch(
            t.contiguous().float(),
            dtype=wdt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
        if path is not None:
            ttnn.dump_tensor(path, out, mode=ttnn.DumpTensorMode.LOCAL)
        _weights.append(out)
        return out

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
        """Row-parallel: shard the INPUT (contraction) dim on tp_axis; the bias
        stays out of the reduction and is added once by the caller.

        Exactly one of the two prepared forms is populated. ``full`` is the
        replicated ``(1, C)`` bias added after the all-gather. ``local`` is the
        same bias fractured over tp_axis, for the scattered-domain add between
        reduce-scatter and all-gather; TP chunk ``d`` lands on device ``d`` for
        both the mesh mapper and the reduce-scatter, which is the convention
        the existing all-gathered column order already depends on."""
        mapper = _mapper(0)
        w = f32(linear.weight.detach().t(), mesh_mapper=mapper)
        if linear.bias is None:
            return w, _RowBias(None, None)
        bias2d = linear.bias.detach().reshape(1, -1)
        if _rs_domain_bias:
            _row_bias_shard_width(int(bias2d.shape[-1]), tp)
            # 4D so it broadcasts against the (1, B, L, C/tp) reduce-scatter output.
            return w, _RowBias(None, f32(bias2d.reshape(1, 1, 1, -1), mesh_mapper=_mapper(-1)))
        return w, _RowBias(f32(bias2d), None)

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

    def _merge_heads(t, B, L):
        """Joint-SDPA output (B, H, L, D) -> (B, L, H*D).

        ``nlp_concat_heads`` emits (B, 1, L, H*D) directly, skipping the
        tile-padded (B, L, H, D) intermediate the permute would materialise.
        The trailing reshape only drops the unit axis -- both sides share the
        same last two dims, so it is a view, not a re-tile."""
        if not _fused_heads:
            t = ttnn.permute(t, (0, 2, 1, 3))  # (B, L, H, D)
            return ttnn.reshape(t, (B, L, inner))
        t = ttnn.experimental.nlp_concat_heads(t)  # (B, 1, L, H*D)
        return ttnn.reshape(t, (B, L, inner))

    def _linear_split3(x, w, b):
        """Fused matmul + q/k/v split, matching tt_dit ColParallelLinear."""
        if not _qkv_split:
            y = _linear(x, w, b)
            Bx, Lx = int(y.shape[0]), int(y.shape[1])
            return tuple(ttnn.slice(y, (0, 0, i * inner), (Bx, Lx, (i + 1) * inner)) for i in range(3))
        M, K, N = x.padded_shape[-2], x.padded_shape[-1], w.padded_shape[-1]
        config = get_matmul_config(M, K, N, get_matmul_core_grid(device))
        return tuple(
            ttnn.experimental.minimal_matmul_split(
                input_tensor=x,
                weight_tensor=w,
                chunks=3,
                dim=-1,
                bias_tensor=b,
                compute_kernel_config=compute_config,
                config=config,
            )
        )

    def _all_reduce(x, mesh_axis=None, bias_local=None):
        """Megatron all-reduce for a row-parallel output: reduce_scatter + all_gather,
        the idiom `models/demos/z_image_turbo` uses for its own flat-TP DiT. Reduces
        across the TENSOR-parallel axis (tp_axis); the sequence-parallel axis is left
        alone (its shards stay sharded).

        ``bias_local`` is a TP-fractured bias applied to the reduce-scatter output.
        The all-gather then replicates an already-biased shard, which is
        bit-identical to biasing the gathered tensor but touches 1/tp of the bytes."""
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
        if bias_local is not None:
            x4 = ttnn.add(x4, bias_local)
        x4 = ccl_manager.all_gather(x4, dim=3, mesh_axis=mesh_axis, use_hyperparams=True, use_persistent_buffer=True)
        return ttnn.reshape(x4, (B, L, Cx))

    def _row_linear(x, w, b):
        # Single-device (tp=1): the all-reduce is a no-op, so fold the bias into the
        # matmul epilogue (ttnn.linear) instead of a standalone dispatch-bound add.
        # Removes one add launch per to_out / to_add_out / FF-down call (8/forward).
        if not sharded:
            if b.full is not None:
                return ttnn.linear(x, w, bias=b.full, compute_kernel_config=compute_config)
            return ttnn.matmul(x, w, compute_kernel_config=compute_config)
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        if b.local is not None:
            return _all_reduce(y, bias_local=b.local)
        y = _all_reduce(y)
        if b.full is not None:
            y = ttnn.add(y, b.full)
        return y

    def _row_linear_mmrs_start(x, w):
        """Fuse latent row-parallel matmul with its TP reduce-scatter.

        Hunyuan keeps activations replicated across TP between projections, so
        the trailing all-gather remains separate.  Bias is intentionally not
        passed here: adding the replicated bias on every TP rank before the
        reduction would multiply it by TP.
        """
        M, K, N = x.padded_shape[-2], x.padded_shape[-1], w.padded_shape[-1]
        full_grid = device.compute_with_storage_grid_size()
        x4 = ttnn.unsqueeze(x, 0)
        _mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tensor=x4,
            weight_tensor=w,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore_fused(tp_axis),
            **get_fused_mmrs_config(M, K, N, full_grid, ccl_manager.num_links),
            bias=None,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl_manager.topology,
            cluster_axis=tp_axis,
            compute_kernel_config=compute_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(tp_axis),
            dtype=wdt,
        )
        return ttnn.squeeze(rs_out, 0)

    def _row_linear_mmrs_finish(rs_out, b):
        """Restore the replicated Hunyuan activation contract after fused MMRS."""
        B, L, C_local = (int(d) for d in rs_out.shape)
        rs4 = ttnn.reshape(rs_out, (1, B, L, C_local))
        if b.local is not None:
            rs4 = ttnn.add(rs4, b.local)
        gathered = ccl_manager.all_gather(
            rs4,
            dim=3,
            mesh_axis=tp_axis,
            use_hyperparams=True,
            use_persistent_buffer=True,
        )
        y = ttnn.reshape(gathered, (B, L, C_local * tp))
        if b.local is None and b.full is not None:
            y = ttnn.add(y, b.full)
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

    def _qkv_heads(x, w, b):
        """(B, S, C) -> q, k, v each (B, H, S, D), heads-major.

        ``nlp_create_qkv_heads`` consumes the fused [q|k|v] projection -- which is
        exactly how each Hunyuan TP shard already stores its weight -- and emits
        the three head-major streams in one op, replacing three slices, three
        reshapes and three permutes. ``transpose_k_heads=False`` keeps K as
        (B, H, S, D) like Q and V; the ring joint SDPA wants all three that way."""
        y = _linear(x, w, b)  # (B, S, 3 * inner)
        Bx, Sx = int(y.shape[0]), int(y.shape[1])
        y = ttnn.reshape(y, (Bx, 1, Sx, 3 * inner))
        return ttnn.experimental.nlp_create_qkv_heads(y, num_heads=heads, transpose_k_heads=False)

    def _rope_bcast_hm(cos, sin, Sx, Dx, dtype):
        # Heads-major twin of _rope_bcast: broadcast against (B, H, S, D), so the
        # unit axis moves from position 2 to position 1.
        cos_b = ttnn.reshape(cos, (1, 1, Sx, Dx))
        sin_b = ttnn.reshape(sin, (1, 1, Sx, Dx))
        if cos_b.dtype != dtype:
            cos_b = ttnn.typecast(cos_b, dtype)
            sin_b = ttnn.typecast(sin_b, dtype)
        return cos_b, sin_b

    def _apply_rope_hm(x4, cos_b, sin_b):
        # x4: (B, H, S, D). Same math as _apply_rope, but the collapse to 2D now
        # reads a tensor whose last two dims are (S, D) -- both tile-aligned --
        # instead of the (H=4, D) axis that pads 4 -> 32.
        Bx, Hx, Sx, Dx = (int(d) for d in x4.shape)
        x2 = ttnn.reshape(x4, (Bx * Hx * Sx, Dx))
        rot = ttnn.matmul(x2, rot_M, compute_kernel_config=compute_config)
        rot4 = ttnn.reshape(rot, (Bx, Hx, Sx, Dx))
        return ttnn.addcmul(ttnn.multiply(x4, cos_b), rot4, sin_b)

    def _apply_rope(x4, cos_b, sin_b):
        # x4: (B, S, H, D); cos_b/sin_b: pre-broadcast (1,S,1,D). out = x*cos + rot(x)*sin.
        Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
        x2 = ttnn.reshape(x4, (Bx * Sx * Hx, Dx))
        rot = ttnn.matmul(x2, rot_M, compute_kernel_config=compute_config)
        rot4 = ttnn.reshape(rot, (Bx, Sx, Hx, Dx))
        # x*cos + rot*sin fused: mul + addcmul (2 ops) instead of mul + mul + add (3).
        return ttnn.addcmul(ttnn.multiply(x4, cos_b), rot4, sin_b)

    def _joint_attention(nh, ne, freqs_cis=None, attn_bias=None, logical_n=None, joint_valid_lengths=None):
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

        The sequence-parallel ring kernel accepts ``joint_valid_lengths`` as a
        per-batch valid prefix. The pipeline packs image/byT5/Qwen valid tokens
        into that prefix, so every padded joint key is excluded from softmax
        without changing the fixed physical shape used by CFG traces."""
        B = int(nh.shape[0])
        Limg = int(nh.shape[1])
        Ltxt = int(ne.shape[1])

        def heads_split(t):
            t = ttnn.reshape(t, (B, -1, heads, dim_head))
            return t

        if _fused_qkv_heads:
            # Heads-major throughout: nlp_create_qkv_heads emits (B, H, S, D)
            # straight from the fused projection, so the tile-padded
            # (B, S, H=4, D) intermediate is never built and the six permutes
            # below are unnecessary. RMS-norm is over D either way. RoPE runs
            # in the same layout (see _apply_rope_hm).
            q, k, v = _qkv_heads(nh, w_qkv, b_qkv)
            q = _rms(q, nq_w)  # (B, H, Limg, D)
            k = _rms(k, nk_w)
            if freqs_cis is not None:
                _cos, _sin = freqs_cis
                cos_b, sin_b = _rope_bcast_hm(_cos, _sin, int(q.shape[2]), int(q.shape[3]), q.dtype)
                q = _apply_rope_hm(q, cos_b, sin_b)
                k = _apply_rope_hm(k, cos_b, sin_b)
            eq, ek, ev = _qkv_heads(ne, w_aqkv, b_aqkv)
            eq = _rms(eq, naq_w)  # (B, H, Ltxt, D)
            ek = _rms(ek, nak_w)
        else:
            # Fused QKV projection+split: one matmul emits the three LOCAL-head
            # streams directly. HY_DIT_QKV_SPLIT=0 retains linear + three slices.
            q, k, v = _linear_split3(nh, w_qkv, b_qkv)
            q = _rms(heads_split(q), nq_w)  # (B, Limg, H, D)
            k = _rms(heads_split(k), nk_w)
            v = heads_split(v)

            # RoPE on the latent stream only (encoder q/k are added un-rotated), matching
            # HunyuanVideo15AttnProcessor2_0 (apply_rotary_emb after norm_q/norm_k).
            if freqs_cis is not None:
                _cos, _sin = freqs_cis
                cos_b, sin_b = _rope_bcast(_cos, _sin, int(q.shape[1]), int(q.shape[3]), q.dtype)
                q = _apply_rope(q, cos_b, sin_b)
                k = _apply_rope(k, cos_b, sin_b)

            eq, ek, ev = _linear_split3(ne, w_aqkv, b_aqkv)
            eq = _rms(heads_split(eq), naq_w)  # (B, Ltxt, H, D)
            ek = _rms(heads_split(ek), nak_w)
            ev = heads_split(ev)

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
                joint_valid_lengths=joint_valid_lengths or [],
            )
        else:
            if joint_valid_lengths is not None:
                raise ValueError("per-batch Hunyuan joint masking currently requires sequence parallelism")
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
        hid_out = _merge_heads(hid_out, B, Limg)
        enc_out = _merge_heads(enc_out, B, Ltxt)

        hid, enc = _run_dual_stream_projection_schedule(
            _mmrs_overlap,
            hidden_start=(
                (lambda: _row_linear_mmrs_start(hid_out, wo))
                if _mmrs_overlap
                else (lambda: _row_linear(hid_out, wo, bo))
            ),
            context_complete=lambda: _row_linear(enc_out, ao_w, ao_b),
            hidden_finish=lambda rs: _row_linear_mmrs_finish(rs, bo),
        )
        return hid, enc

    def _ff_up(x, parts):
        w1, b1, act, w2, b2 = parts
        y = _linear(x, w1, b1)
        return act(y)

    def _ff_down(x, parts):
        _w1, _b1, _act, w2, b2 = parts
        return _row_linear(x, w2, b2)

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
        # L1-fit guard: the sharded kernel allocates several static circular
        # buffers in addition to the tensor shard. A 13f SP8 row has a 400 KiB
        # shard but measured 2.35 MiB of CBs, exceeding Blackhole's 1.5 MiB L1.
        # Include a conservative six-shard estimate and leave headroom for the
        # runtime; use interleaved LN when that estimate does not fit.
        # Keep this optimization to the dispatch-scale shapes it was tuned for.
        # The kernel's reduction CB also grows with block_h in ways not fully
        # represented by the tensor-byte estimate.
        if Mt > 8 or not _layer_norm_shard_fits((Mt * 32), ((Nt // gx) * 32), 2):
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

        joint_valid_lengths = kwargs.get("joint_valid_lengths")
        encoder_query_mask = kwargs.get("encoder_query_mask")
        if encoder_query_mask is not None:
            e = ttnn.multiply(e, encoder_query_mask)
            ne = ttnn.multiply(ne, encoder_query_mask)
        attn_out, ctx_out = _joint_attention(
            nh,
            ne,
            freqs_cis=freqs_cis,
            attn_bias=attn_bias,
            logical_n=kwargs.get("logical_n"),
            joint_valid_lengths=joint_valid_lengths,
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

        if _mmrs_overlap:
            # Both FF-up branches are independent.  Prepare context before the
            # latent MMRS, then place the complete legacy context row projection
            # between latent MMRS and its all-gather finish.
            h_up = _ff_up(nh2, ff_p)
            e_up = _ff_up(ne2, ffc_p)
            _w1, _b1, _act, h_w2, h_b2 = ff_p
            h_ff, e_ff = _run_dual_stream_projection_schedule(
                True,
                hidden_start=lambda: _row_linear_mmrs_start(h_up, h_w2),
                context_complete=lambda: _ff_down(e_up, ffc_p),
                hidden_finish=lambda rs: _row_linear_mmrs_finish(rs, h_b2),
            )
        else:
            h_ff = _ff_down(_ff_up(nh2, ff_p), ff_p)
            e_ff = _ff_down(_ff_up(ne2, ffc_p), ffc_p)
        h = ttnn.addcmul(h, gate_mlp, h_ff)
        e = ttnn.addcmul(e, c_gate_mlp, e_ff)
        if encoder_query_mask is not None:
            e = ttnn.multiply(e, encoder_query_mask)
        if wdt != ttnn.float32:  # keep the block fp32-in/fp32-out for the fp32 glue
            h = ttnn.typecast(h, ttnn.float32)
            e = ttnn.typecast(e, ttnn.float32)
        return h, e

    def _free_weights():
        """Deallocate this block's device weights. Idempotent.

        Weights are NOT offloaded to host -- the torch module they came from is
        still resident, and with HY_DIT_WEIGHT_CACHE a rebuild reloads them in
        ~4s, so there is nothing worth copying back."""
        freed = 0
        errors = []
        for t in _weights:
            try:
                ttnn.deallocate(t)
                freed += 1
            except Exception as err:  # already freed, or never on device
                errors.append(repr(err))
        _weights.clear()
        if errors and freed == 0:
            # Silently returning 0 here once hid the real defect (a weight-cache
            # early return that never registered its tensors), so surface it.
            raise RuntimeError(f"free_weights() released nothing; first error: {errors[0]}")
        return freed

    forward.free_weights = _free_weights
    return forward


def hunyuan_video15_transformer_block(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_transformer_block requires build(device, torch_module) to bind the "
        "block weights; the bare callable has no parameters."
    )
