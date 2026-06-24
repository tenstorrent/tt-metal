# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hand-rolled TT-NN ops for the Qwen3.5 gated-delta-net block.

For now this holds just the depthwise causal conv1d + SiLU that HF runs as
``F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])``.

We deliberately avoid ``ttnn.conv1d`` here. That op targets dense, multi-channel
convolutions (think a real CNN layer), so it spins up circular buffers sized for
gathering every input channel into every output channel — which overflows
per-core L1 once the channel count is as large as ``conv_dim`` (= key_dim*2 +
value_dim, a few thousand). But this conv is *depthwise* (``groups=conv_dim``):
each channel is filtered by its own tiny ``K``-tap kernel, with no cross-channel
mixing at all. That is exactly an FIR filter, and a K-tap FIR is just K shifted,
per-channel-scaled copies of the input added together — pure elementwise
multiply/add, which TT-NN runs cheaply and which sidesteps the L1 blowup.
"""

import math

import torch

import ttnn


def conv1d_weight_taps(conv_weight, kernel_size, mesh_device, *, dtype=ttnn.bfloat16, memory_config=None):
    """Slice the fused depthwise conv weight into ``K`` broadcastable per-channel taps.

    * ``conv_weight`` — HF ``conv1d.weight``, shape ``[D, 1, K]`` (torch on host, or a ttnn tensor).
    * returns a list of ``K`` ttnn tensors, each shaped ``[1, 1, D]``, so they
      broadcast over a ``[B, T, D]`` activation along the channel (last) dim.

    Tap ``k`` is ``weight[:, 0, k]`` — the single conv coefficient that every
    channel applies to the input sample sitting ``k`` steps into its kernel
    window. Pulling these K vectors out once (on host; the weight is tiny) is
    what turns the convolution into plain elementwise multiplies on device.
    """

    w = conv_weight if isinstance(conv_weight, torch.Tensor) else ttnn.to_torch(conv_weight)
    w = w.float()
    D = w.shape[0]
    taps = []
    for k in range(kernel_size):
        # [D] -> [1, 1, D] so the tap broadcasts across batch and sequence.
        tap = w[:, 0, k].reshape(1, 1, D).contiguous()
        taps.append(
            ttnn.from_torch(tap, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=memory_config)
        )
    return taps


def causal_conv1d_silu(x, weight_taps, kernel_size, mesh_device, *, pad, memory_config=None):
    """Depthwise causal conv1d followed by SiLU — matches HF bit-for-bit (up to dtype).

    Reproduces ``F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])`` for the no-cache
    prefill path, where HF's conv1d is depthwise (``groups=conv_dim``),
    ``bias=False``, and left-pads by ``kernel_size - 1`` to stay causal.

    Layout note: HF feeds ``mixed_qkv`` as ``[B, D, T]`` (channels-first, because
    torch ``Conv1d`` demands it). We instead take ``x`` as ``[B, 1, T, D]``
    (channels-*last*) so the per-channel taps broadcast over the last dim and the
    causal shifts run along the sequence dim — the cheap, natural layout for
    TT-NN. Transpose at the call site if your tensor is channels-first.

    * ``x``           — ttnn activation ``[B, 1, T, D]`` (TILE_LAYOUT).
    * ``weight_taps`` — the ``K`` tensors produced by ``tp_common.prepare_conv_taps``
                        (via ``tp_common.shard_small``), surfaced as ``Qwen35GDNWeights.w_taps``.
    * ``pad``         — optional persistent ``[B, 1, K-1, D]`` zero buffer to prepend.
                        Pass one (built once in ``__init__``) to keep this function
                        trace-capturable; omit it and we allocate a fresh one.
    * returns         — ttnn ``[B, 1, T, D]`` (TILE_LAYOUT).
    """
    B, T, D = x.shape[0], x.shape[2], x.shape[3]

    # TODO ttnn.concat for TILE_LAYOUT inputs can return garbage values, may need to switch to ROW_LAYOUT to avoid for now
    # Metal Issue: https://github.com/tenstorrent/tt-metal/issues/47293
    x_padded = ttnn.concat([pad, x], dim=2, memory_config=memory_config)

    # FIR accumulation: out[:, t, :] = Σ_k taps[k] * x_padded[:, t+k, :]. Each
    # term is one tap's contribution — the input slid k steps along the padded
    # sequence, scaled per channel — and we sum the K of them.
    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, :, k : k + T, :]
        # Slicing a tiled tensor on a non-tile-aligned seq offset can drop tile
        # layout; re-tilize before the elementwise multiply.
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(x_slice, weight_taps[k], memory_config=memory_config)
        out = term if out is None else ttnn.add(out, term, memory_config=memory_config)

    # SiLU is fused on at the end, exactly as HF wraps the conv output.
    return ttnn.silu(out, memory_config=memory_config)


def causal_conv1d_silu_update(x, conv_state, weight_taps, kernel_size, *, memory_config=None):
    """Depthwise causal conv1d + SiLU for the cache-update (decode) path.

    The decode-time sibling of :func:`causal_conv1d_silu`. It reproduces FLA's
    ``causal_conv1d_update`` torch fallback::

        h = torch.cat([conv_state, hidden_states], dim=-1)   # prepend cached history
        conv_state.copy_(h[..., -state_len:])                # roll the window forward
        out = F.silu(F.conv1d(h, w, padding=0, groups=D)[..., -seq_len:])

    The convolution itself is the prefill FIR unchanged — same per-channel taps,
    same SiLU. Only two things differ for decode, and they are the whole reason
    this is a separate function:

    * The left context is the *cached* ``conv_state`` (the last real inputs from
      prior steps), NOT zeros. Zero-padding here would make the first ``K-1``
      decode outputs read history as zero and silently corrupt them.
    * We must hand back the *updated* state so the next step has its window — the
      last ``state_len`` columns of the concatenated buffer.

    Layout matches the prefill sibling: channels-last ``[B, 1, T, D]`` so the taps
    broadcast over the last dim and the causal shifts run along the seq dim. No
    bias term — Qwen3.5's conv1d is ``bias=False``.

    ``state_len`` is read off ``conv_state`` exactly as the torch ref reads
    ``conv_state.shape[-1]``. This repo allocates it at the full kernel width
    ``K`` (reset_conv_state), but the offset math below also collapses to the
    prefill case for a ``K-1`` wide state, so either convention is handled.

    * ``x``           — ttnn ``[B, 1, T, D]`` (TILE_LAYOUT); the new decode token(s).
    * ``conv_state``  — ttnn ``[B, 1, state_len, D]`` (TILE_LAYOUT); the carried window.
    * ``weight_taps`` — the ``K`` tensors produced by ``tp_common.prepare_conv_taps``
                        (via ``tp_common.shard_small``), surfaced as ``Qwen35GDNWeights.w_taps``.
    * returns         — ``(out, new_state)``: ``out`` ``[B, 1, T, D]`` post-SiLU and
                        ``new_state`` ``[B, 1, state_len, D]`` (x's dtype) to copy
                        back into the persistent conv-state buffer.
    """
    T = 1  # decode-time conv runs one step at a time; the prefill FIR handles the full seq in one shot
    state_len = conv_state.shape[2]
    # The conv needs K-1 columns of left context; a narrower state can't fill the
    # window. Assert instead of silently emitting garbage — same constraint the
    # torch ref hits (its conv output would come out shorter than seq_len).
    assert state_len >= kernel_size - 1, f"conv_state width {state_len} < kernel_size-1 ({kernel_size - 1})"

    # conv_state lives in its buffer's dtype (this repo: bf16, see reset_conv_state)
    # while x is the bf16 linear output, and ttnn.concat needs both to agree — so
    # match x, the same cast the torch ref makes via `.to(weight.dtype)`. The guard
    # is a no-op while both are bf16, but keeps the op correct for any state dtype.
    # The state is only a shift buffer of recent inputs (no accumulation), so bf16
    # loses nothing here.
    if conv_state.dtype != x.dtype:
        conv_state = ttnn.typecast(conv_state, x.dtype)

    # The one structural change from prefill: prepend the cached history instead
    # of zeros. The buffer becomes [B, 1, state_len + T, D].
    # TODO ttnn.concat for TILE_LAYOUT inputs can return garbage values, may need to switch to ROW_LAYOUT to avoid for now
    # Metal Issue: https://github.com/tenstorrent/tt-metal/issues/47293
    x_padded = ttnn.concat([conv_state, x], dim=2, memory_config=memory_config)

    # Roll the window forward: the next step's left context is the last state_len
    # columns of what we just convolved (torch's conv_state.copy_(h[..., -state_len:])).
    # That tail starts at index T; the slice lands off a tile boundary, so re-tilize
    # before handing it back (causal_conv1d_silu caveat).
    new_state = x_padded[:, :, T:, :]
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)

    # FIR over the last T conv outputs. With a state_len-wide left context the
    # wanted outputs start at off = state_len - (K-1) (== 0 for prefill's K-1 pad,
    # == 1 for this repo's K-wide state), so tap k reads x_padded[off+k : off+k+T].
    off = state_len - (kernel_size - 1)
    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, :, off + k : off + k + T, :]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(x_slice, weight_taps[k], memory_config=memory_config)
        out = term if out is None else ttnn.add(out, term, memory_config=memory_config)

    return ttnn.silu(out, memory_config=memory_config), new_state


def chunk_identity(chunk_size, mesh_device, *, dtype=ttnn.float32, memory_config=None):
    """Build a ``[1, 1, chunk_size, chunk_size]`` identity once on host for the chunk math.

    TT-NN has no ``ttnn.eye`` matrix constructor (``ttnn.identity`` is the elementwise
    pass-through unary, not a matrix), so the ``attn + torch.eye`` of the torch ref and the
    ``I`` term of the lower-triangular inverse must come from a host ``torch.eye`` shipped over
    with ``from_torch`` — exactly the production pattern (create_chunk_masks_seq). The two
    leading size-1 dims let it broadcast over the ``[B, H, num_chunks, C, C]`` chunk batch in an
    ``add``/``matmul``. Built float32 because the inverse it feeds is numerically sensitive.
    """
    eye = torch.eye(chunk_size, dtype=torch.float32).reshape(1, 1, chunk_size, chunk_size)
    return ttnn.from_torch(
        eye,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=memory_config,
    )


def chunk_triangular_masks(chunk_size, mesh_device, *, dtype=ttnn.float32, memory_config=None):
    """Host-built 0/1 lower-triangular masks for the chunk math, applied via multiply.

    We do NOT use ``ttnn.tril`` here. This repo's own production GDN
    (ttnn_delta_rule_seq.py:160-163, create_chunk_masks_seq) documents that
    ``ttnn.tril``/``ttnn.triu`` return WRONG results on this build (wrong diagonal +
    spurious entries), so it builds every mask on host with ``torch.tril`` +
    ``from_torch`` and applies them with ``ttnn.multiply``. We follow that
    de-risked pattern rather than trusting the device tril. Two masks, both
    ``[1, 1, C, C]`` so they broadcast over the rank-5 ``[B, H, nc, C, C]`` chunk
    batch (and rank-4 per-chunk slices):

    * ``lower_incl``   — ``torch.tril(ones, 0)``, 1 on AND below the diagonal.
      Realizes the two trils of the decay_mask (torch 278) and the intra-chunk
      causal ``masked_fill(triu diag1, 0)`` (torch 298, KEEP the diagonal).
    * ``strict_lower`` — ``torch.tril(ones, -1)``, 1 STRICTLY below the diagonal.
      Realizes the ``masked_fill(triu diag0, 0)`` of torch 279 (zeros the diagonal
      too), which is the strictly-lower S of the (I - S) system the inverse solves.
    """
    ones = torch.ones(chunk_size, chunk_size, dtype=torch.float32)

    def _from(m):
        return ttnn.from_torch(
            m.reshape(1, 1, chunk_size, chunk_size),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=memory_config,
        )

    return {
        "lower_incl": _from(torch.tril(ones, diagonal=0)),
        "strict_lower": _from(torch.tril(ones, diagonal=-1)),
    }


# Largest sub-block we hand to the Neumann leaf solver. The whole-block Neumann series is
# stable only on tiny, well-conditioned matrices; real deep-layer chunk matrices reach
# cond ~65, where a 32-wide leaf still diverges on device (the Neumann partials overflow
# even HiFi4 precision before Newton-Schulz can recover the cancellation). 16 is the
# largest leaf that stays stable; 8 works too but only adds sub-tile work for no gain.
_SOLVE_LEAF = 16


def _neumann_newton_schulz_leaf(L, eye, n, kernel_config):
    """Whole-block inverse of a SMALL unit-lower-triangular block via Neumann + Newton-Schulz.

    Writing ``L = I + N`` with ``N = L - I`` strictly lower (hence nilpotent, ``N**n == 0``),
    the Neumann series ``L^{-1} = sum_k (-N)**k`` is exact and is summed in ``ceil(log2(n))``
    doubling matmuls; two Newton-Schulz steps (``X <- X(2I - L X)``) then mop up the matmul
    rounding. This is the original whole-block solver, now demoted to the LEAF of
    :func:`solve_unit_lower_triangular`: it is handed the full ``[.., C, C]`` tile AFTER ``L``
    has been masked to its ``_SOLVE_LEAF``-sized diagonal blocks, so every block it inverts is
    ``<= _SOLVE_LEAF`` wide (cond ~1) — the regime where it was always accurate. Because the
    masked matrix is block-diagonal, ``(-N)**k`` stays block-diagonal, so this single call
    inverts all leaf blocks at once in one set of full-tile matmuls.
    """
    # N = L - I (strictly lower, nilpotent); P starts as (-N)**1 and is repeatedly
    # squared so step s holds (-N)**(2**s); R accumulates the running partial sum.
    N = ttnn.subtract(L, eye)
    P = ttnn.neg(N)
    R = ttnn.add(eye, P)  # f(2) = I + (-N)
    P = ttnn.matmul(P, P, compute_kernel_config=kernel_config)  # (-N)**2
    steps = math.ceil(math.log2(n)) - 1
    for s in range(steps):
        # Doubling identity: f(2n) = f(n) @ (I + (-N)**n).
        R = ttnn.matmul(R, ttnn.add(eye, P), compute_kernel_config=kernel_config)
        # Skip squaring on the last step: that power (>= (-N)**n) is nilpotent-zero and
        # only ever feeds the NEXT iteration's R update, so on the final step it is dead
        # work. Dropping it leaves R bit-identical while saving one full-tile matmul.
        if s < steps - 1:
            P = ttnn.matmul(P, P, compute_kernel_config=kernel_config)

    # Newton-Schulz refinement: each step squares the residual ||I - L X||, cheaply
    # recovering the bits the bf16-mantissa'd matmuls shed even with fp32 accumulation.
    for _ in range(2):
        LX = ttnn.matmul(L, R, compute_kernel_config=kernel_config)
        two_I_minus_LX = ttnn.subtract(ttnn.add(eye, eye), LX)
        R = ttnn.matmul(R, two_I_minus_LX, compute_kernel_config=kernel_config)
    return R


def chunk_solve_masks(chunk_size, mesh_device, *, leaf=_SOLVE_LEAF, dtype=ttnn.float32, memory_config=None):
    """Static 0/1 masks that drive the masked block forward-substitution in
    :func:`solve_unit_lower_triangular`.

    That inverse is computed by MASKING (a multiply) rather than SLICING the matrix apart, so
    every op stays a full ``[.., C, C]`` tile op — no sub-tile slices, no ``concat`` /
    ``to_layout`` re-tilize. Like :func:`chunk_identity` / :func:`chunk_triangular_masks` these
    are minted with one host ``from_torch`` here at init and never mid-forward, keeping the
    chunk graph trace-capturable. We build them with ``torch`` index arithmetic + ``from_torch``
    rather than ``ttnn``: this build's ``ttnn.tril`` is wrong (see :func:`chunk_triangular_masks`)
    and a host bool mask is trivially correct. All masks are ``[1, 1, C, C]`` so they broadcast
    over the ``[B, H, num_chunks, C, C]`` chunk batch. Returns:

    * ``blockdiag`` — 1 where row and col share a ``leaf``-sized diagonal block. Masking
      ``N = L - I`` with it yields a block-diagonal matrix whose ``leaf`` blocks are each
      ``leaf``-nilpotent, so the Neumann leaf inverts them all at once and stably (see
      :data:`_SOLVE_LEAF`).
    * ``offdiag[b]`` for ``b`` in ``2*leaf, 4*leaf, ..., chunk_size`` — 1 on the strictly
      lower off-diagonal half-block of each ``b``-block (rows in its lower ``b/2`` half, cols
      in its upper ``b/2`` half). This is the ``C`` block of the 2x2 partition at scale ``b``
      consumed by the climb step ``L^{-1} = D - D C D``.
    """
    r = torch.arange(chunk_size)[:, None]
    c = torch.arange(chunk_size)[None, :]

    def _from(m):
        return ttnn.from_torch(
            m.to(torch.float32).reshape(1, 1, chunk_size, chunk_size),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=memory_config,
        )

    offdiag = {}
    b = 2 * leaf
    while b <= chunk_size:
        half = b // 2
        offdiag[b] = _from((r // b == c // b) & (r // half != c // half))
        b *= 2

    return {
        "blockdiag": _from(r // leaf == c // leaf),
        "offdiag": offdiag,
    }


def solve_unit_lower_triangular(L, eye, masks, *, compute_kernel_config=None):
    """Invert a batch of UNIT-lower-triangular matrices ``L`` (1s on the diagonal).

    Device-side replacement for the torch reference's in-place forward-substitution loop
    (``for i in range(1, chunk_size): attn[..., i, :i] = ...``). TT-NN tensors are immutable
    (no ``__setitem__``/scatter) and that loop is a strict sequential ragged-slice row solve,
    so it cannot be transcribed directly. We do the SAME block forward substitution, but
    expressed with MASKS on the full tile instead of slicing it into sub-blocks.

    For the 2x2 partition ``L = [[A, 0], [C, B]]`` of a unit-lower-triangular block, the
    closed-form inverse is ``L^{-1} = D - D C D`` with ``D = blockdiag(A^{-1}, B^{-1})`` and
    ``C`` the (embedded) off-diagonal block — exactly what the torch row loop computes. Two
    things make this cheap on device:

    * The leaf inverts ALL ``_SOLVE_LEAF``-blocks at once: masking ``N = L - I`` to its block
      diagonal gives a block-diagonal matrix the Neumann leaf inverts in one set of full-tile
      matmuls (each block is ``_SOLVE_LEAF``-nilpotent, the stable regime).
    * Each climb step (``b = 2*leaf, 4*leaf, ..., chunk_size``) does ``D <- D - D C_b D`` on
      the WHOLE tile, applying every ``b``-block's off-diagonal correction simultaneously.

    So the inverse is the leaf plus ~``2*log2(chunk_size/leaf)`` full-tile matmuls, with NO
    slice / ``concat`` / ``to_layout`` — versus the earlier per-block sliced recursion, which
    fanned out into ~100 small matmuls and a forest of sub-tile slice + re-tilize ops (the
    reason this op was the GDN prefill bottleneck). It is numerically identical to that
    recursion (each leaf runs the same Neumann series on the same diagonal blocks): validated
    rel err ~2e-7 vs the exact inverse on cond~65 matrices, matching the version it replaces.

    Both forms replaced a still-earlier WHOLE-block Neumann + Newton-Schulz inverse that blew
    up (~6e28 vs O(1)) on real cond~65 deep-layer chunks and turned the logits into token-salad:
    there the full ``chunk_size``-nilpotent series overflowed before it could cancel. Block
    forward substitution only ever runs Neumann on tiny ``_SOLVE_LEAF`` blocks, so it stays
    bounded at any conditioning.

    * ``L``     — ``[..., chunk_size, chunk_size]`` float32, unit lower triangular.
    * ``eye``   — ``[1, 1, chunk_size, chunk_size]`` float32 identity (broadcasts over the
                  leading batch dims), e.g. from :func:`chunk_identity`.
    * ``masks`` — dict from :func:`chunk_solve_masks` (``blockdiag`` + ``offdiag[b]``); these
                  encode ``chunk_size`` and the leaf size, so neither is passed separately.
    * ``compute_kernel_config`` — accepted for call-site symmetry with the other chunk ops
                  but DELIBERATELY ignored: this inverse needs HiFi4 (the chunk math's HiFi2
                  is too coarse and lets the leaf series diverge), so we force it below.
    """
    # Force HiFi4 + fp32 accumulation regardless of the caller's config: this is the one
    # chunk op where HiFi2 isn't enough to keep the leaf Neumann series from diverging.
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    N = ttnn.subtract(L, eye)

    # Leaf: mask N to its _SOLVE_LEAF-sized diagonal blocks (block-diagonal, each block
    # _SOLVE_LEAF-nilpotent) and invert every block at once with the Neumann leaf. The result
    # is block-diagonal too; re-mask it to scrub any off-block fp leakage so the climb below
    # consumes an exact blockdiag(A^{-1}, B^{-1}).
    blockdiag = masks["blockdiag"]
    L_blockdiag = ttnn.add(eye, ttnn.multiply(N, blockdiag))
    D = _neumann_newton_schulz_leaf(L_blockdiag, eye, _SOLVE_LEAF, kernel_config)
    D = ttnn.multiply(D, blockdiag)

    # Climb scales: entering step b, D holds the inverse of every (b/2)-block; the off-diagonal
    # correction D <- D - D @ C_b @ D promotes it to the inverse of every b-block (C_b is L's
    # off-diagonal block at scale b, == N there since L's off-diagonal is N's). Associate as
    # (D @ C_b) @ D so each matmul stays a single full-tile op.
    for b in sorted(masks["offdiag"]):
        C = ttnn.multiply(N, masks["offdiag"][b])
        DC = ttnn.matmul(D, C, compute_kernel_config=kernel_config)
        D = ttnn.subtract(D, ttnn.matmul(DC, D, compute_kernel_config=kernel_config))

    return D


def l2norm(x: ttnn.Tensor, dim: int = -1, eps: float = 1e-6, *, memory_config=None):
    """L2-normalize along ``dim``, matching FLA's ``x * rsqrt(sum(x**2) + eps)``.

    Built from primitives rather than ``ttnn.rms_norm``: RMS norm reduces with a
    *mean* (divides by the feature count) and folds ``eps`` in differently, so it
    would silently disagree with FLA. The reduce-then-broadcast shape below is the
    exact transcription of the torch reference (verified PCC ~0.99999, bf16).
    """
    # Σ x² over the feature dim; keepdim leaves a size-1 axis so it broadcasts
    # back over x in the final multiply.
    sq_sum = ttnn.sum(ttnn.square(x), dim=dim, keepdim=True, memory_config=memory_config)
    inv_norm = ttnn.rsqrt(ttnn.add(sq_sum, eps), memory_config=memory_config)
    return ttnn.multiply(x, inv_norm, memory_config=memory_config)
