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


def causal_conv1d_silu(x, weight_taps, kernel_size, mesh_device, *, memory_config=None):
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
    * ``weight_taps`` — the ``K`` tensors returned by :func:`conv1d_weight_taps`.
    * returns         — ttnn ``[B, T, D]`` (TILE_LAYOUT).
    """
    B, T, D = x.shape[0], x.shape[2], x.shape[3]

    # Left-pad the sequence with K-1 zeros. This is the whole trick behind making
    # the conv causal: with the pad in front, output position t only ever reads
    # inputs at or before t. It reproduces HF's `padding=K-1` together with the
    # `[:seq_len]` slice — HF also right-pads, but that slice throws the right
    # tail away, so we simply never create it.
    pad = ttnn.zeros(
        [B, 1, kernel_size - 1, D],
        device=mesh_device,
        dtype=x.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
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
    * ``weight_taps`` — the ``K`` tensors from :func:`conv1d_weight_taps`.
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
            memory_config=memory_config,
        )

    return {
        "lower_incl": _from(torch.tril(ones, diagonal=0)),
        "strict_lower": _from(torch.tril(ones, diagonal=-1)),
    }


def solve_unit_lower_triangular(L, eye, chunk_size, *, compute_kernel_config=None):
    """Invert a batch of UNIT-lower-triangular matrices ``L`` (1s on the diagonal).

    This is the device-side replacement for the torch reference's in-place forward-
    substitution loop (``for i in range(1, chunk_size): attn[..., i, :i] = ...``).
    TT-NN tensors are immutable (no ``__setitem__``/scatter) and that loop is a strict
    sequential, ragged-slice row solve, so it cannot be transcribed. Instead we invert
    the whole block algebraically: writing ``L = I + N`` with ``N = L - I`` strictly
    lower-triangular (hence nilpotent, ``N**chunk_size == 0``), the Neumann series
    ``L^{-1} = sum_k (-N)^k`` is exact and is summed in ``ceil(log2(chunk_size))``
    doubling matmuls. Two Newton-Schulz steps (``X <- X(2I - L X)``) then mop up the
    float32 matmul rounding. This matches the production solver in the experimental GDN
    (ttnn_delta_rule_seq.py ``_solve_lower_triangular_ttnn``), specialized to unit
    diagonal so the ``D = diag(L)`` rescaling drops out. Validated PCC ~0.99998 against
    the torch loop on the real (l2-normalized) chunk matrices, which are well
    conditioned (cond ~1.2); float32 is mandatory here or the inverse diverges.

    * ``L``   — ``[..., chunk_size, chunk_size]`` float32, unit lower triangular.
    * ``eye`` — ``[1, 1, chunk_size, chunk_size]`` float32 identity (broadcasts over the
                leading batch dims), e.g. from :func:`chunk_identity`.
    """
    # N = L - I (strictly lower, nilpotent); P starts as (-N)**1 and is repeatedly
    # squared so step s holds (-N)**(2**s); R accumulates the running partial sum.
    N = ttnn.subtract(L, eye)
    P = ttnn.neg(N)
    R = ttnn.add(eye, P)  # f(2) = I + (-N)
    P = ttnn.matmul(P, P, compute_kernel_config=compute_kernel_config)  # (-N)**2
    for _ in range(math.ceil(math.log2(chunk_size)) - 1):
        # Doubling identity: f(2n) = f(n) @ (I + (-N)**n).
        R = ttnn.matmul(R, ttnn.add(eye, P), compute_kernel_config=compute_kernel_config)
        P = ttnn.matmul(P, P, compute_kernel_config=compute_kernel_config)

    # Newton-Schulz refinement: each step squares the residual ||I - L X||, cheaply
    # recovering the bits the bf16-mantissa'd matmuls shed even with fp32 accumulation.
    for _ in range(2):
        LX = ttnn.matmul(L, R, compute_kernel_config=compute_kernel_config)
        two_I_minus_LX = ttnn.subtract(ttnn.add(eye, eye), LX)
        R = ttnn.matmul(R, two_I_minus_LX, compute_kernel_config=compute_kernel_config)
    return R


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
