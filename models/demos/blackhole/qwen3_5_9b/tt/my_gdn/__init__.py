import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from .weights import load_gdn_weights
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from .operations import (
    causal_conv1d_silu,
    l2norm,
    chunk_identity,
    chunk_triangular_masks,
    solve_unit_lower_triangular,
)


class Qwen35RMSNormGated(LightweightModule):
    def __init__(self, weight: ttnn.Tensor, eps=1e-6):
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: ttnn.Tensor, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = ttnn.typecast(hidden_states, ttnn.float32)
        variance = ttnn.mean(ttnn.square(hidden_states), dim=-1, keepdim=True)
        hidden_states = hidden_states * ttnn.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight
        hidden_states = (
            hidden_states * ttnn.silu(ttnn.typecast(gate, ttnn.float32)) if gate is not None else hidden_states
        )
        hidden_states = ttnn.typecast(hidden_states, input_dtype)
        return hidden_states


class Qwen35GatedDeltaNet(LightweightModule):
    def __init__(
        self,
        args: Qwen35ModelArgs,
        state_dict,
        mesh_device,
    ):
        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_norm_epsilon = args.norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.scale = 1 / (self.head_k_dim**0.5)
        self.weights = load_gdn_weights(mesh_device=mesh_device, state_dict=state_dict, args=args)
        self.mesh_device = mesh_device
        self.recurrent_state = None
        self.cache_params = None
        self.batch_size = args.max_batch_size
        self.chunk_size = args.gdn_chunk_size
        self.norm = Qwen35RMSNormGated(weight=self.weights.w_norm, eps=self.layer_norm_epsilon)

        self.last_recurrent_state = None
        self.conv_state = None
        self.zeroes_recurrent_state = None
        self.reset_recurrent_state()
        self.reset_conv_state()
        self.initialize_params_gated_delta_rule()

    def reset_conv_state(self):
        self.conv_state = ttnn.zeros(
            [self.batch_size, 1, self.conv_kernel_size, self.conv_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

    def reset_recurrent_state(self):
        def _zeroes():
            return ttnn.zeros(
                [self.batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )

        if not self.last_recurrent_state:
            self.last_recurrent_state = _zeroes()
            self.zeroes_recurrent_state = _zeroes()
        else:
            ttnn.copy(self.zeroes_recurrent_state, self.last_recurrent_state)

        return self.last_recurrent_state

    def initialize_params_gated_delta_rule(self):
        self.eye = chunk_identity(self.chunk_size, self.mesh_device)
        self.triangular_masks = chunk_triangular_masks(self.chunk_size, self.mesh_device)

    def chunk_gated_delta_rule(
        self, query, key, value, g, beta, chunk_size, initial_state=None, use_qk_l2norm_in_kernel=False
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)

        # beta, g arrive [B, seq, num_v_heads] (rank-3, head-LAST) from forward_prefill;
        # query/key/value arrive [B, seq, num_v_heads, head_dim] (q/k repeat_interleaved up
        # to num_v_heads first). transpose(1,2) puts heads before seq -> beta/g [B, H, seq],
        # qkv [B, H, seq, D], matching torch's transpose at lines 250-252.
        query, key, value, beta, g = [ttnn.transpose(x, 1, 2) for x in (query, key, value, beta, g)]

        # torch line 251 casts all five to float32 before the chunk math. The lower-
        # triangular inverse and the cross-chunk state scan are numerically sensitive
        # (bf16 tanks the inverse), so promote here rather than relying on the matmul
        # compute config alone — q/k/v/beta arrive bf16 from forward_prefill, g is already
        # float32. ttnn.float32 is usable on Blackhole and this matches the reference.
        query, key, value, beta, g = [ttnn.typecast(x, ttnn.float32) for x in (query, key, value, beta, g)]

        sequence_length = query.shape[2]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        qkv_pad = [(0, 0), (0, 0), (0, pad_size), (0, 0)]  # pad seq (dim 2) at the end only
        bg_pad = [(0, 0), (0, 0), (0, pad_size)]
        query = ttnn.pad(query, padding=qkv_pad, value=0.0)
        key = ttnn.pad(key, padding=qkv_pad, value=0.0)
        value = ttnn.pad(value, padding=qkv_pad, value=0.0)
        beta = ttnn.pad(beta, padding=bg_pad, value=0.0)
        g = ttnn.pad(g, padding=bg_pad, value=0.0)

        total_sequence_length = sequence_length + pad_size
        query = query * self.scale

        v_beta = value * ttnn.unsqueeze(beta, -1)
        k_beta = key * ttnn.unsqueeze(beta, -1)

        # Numerically sensitive chunk math: the lower-triangular inverse and the
        # cross-chunk state scan compound error, so every matmul below accumulates in
        # fp32 (the torch ref casts all five inputs to float32). HiFi2 keeps the matmul
        # mantissa from being truncated to bf16 even when the tensors are float32.
        chunk_math_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Split the padded sequence axis into (num_chunks, chunk_size): rank-4
        # [B, H, L, D] -> rank-5 [B, H, num_chunks, chunk_size, D]. The last two dims
        # (chunk_size=64, D=128) stay tile-aligned, so this is a metadata-only reshape;
        # every op downstream (matmul/cumsum/tril/exp) batches over the leading dims.
        batch_size, num_heads, _, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        num_chunks = total_sequence_length // chunk_size

        def _to_chunks(x):
            return ttnn.reshape(x, [batch_size, num_heads, num_chunks, chunk_size, x.shape[-1]])

        query = _to_chunks(query)
        key = _to_chunks(key)
        value = _to_chunks(value)
        k_beta = _to_chunks(k_beta)
        v_beta = _to_chunks(v_beta)
        # g carries no feature dim, so it stays one rank below the qkv tensors throughout.
        # NOTE: unlike q/k/v this splits the LAST (tiled) dim ([..,L] -> [..,nc,64]) rather
        # than the seq dim, and the cumsum below reduces along that freshly-split axis. The
        # research verified this is safe on the current build (reshape exact, cumsum has only
        # fp32-rounding error); if a future build regresses reshape on the last tiled dim,
        # gate this with a PCC assert.
        g = ttnn.reshape(g, [batch_size, num_heads, num_chunks, chunk_size])

        # Identity (for the +eye / the I of the inverse) and the host-built triangular masks,
        # both [1,1,C,C] so they broadcast over the [B,H,nc] chunk batch. The masks replace
        # every ttnn.tril call below: this repo's production GDN documents ttnn.tril/triu as
        # WRONG on this build and builds masks via from_torch + multiply instead. We follow
        # that de-risked pattern (see chunk_triangular_masks).
        eye = self.eye
        masks = self.triangular_masks
        lower_incl = masks["lower_incl"]
        strict_lower = masks["strict_lower"]

        # Per-chunk cumulative decay (torch line 277). EVERY later use of g wants this
        # cumulative value, not the raw logits.
        g = ttnn.cumsum(g, dim=-1, dtype=ttnn.float32)

        # decay_mask (torch line 278): D[a,b] = g_cum[a] - g_cum[b], with a DOUBLE tril.
        # The first tril (diagonal=0) zeros the strictly-upper entries BEFORE exp; after
        # exp those upper entries would be exp(0)=1, so the SECOND tril re-zeros them.
        # Dropping either tril silently corrupts every attn matrix below. We realize each
        # tril as a multiply by the host-built lower_incl mask (1 on+below diagonal): the
        # first multiply zeros the upper (so exp gives 1 there), the second re-zeros it —
        # bit-identical to torch's two trils, but without trusting ttnn.tril.
        decay_mask = ttnn.subtract(ttnn.unsqueeze(g, -1), ttnn.unsqueeze(g, -2))
        decay_mask = ttnn.multiply(decay_mask, lower_incl)
        decay_mask = ttnn.exp(decay_mask)
        decay_mask = ttnn.multiply(decay_mask, lower_incl)

        # attn (torch line 279): -((k_beta @ key^T) * decay_mask) then masked_fill with
        # triu(diagonal=0) -> 0. masked_fill zeroes the on-and-above-diagonal entries, so
        # the survivors are STRICTLY lower; realize that as a multiply by strict_lower (the
        # diagonal is zeroed too here — a DIFFERENT mask from the intra-chunk one below).
        # transpose_b gives key^T without an explicit transpose. The neg before the mask is
        # harmless (0 stays 0), matching the torch order.
        attn = ttnn.matmul(k_beta, key, transpose_b=True, compute_kernel_config=chunk_math_kernel_config)
        attn = ttnn.multiply(attn, decay_mask)
        attn = ttnn.neg(attn)
        attn = ttnn.multiply(attn, strict_lower)

        # Replace torch's in-place forward-substitution loop (lines 280-284) with an
        # algebraic inverse. The loop builds the strictly-lower part of (I - attn)^{-1}
        # and line 284 adds I; since `attn` here is strictly lower, L = I - attn is unit
        # lower triangular and solve_unit_lower_triangular(L) IS the full (I - attn)^{-1}
        # (the +eye is folded into the k=0 Neumann term). attn is rebound to that inverse.
        L = ttnn.add(eye, ttnn.neg(attn))
        attn = solve_unit_lower_triangular(L, eye, chunk_size, compute_kernel_config=chunk_math_kernel_config)

        # torch lines 285-286: rebind `value` to the inverted-transition-applied v_beta
        # (this is what v_i reads in the chunk loop), and the decay-weighted k_cumdecay.
        value = ttnn.matmul(attn, v_beta, compute_kernel_config=chunk_math_kernel_config)
        k_cumdecay = ttnn.matmul(
            attn,
            ttnn.multiply(k_beta, ttnn.unsqueeze(ttnn.exp(g), -1)),
            compute_kernel_config=chunk_math_kernel_config,
        )

        # Recurrent state carried ACROSS chunks (torch lines 287-291); prefill always
        # starts from zero. Rank-4 (no chunk dim): [B, H, k_head_dim, v_head_dim].
        last_recurrent_state = self.reset_recurrent_state()

        # core_attn_out accumulator (torch line 292): TT-NN has no per-chunk scatter
        # (core_attn_out[:, :, i] = ...), so collect each chunk's slice and concat on the
        # chunk dim after the loop instead of pre-allocating and writing in place.
        core_attn_out_chunks = []

        # Sequential scan over chunks (torch line 296): each iteration depends on the
        # last_recurrent_state the previous one produced, so it is unrolled in Python.
        for i in range(num_chunks):
            # Select chunk i (torch line 297). Indexing the chunk dim (a leading/batch
            # dim, not a tile dim) reduces rank to [B, H, chunk_size, *] and is tile-safe.
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            decay_mask_i = decay_mask[:, :, i]
            g_i = g[:, :, i]

            # Intra-chunk attention (torch line 298): masked_fill with triu(diagonal=1)
            # zeroes only the STRICTLY-upper entries, KEEPING the diagonal. Realize via a
            # multiply by lower_incl (1 on+below) — a DIFFERENT mask from line 279's
            # strict_lower; swapping the two is a silent, plausible-looking correctness bug.
            attn_intra = ttnn.matmul(q_i, k_i, transpose_b=True, compute_kernel_config=chunk_math_kernel_config)
            attn_intra = ttnn.multiply(attn_intra, decay_mask_i)
            attn_intra = ttnn.multiply(attn_intra, lower_incl)

            # Inter-chunk pieces (torch lines 299-302): the carried state's predicted
            # contribution v_prime, the residual v_new, and the decayed-query read of the
            # state. g_i broadcasts over the v_head_dim via a trailing unsqueeze.
            v_prime = ttnn.matmul(
                k_cumdecay[:, :, i], last_recurrent_state, compute_kernel_config=chunk_math_kernel_config
            )
            v_new = ttnn.subtract(v_i, v_prime)
            attn_inter = ttnn.matmul(
                ttnn.multiply(q_i, ttnn.unsqueeze(ttnn.exp(g_i), -1)),
                last_recurrent_state,
                compute_kernel_config=chunk_math_kernel_config,
            )
            out_i = ttnn.add(attn_inter, ttnn.matmul(attn_intra, v_new, compute_kernel_config=chunk_math_kernel_config))
            # Re-insert the chunk axis (size 1) so a later concat on dim=2 rebuilds the
            # [B, H, num_chunks, chunk_size, v_head_dim] tensor.
            core_attn_out_chunks.append(ttnn.unsqueeze(out_i, 2))

            # State update for the next chunk (torch lines 303-306). g_last is the
            # cumulative g at the LAST within-chunk position = the chunk's total decay.
            # Extracting that last column is a within-tile (64-dim) slice, so re-tilize
            # before using it (causal_conv1d_silu caveat).
            g_last = g_i[:, :, chunk_size - 1 : chunk_size]
            g_last = ttnn.to_layout(g_last, ttnn.TILE_LAYOUT)
            # Term 1: decay the old state by exp(g_last) (two trailing dims to broadcast
            # over the [k_head_dim, v_head_dim] state).
            decayed_state = ttnn.multiply(last_recurrent_state, ttnn.exp(ttnn.unsqueeze(g_last, -1)))
            # Term 2: outer product of the remaining-decay-weighted keys with v_new.
            # exp(g_last - g_i) is the decay from each position to the chunk end.
            k_decayed = ttnn.multiply(k_i, ttnn.unsqueeze(ttnn.exp(ttnn.subtract(g_last, g_i)), -1))
            state_update = ttnn.matmul(
                ttnn.transpose(k_decayed, -1, -2), v_new, compute_kernel_config=chunk_math_kernel_config
            )
            last_recurrent_state = ttnn.add(decayed_state, state_update)

        # Reassemble chunks (torch line 310): concat on the chunk dim, then merge
        # (num_chunks, chunk_size) back into the padded sequence axis (tile-safe, both
        # tile-aligned) -> [B, H, total_sequence_length, v_head_dim].
        core_attn_out = ttnn.concat(core_attn_out_chunks, dim=2)
        core_attn_out = ttnn.reshape(core_attn_out, [batch_size, num_heads, total_sequence_length, v_head_dim])

        # Drop the padding rows (torch line 311) and undo the line-45 transpose, then cast
        # back to the input dtype (torch line 312). The seq slice is on a non-tile-aligned
        # boundary while seq is a tile dim, so it must be re-tilized (causal_conv1d_silu
        # caveat) BEFORE the transpose — leaving the dirty tile padding in place corrupts
        # the transpose, which reads whole tiles. Re-tilizing materializes clean rows.
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = ttnn.to_layout(core_attn_out, ttnn.TILE_LAYOUT)
        core_attn_out = ttnn.transpose(core_attn_out, 1, 2)
        core_attn_out = ttnn.typecast(core_attn_out, initial_dtype)

        # forward_prefill unpacks exactly this 2-tuple. The torch ref would None the state
        # (output_final_state defaults False), but we return the real state so the cache /
        # decode path can pick it up; callers that don't need it simply ignore it.
        return core_attn_out, last_recurrent_state

    def forward_prefill(self, hidden_states):
        """
        hidden_states: [B=1, 1, seq_len, hidden_size]
        """
        # attention masking if there is an attention parameter
        weights = self.weights
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[2]

        # mixed_qkv: [B=1, 1, seq_len, conv_dim] — already channels-last out of the linear,
        # which is exactly the layout the FIR conv wants (taps broadcast over the last dim).
        # We deliberately do NOT transpose to channels-first: that's what torch's Conv1d
        # demands, not us, and flipping here would feed the conv the layout it can't use.
        mixed_qkv = ttnn.linear(hidden_states, weights.wqkv)

        z = ttnn.linear(hidden_states, weights.wz)
        z = ttnn.reshape(z, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        # a: [B=1, 1, seq_len, num_v_heads]
        # b: [B=11, 1, seq_len, num_v_heads]
        b = ttnn.linear(hidden_states, weights.wb)
        a = ttnn.linear(hidden_states, weights.wa, dtype=ttnn.float32)

        # Prefill path does not use precomputed states
        breakpoint()
        if self.conv_state is not None:
            # in transformers, you might have to pad mixed_qkv if it is smaller than the conv kernel
            # but, you can also reshape down if the seq_len > conv kernel
            # in prefill, we will just slice down, but we might need to pad up...
            conv_state = mixed_qkv[..., :, -self.conv_kernel_size :, :]
            ttnn.copy(conv_state, self.conv_state)
        else:
            # causal_conv1d_silu's contract is a canonical [B, T, D]; drop the singleton
            # dim 1 that ttnn.linear leaves on. In tile layout this is a metadata-only
            # reshape (the last two dims, which carry the tiles, are untouched).

            mixed_qkv = ttnn.reshape(mixed_qkv, (batch_size, 1, seq_len, self.conv_dim))
            mixed_qkv = causal_conv1d_silu(
                x=mixed_qkv,
                weight_taps=weights.w_taps,
                kernel_size=self.conv_kernel_size,
                mesh_device=self.mesh_device,
            )  # [B, 1, seq_len, conv_dim, dim]

        query, key, value = ttnn.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        beta = ttnn.squeeze(ttnn.sigmoid(b), 1)
        g = ttnn.squeeze(weights.neg_A_log_exp * ttnn.softplus(a + weights.dt_bias), 1)
        if self.num_v_heads // self.num_k_heads > 1:
            query = ttnn.repeat_interleave(query, self.num_v_heads // self.num_k_heads, dim=2)
            key = ttnn.repeat_interleave(key, self.num_v_heads // self.num_k_heads, dim=2)

            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=None,
                use_qk_l2norm_in_kernel=True,
            )
        ## update the recurrent state
        ttnn.copy(last_recurrent_state, self.last_recurrent_state)

        core_attn_out = ttnn.reshape(core_attn_out, (-1, self.head_v_dim))
        z = ttnn.reshape(z, (-1, self.head_v_dim))

        core_attn_out = self.norm(core_attn_out, gate=z)
        core_attn_out = ttnn.reshape(core_attn_out, (batch_size, seq_len, -1))
        out = ttnn.linear(core_attn_out, weights.wo)
        return out
