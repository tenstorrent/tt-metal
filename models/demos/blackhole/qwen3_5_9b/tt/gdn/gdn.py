from enum import Enum

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import tt_all_reduce

from .operations import (
    causal_conv1d_silu,
    causal_conv1d_silu_update,
    chunk_identity,
    chunk_triangular_masks,
    l2norm,
    solve_unit_lower_triangular,
)
from .weights import load_gdn_weights


class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


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
        tt_ccl=None,
        tensor_cache_path=None,
    ):
        # Tensor-parallel context. The delta-rule recurrence is per-value-head, so TP
        # shards by head: every per-head dim below is the LOCAL (per-device) count =
        # global // num_devices. At TP=1 these equal the originals, so the validated
        # single-device shapes are unchanged. tt_ccl drives the one collective this layer
        # needs — the reduce-scatter over the row-parallel out_proj output (TP>1 only).
        self.args = args
        self.num_devices = args.num_devices
        self.tt_ccl = tt_ccl
        tp = self.num_devices

        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads // tp
        self.num_k_heads = args.linear_num_key_heads // tp
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_norm_epsilon = args.norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.scale = 1 / (self.head_k_dim**0.5)  # head dim is NOT sharded, so scale is unchanged
        self.weights = load_gdn_weights(
            mesh_device=mesh_device, state_dict=state_dict, args=args, tensor_cache_path=tensor_cache_path
        )
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
            # bf16, matching the bf16 mixed_qkv it caches: the state is a verbatim
            # shift buffer of recent conv inputs (no accumulation), so fp32 only
            # doubled DRAM/bandwidth for no precision, and fp16 would risk range
            # clipping on activation outliers. bf16 round-trips losslessly.
            dtype=ttnn.bfloat16,
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

        # Persistent zero pads for the prefill path, built once here next to the other
        # per-shape constants so the forward never has to mint them. Allocating a
        # constant buffer (ttnn.zeros / ttnn.pad with a fill value) inside the forward
        # is a host-side write that ttnn trace capture rejects; concatenating against a
        # pre-built buffer is pure device work and captures cleanly. All dims below are
        # already the per-device (post-TP-shard) values, so we do NOT divide by tp.
        #
        # conv_pad: the K-1 zero columns causal_conv1d_silu prepends. bf16 + TILE to
        # match the bf16 mixed_qkv it concats with (same shape family as conv_state).
        self.conv_pad = ttnn.zeros(
            [self.batch_size, 1, self.conv_kernel_size - 1, self.conv_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # chunk_pad_*: the seq-dim tail that rounds the sequence up to a chunk_size
        # multiple in chunk_gated_delta_rule. fp32 + TILE to match q/k/v/beta/g after
        # their typecast. pad_size depends on the runtime seq_len, so we allocate at the
        # full chunk_size width and slice the first pad_size columns at use. One qkv
        # tail serves query/key/value because head_k_dim == head_v_dim (asserted); if
        # they ever diverge, split into a head_k_dim tail and a head_v_dim tail.
        assert self.head_k_dim == self.head_v_dim, (
            "chunk_pad_qkv assumes head_k_dim == head_v_dim so one zero tail serves "
            f"q/k/v; got {self.head_k_dim} vs {self.head_v_dim} — split into per-dim tails"
        )
        self.chunk_pad_qkv = ttnn.zeros(
            [self.batch_size, self.num_v_heads, self.chunk_size, self.head_v_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        self.chunk_pad_bg = ttnn.zeros(
            [self.batch_size, self.num_v_heads, self.chunk_size],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

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
        # Pad the seq dim (dim 2) up to a chunk_size multiple. ttnn.pad(value=0.0) does
        # this in one op, but it materializes a host constant into a fresh buffer — a
        # host-side write that ttnn trace capture forbids, and it issues that write even
        # when pad_size == 0. So when the sequence is already chunk-aligned we skip the
        # op entirely, and otherwise we concat a pad_size-wide slice of the persistent
        # zero tails built in initialize_params_gated_delta_rule. That leaves only a
        # device-side slice + concat in the traced graph. seq_len is tile-aligned (a
        # multiple of 32), so pad_size is one of {0, 32, 64, 96} — always a whole number
        # of tiles, which keeps the concat on the seq (tile) dim well-formed.
        if pad_size > 0:
            assert sequence_length % 32 == 0, (
                f"seq_len {sequence_length} must be tile-aligned (a multiple of 32) for the "
                "trace-safe pad concat, else pad_size straddles a partial tile"
            )
            qkv_tail = self.chunk_pad_qkv[:, :, :pad_size, :]  # [B, H, pad_size, head_dim]
            bg_tail = self.chunk_pad_bg[:, :, :pad_size]  # [B, H, pad_size]
            query = ttnn.concat([query, qkv_tail], dim=2)
            key = ttnn.concat([key, qkv_tail], dim=2)
            value = ttnn.concat([value, qkv_tail], dim=2)
            beta = ttnn.concat([beta, bg_tail], dim=2)
            g = ttnn.concat([g, bg_tail], dim=2)

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

    def recurrent_gated_delta_rule(self, query, key, value, g, beta, initial_state=None, use_qk_l2norm_in_kernel=False):
        """Single-token (decode) recurrent gated-delta scan — torch_recurrent_gated_delta_rule.

        The decode-time sibling of chunk_gated_delta_rule: rather than the chunked
        parallel form, it walks the sequence one position at a time, carrying the
        [B, num_v_heads, head_k_dim, head_v_dim] recurrent (KV) state forward and
        reading the output off it. seq_len is 1 in decode so the loop runs once, but
        it is written for the general case so a short prefill could reuse it.

        CRITICAL ordering (this is the easy bug): the Qwen3.5 reference DECAYS the
        state *before* the read — kv_mem is read off the already-decayed state, not
        the previous step's state. The experimental GDN's decode op
        (ttnn_delta_rule_ops.recurrent_gated_delta_rule_decode_ttnn) uses the FLA
        convention (read first, decay during the write), which is a *different*
        recurrence and would silently diverge here. We follow
        modeling_qwen3_5.py:340-351 exactly.
        """
        initial_dtype = query.dtype
        # l2norm runs on the head-last layout, in the input dtype, BEFORE the
        # transpose — identical to chunk_gated_delta_rule and the torch ref.
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)

        # heads-before-seq, then promote to fp32. q/k/v/beta arrive bf16 (g is
        # already fp32). fp32 is not optional here: the state recurrence compounds
        # every step and decay = exp(g) sits near 1.0, where bf16's ~0.008
        # resolution would quantize the per-step forgetting and accumulate error.
        query, key, value, beta, g = [ttnn.transpose(x, 1, 2) for x in (query, key, value, beta, g)]
        query, key, value, beta, g = [ttnn.typecast(x, ttnn.float32) for x in (query, key, value, beta, g)]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        query = query * self.scale

        # fp32 + HiFi2 + fp32 accumulation on the tiny per-step matmuls (same
        # rationale as the chunk path's chunk_math_kernel_config).
        step_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Carried recurrent state [B, H, head_k_dim, head_v_dim]. Decode continues
        # from the prefill-populated state (passed in); a None initial_state resets
        # to zero (torch line 334-338). We never mutate initial_state in place — the
        # first multiply below allocates a fresh tensor.
        last_recurrent_state = (
            ttnn.typecast(initial_state, ttnn.float32) if initial_state is not None else self.reset_recurrent_state()
        )

        # TT-NN has no per-position scatter (core_attn_out[:, :, i] = ...), so each
        # step's [B, H, 1, v_head_dim] slice is collected and concatenated after.
        core_attn_out_steps = []
        for i in range(sequence_length):
            # [B, H, 1, D] row slices. seq is a tile dim, so a non-aligned slice can
            # drop tile layout (cheap no-op when seq==1); re-tilize before the matmuls.
            q_i = ttnn.to_layout(query[:, :, i : i + 1, :], ttnn.TILE_LAYOUT)
            k_i = ttnn.to_layout(key[:, :, i : i + 1, :], ttnn.TILE_LAYOUT)
            v_i = ttnn.to_layout(value[:, :, i : i + 1, :], ttnn.TILE_LAYOUT)
            # g_i / beta_i as [B, H, 1, 1] to broadcast over the [k, v] state plane.
            g_i = ttnn.reshape(ttnn.exp(g[:, :, i : i + 1]), [batch_size, num_heads, 1, 1])
            beta_i = ttnn.reshape(beta[:, :, i : i + 1], [batch_size, num_heads, 1, 1])

            # 1) decay the state FIRST (torch line 347).
            last_recurrent_state = ttnn.multiply(last_recurrent_state, g_i)
            # 2) read the decayed state: kv_mem = k_i @ state — [B,H,1,Dk] @ [B,H,Dk,Dv]
            #    = [B,H,1,Dv]. The row-vector matmul IS torch's
            #    (state * k.unsqueeze(-1)).sum(-2): a weighted sum over the head_k axis.
            kv_mem = ttnn.matmul(k_i, last_recurrent_state, compute_kernel_config=step_kernel_config)
            # 3) prediction error, gated by beta (torch line 349).
            delta = ttnn.multiply(ttnn.subtract(v_i, kv_mem), beta_i)
            # 4) rank-1 write: state += k_i^T @ delta — [B,H,Dk,1] @ [B,H,1,Dv] is the
            #    outer product torch writes as k.unsqueeze(-1) * delta.unsqueeze(-2).
            #    k_i has a singleton seq dim, so reshape (not transpose) yields the column.
            k_col = ttnn.reshape(k_i, [batch_size, num_heads, k_head_dim, 1])
            outer = ttnn.matmul(k_col, delta, compute_kernel_config=step_kernel_config)
            last_recurrent_state = ttnn.add(last_recurrent_state, outer)
            # 5) read the output with the scaled query: o_i = q_i @ state (torch line 351).
            o_i = ttnn.matmul(q_i, last_recurrent_state, compute_kernel_config=step_kernel_config)
            core_attn_out_steps.append(o_i)

        # Stitch the per-step slices back into [B, H, seq, Dv], undo the
        # heads-before-seq transpose, and restore the caller's dtype (torch 355).
        core_attn_out = (
            core_attn_out_steps[0] if len(core_attn_out_steps) == 1 else ttnn.concat(core_attn_out_steps, dim=2)
        )
        core_attn_out = ttnn.transpose(core_attn_out, 1, 2)
        core_attn_out = ttnn.typecast(core_attn_out, initial_dtype)

        # forward_decode unpacks this 2-tuple and copies the state back into the
        # persistent buffer for the next step (mirrors chunk_gated_delta_rule).
        return core_attn_out, last_recurrent_state

    def forward_prefill(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B=1, 1, seq_len, hidden_size]
        """
        # TODO attention masking needs to be added

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
        # we need to initialize the conv state
        # in transformers, you might have to pad mixed_qkv if it is smaller than the conv kernel
        # but, you can also reshape down if the seq_len > conv kernel
        # in prefill, we will just slice down, but we might need to pad up...
        conv_state = mixed_qkv[..., :, -self.conv_kernel_size :, :]
        # The seq slice lands off a tile boundary (seq is a tile dim), which can drop
        # tile layout; re-tilize before the copy so the persistent conv-state buffer
        # decode reads back verbatim isn't corrupted by stale tile padding.
        conv_state = ttnn.to_layout(conv_state, ttnn.TILE_LAYOUT)
        ttnn.copy(conv_state, self.conv_state)

        # causal_conv1d_silu's contract is a canonical [B, T, D]; drop the singleton
        # dim 1 that ttnn.linear leaves on. In tile layout this is a metadata-only
        # reshape (the last two dims, which carry the tiles, are untouched).
        mixed_qkv = ttnn.reshape(mixed_qkv, (batch_size, 1, seq_len, self.conv_dim))
        mixed_qkv = causal_conv1d_silu(
            x=mixed_qkv,
            weight_taps=weights.w_taps,
            kernel_size=self.conv_kernel_size,
            mesh_device=self.mesh_device,
            pad=self.conv_pad,  # persistent zero pad so the conv stays trace-capturable
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
        out = ttnn.reshape(out, (batch_size, 1, out.shape[-2], out.shape[-1]))
        return tt_all_reduce(
            out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, hidden_states, attention_mask=None):
        # hidden_states [1, 1, B, hidden_size]
        weights = self.weights
        batch_size, seq_len = hidden_states.shape[2], 1

        # TODO attention masking given mask
        conv_state = self.conv_state
        recurrent_state = self.last_recurrent_state

        mixed_qkv = ttnn.linear(hidden_states, weights.wqkv)
        z = ttnn.linear(hidden_states, weights.wz)
        z = ttnn.reshape(z, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        b = ttnn.linear(hidden_states, weights.wb)
        a = ttnn.linear(hidden_states, weights.wa, dtype=ttnn.float32)

        mixed_qkv = ttnn.reshape(mixed_qkv, (batch_size, 1, seq_len, self.conv_dim))
        # causal_conv1d_silu_update returns (out, new_state): the conv output AND the
        # rolled-forward window. Unlike the torch ref (which mutates conv_state in
        # place), we must copy the new window back into the persistent buffer so the
        # NEXT decode step reads the right K-tap context.
        mixed_qkv, new_conv_state = causal_conv1d_silu_update(
            x=mixed_qkv,
            conv_state=conv_state,
            weight_taps=weights.w_taps,
            kernel_size=self.conv_kernel_size,
        )
        ttnn.copy(new_conv_state, self.conv_state)
        query, key, value = ttnn.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        # Decode lays the batch on dim 2 ([1, 1, B, *]), so b/a come out [1, 1, B, Hv].
        # A squeeze(1) would leave [1, B, Hv] — batch and seq swapped relative to the
        # [B, seq=1, Hv] that query/value use and that recurrent_gated_delta_rule's
        # transpose(1,2) expects. Reshape to [B, 1, Hv] instead so all five tensors
        # share the same (batch, seq, head) ordering.
        beta = ttnn.reshape(ttnn.sigmoid(b), (batch_size, seq_len, self.num_v_heads))
        g = ttnn.reshape(
            weights.neg_A_log_exp * ttnn.softplus(a + weights.dt_bias), (batch_size, seq_len, self.num_v_heads)
        )
        if self.num_v_heads // self.num_k_heads > 1:
            query = ttnn.repeat_interleave(query, self.num_v_heads // self.num_k_heads, dim=2)
            key = ttnn.repeat_interleave(key, self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            use_qk_l2norm_in_kernel=True,
        )
        ## update the recurrent state
        ttnn.copy(last_recurrent_state, self.last_recurrent_state)

        core_attn_out = ttnn.reshape(core_attn_out, (-1, self.head_v_dim))
        z = ttnn.reshape(z, (-1, self.head_v_dim))

        core_attn_out = self.norm(core_attn_out, gate=z)
        core_attn_out = ttnn.reshape(core_attn_out, (batch_size, seq_len, -1))
        out = ttnn.linear(core_attn_out, weights.wo)
        out = ttnn.reshape(out, (1, 1, batch_size, out.shape[-1]))
        return tt_all_reduce(
            out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
