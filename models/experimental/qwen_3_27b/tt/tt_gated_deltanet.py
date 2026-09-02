# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet -- linear attention with a gated delta rule.

The important structural difference from softmax attention: instead of a KV cache
that grows with sequence length, this layer carries a FIXED-SIZE recurrent state
[B, 48, 128, 128] plus a small conv state. Memory per token is O(1).

Two algorithms compute the same thing:
  * prefill (T > 1): chunked scan over the sequence
  * decode  (T = 1): single recurrent step
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

D = 5120  # hidden size
NUM_K_HEADS = 16  # query/key heads
NUM_V_HEADS = 48  # value heads
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM  # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 10240 -- q || k || v
CONV_KERNEL = 4
V_PER_K = NUM_V_HEADS // NUM_K_HEADS  # 3 value heads share each key head
L2_EPS = 1e-6  # eps inside the q/k L2 norm (FLA convention)
NORM_EPS = 1e-6  # rms_norm_eps, for the gated output norm
CHUNK = 64  # tokens per chunk in the chunked scan


class TtQwen36GatedDeltaNet(LightweightModule):
    """
    Gated DeltaNet linear attention. 48 of the 64 layers.

    Weights as received. Projections are only partly fused: qkv together,
    z / a / b separate.
        in_proj_qkv    [5120, 10240]   = 2048 + 2048 + 6144  (q, k, v)
        in_proj_z      [5120,  6144]   output gate
        in_proj_b      [5120,    48]   -> beta,  one per value head
        in_proj_a      [5120,    48]   -> alpha, one per value head
        conv1d_weight  [1, 4, 10240]   depthwise, NO bias (permuted at load)
        neg_A [1,1,48]  = -exp(A_log)      dt_bias [1,1,48]
        norm_weight [128]                  out_proj [6144, 5120]

    Pipeline (x [B,T,5120] -> out [B,T,5120]):
        1. project qkv / z / b / a                              _project
        2. depthwise causal conv1d + SiLU on the 10240-wide qkv  _causal_conv1d
        3. split [2048,2048,6144], reshape to heads, and         _project
           repeat_interleave q,k 3x -> 48 heads (inverse of GQA)
        4. L2-normalize q,k; scale q by 1/sqrt(128)             _l2norm_qk
        5. beta = sigmoid(b), g = neg_A * softplus(a+dt_bias)   _gates
        6. delta rule                              _delta_rule_recurrent
           (chunked form still to come -- _chunk_decay is its first piece)
        7. gated norm: normalize o, scale, THEN * silu(z)       _gated_norm
        8. out_proj                                                forward

    conv_state and recurrent_state are held here and advanced by every forward,
    so calls chain into one sequence. reset_state() starts a new one.
    """

    def __init__(
        self,
        device,
        in_proj_qkv: ttnn.Tensor,
        in_proj_z: ttnn.Tensor,
        in_proj_b: ttnn.Tensor,
        in_proj_a: ttnn.Tensor,
        conv1d_weight: ttnn.Tensor,
        neg_A: ttnn.Tensor,
        dt_bias: ttnn.Tensor,
        norm_weight: ttnn.Tensor,
        out_proj: ttnn.Tensor,
        batch: int = 1,
        layer_idx: int = 0,
    ):
        """
        Weights are handed in ALREADY ON DEVICE and shaped in [in, out] layout.
        This module does no loading -- that is one concern, done once, elsewhere.

        Two are not verbatim checkpoint tensors: neg_A is -exp(A_log), and
        conv1d_weight arrives permuted to [1, 4, 10240].
        """
        self.device = device
        self.layer_idx = layer_idx

        self.in_proj_qkv = in_proj_qkv  # [5120, 10240]
        self.in_proj_z = in_proj_z  # [5120,  6144]
        self.in_proj_b = in_proj_b  # [5120,    48]
        self.in_proj_a = in_proj_a  # [5120,    48]

        # A depthwise conv with K=4 is 4 broadcast multiplies of shifted copies of
        # the input, so split [1, 4, 10240] into one [1, 1, 10240] weight vector
        # per kernel position, each broadcasting over [B, T, 10240].
        self.conv_weights = [ttnn.slice(conv1d_weight, (0, i, 0), (1, i + 1, CONV_DIM)) for i in range(CONV_KERNEL)]

        # Both [1, 1, 48], float32, broadcast over [B, T, 48] -- one learned value
        # per HEAD, so all 48 heads forget their state at their own rate.
        #   neg_A   = -exp(A_log) < 0, the head's decay rate. Folded on host.
        #   dt_bias = the head's default timestep, added before the softplus.
        self.neg_A = neg_A
        self.dt_bias = dt_bias

        self.norm_weight = norm_weight  # [128], PLAIN -- no +1 fold, unlike TtQwen36RmsNorm
        self.out_proj = out_proj  # [6144, 5120]

        # Constants, not checkpoint data -- see _l2norm_qk.
        self.l2_eps = L2_EPS / HEAD_K_DIM
        self.q_l2_weight = ttnn.full(
            [HEAD_K_DIM], HEAD_K_DIM**-1.0, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.k_l2_weight = ttnn.full(
            [HEAD_K_DIM], HEAD_K_DIM**-0.5, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        # Chunk masks, all [1, 64, 64] fp32. Depend only on CHUNK, so host-built once.
        ones = torch.ones(CHUNK, CHUNK)

        def _const(t):
            return ttnn.from_torch(
                t.reshape(1, CHUNK, CHUNK), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )

        self.tril_incl = _const(torch.tril(ones))  # i >= j, decay mask
        self.tril_strict = _const(torch.tril(ones, -1))  # i >  j, A has no self-interaction
        self.eye = _const(torch.eye(CHUNK))

        self.batch = batch
        self.reset_state()

    @staticmethod
    def state_shapes(batch_size: int):
        """
        State carried between calls.

        recurrent  [B, 48, 128, 128]  one key->value matrix per value head --
                   the whole memory. Size is independent of T.
        conv       [B, 3, 10240]      previous call's last K-1 rows of mixed_qkv,
                   the causal conv's left context.
        """
        return {
            "recurrent": (batch_size, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
            "conv": (batch_size, CONV_KERNEL - 1, CONV_DIM),
        }

    def reset_state(self):
        """Zero the carried state. Call before starting a new sequence."""
        shapes = self.state_shapes(self.batch)
        self.recurrent_state = ttnn.zeros(
            shapes["recurrent"], device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )
        self.conv_state = ttnn.zeros(shapes["conv"], device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _causal_conv1d(self, mixed_qkv: ttnn.Tensor):
        """
        STEP 2: depthwise causal conv (kernel 4) + SiLU.  [B, T, C] -> [B, T, C]

            out[b, t, c] = sum_i  w[i, c] * x[b, t-3+i, c]

        Depthwise: each channel has its own length-4 filter and never sees another
        channel, so the only mixing is across time. Unrolled into 4 shifted slices
        and a weighted sum -- no matmul, the weights are per-channel scalars.

        self.conv_state is the left context: the previous call's last K-1 rows,
        zeros at the start of a sequence.
        """
        batch, seq_len, _ = mixed_qkv.shape  # mixed_qkv [B, T, 10240]

        padded = ttnn.concat([self.conv_state, mixed_qkv], dim=1)  # [B, 3+T, 10240]

        # Next call's left context: the last 3 rows of `padded`. Still conv INPUT,
        # not output, and taking it from `padded` keeps it correct when T < 3.
        self.conv_state = ttnn.slice(
            padded, (0, seq_len, 0), (batch, seq_len + CONV_KERNEL - 1, CONV_DIM)
        )  # [B, 3, 10240]

        out = None
        for i, w in enumerate(self.conv_weights):  # w [1, 1, 10240]
            # window i holds x[t-3+i] at position t: i=0 -> x[t-3] ... i=3 -> x[t]
            window = ttnn.slice(padded, (0, i, 0), (batch, i + seq_len, CONV_DIM))  # [B, T, 10240]
            term = ttnn.multiply(window, w)  # [B,T,10240] * [1,1,10240] -> [B, T, 10240]
            ttnn.deallocate(window)
            if out is None:
                out = term  # [B, T, 10240]
            else:
                out = ttnn.add(out, term)  # [B, T, 10240]
                ttnn.deallocate(term)
        ttnn.deallocate(padded)

        return ttnn.silu(out)  # [B, T, 10240]

    def _gates(self, b: ttnn.Tensor, a: ttnn.Tensor):
        """
        STEP 3: the two per-head scalars that steer the memory. [B,T,48] each.

            beta = sigmoid(b)                   write strength, in (0, 1)
            g    = neg_A * softplus(a+dt_bias)  log decay, < 0 (used as exp(g))

        Where each term comes from:
            b, a       per-token, from the in_proj_b / in_proj_a matmuls
            dt_bias    LEARNED per-head param [48], straight from the checkpoint
            neg_A      LEARNED per-head param [48]: -exp(A_log), folded on host
                       (checkpoint stores A_log; exp() makes it positive, so
                       neg_A < 0, which is what forces exp(g) into (0,1))

        float32 throughout: exp(g) is applied once per token, so it compounds
        over the whole sequence and bf16 would drift by T=4096.
        """
        b32 = ttnn.typecast(b, ttnn.float32)
        beta = ttnn.sigmoid(b32)
        ttnn.deallocate(b32)

        g = ttnn.typecast(a, ttnn.float32)
        g = ttnn.add(g, self.dt_bias)  # broadcast [1,1,48]
        g = ttnn.softplus(g)
        g = ttnn.multiply(g, self.neg_A)  # broadcast [1,1,48]

        return beta, g

    def _l2norm_qk(self, q: ttnn.Tensor, k: ttnn.Tensor):
        """
        STEP 4: L2-normalize q and k over head_k_dim, and scale q by 1/sqrt(128).


            l2norm(x)         = x * rsqrt(sum(x^2) + eps)      SUM, not mean

        which is just an RMS norm with the constants folded in:
            l2norm(x)         = rms_norm(x, w=1/sqrt(d), eps=eps/d)
            l2norm(x)/sqrt(d) = rms_norm(x, w=1/d,       eps=eps/d)   <- q
        so each is one fused op instead of multiply/sum/add/rsqrt/multiply.
        """
        q = ttnn.rms_norm(q, weight=self.q_l2_weight, epsilon=self.l2_eps)  # [B,T,48,128]
        k = ttnn.rms_norm(k, weight=self.k_l2_weight, epsilon=self.l2_eps)  # [B,T,48,128]
        return q, k

    def _delta_rule_recurrent(self, q, k, v, beta, g):
        """
        STEP 6, reference form -- the delta rule, one token at a time:

            S   <- S * exp(g_t)         decay: forget a fraction of everything
            m    = S^T k_t              what memory currently returns for k_t
            d    = (v_t - m) * beta_t   prediction error, scaled by write strength
            S   <- S + k_t (x) d        write the correction
            o_t  = S^T q_t              read out with the query

        Sequential in T, so this is really the decode algorithm -- prefill wants
        the chunked form. Kept as the definition to validate that against.

        q,k,v [B,T,48,128], beta,g [B,T,48] -> o [B,T,48,128].
        S is self.recurrent_state, updated in place.
        """
        batch, seq_len = q.shape[0], q.shape[1]
        heads = NUM_V_HEADS

        # token-major -> head-major, fp32 for the recurrence
        q = ttnn.typecast(ttnn.permute(q, (0, 2, 1, 3)), ttnn.float32)  # [B,H,T,128]
        k = ttnn.typecast(ttnn.permute(k, (0, 2, 1, 3)), ttnn.float32)  # [B,H,T,128]
        v = ttnn.typecast(ttnn.permute(v, (0, 2, 1, 3)), ttnn.float32)  # [B,H,T,128]
        beta = ttnn.permute(beta, (0, 2, 1))  # [B,H,T], already fp32
        g = ttnn.permute(g, (0, 2, 1))  # [B,H,T]

        state = self.recurrent_state  # [B,48,128,128]

        outputs = []
        for t in range(seq_len):
            q_t = ttnn.slice(q, (0, 0, t, 0), (batch, heads, t + 1, HEAD_K_DIM))  # [B,H,1,128]
            k_t = ttnn.slice(k, (0, 0, t, 0), (batch, heads, t + 1, HEAD_K_DIM))  # [B,H,1,128]
            v_t = ttnn.slice(v, (0, 0, t, 0), (batch, heads, t + 1, HEAD_V_DIM))  # [B,H,1,128]
            g_t = ttnn.reshape(ttnn.slice(g, (0, 0, t), (batch, heads, t + 1)), (batch, heads, 1, 1))
            beta_t = ttnn.reshape(ttnn.slice(beta, (0, 0, t), (batch, heads, t + 1)), (batch, heads, 1, 1))

            decay = ttnn.exp(g_t)  # [B,H,1,1] in (0,1)
            ttnn.multiply_(state, decay)  # in-place so the loop does not churn [B,H,128,128]

            mem = ttnn.matmul(k_t, state)  # [B,H,1,128] @ [B,H,128,128] -> [B,H,1,128]
            delta = ttnn.multiply(ttnn.subtract(v_t, mem), beta_t)  # [B,H,1,128]
            k_col = ttnn.permute(k_t, (0, 1, 3, 2))  # [B,H,128,1]
            ttnn.add_(state, ttnn.matmul(k_col, delta))  # outer product, in-place write

            outputs.append(ttnn.matmul(q_t, state))  # [B,H,1,128]

            for tmp in (q_t, k_t, v_t, g_t, beta_t, decay, mem, delta, k_col):
                ttnn.deallocate(tmp)

        o = ttnn.concat(outputs, dim=2)  # [B,H,T,128]
        return ttnn.permute(o, (0, 2, 1, 3))  # [B,T,48,128]

    def _chunk_decay(self, g_chunks: ttnn.Tensor):
        """
        STEP 6a: within-chunk cumulative decay.  g [N, 1, 64] -> G, decay [N, 64, 64]

            G_i          = g_0 + ... + g_i          running log-decay from chunk start
            decay[i, j]  = exp(G_i - G_j)  for i >= j, else 0

        decay[i,j] is the fraction of a write made at j that survives to i.
        """
        G = ttnn.cumsum(g_chunks, dim=-1)  # [N,1,64]

        G_col = ttnn.permute(G, (0, 2, 1))  # [N,64,1], holds G_i down the rows
        diff = ttnn.subtract(G_col, G)  # [N,64,64], diff[i,j] = G_i - G_j
        ttnn.deallocate(G_col)

        # Mask BEFORE exp: above the diagonal G_i - G_j > 0 and would overflow.
        diff = ttnn.multiply(diff, self.tril_incl)  # [N,64,64]
        decay = ttnn.multiply(ttnn.exp(diff), self.tril_incl)  # re-mask: exp(0)=1 up there
        ttnn.deallocate(diff)

        return G, decay

    def _project(self, x: ttnn.Tensor):
        """
        STEPS 1-2: hidden state -> the six head-shaped tensors the delta rule needs.

        [B, T, 5120] ->
            q, k  [B, T, 48, 128]   (16 heads projected, then repeated 3x)
            v, z  [B, T, 48, 128]
            b, a  [B, T, 48]

        The conv runs on the full 10240-wide mixed_qkv BEFORE the split. Since it
        is depthwise that is purely an efficiency choice -- one op instead of
        three -- and is mathematically identical to convolving q, k, v separately.
        """
        batch, seq_len, _ = x.shape

        mixed_qkv = ttnn.linear(x, self.in_proj_qkv)  # [B, T, 10240]
        mixed_qkv = self._causal_conv1d(mixed_qkv)  # conv + SiLU

        # Plain contiguous split -- no per-head interleaving in Qwen3.5.
        q = ttnn.slice(mixed_qkv, [0, 0, 0], [batch, seq_len, KEY_DIM])
        k = ttnn.slice(mixed_qkv, [0, 0, KEY_DIM], [batch, seq_len, 2 * KEY_DIM])
        v = ttnn.slice(mixed_qkv, [0, 0, 2 * KEY_DIM], [batch, seq_len, CONV_DIM])
        ttnn.deallocate(mixed_qkv)

        q = ttnn.reshape(q, (batch, seq_len, NUM_K_HEADS, HEAD_K_DIM))
        k = ttnn.reshape(k, (batch, seq_len, NUM_K_HEADS, HEAD_K_DIM))
        v = ttnn.reshape(v, (batch, seq_len, NUM_V_HEADS, HEAD_V_DIM))

        # 16 key heads -> 48, so each key head is shared by 3 value heads.
        # This is the inverse of GQA: values outnumber keys here.
        q = ttnn.repeat_interleave(q, V_PER_K, dim=2)
        k = ttnn.repeat_interleave(k, V_PER_K, dim=2)

        z = ttnn.linear(x, self.in_proj_z)  # [B, T, 6144]
        z = ttnn.reshape(z, (batch, seq_len, NUM_V_HEADS, HEAD_V_DIM))

        b = ttnn.linear(x, self.in_proj_b)  # [B, T, 48]  -> beta  (write strength)
        a = ttnn.linear(x, self.in_proj_a)  # [B, T, 48]  -> alpha (decay)

        return q, k, v, z, b, a

    def _gated_norm(self, o: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        """
        STEP 7: gated RMSNorm over head_v_dim.  o, z [B,T,48,128] -> [B,T,48,128]

            out = (o * rsqrt(mean(o^2) + eps) * weight) * silu(z)

        Two things differ from TtQwen36RmsNorm and both are silent if wrong:
        the weight is PLAIN (initialized to ones, no +1 fold), and the norm
        happens BEFORE the gate -- gating first would feed z into the variance.
        """
        normed = ttnn.rms_norm(o, weight=self.norm_weight, epsilon=NORM_EPS)  # [B,T,48,128]
        gate = ttnn.silu(z)  # [B,T,48,128]
        out = ttnn.multiply(normed, gate)  # [B,T,48,128]
        ttnn.deallocate(normed)
        ttnn.deallocate(gate)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        [B, T, 5120] -> [B, T, 5120].

        Prefill path, recurrent delta rule. Consumes and advances the carried
        state, so consecutive calls continue one sequence -- reset_state() starts
        a new one.
        """
        batch, seq_len, _ = x.shape

        # q,k,v,z [B,T,48,128]  b,a [B,T,48]
        q, k, v, z, b, a = self._project(x)

        beta, g = self._gates(b, a)  # step 5 -> [B,T,48] fp32 each
        ttnn.deallocate(b)
        ttnn.deallocate(a)

        q, k = self._l2norm_qk(q, k)  # step 4 -> [B,T,48,128]

        o = self._delta_rule_recurrent(q, k, v, beta, g)  # step 6 -> [B,T,48,128]
        for tmp in (q, k, v, beta, g):
            ttnn.deallocate(tmp)

        o = self._gated_norm(o, z)  # step 7 -> [B,T,48,128]
        ttnn.deallocate(z)

        o = ttnn.reshape(o, (batch, seq_len, VALUE_DIM))  # [B,T,48,128] -> [B,T,6144]
        return ttnn.linear(o, self.out_proj)  # step 8: @ [6144,5120] -> [B,T,5120]
