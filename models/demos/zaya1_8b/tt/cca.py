"""Phase 3: Compressed Convolutional Attention (CCA) for ZAYA1-8B (prefill, B=1).

The two causal depthwise/grouped Conv1d (kernel=2) layers compose into a causal
3-tap mixing that is EXACTLY equivalent to:
    conv_out = qk @ Cm^T + shift1(qk) @ Bm^T + shift2(qk) @ Am^T + bias_total
with Am/Bm/Cm [1280,1280] block-diagonal (10 groups x 128) precomputed on host
from the conv weights. shiftN shifts the sequence down by N (causal, zero-filled).

Pipeline (mirrors HF CCA.forward, no-cache prefill path):
  q=linear_q(h); k=linear_k(h); qk=cat(q,k)
  conv_out = conv_equiv(qk)
  qk_mean_q = (q_heads + repeat(k_heads))/2 ; qk_mean_k = mean over GQA groups
  query = conv_out[:1024] + qk_mean_q ; key = conv_out[1024:] + qk_mean_k
  L2-normalize per head * sqrt(head_dim); key also * temp[kv_head]
  v1=val_proj1(h); v2=val_proj2(shift1(h)); value=cat(v1,v2)
Then ZayaAttention: partial RoPE(64) -> GQA SDPA(causal) -> o_proj.
"""
import torch
import ttnn

from .model_args import ZayaConfig
from .standard import to_dev

C = ZayaConfig
SQRT_HD = float(C.head_dim ** 0.5)


# ----------------------------------------------------------------------------
# host: build conv-equivalent matrices and shift matrices
# ----------------------------------------------------------------------------
def build_conv_equiv(conv0_w, conv0_b, conv1_w, conv1_b):
    """conv0_w [1280,1,2] depthwise; conv1_w [1280,128,2] grouped(10). Returns Am,Bm,Cm [1280,1280], bias [1280]."""
    ch = conv0_w.shape[0]                    # 1280
    G, gs = 10, ch // 10                     # 10 groups of 128
    w0 = conv0_w[:, 0, :].float()            # [1280,2]
    W1 = conv1_w.float()                     # [1280,128,2]
    Am = torch.zeros(ch, ch); Bm = torch.zeros(ch, ch); Cm = torch.zeros(ch, ch)
    bias = conv1_b.float().clone()
    for g in range(G):
        cs = slice(g * gs, (g + 1) * gs)     # output channels of group g
        w0g = w0[cs]                         # [128,2] (input ch == output ch range for depthwise group)
        W1_0 = W1[cs, :, 0]                  # [128 out, 128 in]
        W1_1 = W1[cs, :, 1]
        Am[cs, cs] = W1_0 * w0g[:, 0][None, :]
        Bm[cs, cs] = W1_0 * w0g[:, 1][None, :] + W1_1 * w0g[:, 0][None, :]
        Cm[cs, cs] = W1_1 * w0g[:, 1][None, :]
        bias[cs] += (W1_0 + W1_1) @ conv0_b.float()[cs]
    return Am, Bm, Cm, bias


def shift_matrix(seq_len, k):
    """[S,S] matrix M with M[t, t-k]=1 (causal down-shift by k)."""
    m = torch.zeros(seq_len, seq_len)
    for t in range(k, seq_len):
        m[t, t - k] = 1.0
    return m


def rotate_half_torch(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def to_heads(x, n_heads):
    """[1,S,n_heads*128] -> [1,n_heads,S,128] (slice+unsqueeze+concat; keeps 128 as last dim)."""
    S = x.shape[1]
    hd = C.head_dim
    hs = [ttnn.unsqueeze(ttnn.slice(x, [0, 0, h * hd], [1, S, (h + 1) * hd]), 1) for h in range(n_heads)]
    return ttnn.concat(hs, dim=1)


def from_heads(x, n_heads):
    """[1,n_heads,S,128] -> [1,S,n_heads*128]."""
    S = x.shape[2]
    hs = [ttnn.squeeze(ttnn.slice(x, [0, h, 0, 0], [1, h + 1, S, C.head_dim]), 1) for h in range(n_heads)]
    return ttnn.concat(hs, dim=-1)


# ----------------------------------------------------------------------------
# CCA QKV
# ----------------------------------------------------------------------------
class CCAQKV:
    def __init__(self, device, w, layer, seq_len):
        self.device = device
        self.seq_len = seq_len
        p = f"model.layers.{layer}.self_attn.qkv"
        self.lq = to_dev(w.get(f"{p}.linear_q.weight").t().contiguous(), device)   # [2048,1024]
        self.lk = to_dev(w.get(f"{p}.linear_k.weight").t().contiguous(), device)   # [2048,256]
        self.v1 = to_dev(w.get(f"{p}.val_proj1.weight").t().contiguous(), device)  # [2048,128]
        self.v2 = to_dev(w.get(f"{p}.val_proj2.weight").t().contiguous(), device)
        Am, Bm, Cm, bias = build_conv_equiv(
            w.get(f"{p}.conv_qk.0.weight"), w.get(f"{p}.conv_qk.0.bias"),
            w.get(f"{p}.conv_qk.1.weight"), w.get(f"{p}.conv_qk.1.bias"))
        # conv_out = qk@Cm^T + sh1(qk)@Bm^T + sh2(qk)@Am^T + bias  -> store transposed for ttnn.linear
        self.Cm = to_dev(Cm.t().contiguous(), device)
        self.Bm = to_dev(Bm.t().contiguous(), device)
        self.Am = to_dev(Am.t().contiguous(), device)
        self.conv_bias = to_dev(bias.reshape(1, -1), device)
        self.temp = w.get(f"{p}.temp").float()  # [2] kv-head temperature (applied host-side broadcast)
        self.temp_dev = to_dev(self.temp.repeat_interleave(C.head_dim).reshape(1, 1, -1), device)  # [1,1,256]
        self.temp_heads = to_dev(self.temp.reshape(1, C.n_kv_heads, 1, 1), device)  # [1,2,1,1] head-layout broadcast
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def _shift_seq(self, x, k):
        """Causal down-shift along seq by k (zero-fill) for [1,S,C], tile-native
        (slice + concat along seq, no row-major round-trip)."""
        S, ch = x.shape[1], x.shape[2]
        keep = ttnn.slice(x, [0, 0, 0], [1, S - k, ch])                   # first S-k rows
        z = ttnn.zeros([1, k, ch], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return ttnn.concat([z, keep], dim=1)                              # prepend k zero rows

    def _per_head_l2_scale(self, x, n_heads):
        """x [1,S,n_heads*128] -> normalized per head * sqrt(head_dim). Slice-based
        (avoid making n_heads a tile dim via reshape)."""
        S = x.shape[1]
        hd = C.head_dim
        parts = []
        for h in range(n_heads):
            xh = ttnn.slice(x, [0, 0, h * hd], [1, S, (h + 1) * hd])    # [1,S,128]
            sq = ttnn.sum(ttnn.mul(xh, xh), dim=-1, keepdim=True)       # [1,S,1]
            scale = ttnn.multiply(ttnn.rsqrt(sq), SQRT_HD)
            parts.append(ttnn.mul(xh, scale))
        return ttnn.concat(parts, dim=-1)

    def _l2_h(self, xh):
        """Per-head L2-normalize * sqrt(head_dim), VECTORIZED in head layout
        [1,n_heads,S,128] (one reduce over the last dim instead of an 8-head op loop).
        Pure reorder of _per_head_l2_scale -> token-exact."""
        sq = ttnn.sum(ttnn.mul(xh, xh), dim=-1, keepdim=True)              # [1,n,S,1]
        return ttnn.mul(xh, ttnn.multiply(ttnn.rsqrt(sq), SQRT_HD))        # broadcast over 128

    def _repeat_kv_heads(self, k, groups):
        """k [1,S,n_kv*128] -> [1,S,n_kv*groups*128] interleaved (kv0 x groups, kv1 x groups)."""
        S = k.shape[1]
        parts = []
        for kv in range(C.n_kv_heads):
            sl = ttnn.slice(k, [0, 0, kv * C.head_dim], [1, S, (kv + 1) * C.head_dim])
            parts.extend([sl] * groups)
        return ttnn.concat(parts, dim=-1)

    def _assemble(self, conv, q, k, hidden, v2_src):
        """Build (query,key,value) from conv output, current q/k, the CCA-input hidden
        (for v1) and the v2 source (h_shift for prefill, prev_hs for decode). Generalises over S."""
        S = q.shape[1]
        hd = C.head_dim
        qd = C.cca_q_heads * hd                                            # 1024
        conv_q = ttnn.slice(conv, [0, 0, 0], [1, S, qd])
        conv_k = ttnn.slice(conv, [0, 0, qd], [1, S, C.cca_in_out_ch])
        groups = C.cca_q_heads // C.n_kv_heads                             # 4
        k_rep = self._repeat_kv_heads(k, groups)                           # [1,S,1024]
        mean_q = ttnn.multiply(ttnn.add(q, k_rep), 0.5)                    # [1,S,1024]
        mk_parts = []
        for kv in range(C.n_kv_heads):
            base = kv * groups * hd
            acc = ttnn.slice(mean_q, [0, 0, base], [1, S, base + hd])
            for i in range(1, groups):
                acc = ttnn.add(acc, ttnn.slice(mean_q, [0, 0, base + i * hd], [1, S, base + (i + 1) * hd]))
            mk_parts.append(ttnn.multiply(acc, 1.0 / groups))
        mean_k = ttnn.concat(mk_parts, dim=-1)                             # [1,S,256]
        # Return PRE-L2/PRE-temp (flat). Callers split to heads then L2/temp/rope in
        # head layout (vectorized _l2_h), which is far fewer ops and token-exact.
        q_pre = ttnn.add(conv_q, mean_q)                                   # [1,S,1024]
        k_pre = ttnn.add(conv_k, mean_k)                                   # [1,S,256]
        v1 = ttnn.linear(hidden, self.v1, compute_kernel_config=self.ckc)  # [1,S,128]
        v2 = ttnn.linear(v2_src, self.v2, compute_kernel_config=self.ckc)  # [1,S,128]
        value = ttnn.concat([v1, v2], dim=-1)                             # [1,S,256]
        return q_pre, k_pre, value

    def forward(self, hidden):
        """Prefill. hidden [1,S,2048]. Returns query,key,value (+qk [1,S,1280] for cache)."""
        q = ttnn.linear(hidden, self.lq, compute_kernel_config=self.ckc)   # [1,S,1024]
        k = ttnn.linear(hidden, self.lk, compute_kernel_config=self.ckc)   # [1,S,256]
        qk = ttnn.concat([q, k], dim=-1)                                   # [1,S,1280]
        sh1 = self._shift_seq(qk, 1)
        sh2 = self._shift_seq(qk, 2)
        conv = ttnn.linear(qk, self.Cm, compute_kernel_config=self.ckc)
        conv = ttnn.add(conv, ttnn.linear(sh1, self.Bm, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, ttnn.linear(sh2, self.Am, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, self.conv_bias)                              # [1,S,1280]
        query, key, value = self._assemble(conv, q, k, hidden, self._shift_seq(hidden, 1))
        return query, key, value, qk

    def decode_forward(self, h1, cache, layer):
        """Decode one token. h1 [1,1,2048]. Uses/updates cache.conv_state & prev_hs."""
        q = ttnn.linear(h1, self.lq, compute_kernel_config=self.ckc)       # [1,1,1024]
        k = ttnn.linear(h1, self.lk, compute_kernel_config=self.ckc)       # [1,1,256]
        qk = ttnn.concat([q, k], dim=-1)                                   # [1,1,1280]
        qk_prev2, qk_prev = cache.conv_state[layer]                        # [1,1,1280] each
        conv = ttnn.linear(qk, self.Cm, compute_kernel_config=self.ckc)
        conv = ttnn.add(conv, ttnn.linear(qk_prev, self.Bm, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, ttnn.linear(qk_prev2, self.Am, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, self.conv_bias)                              # [1,1,1280]
        cache.conv_state[layer] = (qk_prev, qk)                            # roll window
        v2_src = cache.prev_hs[layer]                                      # previous token hidden
        cache.prev_hs[layer] = h1
        return self._assemble(conv, q, k, h1, v2_src)


# ----------------------------------------------------------------------------
# CCA attention (rope + GQA SDPA + o_proj)
# ----------------------------------------------------------------------------
class CCAAttention:
    # Weights load once and are seq-agnostic; per-seq cos/sin/causal-mask are cached
    # (cheap), so one model build serves any sequence length across generation steps.
    def __init__(self, device, w, layer, seq_len=None):
        self.device = device
        self.qkv = CCAQKV(device, w, layer, seq_len)
        self.o_proj = to_dev(w.get(f"model.layers.{layer}.self_attn.o_proj.weight").t().contiguous(), device)
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.inv_sqrt_hd = 1.0 / SQRT_HD
        self._seqcache = {}

    def _seq(self, S):
        if S not in self._seqcache:
            from .standard import compute_cos_sin
            cos, sin = compute_cos_sin(S)
            cos_t = to_dev(cos.reshape(1, 1, S, C.rotary_dim), self.device, dtype=ttnn.bfloat16)
            sin_t = to_dev(sin.reshape(1, 1, S, C.rotary_dim), self.device, dtype=ttnn.bfloat16)
            m = torch.triu(torch.ones(S, S), diagonal=1) * (-1e30)
            mask = to_dev(m.reshape(1, 1, S, S), self.device, dtype=ttnn.float32)
            self._seqcache[S] = (cos_t, sin_t, mask)
        return self._seqcache[S]

    def _apply_rope(self, x, n_heads, cos, sin):
        """x [1,n_heads,S,128] -> partial rope on first 64 dims."""
        rd = C.rotary_dim
        S = x.shape[2]
        x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, n_heads, S, rd])
        x_pass = ttnn.slice(x, [0, 0, 0, rd], [1, n_heads, S, C.head_dim])
        half = rd // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [1, n_heads, S, half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [1, n_heads, S, rd])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)                      # rotate_half
        out_rot = ttnn.add(ttnn.mul(x_rot, cos), ttnn.mul(rot, sin))
        return ttnn.concat([out_rot, x_pass], dim=-1)

    def _seq_pos(self, pos):
        """cos/sin at a single absolute position, [1,1,1,rotary_dim]."""
        if ("p", pos) not in self._seqcache:
            from .standard import compute_cos_sin
            cos, sin = compute_cos_sin(pos + 1)
            ct = to_dev(cos[pos].reshape(1, 1, 1, C.rotary_dim), self.device, dtype=ttnn.bfloat16)
            st = to_dev(sin[pos].reshape(1, 1, 1, C.rotary_dim), self.device, dtype=ttnn.bfloat16)
            self._seqcache[("p", pos)] = (ct, st)
        return self._seqcache[("p", pos)]

    def forward(self, hidden, cache=None, layer=None):
        S = hidden.shape[1]
        cos, sin, causal_mask = self._seq(S)
        qpre, kpre, value, qk = self.qkv.forward(hidden)                   # pre-L2 [1,S,1024],[1,S,256],value[1,S,256]
        q = self._apply_rope(self.qkv._l2_h(to_heads(qpre, C.cca_q_heads)), C.cca_q_heads, cos, sin)  # [1,8,S,128]
        k = ttnn.mul(self.qkv._l2_h(to_heads(kpre, C.n_kv_heads)), self.qkv.temp_heads)
        v = to_heads(value, C.n_kv_heads)
        k = self._apply_rope(k, C.n_kv_heads, cos, sin)                    # roped K (stored in cache)
        if cache is not None:                                              # populate decode cache from prefill
            cache.conv_state[layer] = (ttnn.slice(qk, [0, S - 2, 0], [1, S - 1, C.cca_in_out_ch]),
                                       ttnn.slice(qk, [0, S - 1, 0], [1, S, C.cca_in_out_ch]))
            cache.prev_hs[layer] = ttnn.slice(hidden, [0, S - 1, 0], [1, S, C.dim])
            cache.k[layer] = k                                             # [1,2,S,128] roped
            cache.v[layer] = v
        groups = C.cca_q_heads // C.n_kv_heads
        kk = ttnn.repeat_interleave(k, groups, dim=1)
        vv = ttnn.repeat_interleave(v, groups, dim=1)
        scores = ttnn.matmul(q, ttnn.transpose(kk, -2, -1), compute_kernel_config=self.ckc)  # [1,8,S,S]
        scores = ttnn.multiply(ttnn.typecast(scores, ttnn.float32), self.inv_sqrt_hd)
        scores = ttnn.add(scores, causal_mask)
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1), ttnn.bfloat16)
        attn = ttnn.matmul(probs, vv, compute_kernel_config=self.ckc)                        # [1,8,S,128]
        attn = from_heads(attn, C.cca_q_heads)                             # [1,S,1024]
        return ttnn.linear(attn, self.o_proj, compute_kernel_config=self.ckc)          # [1,S,2048]

    def trace_decode(self, h1, ts, layer):
        """Trace-friendly decode: all state read/written in-place on persistent buffers
        in `ts` (TraceState); position carried via ts.onehot/inv/amask/cos/sin inputs.
        h1 [1,1,2048] -> [1,1,2048]."""
        q = ttnn.linear(h1, self.qkv.lq, compute_kernel_config=self.ckc)   # [1,1,1024]
        k = ttnn.linear(h1, self.qkv.lk, compute_kernel_config=self.ckc)   # [1,1,256]
        qk = ttnn.concat([q, k], dim=-1)                                   # [1,1,1280]
        conv = ttnn.linear(qk, self.qkv.Cm, compute_kernel_config=self.ckc)
        conv = ttnn.add(conv, ttnn.linear(ts.p1[layer], self.qkv.Bm, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, ttnn.linear(ts.p2[layer], self.qkv.Am, compute_kernel_config=self.ckc))
        conv = ttnn.add(conv, self.qkv.conv_bias)
        ttnn.copy(ts.p1[layer], ts.p2[layer])                              # p2 <- p1  (roll, before p1 overwrite)
        ttnn.copy(qk, ts.p1[layer])                                        # p1 <- qk
        qpre, kpre, value = self.qkv._assemble(conv, q, k, h1, ts.ph[layer])
        ttnn.copy(h1, ts.ph[layer])                                        # prev_hs <- current
        gq = C.cca_q_heads
        qh = self._apply_rope(self.qkv._l2_h(to_heads(qpre, gq)), gq, ts.cos, ts.sin)     # [1,8,1,128]
        kh = ttnn.mul(self.qkv._l2_h(to_heads(kpre, C.n_kv_heads)), self.qkv.temp_heads)
        kh = self._apply_rope(kh, C.n_kv_heads, ts.cos, ts.sin)            # [1,2,1,128]
        vh = to_heads(value, C.n_kv_heads)
        # masked-write into fixed-MAX KV (position via ts.onehot/ts.inv inputs)
        krep = ttnn.repeat(kh, ttnn.Shape([1, 1, ts.MAX, 1]))             # [1,2,MAX,128]
        ttnn.copy(ttnn.add(ttnn.mul(ts.kc[layer], ts.inv), ttnn.mul(krep, ts.onehot)), ts.kc[layer])
        vrep = ttnn.repeat(vh, ttnn.Shape([1, 1, ts.MAX, 1]))
        ttnn.copy(ttnn.add(ttnn.mul(ts.vc[layer], ts.inv), ttnn.mul(vrep, ts.onehot)), ts.vc[layer])
        groups = gq // C.n_kv_heads
        kk = ttnn.repeat_interleave(ts.kc[layer], groups, dim=1)           # [1,8,MAX,128]
        vv = ttnn.repeat_interleave(ts.vc[layer], groups, dim=1)
        scores = ttnn.matmul(qh, ttnn.transpose(kk, -2, -1), compute_kernel_config=self.ckc)  # [1,8,1,MAX]
        scores = ttnn.add(ttnn.multiply(ttnn.typecast(scores, ttnn.float32), self.inv_sqrt_hd), ts.amask)
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1), ttnn.bfloat16)
        attn = ttnn.matmul(probs, vv, compute_kernel_config=self.ckc)      # [1,8,1,128]
        return ttnn.linear(from_heads(attn, gq), self.o_proj, compute_kernel_config=self.ckc)

    def decode_forward(self, h1, cache, layer):
        """Decode one token at position cache.pos. h1 [1,1,2048]."""
        pos = cache.pos
        cos, sin = self._seq_pos(pos)
        qpre, kpre, value = self.qkv.decode_forward(h1, cache, layer)      # pre-L2 [1,1,1024],[1,1,256],value[1,1,256]
        q = self._apply_rope(self.qkv._l2_h(to_heads(qpre, C.cca_q_heads)), C.cca_q_heads, cos, sin)  # [1,8,1,128]
        k = ttnn.mul(self.qkv._l2_h(to_heads(kpre, C.n_kv_heads)), self.qkv.temp_heads)
        k = self._apply_rope(k, C.n_kv_heads, cos, sin)                    # [1,2,1,128]
        v = to_heads(value, C.n_kv_heads)
        cache.k[layer] = ttnn.concat([cache.k[layer], k], dim=2)           # append along seq -> [1,2,pos+1,128]
        cache.v[layer] = ttnn.concat([cache.v[layer], v], dim=2)
        groups = C.cca_q_heads // C.n_kv_heads
        kk = ttnn.repeat_interleave(cache.k[layer], groups, dim=1)         # [1,8,pos+1,128]
        vv = ttnn.repeat_interleave(cache.v[layer], groups, dim=1)
        scores = ttnn.matmul(q, ttnn.transpose(kk, -2, -1), compute_kernel_config=self.ckc)  # [1,8,1,pos+1]
        scores = ttnn.multiply(ttnn.typecast(scores, ttnn.float32), self.inv_sqrt_hd)        # no mask: q attends all
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1), ttnn.bfloat16)
        attn = ttnn.matmul(probs, vv, compute_kernel_config=self.ckc)                        # [1,8,1,128]
        attn = from_heads(attn, C.cca_q_heads)                             # [1,1,1024]
        return ttnn.linear(attn, self.o_proj, compute_kernel_config=self.ckc)          # [1,1,2048]
