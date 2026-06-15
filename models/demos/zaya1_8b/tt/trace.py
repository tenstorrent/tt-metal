"""ttnn trace for ZAYA1-8B decode (single user or B-user batch-on-M).

Captures the 80-layer decode-step op graph once and replays it with execute_trace,
removing per-op host dispatch (the decode bottleneck). All cross-token state lives in
persistent buffers updated in-place inside the captured graph; per-step position is
carried via persistent input buffers (cos/sin/mask/onehot) updated host-side.

Batch (B>1, lockstep same-length prompts): conv-state / prev_hs buffers are [1,B,*]
(B on the M/"seq" slot, like the MoE matmuls); KV is [B,n_kv,MAX,128] (B on dim0, per-user
attention). cos/sin/amask/onehot are shared across users (same position). B=1 is the
original single-user path unchanged.
"""
import torch
import ttnn

from .model_args import ZayaConfig
from .standard import compute_cos_sin, to_dev
from .cache import ZayaCache

C = ZayaConfig


def _z(shape, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TraceState:
    def __init__(self, device, MAX, B=1):
        self.device = device
        self.MAX = MAX
        self.B = B
        self.pos = 0
        nkv, hd, dim = C.n_kv_heads, C.head_dim, C.dim
        self.hin = _z((1, B, dim), device)
        self.hout = _z((1, B, dim), device)
        self.cos = _z((1, 1, 1, C.rotary_dim), device)        # shared over B (lockstep pos)
        self.sin = _z((1, 1, 1, C.rotary_dim), device)
        self.amask = _z((1, 1, 1, MAX), device, ttnn.float32)
        self.onehot = _z((1, nkv, MAX, hd), device)           # bcast over B
        self.inv = to_dev(torch.ones(1, nkv, MAX, hd), device)
        self.p1, self.p2, self.ph, self.kc, self.vc = {}, {}, {}, {}, {}
        for i in range(C.n_layers):
            if C.is_attention_layer(i):
                self.p1[i] = _z((1, B, C.cca_in_out_ch), device)
                self.p2[i] = _z((1, B, C.cca_in_out_ch), device)
                self.ph[i] = _z((1, B, dim), device)
                self.kc[i] = _z((B, nkv, MAX, hd), device)
                self.vc[i] = _z((B, nkv, MAX, hd), device)


class TracedGenerator:
    def __init__(self, model, device, MAX=128, B=1):
        self.model = model
        self.device = device
        self.B = B
        self.ts = TraceState(device, MAX, B)
        self.trace_id = None
        self._saved = None

    def _write_tokens(self, tids):
        """tids: list of length B (one prev-token id per user)."""
        emb = self.model.embed(torch.tensor([tids], dtype=torch.int32))    # [1,B,2048]
        ttnn.copy(emb, self.ts.hin)

    def _set_pos_inputs(self, pos):
        ts = self.ts
        cos, sin = compute_cos_sin(pos + 1)                                # [pos+1, rotary]
        ttnn.copy(to_dev(cos[pos].reshape(1, 1, 1, C.rotary_dim), self.device), ts.cos)
        ttnn.copy(to_dev(sin[pos].reshape(1, 1, 1, C.rotary_dim), self.device), ts.sin)
        am = torch.zeros(1, 1, 1, ts.MAX)
        am[..., pos + 1:] = -1e30
        ttnn.copy(to_dev(am, self.device, dtype=ttnn.float32), ts.amask)
        oh = torch.zeros(1, C.n_kv_heads, ts.MAX, C.head_dim)
        oh[:, :, pos, :] = 1.0
        ttnn.copy(to_dev(oh, self.device), ts.onehot)
        ttnn.copy(to_dev(1.0 - oh, self.device), ts.inv)

    def prefill(self, ids):
        """Single-user prefill; replicate the populated state to B users (same prompt).
        Returns the single-user first token id."""
        ts = self.ts
        B = self.B
        cache = ZayaCache()
        first = self.model.prefill(ids, cache)
        S = ids.shape[1]
        pad = ts.MAX - S
        for i in range(C.n_layers):
            if not C.is_attention_layer(i):
                continue
            cv1, cv0, ph = cache.conv_state[i][1], cache.conv_state[i][0], cache.prev_hs[i]  # [1,1,*]
            kfull = ttnn.concat([cache.k[i], _z((1, C.n_kv_heads, pad, C.head_dim), self.device)], dim=2)  # [1,nkv,MAX,128]
            vfull = ttnn.concat([cache.v[i], _z((1, C.n_kv_heads, pad, C.head_dim), self.device)], dim=2)
            if B > 1:
                cv1 = ttnn.repeat(cv1, ttnn.Shape([1, B, 1]))              # [1,B,1280]
                cv0 = ttnn.repeat(cv0, ttnn.Shape([1, B, 1]))
                ph = ttnn.repeat(ph, ttnn.Shape([1, B, 1]))               # [1,B,2048]
                kfull = ttnn.repeat(kfull, ttnn.Shape([B, 1, 1, 1]))      # [B,nkv,MAX,128]
                vfull = ttnn.repeat(vfull, ttnn.Shape([B, 1, 1, 1]))
            ttnn.copy(cv1, ts.p1[i])                                       # qk_prev
            ttnn.copy(cv0, ts.p2[i])                                       # qk_prev2
            ttnn.copy(ph, ts.ph[i])
            ttnn.copy(kfull, ts.kc[i])                                     # roped K padded to MAX
            ttnn.copy(vfull, ts.vc[i])
        ts.pos = S
        self._save_state()
        return first

    def _clone(self, t):
        return to_dev(ttnn.to_torch(t), self.device, dtype=t.get_dtype())

    def _save_state(self):
        self._saved = {}
        for i in self.ts.p1:
            self._saved[i] = (self._clone(self.ts.p1[i]), self._clone(self.ts.p2[i]),
                              self._clone(self.ts.ph[i]), self._clone(self.ts.kc[i]),
                              self._clone(self.ts.vc[i]))

    def _reset_state(self):
        for i, (p1, p2, ph, kc, vc) in self._saved.items():
            ttnn.copy(p1, self.ts.p1[i]); ttnn.copy(p2, self.ts.p2[i])
            ttnn.copy(ph, self.ts.ph[i]); ttnn.copy(kc, self.ts.kc[i]); ttnn.copy(vc, self.ts.vc[i])

    def _seed_list(self, seed):
        if isinstance(seed, (list, tuple)):
            return list(seed)
        return [seed] * self.B

    def capture(self, first_token):
        self._write_tokens(self._seed_list(first_token))
        self._set_pos_inputs(self.ts.pos)
        # Warm the program cache first: an op's COLD (first) run issues host->device
        # writes (compile-time args/CB configs), which are forbidden during capture.
        self.model._backbone_trace(self.ts)
        self._reset_state()
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.model._backbone_trace(self.ts)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        self._reset_state()                                               # undo capture-time mutations

    def step(self, token):
        """token: scalar (B=1) or list[B]. Returns next id(s) in the same shape."""
        self._write_tokens(self._seed_list(token))
        self._set_pos_inputs(self.ts.pos)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)
        self.ts.pos += 1
        am = ttnn.to_torch(ttnn.argmax(self.model.lm_head(self.ts.hout), dim=-1)).reshape(-1)
        ids = [int(x) for x in am]
        return ids[0] if self.B == 1 and not isinstance(token, (list, tuple)) else ids

    def generate(self, ids, n):
        first = self.prefill(ids)
        seed = first if self.B == 1 else [first] * self.B
        self.capture(seed)
        cur = seed
        out = [cur]
        for _ in range(n - 1):
            cur = self.step(cur)
            out.append(cur)
        return out
