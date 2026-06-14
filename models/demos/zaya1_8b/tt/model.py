"""Phase 4-5: full ZAYA1-8B model assembly (prefill) on tt-metal.

Wiring (delayed/parallel residual merge), mirroring HF ZayaModel.forward:
  hidden = embed(ids); residual = None; prev_router = None
  for each layer (even=ATT, odd=MoE):
      residual, hidden = res_scale(residual, hidden)          # learned per-channel
      residual = hidden if residual is None else hidden + residual
      hidden   = input_norm(residual)                          # RMSNorm
      hidden   = CCA(hidden)            (ATT)  |  hidden, prev_router = MoE(hidden, prev_router)  (MoE)
  residual, hidden = model_res_scale(residual, hidden); residual = hidden + residual
  hidden = final_norm(residual); logits = lm_head(hidden)
"""
import torch
import ttnn

from .model_args import ZayaConfig, ZayaWeights
from .standard import to_dev, Embedding, RMSNorm, LMHead
from .cca import CCAAttention
from .moe import ZayaMoEBlock

C = ZayaConfig


class ResidualScaling:
    def __init__(self, device, w, prefix, has_residual):
        self.hscale = to_dev(w.get(f"{prefix}.hidden_states_scale").reshape(1, -1), device)
        self.hbias = to_dev(w.get(f"{prefix}.hidden_states_bias").reshape(1, -1), device)
        self.has_residual = has_residual
        if has_residual:  # residual stream kept in fp32 (residual_in_fp32)
            self.rscale = to_dev(w.get(f"{prefix}.residual_scale").reshape(1, -1), device, dtype=ttnn.float32)
            self.rbias = to_dev(w.get(f"{prefix}.residual_bias").reshape(1, -1), device, dtype=ttnn.float32)

    def __call__(self, residual, hidden):
        hidden = ttnn.mul(ttnn.add(hidden, self.hbias), self.hscale)
        if self.has_residual and residual is not None:
            residual = ttnn.mul(ttnn.add(residual, self.rbias), self.rscale)
        return residual, hidden


class ZayaModel:
    def __init__(self, device, seq_len=None, w=None, verbose=True):
        # seq_len is unused (model is seq-agnostic; CCA caches per-seq rope/mask).
        self.device = device
        w = w or ZayaWeights()
        self.embed = Embedding(device, w.embed())
        self.layers = []
        for i in range(C.n_layers):
            pfx = f"model.layers.{i}"
            res = ResidualScaling(device, w, f"{pfx}.res_scale", has_residual=(i != 0))
            inorm = RMSNorm(device, w.get(f"{pfx}.input_norm.weight"))
            if C.is_attention_layer(i):
                sub = CCAAttention(device, w, i)
                self.layers.append(("att", res, inorm, sub))
            else:
                sub = ZayaMoEBlock(device, w, i)
                self.layers.append(("moe", res, inorm, sub))
            if verbose and (i % 10 == 0):
                print(f"  built layer {i}/{C.n_layers}")
        self.model_res = ResidualScaling(device, w, "model.res_scale", has_residual=True)
        self.final_norm = RMSNorm(device, w.final_norm())
        self.lm_head = LMHead(device, w.embed())

    def _backbone(self, input_ids, capture_hidden=None, capture_choices=None, cache=None, decode=False, sparse=True):
        """embedding -> 80 layers -> model_res -> final_norm. Returns final hidden [1,S,2048] bf16.
        cache + decode=False : prefill that also populates the decode cache.
        cache + decode=True  : single-token decode using/updating the cache."""
        hidden = self.embed(input_ids)            # [1,S,2048] bf16
        residual = None                           # kept in fp32 across layers
        prev_router = None
        for i, (kind, res, inorm, sub) in enumerate(self.layers):
            residual, hidden = res(residual, hidden)
            if residual is None:
                residual = ttnn.typecast(hidden, ttnn.float32)
            else:
                residual = ttnn.add(ttnn.typecast(hidden, ttnn.float32), residual)
            hidden = ttnn.typecast(inorm(ttnn.typecast(residual, ttnn.bfloat16)), ttnn.bfloat16)
            if kind == "att":
                if decode:
                    hidden = sub.decode_forward(hidden, cache, i)
                elif cache is not None:
                    hidden = sub.forward(hidden, cache, i)          # prefill + populate cache
                else:
                    hidden = sub.forward(hidden)
            elif decode:
                hidden, prev_router = sub.forward(hidden, prev_router)[:2]          # batched experts, on-device
            else:
                if capture_choices is not None:
                    hidden, prev_router, gate = sub.forward(hidden, prev_router, return_gate=True)
                    capture_choices[i] = ttnn.to_torch(gate).float().reshape(1, -1, 17)[0].argmax(-1)
                else:
                    hidden, prev_router = sub.forward(hidden, prev_router)
            if capture_hidden is not None:
                capture_hidden[i + 1] = ttnn.to_torch(hidden).float()
        residual, hidden = self.model_res(residual, hidden)
        residual = ttnn.add(ttnn.typecast(hidden, ttnn.float32), residual)
        return ttnn.typecast(self.final_norm(ttnn.typecast(residual, ttnn.bfloat16)), ttnn.bfloat16)

    def forward(self, input_ids, capture_hidden=None, capture_choices=None):
        """input_ids [1,S]. Returns full logits [1,S,vocab]."""
        hidden = self._backbone(input_ids, capture_hidden, capture_choices)
        return self.lm_head(hidden)

    def _backbone_trace(self, ts):
        """Trace-able single-token backbone: reads ts.hin, runs layers (ATT via
        trace_decode on persistent buffers, MoE batched), writes ts.hout in-place."""
        hidden = ts.hin
        residual = None
        prev_router = None
        for i, (kind, res, inorm, sub) in enumerate(self.layers):
            residual, hidden = res(residual, hidden)
            if residual is None:
                residual = ttnn.typecast(hidden, ttnn.float32)
            else:
                residual = ttnn.add(ttnn.typecast(hidden, ttnn.float32), residual)
            hidden = ttnn.typecast(inorm(ttnn.typecast(residual, ttnn.bfloat16)), ttnn.bfloat16)
            if kind == "att":
                hidden = sub.trace_decode(hidden, ts, i)
            else:
                hidden, prev_router = sub.forward(hidden, prev_router)[:2]
        residual, hidden = self.model_res(residual, hidden)
        residual = ttnn.add(ttnn.typecast(hidden, ttnn.float32), residual)
        hidden = ttnn.typecast(self.final_norm(ttnn.typecast(residual, ttnn.bfloat16)), ttnn.bfloat16)
        ttnn.copy(hidden, ts.hout)

    def _last_logits(self, hidden):
        S = hidden.shape[1]
        last = ttnn.slice(hidden, [0, S - 1, 0], [1, S, ZayaConfig.dim])  # [1,1,2048]
        return self.lm_head(last)                                         # [1,1,vocab]

    def _argmax_id(self, logits):
        return int(ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(-1)[0])

    def next_token(self, input_ids):
        """Greedy next-token id (no cache). lm_head on LAST position + device argmax."""
        return self._argmax_id(self._last_logits(self._backbone(input_ids)))

    # ---- incremental decode ----
    def prefill(self, input_ids, cache):
        """Run the prompt, populate `cache`, return greedy next-token id."""
        hidden = self._backbone(input_ids, cache=cache)
        cache.pos = input_ids.shape[1]
        return self._argmax_id(self._last_logits(hidden))

    def decode_step(self, token_id, cache, sparse=True):
        """One incremental step from a single token id; updates cache; returns next id."""
        import torch
        ids = torch.tensor([[token_id]], dtype=torch.int32)
        hidden = self._backbone(ids, cache=cache, decode=True, sparse=sparse)   # [1,1,2048]
        cache.pos += 1
        return self._argmax_id(self.lm_head(hidden))

    def generate(self, input_ids, n, sparse=True):
        """Greedy generate n tokens with the incremental cache. Returns list of token ids."""
        from .cache import ZayaCache
        cache = ZayaCache()
        out = [self.prefill(input_ids, cache)]
        for _ in range(n - 1):
            out.append(self.decode_step(out[-1], cache, sparse=sparse))
        return out
