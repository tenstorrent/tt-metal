"""Incremental-decode cache for ZAYA1-8B CCA layers.

Per ATT layer the cross-token state is:
  - conv_state: the last two qk_packed0 vectors (qk[pos-2], qk[pos-1]) for the
    causal k=2 conv, stored as a 2-tuple of [1,1,1280] tensors.
  - prev_hs: the previous token's CCA-input hidden [1,1,2048] (for the v2 stream).
  - k, v: roped KV cache, [1, n_kv, pos, head_dim], grown by concat each step.
MoE / residual / EDA carry no cross-token state (per-forward only).
"""


class ZayaCache:
    def __init__(self):
        self.conv_state = {}   # layer -> (qk_prev2 [1,1,1280], qk_prev [1,1,1280])
        self.prev_hs = {}      # layer -> [1,1,2048]
        self.k = {}            # layer -> [1,n_kv,pos,128]
        self.v = {}            # layer -> [1,n_kv,pos,128]
        self.pos = 0           # number of tokens already processed (= position of next token)
