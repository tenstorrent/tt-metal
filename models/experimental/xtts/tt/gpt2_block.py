"""TTNN port of the XTTS GPT2 decoder block + the GPT head tail (Phase 1).

The reference is HuggingFace GPT2Block as used inside XTTS's `gpt.gpt`
(see _verify_torch_phase1.py for the validated math):

    h = x + attn(ln_1(x))          # causal multi-head self-attention
    out = h + mlp(ln_2(h))         # c_fc -> NewGELU -> c_proj

Key reference quirks handled here:
  * GPT2 uses `Conv1D`, whose weight is stored [in, out] — i.e. already in the
    layout `ttnn.linear` wants (y = x @ W + b), so NO transpose is needed.
  * nn.Linear (mel_head) stores [out, in] and DOES need a transpose.
  * Attention scale = 1/sqrt(head_dim), causal — matches ttnn SDPA defaults.
"""

import ttnn

HIDDEN = 1024
HEADS = 16
HEAD_DIM = 64
EPS = 1e-5


# ---------------------------------------------------------------------------
# weight loading
# ---------------------------------------------------------------------------
def _to_device(torch_tensor, device, *, transpose=False):
    """torch -> ttnn bf16 tile tensor on device. `transpose` for nn.Linear weights."""
    import torch

    t = torch_tensor
    if transpose:
        t = t.t().contiguous()
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def load_block_params(weights, device, prefix="gpt.gpt.h.0."):
    """Build the on-device parameter dict for one GPT2Block.

    `weights` is the torch state-dict slice (keys under `prefix`). Conv1D weights
    are loaded as-is (already [in, out]); LayerNorm weights as 1-D.
    """

    def w(name, **kw):
        return _to_device(weights[prefix + name], device, **kw)

    return {
        "ln_1.weight": w("ln_1.weight"),
        "ln_1.bias": w("ln_1.bias"),
        "attn.c_attn.weight": w("attn.c_attn.weight"),  # [1024, 3072]  (Conv1D, no transpose)
        "attn.c_attn.bias": w("attn.c_attn.bias"),
        "attn.c_proj.weight": w("attn.c_proj.weight"),  # [1024, 1024]
        "attn.c_proj.bias": w("attn.c_proj.bias"),
        "ln_2.weight": w("ln_2.weight"),
        "ln_2.bias": w("ln_2.bias"),
        "mlp.c_fc.weight": w("mlp.c_fc.weight"),  # [1024, 4096]
        "mlp.c_fc.bias": w("mlp.c_fc.bias"),
        "mlp.c_proj.weight": w("mlp.c_proj.weight"),  # [4096, 1024]
        "mlp.c_proj.bias": w("mlp.c_proj.bias"),
    }


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def _split_heads(t):  # [1, s, hidden] -> [1, heads, s, head_dim]
    b, s, _ = t.shape
    t = ttnn.reshape(t, (b, s, HEADS, HEAD_DIM))
    return ttnn.permute(t, (0, 2, 1, 3))


def _merge_heads(t):  # [1, heads, s, head_dim] -> [1, s, hidden]
    b, _, s, _ = t.shape
    t = ttnn.permute(t, (0, 2, 1, 3))
    return ttnn.reshape(t, (b, s, HIDDEN))


def attention(x, p):
    qkv = ttnn.linear(x, p["attn.c_attn.weight"], bias=p["attn.c_attn.bias"])
    q = ttnn.slice(qkv, [0, 0, 0], [qkv.shape[0], qkv.shape[1], HIDDEN])
    k = ttnn.slice(qkv, [0, 0, HIDDEN], [qkv.shape[0], qkv.shape[1], 2 * HIDDEN])
    v = ttnn.slice(qkv, [0, 0, 2 * HIDDEN], [qkv.shape[0], qkv.shape[1], 3 * HIDDEN])
    ttnn.deallocate(qkv)

    q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = _merge_heads(attn)
    return ttnn.linear(out, p["attn.c_proj.weight"], bias=p["attn.c_proj.bias"])


def mlp(x, p):
    h = ttnn.linear(x, p["mlp.c_fc.weight"], bias=p["mlp.c_fc.bias"])
    h = ttnn.gelu(h)  # tanh-approx GELU ~= GPT2 NewGELU (validated by PCC)
    return ttnn.linear(h, p["mlp.c_proj.weight"], bias=p["mlp.c_proj.bias"])


def gpt2_block(x, p):
    h = ttnn.layer_norm(x, weight=p["ln_1.weight"], bias=p["ln_1.bias"], epsilon=EPS)
    x = ttnn.add(x, attention(h, p))
    h = ttnn.layer_norm(x, weight=p["ln_2.weight"], bias=p["ln_2.bias"], epsilon=EPS)
    return ttnn.add(x, mlp(h, p))


# ---------------------------------------------------------------------------
# Full 30-layer stack (prefill) + KV-cache decode
# ---------------------------------------------------------------------------
def load_stack_params(weights, device, n_layers=30):
    """Build the on-device param dict for every GPT2Block layer."""
    return [load_block_params(weights, device, prefix=f"gpt.gpt.h.{i}.") for i in range(n_layers)]


def stack_prefill(x, layers):
    """Run the full decoder stack over a prefill sequence (causal, no cache)."""
    for p in layers:
        x = gpt2_block(x, p)
    return x


def init_kv_cache(device, n_layers, max_seq, batch=1):
    """Pre-allocate per-layer K/V caches: [batch, heads, max_seq, head_dim]."""
    import torch

    caches = []
    for _ in range(n_layers):
        zeros = torch.zeros(batch, HEADS, max_seq, HEAD_DIM, dtype=torch.bfloat16)
        kc = ttnn.from_torch(zeros, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        vc = ttnn.from_torch(zeros.clone(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        caches.append((kc, vc))
    return caches


def _qkv_heads(x, p):
    """c_attn -> split into q/k/v, each [1, heads, seq, head_dim]."""
    qkv = ttnn.linear(x, p["attn.c_attn.weight"], bias=p["attn.c_attn.bias"])
    q = ttnn.slice(qkv, [0, 0, 0], [qkv.shape[0], qkv.shape[1], HIDDEN])
    k = ttnn.slice(qkv, [0, 0, HIDDEN], [qkv.shape[0], qkv.shape[1], 2 * HIDDEN])
    v = ttnn.slice(qkv, [0, 0, 2 * HIDDEN], [qkv.shape[0], qkv.shape[1], 3 * HIDDEN])
    ttnn.deallocate(qkv)
    return _split_heads(q), _split_heads(k), _split_heads(v)


def _attention_prefill(x, p, kc, vc):
    q, k, v = _qkv_heads(x, p)  # [1, heads, S, head_dim]
    ttnn.fill_cache(kc, k, 0)  # populate cache positions 0..S-1
    ttnn.fill_cache(vc, v, 0)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = _merge_heads(attn)
    return ttnn.linear(out, p["attn.c_proj.weight"], bias=p["attn.c_proj.bias"])


def _attention_decode(x, p, kc, vc, cur_pos):
    q, k, v = _qkv_heads(x, p)  # [1, heads, 1, head_dim]
    ttnn.update_cache(kc, k, cur_pos)  # write new token's k/v at seq=cur_pos
    ttnn.update_cache(vc, v, cur_pos)
    qd = ttnn.permute(q, (0, 2, 1, 3))  # -> [1, batch=1, heads, head_dim]
    attn = ttnn.transformer.scaled_dot_product_attention_decode(qd, kc, vc, cur_pos=[cur_pos])
    attn = ttnn.permute(attn, (0, 2, 1, 3))  # -> [1, heads, 1, head_dim]
    out = _merge_heads(attn)
    return ttnn.linear(out, p["attn.c_proj.weight"], bias=p["attn.c_proj.bias"])


def _block_with_attn(x, p, attn_out_fn):
    h = ttnn.layer_norm(x, weight=p["ln_1.weight"], bias=p["ln_1.bias"], epsilon=EPS)
    x = ttnn.add(x, attn_out_fn(h))
    h = ttnn.layer_norm(x, weight=p["ln_2.weight"], bias=p["ln_2.bias"], epsilon=EPS)
    return ttnn.add(x, mlp(h, p))


def stack_prefill_with_cache(x, layers, caches):
    """Prefill the full stack AND populate each layer's KV cache."""
    for p, (kc, vc) in zip(layers, caches):
        x = _block_with_attn(x, p, lambda h, p=p, kc=kc, vc=vc: _attention_prefill(h, p, kc, vc))
    return x


def stack_decode(x, layers, caches, cur_pos):
    """One autoregressive decode step (x is a single token [1,1,1024])."""
    for p, (kc, vc) in zip(layers, caches):
        x = _block_with_attn(x, p, lambda h, p=p, kc=kc, vc=vc: _attention_decode(h, p, kc, vc, cur_pos))
    return x


# ---------------------------------------------------------------------------
# GPT head tail: ln_f / final_norm (LayerNorm) and mel_head (Linear)
# ---------------------------------------------------------------------------
def layer_norm(x, weight, bias):
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=EPS)


def mel_head(x, weight, bias):
    return ttnn.linear(x, weight, bias=bias)
