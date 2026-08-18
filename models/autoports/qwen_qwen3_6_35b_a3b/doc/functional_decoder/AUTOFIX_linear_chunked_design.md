# AutoFix Linear Chunked Design

## Starting Evidence

- Scope: source-only AutoFix side task for `Qwen/Qwen3.6-35B-A3B` functional decoder linear attention. No hardware-facing command, pytest, `tt-smi`, `tt-triage`, or device reset was run.
- Forked subagents: no forked-subagent tool is available in this environment, so this pass followed the AutoFix loop serially.
- Existing diagnostic: `models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/AUTODEBUG_linear_context.md`.
- Worktree: `git status --short` shows `models/autoports/` and `tt_metal/third_party/tt-cluster-descriptors/` as untracked. I did not change implementation, tests, or existing docs.
- Current TTNN implementation: `_QwenLinearAttention.prefill_forward` loops over each token and calls `_step` once per token (`tt/functional_decoder.py:584-600`). `_step` is the decode-style recurrent update (`tt/functional_decoder.py:529-579`).
- HF reference: multi-token forward uses `chunk_gated_delta_rule`, while only cached single-token decode uses `recurrent_gated_delta_rule` (`modeling_qwen3_5_moe.py:520-543`). The fallback chunk rule is `torch_chunk_gated_delta_rule` (`modeling_qwen3_5_moe.py:243-321`).
- Target dimensions from `doc/context_contract.json`: `hidden_size=2048`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`, advertised context `262144`.

## Verdict

Feasible, but not as a small patch and not through `ttnn.experimental.prefix_scan`.

A correct stage-local TTNN prefill can be assembled from existing TTNN ops by processing one HF-style 64-token chunk at a time. The implementation should keep all measured-forward math on TTNN tensors: `ttnn.linear`, `ttnn.slice`, `ttnn.concat`, `ttnn.reshape`, `ttnn.permute`, `ttnn.repeat_interleave`, `ttnn.pad`, `ttnn.cumsum`, elementwise ops, `ttnn.exp`, `ttnn.matmul`, and `ttnn.typecast` where needed. Runtime Torch or host fallback is not required. Torch may still be used at module construction to create static 64x64 masks, matching existing setup-only patterns in this file.

The important design choice is chunk streaming, not materializing all chunks. For `S=262144`, an all-at-once tensor shaped `[1, batch * 32 * 4096, 64, 64]` would be too large. The prefill path should instead run `4096` Python chunk iterations, each using rank-4 tensors with head batch `[1, batch * 32, 64, *]`.

## Tensor Shapes

Symbols:

- `B`: batch.
- `S`: logical prefill length.
- `C`: chunk size, use `64` to match HF fallback and stay tile-aligned.
- `L`: logical length of the current chunk, `1 <= L <= C`.
- `Hk=16`, `Hv=32`, `Dk=128`, `Dv=128`.
- `H = B * Hv`.
- `key_dim = Hk * Dk = 2048`.
- `value_dim = Hv * Dv = 4096`.
- `conv_dim = 2 * key_dim + value_dim = 8192`.

State:

- `conv_state`: four taps, each `[1, 1, B, 8192]`.
- `recurrent_state`: `[1, H, 128, 128]`.

Per chunk:

- `hidden_chunk`: `[1, B, L, 2048]`.
- `mixed_qkv_raw = linear(hidden_chunk, in_proj_qkv)`: `[1, B, L, 8192]`.
- `mixed_qkv = depthwise_causal_conv(mixed_qkv_raw, conv_state)`: `[1, B, L, 8192]`.
- `z = linear(hidden_chunk, in_proj_z)`: `[1, B, L, 4096]`.
- `beta = sigmoid(linear(hidden_chunk, in_proj_b))`: `[1, B, L, 32]`.
- `log_g = -exp(A_log) * softplus(linear(hidden_chunk, in_proj_a) + dt_bias)`: `[1, B, L, 32]`; keep this un-exponentiated for the chunk rule.
- `query`, `key`: split from `mixed_qkv`, reshape/repeat to `[1, H, L, 128]`.
- `value`, `z`: reshape to `[1, H, L, 128]`.
- `beta`, `log_g`: reshape to `[1, H, L, 1]`.
- Padded chunk rule inputs: `[1, H, 64, *]`; pad with zeros when `L < 64`.
- Chunk matrices: `[1, H, 64, 64]`.
- `core`: `[1, H, L, 128]`.
- `out_chunk`: `[1, B, L, 2048]`.

## Minimal Changes

Add one constant:

```python
LINEAR_ATTENTION_CHUNK_SIZE = 64
```

Extend `_QwenLinearAttention.__init__` with setup-only static tensors:

```python
self.linear_chunk_size = LINEAR_ATTENTION_CHUNK_SIZE
self.chunk_lower_mask        # [1, 1, 64, 64], 1 where row >= col
self.chunk_strict_lower_mask # [1, 1, 64, 64], 1 where row > col
self.chunk_eye              # [1, 1, 64, 64]
self.chunk_ones_1x64        # [1, 1, 1, 64]
self.row_prefix_masks[i]    # [1, 1, 64, 64], row i and cols < i
self.row_keep_masks[i]      # [1, 1, 64, 64], zero only row i
```

These can be created from small Torch CPU constants in `__init__` and moved with `_as_device_tensor(...)`. That is setup-only and does not introduce runtime Torch in measured forward. If setup Torch is considered too broad later, replace these with `ttnn.ones`, `ttnn.tril`/`ttnn.triu`, and comparison masks, but that path needs its own op smoke.

Add these helpers inside `_QwenLinearAttention`:

- `_conv_prefill(mixed_qkv_raw, state) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ...]]`
- `_reshape_prefill_heads(mixed_qkv, z, beta, log_g, batch, length) -> tuple[...]`
- `_pad_linear_chunk(x, length, padded_length, last_dim) -> ttnn.Tensor`
- `_solve_chunk_attn(attn0) -> ttnn.Tensor`
- `_chunk_gated_delta_rule(q, k, v, log_g, beta, recurrent_state) -> tuple[ttnn.Tensor, ttnn.Tensor]`
- `_finish_prefill_chunk(core, z, batch, length) -> ttnn.Tensor`

Replace only `_QwenLinearAttention.prefill_forward`; keep `_step` and `decode_forward` unchanged.

## Pseudocode

Vectorized causal conv for one chunk:

```python
def _conv_prefill(self, mixed_qkv_raw, state):
    _, batch, length, _ = _shape(mixed_qkv_raw)

    taps = [ttnn.reshape(t, (1, batch, 1, self.conv_dim)) for t in state.conv_state]
    conv_input = ttnn.concat([taps[1], taps[2], taps[3], mixed_qkv_raw], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    acc = None
    for tap_idx in range(self.cfg.linear_conv_kernel_dim):
        window = _slice(conv_input, (0, 0, tap_idx, 0), (1, batch, tap_idx + length, self.conv_dim))
        part = ttnn.mul(window, self.conv_weights[tap_idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        acc = part if acc is None else ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mixed_qkv = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    history = ttnn.concat([taps[0], taps[1], taps[2], taps[3], mixed_qkv_raw], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    hist_len = _shape(history)[2]
    last4 = _slice(history, (0, 0, hist_len - 4, 0), (1, batch, hist_len, self.conv_dim))
    next_taps = tuple(
        ttnn.reshape(_slice(last4, (0, 0, i, 0), (1, batch, i + 1, self.conv_dim)), (1, 1, batch, self.conv_dim))
        for i in range(4)
    )
    return mixed_qkv, next_taps
```

Head reshape for one chunk:

```python
def _reshape_prefill_heads(self, mixed_qkv, z, beta, log_g, batch, length):
    query = _slice_last(mixed_qkv, 0, self.key_dim)
    key = _slice_last(mixed_qkv, self.key_dim, 2 * self.key_dim)
    value = _slice_last(mixed_qkv, 2 * self.key_dim, self.conv_dim)

    query = ttnn.reshape(query, (batch, length, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
    key = ttnn.reshape(key, (batch, length, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
    if self.repeat_factor != 1:
        query = ttnn.repeat_interleave(query, self.repeat_factor, dim=2)
        key = ttnn.repeat_interleave(key, self.repeat_factor, dim=2)

    value = ttnn.reshape(value, (batch, length, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim))
    z = ttnn.reshape(z, (batch, length, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim))
    beta = ttnn.reshape(beta, (batch, length, self.cfg.linear_num_value_heads, 1))
    log_g = ttnn.reshape(log_g, (batch, length, self.cfg.linear_num_value_heads, 1))

    def fold_heads(x):
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (1, batch * self.cfg.linear_num_value_heads, length, _shape(x)[-1]))

    query = _l2_norm_last_dim(fold_heads(query), self.cfg.linear_key_head_dim)
    key = _l2_norm_last_dim(fold_heads(key), self.cfg.linear_key_head_dim)
    query = ttnn.mul(query, self.cfg.linear_key_head_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return query, key, fold_heads(value), fold_heads(z), fold_heads(beta), fold_heads(log_g)
```

Chunk-local triangular recurrence, equivalent to HF lines 284-294:

```python
def _solve_chunk_attn(self, attn0):
    # attn0 is strict lower triangular [1, H, 64, 64].
    solved = attn0
    for i in range(1, self.linear_chunk_size):
        row = ttnn.mul(solved, self.row_prefix_masks[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        update = ttnn.matmul(row, solved, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
        new_row = ttnn.add(row, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kept = ttnn.mul(solved, self.row_keep_masks[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        solved = ttnn.add(kept, new_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.add(solved, self.chunk_eye, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

This matches the HF row update because `row_prefix_masks[i]` selects only row `i` and columns `< i`; the matmul `row @ solved` can only read already-updated rows `< i`.

Full chunk rule:

```python
def _chunk_gated_delta_rule(self, query, key, value, log_g, beta, recurrent_state):
    # All inputs are padded to [1, H, 64, *].
    log_g = ttnn.cumsum(log_g, dim=2, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    g_rows = ttnn.matmul(log_g, self.chunk_ones_1x64, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    g_cols = ttnn.transpose(g_rows, -2, -1)
    decay = ttnn.exp(ttnn.subtract(g_rows, g_cols, memory_config=ttnn.DRAM_MEMORY_CONFIG), fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    decay = ttnn.mul(decay, self.chunk_lower_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    k_beta = ttnn.mul(key, beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_beta = ttnn.mul(value, beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    kk = ttnn.matmul(k_beta, ttnn.transpose(key, -2, -1), memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    attn0 = ttnn.neg(ttnn.mul(kk, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn0 = ttnn.mul(attn0, self.chunk_strict_lower_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    local_attn = self._solve_chunk_attn(attn0)

    local_value = ttnn.matmul(local_attn, v_beta, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    exp_g = ttnn.exp(log_g, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_cumdecay = ttnn.matmul(local_attn, ttnn.mul(k_beta, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)

    v_prime = ttnn.matmul(k_cumdecay, recurrent_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    v_new = ttnn.subtract(local_value, v_prime, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    content_attn = ttnn.matmul(query, ttnn.transpose(key, -2, -1), memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    content_attn = ttnn.mul(content_attn, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_inter = ttnn.matmul(ttnn.mul(query, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG), recurrent_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    core = ttnn.add(attn_inter, ttnn.matmul(content_attn, v_new, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    g_last = _slice(log_g, (0, 0, self.linear_chunk_size - 1, 0), (1, _shape(log_g)[1], self.linear_chunk_size, 1))
    state_decay = ttnn.mul(recurrent_state, ttnn.exp(g_last, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    state_scale = ttnn.exp(ttnn.subtract(g_last, log_g, memory_config=ttnn.DRAM_MEMORY_CONFIG), fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    state_update_key = ttnn.mul(key, state_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    state_update = ttnn.matmul(ttnn.transpose(state_update_key, -2, -1), v_new, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    recurrent_state = ttnn.add(state_decay, state_update, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return core, recurrent_state
```

Top-level prefill replacement:

```python
def prefill_forward(self, hidden_states, *, linear_state):
    _, batch, seq_len, _ = _shape(hidden_states)
    outputs = []
    state = linear_state

    for start in range(0, seq_len, self.linear_chunk_size):
        end = min(start + self.linear_chunk_size, seq_len)
        length = end - start
        hidden_chunk = _slice(hidden_states, (0, 0, start, 0), (1, batch, end, self.cfg.hidden_size))

        mixed_qkv_raw = ttnn.linear(hidden_chunk, self.in_proj_qkv, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mixed_qkv, conv_state = self._conv_prefill(mixed_qkv_raw, state)
        z = ttnn.linear(hidden_chunk, self.in_proj_z, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        beta = ttnn.sigmoid(ttnn.linear(hidden_chunk, self.in_proj_b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        alpha = ttnn.linear(hidden_chunk, self.in_proj_a, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        log_g = ttnn.mul(ttnn.softplus(ttnn.add(alpha, self.dt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG), beta=1.0, threshold=20.0, memory_config=ttnn.DRAM_MEMORY_CONFIG), self.neg_exp_a_log, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q, k, v, z_heads, beta, log_g = self._reshape_prefill_heads(mixed_qkv, z, beta, log_g, batch, length)
        if length != self.linear_chunk_size:
            q = self._pad_linear_chunk(q, length, self.linear_chunk_size, self.cfg.linear_key_head_dim)
            k = self._pad_linear_chunk(k, length, self.linear_chunk_size, self.cfg.linear_key_head_dim)
            v = self._pad_linear_chunk(v, length, self.linear_chunk_size, self.cfg.linear_value_head_dim)
            z_heads = self._pad_linear_chunk(z_heads, length, self.linear_chunk_size, self.cfg.linear_value_head_dim)
            beta = self._pad_linear_chunk(beta, length, self.linear_chunk_size, 1)
            log_g = self._pad_linear_chunk(log_g, length, self.linear_chunk_size, 1)

        core, recurrent_state = self._chunk_gated_delta_rule(q, k, v, log_g, beta, state.recurrent_state)
        if length != self.linear_chunk_size:
            core = _slice(core, (0, 0, 0, 0), (1, batch * self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim))
            z_heads = _slice(z_heads, (0, 0, 0, 0), (1, batch * self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim))

        out_chunk = self._finish_prefill_chunk(core, z_heads, batch, length)
        outputs.append(out_chunk)
        state = QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    return _concat_dim2_bounded(outputs), state
```

Finish chunk:

```python
def _finish_prefill_chunk(self, core, z, batch, length):
    core = _rms_norm(core, self.norm_weight, self.cfg.rms_norm_eps)
    core = ttnn.mul(core, ttnn.silu(z, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    core = ttnn.reshape(core, (batch, self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim))
    core = ttnn.permute(core, (0, 2, 1, 3))
    core = ttnn.reshape(core, (1, batch, length, self.value_dim))
    return ttnn.linear(core, self.out_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

## TTNN Constraints To Smoke Later

These are not blockers from source inspection, but they need a small TTNN op-legality smoke before the full 262144 probe:

- `ttnn.cumsum(log_g, dim=2, dtype=ttnn.float32)` on `[1, H, 64, 1]`: source documents BF16/FP32, TILE, ranks 1-5, interleaved DRAM/L1. Ensure inputs are tile layout, device resident, interleaved, and not sharded.
- `ttnn.matmul` batched rank-4 forms:
  - `[1, H, 64, 128] @ [1, H, 128, 64] -> [1, H, 64, 64]`
  - `[1, H, 64, 64] @ [1, H, 64, 128] -> [1, H, 64, 128]`
  - `[1, H, 64, 128] @ [1, H, 128, 128] -> [1, H, 64, 128]`
  - `[1, H, 128, 64] @ [1, H, 64, 128] -> [1, H, 128, 128]`
- Broadcast behavior for masks `[1, 1, 64, 64]` against `[1, H, 64, 64]`. If this fails, repeat the masks to `[1, H, 64, 64]` once per module or per batch size.
- `ttnn.matmul(log_g, ones_1x64)` as a broadcast expansion from `[1, H, 64, 1]` to `[1, H, 64, 64]`. If that is awkward, use `ttnn.repeat(log_g, (1, 1, 1, 64))`.
- Mixed or FP32 internal dtype. The closest HF algorithm casts chunk-rule inputs to FP32. If FP32 matmul is too slow or incompatible in one of these shapes, keep `log_g`/decay in FP32 but run value/state matmuls in BF16 and validate PCC against HF.
- Last partial chunk padding: pad only the chunk-rule tensors after causal conv. Do not pad `hidden_chunk` before `in_proj_qkv` and conv, because padded tokens would incorrectly receive nonzero causal-conv output from real left context.

There is no true device-free smoke that can prove these validators, because these TTNN ops validate device-resident tensors. The smallest later check should be a single linear layer with synthetic weights at `seq_len=65` and fallback exceptions enabled, followed by the existing small HF-vs-TTNN PCC path. After that, run the advertised `262144` context probe.

## Final Status

Design verdict: feasible as a stage-local TTNN implementation, with no runtime Torch or host fallback in measured forward, but it is substantial new functional code.

The precise blocker to claiming advertised linear prefill today remains implementation and validation work, not an identified missing TTNN primitive. The likely risk is op legality/performance of the constant-size triangular solve loop and FP32 internal matmuls, which requires a later small device smoke before the long context probe.
