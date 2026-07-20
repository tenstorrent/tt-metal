# KDA API spec & design (Phase 3)

> The contract for the KDA bringup. Reference (torch) API **mirrors** the device (ttnn) API
> param-for-param. Config-in → canonical HF-named weights → reference outputs. Approved
> autonomously per bringup mandate (2026-07-20); async review pending. Design principle: simplest
> architecture that correctly adds the KDA delta by mirroring the existing GDN reference module.

## 1. Scope & non-goals

- **In:** the KDA attention operator + layer, torch reference + ttnn (recurrent) device impl + PCC tests.
- **Out (this pass):** chunked-prefill *on device* (Phase 7), MoE, hybrid 3:1 stacking, full model, perf.

## 2. Design overview — adapt, don't reinvent

Mirror `models/experimental/gated_attention_gated_deltanet/` (torch golden + ttnn + tests). The
**only** semantic change is the forget gate: scalar `g[B,T,H]` → per-channel `g[B,T,HV,K]`
("diagonal gate"). This propagates to exactly three math sites (recurrent decay, chunk cumsum/L_mask,
state update) — see bringup_log Phase 2. All other machinery (delta rule, β, L2-norm, short-conv,
gated-RMSNorm, NoPE) is reused unchanged in shape.

We keep KDA as its **own module** (not a `diagonal_gate: bool` flag on the GDN reference) because the
reference bakes the scalar gate into op shapes; a sibling module mirroring its layout is cleaner than
threading a mode flag through the scalar-shaped ops. Shared, gate-agnostic helpers (l2norm, short-conv,
gated-RMSNorm) are ported verbatim.

## 3. Tensor contract (single source of truth)

```
q, k   : [B, T, H,  K]   fp32/bf16   L2-normed (post short-conv+SiLU); q scaled by K^-0.5
v      : [B, T, HV, V]   fp32/bf16   (short-conv+SiLU)
g      : [B, T, HV, K]   fp32        per-channel decay in LOG space (≤ 0)
beta   : [B, T, HV]      fp32        ∈ (0,1) (sigmoid); ×2 if allow_neg_eigval
state S: [B, HV, K, V]   fp32        recurrent state
o      : [B, T, HV, V]               output (pre gated-RMSNorm/o_proj)
```
`HV % H == 0` (GVA: q/k repeat_interleave by `HV//H`). Defaults: `K=V=128`, `H=HV=32`, `conv=4`.

## 4. Reference (torch) API — `torch_functional/kda_ops.py`

Mirrors FLA `fla/ops/kda` (the authoritative reference; validated against it in Phase 4).

```python
def l2norm(x, eps=1e-6): ...                                   # x * rsqrt(sum(x^2)+eps)

def kda_gate(g, A_log, dt_bias=None, lower_bound=None) -> Tensor:
    # g:[...,H,K] pre-activation; A_log:[H]; dt_bias:[H*K]
    # default : g = -exp(A_log)[:,None] * softplus(g + dt_bias)
    # lb form : g = lower_bound * sigmoid(exp(A_log)[:,None]*(g+dt_bias))

def naive_recurrent_kda(q, k, v, g, beta, scale=None,
                        initial_state=None, output_final_state=False) -> (o, S): ...
def naive_chunk_kda(q, k, v, g, beta, scale=None,
                    initial_state=None, output_final_state=False, chunk_size=64) -> (o, S): ...
```

Layer (`torch_functional/kda_layer.py::KimiDeltaAttentionRef`) mirrors FLA `KimiDeltaAttention`:
constructor `(hidden_size, head_dim=128, num_heads, num_v_heads=None, conv_size=4,
use_short_conv=True, allow_neg_eigval=False, lower_bound=None, norm_eps=1e-5)`; forward
`(hidden_states[B,T,hidden]) -> [B,T,hidden]`. Internally: q/k/v_proj → short-conv+SiLU → reshape →
`kda_gate` → `naive_recurrent_kda`/`naive_chunk_kda` → gated-RMSNorm(sigmoid) → o_proj. NoPE.

## 5. Device (ttnn) API — `tt/ttnn_kda_ops.py` — mirrors §4 param-for-param

```python
def l2norm_ttnn(x, dim=-1, eps=1e-6): ...
def kda_gate_ttnn(g, A_log, dt_bias=None, lower_bound=None, device=None): ...   # -exp(A_log)*softplus(.)
def recurrent_kda_ttnn(q, k, v, g, beta, scale=None,
                       initial_state=None, device=None) -> (o, S): ...
    # per-step: S = S * exp(g_t)[:, :, :, None]           # diagonal decay, [K,1] over V
    #           delta = v_t - (k_t · S); S += (beta*k_t) ⊗ delta
    #           o_t = q_t · S
```
Ops (all exist in-tree): `ttnn.matmul, multiply, add, subtract, exp, rms_norm(L2), transpose,
typecast, reshape, sigmoid, silu, softplus, neg`. **Diagonal decay** = the sole structural change
vs `recurrent_gated_delta_rule_ttnn` (scalar → `[K,1]` broadcast). Chunked device path: **not in this
pass** (fused C++ op is scalar-gate only → Phase 7); reference `naive_chunk_kda` (CPU) covers prefill
correctness meanwhile.

Layer (`tt/ttnn_kda.py::TtKimiDeltaAttention`) mirrors the reference layer; weights loaded by §6.

## 6. Weights (HF names, from `fla/layers/kda.py` + qwen36 loader conventions)

Per-layer `linear_attn` substate. Projections (no bias unless noted), loaded transposed to `[in,out]`:
`q_proj, k_proj, v_proj` (`hidden→H*K` / `HV*V`); short-conv `q_conv1d.weight, k_conv1d.weight,
v_conv1d.weight` (+ optional bias, depthwise, ROW_MAJOR host); gate low-rank `f_proj.0.weight`
(`hidden→V`), `f_proj.1.weight` (`V→HV*K`); `b_proj.weight` (`hidden→HV`); params `A_log` (`[HV]`),
`dt_bias` (`[HV*K]`); output gate `g_proj.0.weight`(`hidden→V`), `g_proj.1.weight/.bias`(`V→HV*V`);
`o_norm.weight` (`[V]`); `o_proj.weight` (`HV*V→hidden`). (For random-init validation, generated
directly; HF-name mapping documented here so a later real-weight load is a drop-in.)

## 7. Validation plan (Phases 4/6)

- **P4 (CPU):** port vs FLA `naive_recurrent_kda`/`naive_chunk_kda` on identical random inputs →
  PCC 1.0 / max|Δ|~1e-6; recurrent≡chunk within 1e-2. Cross-checks the port against the author ref.
- **P6 (device, LoudBox 8×BH):** torch ref vs `recurrent_kda_ttnn` and full `TtKimiDeltaAttention`,
  PCC ≥ 0.98 (recurrent threshold precedent 0.95 for the bf16 delta-rule op; layer ≥0.98 target).
  Rotated/chunked-geometry and determinism checks; explicit validated/NOT-validated ledger.
