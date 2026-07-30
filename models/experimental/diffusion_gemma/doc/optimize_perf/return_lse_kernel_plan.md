# `return_lse` for ttnn plain + chunked SDPA (paged-prefix task T6)

Status: **REVERTED 2026-07-30 — archaeology only.** The work did ship and was device-verified on QB2
on 2026-07-19 (6/6), but its subject lived **outside** `models/experimental/diffusion_gemma/`, in
`ttnn/cpp/ttnn/operations/transformer/sdpa/`, and it had **no consumer**: nothing in the live model
or serving path ever called `return_lse=True`, because the Phase-2 merge it was built for was never
wired into `tt/diffusion_attention.py`. Under the no-shared-edits rule it was therefore reverted with
the rest of the out-of-folder changes, and `tests/test_return_lse.py` was deleted with it. The
11-file diff is recoverable in full from commit `2e18c599bd3`; re-land it as a standalone ttnn PR
**before** wiring the merge, not as a DiffusionGemma-branch carry. `tt/attention_merge.py` (the T7
half) survives untouched — it is pure ttnn and never depended on the reverted producer.
Owns: the LSE identity and scale convention, the two bringup bugs the plan got wrong, the streaming
byte-identity strategy, the remaining scope gap and the flash-merge identity.
See also: [refuted list](../REFUTED.md), [early halt](early_halt.md#absorbed-recon-verdicts) for
why T6 exists.

## What shipped (and was then reverted)

`ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp` declared the T6 `return_lse` variants of
`scaled_dot_product_attention` and `chunked_scaled_dot_product_attention`, emitting the per-row
log-sum-exp as an optional second output so a later step could merge attention over KV partials.
`tests/test_return_lse.py` passed **6/6** on QB2. Both are gone from the tree as of 2026-07-30; the
torch reference the test gated against is kept below so the re-land recipe survives the deletion.

```python
def _reference_lse(q, k, scale, is_causal):
    """torch fp32 reference for the emitted LSE.

    q, k: ``[B, H, S, D]`` fp32. Returns ``[B, H, S, 1]`` fp32 =
    ``logsumexp(scale * Q @ Kᵀ [+ causal mask], dim=-1)``.
    """
    scores = scale * torch.matmul(q, k.transpose(-1, -2))  # [B, H, Sq, Sk]
    if is_causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        causal = torch.tril(torch.ones(sq, sk, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.logsumexp(scores, dim=-1, keepdim=True)
```

The gate has two halves: `return_lse=False` must be **byte-identical** to today for every existing
caller, and the emitted LSE must match `torch.logsumexp`.

> **MEASUREMENT TRAP.** PCC is misleading on the low-variance noncausal LSE because values cluster
> at ~`log(#keys)`. The real gate is an **absolute-error** gate, with PCC ≥ 0.98 only as a sanity
> check.

## The identity being exploited

`LSE_row = scale · m_raw_row + log(l_row) = logsumexp_k(scale · Q·Kᵀ)`, where `m_raw` is the RAW
un-scaled per-row max and `l` is the running softmax denominator — both already computed inside the
shared flash compute, and both already emitted by the ring-joint path.

The scale convention carries over from the ring path to the streaming path because the max reduce
uses the IDENTITY scaler `cb_identity_scale_in` (= 1.0), so the stored max is raw, and the scale is
folded into the **exp** (`exp_packthread_tile_init<true, scale_fp32, ...>` inside
`sub_exp_block_bcast_cols`), not into the max.

The ring path can alias its running-max CB as the LSE CB; the plain streaming path **cannot** — see
[refuted list](../REFUTED.md). Hence a dedicated fp32 `cb_lse_out`.

## Two bugs the plan got wrong

1. `sdpa_inner_loop_step` ALSO needed the `emit_lse` / `cb_lse_out` / `cb_scale_in` template params
   (its `normalize_row` lambda forwards them), plus forwarding at the `sdpa_standard_v2` call site.
2. The LSE emit must **COL-REDUCE** `cur_sum_cb` via `matmul(sum, col_identity)` to obtain the
   scalar `l` BEFORE taking `log` — reading the raw front tile captures only one column's partial
   sum and silently loses `log(#keys)`.

## Byte-identity strategy for existing callers

Every added block is `if constexpr (emit_lse / return_lse)` on device or `if (attrs.return_lse)` on
host; no new CB is allocated when off; compute and writer compile-arg lists **APPEND at the tail** so
existing indices stay stable; and `ttnn::prim::sdpa` keeps returning element `[0]`.

**LSE output tensor spec:** logical shape `[B, NQH, Sq, 1]`, dtype FLOAT32, layout TILE, with column
0 of the padded 32-wide tile holding the value and columns 1–31 pad.

## Open items and risks that survive

* **SCOPE GAP, still real:** the non-streaming fallback `sdpa_standard` (deferred, non-RING) is NOT
  covered, so a host `TT_FATAL` must reject `return_lse` there. gemma4 uses the streaming path, so
  this is an API gap to document rather than a blocker.
* **LIVE-CODE DISCREPANCY, flagged and never resolved:** the task brief said grid `(8,4)`, but the
  live gemma4/DG denoise+prefill SDPA program config in `tt/diffusion_attention.py:72-92` uses
  `grid = CoreCoord(8, 1)` with `q_chunk = k_chunk = 32` for `head_dim >= 512`. Reconcile before
  touching the geometry.
* **Risk that materialized:** `cb_lse_out` is Float32 while the stats/intermediate CBs are
  Float16_b, so the emit needs `pack_reconfig_data_format(cb_lse_out)` around it and a restore
  afterwards — a missing reconfig gives garbage LSE AND corrupts subsequent bf16 packs.
* **Risk to re-check on any change:** `cur.max` is indexed ABSOLUTELY (`lse_row_offset + s`) while
  `cur.sum` is read at its CURRENT FRONT because prior row-groups popped it, so the
  `lse_row_offset += sbh` cadence must track the `normalize_row` call sequence including the
  remainder row-group. An off-by-`sbh` silently MISLABELS LSE rows — clone the proven
  `sink_row_offset` accounting verbatim.
* **L1/CB budget at head_dim 512** on the streaming path (`DHt = vDHt = 16`, where `qk_im` and
  `out_im_A/B` dominate) was called the single riskiest unknown. The added CBs are only ~4–8 KB, but
  the fit must be checked on the exact gemma4 chunked-prefill shape.

## The flash-merge identity any consumer must use

`l = logaddexp(l1, l2)`; `o = o1·exp(l1−l) + o2·exp(l2−l)`, asserted against a full-KV
`out_full`/`lse_full`. This is the equivalence test in the T6 unit test and the identity implemented
by `tt/attention_merge.py::merge_attention_partials` (task T7, device-verified,
`tests/test_attention_merge.py` 3/3).

## Reproduction (after re-landing the extension)

env: see [plan](../../plan.md). The extension and its test are **not in the tree** — restore them
first with `git checkout 2e18c599bd3 -- ttnn/cpp/ttnn/operations/transformer/sdpa/
models/experimental/diffusion_gemma/tests/test_return_lse.py`, then:

```bash
ninja -C build ttnn        # `build` -> `build_Release`
cp build/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
pytest tests/test_return_lse.py
```

A stale `_ttnn.so` will keep accepting `return_lse=True` and mask whether the restore worked, so the
copy is not optional.

The compute kernel is JIT and recompiles on the next device run.

**DEVICE HYGIENE:** streaming SDPA at head_dim 512 is L1-tight, and a trace-region or CB overflow
poisons the device — recover with `tt-smi -r`; eth core 29-25 is the recurring QB2 offender (full
recipe: [plan](../../plan.md)).
