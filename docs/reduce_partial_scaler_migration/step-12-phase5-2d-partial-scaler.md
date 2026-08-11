# Step 12 — Phase 5: the 2-D partial scaler feature has almost no consumers

**Outcome: no code change, and the feature is not recommended.** Phase 5 was scoped on the README's
original claim that "eight kernels mask *both* axes … closing that gap is a new helper feature (2-D
partial support)". Enumerating those kernels and applying the step-8 rule to each shows the feature
would unblock **one** of them, and for that one the mechanism is not expressible with scaler tiles.

## Who the dual-axis kernels actually are

`generate_mask_h_w` has these callers (via their readers):

| Kernel | Reduce shape | Verdict |
|---|---|---|
| `moreh_layer_norm_backward_input_grad_small` | `add_tiles(cb_dyadd, …)` → `reduce(single())` over `cb_dyadd` | **Blocked by step 8** — accumulate-then-reduce |
| `moreh_layer_norm_backward_input_grad_large` | same, `cb_dyadd` / `cb_ydyadd` | **Blocked by step 8** |
| `moreh_group_norm_backward_input_grad_{small,large}` | — | **Same two kernels**: the group-norm factory sets `compute_kernel_file` to the *layer-norm* backward input-grad kernels |
| `moreh_clip_grad_norm_step1` | `add_tiles(cb_correct_xpow, cb_xpowadd)` → `reduce(single())` over `cb_xpowadd` | **Blocked by step 8** |
| `moreh_bias_backward_hw` | one tile per `reduce()` call, `Accumulate` between calls | the only candidate |

So the "eight kernels" are really three distinct compute kernels plus `bias_backward_hw`, and three of
the four are blocked by the accumulation rule — which no scaler feature can fix, because the problem is
that the ragged tile has already been summed into lanes that also carry valid data.

## And for the one candidate, scaler tiles cannot express it

`moreh_bias_backward_hw` reduces `REDUCE_SCALAR` (over H and W together), one tile per call, so it is
structurally fine — it is the same per-call form that made `bias_backward_h` migratable in step 10. The
problem is what a corner tile needs.

`REDUCE_SCALAR` applies the scaler **twice**, row pass then col pass, from a single scaler CB index. Two
independent pieces of evidence for that: the helper's own `static_assert` comment, and
`prepare_reduce_scaler`, which uses `1/sqrt(N)` for `AVG` + `REDUCE_SCALAR` precisely so that the two
applications multiply back to `1/N`.

The partial fill (`fill_each_face_row0_partial`) writes one 16-value vector per face into row 0, and the
LLK indexes that vector **by column for the row pass and by row for the col pass**. So a corner tile
needs `weight(r, c) = [r < mask_h] · [c < mask_w]`, but a single tile can only supply `S(r) · S(c)` for
one vector `S`. That is representable only when `mask_h == mask_w` — and there, usefully, a 0/1 mask
survives being squared (`0² = 0`, `1² = 1`, and the SUM scaler is `1.0`).

Adding a second scaler index to `ReducePartialScaler` does not help: `reduce_tile` takes one scaler tile
index per call and uses it for both passes, so the two axes cannot be given different vectors without a
new LLK entry point.

The remaining option is to **decompose** the ragged 2-D reduce into W-then-H, two 1-D reduces each with
its own partial scaler. The host already does exactly that for Int32 (`"host decomposes Int32 HW reduce
into W-then-H"`). But that costs an intermediate buffer and a second pass, to replace a mask CB and one
`mask_tile` per ragged tile — almost certainly a net loss, and a per-op rewrite rather than a helper
feature.

## Recommendation

Do not build the 2-D partial scaler. Its consumer list is one kernel, and that kernel needs either an
LLK change or a decomposition that costs more than the mask it would remove.

If someone wants the narrow win later: `REDUCE_SCALAR` with `mask_h == mask_w` **is** expressible with a
single partial tile. That would need the `REDUCE_SCALAR` assert relaxed to allow it under that condition,
plus a `toy_reduce_partial` case to confirm the double-application model on device before any op uses
it. It is a micro-feature with one conditional consumer, not the Phase 5 that was scoped.

## What this means for the stated end goal

The README's reachability section needs both limits now, and the 2-D one is the *smaller* of the two:

- **Accumulate-then-reduce** (step 8, step 10, this step) — every moreh norm-style reduce and the three
  dual-axis compute kernels. Permanent unless the kernels are restructured to reduce tiles directly,
  which would cost the L1 residency that made them accumulate in the first place.
- **2-D corner masks** — one kernel, needing an LLK-level change.

Everything else that was reachable has been migrated.
