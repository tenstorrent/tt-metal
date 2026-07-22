# Blaze Models Prefill CI Timeout Bisect — Handoff

Investigating why three jobs in `.github/workflows/blaze-models-prefill-tests.yaml`
(matrix defined in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`) started
timing out on `main` around 2026-07-15: **GLM MoE**, **Kimi Prefill Block**, and
**DSA MLA (GLM-5.1 + GLM-5.2 indexer reuse)**.

This analysis was done from a machine without access to the target hardware, so
findings below are based on CI log inspection and diff review only — **none of the
hypotheses have been verified with an actual rerun on hardware.** That's the next
step for whoever picks this up.

## Method

Total job/step duration in the GitHub Actions UI is misleading — it includes
allocation/cluster-reset overhead that varies run to run. Instead, pull raw logs via:

```
gh api repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs
```

(plain `gh run view --log` / `--log-failed` returns empty for this repo's jobs — the
API endpoint is what works). Parse the `tracy:signpost:49 -` lines (`MLA_START/END`,
`MoE_START/END`, `tt_forward_START/END`) to split each run into **compile/setup
phase** vs **device-execution phase**, then compare that phase breakdown across many
runs/days to see what's flat vs what's moving. Reuse this technique if continuing
the investigation.

## Findings per job

### 1. GLM MoE

Permanently broken since 2026-07-15 21:11, commit `30f8355124a` ("Enable single
galaxy tests for blaze prefill models on exabox") — moved this job from
`bh_galaxy`/single-process pytest to `bh_spsg`/`bh_sc1` exabox runners executed via
`mpirun --pernode`.

- Compile phase is rock-steady ~449–457s across every checked run 07-18→07-21 (was
  286s pre-migration — a persistent 1.6x, unmoving). Looks like a structural
  exabox/compile-host characteristic, not a bisectable main-branch commit.
- Its 20-min timeout budget was never raised to compensate (unlike the other two
  jobs below).
- Separately, the 07-22 run failed via a distinct hardware chip-reinit flake
  (`Error when re-initializing chips!`, retried ~14min in the "Reset cards and
  validate cluster" step) — unrelated to the compile/timeout story, don't conflate
  the two.

**Open question, unresolved:** is `bh_spsg`/`bh_sc1` (exabox) literally different
physical hardware from `bh_galaxy`, or the same boxes relabeled under a different
SKU name? This matters for whether the regression is fixable in software at all.
Nobody has confirmed this yet.

### 2. Kimi Prefill Block

Same initial 07-15 regression (mpirun/exabox migration; budget raised 25→30min to
compensate, but not enough). Recovered transiently by 07-18 (fast pass, 10:45), then
regressed again *permanently* from 07-20 onward on the *same* SKU (`bh_sc1`) — this
rules out infra as the second cause.

- Compile phase (collect → first `MLA_START`) dropped ~2x between 07-20 (~660–730s)
  and 07-21 (~315–325s).
- Attributed to `97a2898` "ds_prefill - Fuse tilize with unified routed expert for
  Blackhole" (#49744) — Kimi Prefill Block exercises the MoE/routed-expert path this
  touches.
- **Ruled out** `928526c` "sparse_mla: fold single-shot onto block-cyclic path"
  (#49719), even though it landed in the same window: verified line-by-line that
  every changed branch in `mla.py` is gated on `self._has_indexer`, which resolves
  `False` for Kimi (`KimiK26Config` carries none of the DSA/indexer config fields,
  so `resolve_has_indexer()` falls through to dense), and the dense-path expressions
  are textually unchanged before/after the diff. The test-file diff for that PR also
  only touches `test_glm_prefill_block`, never `test_kimi_prefill_block`.
  GLM MoE's compile time did *not* move in the same window, which is consistent with
  the surviving attribution being a specific code change rather than a general infra
  improvement.
- Kimi still times out overall even after this recovery — the job runs two
  sequential pytest invocations, each internally budgeted 30 min, and even at the
  improved rate the total exceeds the job's 30-min step budget.

**Verification needed:** revert just `97a2898` on top of current `main` and rerun —
this is currently an inference from timing + diff review, not a confirmed A/B result.

### 3. DSA MLA (GLM-5.1 + GLM-5.2 indexer reuse)

Flat and stable across 7 runs from 07-15 to 07-20 (compile1 ≈196–205s, exec1
≈133–148s, all 14 `MLA_START`/`END` pairs completing every time, both passes and the
two non-timeout fast-fails).

Then a sharp, permanent step between 07-20 08:19 (last good) and 07-21 11:47 (first
of the still-ongoing timeouts): compile1 jumps to ~273s (+34%) and **exec1 triples to
~402–406s**, with only 10/9 of the 14 pairs completing before hitting the 16-min
budget.

Root cause pinned with high confidence: commit
`d8b9784e5520e48fda53f5fbd26174865fc745eb` "ci: migrate sparse/DSA MLA + GLM prefill
block tests to fabric2d" (#50221, 2026-07-20 17:33) switched this exact test's
`DS_SPARSE_FABRIC` from `line` to `fabric2d` (confirmed via `git show` — only the env
var + `-k` selector changed in that commit; **no timeout/SKU values were touched**).

**Verification needed:** concrete testable hypothesis is `line` vs `fabric2d` fabric
on identical code — rerun the test with `-k "...and line"` vs `-k "...and
fabric2d"` on the same hardware rather than assuming this from the diff alone.

## Next steps for whoever has hardware access

1. **Kimi Prefill Block** — revert `97a2898` on `main`, rerun `test_kimi_prefill_block`,
   confirm compile phase reverts to the slower ~660–730s (or stays fast, disproving
   the attribution).
2. **DSA MLA (GLM)** — rerun with `DS_SPARSE_FABRIC=line` vs `DS_SPARSE_FABRIC=fabric2d`
   on identical code to confirm the exec1 3x slowdown is fabric-caused.
3. **GLM MoE** — determine whether `bh_sc1`/exabox is different physical hardware
   from `bh_galaxy`, or the same boxes under a different label/SKU name. This gates
   whether the 1.6x compile-phase regression is fixable at all versus being an
   accepted cost of the exabox migration (in which case the timeout budget just needs
   raising instead).
4. Once each cause is confirmed, decide fixes: revert/patch the offending commit
   (Kimi), pick the correct fabric mode or accept the cost and raise the timeout
   (DSA MLA), and either fix or budget around the exabox compile overhead (GLM MoE).
