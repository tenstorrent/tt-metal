# Upstream PR plan (synthesized from the 4-front analysis swarm, 2026-08-17)

Frozen analysis HEAD: 9d7fd5f5ac6 (re-derive file lists from the FINAL head
before extraction — the branch moved during the audit). Full per-front
detail: A-inventory / B-conventions / C-prereqs / D-hygiene in this dir.

## Where: ONE repo — tenstorrent/tt-metal

tt-llk is an in-tree subtree since PR #41932 (standalone repo frozen,
no pushes since 2026-04-11); precedent #51777 landed LLK + ckernels changes
in a single tt-metal PR. No tt-llk-repo PR exists to file.

## How many and in what order: SEVEN

| # | PR | size | owner (CODEOWNERS) | depends on | state |
|---|----|------|--------------------|------------|-------|
| 1 | stock ttnn.topk silent value corruption fix (topk_final.cpp, 8 lines + regression test carved from test_topk.py:411 block) | tiny | metalium-developers-mmfusedreduce | none | READY after test carve-out — submit first (silent-corruption urgency) |
| 2 | LLK replay for stock topk (ckernel_sfpu_topk.h +283; 1.15–1.20x single-core) | small | mmfusedreduce | none | READY |
| 3 | LLK topk_xl: SFPLOADMACRO merge/rebuild + fused-u16 primitives (+~560 across tt-llk header, ckernels wrapper, compute API) | small-med | LLK owners (precedent #51777 reviewers) | none (additive, behind entry points) | READY |
| 4 | topk_large_indices op line: tree factory, values, TILE I/O + u16, valid_length<k, num_slices, chunk skip, P-cap, multi-rect + hybrid, fused-u16 kernels + op suite (+~5k) | LARGE — consider splitting 4a (pre-hybrid features) / 4b (multi-rect+hybrid+fused) if reviewers balk | metalium-developers-sdpa | PR-3 | BLOCKED on IOMMU pin re-baseline (see blockers) + scrub list |
| 5 | ttnn.topk routing (+307 topk.cpp) + routed tests + sweep large_k suite (+ contract suite as 5b if the flake is resolved) | medium | mmfusedreduce | PR-4 (uses return_values/tile_output/index_dtype params) | BLOCKED on PR-4 + nan_bf16 flake quarantine for 5b |
| 6 | sampling call-site relaxation (575ff18: sampling_1d.py, tt_sampling.py, _utils.py — hand-split _utils.py from the fdb81ed gate hunks into PR-5) | small | models owners | PR-5 live in prod | needs the bit-exactness study reconstructed into PR evidence (original lives in session scratchpad) |
| 7 | deepseek_v3_d_p regather skip (fe1930d: indexer.py, mla.py) | small | deepseek model owners | PR-4 (valid_length contract) | HELD: 8x4 Galaxy validation is cron/dispatch-only CI — requires a pushed branch + manual blaze-models-prefill-tests dispatch BEFORE marking ready |

Not shipped, ever: paper-topk/, TOPK_LEDGER.html + renderer, RADIX/SORTING
campaign docs, baselines/** (force-added CSVs), all underscore-prefixed bench
harnesses, canonical sweep script, tt-llk exploration probes (+25k, six of
which hardcode session /tmp paths). Gather-bug repro test (test_gather.py
+35): would FAIL upstream as written — convert to a filed issue + xfail, or
drop from PRs (draft exists in next-fronts/E).

## Mechanics (Front B/D, mandatory)

- Repo squash-merges and reviewers ENFORCE one category per PR: build each PR
  as a FRESH branch off origin/main, populate via `git checkout <HEAD> --
  <paths>` + hand-split the three mixed files (test_topk.py, _utils.py,
  contract-suite fdb81ed hunks), write fresh commit messages. Never
  cherry-pick: code and campaign hunks are interleaved inside single commits
  (482de67: 5/14 files shippable; 1d8e4a2: 6/7 campaign).
- History cannot be carried: campaign-narrative messages, Claude co-author
  lines, non-TT author emails, and one deleted-fabricated-harness commit.
- Fork PRs auto-run sanity but cannot self-dispatch blackhole-post-commit /
  l2-nightly / galaxy pipelines — a codeowner must trigger; plan reviewer
  pings accordingly.
- Accepted kernel-PR sizes here: 4–28 files / 140–4,600 additions. PR-4 sits
  at the top of that band; pre-agree the 4a/4b split option.

## Hard blockers before ANY op PR (Front C)

1. IOMMU perf pins (test_topk_large_indices.py production_perf_check): already
   ~18% below band pre-fused, ~-35% post-fused; they run NIGHTLY on
   bh_p150b_civ2_viommu. Re-pins must be measured on THAT runner class via a
   dispatched pipeline — push-dependent; without them nightly goes red the
   morning after PR-4 merges.
2. Scrub list (Front D, six items): chunk_skip.hpp:174 scratchpad-path
   comment; two "PR2 triage" comments; stale perf-pin table comment;
   Python/C++ KEEP-IN-SYNC constant drift (_utils.py mirror of topk.cpp
   routing predicate — verify against final topk.cpp incl. rect routing);
   SPDX entity normalization on new files.
3. Discard the UNCOMMITTED .github/workflows/package-and-release.yaml change
   (a GitHub Action SHA-unpin = supply-chain downgrade; present in git status
   since session start, not ours, certainly not shippable).
4. Strays: rm lx-reset, ttnn/ttnn/_ttnn.so.release, n150 mesh descriptor
   yaml, eltwise_poly example (or justify each separately).
5. nan_bf16[routed-W10000] state-dependent flake: quarantine or root-cause
   before the contract suite ships; the suite also needs a pipeline yaml
   entry to be live upstream.

## Sequencing reality

PR-1/2/3 are submittable as soon as extraction + scrub is done (local work,
~half a day). PR-4/5 additionally need the pin re-baseline (push-gated).
PR-6 needs evidence reconstruction; PR-7 needs the Galaxy dispatch. Every
push/dispatch/filing step requires explicit owner authorization per the
standing no-push order.
