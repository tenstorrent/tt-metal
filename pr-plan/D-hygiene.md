# PR-plan front D-hygiene (2026-08-17)

## Verdict

The branch cleanly separates into shippable code (op/LLK/routing/model/test code is near-upstream-quality, SPDX-complete, free of internal tracker references) and a large campaign layer (~50k+ lines under paper-topk/, root docs, ledger/sweep tooling, baselines) that must be excluded wholesale. History cannot be carried: 89 interleaved commits with campaign-narrative messages, non-Tenstorrent author emails, Claude co-author lines, and one deleted-fabricated-harness commit — build each upstream PR as fresh squashed commits cherry-picked/patched onto origin/main. Six concrete in-file cleanups are required before any PR (one kernel comment citing an internal evidence path, two "PR2 triage" comments, a stale perf-pin table, a Python/C++ KEEP-IN-SYNC constant drift, SPDX entity normalization), plus reverting an uncommitted supply-chain-pin downgrade in .github/workflows/package-and-release.yaml. The contract suite and sweep-framework large_k suite are shippable assets after excising the runbook and campaign baselines; the canonical sweep script and ledger renderer are campaign-internal.

## Analysis

AUDIT BASE: merge-base 50a82f835593, branch nkapre/sorting, 89 commits at time of audit (HEAD moved twice DURING the audit — 445760b81dc → f2ba9df49e9 → amended to 9d7fd5f5ac6 "fused-u16 end-to-end merge/rebuild" — other agents are actively committing; re-snapshot before final PR prep). 228 files, ~64.7k insertions at the 445760b measurement point.

=== 1. EXCLUSION LIST (never reaches upstream) ===

ROOT CAMPAIGN DOCS (all new at repo root):
- /home/nachiket/tt-metal/HANDOFF.md — campaign state-of-play doc (commit 1e22b9704a9 "honest state of play").
- /home/nachiket/tt-metal/SORTING.md — architectural-discovery log (referenced 20+ times by THRESHOLD_SELECT_DESIGN.md).
- /home/nachiket/tt-metal/RADIX_BUCKET_GPU.md — literature dossier.
- /home/nachiket/tt-metal/TOPK_LEDGER.html — campaign ledger artifact source.
- /home/nachiket/tt-metal/ci.sh — self-declares campaign-internal: "The branch is local-only: this script is how 'CI ran' until a push is permitted"; hardcodes /tmp/tt-device.lock, tt-llk .venv tt-smi path.

paper-topk/** (ENTIRE TREE, ~100 files): IPDPS draft, briefs, evidence archives with machine-local logs, trials.tsv, dprint dumps. Includes LaTeX build junk that should never have been committed even for the paper: main.aux/.log/.fls/.fdb_latexmk/.bbl/.blg/.out, main.pdf, refs.bib.bak. Evidence files contain internal paths and campaign pin data (e.g. paper-topk/evidence/fused-u16/ added in HEAD commit 9d7fd5f alongside shippable kernel code — the fused-u16 PR extraction must drop the evidence/ half of that commit).

CAMPAIGN TOOLING UNDER tests/ (underscore-prefixed, pytest-invisible, but still repo pollution):
- tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py — hardcodes REPO fallback "/home/nachiket/tt-metal" (line 135), BLAZE_ROOT="/home/nachiket/tt-blaze" (line 308), pre_branch_us campaign pins (lines 422, 439, 470, 487 — "scenarios7 pinning", "I3 landing A/B"). Campaign-internal. DECISION: exclude; its CI role is superseded upstream by the sweep-framework large_k suite (which IS shippable, see §6).
- tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py — exists solely to render TOPK_LEDGER.html. Exclude.
- tests/ttnn/unit_tests/operations/reduction/_topk_sort_bench.py — campaign baseline script (header: "That number is meaningless without knowing what the shipping ops cost"; references campaign LLK bench). Exclude (could ship later as a standalone bench if rewritten).
- tests/ttnn/unit_tests/operations/reduction/baselines/** (comp3, comp4, scope51, smallk_routefix — 11 CSV/JSON/MD files) — campaign pin snapshots consumed by the sweep's pre_branch/regression layers. Exclude.
- tests/ttnn/unit_tests/operations/reduction/TOPK_CONTRACT_RUNBOOK.md — hardcodes /home/nachiket/tt-metal (lines 9, 18), says "per campaign run" (line 86). Exclude, or rewrite to a short header comment inside the test file.
- tests/ttnn/nightly/unit_tests/operations/experimental/_topk_large_indices_{bench,gate_ab.sh,skip_adversarial,skip_debug,skip_diag,skip_hangbattery,skip_telemetry_parse}.py and _topk_routed_bench.py — campaign bench/debug harnesses; gate_ab.sh references "charter item I4" and sed-edits shipping headers per arm. Exclude all 8. (If upstream wants a Tracy bench, _topk_large_indices_bench.py is the only rewrite candidate.)

tt-llk CAMPAIGN MATERIAL:
- tt_metal/tt-llk/tests/docs/THRESHOLD_SELECT_DESIGN.md — commit f9742745551 says "shelved pending go-ahead"; cites SORTING.md ~20x and HANDOFF.md ~4x; unshippable without those docs. Exclude.
- tt_metal/tt-llk/tests/sources/CGTCEQ_RUNBOOK.md — hardcodes "cd /home/nachiket/tt-metal/tt_metal/tt-llk/tests" (line 26). Exclude or fix path to $TT_METAL_HOME-relative.

tt_metal/programming_examples/risc_scan_bench/** (4 files + the CMakeLists.txt hunk adding add_subdirectory(risc_scan_bench)): headers say "prices every RISC-side materialization candidate of the top-k selector campaign (see RADIX_BUCKET_GPU.md gate 2 and the storm/research reports)" (risc_scan_bench.cpp:7-9, kernels/scan_bench.cpp:7-10; RUNBOOK.md similar). DECISION: exclude by default (campaign instrument, not a teaching example — programming_examples is a tutorial directory); revert the one-line CMakeLists.txt hunk with it. If the user wants it upstream as a microbench, it needs header rewrite + relocation (tests/tt_metal micro-benchmarks, not programming_examples) — a separate opt-in PR.

=== 2. UNTRACKED / STATUS JUNK (none committed — keep them out) ===
- lx-reset — tt-smi reset JSON for host "primeradiant", dated 2025-08-26. Delete or move out of repo.
- ttnn/ttnn/_ttnn.so.release — 13.9 MB binary. NOT covered by .gitignore's "*.so" (extension is .so.release). Delete or rename to *.so; add ignore if the rename convention persists.
- tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml — locally generated WH N150 descriptor, unrelated to the BH top-k work. Do not commit.
- tt_metal/programming_examples/eltwise_poly/ — unrelated tutorial experiment (SPDX "© 2025 Tenstorrent Inc.", old entity). Do not commit with this campaign.
- .github/workflows/package-and-release.yaml (working-tree modification, OURS, uncommitted): replaces the SHA-pinned action `pypa/gh-action-pypi-publish@ed0c539a...# v1.13.0` with floating tag `@v1.13.0` — a supply-chain-pin security DOWNGRADE, almost certainly a local convenience hack. Action: `git checkout -- .github/workflows/package-and-release.yaml`. Must never ship.

=== 3. INTERNAL REFERENCES INSIDE SHIPPABLE FILES (fix before PR) ===
Systematic greps over every changed file under ttnn/, tt_metal/hw, tt_metal/tt-llk, models/, tests/ (excluding paper-topk) for: campaign names, /home/nachiket, primeradiant, SORTING/HANDOFF/TOPK_LEDGER, PR-plan numbering, "gate N", "front A-E", "charter", owner emails, TODO/HACK/XXX, first-person tone. Findings in files that SHIP:
(a) ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/topk_large_indices_chunk_skip.hpp:174 — comment "Calibration data: scratchpad storm/tileskip/diag/dprint_k{512,1024,2048}.txt" points at an internal scratchpad path (data now lives in paper-topk/evidence/tileskip/diag/, also excluded). FIX: replace with a self-contained one-line description of the calibration method.
(b) tests/ttnn/unit_tests/operations/data_movement/test_gather.py:275 — "(PR2 triage)" — internal campaign PR-plan numbering. FIX: rewrite as "found while validating routed large-k ttnn.topk".
(c) tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py — one "PR2 triage repro (minimal)" comment. Same fix.
(d) models/common/sampling/_utils.py and sampling_1d.py cite C++ line ranges ("topk.cpp:258-320", "topk.cpp:338-343") — will rot; soften to function names (minor, reviewer-taste).
Everything else is clean: the only GH issue numbers found are PUBLIC upstream trackers (tenstorrent/tt-metal#33492, #44246, #16439, #50215, tt-llk#1340) — fine. "blaze" mentions in tt_metal/hw/.../ckernel_sfpu_sampling.h, softmax_k.h, eltwise_mul_scalar.h are PRE-EXISTING upstream code (not in our diff). No TODO/HACK/XXX or first-person comments in our shipping C++/kernels. The "ledger" in test_topk_contract.py is the suite's own divergence-JSONL mechanism (env TOPK_CONTRACT_LEDGER, default under generated/) — legitimate, not the campaign ledger.

=== 4. REAL CODE DEFECT FOUND DURING AUDIT ===
KEEP-IN-SYNC drift: models/common/sampling/_utils.py:65 `_TOPK_ROUTE_K_MULTIPLE = 16` vs ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:261 `constexpr uint32_t large_k_route_k_multiple = 32` (changed by the TILE-output commit 482de67d779; mirror not updated). Both use it for the `k_rounded <= width` envelope (_utils.py:111 vs topk.cpp:355). Drift is fail-safe by the module's own documented design (mirror-True/C++-False → stock path without indices_tensor), but it violates the file's explicit "KEEP IN SYNC" contract for k in ranges where 16- and 32-rounding straddle the width bound. FIX to 32 before the model-side PR.

=== 5. LICENSE HEADERS ===
All new .py/.cpp/.h/.hpp files in shippable areas carry SPDX pairs — zero source-file gaps. Gaps only in non-source files (CSVs, MDs, one .sh, risc_scan_bench CMakeLists — and existing programming_examples CMakeLists carry no SPDX either, so no issue). INCONSISTENCY: entity mix — most new files use "© 2026 Tenstorrent AI ULC" (matches upstream-current convention, e.g. the pre-existing topk_large_indices.hpp), but test_topk_contract.py, _topk_sort_bench.py, _topk_large_indices_gate_ab.sh, and risc_scan_bench/* use "© 2026 Tenstorrent USA, Inc.", and _topk_large_indices_bench.py uses "© 2026 Tenstorrent USA, Inc." variant formatting. Normalize shipping files to "Tenstorrent AI ULC".

=== 6. UPSTREAM-OWNED TESTS WE MODIFIED — acceptability ===
- tests/ttnn/unit_tests/operations/reduce/test_topk.py: +139 lines, pure additions, BH-gated via is_blackhole, value-exact + index-validity assertions with tie semantics documented. UPSTREAM-READY as-is.
- tests/sweep_framework/sweeps/reduction/topk/topk.py: adds "large_k" suite + invalidations; clean, well-reasoned. UPSTREAM-READY (this is the shippable CI-coverage asset; note it has not been proven in an actual sweeps-infra run — flag in PR).
- tests/ttnn/unit_tests/operations/data_movement/test_gather.py: +35 lines, a genuine RM-gather alignment repro with suspected-mechanism analysis. UPSTREAM-READY after the "(PR2 triage)" scrub; consider filing the suspected RmSingleRowMultiCore reader-alignment bug as a real upstream issue and citing it.
- tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py: +831/-3. The -3 is a SEMANTIC RELAXATION of test_topk_large_indices_valid_length_matches_sliced_input: previously asserted bit-identical indices between valid_length=L and physically-sliced input; now asserts tie-tolerant equality (index diffs must be ties; value multisets equal). Defensible for the multi-core path but it weakens an existing guarantee with NO replacement comment explaining why — add the tie-order rationale comment and call the relaxation out explicitly in the PR body. Also: test_topk_large_indices_production_perf_check carries expected_duration_ns pins that ci.sh's own header declares "stale post-P-cap" and "IOMMU-runner-only" — shipping as-is creates an immediately-failing or environment-fragile perf test. Re-baseline on device before PR (blocked now: no device runs permitted), or split the perf-pin table update into its own commit with fresh numbers.
- Sampling "tests": no sampling test files are modified on the branch (only models/common sampling SOURCE: sampling_1d.py, _utils.py, tt_sampling.py — all upstream-quality prose). The I5 relaxation (575ff18) and regather skip (fe1930d) therefore ship with NO new test coverage in-tree beyond the ledger's CI battery (which is excluded) — the model-side PR needs its parity evidence restated in the PR body, and fe1930d is explicitly PENDING 8x4 validation (do not include it until that gate closes).

=== 7. COMMIT-MESSAGE HYGIENE → SQUASH-VS-CARRY ===
Verdict: SQUASH per PR, mandatory, for five independent reasons:
(1) All 88+ commits carry "Co-Authored-By: Claude Fable 5 / Claude Opus 5 (1M context) <noreply@anthropic.com>" (67+20+ split).
(2) Author identities are nachiket@gmail.com (20) and nachiket@uwaterloo.ca (68+) — neither is nkapre@tenstorrent.com; org CLA/DCO and membership checks will flag.
(3) Messages are campaign-narrative and first-person: "BEAT topk_xl...", "Hypothesis refuted: both my topk flags fail", "Salvage in-flight work; two-thread SFPU floor-break is dead on paper", "RETRACT the end_phase=log(K)-1 'free win'".
(4) History interleaves excluded material with shippable code at fine grain (Ledger/Paper/Evidence commits between op commits; commit 9d7fd5f mixes fused-u16 kernels with paper-topk/evidence files), so no contiguous carry-able range exists.
(5) History contains commits upstream should not see: 3da04d5726e "Delete fabricated benchmark harness; repoint Phase 0 at the real one" and fba8f4847d8 "force-add the gitignored G4 curve + gate ablation CSVs".
Mechanics: for each planned PR, `git checkout -b <pr-branch> origin/main`, apply the relevant file set via `git diff 50a82f...HEAD -- <paths> | git apply` (or cherry-pick then squash), commit fresh with tenstorrent identity and an upstream-toned message. tt-llk caveat: tt_metal/tt-llk is VENDORED here (no .git, files tracked in tt-metal), but tenstorrent/tt-llk remains a separate repo synced into tt-metal — confirm the current contribution flow before deciding whether ckernel_sfpu_topk.h / ckernel_sfpu_topk_xl.h (+ replay commit d64993c, SFPLOADMACRO 7257487, fused-u16) and the tt-llk tests go via a tt-llk PR or directly; a tt-metal-side edit of the vendored copy risks being clobbered by the next sync.

=== 8. PER-FILE CLEANUP ACTION SUMMARY (shippable set) ===
1. topk_large_indices_chunk_skip.hpp:174 — remove internal scratchpad path from comment.
2. test_gather.py + test_topk_large_indices.py — scrub "PR2 triage" (2 sites).
3. test_topk_large_indices.py — re-baseline expected_duration_ns pins (device run required); add tie-order comment at the relaxed valid_length assertion.
4. models/common/sampling/_utils.py:65 — _TOPK_ROUTE_K_MULTIPLE 16 → 32 (sync with topk.cpp:261).
5. SPDX entity normalization to "Tenstorrent AI ULC" on test_topk_contract.py (and any other shipping "USA, Inc." files).
6. Optional: soften hardcoded line-number cross-references in sampling_1d.py/_utils.py comments to function names.
7. `git checkout -- .github/workflows/package-and-release.yaml` (revert pin downgrade).
8. Delete/relocate untracked: lx-reset, ttnn/ttnn/_ttnn.so.release, n150_mesh_graph_descriptor.yaml, eltwise_poly/ (or park them outside the repo).
9. When extracting the fused-u16 PR from 9d7fd5f5ac6, drop its paper-topk/evidence/fused-u16/ files.
10. Ship test_topk_contract.py WITHOUT TOPK_CONTRACT_RUNBOOK.md (or fold a 5-line usage note into the test docstring); ship the sweep-framework large_k suite; exclude _canonical_topk_sweep.py, _topk_ledger_render.py, baselines/**.

## Evidence

- git diff --name-status 50a82f835593..HEAD: 228 files, 64,733 insertions (measured at HEAD=445760b81dc; HEAD has since advanced to 9d7fd5f5ac6)
- ci.sh header lines 1-28: 'The branch is local-only: this script is how "CI ran" until a push is permitted'; also documents 'production_perf_check cells: IOMMU-runner-only, and their expected_duration pins are stale post-P-cap'
- Working-tree diff .github/workflows/package-and-release.yaml: '-uses: pypa/gh-action-pypi-publish@ed0c539abdc6b55ad89f3ea7b8e96860a5ddef80 # v1.13.0' -> '+uses: pypa/gh-action-pypi-publish@v1.13.0' (uncommitted, pin downgrade)
- lx-reset: tt-smi reset JSON, host_name 'primeradiant', time 2025-08-26; ttnn/ttnn/_ttnn.so.release = 13,878,424 bytes, not matched by .gitignore '*.so' (line 85)
- topk_large_indices_chunk_skip.hpp:174: '// scratchpad storm/tileskip/diag/dprint_k{512,1024,2048}.txt.' — only campaign reference surviving in shipping C++ (grep over all changed .cpp/.hpp/.h under ttnn/, tt_metal/hw, tt_metal/tt-llk)
- test_gather.py:275 '(PR2 triage)' and test_topk_large_indices.py 'PR2 triage repro (minimal)' — internal PR-plan numbering in shipping tests
- _canonical_topk_sweep.py:135 REPO fallback '/home/nachiket/tt-metal'; :308 BLAZE_ROOT='/home/nachiket/tt-blaze'; :422/:439/:470/:487 pre_branch_us campaign pins; TOPK_CONTRACT_RUNBOOK.md:9,18 '/home/nachiket/tt-metal', :86 'per campaign run'; CGTCEQ_RUNBOOK.md:26 'cd /home/nachiket/tt-metal/...'
- THRESHOLD_SELECT_DESIGN.md cites SORTING.md ~20x and HANDOFF.md 4x; commit f9742745551 message: 'shelved pending go-ahead'
- risc_scan_bench.cpp:7-9: 'prices every RISC-side materialization candidate of the top-k selector campaign (see RADIX_BUCKET_GPU.md gate 2 and the storm/research reports)'; CMakeLists.txt hunk adds add_subdirectory(risc_scan_bench)
- KEEP-IN-SYNC drift: models/common/sampling/_utils.py:65 '_TOPK_ROUTE_K_MULTIPLE = 16' vs ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:261 'constexpr uint32_t large_k_route_k_multiple = 32'; both feed the k_rounded<=width envelope (_utils.py:111, topk.cpp:355)
- Commit hygiene: 88 commits at audit time, 87 with Co-Authored-By (67 'Claude Fable 5', 20 'Claude Opus 5 (1M context)'); authors 20x nachiket@gmail.com + 68x nachiket@uwaterloo.ca (user is nkapre@tenstorrent.com); messages include 'Hypothesis refuted: both my topk flags fail', 'Salvage in-flight work', 'RETRACT the end_phase=log(K)-1 free win', 3da04d5726e 'Delete fabricated benchmark harness', fba8f4847d8 'force-add the gitignored G4 curve'
- SPDX: zero missing headers on new .py/.cpp/.h in shippable areas (script check); entity mix — 'Tenstorrent AI ULC' (kernels, tt-llk tests, matches pre-existing topk_large_indices.hpp) vs 'Tenstorrent USA, Inc.' (test_topk_contract.py, _topk_sort_bench.py, gate_ab.sh, risc_scan_bench)
- test_topk_large_indices.py diff: only 3 deleted lines = 'assert_equal(bounded_t, sliced_t)' + bit-identical comment, replaced by tie-tolerant value-multiset assertions (semantic relaxation of an existing upstream test, no explanatory comment added)
- Issue references in shipping code are all public upstream trackers: tt-metal#33492, #44246, #16439, #50215, tt-llk#1340 (tt_sampling.py:652-668, ckernel_sfpu_topk.h:134/653/965/1020, sfpu_loadmacro_issue_perf.cpp:133)
- tt_metal/tt-llk has no .git and its files are tracked directly in tt-metal (git ls-files -s shows mode-100644 blobs) — vendored, not a submodule; LLK core changes: ckernel_sfpu_topk.h, experimental/ckernel_sfpu_topk_xl.h (commits d64993c, 725748766a6, 9d7fd5f)
- HEAD moved during audit: 445760b81dc -> f2ba9df49e9 (fused-u16 code under an evidence-titled message) -> amended to 9d7fd5f5ac6 with correct message; 9d7fd5f mixes shipping kernel code with paper-topk/evidence/fused-u16/ files

## Risks

- Branch is a moving target: other agents committed twice during this audit (including one amend). Any PR-extraction file list must be regenerated from the final frozen HEAD, or extractions will silently miss late commits (e.g. fused-u16 landed only mid-audit).
- test_topk_large_indices_production_perf_check ships with expected_duration_ns pins that ci.sh itself declares stale post-P-cap and IOMMU-runner-dependent — upstream CI would fail or flake immediately; re-baselining requires device runs, currently forbidden.
- fe1930d (regather skip) is PENDING 8x4 validation per the campaign ledger — including it in the model-side PR before that gate closes ships an unvalidated multi-chip behavior change.
- tt-llk contribution flow unverified: tt-llk is vendored into tt-metal here, but if upstream still develops LLK in tenstorrent/tt-llk with periodic sync, direct tt-metal edits to tt_llk_blackhole headers could be clobbered by the next sync or rejected in review; confirm the flow before splitting PRs.
- The valid_length assertion relaxation weakens an existing upstream test guarantee; if a reviewer reads it as hiding a regression rather than documenting multi-core tie semantics, it endangers the whole op PR — must be explicitly justified in the PR description.
- Anthropic Co-Authored-By lines and non-corporate author emails across all history make accidental carry (a plain `git push` of the branch) a disclosure/CLA problem in itself — the no-push rule is also protecting hygiene here.
- The _utils.py k-multiple drift (16 vs 32) is documented fail-safe, but only under the assumption that the stock path without indices_tensor is behaviorally identical for sampling callers; if that assumption is wrong for any edge shape, the drift becomes a silent behavior change — fix rather than rely on the fail-safe argument.
- Excluding _canonical_topk_sweep.py removes the branch's only one-command regression harness; the upstream story then rests on test_topk_contract.py + the sweep-framework large_k suite, the latter never yet executed on real sweeps infrastructure.
