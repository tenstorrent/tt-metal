# PR-plan front A-inventory (2026-08-17)

## Verdict

The branch (89 commits, 233 files, +65,170/−137 at pinned HEAD 9d7fd5f5ac6) splits cleanly into 7 upstream PRs for tenstorrent/tt-metal only — no tt-llk PR is needed because tenstorrent/tt-llk is frozen and vendored into tt-metal (tt_metal/tt-llk/README.md: "status: frozen, moved to tt-metal"; files are tracked blobs, no .gitmodules entry). Roughly 60% of the diff (+26.4k campaign docs/paper/ledger, +25.3k tt-llk exploration probes, +1.1k risc_scan_bench, ~+5.5k underscore-prefixed bench harnesses) must NEVER ship; the shippable core is ~12k LOC. Recommended mechanics: per-PR fresh branches off origin/main populated by `git checkout 9d7fd5f5ac6 -- <pathspec>` with fresh squashed commits (cherry-picking is unworkable — code and paper/ledger hunks are interleaved inside single commits, e.g. 482de67, 26abf46, fdb81ed). Merge order: PR-1 (stock bugfix) and PR-2 (LLK replay) anytime; PR-3 (LLK topk_xl) → PR-4 (op core) → PR-5 (routing) → PR-6 (sampling); PR-7 (deepseek regather skip) held pending 8x4 validation. Three files need manual hunk splits (test_topk.py, models/common/sampling/_utils.py, tt-llk test helpers), and the working tree holds four stray untracked files plus an uncommitted SHA-unpin of a GitHub Action that must be discarded.

## Analysis

ALL PATHS AND COMMITS BELOW ARE AT PINNED HEAD 9d7fd5f5ac6 (branch nkapre/sorting), merge-base 50a82f835593 with origin/main. Totals: 89 commits, 233 files, +65,170/−137.

== FRONT A-1: CATEGORY INVENTORY (every changed path classified) ==

(1) OP CORE — topk_large_indices (21 files, +4,151/−75). The op EXISTS at merge-base (verified: `git cat-file -e 50a82f835593:...topk_large_indices_device_operation.cpp` succeeds; merge-base kernels dir held only compute.cpp/reader.cpp/writer.cpp) — so PR-4 is "extend existing experimental op", not "add new op".
Modified: device/kernels/compute.cpp, device/topk_large_indices_device_operation.{cpp,hpp}, device/topk_large_indices_device_operation_types.hpp, device/topk_large_indices_program_factory.{cpp,hpp}, topk_large_indices_nanobind.cpp.
Added kernels: compute_tree.cpp, compute_tree_root.cpp, compute_tree_root_with_values.cpp, compute_with_values.cpp, reader_local.cpp, reader_tile.cpp, writer_flex.cpp, writer_tree.cpp, writer_tree_flex.cpp, writer_tree_with_values.cpp, writer_with_values.cpp, topk_large_indices_chunk_skip.hpp, topk_large_indices_compute_common.hpp, topk_large_indices_writer_flex_common.hpp.
Content from commits: ffb2b3c33cd (column-parallel multi-core), 8794fbb6b09 (log-tree final stage), 42f4823ac4b (valid_length<k contract), 8c9d8c0a9e8 (num_slices override), 68ff59fd732 (opt-in values output), 482de67d779 (TILE-native I/O + uint16 indices), 7d0b76bedf4 (chunk skip), 79709d176f7 (P-cap 128 + rectangle fit), aa721796e3f (skip telemetry + gate A/B), 26abf46feee (multi-rectangle trees + hybrid row split), 9d7fd5f5ac6 (fused-u16 end-to-end merge/rebuild), 9d22032a695 (small op tweak alongside ledger refresh — verify hunk).

(2) TTNN.TOPK ROUTING (1 file, +307): ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp. Content from 809cf5bda41 (large-k 64<k<=2048 route), 15f5659493c (small-k rows the bitonic can't take), fdb81ed027e (MoE-gate carve-out k<=16, padded W in [128,512]), 1d8e4a21244 (multi-row rectangle trees), plus route-side hunks of 68ff59f/482de67. Hard dependency verified: topk.cpp:20 includes topk_large_indices.hpp; ~line 197 calls topk_large_indices(return_values, tile_output, index_dtype) — those three parameters were ADDED by 68ff59f/482de67, so routing cannot merge before op core.

(3) LLK — vendored tt-llk headers + hw wrappers + compute API (4 files, +883/−5):
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h (+283) — content from bb60b823e76 (salvage) + d64993c3273 (default-on replay for topk phase>=4 step loads/stores on BH). NOTE: this header serves the STOCK ttnn.topk LLK too — it is an independent perf change.
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h (+427 from 725748766a6 SFPLOADMACRO-scheduled compare-exchange, internal helpers only: configure_sequences/program_templates/ce_first15/ce_tail/ce_full/record_ce_full — behind existing entry points; +132 from 9d7fd5f fused-u16).
- tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h (+19, 9d7fd5f) and tt_metal/hw/inc/api/compute/experimental/topk_xl.h (+27, 9d7fd5f) — the fused-u16 API surface the op-core kernels compile against (all 5 compute kernels + compute_common.hpp include api/compute/experimental/topk_xl.h).
Repo-structure fact: tt-llk is fully vendored (regular tracked blobs under tt_metal/tt-llk/, nothing in .gitmodules) and the separate tenstorrent/tt-llk repo is FROZEN per its README badge ("moved to tt-metal"). All LLK changes therefore go in tt-metal PRs; no tt-llk PR.

(4) STOCK-TOPK BUGFIX (1 file, +8): ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp, commit 079021d6c12 ("fix silent value corruption for >32 flattened rows"). Its regression test was NOT committed with it — the multi-core local-writer correctness guard lives inside the test_topk.py block added by 809cf5b (test_topk.py:411 "Correctness guard for the multi-core topk local-writer path"). Carve that test out into the bugfix PR.

(5) MODEL-SIDE (5 files, +263/−53):
- 575ff18a1be (I5 sampling relaxation, "take the topk_large_indices route where it fires"): models/common/modules/sampling/sampling_1d.py (74), models/common/sampling/tt_sampling.py (133), models/common/sampling/_utils.py (part).
- fdb81ed027e also touched models/common/sampling/_utils.py (MoE-gate util) → _utils.py (+72) is MIXED across two logical PRs.
- fe1930d50c2 (DSA indexer regather skip, PENDING 8x4 validation): models/demos/deepseek_v3_d_p/tt/mla/indexer.py (33), mla.py (4). indexer.py calls topk_large_indices(valid_length=...) (line ~314) → runtime dependency on op-core valid_length contract (42f4823).

(6) TESTS — shippable (subset of the 31-file/+6,700 tests diff):
- tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py (M, +834 total) → PR-4.
- tests/ttnn/unit_tests/operations/reduce/test_topk.py (M, +139, all from 809cf5b) → MIXED: stock-fix guard (→PR-1) + routed large-k tests incl. bfp8_inf/fallback-predicate tests (→PR-5).
- tests/sweep_framework/sweeps/reduction/topk/topk.py (M, +30, 22563f240c2 large_k suite) → PR-5.
- tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py (A, +1,187, Gate-1 differential contract suite; verified self-contained — zero references to baselines/) → PR-5 or standalone test PR; note fdb81ed added hunks here too.
- tests/ttnn/unit_tests/operations/data_movement/test_gather.py (M, +35, from 809cf5b) → SPECIAL: it is a REPRO for an unfixed upstream gather bug (RmSingleRowMultiCore reader misalignment at index_width>1920: per-core NoC read at byte offset w_start*4 into an aligned CB base → shifted data, "real-but-wrong elements"). The branch does NOT fix gather (68ff59f/482de67 dropped the gather chain from the route instead). As written the test asserts correctness and would presumably FAIL upstream — ship as GitHub issue + xfail test, or not at all (decision needed).

(7) HARNESS/BENCH under tests/ — campaign-only, never ship: tests/ttnn/unit_tests/operations/reduction/{_canonical_topk_sweep.py, _topk_sort_bench.py, _topk_ledger_render.py, TOPK_CONTRACT_RUNBOOK.md, baselines/** (comp3, comp4, scope51, smallk_routefix — 11 CSV/json/md files force-added past the global `*.csv` gitignore rule, .gitignore:8)}; tests/ttnn/nightly/unit_tests/operations/experimental/{_topk_large_indices_bench.py, _topk_large_indices_gate_ab.sh, _topk_large_indices_skip_adversarial.py, _skip_debug.py, _skip_diag.py, _skip_hangbattery.py, _skip_telemetry_parse.py, _topk_routed_bench.py}. Underscore prefix = not pytest-collected; pure campaign instrumentation.

(8) CAMPAIGN DOCS/EVIDENCE — never ship (111 files, +26,454): paper-topk/** (incl. committed LaTeX BUILD ARTIFACTS main.aux/.bbl/.blg/.fdb_latexmk/.fls/.log/.out and main.pdf; evidence/** incl. force-added gitignored CSVs per fba8f4847d8's own message), HANDOFF.md, SORTING.md, RADIX_BUCKET_GPU.md, TOPK_LEDGER.html, ci.sh (repo-root campaign CI battery, a4cf62d/64dbfde).
Also campaign-leaning: tt_metal/programming_examples/risc_scan_bench/** + 1-line CMakeLists.txt hook (+1,119, Gate-2 measurement harness 27c6da71c) — measurement evidence, recommend NOT shipping (or a separate opt-in example PR later). tt_metal/tt-llk/tests/** (54 files, +25,285): overwhelmingly exploration probes whose own commit messages mark them dead ends (73ca67d RETRACT, 85753d7 refuted, bb60b82 dead on paper, 445760b refuted) — perf_pack_exp_histogram, perf_pack_zero_compress, perf_sfpu_count_above, perf_topk_* (9 files), perf_unpack_*, test_pack_*, test_sfpu_count_above, test_cgtceq_perf, cgtceq_analysis.py, THRESHOLD_SELECT_DESIGN.md, CGTCEQ_RUNBOOK.md, + 22 sources/*.cpp|.h. Ship ONLY the minimal regression tests that gate the shipped LLK changes: test_topk_xl_unfused_macro.py + topk_unfused_macro sources (gates 7257487), test_topk_unfused_macro_probe.py + probe source (compile-in gate), and — if they cover the replay-STORE path — test_topk_rebuild_macro.py/test_topk_rebuild_full_macro.py + sources; plus only the needed hunks of the 4 helper files (golden_generators.py +80, llk_params.py +136, perf/test_schemas.py +44, test_variant_parameters.py +158 — MIXED, trim to what shipped tests import).

== FRONT A-2: PROPOSED PR PARTITION ==

PR-1 — "ttnn.topk multi-core: fix silent value corruption for >32 flattened rows". Files: ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp (+8) + the local-writer guard test carved from test_topk.py (~40 LOC). Carries: 079021d6c12 (+ test hunk of 809cf5b). ~50 LOC. No dependencies. Merge first — user-facing correctness fix.

PR-2 — "BH LLK topk: default-on replay for phase>=4 step loads/stores". Files: tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h (+283). Carries: bb60b823e76 + d64993c3273. Independent of everything else, but changes stock-topk codegen on BH → needs its own perf/PCC evidence in the PR body and must pass existing topk LLK/ttnn tests. ~300 LOC.

PR-3 — "BH LLK topk_xl: SFPLOADMACRO-scheduled compare-exchange + fused-u16 merge/rebuild". Files: ckernel_sfpu_topk_xl.h (+559), llk_math_eltwise_unary_sfpu_topk_xl.h (+19), tt_metal/hw/inc/api/compute/experimental/topk_xl.h (+27), + minimal tt-llk regression tests (test_topk_xl_unfused_macro.py, test_topk_unfused_macro_probe.py, matching sources/*.cpp, trimmed helper hunks). Carries: 725748766a6, c8c5ca206c7 (probe), LLK half of 9d7fd5f5ac6. ~1,000–1,500 LOC. Must merge BEFORE PR-4 (op kernels compile against the fused-u16 topk_xl.h API).

PR-4 — "topk_large_indices: multi-core row/column-parallel trees, TILE I/O, u16 indices, values output, chunk skip, fused-u16". Files: all 21 op-core files + tests/ttnn/nightly/.../test_topk_large_indices.py. Carries: ffb2b3c, 8794fbb, 42f4823, 8c9d8c0, 68ff59f, 482de67, 7d0b76b, 79709d1, aa72179, 26abf46, op half of 9d7fd5f (+ op hunk of 9d22032). ~5,000 LOC. Depends on PR-3. Largest review burden — consider a 4a/4b split (4a: multi-core tree paths + values output + TILE/u16; 4b: chunk skip + rectangle fit + hybrid + fused-u16) if reviewers balk; both halves stay on the same file set so the split is by hunk/feature flags, doable but more work.

PR-5 — "ttnn.topk: route small-k/large-k/MoE-gate shapes onto topk_large_indices (BH)". Files: ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp (+307), test_topk.py routed-tests remainder (~100), tests/sweep_framework/sweeps/reduction/topk/topk.py (+30), test_topk_contract.py (+1,187, optional but strong CI story), MoE-gate hunk of models/common/sampling/_utils.py if the carve-out util lives there. Carries: 809cf5b, 15f5659, fdb81ed, 1d8e4a2, route hunks of 68ff59f/482de67, 22563f2. ~450 code + ~1,300 test LOC. Depends on PR-4. Document the tie-semantics contract change (routed path = correct top-k SET, deterministic-but-unspecified tie order — stated in test_topk.py header comment) prominently; that is the most likely reviewer objection.

PR-6 — "Sampling call sites: take the routed topk form". Files: sampling_1d.py, tt_sampling.py, remaining _utils.py hunks. Carries: 575ff18a1be. ~280 LOC. Depends on PR-5 (route must exist to fire). I5 bit-exactness study exists as evidence (paper-topk/evidence/i5-sampling-relaxation/) — summarize in PR body, don't ship the files.

PR-7 — "GLM/DS-V4 DSA indexer: skip top-k TP regather when consumer re-splits". Files: indexer.py, mla.py. Carries: fe1930d50c2. ~37 LOC. HOLD — explicitly PENDING 8x4 validation. Runtime dependency on PR-4 (valid_length contract) given indexer.py's direct topk_large_indices(valid_length=...) call.

Optional PR-0/issue — gather RM multi-core misalignment: file a GitHub issue with the mechanism analysis from the test_gather.py comment; ship the repro as xfail or attach to the issue. Decision needed from user.

== MECHANICS ==
Cherry-pick chains are NOT recommended: at least 6 code commits also touch paper-topk/ or TOPK_LEDGER.html in the same commit (482de67, 26abf46, fdb81ed→_canonical_topk_sweep.py, 9d22032, a667c40, 1d8e4a2), and 9d7fd5f mixes LLK+op+evidence. Instead, for each PR: `git checkout -b <pr-branch> origin/main && git checkout 9d7fd5f5ac6 -- <exact pathspec list> && git add -A && git commit` (fresh squashed commit(s), message citing the measured numbers and the local provenance SHAs). Manual hunk work needed only for: test_topk.py (PR-1 vs PR-5 split), models/common/sampling/_utils.py (PR-5 vs PR-6 split), tt-llk test helpers (trim to shipped-test imports), and — if PR-4 is split 4a/4b — the op-core kernels. Build+test each PR branch in isolation (PR-5's branch must include PR-4's files to compile; stack the branches: pr3 → pr4 on pr3 → pr5 on pr4 → pr6 on pr5).

== WORKING-TREE / ACCIDENTAL-ARTIFACT FLAGS (none may reach any PR) ==
- Uncommitted `M .github/workflows/package-and-release.yaml`: replaces SHA-pinned `pypa/gh-action-pypi-publish@ed0c539...# v1.13.0` with floating `@v1.13.0` — weakens supply-chain pinning; discard (git checkout --).
- Untracked: lx-reset (1,081 B script), ttnn/ttnn/_ttnn.so.release (13.9 MB binary), tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml, tt_metal/programming_examples/eltwise_poly/ (unrelated experiment).
- Uncommitted paper-topk LaTeX rebuild churn (main.aux/.pdf/sections) — campaign-only anyway.
- Committed .gitignore-forced adds: paper-topk/evidence CSVs (fba8f48 says "force-add the gitignored") and tests/.../reduction/baselines/**.csv vs .gitignore:8 `*.csv` — all campaign-only, so no upstream conflict, but any future attempt to ship baselines would fight the ignore rule.
- Committed LaTeX build artifacts (main.aux/.log/.fls/.fdb_latexmk/.pdf) are in history — harmless for the checkout-by-path mechanics, but confirms cherry-pick is the wrong tool.

## Evidence

- git log --oneline 50a82f835593..HEAD → 89 commits at pinned HEAD 9d7fd5f5ac6; git diff --stat → 233 files, +65,170/−137
- Category LOC via git diff --stat 50a82f835593..9d7fd5f5ac6 -- <path>: op core 21 files +4,151/−75; reduction/topk/topk.cpp +307; topk_final.cpp +8; tt-llk headers + hw wrappers 4 files +883/−5; tt-llk tests 54 files +25,285; models 5 files +263/−53; tests/ttnn+sweep 31 files +6,700; campaign docs (paper-topk, HANDOFF.md, SORTING.md, RADIX_BUCKET_GPU.md, TOPK_LEDGER.html, ci.sh) 111 files +26,454; programming_examples 5 files +1,119
- tt-llk is vendored, not a submodule: git ls-files -s tt_metal/tt-llk shows mode-100644 blobs; .gitmodules has no llk entry; tt_metal/tt-llk/README.md badges 'status: frozen' and 'moved to tt-metal' — LLK PRs go to tenstorrent/tt-metal
- Op pre-exists upstream: git cat-file -e 50a82f835593:ttnn/.../topk_large_indices_device_operation.cpp succeeds; merge-base kernels dir = {compute,reader,writer}.cpp only
- Routing→op-core hard dependency: topk.cpp:20 includes topk_large_indices.hpp; comment block at topk.cpp:186-217 documents the route calling topk_large_indices(return_values, tile_output, index_dtype) — parameters added by 68ff59f/482de67
- Op-core→LLK dependency: all 5 compute kernels + topk_large_indices_compute_common.hpp include api/compute/experimental/topk_xl.h; fused-u16 commit 9d7fd5f modifies that API (+27) together with op kernels (grep of kernel includes + git show --stat 9d7fd5f5ac6)
- Stock fix isolation: git show --stat 079021d6c12 = exactly topk_final.cpp +8; its regression guard (test_topk.py:411 'Correctness guard for the multi-core topk local-writer path') was committed by 809cf5b (git log -- test_topk.py shows 809cf5b as sole toucher)
- ckernel_sfpu_topk.h (+283) attribution: git show --name-only bb60b823e76 and d64993c3273 both touch it; ckernel_sfpu_topk_xl.h committed +427 solely from 725748766a6 (internal helpers configure_sequences/ce_first15/ce_tail/ce_full per diff grep)
- Mixed-commit proof (cherry-pick hazard): git show --name-only fdb81ed027e = models/common/sampling/_utils.py + _canonical_topk_sweep.py + test_topk_contract.py + topk.cpp; 482de67/26abf46/9d7fd5f each mix paper-topk evidence with product code
- Model commit split: fe1930d50c2 = indexer.py + mla.py + paper evidence md (indexer.py:314 calls topk_large_indices(valid_length=...)); 575ff18a1be = sampling_1d.py + tt_sampling.py + _utils.py
- Unfixed gather bug repro: test_gather.py +35 (from 809cf5b) comment states RmSingleRowMultiCore reader misalignment at w_start*4 byte offsets for index_width>1920 — a repro, not a fix, on this branch
- Contract suite self-contained: grep 'baselines/' test_topk_contract.py → no matches; .gitignore:8 '*.csv' confirms baselines/evidence CSVs were force-added (fba8f48 commit message says so explicitly)
- Working-tree strays: git status shows uncommitted package-and-release.yaml diff un-pinning pypa/gh-action-pypi-publish from SHA ed0c539... to @v1.13.0; untracked lx-reset (1,081B), ttnn/ttnn/_ttnn.so.release (13,878,424B), n150_mesh_graph_descriptor.yaml, eltwise_poly/

## Risks

- HEAD moved during analysis (main agent committed 9d7fd5f5ac6 mid-run) and may move again — re-pin and re-run the diff --stat totals immediately before executing the partition
- PR-4 at ~5,000 LOC across 21 files is a heavy single review; the suggested 4a/4b hunk-level split adds real manual effort and regression risk — decide before branch surgery
- Routed ttnn.topk changes user-visible tie semantics (index SET with unspecified tie order vs stock's ordering) — upstream reviewers may require an opt-in flag or doc callout; budget for that pushback on PR-5
- PR-2 (replay default-on) alters stock-topk LLK codegen for ALL BH ttnn.topk users, not just the campaign path — it needs standalone perf+PCC evidence on shapes the campaign didn't sweep
- test_gather.py repro would likely FAIL in upstream CI as written (asserts correctness of a still-broken gather path) — must become an xfail or an issue attachment, needs user decision
- Three files require manual hunk splits (test_topk.py, models/common/sampling/_utils.py, tt-llk test helpers); a wrong split silently ships campaign code or breaks a shipped test import
- PR-7 (deepseek regather skip) is explicitly PENDING 8x4 validation — shipping before that validation risks a model-correctness regression on Galaxy meshes
- Which tt-llk tests actually gate the replay-STORE change (PR-2) was inferred, not verified — confirm test_topk_rebuild_*_macro coverage maps to ckernel_sfpu_topk.h paths before trimming the 54-file test set
- The 9d22032a695 op-core hunk (committed alongside a ledger refresh) was not individually inspected — inspect it during PR-4 assembly so no campaign-only tweak ships
- No device runs were performed per hard rules — every per-PR branch must be built and tested on silicon before any push
