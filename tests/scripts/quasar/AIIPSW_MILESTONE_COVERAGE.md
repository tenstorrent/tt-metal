# AIIPSW client milestones → Quasar test coverage

Which AI/IP SW milestone requirement each Quasar test serves, and whether a failure
of that test reaches the release Jira automation.

- **Requirements:** Jira project `AIIPSW` (AI IP SW), monthly release epics
  `AIIPSW-1` (Jul-15), `AIIPSW-5` (Aug-15), `AIIPSW-10` (Sept-15). Success
  criterion for every subticket is *"test passing in the Quasar simulation /
  emulation environment."*
- **Tests:** the three yamls in this directory. For the full row-by-row
  inventory see [QUASAR_TEST_COVERAGE.md](QUASAR_TEST_COVERAGE.md) (generated).
- **Automation:** `.github/actions/scripts/ai_ip_tests.json` decides which failed
  test gets a RELEASE Jira ticket, tagged with its requirement key.

Epic membership and assignees read from Jira on **2026-08-11**; the epics gain
children over time, so re-query before trusting this table.

## Jul-15 — `AIIPSW-1`

| Requirement | Quasar tests | Wired into `ai_ip_tests.json` | Team | Owner |
|---|---|---|---|---|
| **AIIPSW-2** — Runtime: Add FD support *(Done)* | `unit_tests_legacy`: `*DmLoopback*`, `*QuasarComputeKernelMultipleThread*`, `*MultiDmAddTwoInts*` | ✅ yes — and the first two are among the only tests whose failure reaches Jira today | Trinity | Kevin Stevens |
| **AIIPSW-3** — LLK: Compute API calling Yolo related LLK API *(Done)* | 18 Quasar LLK tests in the **tt-llk** repo (`tests/python_tests/quasar/`): `test_eltwise_binary_quasar.py`, `test_eltwise_binary_broadcast_quasar.py`, `test_eltwise_binary_reuse_dest_quasar.py`, `test_eltwise_unary_datacopy_quasar.py`, `test_unary_broadcast_quasar.py`, `test_matmul_quasar.py`, `test_reduce_quasar.py`, `test_pack_quasar.py`, `test_pack_untilize_quasar.py`, `test_unpack_tilize_quasar.py`, `test_unpack_unary_operand_quasar.py`, `test_semaphore_sync_quasar.py`, `test_sfpu_*_quasar.py` (6) | ❌ no — separate repo and CI; not in any tt-metal Quasar yaml | Trinity | Filip Vranic |
| **AIIPSW-9** — ResNet LLK API passing in Horizon emulation *(Done)* | Horizon environment, not Quasar | ➖ out of scope for this map | Horizon | Filip Vranic |

## Aug-15 — `AIIPSW-5`

| Requirement | Quasar tests | Wired into `ai_ip_tests.json` | Team | Owner |
|---|---|---|---|---|
| **AIIPSW-4** — TTNN/Kernel Ops: Quasar ResNet related Kernel Ops *(Done)* | `models/demos/vision/classification/resnet50/quasar/tests/ops/` — **51 test files in the repo, 11 wired** into `quasar_local_tests.yaml` @2x3 (whole-file pytest): `test_add`, `test_padded_slice`, `test_reallocate`, `test_reshape`, `test_reshape_tiled`, `test_sharded_to_interleaved`, `test_slice_write`, `test_tilize`, `test_to_layout`, `test_to_memory_config`, `test_untilize_with_unpadding`. A further 10 model-level tests (e2e, functional, performant, perf, stability) sit in the parent dir, unwired. | ✅ all 11 — but on the emulator path, so **no ticket reaches Jira yet** | Trinity | Borys Bradel |
| **AIIPSW-6** — Runtime: Add FD support for dispatch engine *(Done)* | `unit_tests_api`: `*TensixSingleCoreDirectDramReaderDatacopyWriter`, `*QuasarCRTASharedL1Address*`, `*QuasarCRTAUniqueL1Addresses*`; the new `unit_tests_dispatch` binary (`*QuasarDispatchSInstantiatedAndRunning*` @2x3 and @2x3_DISPATCH); and the whole 14-row `2x3_DISPATCH` suite in `quasar_regression_tests.yaml` | ✅ yes, incl. a config-only wildcard for `2x3_DISPATCH`. Only `*TensixSingleCoreDirectDramReaderDatacopyWriter` @1x3 reaches Jira today | Trinity | Kevin Stevens |
| **AIIPSW-8** — LLK: Quasar int8 support *(Done)* | No named Quasar sim test — LLK-side coverage lives in the tt-llk repo | ❌ no — owning team to name the test(s) | Trinity | Filip Vranic |
| **AIIPSW-12** — LLK: PDL related LLK features *(Done)* | No named Quasar sim test (PDL = the Panoptic-DeepLab vision model) | ❌ no — owning team to name the test(s) | Trinity | Filip Vranic |
| **AIIPSW-13** — Runtime: Profiler debug tool support *(Done)* | `tests/tt_metal/tools/profiler/test_device_profiler.py`: `test_custom_cycle_count_slow_dispatch` @1x3, `test_custom_cycle_count` @2x3_DISPATCH, `test_full_buffer` @2x3_DISPATCH — in `quasar_local_tests.yaml` | ✅ yes (whole-file entry) — emulator path, **no ticket reaches Jira yet** | Trinity | Kevin Stevens |
| **AIIPSW-14** — LLK: Quant/dequant kernels from LLK *(Done)* | No named Quasar sim test — tt-llk repo | ❌ no — owning team to name the test(s) | Trinity | Filip Vranic |
| **AIIPSW-15** — UMD: "Higher Level Layer" support *(Done)* | UMD-side; no tt-metal Quasar yaml row | ❌ no | UMD | _unassigned_ |

## Sept-15 — `AIIPSW-10`

| Requirement | Quasar tests | Wired into `ai_ip_tests.json` | Team | Owner |
|---|---|---|---|---|
| **AIIPSW-7** — TTNN/Kernel Ops: Quasar Llama related Kernel Ops *(In Progress)* | `models/experimental/llama32_1b_quasar/tests/` — 45 isolated op tests under `ops/`, plus module tests under `modules/` (attention, mlp, rmsnorm, rope, lm_head, embedding, sampling) | ❌ none wired into a Quasar yaml yet | Trinity | Borys Bradel |
| **AIIPSW-16** — TTNN/Kernel Ops: Quasar ResNet with conv2D, pool *(In Progress)* | The not-yet-supported half of the ResNet ops dir: `test_conv2d*` (14 files), `test_max_pool2d*` (7), `test_avg_pool2d`, `test_linear` | ❌ none wired yet — these are the ops that do not pass on Quasar today | Trinity | Borys Bradel |

## Coverage growth since this PR was opened

Three test-suite PRs by @kstevensTT landed after the branch point:
[#50647](https://github.com/tenstorrent/tt-metal/pull/50647) (Jul-22),
[#51403](https://github.com/tenstorrent/tt-metal/pull/51403) (Jul-31),
[#52760](https://github.com/tenstorrent/tt-metal/pull/52760) (Aug-10).

| | Before (Jul-17) | Now |
|---|---|---|
| `quasar_regression_tests.yaml` rows | 26 | **56** |
| distinct (group, filter) | 21 | **50** |
| config split | 1x3 22 / 2x3 2 / 2x3_DISPATCH 2 | 1x3 37 / 2x3 5 / **2x3_DISPATCH 14** |
| test groups | 4 | **5** — new `unit_tests_dispatch` |
| runners | gtest | gtest + **pytest** |
| `quasar_local_tests.yaml` | did not exist | **14 pytest rows** (3 profiler, 11 ResNet ops) |
| `quasar_sim_regresion_tests.yaml` | 5 | **5 — unchanged since 2026-06-01** |

## Release test-evidence report

Every Package-and-release run on `stable` produces a shareable record of which
requirements have passing test evidence, via
`.github/actions/scripts/release_test_report.py`. It lands in three places:

- the **workflow run summary**,
- a **`release-test-evidence-<version>` artifact** (markdown), and
- a **Jira issue** in `RELEASE`, one per release version (re-runs update it
  rather than piling up), labelled with every requirement that has passing tests.

**How "executed successfully" is decided.** Two paths:

1. **Authoritative.** The sim CI embeds its full result set — passes included —
   in the check's `output.text` as a JSON block (`rtl-sim-results/v1`). When
   that block is present it is used verbatim and nothing is inferred. Landing
   via tt-umd-simulators!125.
2. **Derived**, for output produced before that block existed. The old manifest
   (`failed_tests.tsv`) listed failures only, so passes were computed as
   *(tests the gate is expected to run)* − *(tests reported failed)*. That is
   only sound when the run completed, so when the check is red with no per-test
   detail, or timed out, or the sim reporter says the manifest was missing, the
   report is marked **INCONCLUSIVE** and claims no passes at all. A truncated or
   malformed JSON block also falls back to this path rather than reporting zero
   passes.

The report deliberately lists requirements with **no** evidence alongside those
with it. On today's gate a green release covers **2 of 12** requirements
(AIIPSW-2 and AIIPSW-6) — the report is only proof of what actually ran, and
saying so is the point.

## Known gaps

1. **The growth is on a Slack path, not the Jira path.** The expanded lists run in
   the GitLab `metal_unit_test_emu_quasar` job, which reports to `#tt-qsr-emu-ci`
   and does not write the `failed_tests.tsv` manifest that feeds the "RTL Sim CI
   test" check. Only the 5-row `quasar_sim_regresion_tests.yaml` (config 1x3)
   reaches Jira. Closing this needs a change in `tensix/tt-umd-simulators`:
   either have the emu job emit the manifest, or teach the reporter to read the
   `gtest-summary/summary.json` it already produces. Until then the AIIPSW-4 and
   AIIPSW-13 entries in `ai_ip_tests.json` are staged, not live.
2. **That emu job is itself unmerged**, living on GitLab branches
   `kstevens/emu-quasar-1x3-testing` and `kstevens/pytest_ci`.
3. **20 DFB implicit-sync tests are unattributed.** `unit_tests_api` gained
   `M2ImplicitSync/DFBImplicitSyncParamFixture_2_0.*`, `ImplicitSync/…` and
   `MeshDeviceFixture.DFB*` in #51403. They look dispatch-adjacent but no
   requirement claims them, so they are deliberately left out of the map rather
   than guessed onto AIIPSW-6.
4. **`unit_tests_legacy/*Bmm` is live but unwatched.** It is one of only four
   tests whose failure can reach Jira today, yet no requirement maps to it.
5. **AIIPSW-8, -12, -14, -15 have no named Quasar test.** They were closed as Done
   on LLK/UMD-side evidence; if they are meant to be gated by the Quasar sim, the
   owning team needs to name the test.
