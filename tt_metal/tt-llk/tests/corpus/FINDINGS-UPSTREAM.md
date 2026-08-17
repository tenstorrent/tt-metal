# FINDINGS-UPSTREAM — itemized upstream ledger

Findings from the SFPI-uplift sweep/corpus work that belong to (or should be
seen by) upstream owners — tt-llk, tt-metal, ttexalens, the sfpi release
pipeline, and any generic Tensix simulator. Each item carries its evidence
pointer. None of these blocks our sweeps (workarounds/fixes are recorded per
item); the ledger exists so the knowledge does not stay trapped in lane
evidence directories.

Status legend: **BUG** = defect to fix upstream; **DIVERGENCE** = two sources
of truth disagree, owner adjudication needed; **CAVEAT** = correct behavior
with a non-obvious consequence others will trip on; **FIXED-HERE** = fixed on
our branch, upstream should take or reimplement the fix.

---

## 1. llk_math_reduce.h:49,51 — INT8 store mod vs simulator legality (DIVERGENCE)

`tt_llk_blackhole/llk_lib/llk_math_reduce.h` lines 49 and 51 (the
`is_int_fpu_en` transpose path) issue `TTI_SFPSTORE(..., InstrModLoadStore::INT8, ...)`
(mod0 = 5). The craq-sim generic-path legality check for SFPSTORE mods
(sfpu store legality, sim source ~line 8695) **rejects mods 1/5/8/10/11/13**,
INT8=5 included — the sim would trap on an instruction production silicon
executes routinely. Either the sim's legality list is too strict (HW accepts
INT8 stores and the sim should model them) or the kernel exercises an
unspecified encoding that happens to work — an owner adjudication either way.

- Evidence: R5 mechanism scout, `~/sfpi-uplift/mechanism-scout-20260817/`
  (`ISA-UNUSED.md` "SFPSTORE mods … rejected by sim legality check (8695)";
  `MECHANISMS.md` "sim/production divergence" note).
- Suggested owner: tt-llk (kernel intent) + craq-sim (legality model).

## 2. tt-llk#1344 — TopK [32,1024] skipped upstream, both impls (BUG, upstream-filed)

`test_topk` at input dimensions [32,1024] is skipped upstream for both the
handwritten and generated impls. This caps our TopK coverage promotion at
[., 128] shapes; the corpus skip ledger classifies it [C] UPSTREAM-BUG (broken
for everyone, not ours). A fix un-skips the node for both paths.

- Evidence: `~/sfpi-uplift/topk-ab-evidence-20260816/EVIDENCE.md` (item 4,
  coverage gaps before promotion);
  `~/sfpi-uplift/coverage-parity/SKIP_LEDGER.md` (class [C], follow-up 8).

## 3. tt-llk#1120 — ReluMin skipped, all instances (BUG, upstream-filed)

Every ReluMin node is skipped upstream, excluding the op from both the
hand and generated coverage paths (skip ledger class [C]; the unary max/min
corpus rows deliberately exclude ReluMin per this issue). A fix un-skips it
for both paths.

- Evidence: `~/sfpi-uplift/coverage-parity/SKIP_LEDGER.md` (class [C] row and
  the unary-corpus exclusion note).

## 4. tt-metal#33492 — stable_sort skipped upstream (BUG, upstream-filed)

The stable_sort TopK variant is skipped upstream (tt-metal side), the second
of the TopK coverage caps recorded before any promotion beyond current shapes
(alongside #1344). Skip ledger class [C].

- Evidence: `~/sfpi-uplift/topk-ab-evidence-20260816/EVIDENCE.md` (item 4);
  `~/sfpi-uplift/coverage-parity/SKIP_LEDGER.md`.

## 5. LLK-profiler zone-id constants hash the source path (CAVEAT, methodology)

Perf/device-profile kernels embed LLK-profiler zone-id constants computed by
hashing the kernel source path. Consequence: the SAME compiler building the
SAME source at two different filesystem paths produces `.text` that differs in
exactly those marker `lui` immediates — byte-identity/parity gates across
machines or worktrees will read false CHANGED. Proven on reduce-sdpa: union
cc1plus at the shim-base path == base cc1plus at the shim-base path, both !=
the shared-path bytes; a probe-off diff vs the Lane W archive was 4 marker
`lui` words only. All parity claims must therefore be made as in-context
pairings at ONE path (compilers varied, tree fixed). Upstream improvement:
derive the zone-id from a repo-relative path (or content hash) so builds are
location-reproducible.

- Evidence: Lane AA methodology finding,
  `~/sfpi-uplift/laneAA-evidence-20260816/README.md` ("Path-provenance
  finding"); extended by R2 (`~/sfpi-uplift/laneR2-evidence-20260817/
  EVIDENCE.md` method note 1).

## 6. 8-byte unpack.elf stub: release vs local sfpi builds (CAVEAT, root-caused)

EVERY locally built cc1plus — including a pristine rebuild of the exact
release baseline commit (bit-reproducible, same sha on rebuild) — emits the
exp-node `unpack.elf` with the 8-byte
`ckernel::coverage::assert_tensor_shape_coverage_unobserved_` stub REMOVED
relative to the pinned sfpi RELEASE compiler (whole `.text` shifts; `math.elf`
and `pack.elf` stay byte-identical). It is common-mode across baseline and
increment, cancels in any lane-vs-lane comparison, and explains why
release-vs-local sweeps show unpack-only drift on every row. Upstream (sfpi
release pipeline): the release build configuration differs from a source
build in a way that changes emitted code — worth pinning down and eliminating
so release binaries are reproducible from source.

- Evidence: R2 finding, `~/sfpi-uplift/laneR2-evidence-20260817/EVIDENCE.md`
  (method note 2, "NEW methodology finding (extends Lane AA finding 2)").

## 7. ttexalens CallstackEntry API drift breaks the assert printer (FIXED-HERE)

ttexalens moved file/line/column off `CallstackEntry` onto
`entry.file_info` (a `DwarfFileLine`); with a fresh venv (tt-exalens 0.3.29)
the tt-llk harness stack-trace printer crashed with
`'CallstackEntry' has no attribute 'file'` — masking the device-side
LLK_ASSERT it was trying to report (which itself leaves the Tensix core hung
until `tt-smi -r`). Fixed on our branch: tolerant printer via
`getattr(entry, "file_info", None) or entry`
(`tests/python_tests/helpers/device.py`, commit 6bd246d2dc
"test(llk): tolerate ttexalens CallstackEntry API drift in assert printer").
Upstream tt-llk should take the tolerant printer (or pin/track the ttexalens
API) so assert reporting survives the drift in both API generations.

- Evidence: commit 6bd246d2dc on this branch; Lane E/L notes (assert masking
  chain: ebreak hang -> TIMEOUT cascade -> printer crash).

## 8. Simulator kept-separator lane-mask hazard (FIXED at craq-sim 9f324140; upstream awareness)

Root cause of the Where sim-vs-silicon divergence: the simulator latched the
SFPLOADMACRO *scheduled store's* lane mask at macro LAUNCH; hardware evaluates
it LIVE at store EXECUTION (per SFPLOADMACRO.md only Addr/Mod0/backdoor are
latched). A 4-slot kept-separator select calendar (misc=0x706) whose ENCC
restore retires the SAME cycle as the store therefore stores under the SETCC
complement on silicon (true lanes left unwritten) while the old sim modeled a
pass — byte-identical binaries were deterministic-RED on silicon and GREEN on
CRAQ. Fixed on craq-sim branch `agent/kept-separator-delivery-model`
(9f324140, base f80a8d64): the store event reads cc/cc_en from the retirement
group's pre-write snapshot; the corrected sim reproduces the exact silicon
signature (mixed FAIL / all_ones FAIL / all_zeros PASS). Compact 0x770
calendars retire the restore a cycle earlier and are correct in both worlds.
Upstream awareness: ANY generic Tensix simulator modeling SFPLOADMACRO
scheduled stores must use live-at-execution lane masks, and compiler-side
macro formation must enforce restore_exec < store_exec (done in sfpi-gcc
`cc-restore-store-race` constraint, b992ec89d34).

- Evidence: `~/sfpi-uplift/where-adjudication-20260817/verdicts/VERDICT.md`;
  craq-sim 9f324140; corrected-sim negative control in
  `~/sfpi-uplift/laneAI-evidence-20260817/`.

---

Maintenance: append new items with the same status legend and an evidence
pointer; strike items when the upstream fix lands (note the landing
commit/issue state rather than deleting the row).
