# Served-source reconciliation, November 10, 2026

Tasks #11/#25, ltx-rt-mac: source preservation and integration review packet.
No serving checkout, model default, production process or shared branch changed.
Immutable evidence is in project run 23.

## Preserved source

| Item | Verified identity |
| --- | --- |
| Existing ltx-rt base | `522ef26916bdd78e2446fd1ff19a4f33e40ecfb6` |
| Recovered LTX head | `0b43a32cda9128299bbd10ce72706bf6d435c254` |
| Recovered tree | `899a0222459f6a5d246df0885f31a3b2c93de081` |
| Local preservation ref | `ttp/t11-preserved-served-0b43a32` |
| Incremental bundle size | 49,532 bytes |
| Bundle SHA256 | `417e594282662d44e1a4898cf6c6208068ed2569d9a125001426885b06b493a5` |

The bundle requires the existing base above. Git verified the bundle, prerequisite
and copied bytes. Its head is 14 commits ahead, zero behind, changing 17 files.
Every original commit is retained. This verifies source, not a live process.

The owner's ledger at ltx-server `c62aad2` reports cs04 LTX on the base,
f101/f102/blx04 LTX on the recovered head, and their H3 slots on main
`15d5b0028220bb994f6d3652fd9948bea04e34c6`. These are distinct identities.
The later owner receipt `da15b541a3a414f4` reports clean detached H3 trees on
f101/f102 at that main commit and a matching build stamp. Source recovery is
unnecessary. At that checkpoint, live-process/adapter attestation was incomplete;
root-only `release.env` was not read. Run 23
`model-source-manifests-update.json` separates source, build and live evidence.
Coordinator receipt `730624765dc2abdb` records the ledger landed at `c62aad234f`.

## Semantic reconciliation

- The recovered tree supplies `vae_key_map`, split-file resolution and explicit
  `LTX25_VAE_FALLBACK_23=1` fallback. The server capability failure observed on
  the base cannot be assumed to reproduce here. Test both providers and record
  the checkpoint actually loaded.
- It preserves independent audio replay, opt-outs, retry/cancellation cleanup,
  encoder trace ordering, and default-off end-hold, end-guide and soft-pin modes.
- Selected source from historical `16e6ed60a663726d1a3b0c8abc235e19a1ca2165`
  was recovered read-only from blx04. Four valid split/monolith/mixed namespace
  mapper contracts agree. Four malformed cases differ: missing statistics, empty
  component name, unknown component and empty checkpoint. The historical mapper
  rejects them; the served mapper returns a map. This identifies a hardening
  candidate, not proof of a valid-model numerical difference.
- The served resolver checks its default conv path and may consult a fallback hub
  path; the historical resolver has a narrower fallback search. Bind actual
  checkpoint identities before porting either policy.
- FastH3 Turbo `d4494e2c1a17cceed9c42fc94284d15a613bb93b` and HyperFlow
  `a99af1c7deaf729b19aa8745f68099f150d9d4b7` share merge base
  `b9f8587ce6c681f380e341753c6f0ea7f5ee441d` with current ltx-rt. They have
  29 and 17 unique commits respectively, while lacking 394 commits reachable
  from ltx-rt. Replacing ltx-rt with either tip loses LTX2.5/Gemma4 work.
  Reconcile candidate-specific patches; do not replace the tree.
- Turbo's inspected `MINIMAX_H3_ADALN_PRECOMPUTE` default is `1`, despite its
  historical quality caveat. Keep it disabled in the high-quality route; adoption
  needs an explicit variant and matched quality tests.
- blx04 contains FastH3 `762cb5411515c172d282811a4dac58df38cafe00`, tree
  `fcc22aad2a23e3f591890e7c9b27bc2b7d5dc4ea`. The owner's backup receipt is
  requested; this task did not duplicate the backup or claim publication.

## Evidence and validation matrix

Existing audio control/lifetime tests: **20 passed** on Python 3.12 using fake
native tensors and AST-extracted methods. Python 3.9 failed on unsupported
`zip(strict=...)`; that run does not establish a product regression.
VAE comparisons record **16 provider/case observations** and the differences above.
Neither test establishes device correctness, quality or speed.

Run 23 contains `source-manifest.json`, `source-comparisons.json`,
`vae-contract-comparison.json`, `validation-matrix.json`, `artifact-manifest.json`,
and the preserved bundle. Hashes cover immutable snapshots and logs, not queues.
The preservation ref is an integration starting point, not a deploy-ready build.

## Main-merge handover, 08:50 PST

GitHub backup ref `ttp/ltx-rt-boxes-0b43a32` now resolves to the preserved
`0b43a32cda9128299bbd10ce72706bf6d435c254`; the backup no longer depends on
the serving host being reachable. The intended `ttp/ltx-rt-main-merge` ref
was absent from the same remote lookup, and its candidate is absent locally.

Owner note `a18a23a0e08e7a93` and sulphur #442's handoff identify candidate
`173bce4455988341ce4358c566d57ec1f9de8a80`, merging main `0de8c7f3aade`
with the preserved LTX history. Its incremental bundle requires that main
commit and has SHA256
`a8cb3f7e26eef37e831441831c78168348160743a947795502a4adb1420ae704`.
This task has requested a local or blx04-accessible copy and the full build log.
The bundle hash here is owner-reported, not locally verified.

The owner's build4 failed in `ttnn_op_transformer` unity compilation:
duplicate `check_gdn_gate` definitions and `q/k/v` constants in gated-delta-rule
sources. This task has not reproduced that failure or selected a fix. Preserve
the candidate's existing `plan.l1_prefetch_max_bytes` conv3d repair. Reproduce
the exact candidate and build settings before changing unity participation or
symbols; then complete the build and relevant tests in an isolated checkout.
Model tasks remain independent of this integration build. Run 37 records the
handover and check retry; required review, device cells and rollout remain open.

| Required cell | Verdict / next evidence |
| --- | --- |
| Bundle, ancestry, copy equality | Pass |
| Audio control lifetime | 20 CPU tests pass |
| VAE semantic comparison | Four valid contracts agree; four malformed cases differ |
| Main integration/build | Consume sulphur #442 candidate and conflict/build receipts |
| LTX2.3 t2v/audio | Combined head; eager/traced/audio on-off, retry/cancel and quality |
| LTX2.5 conv VAE + IC | t2v/ref/fl, actual artifacts, changing sheets, three seeds |
| H3 R2V variants | Actual route; full-quality/Turbo/HyperFlow, checkpoint/adapter |
| Girl-singing cs04 | Owner/broker canonical shape/checkpoint receipt |
| Sustained streams | Capacity and paced qualification for both models separately |
| Independent review | Exact combined integration head after required checks |
| Publication/adoption | Verified remote ref and sulphur integration receipt |

Bind each device cell to source/tree/dirty state, build/runtime, weights/adapters,
inputs, mesh, clock, shape, schedule and options. Reuse #381 only when these match;
its serving pause remains intact. Do not wait on cancelled #420 or duplicate #442.

Current acceptance requires sustained
unique playable video received by the API caller / elapsed wall time >1,
counting concurrent galaxies. Include queue, generation, export and delivery costs;
remove overlap and padding from playable duration. Slow individual clips do not
alone fail this aggregate throughput criterion.

Retain the initial >=60-chunk AND >=300-second net playable video protocol and
the >=1800-second (30-minute) playback protocol, both adapted to concurrent
production. Report per-clip latency, declared startup buffer, concurrency and
participating galaxies separately. Measure unpaced capacity separately from
deliberately paced playback; pacing is not a capacity measurement. Preserve
continuity, bounded host/device memory, quality, changed references and zero
underruns after the declared startup buffer. Finite batches do not qualify as
sustained delivery. Branch alignment closes neither stream order.

Historical contract (superseded): sequential production and mandatory capacity
and service RTF <=0.8 were the earlier gates. Run142's immutable snapshots retain
that history; these thresholds are not current acceptance requirements.

## Delivery boundary

No task policy line permits landing on ltx-rt with `ttp push=True`, and current
combined-source e2e and independent review remain missing. No shared push is
attempted. A delivery successor must read current policy, validate the combined
candidate, use its authorized publication route and obtain sulphur's integration
receipt. sulphur-mac retains ltx-server main and production rollout.

## Continuation checkpoint, task #25

Task #25 resumes the clean task #11 checkpoint `75fb8cb4421f` without changing
model implementations. Run 107 contains command/write/read-back proof, task-record
snapshots and a separate inventory of preserved artifacts. Run 96 has no final
handoff: its remote check was terminated by scheduling cancellation at 11:22 PST
(exit 143), before a successful verdict. It must not be counted as passed.

| Component | Exact preserved head | Validation and review | Delivery state |
| --- | --- | --- | --- |
| Served LTX | `0b43a32cda9128299bbd10ce72706bf6d435c254` | 49,532-byte bundle and ancestry verified; 14 commits retained | GitHub backup ref exists; combined-source qualification pending |
| H3 controller | `3f8a885f5f48b087e937747d4994ee6069d6bc07` | Required hooks and 31 tests passed; independent #15 accepted this exact head | #14 still records review, with no landing or PR |
| H3 encoder | `6081695e3b225ad544c3c18e38c656df73d4373a` | Eight CPU codec tests and applicable hooks passed | Required publication checks and independent review pending |
| H3 transport | `14fa5e4d5c50b24402e12c6d616eabcfb21a0439` | Run135: 36 lightweight required tests passed; 26 prior artifact hashes verified | Review #37 queued; no publication |
| ltx-server measurement | `c19b5fb3f146f6169e7604785b2263e16f8696f3` | Saved final-head receipt: 133 passed, one skipped, diff check passed | Separate repository; no publication recorded |
| Combined main/LTX | `173bce4455988341ce4358c566d57ec1f9de8a80` | Owner-reported unity build failure; no unit-test pass | Bundle/full-log request `836a136e3305d732` unanswered |

The accepted H3 review covers only the controller through `3f8a885f5f48` on
source `15d5b0028220bb994f6d3652fd9948bea04e34c6`. Reuse that verdict without
repeating it. It does not cover the later encoder/transport commits or a different
combined-source head. A pending delivery row is not evidence of a failed review,
and an accepted review is not evidence of a landing. Do not mark #14 superseded
until a verified destination contains the accepted change.

The H3 chain already contains the controller, encoder and transport; do not copy
or recreate another streaming module. Task #20 owns subsequent model-bound H3
receipt work. Measurement commit `c19b5fb3f146` belongs to ltx-server, not tt-metal;
preserve its bundle and coordinate the existing interfaces with #18/#20 instead
of applying a cross-repository cherry-pick.

The run107 finite-measurement snapshot used #22/run101 and #26/run106. They supply
source/config preparation, not new matched console/API results. #22 recovered
five owner warm conv-VAE samples (720p client-ready p50 12.30 seconds, n=3;
448p p50 11.925 seconds, n=2), but still lacks the runnable F4 runtime/model path.
#26 found no H3 runtime, weights or broker on blx06. Its nominal 6/10-second
requests map to 158/243 frames at 24 fps; final ratios must use decoded playable
duration. Neither record qualifies sustained streaming or proves API parity.

At run107, the configured publication-check host was blx06. The former Python bootstrap
failure is not an implementation dependency: run96 initialized Python 3.12.13.
Its interrupted check does require completion. Shared-branch landing remains
disabled by `delivery.code_tasks_may_push=false`; an own-branch draft route still
requires the applicable checks and independent review. No approval of unchanged
H3 code needs to be requested again. Current-source model validation, review of
new integration changes and sulphur-mac's rollout receipt remain separate gates.

Run96's outside-path list alone establishes no mutation. Its recorded commands
read the named skill, memory and verifier files and execute the named binaries;
the source worktree is clean at its preserved head. No restoration is justified
by that list. The #24/run104 report, line 11, assesses run103's `rg` and `sed`
operations as reads; it contains no separate run96 attribution verdict. Run107's
statement that run104 contained no run96/run103 audit was too broad.

Next integration work starts with the existing owner's bundle/full build log,
reproduces the exact unity failure in an isolated authorized checkout, preserves
the conv3d repair, and applies the smallest justified fix. The model benchmarks
continue independently. No changed streaming implementation, build success,
publication or production adoption is claimed by this checkpoint.

## Resumed evidence, 12:58 PST

User message #77 supplies the model bindings and supersedes the runtime-discovery
waits above. H3 uses `~/tt-metal-ltx-rt-b` on f101/f102 at main `15d5b0028220`,
base weights `/home/sulphur/models/minimax-h3`, and the
`lightx2v-h3-turbo` four-step adapter through cs04 `/v1`. LTX uses
`~/tt-metal-ltx-rt` at `0b43a32cda91`. Sulphur-mac owns F4 staging directly from
f101; its planned staging is not a completion receipt. No runtime rebuild or
duplicate model transfer is required by this source-reconciliation task.

Run107's required document check on `f200eec0a6fb` timed out without check output
after 608 seconds (exit 124); its detached wrapper ended with exit 1. Run125's
locked blx06 inspection found both exact temporary sandboxes absent and no
matching process, and closed machine-change record #119. Record #118 had already
closed automatically. The analogous #20/run119 required check also timed out;
its targeted CPU results remain distinct from a required-check pass.

Run125 extends the operation audit to runs109/115 using local transcripts only.
Their flagged commands search memory with `rg` and read the skill with `sed`;
neither writes those paths. Tool output establishes shell execution despite the
separate environment-dump guard notices. No historical before/after baseline
exists, and unrelated writers remain unattributed. No protected file was reopened
or restored. Detector note `656ee47112d63e0b` was read by ttproject-mac.

The run125 remote snapshot records `ltx-rt` as `522ef26916bd`; the LTX backup still
resolves to `0b43a32cda91`, and `ttp/ltx-rt-main-merge` is absent. #14 remains in
review without a recorded landing, while its exact-head acceptance by #15 remains
valid. Owner task sulphur-mac #442 remains queued; request `836a136e3305d732`
still has no named reply. Its combined candidate and full build log are needed
for that integration step. These facts do not gate independent #18/#20 work.

## Owner identity and delivery update, November 10, 2026

[Source/config manifest](served_source_config_20261110.json) preserves the three
sulphur-mac #494 receipts `a078cdf5c764f649`, `a9fcb4a71b3a8638` and
`8717765e1628189a`. These are owner-reported identities, not new device results.
The manifest keeps H3 tt-metal `15d5b0028220`, ltx-server `874684a`, and LTX
`0b43a32cda91` distinct. The benchmark prefix `main6653391` still needs its
repository mapped; it does not replace either model-source identity.

H3 base revision `42ed227ee7df40d41602854ae760620d6eb651fe` supplies distinct
`transformer/` FL2V and `transformer_ref/` R2V shards. Turbo uses separate v1.0
adapters, each 1,383,677,808 bytes; their full SHA256 values and the HyperFlow
identity are preserved in the manifest. The existing F4 v0.1 R2V adapter remains
excluded from matched served-baseline comparisons. No large weights were rehashed.

Public requests use `POST /v1` with `model=h3-fast`; references select the R2V
twin. `h3-fast-r2v` is an internal registry name rejected at the public edge.
The receipt binds four NFE (`H3_STEPS=5`), strength 1, video shift 6, tree-default
audio shift, 24 fps, audio enabled and a 4x8 mesh. Nominal six-/ten-second requests
produce 158/243 frames; performance ratios require measured new playable duration.
Genuine first/last conditioning and reference conditioning remain separate cells.

Existing sulphur-mac #557 owns staging on OM-F4-GBH02 via request
`0af212b741dd1979`. No alternate destination or duplicate copy is requested.
Consume its exact paths, hashes, ownership, admission and free-disk receipt when
available. Staging completion and a verified FastH3 timing packet remain absent.

Run135 confirms run125's upload timeout (inner exit 124, wrapper exit 1), local
PID 92778 absent, and the locked remote probe showing sandbox
`/tmp/ttp-check.yZEF4tXv` absent with no matching processes. Change #132 closed at
13:51 PST; #129 was also closed by task #20. No duplicate cleanup was performed.
The effective delivery configuration now has no `checks_host` override; unchanged
lightweight document validation runs locally through `ttp checks`. Heavy checks
remain remote. Run142 records the final committed-head check result separately.

Maintainer receipts `f5becd9a5b2768ed` and `5e18f5ce27efc504`, plus run135's
invocation audit, correct the earlier protected-access characterization: ordinary
SSH use of its own configuration was a guard false positive, not worker misconduct.
Selective `printenv TTP_RUN_DIR` calls also triggered false positives. The recorded
`rg`/`sed` operations were reads, not confirmed writes. No historical content
baseline proves unrelated authorship; no protected contents were reopened or
restored, and the guard stays enabled. This run uses individual variable reads.

Review #15 accepted controller `3f8a885f5f48` with no blocking findings and reused
its exact-head hooks/31-test receipt. Its scope does not cover encoder/transport
or a newly combined source tree. Review #37 owns transport `14fa5e4d5c50` and
prerequisite provenance; integration must consume its accepted head before
publication. Neither #14 nor #20 is marked landed or superseded. The existing
main/LTX build candidate still needs its owner bundle and complete failure log.

R-1ac7 requires all four H3/LTX 2.5 IC-LoRA FL2V/R2V qualification cells on TYO2,
with enforced 160 W read back for every participating chip, actual AICLK and
throttling recorded. Those measurements are absent here. CPU passes and identity
receipts establish neither native acceptance nor sustained caller throughput >1.
All four cells remain unqualified without the required evidence. Required
review, validated publication and sulphur-mac's integration receipt remain open.

[authored by AI]
