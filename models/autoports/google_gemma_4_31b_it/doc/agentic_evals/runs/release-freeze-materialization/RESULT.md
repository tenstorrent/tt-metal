# Prepared freeze materialization

`prepared.json` contains exact stock dispatch argv for current80/C8 and banked40/C8 fallback. Each candidate's dry/release argv is identical; source task arrays exactly match its registered cohorts.json rung (checked programmatically). No selection, dispatch, frozen-pass or optional image-build success is claimed. No source or central ledger changed;0 device hours used.

80 candidate: TT eef534e5cd6bcaf37957647de4d8d4eff4f0e047, TTI42dc40754e465cf8c9ec993027ff59d3c936832c.40 fallback: sameTT, TTIe5a9a7638a1f46f22c0d89acf2ac333da5d53c02. Both use proven digest bf1ada4081f7fa338ed8495a02324390d3b6124861cd625a0d2fb2d6a255d425, pinned plugin ca7a18246e4111629ed6cb4a7d1192002749fccf, C8/B8/maxseq8/pool20480. Config/recipe/limits/HF revisions are explicit in the packet. Source inspection of42dc eval entry confirms max_concurrent8, revision842da, max_length262144, request14400, output126976, thinkingtrue, streamfalse,temp1/topk20/topp.95. Do not mistake client effective max_length262080 after harness adjustment for an advertised-context reduction.

Only pre-freeze source/image substitution is allowed: coordinator may replace TT/image together with a matched actually accepted immutable pair and CI native-response confirmation. Optional f579 build42334508049 is pending, not passed by this evidence. Operator9 permits fallback without waiting. TTI tasklist/concurrency/limits remain fixed per candidate; changes require a newly materialized candidate and dry. Once dry succeeds, no semantic substitution is allowed. The separate local benchmark arm retains its existing f579+af1ef freeze and source hashes at `runs/framed-benchmark-dry/freeze.json`; only its documented fresh ARTIFACT_ROOT output substitution is permitted. Distinct per-arm source identities must remain explicit.

## Stronger current no-score dry criterion

The earlier aggregate-reply-only procedure is superseded for this freeze by complete **initial8 native replies**, not merely eight arbitrary completed requests. Use the private cumulative observer `bringup/artifacts/ci-native-large-reply-visibility/observe.py` (hashes recorded in prepared.json), with --clients8 and an isolated actual evaluator-start boundary. Its observe.py:78-95 accepts old TTNATIVEHEX1 and new TTNATIVEHEX2; framing.py validates all chunks, declared size, SHA256 and JSON before reconstruction. Pending/corrupt/incomplete records are not replies.

Invocation template (private observation only, not a harness wrapper):

```
python bringup/artifacts/ci-native-large-reply-visibility/observe.py --snapshot <complete-live-log> --state <fresh-private-state.json> --output <private-observation.json> --boundary-marker 'Running generate_until requests' --fetch-start <UTC> --fetch-end <UTC> --clients 8 --year 2026
```

The coordinator must verify boundary uniqueness/isolation, exact frozen inputs/image/module/full60/context/settings/IDs and observe active_rows8 during evaluator traffic. Require `initial_c_identified=true`, `all_initial_server_replies_observed=true`, no compromised source coverage, no pending/framing/truncation/API errors, complete finished native chat choices for all eight distinct initial received request IDs. Native response construction is not client acknowledgement: also preserve successful parsed client progress and no API/sentinel errors. HTTP200 or progress8/N alone cannot substitute for identity-matched native replies.

Observer intentionally returns `no_score_currently=unknown`. Coordinator must separately check freshly fetched latest event times, fetch interval/lag, current request activity and live job metadata, plus k<N and absence of aggregation markers in a complete up-to-date snapshot. Never infer current no-score from stale logs or jobstatus alone. If scoring overtakes observation, this is not a clean no-score dry. Do not accept a complete rehearsal absent an explicit exception.

GPQA has zero task sandboxes/task environment images, so those pull/concurrency obligations are vacuous; model server Docker/source/readiness still require actual evidence. Preserve live logs/observer hashes/run-job metadata **before** stock cancellation, then use the authorized CI cancellation route only after criteria are met. Do not SIGTERM a local device process. Stock cancellation may omit final artifacts; saved native evidence is necessary. The local benchmark's existing dry is a separate arm, not proof that GPQA passed.

## Detect stock workflow drift without modifying it

Read-only check now returned dispatch workflow blob82613c81987a7093e15ad4477e5b5391a35ba32e and tt-shield branch commit ba2f03318608be52c5cb2085469598a89dcfa0fc. These are observations, not a lock: manual dispatch references mutable `vvukoman/enable-on-dispatch-cross-repo-trigger`, which invokes relative stock build/test workflows. Do not edit or pin the workflow yourself.

Immediately before dry and release, record `gh api repos/tenstorrent/tt-agentic-bringup-qb2/contents/.github/workflows/manual-tt-shield-dispatch.yml` blobSHA and `gh api repos/tenstorrent/tt-shield/commits/vvukoman/enable-on-dispatch-cross-repo-trigger` commitSHA. For each actual run preserve run API `head_sha`, `path`, `referenced_workflows` (when exposed), and startup log `Uses: ... (SHA)` plus resolved action SHAs. Existing logs show this actual resolved reusable-workflow SHA. Compare actual dry versus release provenance, not just pre-dispatch branch lookups (race possible). Preserve relevant workflow blob hashes at those commits too; external action refs may move independently.

If drift is found before release, the final frozen route is unproved and needs review/re-dry; do not silently claim same route. A mismatch first discovered after release starts is a release limitation/failure to report, not a reason to fix/re-dispatch the release. The stock interface cannot guarantee a mutable upstream branch will not move between lookups/dispatch. No stricter guarantee is claimed.

Remaining decisions/evidence: actual80 result and fit versus40 fallback, chosen TT/image pair, optional logging CI confirmation, final measured queue+tail projection, fresh GPQA dry evidence and unchanged actual workflow provenance. The packet intentionally leaves these to the coordinator.
