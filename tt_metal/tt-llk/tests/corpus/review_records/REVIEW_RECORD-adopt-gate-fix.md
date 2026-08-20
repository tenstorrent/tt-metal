# Review record — cross-run adoption source-root provenance gate (adopt-gate-fix)

Defect citation: wave-12 review, master-HANDOFF ledger 19 (the sweep cell-reuse
violation).  Proven by five EXECUTED negatives in that review: _adopt_prev_cell
(sweep_2x2.py) adopted cells from --prev-run roots checking only cell-level
greenness + jobkey + .text-hash match — it never checked the SOURCE ROOT's
provenance.  All five constructed dirty-root shapes were consumed:
  1. foreign-pin roots (adopted with zero provenance recorded);
  2. roots with NO pin record at all;
  3. QUARANTINED-marked roots;
  4. CONTAMINATION-NOTE.md roots;
  5. --skip-craq-gate-tainted roots consumed by an untainted run.
Plus the laundering corollary: transitive adoption (copytree then OVERWRITE
REUSED_FROM.txt) erased the run that actually touched silicon.

## Scope of this fix (branch adopt-gate-fix, base 7f1f0be9d9)

1. `Sweep._prev_root_provenance` (sweep_2x2.py): once-per-root cached gate run
   before ANY cell is adopted from a --prev-run root:
   - REFUSE on the same dirt markers newest_clean_runs skips (root-name
     *CONTAMINATED*/*quarantine*, QUARANTINED marker file,
     CONTAMINATION-NOTE.md at root level);
   - REFUSE fail-closed when the root has no readable pin record (PIN_STAMP
     first line, then preflight.json cc1plus_sha256 — the wrapper guard's own
     order and its own rationale: unknown provenance is how contamination
     starts);
   - REFUSE when the root's preflight records craq_gate_skipped taint and THIS
     run gates on CRAQ; a taint-MATCHED run adopts and the taint line is
     propagated into this run's MANIFEST.txt per adopted cell;
   - a recorded pin differing from this run's pin does NOT refuse (the
     per-cell .text key against THIS run's classify hashes protects the
     number) but prints a loud CROSS-PIN ADOPTION line and the mismatch is
     recorded per adoption.
2. Transitive origin preservation: `Sweep._reuse_chain` — a source cell's own
   REUSED_FROM.txt chain is EXTENDED, never overwritten; the final marker
   lists the full chain oldest-first with each hop's root path AND recorded
   pin, so the run that touched silicon is always entry 0 (legacy single-line
   markers parse as an origin hop with pin "unrecorded").
3. Per-adoption provenance: scoreboard.json reused_cells entries (and the
   printed reuse line) now carry source_pin (cc1plus sha 12-hex prefix),
   source_taint (bool) and origin_root (first hop of the chain).
4. Stale-comment eradication: the four "annotator-only today" --prev-run
   comments (sweep_wrapper_lib.sh newest_clean_runs header + weekly/nightly/
   headline wrappers) now describe the real consumers and this gate.
5. RED selftests in selftest_sweep_core_overhaul.py (constructed-negative
   pattern of that harness): foreign-pin-recorded (adopt + provenance
   recorded + loud line), no-pin-record (refused), QUARANTINED (refused),
   CONTAMINATION-NOTE (refused), taint-mismatch (refused), taint-match
   (adopted + MANIFEST taint propagated), transitive chain (full chain
   preserved), scoreboard provenance fields.  Verified RED against the base
   commit's sweep_2x2.py: all 8 new checks FAIL on 7f1f0be9d9 (the five
   dirty-root shapes all ADOPT, the taint line is absent, and the transitive
   marker names only the intermediate run — origin laundered), selftest exits
   1; ALL GREEN on the fixed code.
6. y2038 fixture fix (separate commit): selftest_sweep_wrapper_lib.sh used
   year-2099 `touch -t` mtimes; on this box's y2038-clamping filesystem the
   ordering scrambled and 3 newest_clean_runs cases FAILED at every wrapper
   preflight (FATAL-first — every scheduled sweep refused; ledger 18 open).
   Dates moved to 2033 preserving ordering semantics; re-executed on this
   box: 3 FAIL -> 17/17 ALL PASS.

## Gates run at this commit stack (this box, no silicon, no toolchain)

- python3 ast-parse: sweep_2x2.py + selftest_sweep_core_overhaul.py OK
- bash -n: sweep_wrapper_lib.sh, selftest_sweep_wrapper_lib.sh, all three
  wrappers OK
- selftest_sweep_core_overhaul.py: 32 PASS / 0 FAIL (incl. the 8 new
  provenance checks; RED-verified against base as above)
- selftest_enforcement_gates.py: 16 PASS, rc 0
- selftest_conf_lint.sh: rc 0; conf_lint.sh on the REAL conf: GREEN
  (cc1plus 01aed0d8d58d…, sim pins verified)
- selftest_sweep_2x2_report.py: 23 PASS, rc 0
- selftest_batched_silicon.py: 51 PASS, rc 0
- selftest_sweep_wrapper_lib.sh: 17/17 ALL PASS on this box (post-y2038 fix;
  3 FAIL before)
- selftest_witness_preflight.py (18 PASS), selftest_dejagnu_gate.sh,
  selftest_corpus_leg_store.py (11 PASS), selftest_corpus_watch.py (12 PASS):
  all rc 0

No conf/pin/baseline/measurement changes.  No device jobs.

## Independence disclosure

Authored by the auditor side: this fix was written and verified by the
reviewing session lineage that found the defect (wave-12), not by the lane
that introduced the adoption prober (laneDA/DC).  Per the ledger-17
precedent, that is disclosed here rather than treated as independent review;
the gate's selftests are constructed negatives executed against both the
broken base and the fix.
