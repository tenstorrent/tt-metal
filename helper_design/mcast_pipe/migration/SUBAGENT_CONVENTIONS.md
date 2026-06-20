# mcast_pipe migration — per-tier subagent conventions (read first)

You are migrating a TIER of kernel call sites from **raw open-coded mcast+handshake primitives** to the
materialized helper **`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`** (MCAST_PIPE_API_VERSION 7). Work from
`/localdev/sjovic/tt-metal`. Device is a single Wormhole b0, SHARED + SEQUENTIAL — never run device
tests in parallel, never in the background.

## Environment / build
- `source python_env/bin/activate` before any test.
- Kernel-only edits (.cpp/.hpp under .../kernels/ and the helper header) need **NO host rebuild** — JIT
  compiles them at test time. You are only editing kernel code, so do NOT run ./build_metal.sh.

## The v7 helper API (target form)
Read `helper_design/mcast_pipe/proposed_helpers.md` and the helper header for the authoritative API.
Quick shape (translate raw primitives INTO this):
- **Sender:** `dataflow_kernel_lib::SenderPipe<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES,
  PRE_HANDSHAKE, CONSUMER_READY_SEM_ID, DataReadySignal>(noc, McastRect<NOC_ID>{x0,y0,x1,y1})` then
  `.send(src,dst,size)`. Control/flag-only: `.send_signal()` (no arg).
- **Receiver:** `dataflow_kernel_lib::ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID>(noc)`
  then `.receive(sender_x, sender_y)` (sender coords passed to the call). Control: `.receive_signal()`.
- **PRE_HANDSHAKE=false** for a pure-signal sender that never gates on a consumer-ready sem → then OMIT
  CONSUMER_READY_SEM_ID (trailing default UNUSED_SEM_ID), or the new static_assert fires.
- The ctor sets the local data-ready cell VALID; `send()` re-asserts VALID per call (the 20cf0df46ee fix)
  — so a rotating-role / loopback core that clobbers the cell each round is handled.
- **BEST REFERENCE**: find an already-migrated sibling and copy its exact spelling. To list migrated
  call sites: `git grep -l "SenderPipe\|ReceiverPipe" ttnn/ models/ tt_metal/`. Match a sibling of the
  same role/op-family (e.g. for a sharded-gn sender, look at a migrated sharded-ln/gn kernel).

## Per-kernel loop (run-all mode)
For each kernel in the tier, sequentially:
1. `git status` clean-ish checkpoint (note HEAD).
2. Read the kernel + a migrated sibling. Rewrite the open-coded mcast+handshake block to use the helper.
   Keep any HOLE interleaving (gather/DRAM reads between phases) by using the separately-callable phases
   (`send_signal()` / data `send()` / `receive()`), per the audit note for that kernel.
3. **Smoke** ONE parametrization first with `--dev` (catches compile errors + hangs):
   `scripts/run_safe_pytest.sh --dev "<file>::<test>[<one full nodeid>]"`. Collect nodeids with
   `python -m pytest <file>::<test> --collect-only -q` then pick one that hits the kernel (match the
   factory dispatch condition in the test_map validation note).
   - NOTE: pytest `-k` cannot contain `=`. Pass the FULL nodeid as a positional arg (quote it; it may
     contain spaces/parens) instead of `-k`.
4. If smoke PASSES: run the family validation with `--run-all` for full counts:
   `scripts/run_safe_pytest.sh --run-all "<file>::<test>"` (or the specific nodeid family).
   Confirm the kernel actually JIT-built: `grep -rl <kernel_basename> generated/` (or the build cache).
5. On PASS (0 failed, no hang):
   - `git add <kernel>` and commit: `apply mcast_pipe to <kernel basename>` with a one-line body noting
     the test + counts. END the message with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
   - LEDGER WRITE-BACK (mandatory): in `helper_design/mcast_pipe/migration/ledger.json` set the entry's
     status=migrated, migrated_api_version=7, commit=<hash>, last_verified="2026-06-20". Use this exact
     idiom so the pre-commit EOF hook is a no-op (json.dump omits trailing newline → hook aborts commit):
       ```python
       json.dump(led, open(p,'w'), indent=1)   # then:
       open(p,'a').write('\n')                  # add trailing newline
       ```
     Commit the ledger alongside (or immediately after) the kernel commit.
   - Write `helper_design/mcast_pipe/migration/log/<kernel_basename>.md`: diff summary, test + counts,
     lines removed, notes.
6. On FAIL or HANG:
   - Try to fix (wrong call-site wiring, sem id, rect, PRE_HANDSHAKE, num_dests). Retry.
   - After ~2 failed attempts — ESPECIALLY a hang — escalate: spawn the `ttnn-expert-debugger` agent with
     the kernel path, the failing nodeid, the triage report path (printed as `SAFE_PYTEST: triage report:`),
     and what you tried. Use its diagnosis to attempt one more fix.
   - If still failing AND it's a **helper DESIGN limitation** (the API itself would need a new mode/
     topology — e.g. COUNTER inc_multicast, value-carrying flag, multi-mcast-per-call, a missing fork):
     DO NOT modify the helper. `git restore <kernel>` (keep tree green), set ledger status=quarantined
     (or deferred) with a one-line `census_note` reason, write the log, and MOVE ON.
   - If it's just a stubborn migration bug you can't crack: same revert + quarantine with reason.

## Commit hygiene
- Per-kernel atomic commits (kernel + its ledger write-back). Never one big commit.
- Pre-commit hooks may reformat/modify files and ABORT the commit. If a commit aborts: re-`git add` the
  hook-modified files and re-commit. Verify `git rev-parse HEAD` actually advanced.
- NEVER push, rebase, or reset. Local commits only.
- NEVER run `tt-smi -r` — run_safe_pytest.sh resets the device automatically.

## Return contract (compact — NO diffs, NO full logs)
Return ONLY:
```
tier: <N>
per_kernel:
  - kernel: <basename>, status: migrated|quarantined|deferred, validation: pass|fail|hang,
    commit: <hash or ->, diff_lines_removed: <n>, note: <one line>
tier_totals: {migrated, quarantined, deferred}
artifacts: [log file paths]
```
Verbose detail goes to the log files on disk, not your return message.
