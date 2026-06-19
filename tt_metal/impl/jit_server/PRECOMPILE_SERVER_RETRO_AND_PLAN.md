# Remote JIT Precompile Server — Retrospective & Path to Prod

**Branch:** `mstaletovic/server_precompile_investigation`
**Goal of this doc:** (1) record what was built, what broke, and what was fixed; (2) lay out the
concrete cleanup + wiring steps to run `run_eval.py` kernel-generation runs against the remote JIT
compile server in a reliable "prod" configuration.

Companion docs: `SERVER_HANG_HANDOFF.md` (deep dive on the hang root-cause + keepalive validation +
benchmarks), and `tt_ops_code_gen/eval/PRECOMPILE_EVAL_HANDOFF.md` (eval-integration design notes).

---

## Part 1 — Retrospective

### 1.1 What we set out to do
Offload the JIT kernel-**compile** step of a TTNN run to a remote 128-core compile **server (farm)**,
driven by the `--precompile` warm pass (`tests/plugins/up_front_collect.py` →
`ttnn.graph.up_front_compile`). The warm pass collects every distinct program in a test selection,
compiles them all in parallel (locally, or shipped to the server), and populates the on-disk JIT
cache so the real run is execute-only. This is the lever to make broad eval runs (hundreds–thousands
of distinct kernels) fast on core-starved client boxes.

### 1.2 The headline bug: permanent warm-pass wedge
At high concurrency (`--precompile-workers 128`, server cache cold so every kernel round-trips), the
warm pass **hung forever** — one `parallel_compile` worker blocked in
`JitCompileRpcSession::wait_all() → capnp Promise::wait() → epoll_wait`, waiting for a `CompileResults`
response that never arrived. Reproduced reliably at ~1392 programs / ~4000 kernels (the reconstructed
`rms_norm` golden suite), never at ~200.

**Root cause (proven with `/proc` + `ss`, not just gdb):** a **half-open TCP connection**. At hang
time the client held N ESTABLISHED connections with empty queues while the server **and** the host
Docker `docker-proxy` showed **zero** connections — the server-side socket was torn down under load
(through Docker's userland port-proxy, host `:54210` → container `:5555`) **without** the response
being delivered and **without** a FIN/RST reaching the client. capnp waited forever; `wait_all()` had
**no timeout and no keepalive**, so a single lost response wedged the entire warm pass. The server was
idle with `received == done` — it had compiled and logged every request; the loss was purely at the
transport/proxy layer.

### 1.3 Bugs found & fixed (in commit order on the branch)
1. **`firmware: clear NoC packet tags at idle-erisc startup`** (`ca8abbafa69`) — unrelated dispatch
   firmware hygiene surfaced during triage.
2. **`jit: make remote-compiled kernels cacheable + reusable`** (`9feb97d0e57`) — server-warmed
   kernels are now read back from the on-disk cache by the local run (server-warm → local-run
   handoff). Without this the warm pass warmed nothing reusable.
3. **`jit_server: fix permanent warm-pass hang on lost remote-compile response`** (`bf09b02bb70`) —
   the cure:
   - **Client app-timeout + local fallback.** `wait_all()` races each RPC response against a
     `kj::Timer` (`TT_METAL_JIT_SERVER_TIMEOUT_S`, default 240 s, 0 = unbounded) → throws typed
     `RemoteCompileTransportError`; `program.cpp` catches it and **falls back to a local compile** of
     that program (reusing the already-prepped `submitted_kernels`). Converts permanent wedge → one
     slow program.
   - **Server hardening** (not the cure): blocking `listen_promise.exclusiveJoin(shutdown).wait()`
     loop replacing the non-idiomatic poll-spin, plus a catch-all around the compile fulfillment.
     Measured: did **not** reduce drop rate — confirming the drop is transport-layer.
4. **TCP keepalive (the new, uncommitted change in this session)** — `jit_compile_rpc_client.cpp`
   (`.cpp` only): swapped `capnp::EzRpcClient` for a manual `TwoPartyClient` over a `kj::AsyncIoStream`
   so we can arm `SO_KEEPALIVE` + `TCP_KEEPIDLE/INTVL/CNT` + `TCP_USER_TIMEOUT` on the socket before
   wrapping it. A genuinely-dead half-open connection is now detected by the **kernel** in ~11–15 s
   (idle 5 + 3×2, user-timeout 15 s) regardless of how long a legitimate slow compile takes — fixing
   the app-timeout's fundamental false-positive problem (a 20 s app timeout caused 25–42 *spurious*
   fallbacks on slow-but-alive connections at q150/q200). Knobs:
   `TT_METAL_JIT_SERVER_KEEPALIVE` (default 1), `..._KEEPALIVE_IDLE_S/INTVL_S/CNT`, `..._USER_TIMEOUT_MS`.

### 1.4 Validation done this session (2026-06-19)
- **Keepalive A/B (airtight), `TIMEOUT_S=0` so only keepalive can act, full suite @128 server-cold:**
  - keepalive ON, 0 drops → completed (no false positives across ~4000 kernels).
  - keepalive ON, 3 drops → all caught as `Connection timed out` → local fallback → completed.
  - keepalive OFF → **WEDGE** (killed at 440 s). ⇒ the dead-connection check is **useful**.
- **Packet-loss re-confirmed server-side:** `peak_inflight=128`, no errors logged, drains to 0
  sockets — server delivered everything; loss is the Docker proxy under 128-way concurrency.
- **Clean E2E (full suite):** cold-inline **2088 s** vs server-precompile **182.5 s compile + 50.85 s
  run ≈ 272 s = ~7.7×**.
- **Farm-vs-inline crossover** (cold both sides, ccache disabled — see gotcha): crossover ≈ 10
  programs; farm speedup grows 1.8× @20 → 3.1× @50 → 3.7× @100 programs.

### 1.5 Issues / gotchas hit along the way (so we don't repeat them)
- **ccache silently contaminated the first farm-vs-inline sweep.** The login profile sets
  `TT_METAL_CCACHE_KERNEL_SUPPORT=1` and the build wraps `ccache g++` **unconditionally** — setting
  `TT_METAL_CCACHE_KERNEL_SUPPORT=0` does **not** disable it, and ccache (`/localdev/mstaletovic/.ccache`)
  is a separate store from `TT_METAL_CACHE` so clearing the JIT cache doesn't touch it. With nested
  test sets the inline runs got hits and ran ~1.5–2× too fast. Fix for cold A/Bs: **`CCACHE_DISABLE=1`**.
- **Dirty-device crashes between repro reps.** A killed/wedged warm pass leaves dispatch cores
  running → next device open aborts with `Read unexpected run_mailbox value: 0x40`. `repro_warmpass.sh`
  doesn't reset between reps; the sanctioned reset is `run_safe_pytest.sh` (resets a dirty device at
  startup under flock). Reset after any killed rep.
- **`.so` install-copy trap.** `build/lib/*.so` are install copies; `ninja _ttnn.so` alone doesn't
  update the loaded lib. Use `cmake --build build_Release --target install` and verify the fix is live
  before benchmarking. (Confirmed this session: keepalive strings present in `libtt_metal.so`.)
- **`run_safe_pytest --precompile` mislabels the warm pass `✗ warmup FAILED (exit 1)`.** The
  NO_DISPATCH warm-pass bodies legitimately exit 1, but the JIT cache *was* warmed. The wrapper treats
  exit 1 as failure → `e2e_results.csv` `total_s`/`warmup_s` come out `NA`. Cosmetic/metrics bug, not
  a correctness bug (see plan item C4).
- **fd "leak" was a non-issue** — the ~900 fds are `libtracy` perf_event handles (128 CPUs × 7),
  static across idle + load; sockets reclaim cleanly.

---

## Part 2 — Plan: clean up & get to prod

Tackle in order. Each item is independently verifiable. ☐ = todo.

### A. Land the code (the actual feature)
- ☐ **A1.** Commit the keepalive change: `tt_metal/impl/jit_server/jit_compile_rpc_client.cpp`
  (`.cpp`-only). Message: client TCP keepalive for half-open dead-connection detection + env knobs.
- ☐ **A2.** Commit the collect speedup: `tests/plugins/up_front_collect.py` (no-op
  `eval.metrics.check_output` during collect — skips the addr-0 readback + metrics on garbage output).
- ☐ **A3.** Submodule `tt_ops_code_gen`: decide on the uncommitted `eval/metrics.py` (M) — commit or
  revert in the submodule, then bump the submodule pointer in the superproject. Don't leave it dirty
  (`m`).
- ☐ **A4.** Update `SERVER_HANG_HANDOFF.md` + this doc are already committed-ready; include them.

### B. Make the server usable from `run_eval.py` (the integration gap)
Today the server is **purely env-driven** — there is **no** JIT-server reference in the eval runner
code; `run_eval.py` does `env = os.environ.copy()` and forwards, so exporting the right vars before
the run is sufficient *mechanically*, but there's no validation and no documented runbook.
- ☐ **B1. Decide the eval server-config contract** (open decision from `PRECOMPILE_EVAL_HANDOFF.md`
  §"Config correctness"): if `TT_METAL_JIT_SERVER_ENABLE=1`, **require** `TT_METAL_JIT_PREPROCESS=1`
  (preprocess-and-ship is mandatory for arbitrary agent-generated kernels) and a reachable
  `TT_METAL_JIT_SERVER_ENDPOINT`. Add a pre-flight check (TCP connect to the endpoint) in
  `eval_test_runner.sh` that aborts early with a clear message rather than silently degrading.
- ☐ **B2. Decide fallback policy:** default = degrade to local compile if the server is unreachable
  (`TT_METAL_JIT_SERVER_ENABLE=0`); opt-in `EVAL_REQUIRE_JIT_SERVER=1` = abort loudly (for
  core-starved hosts that genuinely need the farm). The keepalive + app-timeout already handle
  mid-run drops; B2 is about *startup* reachability.
- ☐ **B3. Default keepalive ON in the eval path**, app-timeout as a coarse backstop (e.g.
  `TT_METAL_JIT_SERVER_TIMEOUT_S=240`, `TT_METAL_JIT_SERVER_KEEPALIVE=1`). Document in the runner.
- ☐ **B4. Write the canonical runbook** (env block below) into `PRECOMPILE_EVAL_HANDOFF.md` and/or the
  eval-launch skill so a server-backed run is one documented invocation.

Canonical server env block for a run_eval / eval run:
```bash
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54210
export TT_METAL_JIT_PREPROCESS=1            # mandatory with the server
export TT_METAL_JIT_SERVER_KEEPALIVE=1      # half-open detection (~11-15s)
export TT_METAL_JIT_SERVER_TIMEOUT_S=240    # coarse backstop
export EVAL_PRECOMPILE=1                     # warm phase on (auto-on for broad runs anyway)
export EVAL_PRECOMPILE_WORKERS=128           # farm width
```

### C. Clean the repo working tree
- ☐ **C1. Repro dir is 102 MB / 140 files, only 10 committed.** Purge untracked logs/caches; keep the
  *valuable* artifacts. Verify `.gitignore` covers `*.log`, `*_cache*/`, `*_ccache*/`, `sets/`,
  `cold_cache_*`. Concretely:
  - Keep (and commit) the durable scripts + results: `repro_warmpass.sh`, `repro_farm_vs_inline.sh`,
    `keepalive_validation_results.csv`, `farm_vs_inline_results.csv` (+ `..._ccacheON_contaminated.csv`
    as a cautionary record, or drop it), `diag_drops.sh`, `repro_server_knee.sh`, `collect_ids.py`.
  - Delete: all `*.log` (2 MB+ each), `*_cache*/`, `*_ccache*/`, `cold_cache_*`, `prof.out`, the
    `matrix_*`/`cross_*`/`kat_*`/`scd_*`/`sk_*`/`sp*_*` one-off logs, `fvi_*.log`, `fvi_ccache_probe/`.
  - Decision: do we keep this whole `repro/` tree on the prod branch at all, or move it to a
    throwaway branch / `.gitignore` the lot? Recommend: keep ~5 canonical scripts + the 2 results
    CSVs + the 2 handoff docs; gitignore everything else.
- ☐ **C2. Remove the `rms_norm` reconstruction** — `ttnn/ttnn/operations/rms_norm/` and
  `tests/ttnn/unit_tests/operations/rms_norm/` are **repro-only** (rms_norm is nuked on the eval
  branch and was reconstructed from the eval DB to drive the hang repro). Do NOT commit them; delete
  or gitignore. Confirm nothing else now depends on them.
- ☐ **C3. Submodule untracked files:** `eval/COLLECT_PASS_DESIGN.md`, `eval/PRECOMPILE_EVAL_HANDOFF.md`,
  `settings.local.json`, `worktrees/` — commit the two design docs into the submodule (they're the
  real handoff record), gitignore `settings.local.json` + `worktrees/`.
- ☐ **C4. Fix the `precompile_warm()` exit-1 misclassification** (`scripts/run_safe_pytest.sh:275`
  and the mirror in `eval/eval_test_runner.sh`): treat warm-pass exit 1 as success when the collect
  log shows `compiled N programs … errors=0`. Restores honest `total_s`/`warmup_s` metrics. Low-risk,
  high-clarity.

### D. Verify the deployed binaries & server
- ☐ **D1. Client lib is live:** after A1, `cmake --build build_Release --target install`; confirm the
  keepalive symbol is in the loaded `libtt_metal.so` (`strings … | grep TT_METAL_JIT_SERVER_KEEPALIVE`).
  **Note for run_eval:** it clones a fresh tree and rebuilds — so the fix must be **committed** (A1)
  for the clone to get it; a working-tree-only change won't propagate to a clone.
- ☐ **D2. Server parity & hygiene** (`bgdepyc01`, container
  `bgdepyc01-special-mstaletovic-for-reservation-24729`, host `:54210`):
  - The server currently runs on branch `mstaletovic/jit-preprocess-and-ship`. Keepalive/timeout are
    **client-only** so the protocol is compatible (verified: this session's runs succeeded). Still,
    confirm the server tree is at the intended prod commit and rebuild if needed
    (`cmake --build /localdev/.../build_Release --target jit_compile_server` **in the container**).
  - Drop the leftover ccache-experiment env on next restart (0 benefit; preprocess `.ii` is
    uncacheable). Restart cleanly in the tmux `jit_server` window; verify `received==done`, drains to
    0 sockets, and `echo > /dev/tcp/bgdepyc01/54210` from the client.
  - Enable periodic metrics (`TT_METAL_JIT_SERVER_LOG_INTERVAL_MS=1000`) for observability during
    prod runs.

### E. Pre-flight a real eval run against the server
- ☐ **E1. Reset device to a clean baseline** (single case via `run_safe_pytest.sh`).
- ☐ **E2. Smoke test:** one small server-backed `run_eval.py` (or `eval_test_runner.sh`) run on a
  real eval op (NOT the reconstructed rms_norm) — confirm warm pass routes to the server (server
  `count=` climbs), 0 wedges, fallbacks are rare and recover, and golden results are unaffected.
- ☐ **E3. Scale test:** a broad run (hundreds of programs) @128 — confirm the ~3–7× e2e win holds and
  keepalive catches any drop without wedging.
- ☐ **E4.** Capture a short "prod readiness" note (numbers + any residual fallbacks) and we're done.

### Definition of "prod ready"
Keepalive + collect-speedup committed (A); server config validated/documented in the eval path (B);
working tree clean, repro junk gone, rms_norm reconstruction removed (C); client lib live + server
confirmed compatible/healthy (D); a broad server-backed `run_eval.py` run completes with the expected
speedup and no wedge (E).
