# JIT Compile Server — Hang Handoff

**Goal:** fix the wedge where a client's parallel JIT warm-pass (`--precompile`, many workers)
routed to the remote compile server hangs **forever** waiting on a compile response.

This doc gives you: how to connect to the server, how to reproduce, the diagnosis so far
(with file:line evidence), and ranked fix directions. Branch: `mstaletovic/agent_eval`.

---

## 1. TL;DR of the bug

Under a high-concurrency remote warm pass (`--precompile --precompile-workers 128` with the JIT
server enabled), **one `parallel_compile` worker blocks permanently** in
`JitCompileRpcSession::wait_all()` → capnp `Promise::wait()` → `epoll_wait`, waiting for a
`CompileResults` response that never arrives. The whole warm pass never joins → the run hangs.

- **It's a rare race that scales with remote-request volume.** Reproduced **0/12** at ~203 kernels,
  but **2/2 at ~843 kernels** (server cache cold, so every kernel must compile + round-trip).
- At hang time the **server is idle and `received == done`** (it finished and logged every request
  it got) yet the client still waits → the lost message is at the **transport/delivery** layer, and
  the client has **no timeout** so a single lost response wedges forever.

The two independent defects (either fix alone makes it non-fatal):
1. **Client has no timeout** on the wait — turns any lost message into a permanent hang.
2. **Server uses a non-idiomatic hand-rolled `poll()` loop** as the single pump for all connections
   + all cross-thread compile-completion fulfillments — the suspect locus for a dropped response.

---

## 2. How to connect to the server

The server runs on **`bgdepyc01`** inside docker container
**`bgdepyc01-special-mstaletovic-for-reservation-24729`** (mstaletovic's reservation; you can
restart it freely).

```bash
C=bgdepyc01-special-mstaletovic-for-reservation-24729
ssh bgdepyc01                          # then: docker exec $C bash -c '...'
# one-liner pattern used throughout:
ssh -o ConnectTimeout=10 bgdepyc01 "docker exec $C bash -c '<cmd>'"
```

Facts about the running server:
- Binary: `./build_Release/tools/jit_compile_server`, cwd `/localdev/mstaletovic/tt-metal`.
- Runs in tmux window **`jit_server`** under a restart wrapper (`...; echo SERVER_EXIT rc=$? ... > jit_server_exit.log`),
  with `LD_PRELOAD=/localdev/mstaletovic/tt-metal/crash_handler.so`.
- Listens on `0.0.0.0:5555` **inside** the container; docker-mapped to **`bgdepyc01:54210`** on the host.
- Log: `/localdev/mstaletovic/tt-metal/jit_compile_server.log` (server's cwd).
- Server's own JIT `.elf` cache: `/tmp/tt-metal-cache/<build_key>/kernels/<kernel>/...`.
- Currently launched with `TT_METAL_CCACHE_KERNEL_SUPPORT=1` (from an experiment — **note: ccache gives
  0 benefit on the server because preprocess-mode ships pre-expanded `.ii`, which ccache marks
  uncacheable; safe to drop on next restart**).

Restart it (in the tmux window) — set any new env on the launch line:
```bash
ssh bgdepyc01 "docker exec $C bash -c '
  tmux send-keys -t jit_server C-c; sleep 3
  tmux send-keys -t jit_server \"LD_PRELOAD=/localdev/mstaletovic/tt-metal/crash_handler.so TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:5555 ./build_Release/tools/jit_compile_server > jit_compile_server.log 2>&1\" Enter'"
```
Verify up: `echo > /dev/tcp/bgdepyc01/54210` (host) and `pgrep -f jit_compile_server` (container).
Enable periodic metrics with `TT_METAL_JIT_SERVER_LOG_INTERVAL_MS=1000` (logs inflight/queued/peak).

**Client side** (point a tt-metal run at the server):
```bash
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54210
export TT_METAL_JIT_PREPROCESS=1        # ship self-contained .ii (works for arbitrary dev kernels)
```

---

## 3. How to reproduce the hang

Need a workload with **many distinct kernels** and a **server-cold** cache (so every kernel
compiles + round-trips — that's what makes the race fire).

1. Reconstruct an op with lots of kernels. `rms_norm` (eval DB run 456) at 500 golden cases →
   ~843 kernels reproduced it 2/2. Reconstruct with:
   ```python
   # from tt_ops_code_gen root, PYTHONPATH=. ; eval.db.get_kernels/get_host_code/get_artifacts(conn, 456)
   ```
   (see the eval `db.py` accessors). Write host_code → `ttnn/ttnn/operations/rms_norm/`,
   kernels → `.../rms_norm/kernels/`, tests/artifacts by their repo-relative `name`.
2. **Dewarm the server `.elf` cache** so it must compile fresh (this is essential):
   ```bash
   ssh bgdepyc01 "docker exec $C bash -c 'find /tmp/tt-metal-cache -type d -name \"rms_norm*\" -exec rm -rf {} + ; echo dewarmed'"
   ```
3. Run the warm pass @128 against the server (client env from §2 set), cold local cache, under a
   timeout so it can't wedge your shell:
   ```bash
   export PYTHONPATH=$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen   # eval pkg importable (repo-root `eval` symlink)
   export TT_METAL_CACHE=$PWD/<fresh_cold_dir>
   mapfile -t IDS < <first_500_golden_nodeids>
   timeout --signal=INT --kill-after=20 360 \
     scripts/run_safe_pytest.sh --precompile --precompile-workers 128 --run-all "${IDS[@]}"
   ```
   **Wedge signature:** the warm pass prints `PRECOMPILE: ===== warmup =====` but **never** prints
   `compiled N programs in Xs` (i.e. `up_front_compile` never returns); `timeout` kills it (exit 124/137).

To confirm it's the wedge (not just slow): while hung, the **client** has a thread in
`wait_all → Promise::wait → epoll_wait`, and the **server** is idle with `received == done`.

---

## 4. Diagnosis so far (with evidence)

### Client — where it blocks (gdb-confirmed)
```
up_front_compile::parallel_compile()::$_0
  → CompileProgram → ProgramImpl::compile          (tt_metal/impl/program/program.cpp, remote path ~2244–2280)
  → RemoteCompileCoordinator::finish()             (remote_compile_coordinator.cpp:83; wait loop at :97)
  → JitCompileRpcSession::wait_all()               (jit_compile_rpc_client.cpp:272–281)  ← NO TIMEOUT
  → capnp Promise::wait() → kj::EventLoop::wait → epoll_wait   ← blocked forever
```
`wait_all()` is a bare `promise.wait(waitScope)` per request, no timeout, no failure path. **This is
what makes a lost response fatal rather than slow.**

### Server — finished everything, response not delivered
- The compile handler offloads to a 134-thread `tf::Executor`; on completion it calls the
  cross-thread fulfiller (`jit_compile_service.cpp:299`), and a `.then` on the event loop serializes
  the response (`:306`). The "done" counter increments right before `fulfill()` (`:296`).
- At hang time: server threads sleeping, `0` live `cc1plus`, and **`received == done`** → the server
  completed (and "done"-logged) every request it received. So the missing message is in
  **request-receipt or response-delivery**, not compute.

### The suspect locus — non-idiomatic event loop
`jit_compile_server_controller.cpp:90–100` does **not** block on the listener; it hand-rolls:
```cpp
while (!should_stop_) {
    auto count = wait_scope.poll();      // NON-blocking: runs only already-ready events
    listen_promise.poll(wait_scope);
    if (count > 0) last_events = now; else if (now-last_events > 10ms) sleep_for(1ms);
}
```
Every one of the (up to 128) connections' socket I/O **and** every cross-thread `fulfill()` must be
pumped through this one `poll()`. Idiomatic capnp would block in `epoll` via
`listen_promise.wait(wait_scope)` / `runForever()`, woken reliably by both socket events and the
`kj::Executor` cross-thread wakeup. The poll-spin is the most plausible place a fulfillment/response
gets dropped under load. (Confirmed live: the server's event-loop thread sits in `hrtimer_nanosleep`
= the 1 ms sleep, not `epoll_wait`.)

### Latent server bug (didn't fire in the observed case, but real)
`jit_compile_service.cpp:288` catches `const std::exception&` only. `kj::Exception` is **not**
std-derived, so a kj/capnp exception thrown inside the `thread_pool_.silent_async` lambda would
escape **before** `(*fulfiller)->fulfill()` at `:299` → that RPC promise never completes → client
hangs. (Observed hang had `done==received`, so this wasn't the trigger *that* time, but it's a real
second path to the same symptom.)

### Connection / dedup model (context)
Each `parallel_compile` worker builds its **own** `RemoteCompileCoordinator` (a per-compile local in
`program.cpp:2253`) → its own `JitCompileRpcSession` → its own `capnp::EzRpcClient`. With one
endpoint, 128 workers = **128 concurrent connections into one `TwoPartyServer` event loop**.

### Ruled out
- **Thread-pool / deduper deadlock:** NOT the cause. In `in_flight_compile_deduper.hpp`, a waiter
  only exists once the owner has claimed the entry, and the owner holds its pool thread continuously
  from claim through `erase` (no yield) — so a waiter's owner is always running, never starved.
- **ccache:** server doesn't benefit (preprocess `.ii` is uncacheable) and isn't implicated.

---

## 5. Fix directions (ranked)

1. **Client `wait_all()` timeout + local-compile fallback** — *the* fix; highest value, no server
   redeploy. Give the capnp `wait()` a deadline (`kj::Timer`/`afterDelay().exclusiveJoin(promise)`);
   on timeout, fail that kernel back to a **local** compile instead of blocking forever. Converts
   "permanent wedge" → "one slow kernel." File: `jit_compile_rpc_client.cpp:272`
   (and surface the timeout through `RemoteCompileCoordinator::finish()` so the coordinator retries
   locally). The local path already exists (`program.cpp` non-remote branch).

2. **Server: replace the poll-spin with a blocking loop** —
   `listen_promise.exclusiveJoin(shutdown).wait(wait_scope)`, where `shutdown` is a cross-thread paf
   fulfilled by `stop()`. Blocks in `epoll`, woken by both socket and cross-thread events, still
   shuts down cleanly. Removes the fragile single-pump delivery. File:
   `jit_compile_server_controller.cpp:90–100`.

3. **Server: also catch `kj::Exception` / `...`** at `jit_compile_service.cpp:288` and still
   `fulfill()` with `success=false`, so no compile path can silently drop a fulfillment.

4. **Reduce fan-out / improve dedup** (efficiency + lowers race probability): share a small session
   pool across `parallel_compile` workers instead of 128 independent connections, and/or have the
   `InFlightCompileDeduper` retain completed entries briefly so post-compile duplicate requests
   coalesce instead of recompiling.

Recommended order: **(1) first** (stops the bleeding regardless of server cause), then **(2)+(3)**
to actually remove the lost-response source, then **(4)** if you want @128 to be efficient.

---

## 6. Build & deploy

- **Client fix** (`jit_compile_rpc_client.cpp`, `remote_compile_coordinator.cpp`, `program.cpp`):
  rebuild + **install** the lib — `build/lib/*.so` are install COPIES, so `ninja _ttnn.so` alone
  won't update the loaded lib:
  ```bash
  cmake --build build_Release --target install     # propagates _ttnn.so
  ```
  Verify the fix is actually live before benchmarking (a stale `_ttnn.so` silently hides changes).
- **Server fix** (`jit_compile_server.cpp`, `jit_compile_service*.cpp`, `jit_compile_server_controller.cpp`):
  build inside the **container** (it has the newer glibc the server needs):
  ```bash
  ssh bgdepyc01 "docker exec $C bash -c 'cd /localdev/mstaletovic/tt-metal && cmake --build build_Release --target jit_compile_server'"
  ```
  then restart the tmux `jit_server` window (see §2).

---

## 7. Verify the fix

- **Re-run the §3 repro** (843+ kernels, server `.elf` dewarmed, @128) **multiple times** — it must
  complete every time (no `timeout` kill; `compiled N programs in Xs` always prints).
- For the client-timeout fix specifically: simulate a lost response (e.g. temporarily make the
  server drop/delay one response, or kill+restart the server mid-warm-pass) and confirm the client
  **falls back to local compile** and the run still passes, instead of hanging.
- Watch the server stays healthy: `received == done` and it drains to idle; no orphaned
  fulfillers / fd leak under repeated @128 load.

## 8. Key files
- Client: `tt_metal/impl/jit_server/jit_compile_rpc_client.cpp`,
  `tt_metal/impl/jit_server/remote_compile_coordinator.cpp`,
  `tt_metal/impl/jit_server/in_flight_compile_deduper.hpp`,
  `tt_metal/impl/program/program.cpp` (remote path ~2244–2280).
- Server: `tt_metal/tools/jit_compile_server/jit_compile_server.cpp`,
  `tt_metal/impl/jit_server/jit_compile_service.{hpp,cpp}`,
  `tt_metal/impl/jit_server/jit_compile_server_controller.cpp`,
  `tt_metal/impl/jit_server/rpc.capnp`.
- Parallel warm-pass caller: `ttnn.graph.up_front_compile` / `tests/plugins/up_front_collect.py`.

---

## 9. RESOLUTION (2026-06-17)

**Status: reproduced, root-caused, fixed, verified.**

### Reproduction
The reconstructed `rms_norm` golden suite (`eval/golden_tests/rms_norm/test_golden.py`,
5041 cases → **1392 unique programs / ~4000 distinct kernel compiles**) wedges the warm pass
**reliably** at `--precompile-workers 128` with a cold local cache + dewarmed server `.elf`
cache. Driver: `tt_metal/impl/jit_server/repro/repro_warmpass.sh` (runs ONLY the warm pass under
the device flock + a timeout — no heavy real run). Signature: never prints
`UP_FRONT_COLLECT: compiled N programs`, killed by `timeout` (exit 137). 504-kernel runs did
**not** reproduce (0/1); 1392-kernel runs reproduced 2/2.

### Root cause (pinned with /proc + ss evidence, `repro/gdb_wedge_*.txt`)
At hang time, two `parallel_compile` workers are blocked in
`JitCompileRpcSession::wait_all() → Promise::wait() → epoll_wait()` while the **server is idle
and `received == done`** (it compiled and logged every request). The decisive new evidence:

- The wedged client holds **2 ESTABLISHED TCP connections** to the server (`ss`/`/proc/net/tcp`),
  both with **empty Recv-Q/Send-Q**.
- The **server side has ZERO connections** on its port (container `:5555` and host docker-proxy
  `:54210` both show none).

i.e. a **half-open connection**: the server-side socket was torn down (under load, through the
docker userland port-proxy) **without the response being delivered and without the FIN/RST
reaching the client**. capnp on the client is still waiting for a response that can never come,
and `wait_all()` had **no timeout and no keepalive**, so a single lost response wedges the warm
pass forever. (The server also accumulated ~900 open fds under repeated @128 load.)

### Fixes applied
1. **Client timeout + local fallback (THE fix — makes the hang impossible).**
   - `jit_compile_rpc_client.{hpp,cpp}`: `wait_all()` now races each RPC response against a
     `kj::Timer` deadline (default **240 s**, `TT_METAL_JIT_SERVER_TIMEOUT_S`, `0`=legacy
     unbounded). On timeout/disconnect it throws a typed `RemoteCompileTransportError`
     (distinct from a genuine `success=false` compile error, which still propagates).
   - `program.cpp` (remote path): catches `RemoteCompileTransportError` from
     `coordinator.finish()` and **falls back to a local compile** of the program, reusing the
     already-prepped `submitted_kernels` (re-running `prep_kernel` would re-add the
     `ALIGN_LOCAL_CBS_TO_REMOTE_CBS` reserved define and assert). `ensure_kernel_binaries` is
     cache-aware, so kernels the server *did* deliver are read from disk; only the lost ones
     recompile. Converts "permanent wedge" → "one slow, locally-compiled program".
2. **Server: blocking event loop (root-cause reduction).**
   `jit_compile_server_controller.{hpp,cpp}`: replaced the non-idiomatic non-blocking
   `wait_scope.poll()` + 1 ms-sleep spin (the single pump for all 128 connections' socket I/O
   **and** every cross-thread compile fulfillment) with an idiomatic blocking
   `listen_promise.exclusiveJoin(shutdown).wait(wait_scope)`. Blocks in `epoll`, woken reliably by
   both socket events and cross-thread fulfillments; `shutdown` is a cross-thread paf fulfilled by
   `stop()`. Verified to still shut down cleanly (2 s on SIGINT).
3. **Server: catch-all fulfillment.**
   `jit_compile_service.cpp`: the pool lambda now also catches `kj::Exception` and `...` (not just
   `std::exception`) before `fulfill()`, closing the latent path where a non-std exception escapes
   and the RPC promise never completes.

### Verification
- **Before:** full @128 run wedges 2/2 (+ a 3rd here) — never prints `compiled N programs`,
  killed by timeout (exit 137).
- **Client fix alone** (server *unchanged*, still dropping): full @128 run **completed** —
  `compiled 1392 programs in 249.9s (workers=128, errors=0)`; 3 responses were lost
  (programs 737/1315/1817), each timed out and **fell back to local compile**. No wedge.
  (`repro/verify_clientfix_*.log`.)
- **Both fixes deployed:** server rebuilt in-container + restarted; clean SIGINT shutdown
  confirmed (2 s); **two** full @128 runs **completed** — `compiled 1392 programs in 180.0s` and
  `in 259.3s`; 3 and 4 responses lost + fell back to local. No wedge.
  (`repro/verify_bothfix_*.log`, `repro/verify_run3_*.log`.)
- **Net: 3/3 post-fix runs complete (vs 3/3 wedges before).** Lost-response count is unchanged by
  the server fix (3–4 per run) — the client timeout + local fallback is what unwedges it.

**Honest finding on the server fix:** the blocking event loop did **not** reduce the drop rate
(≈3 lost responses both with and without it) and the server's fd count stayed ≈900. That
confirms the lost response is at the **transport / docker userland-proxy layer under load**, not
the poll-spin — so the **client timeout + local fallback is the actual cure**; the server changes
(blocking loop, catch-all) are correct hardening (idiomatic, no CPU spin, closes the latent
kj-exception drop path) but are not what unwedges the run. The residual warm-pass `errors` seen in
one run were pre-existing transient `preprocess-and-ship: -E failed` / partial-ELF-rename flakiness
during normal remote submit (fired *before* the timeout fallbacks), independent of this fix, and
are best-effort (those programs simply cold-compile in the real run). A separate follow-up worth
filing: the server-side fd accumulation (~900 fds) under repeated @128 load.

### Build / deploy notes
- Client/program changes: `cmake --build build_Release --target install` on the **client** box
  (build/lib/*.so are install copies; `ninja _ttnn.so` alone is not enough).
- Server changes: copy `jit_compile_server_controller.{hpp,cpp}` + `jit_compile_service.cpp` to
  the container checkout, `cmake --build /localdev/mstaletovic/tt-metal/build_Release --target
  jit_compile_server`, restart the tmux `jit_server` window.

### Debug artifacts (committed under `tt_metal/impl/jit_server/repro/`)
`repro_warmpass.sh` (focused warm-pass repro), `repro_server_hang.sh` (full run_safe_pytest
repro), `collect_ids.py`, `all_golden_nodeids.txt` / `first500_nodeids.txt`, `gdb_wedge_*.txt`
(wedged-client backtraces), and the `verify_*`/`run_all_*` logs.

---

## 10. KEEPALIVE VALIDATION + CLEAN E2E (2026-06-19)

Follow-up after the §9 client-timeout fix: added **TCP keepalive** as a *second, latency-independent*
dead-connection detector and validated it; then captured clean end-to-end compile times.

### Why keepalive (vs the app-level timeout from §9)
The app timeout (`TT_METAL_JIT_SERVER_TIMEOUT_S`) has an unavoidable tension: it must exceed the
**whole-batch** latency or it false-positives on slow-but-alive responses. Measured directly —
`repro/server_push_t20_results.csv`: a **20 s** app timeout caused **25 / 42 spurious fallbacks** at
q150 / q200 (queued, still-alive responses misclassified as dead), while a 240 s timeout avoided that
but re-opens a long wedge window on a *real* drop. TCP keepalive sidesteps this: it probes the
**connection** (idle 5 s + 3×2 s = ~11 s, plus `TCP_USER_TIMEOUT=15 s`), so a genuinely-dead
half-open is killed in ~11–15 s **regardless** of how long a legitimate compile takes — a
slow-but-alive connection keeps ACKing probes and is never touched.

Implementation (`jit_compile_rpc_client.cpp`, client-only — server unchanged): replaced
`capnp::EzRpcClient` with a manual `TwoPartyClient` over a `kj::AsyncIoStream` so we can arm
`SO_KEEPALIVE` + `TCP_KEEPIDLE/INTVL/CNT` + `TCP_USER_TIMEOUT` on the socket before wrapping it.
Knobs: `TT_METAL_JIT_SERVER_KEEPALIVE` (default 1), `..._KEEPALIVE_IDLE_S` (5), `..._INTVL_S` (2),
`..._CNT` (3), `..._USER_TIMEOUT_MS` (15000). A dead connection surfaces as
`::read(...): Connection timed out` → typed `RemoteCompileTransportError` → local-compile fallback
(same path as the app timeout).

### A/B result (airtight) — `repro/ka_validate_*.log`, `repro/keepalive_validation_results.csv`
Full reconstructed rms_norm suite @128, **server `.elf` dewarmed each run**, **`TIMEOUT_S=0`** (app
timeout OFF, so *only* keepalive can break a dead connection):

| run | keepalive | drops | outcome |
|-----|-----------|-------|---------|
| kaON r1 | ON | 0 | **COMPLETED** `compiled 1392 programs in 179.2s` — no false positives across ~4000 kernels |
| kaON r2 | ON | 3 | **COMPLETED** `... in 184.9s`; 3 dead conns detected as `Connection timed out` → local fallback |
| kaOFF r1 | OFF | ≥1 | **WEDGE** — killed at 440 s wall, never printed `compiled N programs` (legacy unbounded wait) |

Server side during these runs (`jit_compile_server.log`): `peak_inflight=128`, **no errors /
disconnects logged, drains to 0 sockets** — i.e. the server delivered every request; the loss is
purely the Docker userland port-proxy under 128-way concurrency (confirms §9 root cause with fresh
evidence). **Verdict: the dead-connection check is useful** — it converts a permanent wedge into a
~11–15 s detection + local fallback, *without* the app timeout's false-positive problem. Recommend
keepalive ON as the primary mechanism and the app timeout as a coarse backstop (or 0/off).

### Clean E2E (full suite: 1680 pass / 2941 skip / 420 xfail; 1392 programs / ~4000 kernels)
| config | warm-pass compile | warm run | note |
|--------|-------------------|----------|------|
| cold-inline (no precompile) | — | **2088 s** | kernels compile on-demand during the run |
| precompile-server @128 (keepalive on) | **182.5 s** (221 s phase incl. device open) | **50.85 s** | 0 errors, 0 fallbacks; run reuses warm on-disk cache |

⇒ server-precompile e2e **~272 s vs ~2120 s cold = ~7.7×**.

### Farm-vs-inline crossover sweep (`repro/repro_farm_vs_inline.sh`, `farm_vs_inline_results.csv`)
N distinct programs, **fully cold both sides** (fresh `TT_METAL_CACHE` each iter; **ccache disabled
via `CCACHE_DISABLE=1`** — see caveat). local-inline = compile-on-demand during the run (this box is
cpuset-pinned to **8 cores**); remote-farm = server precompile @128 + warm run. 0 fallbacks across
the whole sweep.

| N (prog) | kernels | inline (cold) | farm warm+run = total | farm speedup |
|----------|---------|---------------|-----------------------|--------------|
| 5   | 18  | 13.9 s | 12.6+5.2 = **17.8 s** | 0.78× |
| 10  | 33  | 21.5 s | 14.1+5.3 = **19.4 s** | 1.11× |
| 20  | 63  | 37.5 s | 15.3+5.8 = **21.1 s** | 1.78× |
| 30  | 90  | 49.1 s | 16.0+5.9 = **21.9 s** | 2.24× |
| 50  | 150 | 78.8 s | 19.1+6.6 = **25.7 s** | 3.07× |
| 70  | 189 | 100.3 s | 22.3+6.9 = **29.2 s** | 3.43× |
| 100 | 234 | 126.4 s | 25.1+9.2 = **34.3 s** | 3.69× |

Crossover ≈ **10 programs**: below it the farm's fixed cost (warm-pass device-open + collect + a
second device-open for the run, ~17 s floor) loses to just compiling ~30 kernels locally; above it
the farm wins and the gap widens (inline scales ~linearly with kernel count on 8 cores; farm compile
is near-flat on 128 cores). Extrapolates to the 7.7× full-suite number above (1392 programs).

**CCACHE CAVEAT (important, cost me a redo).** The login profile sets
`TT_METAL_CCACHE_KERNEL_SUPPORT=1` and the JIT build wraps `ccache g++` **unconditionally** —
`TT_METAL_CCACHE_KERNEL_SUPPORT=0` does **not** turn ccache off. A first pass left ccache on: with
the firmware prime + **nested** sets (p5⊂p10⊂…), the local-inline runs got ccache hits from earlier
iterations and ran ~1.5–2× too fast (e.g. N=70 inline 49.8 s ccache-warm vs **100.3 s** truly cold),
collapsing the apparent farm advantage. The farm path can't use local ccache (preprocess-shipped
`.ii` is uncacheable), so only inline was understated. Fix = ccache's own switch **`CCACHE_DISABLE=1`**
(pure passthrough); verified cold via an isolated `CCACHE_DIR` probe that stayed empty (0 objects).
Always disable ccache with `CCACHE_DISABLE=1` for cold-compile A/Bs on this box.

NOTE: `run_safe_pytest --precompile`
still logs `✗ warmup FAILED (pytest exit 1) -> warmed NOTHING; running COLD` — this is a
**misclassification**: the warm pass's NO_DISPATCH test bodies legitimately exit 1, but the JIT
cache *was* warmed (proven by the 2088 s → 51 s run drop). `precompile_warm()` in
`scripts/run_safe_pytest.sh:275` should treat exit 1 (body failures) as success when the collect
log shows `compiled N programs ... errors=0`; today the headline `total_s`/`warmup_s` in
`e2e_results.csv` come out `NA` because of this. Open cleanup, not a correctness bug.
