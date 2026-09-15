<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# JIT Compile Server — User's Guide

> ## Experimental. Not for production.
>
> This feature is intended to **accelerate development and CI iteration**, not to run in
> production or on shared/untrusted infrastructure. Before you use it, understand the
> security posture:
>
> - **No authentication.** The server trusts any client that can open a TCP connection to it.
> - **No encryption.** The transport is Cap'n Proto over plain TCP. If the link crosses an
>   untrusted network, tunnel it (SSH, WireGuard, VPN).
> - **Remote code execution is the feature.** The client tells the server which compiler to
>   run and with which flags; the server runs it. Anyone who can reach the endpoint can
>   execute code as the server's user. (Shell injection specifically is mitigated —
>   `posix_spawn` with an explicit argv, never `system()` — and client-supplied path
>   components are rejected if absolute or containing `..`. That hardens the edges; it does
>   not change the fact that the server exists to run compilers on the client's behalf.)
> - **Default bind is `localhost:9876`**, so out of the box only local processes can reach
>   it. Widening the bind address is an explicit, deliberate act.
>
> Recommendation: run it inside a trusted cluster network or behind a firewall, never on a
> public interface, and never with elevated privileges.

---

## 1. What it is

Normally each tt-metal process compiles its own kernels with its own local compiler
processes, caching the results under `TT_METAL_CACHE`. The JIT compile server moves that
compilation to one or more separate server processes:

1. The client computes the kernel hash and generates the kernel's source/header files as usual.
2. If a valid ELF already exists in the client's local `TT_METAL_CACHE`, **nothing is sent** —
   the local cache always wins.
3. Otherwise the client sends a compile request (compiler path, flags, defines, include
   string, source list, and the generated files' contents) to a server and waits.
4. The server compiles and links into its own on-disk cache, then returns the ELF bytes.
5. The client writes the ELF into its local cache and loads it normally.

Two layers of deduplication make this worthwhile:

- **Per-process (client side):** identical kernel hashes submitted within one process
  collapse into a single request.
- **Global, in-flight (server side):** concurrent requests for the same
  `build_key` + kernel hash — *even from different processes on different hosts* — collapse
  into a single compile. Late arrivals wait on the first one's result rather than
  recompiling. This is the layer a local on-disk cache cannot give you.

Plus the server's own on-disk cache, which serves repeat requests after the fact.

**Kernel-to-server routing is deterministic:** endpoint index = `kernel_hash % num_endpoints`.
Every client that is given the **same endpoint list in the same order** routes a given
kernel to the same server. That is what makes cross-process and cross-host dedup and cache
reuse work. If clients get different lists, or the same hosts in a different order, you
still get correct results but you fragment the dedup and the cache.

---

## 2. When to expect a performance gain

Two different mechanisms are at work, and they are worth keeping apart because they scale
very differently.

**Avoided compiles.** In-flight dedup and the server's on-disk cache make a kernel get
compiled once instead of once per process. This requires **kernel overlap** — the same kernel
hash being needed by more than one compiling process, or by the same process more than once
over time. Essentially all of the speedup lives here, and it scales with how much your
kernel sets overlap.

**Relocated compiles.** Separately, and even with zero overlap, the compiling itself happens
on the server's CPU instead of the application's. This does not reduce the total amount of
work — it only changes who pays for it — and it adds RPC round-trips and ELF transfer.
So it helps only when the application host's CPU is the constraint and the server's is not.
Treat it as a secondary effect: it is what makes the co-located multi-host setup (§4b)
affordable, not a reason to deploy the server on its own.

| Situation | Expect | Mechanism |
| --- | --- | --- |
| N processes compile largely the same kernel set, concurrently | Best case — ideally 1 compile instead of N | avoided compiles |
| Same kernels requested later by a different process or host | Server cache hit; near-free | avoided compiles |
| No overlap, but the application host is CPU-contended while the server host is idle | Modest, and it can still be eaten by the RPC and transfer overhead | relocated compiles |
| One process, all-unique kernels, cold caches, uncontended host | **Net loss.** You pay RPC, source shipping, and ELF transfer for work the local parallel build would have done anyway | neither |
| Warm local `TT_METAL_CACHE` | No requests sent at all — identical to not using the server | n/a |

Rule of thumb: if your workload is a single process with a unique kernel set, don't bother.
If it is many processes converging on a shared kernel set, the server is where the win is.

---

## 3. Build and start the server

```bash
# Build (from the tt-metal root)
./build_metal.sh --enable-ccache          # or: cmake --build build --target jit_compile_server

# Run it. Defaults: bind localhost:9876, cache root /tmp/tt-metal-cache/
./build/tools/jit_compile_server
```

The server **never opens a device** — no accelerator required on the machine that runs it.
What it does need is the tt-metal source tree and the SFPI toolchain present at the absolute
paths the *client* names in its requests (see §5). The server takes no configuration from
`TT_METAL_HOME` itself; the tree just has to be there.

A more realistic invocation, with a relocated cache root and a widened bind:

```bash
export TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:9876        # widen the bind — see the disclaimer
export TT_METAL_JIT_SERVER_CACHE_ROOT=/local/jit-server-cache/
./build/tools/jit_compile_server
```

It shuts down cleanly on `SIGINT`/`SIGTERM`.

Concurrency: the server compiles on a Taskflow executor sized to
`std::thread::hardware_concurrency()`. One server process per host is enough; it will use
the whole box.

### Cache root

`TT_METAL_JIT_SERVER_CACHE_ROOT` is the directory where the server stores compiled objects,
ELFs, and uploaded firmware, partitioned by `build_key`. It is **not** the same as a
client's `TT_METAL_CACHE`: clients still write their own local copy of each ELF after the
server returns it. The server cache exists so a later request for the same kernel (from
this process or another) can skip recompilation.

The default is `/tmp/tt-metal-cache/`. Relocate it when `/tmp` is too small or too
ephemeral — kernel object files and ELFs accumulate for every unique kernel the server has
ever seen, and the cache is never garbage-collected. A few large workloads can fill a
tmpfs. Point it at a node-local disk with enough headroom, and keep it distinct from any
client `TT_METAL_CACHE` (the layouts differ). If you share a host with clients that have
neither `TT_METAL_CACHE` nor `$HOME` set (common in containers), they also fall back to
`/tmp/tt-metal-cache/` — set both explicitly to avoid colliding.

### Point a client at it

```bash
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876
pytest tests/... # or your application
```

That's the whole client-side change. If `TT_METAL_JIT_SERVER_ENABLE=1` is set but no
endpoint is configured, the client fails fast with a clear error rather than silently
falling back to local compilation.

---

## 4. Deployment patterns

### 4a. Single host, multiple processes

*Several ranks, pytest workers, or model instances on one machine.*

Each process needs its **own** `TT_METAL_CACHE` (a shared cache directory across concurrent
processes causes compilation conflicts). That isolation is exactly why local caching gives
you no cross-process reuse: process 2 cannot see what process 1 just compiled. The server
fixes that, and the in-flight deduper means simultaneous starts collapse instead of racing.

```bash
# Terminal 1 — server
./build/tools/jit_compile_server

# Terminals 2..N — workloads, one private cache each
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876
TT_METAL_CACHE=/tmp/cache_rank0 python my_workload.py --rank 0 &
TT_METAL_CACHE=/tmp/cache_rank1 python my_workload.py --rank 1 &
wait
```

Nothing special about the source tree here: the server runs from the same checkout as the
application, so §5's shared-tree requirement is satisfied for free.

### 4b. Multiple hosts, servers co-located with the workload

*Multi-host run; one server per host; no extra machines.*

This does **not** require dedicated hardware. During the compile phase the application is
mostly waiting, so the CPU it isn't using is what the server spends. The work is shifted,
not duplicated.

The gain over "just use a local-disk `TT_METAL_CACHE` on each host" is structural: local
caches cannot be reused across hosts, so every host pays for every kernel. With the server
fleet, a kernel needed by ranks on 4 hosts is compiled once — by whichever server the
hash routes to — and the other three get it from that server's cache or from the in-flight
dedup. **Whenever kernel sets overlap across hosts, the gain is real, not speculative.**

```bash
# On every host: start a server (widened bind so peers can reach it)
TREE=/nfs/tt-metal      # the shared tree, same absolute path everywhere
for h in host0 host1 host2 host3; do
  ssh $h "cd $TREE && \
    TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:9876 \
    TT_METAL_JIT_SERVER_CACHE_ROOT=/local/jit-cache/ \
    nohup ./build/tools/jit_compile_server > /tmp/jit_server.log 2>&1 &"
done

# Launch the workload with the identical, identically-ordered endpoint list everywhere
export ENDPOINTS=host0:9876,host1:9876,host2:9876,host3:9876
mpirun -H host0,host1,host2,host3 \
  -x TT_METAL_HOME -x LD_LIBRARY_PATH \
  -x TT_METAL_JIT_SERVER_ENABLE=1 \
  -x TT_METAL_JIT_SERVER_ENDPOINTS="$ENDPOINTS" \
  ./my_workload
```

Two things to get right:

- **Identical endpoint list, identical order, on every rank.** Otherwise routing diverges
  and dedup fragments.
- **Per-rank `TT_METAL_CACHE`.** Ranks on the same host must not share one. If `$HOME` is on
  NFS, prefer a node-local path — a shared-filesystem cache adds latency and contention to
  the local-hit fast path that is supposed to be cheap.

With `tt_run`/`ttrun.py`, `TT_METAL_HOME` and `TT_METAL_CACHE` are forwarded to ranks, and
the JIT server variables can be passed the same way:

```bash
tt_run --rank-binding ranks.yaml \
  env TT_METAL_JIT_SERVER_ENABLE=1 \
      TT_METAL_JIT_SERVER_ENDPOINTS="$ENDPOINTS" \
      pytest -svv tests/...
```

### 4c. CPU farm (scale-out)

*Servers on a separate pool of machines, typically many-core, no accelerators.*

Best suited to **cache warming**: run a job that touches every kernel configuration you
care about, let a big CPU pool chew through it in parallel, and end up with hot server
caches (and hot client caches if the warming job runs where the workload will run).

```bash
# Farm side: N many-core CPU hosts, each running one server with a persistent cache root
for i in $(seq 0 15); do
  ssh cpu$i "cd /nfs/tt-metal && \
    TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:9876 \
    TT_METAL_JIT_SERVER_CACHE_ROOT=/local/jit-cache/ \
    nohup ./build/tools/jit_compile_server > /tmp/jit_server.log 2>&1 &"
done

# Warming client
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=$(seq -s, -f 'cpu%g:9876' 0 15)
python warm_all_kernels.py
```

This is the pattern most likely to hit the source-visibility problem in §5, because the
farm hosts are not the hosts that generated the kernel sources. Read §5 before deploying it.

---

## 5. Limitations

### 5a. Client and server must share the same source tree

The request carries the *recipe*, not the world. Include paths, compiler flags, the linker
script path, and the compiler path itself are all sent as **absolute paths that the server
resolves on its own filesystem**. Only the per-kernel generated files (the `.cpp`/`.h`/`.hpp`
the client generated into that kernel's build directory) travel over the wire.

Consequently the server must have, at the same absolute paths as the client:

- the same tt-metal source tree, at the same commit and with the same local modifications;
- the same SFPI/RISC-V toolchain (the `g++` path is taken from the client's request);
- any other file named by an absolute path in the flags.

How this plays out per pattern:

- **Single host (§4a):** free. The application and the server run from the same checkout.
- **Multiple hosts (§4b):** usually satisfied by **NFS** — every host mounts the same tree at
  the same path. If the tree is instead replicated per host, *you* are responsible for
  keeping the replicas byte-identical and mounted at the same path. A stale replica does
  not produce a clean error; it produces confusing compile failures, or worse, a binary
  built from different sources than you think.
- **CPU farm (§4c):** the farm hosts must mount the tree too, or use preprocess mode below.

### 5b. Application-generated kernel sources must be visible to the server

A direct consequence of 5a. When a kernel is registered by file path, the generated
`kernel_includes.hpp` contains literally `#include "<absolute path to the kernel source>"`,
and the server opens that path on its own filesystem. If your framework **generates** kernel
source into a scratch/temp directory at runtime, the server needs to be able to read that
directory.

- On a shared filesystem (NFS scratch), this works with no extra configuration.
- Without a shared filesystem — the typical CPU-farm case — the server cannot see the
  generated source and the compile fails.

The escape hatch is **`TT_METAL_JIT_PREPROCESS=1`** on the client:

```bash
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=cpu0:9876,cpu1:9876
export TT_METAL_JIT_PREPROCESS=1        # ship self-contained translation units
python warm_all_kernels.py
```

In this mode the client runs the preprocessor (`-E`) itself with the exact compile flags and
ships the resulting self-contained `.ii` — headers and defines inlined. The server then needs
nothing but the toolchain: no include tree, no source files, no shared filesystem. The
source tree is not even read or sent.

**Do not enable it otherwise.** The preprocessing runs on the *client*, which is the CPU you
were trying to free, and the payloads get substantially larger. It also disables server-side
object reuse for those units: a `.ii` has no real include tree, so no valid object dependency
hash can be computed, and the server conservatively recompiles next time. (Client-side reuse
still works — it rides on a `.fulldephash` sidecar written next to the ELF after a successful
compile.) Reserve preprocess mode for the case it exists to solve: a farm that cannot see
your sources.

### 5c. Other limitations and gotchas

- **Precompiled kernels are not supported in remote mode.** A kernel with a
  `PrecompiledKernelConfig` is not handled on the remote path.
- **Recipe changes are not fully fingerprinted server-side.** Local builds write a
  `.build_state` hash over the full compile/link recipe and force a rebuild on mismatch. The
  server does not; it partitions its cache by `build_key` + kernel path and validates via
  dependency hashes. In practice the kernel path includes the compile hash, so this is
  almost always fine — but reusing the same `build_key` and kernel path while changing only
  flags could theoretically serve a stale object. If you are changing build flags and see
  something impossible, wipe `TT_METAL_JIT_SERVER_CACHE_ROOT`.
- **Firmware is uploaded once per (endpoint, build_key) per client process** and must remain
  in the server's cache root. If you clear the server cache while clients are running,
  compiles fail until those clients restart. Don't point the cache root at something that
  gets cleaned under you. See §3 for what the cache root is and why you would relocate it.
- **`TT_METAL_JIT_SERVER_ENDPOINT` means different things on each side:** the *bind* address
  for the server, a single-endpoint *target* for the client. Don't export one value into both
  roles by accident; prefer `TT_METAL_JIT_SERVER_ENDPOINTS` on clients and keep
  `TT_METAL_JIT_SERVER_ENDPOINT` for the server's bind.
- **A server failure is fatal to the client.** There is no automatic fallback to local
  compilation — an unreachable endpoint or a failed compile throws. For unattended runs,
  keep `TT_METAL_JIT_SERVER_ENABLE` easy to turn off.
- **`TT_METAL_FORCE_JIT_COMPILE=1` bypasses all reuse**, including the client-side gate that
  skips the remote round-trip. Every kernel gets sent. Useful for benchmarking the server,
  wasteful otherwise.

---

## 6. Environment variable reference

### Client (the tt-metal application)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TT_METAL_JIT_SERVER_ENABLE` | unset (off) | `1` enables the remote compile path. Any other value is off. |
| `TT_METAL_JIT_SERVER_ENDPOINTS` | unset | Comma-separated `host:port` list. Must be identical and identically ordered across all clients that should share dedup. |
| `TT_METAL_JIT_SERVER_ENDPOINT` | unset | Single-endpoint fallback, used only when `..._ENDPOINTS` is unset or empty. |
| `TT_METAL_JIT_PREPROCESS` | unset (off) | Set (any value) to preprocess on the client and ship self-contained `.ii`. See §5b — use only when the server cannot see your sources. |
| `TT_METAL_CACHE` | `$HOME/.cache/tt-metal-cache/`, else `/tmp/tt-metal-cache/` | Local kernel cache. Must be unique per concurrent process. Checked before any remote request. |
| `TT_METAL_FORCE_JIT_COMPILE` | unset | Set to bypass all ELF reuse, local and remote. |

### Server (`jit_compile_server`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TT_METAL_JIT_SERVER_ENDPOINT` | `localhost:9876` | Bind address. Use `0.0.0.0:9876` or a specific interface to accept remote clients — see the disclaimer. |
| `TT_METAL_JIT_SERVER_CACHE_ROOT` | `/tmp/tt-metal-cache/` | Directory for compiled objects, ELFs, and firmware. Grows without bound; relocate off `/tmp` when disk is tight. See §3. |

---

## 7. Quick troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `TT_METAL_JIT_SERVER_ENABLE is set but no compile-server endpoints are configured` | Neither `..._ENDPOINTS` nor `..._ENDPOINT` set on the client. |
| `Failed to connect to remote JIT compile server at <ep>` | Server not running, wrong port, firewall, or server bound to `localhost` while the client is remote. |
| Compile fails with "No such file or directory" on a header or kernel source | §5a/§5b: the server does not see the same tree, or the generated source isn't on a shared filesystem. Fix the mount, or use `TT_METAL_JIT_PREPROCESS=1`. |
| `Firmware artifact not found for build_key ...` | Server cache root was cleared after the client uploaded firmware. Restart the client (or don't clear the cache mid-run). |
| `Absolute <field> is not allowed` / `must not contain '..'` | The server rejected a client-supplied path component. Expected for a malformed or mismatched client; report it if it happens in a normal run. |
| No speedup at all | Kernel sets likely don't overlap — see §2. |
| Slower than local | Single process with unique kernels, or `TT_METAL_JIT_PREPROCESS` left on unnecessarily. |
