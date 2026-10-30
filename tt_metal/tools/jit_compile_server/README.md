<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
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
> - **Remote code execution is the feature.** The client specifies the compiler and flags;
>   the server runs them as its own user. `posix_spawn` with an explicit argv (never
>   `system()`) and rejection of absolute/`..` path components harden the edges; they do not
>   change that anyone who can reach the endpoint can compile as the server.
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

1. The client generates the kernel's JIT files and sends a compile request (compiler path,
   flags, defines, include string, source list, and the contents of those JIT-generated
   files) to a server and waits.
2. The server compiles and links into its own on-disk cache, then returns the ELF bytes.
3. The client writes the ELF into its local `TT_METAL_CACHE` and loads it.

A later request for the same kernel still goes to the server; the *server's* cache is what
avoids compiling it again. The local copy is for loading, not for skipping the RPC.
(Preprocess mode is different; see §5b.)

Two layers of deduplication make this worthwhile:

- **Per-process (client side):** identical kernel hashes submitted within one process
  collapse into a single request.
- **Global, in-flight (server side):** concurrent requests for the same
  `build_key` + kernel hash — *even from different processes on different hosts* — collapse
  into a single compile. Late arrivals wait on the first one's result rather than
  recompiling. This is the in-flight layer a local on-disk cache cannot give you.

Plus the server's own on-disk cache, which serves repeat requests after the fact.

**Kernel-to-server routing is deterministic:** endpoint index = `kernel_hash % num_endpoints`.
Every client that is given the **same endpoint list in the same order** routes a given
kernel to the same server. That is what makes cross-process and cross-host dedup and cache
reuse work. If clients get different lists, or the same hosts in a different order, you
still get correct results but you fragment the dedup and the cache.

---

## 2. When to expect a performance gain

Two cases actually pay off.

**Kernel overlap.** In-flight dedup and the server's on-disk cache compile a kernel once
instead of once per process. That needs the same kernel hash on more than one process, or
more than once over time. This is the usual win for §4a and §4b: the compile still runs on
the workload hosts' CPUs, but overlapping kernels are not compiled N times. A shared
`TT_METAL_CACHE` has no in-flight dedup across processes, so near-simultaneous local
misses can all compile. The server eliminates that. A host-local cache also cannot share
across hosts.

**Scale-out.** A farm with *more* compile CPU than the application hosts, given a large
unique kernel list. Kernels hash-shard across endpoints
(`kernel_hash % num_endpoints`), so the farm compiles them in parallel in a way a local
thread pool cannot. That is the usual reason to put compile on a separate set of machines
(§4c, typically cache warming). Relocating onto a similar-size box without adding capacity
is possible but rarely the win: workload hosts already have CPU, and you still pay RPC
and ELF transfer.

| Situation | Expect |
| --- | --- |
| N processes compile largely the same kernel set, concurrently | Best case — ideally 1 compile instead of N |
| Same kernels requested later by a different process or host | Server cache hit; near-free |
| Large unique kernel list on a scaled-out server pool | Wins if the farm has more CPU than the application hosts |
| Unique kernels, same amount of compile CPU as compiling locally | **Net loss** — RPC and transfer for work local compile would have done |

Rule of thumb: deploy for overlap, or for a large unique kernel list on a bigger CPU pool.

---

## 3. Build and start the server

```bash
# Build (from the tt-metal root)
./build_metal.sh --enable-ccache          # or: cmake --build build --target jit_compile_server

# Run it. Defaults: bind localhost:9876, cache root /tmp/tt-metal-cache/
./build/tools/jit_compile_server
```

The server **never opens a device**. It does need the tt-metal source tree and the SFPI
toolchain at the absolute paths the *client* names in its requests (see §5). The server
does not read `TT_METAL_HOME`; the tree just has to be on disk at those paths.

A typical invocation with a relocated cache root and a widened bind:

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

`TT_METAL_JIT_SERVER_CACHE_ROOT` is where the server stores compiled objects, ELFs, and
uploaded firmware, partitioned by `build_key`. It is **not** a client's `TT_METAL_CACHE`:
clients still write their own ELF copy after the server returns it. The server cache lets
a later request for the same kernel skip recompilation.

The default is `/tmp/tt-metal-cache/`. Relocate it when `/tmp` is too small or ephemeral —
objects and ELFs accumulate for every unique kernel, with no garbage collection, and a few
large workloads can fill a tmpfs. Use a node-local disk with headroom.

Keep it distinct from any client `TT_METAL_CACHE`; the two layouts differ. There is a real
collision to avoid when the server shares a host with its clients (§4a, §4b): a client
whose `TT_METAL_CACHE` is unset falls back to `$HOME/.cache/tt-metal-cache/`, but if
`$HOME` is also unset or missing — common in containers — it falls back to
`/tmp/tt-metal-cache/`, which is exactly the server's default. Set both variables
explicitly in that case.

### Point a client at it

```bash
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876
pytest tests/... # or your application
```

That's all on the client. If `TT_METAL_JIT_SERVER_ENABLE=1` is set with no endpoint, the
client fails fast instead of falling back to local compilation.

---

## 4. Deployment patterns

### 4a. Single host, multiple processes

*Several ranks, pytest workers, or model instances on one machine.*

Concurrent processes on one host **can** share a `TT_METAL_CACHE` directory — some tests
already do. The local cache is lock-free, though, so it has no in-flight dedup across
processes: if several ranks miss at about the same time, they can all compile the same
kernel. The compile server eliminates that duplication.

```bash
# Terminal 1 — server
./build/tools/jit_compile_server

# Terminals 2..N — workloads; a shared cache dir is fine
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876
export TT_METAL_CACHE=/tmp/tt-metal-cache-shared
python my_workload.py --rank 0 &
python my_workload.py --rank 1 &
wait
```

The server runs from the same checkout as the application, so §5's shared-tree requirement
is free.

### 4b. Multiple hosts, servers co-located with the workload

*Multi-host run; one server per host; no extra machines.*

The compile still uses these hosts' CPUs — you are not adding capacity. The reason to do
it is **cross-host overlap**: unlike a per-host local-disk `TT_METAL_CACHE`, a kernel
needed on several hosts is compiled once (whichever server the hash routes to) and the
others take the cache or in-flight result. Local disk cannot do that.

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
- **Prefer a node-local `TT_METAL_CACHE` if `$HOME` is on NFS.** The local-hit fast path is
  supposed to be cheap; a shared-filesystem cache adds latency. Ranks on the same host may
  share that directory.

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

This is the pattern most likely to hit the source-visibility problem in §5: the farm has
the tt-metal tree, but not necessarily the kernel sources the application generated at
runtime.

---

## 5. Limitations

### 5a. Client and server must share the same source tree

The request carries the *recipe*, not the world. Include paths, compiler flags, the linker
script path, and the compiler path itself are all sent as **absolute paths that the server
resolves on its own filesystem**. What travels over the wire as file *contents* is only the
JIT-generated files in that kernel's build directory (`.cpp`/`.h`/`.hpp` wrappers). The
original kernel source is not sent; if it is referenced by absolute path, the server must
be able to open that path itself (see §5b).

Consequently the server must have, at the same absolute paths as the client:

- the same tt-metal source tree, at the same commit and with the same local modifications;
- the same SFPI/RISC-V toolchain (the `g++` path is taken from the client's request);
- any other file named by an absolute path in the flags.

How this plays out per pattern:

- **Single host (§4a):** free. The application and the server run from the same checkout.
- **Multiple hosts (§4b):** usually **NFS** at the same path on every host. If the tree is
  replicated instead, keep replicas byte-identical at that path. A stale replica does not
  fail cleanly; it can build from different sources than you think.
- **CPU farm (§4c):** the farm hosts must mount the tree too, or use preprocess mode below.

### 5b. Application-generated kernel sources must be visible to the server

A direct consequence of 5a. When a kernel is registered by file path, the JIT-generated
`kernel_includes.hpp` contains literally `#include "<absolute path to the kernel source>"`,
and the server opens that path on its own filesystem. If your framework **generates** that
kernel source into a scratch/temp directory at runtime, the server needs to be able to read
that directory.

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

In this mode the client runs `-E` with the exact compile flags and ships a self-contained
`.ii` (headers and defines inlined). The server compiles that without an include tree or
kernel sources. Link still needs the linker script and extra link objects at the absolute
paths in the recipe, plus `g++` at the requested path; firmware is uploaded separately.

**Do not enable it otherwise.** Preprocessing runs on the *client*, payloads get larger, and
`.ii` units have no include tree so the server cannot write an object dephash and
conservatively recompiles next time. Use this only when the farm cannot see your sources.

After a successful preprocess compile, the client writes `.fulldephash` and `.build_state`
next to the ELF. A later run that finds those still valid **skips the RPC** and loads the
local ELF.

### 5c. Other limitations and gotchas

- **Precompiled kernels are not supported in remote mode.** A kernel with a
  `PrecompiledKernelConfig` is not handled on the remote path.
- **Recipe changes are not fully fingerprinted server-side.** Local builds store a
  `.build_state` hash of the compile/link recipe. The server only partitions by
  `build_key` + kernel path and checks dependency hashes. The kernel path usually includes
  the compile hash, so this is almost always fine — but changing only flags under the same
  `build_key` and kernel path could theoretically reuse a stale object. Wipe
  `TT_METAL_JIT_SERVER_CACHE_ROOT` if that happens.
- **Firmware is uploaded once per (endpoint, build_key) per client process** and must stay
  in the cache root. Clearing the cache while clients are running fails compiles until
  those clients restart.
- **`TT_METAL_JIT_SERVER_ENDPOINT` means different things on each side:** bind address on
  the server, single-endpoint target on the client. Prefer `TT_METAL_JIT_SERVER_ENDPOINTS`
  on clients.
- **A server failure is fatal to the client.** There is no automatic fallback to local
  compilation — an unreachable endpoint or a failed compile throws. For unattended runs,
  keep `TT_METAL_JIT_SERVER_ENABLE` easy to turn off.
- **`TT_METAL_FORCE_JIT_COMPILE=1` bypasses local JIT reuse** (and the preprocess RPC skip
  in §5b). Useful for benchmarking the server, wasteful otherwise.

---

## 6. Environment variable reference

### Client (the tt-metal application)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TT_METAL_JIT_SERVER_ENABLE` | unset (off) | `1` enables the remote compile path. Any other value is off. |
| `TT_METAL_JIT_SERVER_ENDPOINTS` | unset | Comma-separated `host:port` list. Must be identical and identically ordered across all clients that should share dedup. |
| `TT_METAL_JIT_SERVER_ENDPOINT` | unset | Single-endpoint fallback, used only when `..._ENDPOINTS` is unset or empty. |
| `TT_METAL_JIT_PREPROCESS` | unset (off) | Set (any value) to preprocess on the client and ship self-contained `.ii`. See §5b — use only when the server cannot see your sources. |
| `TT_METAL_CACHE` | `$HOME/.cache/tt-metal-cache/`; falls back to `/tmp/tt-metal-cache/` when `$HOME` is unset or does not exist | Where the client writes returned ELFs. Concurrent processes on one host may share it. The fallback path collides with the server's default cache root — see §3. |
| `TT_METAL_FORCE_JIT_COMPILE` | unset | Bypass local JIT reuse, and the preprocess RPC skip in §5b. |

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
| No speedup at all | No kernel overlap, and no extra compile CPU — see §2. |
| Slower than local | Unique kernels on similar CPU, or `TT_METAL_JIT_PREPROCESS` left on unnecessarily. |
