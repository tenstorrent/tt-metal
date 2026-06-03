# Briefing: Stand Up a Remote JIT Compile Server (server-side agent)

**You are the build server.** Another machine (the *client*) runs tt-metal tests and
wants to offload its just-in-time **kernel compilation** to you over the network. Your
job: run the `jit_compile_server` daemon, make sure your environment is a faithful twin
of the client's build environment, confirm the client can reach you, and report your
endpoint back. You do **not** run any tests and you do **not** need Tenstorrent hardware.

This document is self-contained. Read §1–§3 to understand *why*, then execute §4 step by
step. §6 is your troubleshooting map. Everything here is grounded in the actual code:
`tt_metal/tools/jit_compile_server/jit_compile_server.cpp` (the daemon),
`tt_metal/impl/jit_server/` (RPC + client coordinator), and
`tt_metal/jit_build/build.cpp` (how compile recipes are formed).

---

## 1. What you are doing, in one paragraph

The client compiles ~5 RISC-V kernels per program, and that compilation is the dominant
cost of a cold test run. When `TT_METAL_JIT_SERVER_ENABLE=1`, the client's compile path
(`ProgramImpl::compile`, `tt_metal/impl/program/program.cpp:1943`) stops compiling
locally and instead, for each *unique* kernel, sends you a **compile recipe** over a
Cap'n Proto RPC. You run the compiler, and ship the resulting `.elf` bytes back. The
client writes those binaries into its own on-disk cache, so its subsequent real test run
is a warm cache hit. You are a stateless-ish compile worker; the client is the brain.

```
CLIENT (has the device, runs tests)                 YOU (the server, CPU-only)
───────────────────────────────────                 ──────────────────────────
for each unique kernel:
   build recipe (gpp, flags, srcs, -I, defines)
   ── uploadFirmware(build_key, [fw.elf]) ─────────►  store firmware in your cache root
   ── compile(CompileRequest) ─────────────────────►  run gpp via posix_spawn
                                                       link against uploaded firmware
   ◄── CompileResponse{ success, elfBlobs[] } ──────  read the .elf back, return bytes
   write each .elf into the client's local cache
```

---

## 2. The one thing that makes or breaks this: **environment parity**

The client's recipe references the compiler, the include trees, the static kernel
sources, and the linker script **by absolute path, rooted at the client's repo
directory** (`build.cpp` builds every `src` as `env_.root_ + src`, every `-I` as
`root_ + "..."`, and `gpp` as `root_ + "runtime/sfpi/compiler/bin/riscv-tt-elf-g++"`).
The server runs those paths *as-is*. Therefore:

> **You must have an identical tt-metal checkout, built, at the *same absolute path* as
> the client, using the *same sfpi compiler version*.**

Three reasons, all hard requirements:

1. **Absolute source/include/linker/compiler paths must resolve on your filesystem.** If
   the client's root is `/abc/tt-metal` and yours is `/xyz/tt-metal`, every `-I/abc/...`
   and every `/abc/.../trisck.cc` the client sends will be "file not found" on your box.
2. **The server binary's `RUNPATH` is baked to absolute `build_Release/...` paths.** The
   `jit_compile_server` executable finds `libtt_metal.so` only at the absolute path it
   was built at. Wrong path ⇒ it won't even start (unless you override `LD_LIBRARY_PATH`).
3. **Binary identity of the output ELF.** The client computes a `build_key` =
   hash(arch, cflags, lflags, defines, **sfpi `g++ --version` string**) — see
   `build.cpp:312-339`. The ELF you produce must be byte-equivalent to what the client
   would have built locally, or you'll silently poison its cache. Same sfpi version ⇒ safe.

**The simplest way to satisfy all three at once: replicate the client's repo (content +
build + sfpi) to the identical absolute path.** `rsync` does this in one shot. If you
genuinely cannot create that path on your box, a symlink/bind-mount from the client's
absolute path to wherever you keep the repo is an acceptable substitute (all that matters
is that the paths *resolve*).

What you do **not** need:
- **No Tenstorrent device / hardware.** The daemon never opens a device (it has no
  `CreateDevice`/`MetalContext` dependency — that's by design, see the comment at
  `jit_compile_server.cpp:42`). Any CPU box works; more cores = more parallel compiles.
- **No firmware build.** The client uploads its prebuilt "weakened" firmware ELF to you
  (the `uploadFirmware` RPC); you just store and link against it.
- **No Python env / pytest.** You only run the compiler.

---

## 3. The handshake you need from the client (ask for this first)

Before you can replicate anything, get these **authoritative values from the client
machine** (the repo root is environment-driven, so don't guess it). Have the human run
this **on the client box, inside the exact shell/env the tests run in**, and paste you
the output:

```bash
# ---- run on the CLIENT, in the test environment ----
cd <the repo the client tests from>
source python_env/bin/activate 2>/dev/null
echo "ROOT=${TT_METAL_HOME:-$(pwd)}"                 # the absolute root embedded in every path
echo "COMMIT=$(git rev-parse HEAD)"
echo "CACHE=${TT_METAL_CACHE:-<default: \$HOME/.cache or repo>}"
echo "CCACHE=${TT_METAL_CCACHE_KERNEL_SUPPORT:+yes}"  # if 'yes', gpp is prefixed with 'ccache '
GPP="${TT_METAL_HOME:-$(pwd)}/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
[ -x "$GPP" ] || GPP=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++
echo "GPP=$GPP"
"$GPP" --version | head -1                            # the sfpi version folded into build_key
```

You will replicate exactly: **ROOT** (same absolute path), **COMMIT** (same source),
**sfpi version** (same toolchain), and install **ccache** if `CCACHE=yes`.

> For reference, at the time this briefing was generated the client clone was at commit
> `1f5336b1019761a62599c2f447116ea0e15bef33`, sfpi `tenstorrent/sfpi:7.48.0[595]` (gcc
> 15.1.0), `TT_METAL_CCACHE_KERNEL_SUPPORT=1`. **Trust the live handshake over these
> numbers** — the client may have moved on.

---

## 4. Setup procedure

Throughout, let `TTM_ROOT` = the client's `ROOT` from §3 (e.g.
`/localdev/mstaletovic/tt-metal`). Use the **same value** on your box.

### Step 0 — Record your own coordinates
```bash
hostname
hostname -I | awk '{print $1}'      # your IP(s) — the client must be able to reach one of these
nproc                                # your core count (your compile parallelism)
command -v rsync ccache              # tools you'll likely need
```
Note: if `hostname -I` shows a `172.17.x.x` address, you're on a Docker bridge — that IP
is usually **not** reachable from another host. See §5 for the networking decision.

### Step 1 — Get the repo onto your box at the identical absolute path

Pick **one** path. (a) is by far the most robust.

**(a) Replicate the client's built repo via rsync (recommended).** Run on the **client**,
pushing to you — this copies sources, the `build_Release/` tree (binary + `libtt_metal.so`
+ sfpi), so versions and paths match automatically:
```bash
# on the CLIENT (it has the built tree):
rsync -a --info=progress2 \
  "$TTM_ROOT/" \
  <you>@<your-host>:"$TTM_ROOT/"      # SAME absolute path on the destination
```
If you can't write `TTM_ROOT` directly on your box, rsync to a path you own and then
`ln -s /your/real/location "$TTM_ROOT"` (create parent dirs first) so the path resolves.

**(b) Shared filesystem.** If `TTM_ROOT` is on NFS/shared storage already visible from
your box at the same path, you need to copy nothing — skip to Step 2. (Verify:
`ls "$TTM_ROOT/build_Release/tools/jit_compile_server"` works on your box.)

**(c) Fresh clone + build (last resort, slow).** Only if rsync/sharing is impossible:
```bash
git clone <tt-metal remote> "$TTM_ROOT" && cd "$TTM_ROOT"
git checkout <COMMIT from §3>
git submodule update --init --recursive   # sfpi lives in runtime/sfpi
./build_metal.sh                          # heavy; produces build_Release/ + the server binary
```
This only pays off if you'll reuse the box a lot — the whole point is to avoid compiling.

### Step 2 — Verify your twin before launching
```bash
cd "$TTM_ROOT"
# Binary present and its libs resolve (RUNPATH points at this absolute path):
ldd build_Release/tools/jit_compile_server | grep -E 'tt_metal|not found'   # expect NO 'not found'
# Compiler present at the path the client will name, and version MATCHES §3:
runtime/sfpi/compiler/bin/riscv-tt-elf-g++ --version | head -1
# ccache present iff client had CCACHE=yes:
command -v ccache
# Source/commit matches:
git rev-parse HEAD     # must equal COMMIT from §3
```
If `ldd` shows `not found`, the repo is at the wrong absolute path (or you rsync'd to a
different one without the symlink) — fix Step 1. As a stopgap you can
`export LD_LIBRARY_PATH="$TTM_ROOT/build_Release/lib:$TTM_ROOT/build_Release/tt_metal"`,
but that does **not** fix the source/include path problem — the path must still match.

### Step 3 — Launch the daemon
```bash
cd "$TTM_ROOT"
export TT_METAL_JIT_SERVER_ENDPOINT=0.0.0.0:9876        # bind all interfaces (see §5 for safer options)
export TT_METAL_JIT_SERVER_CACHE_ROOT=/tmp/tt-metal-jit-server-cache/   # your scratch; needs disk space
mkdir -p "$TT_METAL_JIT_SERVER_CACHE_ROOT"
./build_Release/tools/jit_compile_server
```
On success it logs:
```
JIT compile server listening on 0.0.0.0:9876
JIT compile server cache root: /tmp/tt-metal-jit-server-cache/
```
Leave it running in the foreground (or under `tmux`/`nohup`/`systemd`). It handles
`SIGINT`/`SIGTERM` for clean shutdown. Defaults if you set nothing: binds
`localhost:9876`, cache root `/tmp/tt-metal-cache/`.

### Step 4 — Prove you're listening and reachable
```bash
ss -ltnp | grep 9876            # on YOU: confirm the listen socket exists
```
Then from the **client**, confirm the path is open end-to-end:
```bash
nc -vz <your-ip> 9876           # on the CLIENT: expect 'succeeded'/'open'
```
(Optional toolchain smoke test, on YOU — proves sfpi works without involving the client:
```bash
echo 'int main(){return 0;}' > /tmp/t.cpp
runtime/sfpi/compiler/bin/riscv-tt-elf-g++ -c /tmp/t.cpp -o /tmp/t.o && echo "sfpi OK"
```)

### Step 5 — Report back
Tell the user / client the single line they need:
```
ENDPOINT = <reachable-ip-or-host>:9876     # e.g. 10.0.0.5:9876
```
The client then runs its tests with:
```bash
TT_METAL_JIT_SERVER_ENABLE=1 \
TT_METAL_JIT_SERVER_ENDPOINTS=<your-endpoint>[,<second-endpoint>...] \
EVAL_PRECOMPILE=1 \
  scripts/run_safe_pytest.sh --run-all <test_file.py>
```
(`EVAL_PRECOMPILE=1` is the client's up-front parallel pass — it fans every unique
program at you at once instead of trickling them. Multiple servers can be listed
comma-separated; the client shards kernels across them by hash.)

---

## 5. Networking decision (do this consciously)

The transport is **plain TCP, Cap'n Proto, no auth, no encryption** (see the security
banner at `jit_compile_server.cpp:26-37`). Pick the binding that matches your network:

- **Both boxes on a trusted LAN / lab subnet, real routable IPs:** bind
  `0.0.0.0:9876`, give the client `<your-lan-ip>:9876`. Make sure no firewall blocks 9876.
- **You're in a container with a `172.17.x.x` bridge IP (not reachable cross-host):**
  bind to the **host**, not the container. Either run the server on the host, publish the
  container port (`docker run -p 9876:9876`), or use the SSH tunnel below.
- **Untrusted or unknown network:** do **not** expose the port. Use an SSH tunnel so the
  client reaches you over loopback only:
  ```bash
  # run on the CLIENT: forward client's localhost:9876 -> your localhost:9876
  ssh -N -L 9876:localhost:9876 <you>@<your-host>
  ```
  Then bind the server to `localhost:9876` (the default) and give the client
  `TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876`.

Because the client tells you which compiler to run, a client that can reach you has code
execution on your box by design. Only accept connections from machines you trust.

---

## 6. Troubleshooting map (symptom → cause → fix)

Watch your daemon's stdout. A healthy run logs `uploadFirmware build_key=...` once per
build_key, then `compile <kernel>: targets=...` and `done <kernel>: <ms>ms, elfs=N` per
kernel. Failures log `FAIL <kernel>: <msg>`.

| What you see | Cause | Fix |
|---|---|---|
| `sfpi not found at .../runtime/sfpi or /opt/tenstorrent/sfpi` (server won't start) | Repo not at `TTM_ROOT`, or submodules missing | Step 1 — replicate to the same absolute path; ensure `runtime/sfpi` exists |
| `ldd` shows `libtt_metal.so => not found` | Binary's RUNPATH (absolute) doesn't resolve | Same path as client (Step 1), or `LD_LIBRARY_PATH` stopgap (Step 2) |
| `FAIL ...: ... No such file or directory` on a `-I` path or a `.cc`/`.cpp` source | Your repo is at a **different absolute path** than the client's | Match `TTM_ROOT` exactly, or symlink the client's path to yours |
| `FAIL ...: Firmware artifact not found for build_key ... Ensure the client uploads firmware via uploadFirmware RPC` | Your cache root was wiped after upload, or client never uploaded | Don't clear `TT_METAL_JIT_SERVER_CACHE_ROOT` mid-session; have client reconnect (it re-uploads per build_key) |
| `FAIL ...: <compiler errors>` | sfpi **version mismatch** vs the client, or stale objects | Make `g++ --version` match §3 exactly; use a **fresh** `TT_METAL_JIT_SERVER_CACHE_ROOT` |
| Client error: `Failed to connect to remote JIT compile server at <ep>: ...` | Not listening, wrong IP/port, firewall, or container IP not routable | §4 Step 4 + §5 (tunnel or host binding) |
| Client error: `Response count mismatch for endpoint ...` / `ccache: command not found` in server log | Client uses `ccache` (`CCACHE=yes`) but you don't have it | `apt install ccache` (or matching pkg) on your box |
| `Absolute ... is not allowed` / `must not contain '..'` | Path-traversal guard tripped (shouldn't happen with a legit client) | Indicates a malformed/hostile request; verify the client is the real tt-metal |
| Tests run but a later **local** run gets wrong results / corrupt kernels | ELF poisoning from a toolchain mismatch | Stop the server; ensure exact sfpi parity; wipe the client's cache and the server cache; rerun |

A subtle correctness note (from the daemon's own TODO at `jit_compile_server.cpp:47`):
the server keys its cache by `build_key` + kernel path and uses `.dephash` dependency
tracking, but does **not** implement the local build's `build_state_hash_`. So if you
reuse a cache root across runs while flags change in ways not reflected in dependency
hashes, you could serve a stale `.o`. **Use a fresh `TT_METAL_JIT_SERVER_CACHE_ROOT` per
distinct build environment** and you avoid this entirely.

---

## 7. Teardown

`Ctrl-C` (or `kill -TERM`) the daemon — it shuts down cleanly. The cache under
`TT_METAL_JIT_SERVER_CACHE_ROOT` is just scratch; delete it to reclaim disk. Nothing
persists on the client side except the warmed binaries it already wrote into its own
cache, which are exactly what it would have built locally.

---

## 8. Quick reference

**Server env vars** (read by `jit_compile_server`):
| Var | Default | Meaning |
|---|---|---|
| `TT_METAL_JIT_SERVER_ENDPOINT` | `localhost:9876` | Bind address. `0.0.0.0:9876` to accept remote. |
| `TT_METAL_JIT_SERVER_CACHE_ROOT` | `/tmp/tt-metal-cache/` | Where the server writes objs/elfs/firmware. |

**Client env vars** (read by tt-metal at compile time):
| Var | Meaning |
|---|---|
| `TT_METAL_JIT_SERVER_ENABLE=1` | Turn on remote compile. |
| `TT_METAL_JIT_SERVER_ENDPOINTS` | Comma-separated `host:port` list (or `TT_METAL_JIT_SERVER_ENDPOINT` singular). |
| `EVAL_PRECOMPILE=1` | (Eval suites) up-front parallel pass that fans all programs at the server at once. |

**Key files** (if you need to read the source):
- `tt_metal/tools/jit_compile_server/jit_compile_server.cpp` — the daemon (`main`, `compile_callback`, `upload_firmware_callback`).
- `tt_metal/impl/jit_server/rpc.capnp` — the wire schema (`CompileRequest`, `TargetRecipe`, `CompileResponse`).
- `tt_metal/impl/jit_server/remote_compile_coordinator.cpp` — the client side (what gets sent, how ELFs are written back).
- `tt_metal/impl/program/program.cpp:1943` — where the client switches into remote mode.
- `tt_metal/jit_build/build.cpp:312-339` (build_key) and `:813-874` (`export_target_recipe`, the recipe you receive).
