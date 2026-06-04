# JIT compile-server environment fingerprint (Tier 1)

**Status:** draft for review. No build inputs touched yet — this document is the
reviewable proposal; apply on approval.

**Goal:** make client↔server tree divergence **fail loud and early** instead of
silently returning stale/wrong ELFs (the §2.4 bug, which also poisons the *local*
on-disk cache). This does **not** decouple the states — client and server still need
matching content. It converts the failure mode from *silent-wrong* to *one-line abort*.

This is the verification half of the problem (prove the bytes match). It is independent
of, and complementary to, whatever transport you use to *get* the bytes there (rsync,
shared FS, content upload).

---

## 1. What we are protecting

The server runs the compiler against **its own filesystem** at client-supplied absolute
paths (`jit_compile_server.cpp:195,203`). Three things must match for the remote ELF to
equal what the client would have built locally:

| Surface | Risk | Why current hashes miss it |
|---|---|---|
| **Toolchain** (`gpp`, sfpi) | version drift ⇒ different codegen | nothing checks it cross-machine |
| **Kernel sources** (`srcs`) | uncommitted/edited kernel ⇒ stale compile | `build_state_hash_` hashes the *path list*, not content; server `.dephash` is over its *own* (stale) files |
| **Header closure** (`-I` tree) | branch/HAL drift ⇒ wrong defines/layout | same as above |

No value in the protocol today is a *content* fingerprint (`rpc.capnp` has none).

---

## 2. Design: two layers, strict abort

### Layer A — session handshake (the must-have)
One new RPC, `verifyEnvironment`, gated **once per endpoint** per client process (exactly
like the existing firmware gate `s_fw_gate_`). Sent before the first compile to each
endpoint. Verifies:

- **Toolchain identity** — content hash of the resolved compiler binary + its `--version`
  string. Cheap (one file), critical, **git-independent**.
- **Tree fingerprint** — `git HEAD` + a hash over `git status --porcelain` and the content
  of every dirty/untracked file under the build-relevant roots
  (`tt_metal/`, `ttnn/`, kernel dirs). Cheap because clean tracked files are pinned by the
  commit hash; only the dirty set is hashed. Catches branch mismatch, HAL drift, **and the
  uncommitted-edit case** (the dirty kernel shows up in the porcelain digest).

If either side is **not** a git checkout, the tree fingerprint degrades to a loud warning
(toolchain check still hard-fails). See §6.

### Layer B — per-target source hashes (precision hardening)
Add a content hash per entry in `srcs` to `TargetRecipe`, populated client-side, verified
server-side **before** compiling that source. This is **git-independent** and catches the
exact edited-kernel case even when Layer A's git path is unavailable, and even for sources
outside the digested roots. No chicken-and-egg: `srcs` are explicitly listed, so the server
can hash them directly without needing the post-compile `.d` closure.

**Recommendation:** ship both. Together they give git-independent protection on the
high-risk surface (toolchain + actual sources) and git-accelerated coverage on the broad
surface (headers). Layer A alone is the minimum that closes §2.4 when both sides are git
checkouts (our current reality).

### Failure policy
On any mismatch the client **aborts the run** with an actionable message — it does **not**
fall back to local compile and does **not** warn-and-continue (silent-wrong is the whole
thing we're killing). Escape hatch: `TT_METAL_JIT_SERVER_SKIP_VERIFY=1` for users who
knowingly accept the risk.

---

## 3. Protocol changes — `tt_metal/impl/jit_server/rpc.capnp`

```capnp
struct EnvFingerprint {
    rootPath       @0 :Text;    # client TT_METAL_HOME (server checks its own tree at this path)
    gppRaw         @1 :Text;    # the gpp invocation string (may be "ccache <path>")
    toolchainHash  @2 :UInt64;  # hash(resolved compiler binary bytes) ^ hash(--version)
    treeDigest     @3 :UInt64;  # git-based tree fingerprint (0 if git unavailable on client)
    gitDescribe    @4 :Text;    # human-readable: "<sha>[-dirty(<n> files)]" for diagnostics
    treeDigestValid@5 :Bool;    # false if client could not compute treeDigest (no git)
}

struct VerifyEnvironmentResponse {
    match              @0 :Bool;
    errorMessage       @1 :Text;
    serverGitDescribe  @2 :Text;    # server's view, for the diff message
    serverToolchainHash@3 :UInt64;
    serverTreeDigest   @4 :UInt64;
    serverTreeValid    @5 :Bool;
}

# Layer B: add to TargetRecipe
struct TargetRecipe {
    # ... existing fields ...
    srcHashes @13 :List(UInt64);  # content hash per entry in `srcs`, same order
}

interface JitCompile {
    compile         @0 (request :CompileRequest)        -> (response :CompileResponse);
    uploadFirmware  @1 (request :UploadFirmwareRequest) -> (response :UploadFirmwareResponse);
    verifyEnvironment @2 (request :EnvFingerprint)      -> (response :VerifyEnvironmentResponse);  # NEW
}
```

Mirror these as plain DTOs in `types.hpp` (`EnvFingerprint`, `VerifyEnvironmentResponse`)
and add `fill_*`/`read_*` helpers in `jit_compile_rpc_client.cpp` alongside the existing
firmware ones.

---

## 4. Shared hashing helpers — new `tt_metal/impl/jit_server/fingerprint.{hpp,cpp}`

Used by **both** client and server so the two sides compute identically.

```cpp
namespace tt::tt_metal::jit_server {

// hash(resolved compiler binary bytes) ^ hash(`gpp --version` stdout).
// Resolves "ccache <path>" / bare paths to the real compiler file.
uint64_t compute_toolchain_hash(const std::string& gpp_raw);

// git HEAD + hash(porcelain + content of each dirty/untracked file under `roots`).
// Returns {digest, valid}; valid=false when `root` is not a git work tree.
struct TreeFingerprint { uint64_t digest; bool valid; std::string describe; };
TreeFingerprint compute_tree_fingerprint(
    const std::string& root, const std::vector<std::string>& rel_roots);

// content hash of a single file (used for srcHashes). Wraps utils::read_file_bytes.
uint64_t compute_file_hash(const std::string& path);

}  // namespace
```

Implementation notes:
- Reuse `tt::StableHasher` (`common/stable_hash.hpp`: `update(string_view)`,
  `update(const void*, size)`, `update(uint64_t)`, `digest()`), the same hasher
  `build_state_hash_` already uses (`build.cpp:461`) — so values are stable and comparable.
- `compute_file_hash`: `auto b = tt::jit_build::utils::read_file_bytes(p); h.update(b.data(), b.size());`
- `compute_tree_fingerprint`: shell `git -C <root> rev-parse HEAD` and
  `git -C <root> status --porcelain -- <rel_roots>` (via popen or the existing
  `utils::exec_command`); fold the porcelain text into the hasher, then for each `M`/`??`
  path fold in `compute_file_hash(path)`. Sorting the porcelain lines makes it order-stable.
- `rel_roots` default: `{"tt_metal", "ttnn", "runtime/sfpi"}` — tune to the compile surface.

---

## 5. Client changes

### 5a. Compute the fingerprint once per process (cached)
The client tree + toolchain are fixed for the process. Compute lazily, guard with a mutex,
cache by `(rootPath, gppRaw)`.

### 5b. Gate verification per endpoint — `remote_compile_coordinator.{hpp,cpp}`
Add a gate mirroring the firmware gate, keyed by **endpoint** (the client tree is global;
the server tree is per endpoint):

```cpp
// remote_compile_coordinator.hpp  (new statics + method)
static std::mutex s_env_gate_mutex_;
static std::unordered_map<std::string, std::shared_future<void>> s_env_gate_;  // key: endpoint
void ensure_environment_verified(std::size_t endpoint_index);
```

```cpp
// remote_compile_coordinator.cpp
void RemoteCompileCoordinator::ensure_environment_verified(std::size_t ep) {
    if (std::getenv("TT_METAL_JIT_SERVER_SKIP_VERIFY")) return;
    const std::string& endpoint = endpoints_[ep];
    // ... s_env_gate_ once-per-endpoint dance, identical shape to ensure_firmware_uploaded ...
    const auto& dev_env = BuildEnvManager::get_instance().get_device_build_env(device_build_id_);
    jit_server::EnvFingerprint fp = jit_server::client_fingerprint(   // cached compute (§5a)
        dev_env.build_env.get_root_path(), dev_env.build_env.get_gpp());
    ensure_session(ep);
    auto resp = sessions_[ep]->verify_environment(fp);
    TT_FATAL(resp.match,
        "Remote JIT server environment mismatch at {}.\n"
        "  client: {}\n  server: {}\n"
        "  toolchain {} (client) vs {} (server)\n"
        "  -> re-sync the server checkout (rsync/branch) before running, "
        "or set TT_METAL_JIT_SERVER_SKIP_VERIFY=1 to bypass.\n{}",
        endpoint, fp.gitDescribe, resp.serverGitDescribe,
        fp.toolchainHash, resp.serverToolchainHash, resp.errorMessage);
}
```

Call it in `submit()` immediately after `ensure_session(ep_idx)`, before
`ensure_firmware_uploaded(ep_idx)` (`remote_compile_coordinator.cpp:68-69`):

```cpp
ensure_session(ep_idx);
ensure_environment_verified(ep_idx);   // NEW — fail before any compile work
ensure_firmware_uploaded(ep_idx);
```

Add `JitCompileRpcSession::verify_environment(const EnvFingerprint&)` to
`jit_compile_rpc_client.{hpp,cpp}`, mirroring `upload_firmware` (1:1 with the existing
`send().wait(getWaitScope())` shape).

### 5c. Layer B — populate `srcHashes` (`build.cpp` `export_target_recipe`)
Right after the `srcs` loop (`build.cpp:823`):

```cpp
for (const auto& src : srcs_) {
    target.srcs.push_back(src);
    target.src_hashes.push_back(jit_server::compute_file_hash(src));  // NEW
}
```
and serialize it in `fill_target_recipe` (`jit_compile_rpc_client.cpp:64`).

---

## 6. Server changes

### 6a. `verifyEnvironment` handler — `jit_compile_service.{hpp,cpp}`
Add a `VerifyEnvironmentCallback` alongside the existing two callbacks; deserialize the
`EnvFingerprint`, run the callback on the thread pool (same pattern as `compile`), serialize
the `VerifyEnvironmentResponse`.

### 6b. Callback — `jit_compile_server.cpp`
Recompute the server's own values over its tree at the client-supplied `rootPath`/`gppRaw`,
compare, return the diff:

```cpp
VerifyEnvironmentResponse verify_environment_callback(const EnvFingerprint& c) {
    VerifyEnvironmentResponse r;
    uint64_t srv_tool = compute_toolchain_hash(c.gpp_raw);
    auto srv_tree = compute_tree_fingerprint(c.root_path, kDefaultRelRoots);
    r.server_toolchain_hash = srv_tool;
    r.server_tree_digest = srv_tree.digest;
    r.server_tree_valid = srv_tree.valid;
    r.server_git_describe = srv_tree.describe;

    bool tool_ok = (srv_tool == c.toolchain_hash);
    bool tree_ok = (!c.tree_digest_valid || !srv_tree.valid)   // degrade if either lacks git
                       ? true
                       : (srv_tree.digest == c.tree_digest);
    if (!tool_ok)        r.error_message += "toolchain hash mismatch (sfpi version differs); ";
    if (c.tree_digest_valid && srv_tree.valid && !tree_ok)
                         r.error_message += "source tree fingerprint mismatch; ";
    if ((!c.tree_digest_valid || !srv_tree.valid))
                         r.error_message += "WARNING: tree verification skipped (no git on one side); ";
    r.match = tool_ok && tree_ok;   // toolchain is always hard; tree only when both have git
    return r;
}
```
Register it in `main()` next to `compile_callback`/`upload_firmware_callback`.

### 6c. Layer B — verify `srcHashes` pre-compile (`compile_one`, `jit_compile_server.cpp:185`)
Before invoking the compiler on `target.srcs[src_index]`:

```cpp
if (!target.src_hashes.empty()) {
    uint64_t actual = compute_file_hash(target.srcs[src_index]);
    if (actual != target.src_hashes[src_index]) {
        throw std::runtime_error(fmt::format(
            "Source content mismatch for {} (client {} != server {}). "
            "Server checkout is stale; re-sync before compiling.",
            target.srcs[src_index], target.src_hashes[src_index], actual));
    }
}
```
This surfaces as a `compile FAIL` the coordinator already turns into a `TT_FATAL`
(`remote_compile_coordinator.cpp:108`).

---

## 7. Edge cases

- **Server not a git checkout** (rsync without `.git`): Layer A tree check degrades to a
  warning; toolchain check + Layer B src hashes still hard-fail. Mitigation: rsync `.git`,
  or rely on Layer B. Document in `REMOTE_JIT_SERVER_SETUP.md`.
- **`ccache` in `gpp`**: `compute_toolchain_hash` must resolve `ccache <path>` to the real
  compiler binary before hashing (and not require ccache to hash). The *existing* ccache
  PATH requirement on the server is unchanged.
- **Equal clean HEAD ⇒ tracked files identical** by git's content model, so the tree digest
  only needs to enumerate the dirty/untracked set — keeps it cheap.
- **Gitignored generated headers** under the roots are caught as untracked (`??`) by
  porcelain, so they're folded into the digest.
- **Concurrency**: verification is gated once per endpoint via `s_env_gate_`; later threads
  block on the shared_future, same as firmware.
- **Cost**: one extra RPC per endpoint per process (negligible against thousands of
  compiles); per-request Layer B adds one `read_file_bytes`+hash per source (a few files).

---

## 8. Test plan

1. **Unit** (`tests/tt_metal/tt_metal/jit_build/`): `compute_file_hash` stability;
   `compute_tree_fingerprint` changes iff a tracked/dirty file changes; toolchain hash
   changes iff the binary/version changes.
2. **Matching trees** (mock device, `test_compile_stress.cpp` style): handshake `match=true`,
   compiles succeed — no regression.
3. **Divergence — the §2.4 reproduction**: edit a kernel on the client only; assert the run
   aborts at `ensure_environment_verified` (Layer A) **and** at `compile_one` if Layer A is
   bypassed (Layer B), with messages naming the file/commit. This is the test that proves we
   killed the silent-wrong failure.
4. **Toolchain mismatch**: point server at a different sfpi; assert hard abort.
5. **No-git fallback**: strip `.git` on server; assert warning + Layer B still catches an
   edited source.
6. **Bypass**: `TT_METAL_JIT_SERVER_SKIP_VERIFY=1` restores today's behavior.

---

## 9. Files touched

| File | Change |
|---|---|
| `tt_metal/impl/jit_server/rpc.capnp` | `EnvFingerprint`, `VerifyEnvironmentResponse`, `verifyEnvironment` RPC, `srcHashes` |
| `tt_metal/impl/jit_server/types.hpp` | matching DTOs |
| `tt_metal/impl/jit_server/fingerprint.{hpp,cpp}` | **new** shared hashing helpers |
| `tt_metal/impl/jit_server/jit_compile_rpc_client.{hpp,cpp}` | `verify_environment` session call + fill/read; serialize `srcHashes` |
| `tt_metal/impl/jit_server/jit_compile_service.{hpp,cpp}` | `verifyEnvironment` handler + callback type |
| `tt_metal/impl/jit_server/remote_compile_coordinator.{hpp,cpp}` | `s_env_gate_`, `ensure_environment_verified`, call in `submit()` |
| `tt_metal/tools/jit_compile_server/jit_compile_server.cpp` | `verify_environment_callback`; `srcHashes` check in `compile_one`; register in `main()` |
| `tt_metal/jit_build/build.cpp` | populate `src_hashes` in `export_target_recipe` |
| `tt_metal/jit_build/types.hpp` | add `src_hashes` to `TargetRecipe` |
| `tests/tt_metal/tt_metal/jit_build/` | tests per §8 |
| `REMOTE_JIT_SERVER_SETUP.md` | document the check + `.git`/skip-verify notes |

**Effort:** the meat is `fingerprint.cpp` (git/tree digest) + the capnp/DTO plumbing; the
coordinator/service hooks are ~1:1 copies of the existing firmware path. Layer A is the
minimum to close §2.4 today; Layer B is cheap and worth shipping with it.
