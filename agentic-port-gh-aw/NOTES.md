# Replacing `agentic_port` with GitHub Agentic Workflows — working notes

Scratch file for the feasibility investigation. Not part of any repo.
Canvas version: `~/.cursor/projects/localdev-ebanerjee/canvases/codegen-port-gh-aw-feasibility.canvas.tsx`

**Goal:** replace the ~41k-line `agentic_port/` porting pipeline (on `tt-dm-codegen@codegen_agentic_port`)
with a gh-aw agentic workflow that runs in **tt-metal** CI, ideally **on a card**, in a couple thousand lines.

Explicitly NOT wanted: a gh-aw workflow in tt-metal that shells out to the existing pipeline.
(That is what PR #52344 does today.)

---

## 1. Verdict

**Feasible.** Every capability needed is a documented gh-aw feature, and tt-metal already runs gh-aw
in production. One unproven item: whether the agent's own bash can open `/dev/tenstorrent` from
inside the AWF sandbox. Being tested now (see §8).

Estimated line count: **41,240 → ~2,400**, of which ~1,200 is the perf/correctness gate that should
stay as code.

---

## 2. Line-count accounting (measured 2026-08-07)

`cd tt-dm-codegen && git diff --numstat main...HEAD`

| Subsystem | Today | Est. after | Note |
|---|---:|---:|---|
| Orchestrator phases, state, resume, retry | 10,667 | ~150 | Hand-rolled workflow engine. Actions *is* that engine. |
| Pipeline self-tests | 6,861 | ~200 | Mostly tests of the plumbing that stops existing. |
| Verify harness (perf + correctness gates) | 4,267 | ~1,200 | **Survives.** Domain knowledge, not orchestration. |
| Manifests + run logs | 3,223 | ~100 | Logs → Actions artifacts + PR body. |
| Knowledge / prompts / contracts | 2,847 | ~600 | Becomes the workflow markdown body. |
| Cross-arch CI dispatch | 2,163 | ~60 | → `dispatch-workflow` safe output + matrix job. |
| Driver + lib (build, exec transport) | 1,242 | ~50 | Daemon/device-lock exists to share a card; Actions owns that. |
| README, run.sh, residual skills | 572 | ~40 | |
| **Subtotal inside `agentic_port/`** | **31,842** | | 209 files |
| Supporting unit tests outside `agentic_port/` | 9,398 | ~0 | |
| **TOTAL** | **41,240** | **~2,400** | 289 files changed vs main |

---

## 3. What the pipeline does (8 phases)

Driven by `agentic_port/orchestrator/run.py` via `@register_stage` in `shared/registry.py`.

1. **Intake** — deterministic. Native vs `generic_op` eligibility screen. Gate: device win ≥ **1.05x**
   (`checks.MIN_DEVICE_WIN_MARGIN`). `--skip-intake` for offline dev.
2. **Classify** — 2a: LLM writes/validates a manifest. 2b: full-ledger device qualify; seeds
   *provisional demotions*. Gate: model-traced coverage ≥ **10%** (`MIN_PORT_COVERAGE_FRACTION`).
3. **Scaffold** — deterministic kernel copy + SPDX + CMake registration; LLM fills host stubs.
4a. **Translate** — LLM writes the ProgramFactory + `supported_by_codegen` + `is_demoted`.
4b. **Review** — read-only LLM, JSON verdict. Loop bounded by `per_phase_retry["4"]` (default 3).
5. **Build** — `./build_metal.sh -e` + import/callability check. Compile fail → reroute to 4a.
6. **Correctness** — `skills/verify` adapter, band=correctness. PCC vs torch golden, program-cache
   stability, routing checks for demoted / `scope:out`.
7. **Performance** — band=performance. Multi-ratio gates (§4). LLM only for `demote_analysis`.
8. **Conclude** — deterministic. Ship bar, emit coverage test, `gh pr`, cross-arch dispatch.

Also: `update.py` (full re-drive on generator drift) and `sync_op.py` (surgical kernel/host sync,
no correctness/perf).

---

## 4. The hard parts — MUST STAY AS CODE

### Perf gate thresholds (`skills/verify/lib/gates.py` + `constants.py`)
- **Wall**: ratio ≥ **0.98** (`WALL_TIE_BAND`) OR absolute deficit ≤ **3 µs** (`WALL_TIE_ABS_US`). Min-of-30 sampling.
- **Device vs native**: ratio ≥ **1.0**, paired-median CI noise guards, **300 ns** absolute escape (`DEVICE_VS_NATIVE_TIE_ABS_NS`).
- **Device vs generic_op**: ≥ **0.95** (`DEVICE_VS_GENERIC_TIE_BAND`). Inconclusive CI → pass-but-flagged.
- **Update mode vs previously shipped port**: ≥ **0.95** — hard fail, no waiver.
- **Op-level**: needs ≥1 strict win (wall or device > 1.0 vs native).
- **Device measurement**: **5** independent Tracy windows. Program-cache growth during timed reps = hard error.

### Demotion routing ("no hardcoded shapes")
Two-predicate contract, enforced in `phase4a_translate/stage.py::artifacts_missing()`:
- `supported_by_codegen` = **correctness only** (the sole check in the codegen prim's validate TT_FATAL).
- `is_demoted` = **perf gate**, only bites in the `auto` branch.
- So `auto = supported && !demoted` always wires the same way. `is_demoted` required even when empty.

Flow: 2b seeds provisional → 7 `classify_perf_failures()` splits **fix** (generic won on device, or
regression vs previous port → reroute to 4a) vs **demote** (generic never won) → LLM `demote_analysis`
rules on candidates → `routing.py` (`plan_demotion_rebuild`, `plan_perf_resolution_fallback`), up to
**8** resolution rounds / **6** demotion-rebuild rounds.

Reviewer-facing rule as stated in the tilize PR: *demote only when a config loses by more than
**300 ns** AND more than **5%***. Structural predicates, never a shape table.

### Coverage ledger (`lib/ledger.py`, 511 lines)
Enumerates codegen sweep grid vs upstream native sweep vs `port_scope`. Three scopes: `in`, `out`
(routing fallback), dropped. Drives all parameterized harness cases.

### Emitted contract tests (`phase8_conclude/emit_test.py`, 533 lines)
Generated pytest: `test_<op>_codegen_program_cache_hit`, unknown `implementation=` → `RuntimeError`,
sharded output memory config → native fallback via `_ROUTED`. Parsed with `ast.parse` before commit.

---

## 5. gh-aw capability findings (v0.84.x)

Docs: https://github.github.com/gh-aw/
Schema: https://raw.githubusercontent.com/github/gh-aw/main/pkg/parser/schemas/main_workflow_schema.json
(cached at `~/.cursor/projects/localdev-ebanerjee/agent-tools/451531c6-*.txt`)

| Need | Status | Evidence |
|---|---|---|
| Agent in tt-metal CI | **proven** | 4 workflows on main: `repo-assist`, `silencer`, `ci-failure-triage`, `daily-repo-status` |
| Card runner | **proven** | `runs-on` takes self-hosted label arrays (AND semantics) |
| Device in job | **proven (job level)** | top-level `container:` has `image`/`volumes`/`options` (string) → `--device /dev/tenstorrent` |
| Private repo checkout | **proven** | `CODEGEN_REPO_TOKEN` already used in PR #52344; also `checkout:` multi-repo |
| Long jobs | **proven** | `timeout-minutes` passthrough; 360-min cap is GitHub-hosted only, self-hosted = 5 days. PR #52344 uses 720. |
| PR / branch / comment / dispatch | **proven** | safe-outputs: `create-pull-request`, `push-to-pull-request-branch`, `add-comment`, `dispatch-workflow` (allowlist), `upload-asset`, `create-issue` |
| tt-dm-codegen ticket | **proven** | `create-issue: target-repo: owner/repo` + `allowed-repos` |
| No new API key | **proven** | `engine: copilot` + `copilot-requests: write` bills to Actions token |
| **Agent bash opens the card** | **OPEN** | AWF bind-mounts `/dev:/host/dev:ro` but passes no `--device` rule → cgroup device filter should deny `open()` |

Other useful frontmatter: `steps:` / `pre-steps:` / `post-steps:` (deterministic, run in agent job),
`jobs:` (full custom jobs before agent, incl. `jobs.agent.needs`), `runs-on-slim:` (framework jobs,
default `ubuntu-slim`), `sandbox.agent.mounts` (`src:dst:ro|rw`), `sandbox.agent.args` (x-internal,
appended to AWF invocation), `sandbox.agent.memory`, `repo-memory: true`, `tools.github.mode:
gh-proxy|local|remote` (gh-proxy avoids the Docker-backed MCP server), `max-ai-credits`.

### The `sandbox.agent: false` escape hatch is NOT available on tt-metal — verified 2026-08-07

`gh aw compile` auto-enables `strict: true`, and strict mode hard-rejects it:

```
error: strict mode: 'sandbox.agent: false' is not allowed because it disables the agent
sandbox firewall. Remove 'sandbox.agent: false' or set 'strict: false' to disable strict mode.
```

And `strict: false` workflows **cannot run on public repositories** — tenstorrent/tt-metal is public.
So the two settings are mutually exclusive here. Remaining escapes if AWF blocks the device:
`sandbox.agent.mounts`, `sandbox.agent.args` (x-internal), or the PR #52344 split.

---

## 6. tt-metal CI facts

- **SKU → runner labels**: `.github/sku_config.yaml` is the source of truth.
  - `wh_n150` = `[N150, in-service, cloud-virtual-machine]` (least contended)
  - `wh_n300_perf` = `[N300, in-service, bare-metal, pipeline-perf]`
  - `bh_p150_perf` = `[in-service, arch-blackhole, P150, pipeline-perf]`
- **Card job shape** (from `port-device.yaml`): `container.image` = dev docker image,
  `volumes: [workspace/docker-job:/work, /dev/hugepages-1G:/dev/hugepages-1G]`,
  `options: "--device /dev/tenstorrent"`. Skip hugepages on `viommu` labels.
- **Build**: `.github/workflows/build-artifact.yaml`, reusable, `timeout-minutes: 100`, takes `tracy:`.
  S3 ccache at `garage.garage.svc.cluster.local:3900` — **cluster-internal, likely unreachable from card runners**.
- **CRITICAL**: the artifact `ttm_any.tar.zst` packages **outputs only** —
  `ttnn/ttnn/*.so build/lib build/programming_examples build/test build/tools runtime tt_metal/pre-compiled`.
  **No `CMakeCache.txt`, no object files** → cannot incrementally rebuild from the artifact.
  An agent editing C++ must do one full build in-job (dev image has the toolchain) or cache the build tree.
- **Consumed via** `.github/actions/setup-job` (download artifact, untar, `uv pip install` wheel).
- `test-dispatch.yaml` = closest "run arbitrary command on a card after a build" primitive.
- Perf measurement: `TT_METAL_DEVICE_PROFILER=1` + `python -m tracy -r -p ...` →
  `cpp_device_perf_report.csv` → `tools/tracy/process_ops_logs.py` → `ops_perf_results_*.csv`.
- `CODEGEN_REPO_TOKEN` is **not** referenced anywhere on main; only on PR #52344's branch.

---

## 7. Prior art / exemplars

- **PR #52344** `ebanerjee/codegen-port-agent-workflows` — `port-agent.md` (423 lines, gh-aw,
  no card, dispatches) + `port-device.yaml` (410 lines, card, mechanical) + 2125-line lock file.
  **Parked, never completed a run.** Calls `agentic_port.orchestrator.run` → does not replace anything.
  Notes in its body: `workflow_dispatch` only resolves against the default branch (hence the
  `workflow_call` trigger); `CODEGEN_PORT_EXCLUSIVE_DEVICE=1` asserts sole card ownership.
- **Port PRs by mmoscickiTT**: #52312 concat, #52074 gather, #51919 tilize, #51913 permute,
  #50700 repeat_interleave, #50699 pad. Shape of the deliverable: ~4,000 lines C++ over ~23 files
  (`ttnn/cpp/ttnn/operations/data_movement/<op>/codegen/{,kernels}/`, + `sources.cmake`,
  `CMakeLists.txt`, `<op>.cpp` auto branch, nanobind, and a routing test under
  `tests/ttnn/nightly/.../test_<op>_codegen_routing.py`), plus a very detailed PR body:
  perf table, mermaid xychart, "Why it is faster", trailing-configs table, internal gate detail.

---

## 8. Open question being tested

**Can a process inside the AWF agent sandbox open `/dev/tenstorrent`?**

Design: one gh-aw workflow on a card runner, no `container:` (so AWF has a Docker daemon on the
host). A deterministic `steps:` pre-step probes the device **outside** the sandbox (control), then
the agent runs the same probe **inside** the sandbox (treatment). Difference isolates AWF.

Fallbacks, in order: `sandbox.agent.mounts` → `sandbox.agent: false` + `tools.github.mode: gh-proxy`
→ PR #52344's split (agent off-card, deterministic device job) which keeps most of the line-count win
but worsens iteration latency.

### RESULT — answered 2026-08-07. **The agent cannot open the card.**

Run: https://github.com/tenstorrent/tt-metal/actions/runs/31194862434 (branch `ebanerjee/awf-device-probe`, THROWAWAY — delete it).
Workflow compiled with gh-aw v0.85.4, ran on `[N150, in-service, cloud-virtual-machine]`, whole run green, agent leg took 19 s.

| Leg | Result |
|---|---|
| **CONTROL** (ordinary step, outside sandbox) | `uid=1000(ubuntu)`, groups include `docker`. `/dev/tenstorrent/0` present. `OPEN OK O_RDWR`, `OPEN OK O_RDONLY`. **Device is openable.** |
| **TREATMENT** (agent bash, inside AWF) | `/dev/tenstorrent/0` **visible** — mode `0o666`, chardev `240:0` — but `open()` fails **EPERM (errno 1)** on both `O_RDWR` and `O_RDONLY`. Container confirmed: 11 `AWF_*` vars, hostname `3bdfcc57060d`, AWF chroot as host user `ubuntu`. |

Exactly the predicted failure: AWF bind-mounts `/dev` so the node is *visible*, but the container
carries no `--device` rule, so the device cgroup denies `open()`.

### Escapes — all closed under strict mode (verified by compile)

tt-metal is public → `strict: false` cannot run there → `gh aw compile` auto-sets `strict: true`.

| Escape | Status |
|---|---|
| `sandbox.agent: false` | **rejected**: `strict mode: 'sandbox.agent: false' is not allowed because it disables the agent sandbox firewall` |
| `sandbox.agent.args: ["--enable-dind"]` | **rejected**: `strict mode: 'sandbox.agent.args' is not allowed because it is an internal implementation detail` |
| `sandbox.agent.mounts: ["/dev/tenstorrent:/dev/tenstorrent:rw"]` | **compiles**, but a bind mount grants no device-cgroup rule — expected to still EPERM. *Untested.* |
| AWF's own `container.enableDind`, `security.allowHostPorts`, `security.legacySecurity` | **not exposed in gh-aw frontmatter at all** (0 hits in the gh-aw JSON schema) |

AWF config schema (`awf-config.schema.json` v0.27.44) has `container.mounts` but **no devices field**.

### Consequence for the architecture

The agent process and the card **must be separate processes**. This does not sink the plan — it
pins it to the PR #52344 shape (agent reasons, a deterministic leg measures) rather than the
single-process loop. The line-count win is largely unaffected because the gate harness was always
going to be deterministic code; what suffers is **iteration latency**, since a translate → build →
measure → re-translate cycle can no longer happen inside one agent turn.

---

## 9. ROUND 2 — `mcp-scripts` SOLVES IT. Verified 2026-08-07.

Run: https://github.com/tenstorrent/tt-metal/actions/runs/31195804593 — green, agent leg 37 s.
Three legs, one N150, same job.

| Leg | Hostname | Container markers | `open()` |
|---|---|---|---|
| CONTROL — ordinary step | `tt-metal-ci-vm-136` | — | **OK** (RDWR + RDONLY) |
| TREATMENT — agent bash in AWF | `37f3f355a679` | **11** `AWF_*` vars | **EPERM** (errno 1), both modes |
| **MCPTOOL — `probe_device` mcp-script** | **`tt-metal-ci-vm-136`** | **0** `AWF_*` vars | **OK** (RDWR + RDONLY) |

Same `/dev/tenstorrent/0` node (mode `0o666`) visible to all three. The hostname flipping from the
container hash back to the CI VM name, and the `AWF_*` count dropping 11 → 0, confirms the tool
really did execute on the runner host.

The gh-aw docs are accurate: mcp-scripts *"run as an HTTP MCP server on the GitHub Actions runner,
**outside the agent container**"*, and the agent reaches them over `host.docker.internal`. Visible in
the compiled lock file as a plain host step, `bash "${RUNNER_TEMP}/gh-aw/actions/start_mcp_scripts_server.sh"`,
with the agent's MCP config pointing at `http://host.docker.internal:$GH_AW_MCP_SCRIPTS_PORT`.

### Why this is the answer

The tight loop is restored **without weakening the sandbox**. The agent still cannot touch the card,
which is the correct security posture and the one strict mode insists on; it calls a declared tool
that does. The trust boundary stays exactly where gh-aw wants it.

This maps cleanly onto the architecture the old pipeline already had: `tt-device-mcp` /
`DaemonClient` existed to serialise card access behind an RPC seam. `mcp-scripts` **is** that seam,
supplied by the framework. So the port workflow becomes:

```yaml
mcp-scripts:
  build-and-measure:
    description: "Rebuild the op and return the gate verdict as JSON"
    timeout: 3600            # accepted; 300 verified working. Default is 60.
    inputs: { op: {...}, cases: {...} }
    run: |
      ./build_metal.sh -e && python3 .github/scripts/port_gate.py --op "$INPUT_OP" ...
```

Agent edits C++ → calls `build-and-measure` → reads the JSON verdict → edits again. All in one
job, on one card, with the agent sandboxed throughout.

### Caveats to check when building the real thing

- **`timeout`** is per tool call, in seconds; 300 verified. A cold `build_metal.sh` may exceed even
  a generous value — do the first build in a `steps:` pre-step and let the tool only do incremental
  rebuilds. (See §6: the build artifact has no object files, so the first build is expensive.)
- **Output >500 chars is spilled to a file** and the agent gets a path. Return compact JSON, or
  confirm the agent can read the spill path from inside AWF.
- **Always give the agent a `noop` safe output.** In both rounds the agent felt obliged to make a
  safe-output call; in round 2, with only the auto-injected `create-issue` available, it filed
  issues #52480 and #52477 despite explicit instructions not to. Both closed as not-planned.
- The agent job ran fine on a card runner with no `container:` — gh-aw auto-installed Node 24, AWF
  chrooted as host user `ubuntu`, model resolved `auto` → `claude-sonnet-5`.

### Revised verdict

No blocking unknown remains. The single-job, on-card, tight-loop architecture is achievable, with
the measurement behind an `mcp-scripts` tool rather than in the agent's own bash — which is a better
design than the one originally sketched, not a compromise.

---

## 10. THREAT MODEL AND HARDENING RULES

Because `mcp-scripts` deliberately runs outside AWF, the tool body *is* the trust boundary. This
section is the design constraint set for the real workflow. Written 2026-08-07 after the round-2
probe.

### 10.1 The principal, not the capability

The instinct to compare this against "an agent loose on a card" is the wrong baseline. Every card CI
job in tt-metal already compiles and runs arbitrary repository code with full device access — that
is what hardware testing is. This workflow introduces **no new capability class**. What it
introduces is a new *principal* deciding what that code does: a model, and transitively anyone who
can influence the model's inputs.

So the danger is set almost entirely by the trigger surface:

- `workflow_dispatch` → principal is a human with write access. Essentially the status quo.
- `issue_comment` / `pull_request` from arbitrary users → untrusted text reaches the agent, and
  prompt injection converts directly into whatever the tool permits.

### 10.2 The inherent exposure — state it plainly

**You cannot port C++ ops without compiling and executing agent-authored C++.** The agent writes a
program factory, it is linked into `ttnn`, and a test loads `ttnn` and runs it. At that moment
agent-authored code is executing as a host process with device access. Two corollaries that are easy
to miss:

- `sources.cmake` lives *inside* each op directory (`ttnn/cpp/ttnn/operations/<op>/sources.cmake`),
  so a write-allowlist scoped to the op directory necessarily includes it — and CMake has
  `execute_process()`. Build-time execution is in scope, not just test-time.
- Input hardening (§10.4, H1) blocks the *cheap* path to a shell. It does not close the C++ path. A
  determined or injected agent gets host execution with more steps.

The honest security argument is therefore **not** "the agent is sandboxed." It is: *the on-card job
is an untrusted-code execution context, tt-metal already has a posture for that, and we keep the job
worth nothing to steal.* Everything in §10.4 follows from that framing.

### 10.3 Threats

| # | Threat | Primary control |
|---|---|---|
| T1 | Shell injection via tool inputs → host RCE outside AWF | H1 |
| T2 | Agent-authored C++/CMake executes at build or test time (inherent) | H2 |
| T3 | Theft of `CODEGEN_REPO_TOKEN` / `GITHUB_TOKEN` from the runner | H2, H6 |
| T4 | Exfiltration over the host's unfiltered network | H2 |
| T5 | Self-escalation: agent edits the gate script, workflow, or build script that enforces policy | H3, H5 |
| T6 | Manifest laundering: agent writes a test file, then selects it as "pre-existing" | H3, H4 |
| T7 | Wedged card / runaway loop burning card time | H7, H8 |
| T8 | Bricking via firmware flash | H2, H9 |
| T9 | Prompt injection from untrusted trigger content | H10 |
| T10 | Unreviewed change reaching `main` | H11 |

T4 deserves a note: AWF's egress allowlist governs the *agent container*. The mcp-scripts server is
a host process and is not behind it. Inferred from the architecture, not yet measured — one curl to
a non-allowlisted domain from a tool body would settle it, and it is worth doing.

### 10.4 Hardening rules

**H1 — Inputs are data, never command fragments.** Tool inputs are typed and `enum`-constrained
where possible. Nothing is ever interpolated into a shell. The tool body passes inputs to a
checked-in script via `argv` or a JSON file. No `${{ }}` expression appears anywhere in a tool body
(that is plain Actions script injection; gh-aw sanitizes the *prompt*, not a `run:` block).
Structured values like shapes are validated as bounded lists of ints, not free strings.

**H2 — The tool is a thin launcher; the payload runs contained.** The tool must not build or test
directly on the host. It launches them in a container:

```
docker run --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
           --network none -v "$WORKSPACE:$WORKSPACE" ...
```

This is idiomatic here — `.github/actions/docker-run/action.yml` already defaults `device` to exactly
that passthrough. `--network none` is the important addition: it means agent-authored C++ keeps the
card but loses exfiltration (T4), the ability to fetch tooling (T8), and reach into the runner
beyond the workspace (T3). This single rule converts the inherent exposure of §10.2 from "host
compromise" into "workspace and card only."

**H3 — Write-path allowlist, recomputed from the base SHA on every call.** Before doing anything,
the gate script runs `git diff --name-only <BASE_SHA>` and rejects if any changed path falls outside
the op source directories. This is what actually protects test files, build scripts, `.github/**`,
and the gate script itself. It must be enforced *in the tool*, not stated in the prompt — prompt
rules are not enforcement.

**H4 — Freeze the test manifest before the agent starts.** See §10.5.

**H5 — Stage the enforcement outside the workspace.** At job start, copy the gate script to
`$RUNNER_TEMP/gate/` and invoke that copy, so the thing checking the diff is not itself inside the
tree being diffed. Defence in depth with H3.

**H6 — Keep the job worth nothing to steal.** No secrets in any tool `env:`. Check out tt-dm-codegen
in a `steps:` pre-step at a **pinned SHA** with `persist-credentials: false`, so
`CODEGEN_REPO_TOKEN` is never left in `.git/config` for agent code to read. The tool never needs the
token. Keep `permissions:` minimal (`contents: read`).

**H7 — Budget the card.** A per-call `timeout` (300 s verified; default is 60), a job
`timeout-minutes`, and an invocation counter persisted by the tool that refuses after N calls.
gh-aw has no per-tool call cap, so this has to be implemented in the tool body.

**H8 — Reset on exit.** An `always()` cleanup step running `tt-smi -r`, so a wedged card does not
poison the next job. Routine in tt-metal CI already; `ensure-bh-links-online` is the
reset-and-recheck pattern to copy.

**H9 — Nothing that can flash.** `tt-flash` appears nowhere in `.github/` and must not be added to
the workflow's image. With H2's `--network none`, agent code cannot fetch it either. Bricking
requires a firmware write; without a flashing tool the realistic worst case is a wedged card, which
H8 recovers.

**H10 — Human triggers only, at least to start.** `workflow_dispatch` (and optionally a
`slash_command` restricted to write-access users). Never `pull_request_target` or `issue_comment` on
the card-holding job.

**H11 — Output only through safe-outputs.** `create-pull-request` for the product, plus `noop` —
which is a real safe-output (confirmed in the schema) and must be declared, because in both probe
rounds the agent felt obliged to make some safe-output call and, lacking `noop`, filed unwanted
issues. Any tt-dm-codegen ticket or PR goes through safe-outputs too, never through a tool.

### 10.5 Enforcing "only pre-existing tests"

The governing rule: *the workflow may only run tests that already exist in tt-metal or
tt-dm-codegen.* Two mechanisms, both required, because either alone is defeatable.

1. **Frozen manifest.** Before the agent runs, enumerate the allowed tests from the **pristine base
   commit** — a clean worktree at `BASE_SHA` for tt-metal and the pinned SHA for tt-dm-codegen — and
   write the manifest outside the workspace. The tool accepts a `test_id` only if it is in that
   frozen list. Collecting from the *working tree* instead would be the whole bug: the agent writes
   `test_evil.py`, and it is now "pre-existing."
2. **Test paths are unwritable.** H3's allowlist excludes every test path, so the agent cannot edit
   an existing test into something else. Without this, a frozen list of test IDs is worthless
   because the IDs still resolve to mutable content.

Note the check must resolve the *content* as well as the ID: verify on each call that no test path
appears in the diff, rather than trusting that the manifest still describes what is on disk.

**What this rule does and does not bound.** It bounds the *harness* — the agent picks from a fixed
menu of measurements and never composes a command line. It does **not** bound the *payload*: the
code under test is agent-authored by construction (§10.2), and it dispatches agent-authored kernels
to the card. That is unavoidable in a porting pipeline and is exactly why H2 matters more than any
other rule here.

### 10.6 Residual risk

After H1–H11, a fully adversarial agent can: write C++ that runs in a network-isolated container
with the card attached, wedge or hang that card, and produce a bad PR. It cannot reach the network,
read runner credentials, modify tests or build scripts, run anything outside the frozen manifest,
flash firmware, or merge anything.

That is a strictly *smaller* privilege set than any human contributor pushing a branch to tt-metal
today, and the worst outcome is a wedged card recovered by `tt-smi -r`. On that basis the workflow is
safe to run in CI under a human-only trigger.

**Open item:** confirm empirically whether the mcp-scripts host process is outside AWF's egress
allowlist (T4). Cheap to test, and it determines how load-bearing `--network none` really is.

---

## 11. DECISION RECORD — accepted risk posture (2026-08-07)

**Decision (Evan):** the agent's output is treated as trusted-enough. Rationale: any developer with
permission to run these workflows could already do equivalent damage if they were crafty, and the
ops in question are code-generated to begin with. §10 is therefore **not a gate on building the
prototype**. Build the straightforward thing.

This does not make §10 dead weight, but it does change *why* most of the rules exist, and the
re-justification matters because it changes which ones to actually implement.

### What survives, on new grounds

The rules that survive mostly stop being security controls and become **result-integrity** and
**cost** controls:

- **H3 / H4 (write-path allowlist + frozen test manifest) — keep, and they get more important, not
  less.** The threat they now defend against is not an attacker; it is reward hacking. An agent that
  can edit a test or add its own is an agent that can make its own performance numbers come out
  right. Since the entire product of this pipeline is a *performance claim*, an unfalsifiable claim
  is worse than no pipeline. This is the one place worth being strict.
- **H7 (invocation budget, timeouts) — keep.** Card time is a shared, contended resource. This is
  now about not letting a confused agent burn a runner for an hour, which is a cost and availability
  concern that survives any trust assumption.
- **H8 (`tt-smi -r` on exit) — keep.** Pure operational hygiene, already standard in tt-metal CI.
  A wedged card poisons the *next* job, which is someone else's problem.
- **H11 (safe-outputs, including `noop`) — keep.** `noop` is a functional requirement, not a
  control: without it the agent invents a safe-output call and files junk issues (observed twice).
- **H1 (typed inputs, no shell interpolation) — keep, it is free.** As much about not breaking on a
  stray quote as about injection.
- **H6 (`persist-credentials: false`) — keep, it is one line** and standard practice.

### What to drop

- **`--network none` (part of H2): drop.** Under the accepted posture its only job was blocking
  exfiltration, and it has a real cost — ttnn tests routinely fetch model weights and pip
  dependencies, so an isolated network would break legitimate runs. Still run build and test in a
  container, but for environment reproducibility and the existing `docker-run` convention, not
  isolation. This is an active simplification, not an omission.
- **H5 (staging the gate script outside the workspace): drop.** Defence-in-depth against a
  self-escalating agent; redundant once H3 is in place and the agent is trusted.
- **H9 (no flashing tools): drop as an explicit rule.** Already true by default; nothing to do.
- **H10 (human-only trigger): keep initially, but for cost, not safety.** `workflow_dispatch` while
  iterating so runs are deliberate. Relaxing it later is a scheduling question, not a security one.
- **T4 open item: deprioritised.** Only mattered because `--network none` depended on it. Worth
  knowing eventually; not a prerequisite.

### Net effect on the prototype

The hardening collapses to a short list that a reasonable engineer would write anyway: validated
inputs, a diff check that keeps the agent out of the test files, budgets, and a card reset on exit.
None of it is architectural. Proceed directly to the §9 design — single on-card job, agent in AWF,
measurement behind an `mcp-scripts` tool.
