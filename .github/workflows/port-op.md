---
description: |
  Port a tt-dm-codegen generic_op into tt-metal as a C++ program factory, on a card, in one job.

  An op that exists only as a tt-dm-codegen generic_op already has a working kernel and a measured
  device-time win, but it pays a per-call program-descriptor rebuild that eats the win at wall
  clock. Porting it means transliterating the generator's builder into a cached DeviceOperation and
  wiring the native/codegen routing, then proving on silicon that the win survives.

  The deterministic parts run before the agent: kernels are copied, the build is registered, the
  tree is compiled, and a native baseline is measured. The agent writes C++ and calls two tools --
  `build` and `verify` -- until the gates pass, then opens a draft PR. It never picks the cases,
  the thresholds, or the measurement method, and `verify` refuses to run at all if the working tree
  has changed outside the port's own files.

  V1 scope: one op, one arch (N150), `workflow_dispatch` only. `is_demoted()` ships as a stub.

on:
  workflow_dispatch:
    inputs:
      op:
        description: "Op to port; must have a manifest at agentic_port/manifests/<op>.yaml"
        required: true
        type: string
        default: pad
      category:
        description: "ttnn/cpp/ttnn/operations/<category>/<op>"
        required: false
        type: string
        default: data_movement
      codegen-ref:
        description: "tt-dm-codegen ref to port from"
        required: false
        type: string
        default: codegen_agentic_port
      docker-image:
        description: "Dev image carrying both the build toolchain and the runtime"
        required: false
        type: string
        default: ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
      perf-limit:
        description: "Number of in-scope cases to measure per performance band"
        required: false
        type: string
        default: "24"

permissions:
  contents: read
  copilot-requests: write

engine: copilot

network: defaults

# One port at a time per op: two concurrent runs would fight over the same card and produce
# timings that are noise rather than measurement.
concurrency:
  group: "gh-aw-port-op-${{ inputs.op }}"

timeout-minutes: 330

runs-on: [N150, in-service, cloud-virtual-machine]

# Same container recipe as bisect-dispatch, the existing build-and-test-on-a-card job: the dev
# image carries the build toolchain and the runtime, and the card is passed through. Because the
# job itself is containerised, the tools below inherit device access rather than needing it
# granted separately.
container:
  image: ${{ inputs.docker-image }}
  volumes:
    # Subdir to work around https://github.com/actions/runner/issues/691
    - ${{ github.workspace }}/docker-job:/work
    - ${{ github.workspace }}/codegen:/codegen
    - /dev/hugepages-1G:/dev/hugepages-1G
    # HOME is hardcoded: https://github.com/actions/runner/issues/863
    - /home/ubuntu/.ccache-ci:/github/home/.ccache
  options: >
    --device /dev/tenstorrent
    --group-add 1457
    --tmpfs /tmp

steps:
  # The frontmatter schema carries no job-level or container-level `env`, so the environment is
  # established here. Everything written to $GITHUB_ENV is visible to the later steps and to the
  # tools, which run as part of this same job.
  - name: Configure environment
    run: |
      {
        echo "TT_METAL_HOME=/work"
        echo "PYTHONPATH=/work:/work/ttnn:/work/tools:/codegen"
        echo "LD_LIBRARY_PATH=/work/build/lib"
        echo "ARCH_NAME=wormhole_b0"
        echo "CCACHE_TEMPDIR=/tmp/ccache"
        echo "CCACHE_BASEDIR=/work"
        echo "TRACY_NO_INVARIANT_CHECK=1"
        echo "TRACY_NO_ISA_EXTENSIONS=1"
      } >> $GITHUB_ENV
      mkdir -p /tmp/ccache

  - name: Checkout tt-metal
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
    with:
      fetch-depth: 0
      submodules: recursive
      path: docker-job
      # The agent has no business holding a push token; the pull-request safe output does the
      # pushing from outside the sandbox.
      persist-credentials: false

  # Read-only, credentials not persisted: the agent must never be able to push to the generator
  # repo, and the port only ever reads from it.
  - name: Checkout tt-dm-codegen
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
    with:
      repository: tenstorrent/tt-dm-codegen
      ref: ${{ inputs.codegen-ref }}
      token: ${{ secrets.CODEGEN_REPO_TOKEN }}
      persist-credentials: false
      path: codegen

  - name: Record the base commit
    working-directory: /work
    run: |
      git config --global --add safe.directory /work
      git config --global --add safe.directory /codegen
      echo "PORT_BASE_SHA=$(git rev-parse HEAD)" >> $GITHUB_ENV
      echo "PORT_OP=${{ inputs.op }}" >> $GITHUB_ENV
      echo "PORT_CATEGORY=${{ inputs.category }}" >> $GITHUB_ENV
      echo "PORT_MANIFEST=/codegen/agentic_port/manifests/${{ inputs.op }}.yaml" >> $GITHUB_ENV
      echo "PORT_LIMIT=${{ inputs.perf-limit }}" >> $GITHUB_ENV

  # Scaffold before building, not after: this registers the codegen sources in CMake once, so
  # every rebuild the agent triggers is a plain incremental compile with no re-configure.
  - name: Scaffold the port
    working-directory: /work
    run: |
      python3 .github/scripts/port/scaffold.py \
        --op "${{ inputs.op }}" --category "${{ inputs.category }}" \
        --ttmetal-home /work --codegen-root /codegen

  # The remote ccache credentials are bound to this step alone and never written to $GITHUB_ENV.
  # Only this full build needs them; the agent's rebuilds recompile three translation units
  # against an already-warm local tree, so leaking shared-cache write access into the agent's
  # environment would buy nothing and risk poisoning the cache for every other build in CI.
  - name: Build tt-metal
    working-directory: /work
    timeout-minutes: 150
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.GARAGE_S3_ACCESS_KEY }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.GARAGE_S3_SECRET_KEY }}
      AWS_DEFAULT_REGION: garage
      CCACHE_REMOTE_ONLY: "true"
      CCACHE_COMPRESS: "true"
      CCACHE_REMOTE_STORAGE: "s3://ccache|region=garage|prefix=tt-metal|endpoint_url=http://garage.garage.svc.cluster.local:3900"
    run: |
      if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "::warning::Garage S3 credentials missing; this build will be cold."
      fi
      ccache -z || true
      ./build_metal.sh --build-dir build --build-type Release --enable-ccache --enable-profiler
      ccache -sv || true

  - name: Native baseline
    working-directory: /work
    continue-on-error: true
    run: |
      python3 .github/scripts/port/gate.py \
        --op "${{ inputs.op }}" --manifest "$PORT_MANIFEST" --category "${{ inputs.category }}" \
        --band performance --repo /work --work /tmp/port-baseline \
        --limit "${{ inputs.perf-limit }}" --skip-write-check || true
      echo "Baseline recorded (the ported leg is expected to fail here: it is still a stub)."

tools:
  edit:
  bash: [":*"]

# Both tools are thin launchers over checked-in scripts. No agent-supplied text is interpolated
# into a shell command; `band` is a closed enum the compiler enforces.
mcp-scripts:
  build:
    description: >-
      Rebuild ttnn after editing the codegen sources. Returns compiler diagnostics on failure and a
      short success line otherwise. Call this after every batch of edits, before calling verify.
    timeout: 2400
    run: |
      cd /work
      cmake --build build --target install -j"$(nproc)" 2>&1 | tail -120
      exit "${PIPESTATUS[0]}"
  verify:
    description: >-
      Grade the port on real hardware and return the verdict as JSON. `correctness` checks
      bit-exactness against a torch golden for in-scope cases and native fallback for out-of-scope
      ones; `performance` measures wall clock and device kernel duration against native and against
      the tt-dm-codegen prototype. Refuses to run if the working tree has changed outside the
      port's own files. You cannot change what it measures or where the thresholds sit.
    timeout: 3600
    inputs:
      band:
        type: string
        description: "One of: correctness, performance, both."
    run: |
      cd /work
      # The schema has no enum, so the allowlist lives here. `band` is the only agent-supplied
      # value in this file and it never reaches a shell as anything but a matched literal.
      case "$INPUT_BAND" in
        correctness|performance|both) ;;
        *) echo "verify: band must be correctness, performance, or both (got: $INPUT_BAND)" >&2; exit 2 ;;
      esac
      python3 .github/scripts/port/gate.py \
        --op "$PORT_OP" --manifest "$PORT_MANIFEST" --category "$PORT_CATEGORY" \
        --band "$INPUT_BAND" --repo /work --base-sha "$PORT_BASE_SHA" --limit "$PORT_LIMIT"

safe-outputs:
  mentions: false
  create-pull-request:
    draft: true
    title-prefix: "[codegen-port] "
    labels: [codegen-port, gh-aw]
  # Without a no-op the agent has only one way to end, and an agent with a single available action
  # will take it -- filing a PR for a port it knows does not work. This makes stopping expressible.
  noop:
---

# Port `${{ inputs.op }}` to a C++ program factory

You are porting the tt-dm-codegen generic_op **`${{ inputs.op }}`** into tt-metal as a native C++
`DeviceOperation`, and proving on silicon that it beats the existing implementation.

The tree is already scaffolded and built. The kernels are copied, the three `codegen/` source files
exist as empty stubs, they are registered in CMake, and a native baseline has been measured. Your
job is to fill in those stubs and wire the routing.

## Where things are

| What | Where |
| --- | --- |
| tt-metal checkout (your working tree) | `/work` |
| The op you are porting | `/work/ttnn/cpp/ttnn/operations/${{ inputs.category }}/${{ inputs.op }}/` |
| Your stubs to fill in | `.../${{ inputs.op }}/codegen/` |
| tt-dm-codegen (read-only) | `/codegen` |
| The manifest — source of truth | `/codegen/agentic_port/manifests/${{ inputs.op }}.yaml` |
| The porting guide — read this first | `/codegen/agentic_port/knowledge/porting-guide.md` |
| The generator you are transliterating | the manifest's `codegen_builder` path, under `/codegen` |
| **A finished, merged port to imitate** | `/work/ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/` |

Read the manifest and the porting guide in full before you write anything. The merged `repeat`
port is the single most useful thing here: it is a complete, reviewed example of exactly the
deliverable you are producing, including its routing in `repeat/repeat.cpp`. Follow its structure.

## What to write

**A. Program factory.** Transliterate the manifest's `codegen_builder` into a declarative
`create_descriptor` in `codegen/${{ inputs.op }}_codegen_program_factory.cpp`, per the porting
guide's factory-translation section. Wire real circular-buffer and kernel args from the manifest's
`cache_key_fields`, not placeholders.

Every literal you write — CB sizing, work-split math, compile-time args, routing constants — must
trace to a manifest field, a builder constant, a device query, or a tensor property. A number that
merely happens to make one case pass is a defect even when the case passes.

**B. Routing.** All four pieces, in the shared files, not in new ones:

1. `supported_by_codegen()` in `codegen/${{ inputs.op }}_codegen_supported.{hpp,cpp}`, transcribed
   from the generator sweep's `invalidate_vector` plus any op guards. This predicate is about
   **correctness and device-resource feasibility, never performance**. Two things it must get
   right, both of which the sweep cannot tell you:

   - **Bound every dimension-scaled circular buffer.** If a CB's footprint (page size × depth)
     scales with a tensor dimension, reject configs whose minimum viable CB does not fit in per-core
     L1, and scale depth down to what does fit. The Python reference gets this free from
     `ProgramFactory.assemble`'s preflight; a port that drops it turns an `auto` call that should
     have fallen back to native into a hard allocation failure. Sweeps only ever run shapes small
     enough to fit, so `invalidate_vector` will never mention this. Walk **every leaf of your
     dispatch tree** — a guard written for the default builder leaves the alternates unbounded.
   - **Transcribe general conditions, not example shapes.** Each `scope: out` case in the manifest
     is one representative of a general condition stated in its `note`. Write the condition (e.g.
     `logical[-2] % TILE_H != 0`), never an exact-match on the tuple. An exact-match lets every
     other shape meeting the same condition wrongly reach codegen, and it will be rejected on
     review. If you cannot name the general condition from the `note`, read the op-orchestration
     source's own guard and transcribe that.

   It must agree with the manifest: every `scope: in` case returns `true`, every `scope: out` case
   returns `false`.

2. `is_demoted()` in the same files. This is the performance routing gate, consulted **only** by
   the `auto` branch. **For v1, emit it returning `false` for everything** — demotion analysis is
   deliberately out of scope for this run. Emit it anyway so the routing shape is right and later
   work has somewhere to land.

3. The host free function `ttnn::${{ inputs.op }}(..., implementation=)` in the shared
   `${{ inputs.op }}.cpp`:
   - `"auto"` (the default) → codegen iff `supported_by_codegen(attrs) && !is_demoted(attrs)`,
     else the native prim.
   - `"native"` → always the native prim, unconditionally.
   - `"codegen"` → check `supported_by_codegen()` **first** and `TT_FATAL` if it returns false,
     rather than silently falling back. Do not consult `is_demoted()` here; forced codegen must
     still run demoted-but-correct cases.
   - **Account for every parameter the native free function takes.** The ones controlling placement
     or execution rather than tensor content — `use_multicore`, `sub_core_grids`, worker grid
     overrides, compute-kernel config — are honoured by none of the codegen builders, which place
     work over the full `compute_with_storage_grid_size()`. When any of them is set, `"auto"` must
     route to native and `"codegen"` must `TT_FATAL` naming it. Accepting the call and ignoring the
     control lands work on cores the caller deliberately reserved. Put this in its own predicate,
     shared by both branches, and keep it out of `supported_by_codegen()`.

4. `prim::${{ inputs.op }}_codegen` validate, in
   `codegen/${{ inputs.op }}_codegen_device_operation.cpp`: a second `TT_FATAL(supported_by_codegen(...))`
   on cache miss, plus **the native op's own structural `TT_FATAL`s** — device storage, non-null
   buffer, rank and alignment invariants. `supported_by_codegen()` only asks about layout, dtype and
   memory config, all of which answer perfectly well for a host-side or deallocated tensor, so a
   scope-gate-only validate lets one through and the first symptom is a null-buffer dereference
   where native gave a clear error.

5. The Python binding: expose the `implementation` kwarg on the **existing** binding in
   `${{ inputs.op }}_nanobind.cpp`, defaulting to `"auto"`.

**C. Remaining hooks.** Fill in `compute_output_specs`, `create_output_tensors`, and any other
DeviceOp hooks per the porting guide's checklist for this op's tier.

## How to iterate

1. Write or fix code.
2. Call **`build`**. Fix what it reports. Repeat until it compiles.
3. Call **`verify`** with `band: correctness`. Fix real defects until it passes.
4. Call **`verify`** with `band: performance`.
5. When the verdict is `win`, open the draft PR.

Read the verdict rather than pattern-matching on it:

- **`win`** — all gates pass and something is genuinely faster. Open the PR.
- **`back-to-translate`** — a gate failed. `failing` names the cases. Go fix the code.
- **`not-a-candidate`** — every gate held but nothing actually got faster. This port is not worth
  landing. **Do not open a PR.** Use the no-op output and explain what you measured.
- **`blocked`** — the harness could not run, or the tree changed outside the port's own files. Read
  the error. If it is the write-path guard, revert the stray edit; do not try to work around it.

## Rules

**Do not go looking for a prior implementation of this op.** No `git log`, `git show`, or `git
branch` spelunking, no diffing against other branches, no reading a port from another checkout.
Anything you find that way is stale scratch work, not a reference, and using it invalidates this
run. Build the port from the manifest, the generator source, the porting guide, the merged `repeat`
port, and nothing else.

**Do not write tests.** No new `test_*.py`, no ad-hoc device scripts, no edits to any existing
test, harness, or gate script. Correctness and performance are measured for you by `verify`,
against cases derived from the generator's own sweep. `verify` recomputes the diff from the base
commit every time it runs and refuses if anything outside the port's own files changed, so editing
the harness does not get you a passing grade — it gets you a `blocked` verdict.

**Do not touch git history.** No `git push`, no `git commit --amend`, nothing that rewrites
history, and never target `main`. Leave your work as uncommitted edits; the pull-request output
handles the rest.

**Do not invent numbers.** Every performance figure you report must come from a `verify` verdict.
Do not estimate, extrapolate, or restate a ratio you did not measure.

## The pull request

Open it as a draft, describing:

- **What changed** — the file list and the shape of the port.
- **Why this is faster** — the mechanism, grounded in source. Name the specific native cost that is
  avoided and the generated design choice that avoids it. Do not claim causality you cannot point
  at, and do not write "codegen is faster."
- **Measured performance** — a table straight from the `verify` verdict: per-configuration wall
  ratio, device-vs-native, and device-vs-prototype, plus the case counts by scope.
- **Routing** — the conditions under which `auto` falls back to native, and why each one exists.
- **Known gaps** — anything you could not resolve, and anything the manifest asserts that the
  hardware contradicted. Say so plainly rather than quietly working around it. Note that
  `is_demoted()` is a v1 stub.

If the verdict is not `win`, do not open a pull request at all. Report what you measured and stop.
