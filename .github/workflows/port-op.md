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
  # Prototype shakedown only. `workflow_dispatch` resolves against the default branch, so a
  # workflow that is not yet on main cannot be dispatched -- this push trigger is how v1 gets
  # exercised before it lands. Remove it when this merges; every input falls back to its default
  # on a push, which is why the defaults are duplicated inline throughout.
  push:
    branches: [ebanerjee/port-op-dryrun]
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

# One port at a time per branch: two concurrent runs would fight over the same card and produce
# timings that are noise rather than measurement. Keyed on the ref rather than the op because
# workflow-level fields are evaluated before a push event has any inputs to read.
concurrency:
  group: "gh-aw-port-op-${{ github.ref_name }}"

timeout-minutes: 330

runs-on: [N150, in-service, cloud-virtual-machine]

# Deliberately NOT a job-level `container:`. The agent firewall and the MCP gateway both start
# their own containers, so putting the job inside one would nest Docker and break them. Instead the
# job stays on the runner host -- which a probe confirmed can open /dev/tenstorrent -- and the
# toolchain lives in a long-lived container that the steps and tools exec into. The build image and
# the card are therefore available without the agent ever holding either.

# These run before the checkout, which is the whole point: a runner that cannot host the agent
# should cost seconds to reject, not the half hour it takes to reach the step that finds out.
pre-steps:
  # gh-aw pins its MCP gateway to 127.0.0.1:8080 and starts it by removing any leftover
  # `awmg-mcpg` container and immediately rebinding the port -- 1.3 ms apart in run
  # 31218783110, which failed with `address already in use` because the dying container still
  # held the binding. These are persistent VMs, so leftovers are normal and the port is not
  # configurable from frontmatter. Clearing them here gives the old binding minutes to drop, and
  # a port held by something else becomes a fast failure that names the holder. Removing another
  # job's gateway is not a new hazard -- gh-aw removes that container by name unconditionally
  # itself -- and a card runner only ever hosts one job, since the card is exclusive.
  #
  # `portdev` is removed for a different reason: this workflow leaks it (see post-steps), and a
  # surviving one would make `docker run --name portdev` fail outright.
  - name: Clear leftover state from earlier runs on this runner
    run: |
      docker rm -f awmg-mcpg portdev 2>/dev/null || true

      if ! command -v ss >/dev/null; then
        echo "ss is unavailable; leaving the port check to gh-aw"
        exit 0
      fi
      for _ in $(seq 1 15); do
        if ! ss -ltnH 'sport = :8080' | grep -q .; then
          echo "gateway port 8080 is free"
          exit 0
        fi
        sleep 2
      done

      # Naming the holder is the whole value of failing here: a container means leftover state and
      # a rerun will land somewhere clean, whereas a host service means this runner pool cannot
      # host a gh-aw agent at all. `-p` needs root to attribute sockets owned by other users.
      echo "::error::Port 8080 is still bound after 30s. The gh-aw MCP gateway binds it and cannot be moved, so this runner cannot host the agent."
      sudo -n ss -ltnp 2>/dev/null || ss -ltn || true
      docker ps || true
      exit 1

steps:
  # The frontmatter schema carries no job-level or container-level `env`, so the environment is
  # established here. Everything written to $GITHUB_ENV is visible to the later steps and to the
  # tools, which run as part of this same job.
  - name: Configure environment
    run: |
      {
        echo "PORT_CONTAINER=portdev"
        echo "TT_METAL_HOME=/work"
      } >> $GITHUB_ENV
      mkdir -p /tmp/ccache

  # Checked out at the workspace root, not a subdirectory: gh-aw's own later steps assume the
  # workspace root is the repository, and a subdirectory checkout fails them.
  - name: Checkout tt-metal
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
    with:
      # Shallow: the only history this run needs is the base commit the write-path guard diffs
      # against, and that is HEAD. Full history plus recursive submodules on a card runner cost
      # tens of minutes for nothing.
      fetch-depth: 1
      submodules: recursive
      # The agent has no business holding a push token; the pull-request safe output does the
      # pushing from outside the sandbox.
      persist-credentials: false

  # Read-only, credentials not persisted: the agent must never be able to push to the generator
  # repo, and the port only ever reads from it. `actions/checkout` cannot place a repo outside the
  # workspace, so it lands in a dotted directory that is then excluded from tt-metal's index --
  # otherwise the whole generator tree reads as untracked files and the write-path guard, which
  # counts untracked files, would refuse every verify call.
  - name: Checkout tt-dm-codegen
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
    with:
      repository: tenstorrent/tt-dm-codegen
      ref: ${{ inputs['codegen-ref'] || 'codegen_agentic_port' }}
      token: ${{ secrets.CODEGEN_REPO_TOKEN }}
      persist-credentials: false
      path: .codegen

  - name: Record the base commit
    run: |
      echo ".codegen/" >> .git/info/exclude
      echo "PORT_BASE_SHA=$(git rev-parse HEAD)" >> $GITHUB_ENV
      echo "PORT_OP=${{ inputs.op || 'pad' }}" >> $GITHUB_ENV
      echo "PORT_CATEGORY=${{ inputs.category || 'data_movement' }}" >> $GITHUB_ENV
      echo "PORT_MANIFEST=/codegen/agentic_port/manifests/${{ inputs.op || 'pad' }}.yaml" >> $GITHUB_ENV
      echo "PORT_LIMIT=${{ inputs['perf-limit'] || '24' }}" >> $GITHUB_ENV

  # One container for the whole run, holding the toolchain, the runtime and the card. Keeping it
  # alive across steps is what makes the agent's rebuilds incremental: the build tree, the local
  # ccache and the CMake configure all persist between tool calls.
  #
  # `--network host` is load-bearing, not tidiness. The remote ccache lives at a Kubernetes service
  # name (`garage.garage.svc.cluster.local`) that resolves on the runner host; a container on the
  # default bridge network does not reliably inherit that resolution, every object misses, and the
  # build goes from minutes to over an hour. Sharing the host network namespace gives the container
  # exactly the DNS and routing the host has.
  - name: Start the toolchain container
    run: |
      docker pull "${{ inputs['docker-image'] || 'ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest' }}"
      docker run -d --name portdev \
        --device /dev/tenstorrent \
        --group-add 1457 \
        --network host \
        -v "${{ github.workspace }}:/work" \
        -v "${{ github.workspace }}/.codegen:/codegen" \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -e TT_METAL_HOME=/work \
        -e PYTHONPATH=/work:/work/ttnn:/work/tools:/codegen \
        -e LD_LIBRARY_PATH=/work/build/lib \
        -e ARCH_NAME=wormhole_b0 \
        -e TRACY_NO_INVARIANT_CHECK=1 \
        -e TRACY_NO_ISA_EXTENSIONS=1 \
        -e CCACHE_TEMPDIR=/tmp/ccache \
        -e CCACHE_BASEDIR=/work \
        -w /work \
        "${{ inputs['docker-image'] || 'ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest' }}" sleep infinity
      docker exec portdev git config --global --add safe.directory /work
      docker exec portdev git config --global --add safe.directory /codegen
      docker exec portdev bash -c 'ls /dev/tenstorrent && echo "card visible in container"'

  # Scaffold before building, not after: this registers the codegen sources in CMake once, so
  # every rebuild the agent triggers is a plain incremental compile with no re-configure.
  - name: Scaffold the port
    run: |
      docker exec portdev python3 .github/scripts/port/scaffold.py \
        --op "${{ inputs.op || 'pad' }}" --category "${{ inputs.category || 'data_movement' }}" \
        --ttmetal-home /work --codegen-root /codegen

  # Diagnostic, and deliberately allowed to fail. Run 31218783110 built with credentials present
  # and still reported `Errors: 1395` against remote storage -- every request failed, which is a
  # transport problem, not the cache-key mismatch it was first read as. `garage.…svc.cluster.local`
  # is cluster-internal DNS and this runner pool may sit outside that cluster, which would also
  # explain why the 2.6-minute warm build only ever reproduced on the in-cluster `civ2` pool.
  # Resolving and connecting here records the answer next to the build instead of leaving it to be
  # inferred from hit rates.
  - name: Probe the remote ccache endpoint
    continue-on-error: true
    run: |
      docker exec portdev getent hosts garage.garage.svc.cluster.local
      docker exec portdev curl -sS -m 10 -o /dev/null \
        -w 'ccache endpoint: HTTP %{http_code} in %{time_total}s\n' \
        http://garage.garage.svc.cluster.local:3900/

  # The remote ccache credentials are bound to this step alone and never written to $GITHUB_ENV.
  # Only this full build needs them; the agent's rebuilds recompile three translation units
  # against an already-warm tree, so leaking shared-cache write access into the agent's
  # environment would buy nothing and risk poisoning the cache for every other build in CI.
  - name: Build tt-metal
    timeout-minutes: 150
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.GARAGE_S3_ACCESS_KEY }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.GARAGE_S3_SECRET_KEY }}
    run: |
      if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "::warning::Garage S3 credentials missing; this build will be cold."
      fi
      docker exec portdev ccache -z || true
      # Tracy is on by default and there is no --enable-profiler flag; passing one makes
      # build_metal.sh exit immediately on an unknown argument. The build is deliberately the last
      # command so its status is the step's -- a trailing `ccache -sv || true` masked a failed
      # build behind a green step on the first shakedown run.
      # CCACHE_REMOTE_ONLY is off: while the remote is erroring on every request, it would leave
      # ccache doing nothing at all, and local storage still pays for itself across the rebuilds
      # the agent triggers within this job. CCACHE_LOGFILE is what turns the next remote failure
      # into a readable cause rather than an error count.
      docker exec \
        -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
        -e AWS_DEFAULT_REGION=garage \
        -e CCACHE_COMPRESS=true -e CCACHE_LOGFILE=/tmp/ccache.log \
        -e "CCACHE_REMOTE_STORAGE=s3://ccache|region=garage|prefix=tt-metal|endpoint_url=http://garage.garage.svc.cluster.local:3900" \
        portdev ./build_metal.sh --build-dir build --build-type Release --enable-ccache

  - name: ccache summary
    if: always()
    continue-on-error: true
    run: |
      docker exec portdev ccache -sv || true
      # The statistics report a remote error count but never the reason; the log has it.
      docker exec portdev bash -c 'grep -m 20 -i "remote\|s3\|http" /tmp/ccache.log || echo "no remote entries in the ccache log"' || true

  # The dev image already carries a working environment with ttnn and torch; `create_venv.sh`
  # refuses to run because it will not clobber the existing /opt/venv, and it is not needed. The
  # one real gap is graphviz, which `ttnn.graph` imports unconditionally at `import ttnn`.
  # Everything the harness needs is asserted here rather than discovered later, so a missing
  # dependency fails this step instead of surfacing as a mysterious `blocked` verdict.
  - name: Prepare the Python environment
    timeout-minutes: 15
    run: |
      # The image's /opt/venv has no pip -- tt-metal provisions with uv, so install with uv and
      # only fall back to pip if this image ever changes.
      docker exec portdev bash -c '
        set -e
        if command -v uv >/dev/null; then
          uv pip install --python "$(command -v python3)" graphviz pyyaml
        else
          python3 -m pip install --quiet graphviz pyyaml
        fi
      '
      docker exec portdev python3 -c 'import ttnn, torch, yaml, graphviz; print("harness imports OK")'

  # This exists to prove the harness and the card work before the agent starts, so that a later
  # failure is unambiguously the agent's code. The ported leg is still an empty stub, so gate.py
  # will report a failing verdict and exit non-zero -- that is expected and is not what this step
  # grades. What it grades is whether the harness ran at all: a `blocked` verdict, an empty ledger,
  # or missing device attribution all mean the numbers the agent later produces would be
  # meaningless. Run 31203909328 sailed through this step green with an unbuilt tree because it was
  # `continue-on-error` and swallowed everything, which is exactly the failure being closed here.
  - name: Native baseline
    run: |
      docker exec portdev python3 .github/scripts/port/gate.py \
        --op "${{ inputs.op || 'pad' }}" --manifest "/codegen/agentic_port/manifests/${{ inputs.op || 'pad' }}.yaml" \
        --category "${{ inputs.category || 'data_movement' }}" --band performance --repo /work \
        --work /tmp/port-baseline --limit "${{ inputs['perf-limit'] || '24' }}" --skip-write-check \
        > /tmp/baseline.json || true
      cat /tmp/baseline.json

      python3 - /tmp/baseline.json <<'PY'
      import json, sys

      try:
          report = json.loads(open(sys.argv[1]).read() or "{}")
      except json.JSONDecodeError as exc:
          sys.exit(f"the baseline produced no parseable verdict ({exc}); the harness did not run")

      verdict = report.get("verdict") or report.get("performance", {}).get("verdict")
      if verdict in (None, "blocked"):
          sys.exit(f"baseline verdict is {verdict!r}: {report.get('error', 'see the report above')}")
      if not report.get("ledger_counts", {}).get("in"):
          sys.exit("the ledger produced no in-scope cases, so there is nothing to measure")

      # The op-code note is the only positive evidence that the native and generic legs really
      # dispatched on the card and that profiler rows lined up with them. Without it the device
      # band is inconclusive, and it will still be inconclusive once the agent is done.
      codes = next((n for n in report.get("notes", []) if n.startswith("op codes:")), None)
      if codes is None or "native=" not in codes:
          sys.exit("no device attribution for the native leg; " + "; ".join(report.get("notes", [])))
      print(f"baseline harness OK -- {codes}, verdict {verdict!r} (a failing verdict is expected here)")
      PY

# Nothing removed `portdev` before this, so every failed run left a container holding
# /dev/tenstorrent and the host network namespace on a shared VM -- four of them by run
# 31218783110. That is the same kind of debris that broke that run, where the leftover was gh-aw's
# own gateway container. This is also H8 of the threat model, specified and never implemented:
# reset the card so a wedged device becomes this job's problem rather than the next job's.
post-steps:
  - name: Release the card and remove the toolchain container
    if: always()
    run: |
      docker exec portdev tt-smi -r || tt-smi -r || echo "no tt-smi on the host or in the image; card not reset"
      docker rm -f portdev || true

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
      docker exec portdev bash -c 'cmake --build build --target install -j"$(nproc)" 2>&1 | tail -150; exit "${PIPESTATUS[0]}"'
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
      # The schema has no enum, so the allowlist lives here. `band` is the only agent-supplied
      # value in this file, and matching it against literals before use means nothing the agent
      # writes ever reaches a shell as anything but one of these three words.
      case "$INPUT_BAND" in
        correctness|performance|both) BAND="$INPUT_BAND" ;;
        *) echo "verify: band must be correctness, performance, or both (got: $INPUT_BAND)" >&2; exit 2 ;;
      esac
      docker exec portdev python3 .github/scripts/port/gate.py \
        --op "$PORT_OP" --manifest "$PORT_MANIFEST" --category "$PORT_CATEGORY" \
        --band "$BAND" --repo /work --base-sha "$PORT_BASE_SHA" --limit "$PORT_LIMIT"

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

# Port a codegen op to a C++ program factory

You are porting the tt-dm-codegen generic_op **`${{ inputs.op || 'pad' }}`** into tt-metal as a native C++
`DeviceOperation`, and proving on silicon that it beats the existing implementation.

The tree is already scaffolded and built. The kernels are copied, the three `codegen/` source files
exist as empty stubs, they are registered in CMake, and a native baseline has been measured. Your
job is to fill in those stubs and wire the routing.

## Where things are

All paths below are relative to your working directory. The build and measurement tools run inside
a container where these same files appear under `/work` and `/codegen`, so a path in a compiler
diagnostic will be prefixed differently from the path you edit. They are the same file.

| What | Where you edit it |
| --- | --- |
| tt-metal checkout (your working tree) | the workspace root |
| The op you are porting | `ttnn/cpp/ttnn/operations/${{ inputs.category || 'data_movement' }}/${{ inputs.op || 'pad' }}/` |
| Your stubs to fill in | `.../${{ inputs.op || 'pad' }}/codegen/` |
| tt-dm-codegen (read-only) | `.codegen/` |
| The manifest — source of truth | `.codegen/agentic_port/manifests/${{ inputs.op || 'pad' }}.yaml` |
| The porting guide — read this first | `.codegen/agentic_port/knowledge/porting-guide.md` |
| The generator you are transliterating | the manifest's `codegen_builder` path, under `.codegen/` |
| **A finished, merged port to imitate** | `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/` |

Read the manifest and the porting guide in full before you write anything. The merged `repeat`
port is the single most useful thing here: it is a complete, reviewed example of exactly the
deliverable you are producing, including its routing in `repeat/repeat.cpp`. Follow its structure.

## What to write

**A. Program factory.** Transliterate the manifest's `codegen_builder` into a declarative
`create_descriptor` in `codegen/${{ inputs.op || 'pad' }}_codegen_program_factory.cpp`, per the porting
guide's factory-translation section. Wire real circular-buffer and kernel args from the manifest's
`cache_key_fields`, not placeholders.

Every literal you write — CB sizing, work-split math, compile-time args, routing constants — must
trace to a manifest field, a builder constant, a device query, or a tensor property. A number that
merely happens to make one case pass is a defect even when the case passes.

**B. Routing.** All four pieces, in the shared files, not in new ones:

1. `supported_by_codegen()` in `codegen/${{ inputs.op || 'pad' }}_codegen_supported.{hpp,cpp}`, transcribed
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

3. The host free function `ttnn::${{ inputs.op || 'pad' }}(..., implementation=)` in the shared
   `${{ inputs.op || 'pad' }}.cpp`:
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

4. `prim::${{ inputs.op || 'pad' }}_codegen` validate, in
   `codegen/${{ inputs.op || 'pad' }}_codegen_device_operation.cpp`: a second `TT_FATAL(supported_by_codegen(...))`
   on cache miss, plus **the native op's own structural `TT_FATAL`s** — device storage, non-null
   buffer, rank and alignment invariants. `supported_by_codegen()` only asks about layout, dtype and
   memory config, all of which answer perfectly well for a host-side or deallocated tensor, so a
   scope-gate-only validate lets one through and the first symptom is a null-buffer dereference
   where native gave a clear error.

5. The Python binding: expose the `implementation` kwarg on the **existing** binding in
   `${{ inputs.op || 'pad' }}_nanobind.cpp`, defaulting to `"auto"`.

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
