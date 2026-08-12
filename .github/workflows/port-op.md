---
description: |
  Port a tt-dm-codegen generic_op into tt-metal as a C++ program factory, proven on a card.

  An op that exists only as a tt-dm-codegen generic_op already has a working kernel and a measured
  device-time win, but it pays a per-call program-descriptor rebuild that eats the win at wall
  clock. Porting it means transliterating the generator's builder into a cached DeviceOperation and
  wiring the native/codegen routing, then proving on silicon that the win survives.

  Two runners, because no single one can do both halves. This job holds the agent and runs on a
  GitHub-hosted runner, the only place `api.githubcopilot.com` is reachable. It has no compiler
  worth using and no card, so every build and every measurement is dispatched into CIv2 as a
  `port-measure.yaml` run and waited on. The agent writes C++ and calls two tools -- `build` and
  `verify` -- which are launchers over that dispatch, then opens a draft PR carrying the verdict it
  reached. It never picks the cases, the thresholds, or the measurement method, and `verify`
  refuses to run at all if the working tree has changed outside the port's own files or the
  generated test has drifted.

  V1 scope: one op, one arch (N150), `workflow_dispatch` only.

on:
  # Prototype shakedown only. `workflow_dispatch` resolves against the default branch, so this
  # workflow cannot be started that way until it lands on main -- the push trigger is how v1 gets
  # exercised before then. Remove it when this merges; every input falls back to its default on a
  # push, which is why the defaults are duplicated inline throughout.
  #
  # Note that `port-measure.yaml`, which this job dispatches to, needs no such accommodation and no
  # merge either: it triggers on the push of the scratch ref, and a push event runs the workflow
  # file from the pushed commit.
  push:
    branches: [ebanerjee/port-op-dryrun]
  workflow_dispatch:
    inputs:
      op:
        description: "Op to port; must have a manifest at agentic_port/manifests/<op>.yaml"
        required: true
        type: string
        default: untilize
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
      perf-limit:
        description: "Number of in-scope cases to measure per performance band"
        required: false
        type: string
        default: "24"
      runner-label:
        description: "CIv2 pool the dispatched build and measurement land on"
        required: false
        type: string
        default: '["tt-ubuntu-2204-N150-viommu-stable"]'

# Unchanged, and that is the point. The launcher has to push a scratch ref -- the working tree
# cannot travel inline, since the `pad` port alone was 118 KB -- but that authority does not come
# from this job's token. It comes from a PAT placed by the pre-step below, so the agent job stays
# read-only and gh-aw's strict mode, which rejects *any* write permission here, stays on.
#
# A PAT rather than a broader `GITHUB_TOKEN`, and not for scope: GitHub does not let pushes made
# with `GITHUB_TOKEN` start workflow runs, and the push is what starts the measurement.
permissions:
  contents: read
  copilot-requests: write

# The MCP gateway cancels a tool call after its own deadline, independent of the per-tool `timeout`
# values further down -- those bound the handler process, this bounds the request. It defaults to 60
# seconds, so `build` died at 1m00s four times in a row on the first real run, each retry starting
# another CIv2 build that then ran on unwatched.
#
# 10m is the ceiling gh-aw allows, not a chosen number. It is enough for `build`, which measured 8.5
# minutes, though not with much room once the pool is busy. It is nowhere near `verify`, which is a
# build plus the bands. Making `verify` work at all means the launcher cannot block: it has to start
# the run, return a handle, and be polled. Until that lands, treat `verify` as known-broken rather
# than merely slow.
engine:
  id: copilot
  mcp:
    tool-timeout: 10m

# The agent firewall's API proxy meters AI credits on every request, and it prices them from a table
# keyed by model. `engine: copilot` with no model pinned resolves to `auto`, which is not in that
# table, so the proxy rejected every request and the agent made no changes at all:
#
#   400 Model "auto" has no AI credits pricing and no default pricing is configured.
#
# It retried three times and gave up in 43 seconds. Nothing to do with this runner pool -- it would
# fail the same way anywhere -- but it is the last thing between a working CIv2 job and an agent that
# actually ports the op.
#
# A fallback rate rather than pinning a priced model, because pinning one would change which model
# does the work, and the point of this run is to test the harness rather than to re-tune the agent.
# The rates are the ones the proxy's own error message suggests, in dollars per million tokens; they
# affect credit accounting only, never behaviour, and are worth revisiting if that accounting has to
# be accurate for a model `auto` chose.
models:
  default-ai-credits-pricing:
    input: 3.0
    output: 15.0

network: defaults

# One port at a time per branch: two concurrent runs would fight over the same card and produce
# timings that are noise rather than measurement. Keyed on the ref rather than the op because
# workflow-level fields are evaluated before a push event has any inputs to read.
concurrency:
  group: "gh-aw-port-op-${{ github.ref_name }}"

# Against the 360-minute ceiling on a hosted runner, leaving room for the post-steps to still run.
# The budget is the real constraint on this design: a compile check is a build, call it 20 minutes
# with queue, and a full verify adds the device job for 35-45. That is six to eight cycles for the
# whole port, and the `pad` run spent six on performance re-checks alone -- which is what the
# batching instruction in the prompt is defending.
timeout-minutes: 350

# Not CIv2, though every build and measurement lands there. CIv2 has no direct egress; it leaves
# through `proxy.restricted-proxy.svc.cluster.local`, whose allowlist does not carry
# `api.githubcopilot.com`, so the agent process starts in-cluster and then fails every model request
# with a 403 from squid. A hosted runner is outside the cluster and reaches the API natively. It
# cannot build tt-metal -- 4 cores, and a probe watched the rate collapse from 145 targets a minute
# to 10-15 as it crossed into the ttnn unity builds, which is exactly the part an edit to an op
# touches -- so it builds nothing at all. See `port-measure.yaml`, which does.
runs-on: ubuntu-latest

pre-steps:
  # The credential the launcher pushes and dispatches with, placed where the agent cannot read it.
  #
  # The obvious way to do this is wrong, and nothing warns you at runtime. A `${{ secrets.X }}`
  # inside an `mcp-scripts` body is interpolated **verbatim** into the generated handler under
  # `${RUNNER_TEMP}/gh-aw/mcp-scripts/`, and awf mounts that whole directory into the sandbox
  # read-only -- so the agent reads the secret with `cat`. The environment is no better: awf runs
  # with `--env-all` minus a fixed five-name exclusion list, which means anything any step wrote to
  # `$GITHUB_ENV` is inside the sandbox too.
  #
  # Both halves of this step are what avoid that. `pre-steps` are ordinary workflow steps, so a
  # step-level `env:` here stays scoped to this step and never reaches the agent step's environment;
  # and `$HOME` is in none of awf's mounts, which are `${RUNNER_TEMP}/gh-aw`, the workspace, the
  # tool cache and `/tmp/gh-aw`. HANDOFF.md records how that was established, and it is worth
  # re-checking if gh-aw's generator ever changes.
  #
  # `CODEGEN_REPO_TOKEN` is the fallback because it is what exists today: a classic PAT with `repo`,
  # already SSO-authorised for the org, already used by the measure job to check out the generator.
  # It is also far broader than this needs -- `repo` reaches every repository its owner can, not
  # just this one -- so `PORT_PUSH_TOKEN` is preferred whenever someone provisions it, and a
  # fine-grained token scoped to tt-metal with `contents: write` and `actions: read` is all the
  # launcher actually uses. The breadth is why dispatch.py refuses to push a snapshot that touches
  # `.github/`: see the comment there, the risk is specific and it is not about the push.
  - name: Place the dispatch credential outside the sandbox
    env:
      PORT_DISPATCH_TOKEN: ${{ secrets.PORT_PUSH_TOKEN || secrets.CODEGEN_REPO_TOKEN }}
    run: |
      if [ -z "$PORT_DISPATCH_TOKEN" ]; then
        echo "::error::Neither PORT_PUSH_TOKEN nor CODEGEN_REPO_TOKEN is set. The launcher cannot push a scratch ref without one, so build and verify would both fail on the agent's first tool call."
        exit 1
      fi
      install -d -m 700 "$HOME/.port-dispatch"
      umask 077
      printf '%s' "$PORT_DISPATCH_TOKEN" > "$HOME/.port-dispatch/token"
      echo "credential placed, $(wc -c < "$HOME/.port-dispatch/token") bytes"

steps:
  # Checked out at the workspace root, not a subdirectory: gh-aw's own later steps assume the
  # workspace root is the repository, and a subdirectory checkout fails them.
  - name: Checkout tt-metal
    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
    with:
      # Shallow: the only history this run needs is the base commit the write-path guard diffs
      # against, and that is HEAD.
      fetch-depth: 1
      # No submodules. Nothing is built here, and the launcher pushes a commit whose parent is the
      # checked-out HEAD, so the submodule gitlinks come along unchanged from the parent commit --
      # the CIv2 side then checks out `submodules: recursive` against them and builds. Cloning them
      # here would cost several minutes per run to produce a tree nothing reads.
      submodules: false
      # The agent has no business holding a push token, and the launcher does not use this one:
      # it authenticates from the file the pre-step wrote, outside the sandbox.
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
      # The generator tree is a checkout inside the workspace, so without this every one of its
      # files reads as an untracked file: the launcher's `git add -A` would sweep the whole thing
      # into the scratch commit, and gate.py's write-path guard, which counts untracked files,
      # would refuse to measure anything.
      echo ".codegen/" >> .git/info/exclude
      {
        echo "PORT_BASE_SHA=$(git rev-parse HEAD)"
        echo "PORT_OP=${{ inputs.op || 'untilize' }}"
        echo "PORT_CATEGORY=${{ inputs.category || 'data_movement' }}"
        echo "PORT_CODEGEN_REF=${{ inputs['codegen-ref'] || 'codegen_agentic_port' }}"
        echo "PORT_LIMIT=${{ inputs['perf-limit'] || '24' }}"
        echo "PORT_RUNNER_LABEL=${{ inputs['runner-label'] || '["tt-ubuntu-2204-N150-viommu-stable"]' }}"
      } >> $GITHUB_ENV

  # `scaffold.py` imports `yaml` and `ttnn_names`, whose own `import ttnn` sits inside a function
  # that the scaffold pass never calls. So the whole pass runs here with PyYAML and nothing else --
  # no container, no build, no card. Only `--emit-test-only` needs a real ttnn, and that is why it
  # moved to the baseline dispatch.
  - name: Install what the scaffold pass needs
    run: python3 -m pip install --quiet pyyaml

  # Scaffolding before the first build is what makes every later build incremental: the codegen
  # sources are registered in CMake once, here, so a rebuild is a compile rather than a reconfigure.
  # That mattered more when builds were local and incremental; it still matters because a CMake
  # reconfigure inside `build-artifact.yaml` would be a change to a file the write-path guard
  # forbids the agent from touching.
  - name: Scaffold the port
    run: |
      python3 .github/scripts/port/scaffold.py \
        --op "$PORT_OP" --category "$PORT_CATEGORY" \
        --ttmetal-home . --codegen-root .codegen

      dir="ttnn/cpp/ttnn/operations/${PORT_CATEGORY}/${PORT_OP}/codegen"
      if [ ! -d "$dir" ]; then
        echo "::error::scaffold.py reported success but $dir does not exist"
        exit 1
      fi
      echo "scaffolded $(find "$dir" -type f | wc -l) stub files"

  # Everything the agent needs that only a card can produce, fetched before a single credit is
  # spent: the ledger, the coverage report, a golden checked against native, a native baseline that
  # proves the harness runs at all, and the generated routing test, which is part of the deliverable
  # and cannot be rendered here because rendering it expands the generator's sweep and imports ttnn.
  #
  # It is also the cheapest possible test of the dispatch path itself. A broken launcher, a missing
  # credential or a `port-measure.yaml` that is not on the default branch all fail here, in the one
  # place where the failure is unambiguous and the agent has not yet been started.
  #
  # No `timeout-minutes` here even though it wants one: gh-aw drops the field from custom steps. The
  # launcher's own ceiling is what bounds this.
  - name: Measure the native baseline on CIv2
    run: python3 .github/scripts/port/dispatch.py --mode baseline

post-steps:
  # Run 31406186048 wrote a correct port -- 184/184 cases bit-exact -- and threw it away. The
  # performance verdict never reached `win`, the agent correctly declined to open a PR, and the C++
  # existed nowhere but this workspace. The `agent` artifact holds only the transcript, and the
  # transcript truncates file bodies (`56 lines...`), so nothing was recoverable: a 20-minute build,
  # eleven `verify` calls and an hour of card time produced a log message. This runs before the
  # verdict is known and regardless of it, because the run that most needs its work preserved is
  # exactly the run that is not going to open a PR.
  - name: Preserve the port whatever the verdict
    if: always()
    run: |
      mkdir -p /tmp/port-artifact
      dir="ttnn/cpp/ttnn/operations/${PORT_CATEGORY}/${PORT_OP}"
      # The scaffolded codegen tree is untracked, so a diff would omit the very files the agent was
      # asked to fill in. Archive the op directory outright, and keep a diff for the few tracked
      # files the port may touch (sources.cmake, CMakeLists.txt, the op's own .cpp/.hpp).
      tar -czf /tmp/port-artifact/port-tree.tar.gz "$dir" || echo "::warning::no $dir to archive"
      git diff "${PORT_BASE_SHA}" -- . > /tmp/port-artifact/tracked.diff || true
      git status --porcelain > /tmp/port-artifact/status.txt || true
      # The generated routing test is untracked and lives outside the op directory, so neither the
      # archive above nor the diff would carry it. It is part of the deliverable.
      routing="tests/ttnn/nightly/unit_tests/operations/${PORT_CATEGORY}/test_${PORT_OP}_codegen_routing.py"
      cp "$routing" /tmp/port-artifact/ 2>/dev/null || echo "::warning::no generated routing test at $routing"
      # What each dispatch brought back. The per-case measurements behind every verdict live in the
      # results artifacts of the measure runs, which are separate runs nobody will think to look up;
      # copying them here puts the whole history of the port in one place. They are also the only
      # record of which cases were marginal and by how much, which is what judging the noise floor
      # needs after the fact -- the transcript truncates them.
      cp -r "$HOME"/.port-dispatch/results-* /tmp/port-artifact/ 2>/dev/null \
        || echo "::warning::no dispatch results to copy"
      ls -l /tmp/port-artifact/

  - name: Upload the port
    if: always()
    continue-on-error: true
    uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
    with:
      name: port-${{ env.PORT_OP }}-${{ github.run_id }}-${{ github.run_attempt }}
      path: /tmp/port-artifact
      if-no-files-found: warn

  # A hosted runner is destroyed with the job, so this is not about the machine. It is about the
  # repository: a dispatch that dies between pushing and its own cleanup leaves a scratch branch
  # behind, and nothing else would ever remove it.
  - name: Delete any scratch refs a dispatch left behind
    if: always()
    env:
      PORT_DISPATCH_TOKEN: ${{ secrets.PORT_PUSH_TOKEN || secrets.CODEGEN_REPO_TOKEN }}
    run: |
      [ -n "$PORT_DISPATCH_TOKEN" ] || exit 0
      shred -u "$HOME/.port-dispatch/token" 2>/dev/null || rm -f "$HOME/.port-dispatch/token"
      remote="https://x-access-token:${PORT_DISPATCH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
      # Scoped to this run's refs, not to the prefix: another port may legitimately be in flight.
      # The launcher names them "<op>-<mode>-<run id>-<uuid>", hence the leading wildcard.
      leftovers=$(git ls-remote --heads "$remote" "refs/heads/port-op-scratch/*-${GITHUB_RUN_ID}-*" \
        | awk '{print $2}')
      for ref in $leftovers; do
        echo "removing leftover $ref"
        git push "$remote" ":$ref" >/dev/null 2>&1 || echo "::warning::could not delete $ref"
      done

tools:
  edit:
  bash: [":*"]

# All three tools are the same checked-in launcher, which pushes the working tree to a scratch ref
# -- which is itself what starts `port-measure.yaml` on CIv2. No agent-supplied text is interpolated
# into a shell command; `band` is a closed enum, checked against literals below, and checked a
# second time on the far side because the commit message is the transport.
#
# Starting and collecting are separate tools because they have to be. The MCP gateway cancels a tool
# call at 600s and gh-aw will not compile a higher ceiling, which is shorter than a build and a
# fraction of a verify. So `build` and `verify` return as soon as the run exists, and `wait` collects
# it in bounded slices. The per-tool timeouts below sit under the gateway's, so the launcher stops
# itself rather than being killed mid-flight -- being killed is what stranded four builds on CIv2 on
# the first real run, each holding a card that nobody was left to read.
#
# `--as-tool` is not optional and is the other half of that lesson. gh-aw runs these handlers through
# `execFile` and rejects the promise on a non-zero exit, so the agent is shown `Command failed:
# wait.sh (exit code: 4)` with the actual answer demoted below it. Twice now the agent has read a
# delivered answer as a broken tool and called the tool again -- once against the gateway timeout,
# once against a compiler diagnostic and a `wait` that had only said "not finished yet". Under the
# flag the exit code says nothing and the text says everything; a genuine harness failure still
# exits non-zero, because that one really is a broken tool.
mcp-scripts:
  build:
    description: >-
      Start a compile of your working tree on CIv2 and return a handle immediately. Returns in under
      a minute; the build itself takes roughly ten. Collect the compiler diagnostics by calling
      `wait` with the handle. Batch every edit you intend to make before calling this.
    timeout: 540
    run: |
      python3 "${GITHUB_WORKSPACE}/.github/scripts/port/dispatch.py" --mode build --start --as-tool
  verify:
    description: >-
      Start grading the port on real hardware and return a handle immediately; collect the verdict
      with `wait`. `correctness` checks bit-exactness against a golden for in-scope cases and native
      fallback for out-of-scope ones; `performance` measures wall clock and device kernel duration
      against native and against the tt-dm-codegen prototype. Refuses to run if the working tree has
      changed outside the port's own files. You cannot change what it measures or where the
      thresholds sit. It builds before it measures, so it takes about forty minutes to come back; you
      can afford a handful of these in a whole run.
    timeout: 540
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
      python3 "${GITHUB_WORKSPACE}/.github/scripts/port/dispatch.py" --mode verify --band "$BAND" --start --as-tool
  wait:
    description: >-
      Collect a build or verify started earlier, given its handle. Blocks for up to seven minutes and
      then returns either the result or a note that the run is still going. That note is not a
      failure and says nothing about your port: call `wait` again with the same handle. Expect two
      calls for a build and five or six for a verify.
    timeout: 540
    inputs:
      handle:
        type: string
        description: The handle returned by build or verify.
    run: |
      # Handles are generated by the launcher and only ever echoed back, but this is agent-supplied
      # text reaching a command line, so it is constrained to the shape the launcher emits rather
      # than trusted. The launcher looks it up in a directory the sandbox cannot write to.
      case "$INPUT_HANDLE" in
        *[!a-zA-Z0-9-]*|"") echo "wait: '$INPUT_HANDLE' is not a handle; pass the one build or verify printed" >&2; exit 2 ;;
      esac
      python3 "${GITHUB_WORKSPACE}/.github/scripts/port/dispatch.py" --wait "$INPUT_HANDLE" --as-tool

safe-outputs:
  mentions: false
  create-pull-request:
    draft: true
    title-prefix: "[codegen-port] "
    labels: [codegen-port, gh-aw]
  # Without a no-op the agent has only one way to end, and an agent with a single available action
  # will take it -- filing a PR for a port it knows does not work. This makes stopping expressible.
  # It is now reserved for a port that is not correct or a harness that could not run: a correct port
  # that lost on performance is a reviewable finding, and the prompt sends that to a draft PR.
  noop:
---

# Port a codegen op to a C++ program factory

You are porting the tt-dm-codegen generic_op **`${{ inputs.op || 'untilize' }}`** into tt-metal as a native C++
`DeviceOperation`, and proving on silicon that it beats the existing implementation.

The tree is already scaffolded. The kernels are copied, the three `codegen/` source files exist as
empty stubs, they are registered in CMake, the routing test has been generated, and a native
baseline has been measured on real hardware. Your job is to fill in those stubs and wire the routing.

## Your tools run somewhere else, and they are slow

There is no compiler and no Tenstorrent card on this machine. `build` and `verify` push your working
tree to a scratch branch and start a job on a cluster runner that has both. **A `build` takes about
ten minutes to come back and a `verify` about forty.** Nothing you can do makes them faster, and the
whole run has a few hours in it, so plan on roughly six to eight of them.

Both return a handle straight away rather than the answer. You collect the answer with `wait`, which
blocks for up to seven minutes and then either gives you the result or tells you the run is still
going. **If it says the run is still going, that is not a failure and it is not about your port —
call `wait` again with the same handle.** A build usually takes two `wait` calls and a verify five or
six. Do not start a second `build` because the first has not come back yet; starting one cancels
whatever was already running, so you would be throwing away the answer you are waiting for.

What that changes about how you work:

- **Finish everything you can before calling `build`.** A compile error in one file and a compile
  error in three files cost exactly the same. Read all the sources, write the whole port, re-read
  what you wrote looking for the obvious mistakes, and only then build. Calling `build` on a file
  you already know is unfinished wastes a fifth of your budget.
- **`verify` is scarce.** Get correctness passing, then measure performance once. Iterating on
  performance is what exhausted the budget in an earlier run of this workflow.
- Never call either tool twice on an unchanged tree. The answer will be the same and you will have
  spent twenty minutes learning that.

Because the build happens elsewhere, a compiler diagnostic names the file as `/work/...` while you
edit it under the workspace root. Same file, different prefix.

## Where things are

All paths below are relative to your working directory.

| What | Where you edit it |
| --- | --- |
| tt-metal checkout (your working tree) | the workspace root |
| The op you are porting | `ttnn/cpp/ttnn/operations/${{ inputs.category || 'data_movement' }}/${{ inputs.op || 'untilize' }}/` |
| Your stubs to fill in | `.../${{ inputs.op || 'untilize' }}/codegen/` |
| tt-dm-codegen (read-only) | `.codegen/` |
| The manifest — source of truth | `.codegen/agentic_port/manifests/${{ inputs.op || 'untilize' }}.yaml` |
| The porting guide — read this first | `.codegen/agentic_port/knowledge/porting-guide.md` |
| The generator you are transliterating | the manifest's `codegen_builder` path, under `.codegen/` |
| **A finished, merged port to imitate** | `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/` |

Read the manifest and the porting guide in full before you write anything. The merged `repeat`
port is the single most useful thing here: it is a complete, reviewed example of exactly the
deliverable you are producing, including its routing in `repeat/repeat.cpp`. Follow its structure.

## What to write

**A. Program factory.** Transliterate the manifest's `codegen_builder` into a declarative
`create_descriptor` in `codegen/${{ inputs.op || 'untilize' }}_codegen_program_factory.cpp`, per the porting
guide's factory-translation section. Wire real circular-buffer and kernel args from the manifest's
`cache_key_fields`, not placeholders.

Every literal you write — CB sizing, work-split math, compile-time args, routing constants — must
trace to a manifest field, a builder constant, a device query, or a tensor property. A number that
merely happens to make one case pass is a defect even when the case passes.

**B. Routing.** All four pieces, in the shared files, not in new ones:

1. `supported_by_codegen()` in `codegen/${{ inputs.op || 'untilize' }}_codegen_supported.{hpp,cpp}`, transcribed
   from the generator sweep's `invalidate_vector` plus any op guards. This predicate is about
   **correctness and device-resource feasibility, never performance**. Two things it must get
   right, both of which the sweep cannot tell you:

   - **Bound every device resource whose footprint scales with a dimension.** For each one your
     factory allocates, the factory must scale down to what the device admits and this predicate
     must reject whatever still does not fit — at **every leaf of your dispatch tree**, since a
     bound written for the default builder leaves the alternates unbounded. The sweep cannot tell
     you about any of this: `invalidate_vector` only ever ran shapes small enough to fit, so a
     predicate derived from it silently claims every size, and `auto` then fails allocation on a
     config the predicate advertised instead of falling back to native. The porting guide works
     this through for circular buffers, which is the instance you will meet first.
   - **Transcribe general conditions, not example shapes.** Each `scope: out` case in the manifest
     is one representative of a general condition stated in its `note`. Write the condition (e.g.
     `logical[-2] % TILE_H != 0`), never an exact-match on the tuple. An exact-match lets every
     other shape meeting the same condition wrongly reach codegen, and it will be rejected on
     review. If you cannot name the general condition from the `note`, read the op-orchestration
     source's own guard and transcribe that.

   It must agree with the manifest: every `scope: in` case returns `true`, every `scope: out` case
   returns `false`.

2. `is_demoted()` in the same files. This is the performance routing gate, consulted **only** by
   the `auto` branch, and it answers a different question from `supported_by_codegen()`: not "can
   codegen serve this?" but "should it?"

   You cannot know the answer in advance, and you must not guess it. Start it returning `false` for
   everything, then let the measurements tell you. `verify`'s performance verdict reports
   `routing.demotion_candidates` — in-scope cases that the port measurably loses on. Each one is a
   choice: make the generated path faster, or demote it. Demoting is legitimate; a configuration the
   generated kernel is genuinely worse at should route to native, and a case you demote is not
   graded.

   Two constraints on taking that option. **Write the general condition, not the case list** — same
   standard as `supported_by_codegen()`, and for the same reason: a predicate that exact-matches the
   measured tuples leaves every neighbouring shape with the same problem routed to codegen. Name the
   property that makes those cases slow. And **demotion is capped**: past
   `routing.demotion_cap` of the measured in-scope cases the verdict becomes `not-a-candidate`,
   because a port serving a minority of its own declared scope is not worth maintaining. If you find
   yourself demoting to get to a `win`, the answer is that this port is not a win.

   `routing.demoted_but_faster` is the opposite mistake: a case you routed away that measured faster
   under forced codegen. Those demotions cost performance for nothing; remove them.

3. The host free function `ttnn::${{ inputs.op || 'untilize' }}(..., implementation=)` in the shared
   `${{ inputs.op || 'untilize' }}.cpp`:
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

4. `prim::${{ inputs.op || 'untilize' }}_codegen` validate, in
   `codegen/${{ inputs.op || 'untilize' }}_codegen_device_operation.cpp`: a second `TT_FATAL(supported_by_codegen(...))`
   on cache miss, plus **the native op's own structural `TT_FATAL`s** — device storage, non-null
   buffer, rank and alignment invariants. `supported_by_codegen()` only asks about layout, dtype and
   memory config, all of which answer perfectly well for a host-side or deallocated tensor, so a
   scope-gate-only validate lets one through and the first symptom is a null-buffer dereference
   where native gave a clear error.

5. The Python binding: expose the `implementation` kwarg on the **existing** binding in
   `${{ inputs.op || 'untilize' }}_nanobind.cpp`, defaulting to `"auto"`.

**C. Remaining hooks.** Fill in `compute_output_specs`, `create_output_tensors`, and any other
DeviceOp hooks per the porting guide's checklist for this op's tier.

## How to iterate

1. Read everything first: the manifest, the porting guide, the generator's builder, the `repeat`
   port. This costs you nothing and every mistake it prevents costs twenty minutes.
2. Write the whole port — factory, all four routing pieces, the remaining hooks. Then re-read it.
3. Call **`build`**, then **`wait`** until it returns. Fix everything it reports in one pass, then
   build again.
4. Call **`verify`** with `band: correctness`, then **`wait`** until it returns. Fix real defects
   until it passes.
5. Call **`verify`** with `band: performance`, then **`wait`** until it returns.
6. Open the draft PR once correctness is a full pass and you have stopped improving performance,
   whatever the performance verdict turned out to be.

Read the verdict rather than pattern-matching on it:

- **`win`** — all gates pass and something is genuinely faster.
- **`back-to-translate`** — a gate failed. `failing` names the cases that failed on their own, and
  `routing.demotion_candidates` names every case implicated, including the ones that failed only as
  part of a class. Go fix the code, or route those cases away — see `is_demoted()` above.
- **`not-a-candidate`** — either every gate held and nothing got faster, or the port demoted more of
  its scope than the cap allows. Check `routing.demoted_fraction` to see which.
- **`blocked`** — the harness could not run, or the tree changed outside the port's own files. Read
  the error. If it is the write-path guard, revert the stray edit; do not try to work around it.

A tool can also fail without producing a verdict at all — the dispatch could not complete, the
remote job died, no results came back. That is infrastructure, not your port. Say so and stop rather
than rewriting working code to appease it.

**A verdict short of `win` is not a reason to stop without a pull request.** See below.

**Do not re-run `verify` on unchanged code.** It costs forty minutes of your budget per call, and the
wall-clock noise that once made repeated calls look informative is now absorbed by the gate itself: a
case under the tie band is recorded as `marginal` instead of failing, and only a case below the noise
floor, or marginals concentrated in one class, refuses the port. So `back-to-translate` means the code
needs changing, not that the measurement was unlucky. Read `failing`,
`summary.wall_failing_strata` and `routing.demotion_candidates`, and fix what they point at.

## Rules

**Do not go looking for a prior implementation of this op.** No `git log`, `git show`, or `git
branch` spelunking, no diffing against other branches, no reading a port from another checkout.
Anything you find that way is stale scratch work, not a reference, and using it invalidates this
run. Build the port from the manifest, the generator source, the porting guide, the merged `repeat`
port, and nothing else.

**Do not write tests, and do not edit the generated one.** No new `test_*.py`, no ad-hoc device
scripts, no edits to any existing test, harness, or gate script. Correctness and performance are
measured for you by `verify`, against cases derived from the generator's own sweep. `verify`
recomputes the diff from the base commit every time it runs and refuses if anything outside the
port's own files changed, so editing the harness does not get you a passing grade — it gets you a
`blocked` verdict.

One test does ship with your port, and it is generated:
`tests/ttnn/nightly/unit_tests/operations/<category>/test_<op>_codegen_routing.py`. It was rendered
from the coverage ledger during the baseline measurement and is already in your working tree, so it
is part of your deliverable — include it in the pull request and describe what it covers. It asserts
that every out-of-scope case falls back to native under `auto`, which is the same thing `verify`'s
correctness band checks, so if that band passes, this test passes.

`verify` re-renders the file and compares it on every call; a mismatch is a `blocked` verdict. Do not
edit it, and do not try to regenerate it here — rendering it needs a real ttnn and a card, neither of
which this machine has. If it goes missing or drifts, that is a harness problem to report, not one to
work around.

**Do not touch git history.** No `git push`, no `git commit --amend`, nothing that rewrites
history, and never target `main`. Leave your work as uncommitted edits: that is exactly what `build`
and `verify` snapshot and send to the machine that runs them, and it is what the pull-request output
turns into a branch. Committing here does not break them, but it buys nothing and it is one more
thing that can go wrong.

**Do not invent numbers.** Every performance figure you report must come from a `verify` verdict.
Do not estimate, extrapolate, or restate a ratio you did not measure.

## The pull request

**Open one whenever the correctness band is a full pass.** Correct code that compiles and routes
properly is reviewable work regardless of what the performance verdict says, and a measured negative
result is a finding worth keeping rather than a reason to throw the port away. Run 31406186048 wrote
184 out of 184 bit-exact cases, never reached `win`, opened nothing, and left nothing behind but a log
line. That is the outcome this rule exists to prevent.

Use the no-op output only when correctness fails or the verdict is `blocked` — there is nothing to
review in a port that does not work.

Put the performance verdict in the title after the op name, in brackets: `[win]`,
`[not-a-candidate]`, or `[back-to-translate]`. A reviewer must not have to read the body to find out
whether the thing is faster.

Open it as a draft, describing:

- **What changed** — the file list and the shape of the port, including the generated routing test.
- **Why this is, or is not, faster** — the mechanism, grounded in source. Name the specific native
  cost that is avoided and the generated design choice that avoids it. Do not claim causality you
  cannot point at, and do not write "codegen is faster." A verdict short of `win` gets the same
  treatment in reverse: name the cost the generated path pays that native does not.
- **Measured performance** — a table straight from the `verify` verdict: per-configuration wall
  ratio, device-vs-native, and device-vs-prototype, plus the case counts by scope.
- **What was measured** — from `performance.coverage`, one row per stratum: the class, how many
  ledger cases it holds (`cases_in_ledger`), how many were measured, and its worst wall and
  device ratios. Then state `coverage.axes` and, if `coverage.axes_dropped` is non-empty, say which
  parameters the sample does *not* resolve across. If `coverage.complete` is false, name the
  unmeasured classes. A ratio quoted without this is a claim about the sample, not about the op.
- **Routing** — the conditions under which `auto` falls back to native, and why each one exists.
  Separate the `supported_by_codegen()` conditions from the `is_demoted()` ones, and for demotion
  give the measurement that motivated each condition. List `routing.demoted` and
  `routing.demoted_fraction`.
- **Known gaps** — anything you could not resolve, and anything the manifest asserts that the
  hardware contradicted. Say so plainly rather than quietly working around it. Include every entry
  in the verdict's `notes`, and `routing.demoted_but_faster` if it is non-empty.

Every number in that body must be a field from a `verify` verdict, and the surrounding prose must be
something the fields support. If a section has no data behind it, write that instead of writing
around it.
