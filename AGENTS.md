# AGENTS.md — instructions for coding agents authoring changes

## Determine whether you are a CI runner

The CI-runner instructions below describe the GitHub Copilot coding-agent job
configured by `.github/workflows/copilot-setup-steps.yml`. Apply them only when
both of these are true:

- `GITHUB_ACTIONS=true` is present in the environment.
- The task is running as that GitHub Copilot coding-agent job, or the runtime
  explicitly identifies itself as that job.

`CI=true`, `CODEX_CI=1`, being an automated coding agent, running remotely, or
running in a container is not enough. If you cannot establish both conditions,
assume you are **not** this CI runner.

Non-CI agents must ignore the entire section below. In particular, do not adopt
its assumptions about the installed toolchain, Docker, ccache, or accelerator
availability. Inspect your actual environment and use any available local or
remote Tenstorrent hardware and task-specific instructions.

## If you are a CI runner

This section is for agents **writing** changes to tt-metal on the GitHub Copilot
coding-agent runner described above.

### Related instruction files

| File | Audience | Purpose |
| --- | --- | --- |
| `AGENTS.md` (this file) | cloud agent | how to author and **verify** a change |
| `.github/copilot-instructions.md` | code review | cross-cutting review criteria |
| `.github/instructions/*.instructions.md` | code review | path-scoped review criteria (all carry `excludeAgent: "cloud-agent"`) |

The review files describe how to critique a PR. They are not a specification
for your own work.

### Your environment

You run on an internal runner. The tt-metal toolchain is **not** installed on
that host — it lives in the CI build image, and you reach it through a wrapper
script. Do not try to install compilers or dependencies; do not run
`install_dependencies.sh`.

What the host does have: `docker`, a checkout with submodules already
initialised, and a shared remote ccache.

### Match the check to the change

| You changed | What to run |
| --- | --- |
| C++, headers, kernels — `.cpp` / `.hpp` under `tt_metal/`, `ttnn/`, `tt_stl/`, `tt-train/` | **Build.** See below. |
| nanobind bindings (the C++ behind the Python API) | **Build** — this is C++. |
| CMake — `CMakeLists.txt`, `sources.cmake`, `cmake/`, `build_metal.sh` | **Build**, or at minimum `--configure-only` if you only need to prove configure still works. |
| Python only | No build. Formatting is enforced by `pre-commit` (black, isort, autoflake). |
| YAML, workflows | No build. `pre-commit` runs yamllint and check-yaml. |
| Docs, markdown, CODEOWNERS | No build. |

`.pre-commit-config.yaml` defines the formatting and lint hooks the repo enforces
(including `clang-format` and `gersemi` for CMake). Run them if available; do not
treat their absence in your environment as a reason to skip the table above.

**If you are unsure whether your change affects the build, build it.**

### Building

If you changed C++ or CMake, compile before opening the PR.

From the repository root:

```bash
.github/scripts/copilot-build.sh
```

That runs `build_metal.sh --enable-ccache` inside the CI build image against
your working tree, and prints a ccache summary when it finishes. Any arguments
you pass go straight through to `build_metal.sh`:

| Command | When |
| --- | --- |
| `.github/scripts/copilot-build.sh` | default — the usual case |
| `… --configure-only` | prove CMake still configures, without compiling (~6 min) |
| `… --build-metal-tests` | you changed something under `tt_metal/` with tests |
| `… --build-ttnn-tests` | likewise for `ttnn/` |
| `… --build-programming-examples` | you touched `tt_metal/programming_examples/` |
| `… --build-tt-train` | you touched `tt-train/` |
| `… -b Debug` | you need assertions to reproduce something |

`--enable-ccache` is always applied for you. Build the narrowest thing that
actually exercises your change; do not reach for `--build-all`.

If the wrapper warns that Garage credentials are missing, you are building
against a cold cache and it will most likely not finish. Say so in the PR
rather than burning the session on it.

### What to do about a build

- **Builds clean** — say so explicitly in the PR description, including the
  exact command you ran.
- **Fails to build** — fix it and rebuild. Do not open the PR and let CI find
  a compile error you could have caught.
- **Did not need a build** (see the table above) — say which check you ran
  instead, e.g. that it is a docs-only change.
- **Genuinely cannot build** (cold cache, docker unavailable, environment
  problem) — open the PR anyway, and state in the description that the change
  is **unverified**, and why.

Do not claim you ran anything you did not run.

### Hardware availability

The Copilot setup introduced with these instructions was designed for a runner
with no Tenstorrent accelerator attached. That is an environment assumption,
not a repository-wide fact. Before ruling out device tests, check for
`/dev/tenstorrent` and for any hardware reservation or remote-execution tools
provided to the task.

If accelerator access is available, use the applicable hardware instructions
and do not claim that silicon verification is unavailable. If it is not
available, device tests, performance measurements, and hardware-dependent
behaviour cannot be checked in this environment; compilation and host-side unit
tests remain in scope.

If a change's correctness depends on device behaviour, say so.

Never state a performance improvement without measurements.

### Scope discipline

- Change the minimum needed to solve the stated issue.
- New source files go in the relevant `sources.cmake`, not into
  `CMakeLists.txt` build structure.
- Adding an external dependency (`find_package`, `CPMAddPackage`,
  `FetchContent_Declare`, a new `third_party/` submodule) requires infra team
  review. If the issue seems to need one, stop and say so in the PR rather than
  adding it.
