# AGENTS.md — instructions for coding agents authoring changes

This file is for agents **writing** changes to tt-metal (GitHub Copilot cloud
agent and equivalents).

It is deliberately separate from the review instructions:

| File | Audience | Purpose |
| --- | --- | --- |
| `AGENTS.md` (this file) | cloud agent | how to author and **verify** a change |
| `.github/copilot-instructions.md` | code review | cross-cutting review criteria |
| `.github/instructions/*.instructions.md` | code review | path-scoped review criteria (all carry `excludeAgent: "cloud-agent"`) |

The review files describe how to critique someone else's PR. Do not treat them
as a specification for your own work.

## Your environment

You run on an internal runner inside the CI build image, so the full toolchain
is already installed — do **not** try to install compilers or dependencies:

- `clang++-20`, `cmake`, `ninja`, `ccache`, `python3`
- A shared remote ccache. Warm-cache incremental builds take minutes; a cold
  full build takes over an hour.

## You are expected to compile your change

**A change that has not been compiled is not finished.** Historically, changes
authored here were opened without ever being built, and validation was left
entirely to post-hoc CI. Do not do that.

Build from the repository root:

```bash
./build_metal.sh --enable-ccache
```

Always pass `--enable-ccache`. Without it you lose the shared cache and the
build will not finish within your session.

Useful flags (see `./build_metal.sh --help` for the full list):

| Flag | When |
| --- | --- |
| `--build-tests` | you added or changed a test |
| `--build-metal-tests` / `--build-ttnn-tests` | narrow the above to one area |
| `--build-programming-examples` | you touched `tt_metal/programming_examples/` |
| `--build-tt-train` | you touched `tt-train/` |
| `-b Debug` | you need assertions to reproduce something |
| `--configure-only` | you only need to prove CMake still configures |

Build the narrowest thing that actually exercises your change. Do not reach for
`--build-all`.

## What to do about the result

- **Builds clean** — say so explicitly in the PR description, including the
  exact command you ran.
- **Fails to build** — fix it and rebuild. Do not open the PR and let CI find
  a compile error you could have caught.
- **Genuinely cannot build** (missing hardware, cold cache, environment
  problem) — open the PR anyway, but state plainly in the description that the
  change is **unverified** and why. An honest "not compiled" is far more useful
  to a reviewer than silence.

Do not claim you ran anything you did not run.

## Things you cannot verify here

The runner has no Tenstorrent accelerator attached, so anything requiring real
silicon — device tests, performance measurements, hardware-dependent
behaviour — cannot be checked in your environment. Compilation and host-side
unit tests are in scope; on-device results are not.

If a change's correctness depends on device behaviour, say so rather than
implying it was validated.

Never state a performance improvement without measurements. A PR claiming a
speedup with no before/after numbers will be flagged in review.

## Scope discipline

- Change the minimum needed to solve the stated issue.
- New source files go in the relevant `sources.cmake`, not into
  `CMakeLists.txt` build structure.
- Adding an external dependency (`find_package`, `CPMAddPackage`,
  `FetchContent_Declare`, a new `third_party/` submodule) requires infra team
  review. If the issue seems to need one, stop and say so in the PR rather than
  adding it.
