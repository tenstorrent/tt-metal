# Tenstorrent TT-Metal / TT-Metalium – Copilot Instructions

## Core Components
- tt_metal/** -  core runtime, device APIs, allocators, low-level kernels
- ttnn/** - higher-level op layer and Python/C++ integration
- tools/** - executable tools used to help with scaleout and debugging
- tt-train/** - training library built on top of ttnn

## Code Review Priorities
- Focus first on **correctness**, **ABI/API stability**, and **performance regressions**
- Prefer **compile-time safety** **clarity** and **simplicity** over cleverness.
- Avoid macros when templates or constexpr utilities suffice.
- Advise against addition of new dependencies if similar ones already exist.
- Review code for correctness, readability, security, memory safety, and best practices.
- Flag anytime someone uses templates excessively or unnecessarily.
- Strongly discourage usage of enable_if, SFINAE, and recursive template instantiations unless absolutely necessary.

## Must-run mental model
- Assume a **large C++20 codebase** with heavy headers; keep compile times in mind (avoid unnecessary includes; prefer forward declarations and PIMPL where appropriate).
- Respect existing **clang-tidy** profile (bugprone, performance, modernize, readability, cppcoreguidelines). If suggesting changes, point to the relevant rule and a minimal diff.
- Treat **rule-of-5/0**, **virtual destructor requirements**, and **avoid const/ref data members** issues as first-class findings; suggest safe fixes.

## Build & test expectations
- For PRs touching C++/kernels/runtime, recommend:
  - Build: `./build_metal.sh --build-all`
  - Lint: `cmake --preset clang-tidy; cmake --build --preset clang-tidy`
- If suggesting refactors, include a quick **compile-time impact note** (header churn, template instantiation).

## Reviews should:
- Cite **exact files/lines** and propose **minimal diffs**.
- Flag any **ABI risk** (public headers) or **dependency version traps** (e.g., protobuf) and propose mitigations.
- Prefer **unit tests** in the nearest tests target. If risky change, request a **micro-benchmark** or existing perf harness.

## Security & reliability
- Watch for UB (aliasing, lifetime, overflow), data races, and unsafe casts.
- For kernel/firmware work, confirm **bounds checks** and **synchronization** (mailboxes, queues) are preserved.

## Document only changes
- Changes that involves documentations only (e.g. typo fixes, new tech reports) should have the PR title prefixed with [skip ci] to avoid running unnecessary checks.
Respond tersely, with code blocks for actionable diffs.
