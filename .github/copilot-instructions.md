# Tenstorrent TT-Metal / TT-Metalium – Copilot Repo Instructions

When performing a code review in this repository:

## Priorities
- Focus first on **correctness**, **ABI/API stability**, **performance regressions**, and **architecture-agnostic design** (no new architecture-specific `#ifdef`s; prefer HAL queries and runtime capability checks).
- Prefer **compile-time safety** and **clarity** over cleverness. Avoid macros when templates or constexpr utilities suffice.
- Limit addition of new dependencies if similar ones already exist.
- Review code for correctness, readability, security, memory safety, and best practices.
- Flag anytime someone uses templates excessively. Strongly discourage usage of enable_if, SFINAE, and recursive template instantiations unless absolutely necessary.

## Must-run mental model
- Assume a **large C++20 codebase** with heavy headers; keep compile times in mind (avoid unnecessary includes; prefer forward declarations and PIMPL where appropriate).
- Respect existing **clang-tidy** profile (bugprone, performance, modernize, readability, cppcoreguidelines). If suggesting changes, point to the relevant rule and a minimal diff.
- Treat **rule-of-5/0**, **virtual destructor requirements**, and **avoid const/ref data members** issues as first-class findings; suggest safe fixes.

## Build & test expectations
- For PRs touching C++/kernels/runtime, recommend:
  - Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
  - Build: `cmake --build build -j`
  - Tests: `ctest --test-dir build --output-on-failure`
  - Lint: `clang-tidy` using the repo `.clang-tidy` (don’t propose rules outside our profile unless gated behind the “extra checks” pipeline).
- If suggesting refactors, include a quick **compile-time impact note** (header churn, template instantiation).

## Reviews should:
- Cite **exact files/lines** and propose **minimal diffs**.
- Flag any **ABI risk** (public headers) or **dependency version traps** (e.g., protobuf) and propose mitigations.
- Prefer **unit tests** in the nearest tests target. If risky change, request a **micro-benchmark** or existing perf harness.

## Security & reliability
- Watch for UB (aliasing, lifetime, overflow), data races, and unsafe casts.
- For kernel/firmware work, confirm **bounds checks** and **synchronization** (mailboxes, queues) are preserved.

Respond tersely, with code blocks for actionable diffs.
