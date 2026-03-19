# TDD — framework notes

Use this file when the stack is known; otherwise follow project conventions.

## Principles (any stack)

- Name tests after **behavior** (`returns_empty_list_when_no_matches`), not
  after the method name only.
- One logical behavior per test when possible; share setup with fixtures/helpers
  per project style.
- On failure, the reader should see **what was expected**, **what happened**,
  and **which case** (data/inputs).

## pytest (Python)

- Prefer `assert actual == expected` with clear variable names; pytest rewrites
  asserts for useful output.
- For contracts: `pytest.raises(ExpectedError, match="...")` when errors are
  part of the API.
- Use `@pytest.mark.parametrize` for multiple inputs instead of copy-paste.

## Jest / Vitest (JavaScript / TypeScript)

- Use `expect(x).toEqual(y)` for deep equality; add a second argument message
  when the default diff is not enough.
- `expect(() => fn()).toThrow(expected)` for error contracts.

## Go (`testing`)

- Use `t.Helper()` in helpers; `t.Fatalf` / `t.Errorf` with formatted context.
- Table-driven tests with a `name` field per row for readable failure output.

## Rust

- Prefer `assert_eq!`, `assert_ne!`, or custom messages via `assert!(cond, "…")`.
- For `Result`, match on `Ok`/`Err` explicitly in tests.

## C++ (GoogleTest)

- Use `TEST(Suite, Name)` for free functions; `TEST_F(Fixture, Name)` when setup
 /teardown or shared state lives in a `::testing::Test` subclass.
- Prefer `EXPECT_*` when later assertions still add signal; use `ASSERT_*`
  when the rest of the test is invalid if the check fails.
- Equality / contracts: `EXPECT_EQ`, `EXPECT_NE`, `EXPECT_TRUE`, `EXPECT_STREQ`
  for C strings; `EXPECT_THROW(stmt, Type)` and `EXPECT_DEATH` / `ASSERT_DEATH`
  when error behavior is specified.
- Add context with the `<<` stream operator on any assertion, or `SCOPED_TRACE`
  / `GTEST_SKIP()` when a case does not apply.
- Parameterized suites: `TEST_P` + `INSTANTIATE_TEST_SUITE_P` for many inputs
  without duplicated bodies.

## CMake

- Declare tests with `enable_testing()` then `add_test(NAME … COMMAND …)` so
  `ctest` can run them; use generator expressions and `$<TARGET_FILE:…>` when the
  command is a built binary.
- Prefer a dedicated test target (`add_executable` / `gtest_discover_tests` from
  `GoogleTest` module, or `FetchContent`/`find_package` for GTest) and
  `target_link_libraries(… GTest::gtest_main)` rather than ad hoc invocations.
- During TDD, narrow runs: `ctest -R pattern -V` (regex on test name) or run the
  test executable with `--gtest_filter=Suite.Name` for fast red-green loops.
- Keep test binaries out of install rules unless the project intentionally
  ships them; mirror the repo’s pattern for labels, fixtures, and env vars
  (`set_tests_properties`).

## Bash

- Prefer a test runner with structured output over one-off scripts: **Bats**
  (`@test`, `run`, `assert`, `assert_output`, `assert_line`) or **shunit2**
  (`assertEquals`, `fail`) so failures name the example.
- For tiny checks without a framework, use `[[ … ]]` with an explicit failure
  path: `if ! [[ … ]]; then echo "…context…" >&2; exit 1; fi` (avoid silent
  `test` failures).
- Quote every expansion; use `set -euo pipefail` in test drivers when the
  project allows it. For scripting rules (quoting, strict mode), follow the
  repo’s bash-safety skill if present.
- Contract tests: assert exit codes (`run …; [[ $status -eq … ]]`), stdout/stderr
  snippets, and file state after the command under test.
