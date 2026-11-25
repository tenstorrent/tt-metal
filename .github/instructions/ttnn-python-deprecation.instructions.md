---
description: 'PR review for deprecations in the TTNN Python API — FutureWarning signal, shared helper, same-PR migration, docstring note, manifest entry, and grace period'
applyTo: 'ttnn/**/*nanobind*.cpp,ttnn/**/*nanobind*.hpp,ttnn/**/nanobind/**,ttnn/cpp/ttnn-nanobind/**,ttnn/ttnn/**/*.py,tests/ttnn/**/*.py'
excludeAgent: "cloud-agent"
---

# TTNN Python Deprecation Review

This policy covers every new deprecation in the TTNN Python API: functions and arguments bound from C++ with nanobind, and functions written in pure Python. It is the C++ public API deprecation policy in [`metal-api.instructions.md`](metal-api.instructions.md) ("Modifying or Deleting Stable APIs"), applied to the Python API, plus the Python-specific rules below.

The category is `FutureWarning` because Python shows it by default, the way the compiler shows a C++ deprecation, whereas `DeprecationWarning` is hidden outside `__main__`.

Apply these rules to deprecations a PR adds or changes. Some existing deprecations predate this policy and use only a log message or a docstring note. Do not use them as examples. They can move to the helper. See "Moving an existing deprecation to the helper".

## Terms

- **Final release**: a `ttnn` version on PyPI with no `rc` or `dev` suffix. Its content is the git tag with the same name, for example `v0.82.0`. Daily `-dev` builds and release candidates are not final releases.
- **First release** of a deprecation: the earliest final release whose content has the deprecation, that is the `[[deprecated]]` attribute or the `warn_deprecated` call.
- **Removal window**: the period in which a PR can remove a deprecated interface. It opens four weeks after the PyPI upload of the first release, or on the `earliest_removal_date` or a promised date if that is later.
- **Transitional entry**: a `.github/deprecations.json` entry whose `introduced_by_pr` merged at or before #65989, the PR that added this policy.

## 🔴 CRITICAL

### Signal

- **FutureWarning through the shared helper, and only that way.** A deprecated function or argument calls `warn_deprecated(what, instead)`, specified in "Shared helper" below. Pure-Python code calls the Python helper. Bindings call the C++ wrapper. Flag a deprecation signalled by a log message (`log_warning`, `logger.warning`, …), by a hand-written `warnings.warn`, or by `PyErr_WarnEx`. Flag a deprecation that is only a docstring note.
- **Message format.** The helper builds exactly `<what> is deprecated and will be removed; use <instead> instead.` With `earliest_removal_date="YYYY-MM-DD"`, it builds `<what> is deprecated and will be removed no earlier than YYYY-MM-DD; use <instead> instead.` A date is a minimum. The removal window can open later than the date, but never earlier. Give a date in the message only through `earliest_removal_date`. A docstring can repeat that date. `instead` is required. It completes the sentence "use … instead". When there is no replacement, phrase the action the caller should take so it fits, for example `instead="a call without correction"`.
- **`what` names the public Python name, fully qualified.** A function: `what="ttnn.sigmoid_accurate"`. An argument: `what="the correction argument of ttnn.sum"`. An attribute: `what="ttnn.LayerNormDefaultProgramConfig.legacy_rsqrt"`. When one binding wrapper serves several operations, each operation passes its own name. Flag a generic `what` such as "this argument".
- **Attribution.** The warning points at the first stack frame outside the TTNN Python package. That frame is the user's code, whether the call goes through TTNN's operation wrappers or straight into a binding. Flag code that passes its own `stacklevel` or warns from somewhere other than the helper.
- **Turning warnings into errors stops the call.** When a filter turns the warning into an error (`-W error::FutureWarning`, `warnings.simplefilter("error")`, pytest's `filterwarnings`), the `FutureWarning` reaches the caller and the deprecated code path doesn't run. Call the helper before doing any work for the deprecated function or argument. Flag bindings that catch and swallow the helper's exception or call it after the work is done.

### Same-PR migration

- The PR that adds the deprecation migrates every in-repo caller: TT-Metalium, TTNN, tests, models and tt-train. Only compatibility tests may keep calling the deprecated interface. They assert the warning with `pytest.warns(FutureWarning)`.
- No PR adds new uses of an already-deprecated interface. The exception is the Python binding of a deprecated C++ function. It keeps calling that function during the grace period.

### Grace period and removal

The same process as the C++ public API:

- The deprecated function or argument stays until its **removal window** opens: four weeks after the PyPI upload of its first release, or on its `earliest_removal_date` if that is later. Wheel users see the warning only in a final release, so the window counts from the release, not from the merge.
- The removal window applies to every deprecated interface in any directory, also in `experimental/` and in TTNN C++. A deprecation is a promise to users.
- Removal happens in a **separate PR**, which also deletes the manifest entry.
- Flag any PR that removes a public TTNN Python function or argument without a prior deprecation, or before its removal window opens.

## 🟡 IMPORTANT

### Docstring

The docstring of the deprecated function, or the docstring entry of the deprecated argument, gets a Sphinx `deprecated` note. Its version argument is the minor version from the latest `-dev` tag on `main`. For example, `git tag --merged origin/main --list 'v*-dev*' --sort=-creatordate | head -1` prints `v0.83.0-dev20261108`, so the note reads:

```rst
.. deprecated:: 0.83
   Use ``ttnn.new_op`` instead.
```


### Manifest

Add one entry to `.github/deprecations.json`, the tracked-deprecations manifest that the deprecation reaper reads. Use the existing fields unchanged: `id`, `description` (naming the replacement), `files`, `introduced_by_pr` (this PR's number), `grace` (at least `4w`) and `owners`. Do not compute or write a date.

The reaper reads `grace` as follows:

- For a transitional entry, it counts from the merge of `introduced_by_pr`.
- For a later entry, it counts from the first release. It finds the first release from the deprecations that `introduced_by_pr` adds. If it cannot find them, it lists the entry as an entry to fix.

Until the reaper can find the first release, it counts every entry from the merge. Its reminder can then come before the removal window opens.

### Deprecated C++ functions behind a binding

- A binding that exposes a `[[deprecated]]` C++ function calls it with no warning suppression. Flag a `#pragma GCC diagnostic` block that the PR adds or changes to silence `-Wdeprecated-declarations` in binding code. The build already controls this warning:
  - By default, the top-level `-Wno-deprecated-declarations` turns it off for all targets.
  - With `TT_METAL_DEPRECATION_WARNINGS=ON`, the build reports each call as a warning and never as an error, also when `CMAKE_COMPILE_WARNING_AS_ERROR` is on.

  A pragma does not change the build result. It only hides the call from the deprecation report.
- A binding is not a compatibility shim. The exception for warning suppressions in `scripts/validate_api/README.md` covers the Metal host API only, not binding code.
- Do not flag a pragma that is already on `main` in lines the PR does not change. A PR that edits such a binding can remove the pragma.
- Deprecating a Python function or argument also deprecates the C++ interface it binds to, when one exists. The same PR marks that C++ function, or the overload that takes the argument, `[[deprecated]]` under [`metal-api.instructions.md`](metal-api.instructions.md). The Python binding then warns the caller and still calls the deprecated C++ function. Flag a PR that deprecates a Python function or argument while the corresponding C++ interface stays undeprecated, and warn the author. This doesn't apply when the Python function or argument exists only in the binding and has no C++ counterpart.

### Moving an existing deprecation to the helper

Existing deprecations may move to the helper at any time.

- **It already emitted a Python warning.** It is the same deprecation. The move changes the category to `FutureWarning` and the message to the helper's format, but the removal window does not restart. The window counts from the first release that had the old warning. Its manifest entry's `introduced_by_pr` is the PR that first deprecated it.
- **It was only a docstring note or a log message.** Users never got a Python warning, so its removal window counts from the first release that has the helper's warning. The moving PR adds the manifest entry with itself as `introduced_by_pr`.
- **Its removal window is already open.** It stays deprecated. Moving it does not decide its removal. That stays its owners' decision.

### Repo test configuration

Leave `pytest.ini` as it is. TTNN `FutureWarning`s stay warnings in the repo's own test runs, so compatibility tests and in-flight migrations don't break the suite.

## Shared helper

The helper is `ttnn/ttnn/_deprecation.py` (Python) and `ttnn/cpp/ttnn-nanobind/deprecation.hpp` (C++). Every deprecation reuses it. A PR that changes it keeps this contract and the test described below.

### Python: `warn_deprecated(what: str, instead: str, *, earliest_removal_date: str | None = None) -> None`

- Lives in a private module of the TTNN Python package, `ttnn/ttnn/_deprecation.py`, and is not re-exported from `ttnn`.
- The module imports only the standard library. Importing it never loads `_ttnn`, so the C++ wrapper can import it at any time without an import cycle, even while `ttnn/__init__.py` is still running.
- Builds the message in the format above. `what` and `instead` are required and non-empty. `earliest_removal_date` is optional and keyword-only. If it is given, it must be a `YYYY-MM-DD` date, or the helper raises `ValueError`.
- Starts at its caller's frame and walks outward to the first frame whose file is not inside the TTNN package directory, which is the directory that contains `_deprecation.py`. It then calls `warnings.warn(message, FutureWarning, stacklevel=<that frame's depth>)`. If every Python frame is inside the package, it attributes the warning to the outermost frame.
- Works on Python 3.10, the minimum the package supports, so it can't use the `skip_file_prefixes` parameter of `warnings.warn`, which was added in 3.12.
- Doesn't catch anything. When a filter turns the warning into an error, `warnings.warn` raises and the exception propagates to the caller.

Attributing the warning to the user's frame is what gives users one warning per call site under Python's default filter, and lets them filter by their own module.

### C++: `ttnn::warn_deprecated(std::string_view what, std::string_view instead, std::string_view earliest_removal_date = {})`

An empty `earliest_removal_date` passes `None` to the Python helper.


- An `inline` function in the header-only `ttnn/cpp/ttnn-nanobind/deprecation.hpp`. It is the whole C++ interface: a binding calls it on the first line of its body, whatever is deprecated (a function, an argument detected with a `std::optional` that defaults to `None`, an overload, a method or a property). Flag new deprecation options on `ttnn::bind_function` or template wrappers until several real call sites repeat the same code.
- Holds the GIL with `nb::gil_scoped_acquire`, because bindings registered through `ttnn::bind_function` run with the GIL released. It imports `ttnn._deprecation` and calls `warn_deprecated`.
- Imports the module on every call, never while the extension module is being set up in `NB_MODULE`. The import is a lookup in `sys.modules`.
- Doesn't keep the Python function in a `static nb::object`. A static's destructor runs after the interpreter has shut down and crashes the process at exit.
- Lets an exception from the call propagate as `nb::python_error`. The binding stops before it runs the deprecated path, and nanobind hands the `FutureWarning` back to the caller.
- Contains no message formatting and no stack walking. That logic exists once, in Python.
- Only binding code calls it: sources in `ttnn/cpp/ttnn-nanobind/` and the `*_nanobind.cpp` and `*_nanobind.hpp` files, which build into the Python extension. Code in the TTNN C++ library (`TTNN::CPP`) never emits a Python warning. A deprecation in the C++ API uses `[[deprecated]]`, as in [`metal-api.instructions.md`](metal-api.instructions.md).

### The helper's test

`tests/ttnn/unit_tests/base_functionality/test_deprecation.py`, outside the TTNN package. It needs no device and calls no production function, so it doesn't depend on which functions happen to be deprecated. It calls `ttnn._deprecation.warn_deprecated` and checks:

1. The category is `FutureWarning`.
2. The exact message format.
3. The warning's filename and line are the test's own, both on a direct call and through a frame inside the TTNN package. The test makes such a frame without TTNN code by running `compile(...)` with a filename inside the package directory, which is what a call through TTNN's operation wrappers looks like to the helper.
4. Under `warnings.simplefilter("error")`, the call raises `FutureWarning` and the exception propagates.
5. With `earliest_removal_date`, the message has the date. An `earliest_removal_date` in any form other than `YYYY-MM-DD` raises `ValueError`.

The C++ wrapper has no dedicated test. Its logic is a GIL acquire and one call; every test that calls a binding using it exercises it.

## Review Checklist

- [ ] The deprecation emits `FutureWarning` through `warn_deprecated`, with no log message, hand-written `warnings.warn` or `PyErr_WarnEx`
- [ ] The message follows the helper's format, and a date appears in it only through `earliest_removal_date`
- [ ] `what` is the fully qualified public Python name, per operation
- [ ] The C++ interface behind a deprecated Python function or argument, if any, is `[[deprecated]]` in the same PR
- [ ] No `#pragma` that the PR adds or changes silences `-Wdeprecated-declarations` in binding code
- [ ] The helper runs before any work for the deprecated function or argument, and its exception isn't swallowed
- [ ] Only binding code calls the C++ wrapper; no code in `TTNN::CPP` emits a Python warning
- [ ] Every in-repo caller is migrated; the remaining callers are compatibility tests using `pytest.warns(FutureWarning)`
- [ ] No new uses of an already-deprecated interface, except the binding of a deprecated C++ function
- [ ] The docstring has `.. deprecated:: <minor version>` with the version from the latest `-dev` tag on `main`
- [ ] `.github/deprecations.json` has one entry with the existing fields and a grace of at least `4w`
- [ ] A removal is in its own PR, after the removal window opens: four weeks after the PyPI upload of the first release, or the `earliest_removal_date` if that is later
- [ ] `pytest.ini` doesn't turn `FutureWarning` into errors
