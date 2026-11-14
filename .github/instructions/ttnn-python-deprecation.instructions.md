---
description: 'PR review for deprecations in the TTNN Python API — FutureWarning signal, shared helper, same-PR migration, docstring note, manifest entry, and grace period'
applyTo: 'ttnn/**/*nanobind*.cpp,ttnn/**/*nanobind*.hpp,ttnn/**/nanobind/**,ttnn/cpp/ttnn-nanobind/**,ttnn/ttnn/**/*.py,tests/ttnn/**/*.py'
excludeAgent: "cloud-agent"
---

# TTNN Python Deprecation Review

This policy covers every new deprecation in the TTNN Python API: functions and arguments bound from C++ with nanobind, and functions written in pure Python. It is the C++ public API deprecation policy in [`metal-api.instructions.md`](metal-api.instructions.md) ("Modifying or Deleting Stable APIs"), applied to the Python API, plus the Python-specific rules below.

The category is `FutureWarning` because Python shows it by default, the way the compiler shows a C++ deprecation, whereas `DeprecationWarning` is hidden outside `__main__`.

Apply these rules to deprecations a PR adds or changes. Some existing deprecations predate this policy: two hand-written `DeprecationWarning`s and several log-message deprecations. Don't use them as examples.

## 🔴 CRITICAL

### Signal

- **FutureWarning through the shared helper, and only that way.** A deprecated function or argument calls `warn_deprecated(what, instead)`, specified in "Shared helper" below. Pure-Python code calls the Python helper. Bindings call the C++ wrapper. Flag a deprecation signalled by a log message (`log_warning`, `logger.warning`, …), by a hand-written `warnings.warn`, or by `PyErr_WarnEx`. Flag a deprecation that is only a docstring note.
- **Message format.** The helper builds exactly `<what> is deprecated and will be removed; use <instead> instead.` Flag messages, docstrings or comments that give a calendar removal date: the grace period starts at merge and is tracked in the manifest. `instead` is required. It completes the sentence "use … instead". When there is no replacement, phrase the action the caller should take so it fits, for example `instead="a call without correction"`.
- **Attribution.** The warning points at the first stack frame outside the TTNN Python package. That frame is the user's code, whether the call goes through TTNN's operation wrappers or straight into a binding. Flag code that passes its own `stacklevel` or warns from somewhere other than the helper.
- **Turning warnings into errors stops the call.** When a filter turns the warning into an error (`-W error::FutureWarning`, `warnings.simplefilter("error")`, pytest's `filterwarnings`), the `FutureWarning` reaches the caller and the deprecated code path doesn't run. Call the helper before doing any work for the deprecated function or argument. Flag bindings that catch and swallow the helper's exception or call it after the work is done.

### Same-PR migration

- The PR that adds the deprecation migrates every in-repo caller: TT-Metalium, TTNN, tests, models and tt-train. Only compatibility tests may keep calling the deprecated interface. They assert the warning with `pytest.warns(FutureWarning)`.
- No PR adds new uses of an already-deprecated interface.

### Grace period and removal

The same process as the C++ public API:

- The deprecated function or argument stays for at least **four weeks** after the deprecation merges to `main`, or longer if a later date was promised.
- Removal happens in a **separate PR**, which also deletes the manifest entry.
- Flag any PR that removes a public TTNN Python function or argument unless a deprecation of it has been on `main` for at least four weeks.

## 🟡 IMPORTANT

### Docstring

The docstring of the deprecated function, or the docstring entry of the deprecated argument, gets a Sphinx `deprecated` note. Its version argument is the minor version from the latest `-dev` tag on `main`. For example, `git tag --merged origin/main --list 'v*-dev*' --sort=-creatordate | head -1` prints `v0.83.0-dev20261108`, so the note reads:

```rst
.. deprecated:: 0.83
   Use ``ttnn.new_op`` instead.
```

The version is required. Without one, Sphinx renders the first word of the text as the version, as in "Deprecated since version Use: …".

### Manifest

Add one entry to `.github/deprecations.json`, the tracked-deprecations manifest that the deprecation reaper reads. Use the existing fields unchanged: `id`, `description` (naming the replacement), `files`, `introduced_by_pr` (this PR's number), `grace` (at least `4w`) and `owners`. The grace period starts when the PR merges, because the reaper counts from the PR's merge time. Don't compute or write a date.

### Repo test configuration

Leave `pytest.ini` as it is. TTNN `FutureWarning`s stay warnings in the repo's own test runs, so compatibility tests and in-flight migrations don't break the suite.

## Shared helper

The helper doesn't exist yet. The PR that adds the first new TTNN Python deprecation adds it, following this contract, together with the test described below. Every later deprecation reuses it.

### Python: `warn_deprecated(what: str, instead: str) -> None`

- Lives in a private module of the TTNN Python package, `ttnn/ttnn/_deprecation.py`, and is not re-exported from `ttnn`.
- Builds the message in the format above. Both arguments are required and non-empty.
- Starts at its caller's frame and walks outward to the first frame whose file is not inside the TTNN package directory, which is the directory that contains `_deprecation.py`. It then calls `warnings.warn(message, FutureWarning, stacklevel=<that frame's depth>)`. If every Python frame is inside the package, it attributes the warning to the outermost frame.
- Works on Python 3.10, the minimum the package supports, so it can't use the `skip_file_prefixes` parameter of `warnings.warn`, which was added in 3.12.
- Doesn't catch anything. When a filter turns the warning into an error, `warnings.warn` raises and the exception propagates to the caller.

Attributing the warning to the user's frame is what gives users one warning per call site under Python's default filter, and lets them filter by their own module.

### C++: a thin wrapper in `ttnn/cpp/ttnn-nanobind/`

- Takes the same two parameters as the Python function.
- Holds the GIL with `nb::gil_scoped_acquire`, because bindings registered through `ttnn::bind_function` run with the GIL released. It imports `ttnn._deprecation` and calls `warn_deprecated`.
- Lets an exception from the call propagate as `nb::python_error`. The binding stops before it runs the deprecated path, and nanobind hands the `FutureWarning` back to the caller.
- Contains no message formatting and no stack walking. That logic exists once, in Python.

### The helper's test

One test at the public `ttnn` Python API, in a test module under `tests/ttnn/`, outside the TTNN package. It calls the newly deprecated function or argument and checks four things:

1. The category is `FutureWarning`.
2. The message matches the format.
3. The warning's filename is the test module, which shows that attribution works through TTNN's operation wrappers.
4. Under `warnings.simplefilter("error")`, the call raises `FutureWarning` and the deprecated path has no effect.

The same test shape applies when the first caller is a pure-Python API. `test_extract_resource_usage_per_core_deprecated_kwarg` in `tests/ttnn/unit_tests/base_functionality/test_graph_trace_utils.py` is prior art for calling a deprecated argument at the public API.

## Review Checklist

- [ ] The deprecation emits `FutureWarning` through `warn_deprecated`, with no log message, hand-written `warnings.warn` or `PyErr_WarnEx`
- [ ] The message follows `<what> is deprecated and will be removed; use <instead> instead.` and contains no date
- [ ] The helper runs before any work for the deprecated function or argument, and its exception isn't swallowed
- [ ] Every in-repo caller is migrated; the remaining callers are compatibility tests using `pytest.warns(FutureWarning)`
- [ ] No new uses of an already-deprecated interface
- [ ] The docstring has `.. deprecated:: <minor version>` with the version from the latest `-dev` tag on `main`
- [ ] `.github/deprecations.json` has one entry with the existing fields and a grace of at least `4w`
- [ ] A removal is in its own PR, at least four weeks after the deprecation merged to `main`
- [ ] If this PR adds the helper, it follows the contract above and includes the four-check test
- [ ] `pytest.ini` doesn't turn `FutureWarning` into errors
