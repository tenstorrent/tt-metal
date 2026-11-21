# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared TTNN Python deprecation helper.

They call the helper directly and use no device and no production function, so they don't depend on
which TTNN functions happen to be deprecated.
"""

import functools
import inspect
import os
import types
import warnings

import pytest

import ttnn._deprecation
from ttnn._deprecation import warn_deprecated

# The helper treats every file in the directory that contains _deprecation.py as TTNN's own code.
TTNN_PACKAGE_DIR = os.path.dirname(ttnn._deprecation.__file__)


# The helper looks only at the filename recorded in each frame's code object. The functions below are templates:
# the tests never call them as they are, but copy them with that filename replaced. A copy whose filename is inside
# the TTNN package looks to the helper exactly like TTNN's own code, such as the Python operation wrappers that every
# binding call goes through, without depending on any real TTNN function. A copy with importlib's bootstrap filename
# looks like the frames between an `import` statement and the module being imported.
def operation_wrapper(what, instead):
    return binding(what, instead)


def binding(what, instead):
    warn_deprecated(what, instead)


def package_init(import_module, submodule_code, what, instead):
    return import_module(submodule_code, what, instead)


def import_module(function, *args):
    return function(*args)


def caller(what, instead):
    warn_deprecated(what, instead)


def _with_filename(function, filename, namespace):
    # The copy looks up globals, such as `binding`, in `namespace` rather than in this module.
    return types.FunctionType(function.__code__.replace(co_filename=filename), namespace, function.__name__)


def _ttnn_code():
    filename = os.path.join(TTNN_PACKAGE_DIR, "operations", "_stand_in_operation.py")
    namespace = {"warn_deprecated": warn_deprecated}
    for function in (operation_wrapper, binding, package_init):
        namespace[function.__name__] = _with_filename(function, filename, namespace)
    return namespace


def _through_ttnn_wrappers():
    return _ttnn_code()["operation_wrapper"]


def _during_nested_import_of_ttnn():
    # The caller imports TTNN, whose package __init__ imports a submodule that calls a deprecated function:
    # caller -> importlib -> TTNN __init__ -> importlib -> TTNN submodule -> helper.
    # functools.partial adds no Python frames.
    bootstrap_import = _with_filename(import_module, "<frozen importlib._bootstrap>", {})
    ttnn_code = _ttnn_code()
    ttnn_package_init = functools.partial(ttnn_code["package_init"], bootstrap_import, ttnn_code["operation_wrapper"])
    return functools.partial(bootstrap_import, ttnn_package_init)


def test_warns_with_future_warning_and_the_policy_message():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("ttnn.old_op", "ttnn.new_op")

    assert len(caught) == 1, f"expected one warning, got {[str(w.message) for w in caught]}"
    assert caught[0].category is FutureWarning, f"expected FutureWarning, got {caught[0].category}"
    assert str(caught[0].message) == "ttnn.old_op is deprecated and will be removed; use ttnn.new_op instead."


@pytest.mark.parametrize(
    "make_call",
    [lambda: warn_deprecated, _through_ttnn_wrappers, _during_nested_import_of_ttnn],
    ids=["direct", "through_ttnn_wrappers", "during_nested_import_of_ttnn"],
)
def test_warning_points_at_the_callers_line(make_call):
    deprecated_call = make_call()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        deprecated_call("ttnn.old_op", "ttnn.new_op")

    locations = [(w.filename, w.lineno) for w in caught]
    assert locations == [(__file__, expected_line)], f"warning attributed to {locations}"


def test_frames_without_a_source_file_are_outside_ttnn_even_from_the_package_directory(monkeypatch):
    monkeypatch.chdir(TTNN_PACKAGE_DIR)
    string_caller = _with_filename(caller, "<string>", {"warn_deprecated": warn_deprecated})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        string_caller("ttnn.old_op", "ttnn.new_op")

    filenames = [w.filename for w in caught]
    assert filenames == ["<string>"], f"warning attributed to {filenames}"


def test_warnings_as_errors_raise_future_warning_through_ttnn_frames(expect_error):
    operation_wrapper = _through_ttnn_wrappers()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with expect_error(FutureWarning, r"^ttnn\.old_op is deprecated and will be removed; use ttnn\.new_op"):
            operation_wrapper("ttnn.old_op", "ttnn.new_op")


@pytest.mark.parametrize("what, instead, empty", [("", "ttnn.new_op", "what"), ("ttnn.old_op", "", "instead")])
def test_empty_what_or_instead_is_rejected(what, instead, empty, expect_error):
    with expect_error(ValueError, f"non-empty '{empty}'"):
        warn_deprecated(what, instead)
