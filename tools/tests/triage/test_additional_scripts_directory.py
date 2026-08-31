# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Unit tests for --additional-scripts-directory. These need no hardware - the fixture scripts under
# additional_scripts/ and additional_scripts_extra/ ignore the context, so run_script() can be
# driven with a stub one. additional_scripts_broken/ holds the things that are supposed to go wrong:
# a dependency that does not exist, a file name that collides with another directory's, a module that
# exits at import, and a package __init__.py that must never be loaded as a script.

import os
import sys

import pytest


metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")
sys.path.insert(0, triage_home)


from triage import ScriptArguments, TTTriageError, TriageScript, resolve_scripts_directories, run_script

# Realpath'd, because that is the spelling triage normalises every directory to - so these constants
# can be compared directly against what it discovers and reports.
TESTS_HOME = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIRECTORY = os.path.join(TESTS_HOME, "additional_scripts")
EXTRA_SCRIPTS_DIRECTORY = os.path.join(TESTS_HOME, "additional_scripts_extra")
BROKEN_SCRIPTS_DIRECTORY = os.path.join(TESTS_HOME, "additional_scripts_broken")

# Printed by additional_scripts/check_additional_directory.py. Spelled out here rather than imported
# so the assertion is on what the script actually writes to stdout.
SUCCESS_MESSAGE = "check_additional_directory ran from an additional scripts directory"


class StubContext:
    """Stands in for a ttexalens Context. The fixture scripts never touch it."""


@pytest.fixture(autouse=True)
def isolate_module_state():
    """Undo the sys.path and sys.modules writes that resolving directories and loading scripts do.

    Scripts are imported by bare module name, and the directories stay on sys.path for the rest of
    the run by design - which would leak into the other test files sharing this pytest process.
    """
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


def make_args(*directories: str) -> ScriptArguments:
    return ScriptArguments({"--additional-scripts-directory": list(directories)})


def both_directories() -> ScriptArguments:
    return make_args(SCRIPTS_DIRECTORY, EXTRA_SCRIPTS_DIRECTORY)


@pytest.fixture
def scripts_directories() -> list[str]:
    """The full search list - built-in plus both fixture directories - as every entry point builds it.

    Loading and discovery need this: resolving is what puts the directories on sys.path, so without
    it check_additional_directory cannot import the provider that lives in the other directory.
    """
    return resolve_scripts_directories(both_directories())


def test_script_in_additional_directory_runs_by_bare_name(capsys):
    """The whole feature end to end: --run=<name> resolves out of tree, and the script runs."""
    result = run_script(
        script_path="check_additional_directory",
        args=both_directories(),
        context=StubContext(),
        return_result=True,
    )

    assert SUCCESS_MESSAGE in capsys.readouterr().out
    assert [row.message for row in result] == [SUCCESS_MESSAGE]
    # Proof the two providers really ran and were reached through their own directories.
    assert result[0].dependencies == "additional_provider, additional_extra_provider"


def test_flag_is_honoured_without_main():
    """run_script() is a second entry point; the directories must be read there too, not just main()."""
    result = run_script(
        script_path=os.path.join(SCRIPTS_DIRECTORY, "check_additional_directory.py"),
        args=None,
        context=StubContext(),
        argv=[f"--additional-scripts-directory={EXTRA_SCRIPTS_DIRECTORY}"],
        return_result=True,
    )

    assert [row.message for row in result] == [SUCCESS_MESSAGE]


def test_bare_dependency_names_resolve_in_every_searched_directory(scripts_directories):
    """system_info is built-in, additional_provider is a sibling, the third is in the other directory."""
    script_path = os.path.join(SCRIPTS_DIRECTORY, "check_additional_directory.py")
    scripts = TriageScript.load_all(script_path, scripts_directories)

    assert set(scripts[script_path].config.depends) == {
        os.path.join(triage_home, "system_info.py"),
        os.path.join(SCRIPTS_DIRECTORY, "additional_provider.py"),
        os.path.join(EXTRA_SCRIPTS_DIRECTORY, "additional_extra_provider.py"),
    }


def test_missing_dependency_names_the_script_that_wants_it():
    """A typo in `depends` must say who asked for what, not surface as a bare ModuleNotFoundError."""
    broken = os.path.join(BROKEN_SCRIPTS_DIRECTORY, "broken_dependency_check.py")

    with pytest.raises(TTTriageError) as excinfo:  # allow-pytest.raises: no expect_error fixture
        TriageScript.load_all(broken)

    message = str(excinfo.value)
    assert "broken_dependency_check.py depends on no_such_provider.py" in message
    assert "does not exist" in message
    # The directories it looked in are listed, so the author can see why it was not found.
    assert BROKEN_SCRIPTS_DIRECTORY in message and os.path.realpath(triage_home) in message


def test_dependency_search_path_lists_each_directory_once(scripts_directories):
    """A script's own directory is usually already in the list; it must not be searched twice."""
    for directory, script_name in [
        (triage_home, "system_info.py"),
        (SCRIPTS_DIRECTORY, "additional_provider.py"),
        (EXTRA_SCRIPTS_DIRECTORY, "additional_extra_provider.py"),
    ]:
        script = TriageScript.load(os.path.join(directory, script_name), scripts_directories)

        search_path = script.dependency_search_path
        assert len(search_path) == len(set(search_path)), search_path
        # The script's own directory still wins, so a sibling beats a same-named built-in.
        assert search_path[0] == os.path.realpath(directory)
        assert set(search_path) == set(scripts_directories)


def test_discovery_spans_built_in_and_additional_directories(scripts_directories):
    scripts = TriageScript.discover_all(scripts_directories)

    names = {os.path.basename(path) for path in scripts}
    assert {"check_additional_directory.py", "additional_provider.py", "additional_extra_provider.py"} <= names
    assert "system_info.py" in names, "built-in scripts must still be discovered"


def test_script_directory_is_searched_even_when_not_passed_as_a_flag():
    """Running an out-of-tree script directly must resolve a dependency sitting next to it."""
    consumer = os.path.join(EXTRA_SCRIPTS_DIRECTORY, "check_extra_directory.py")

    # No additional directories at all - the script's own directory comes first in the search path.
    scripts = TriageScript.load_all(consumer)

    sibling = os.path.join(EXTRA_SCRIPTS_DIRECTORY, "additional_extra_provider.py")
    assert scripts[consumer].config.depends == [sibling]
    assert [s.name for s in scripts[consumer].depends] == ["additional_extra_provider.py"]


def test_package_init_is_not_treated_as_a_script():
    """A consuming repo's __init__.py must not be imported as a module named '__init__'."""
    scripts = TriageScript.discover_all([BROKEN_SCRIPTS_DIRECTORY])

    assert "__init__.py" not in {os.path.basename(path) for path in scripts}
    # It was skipped by name, not imported and then rejected.
    assert "__init__" not in sys.modules


def test_one_directory_under_several_spellings_is_scanned_once(scripts_directories):
    spellings = [
        SCRIPTS_DIRECTORY,
        SCRIPTS_DIRECTORY + os.sep,
        os.path.join(SCRIPTS_DIRECTORY, "..", "additional_scripts"),
    ]
    scripts = TriageScript.discover_all(spellings)

    assert sorted(os.path.basename(path) for path in scripts) == [
        "additional_provider.py",
        "check_additional_directory.py",
    ]


def test_same_file_name_in_two_directories_loses_the_shadowed_copy():
    """Only one file can own a module name, so the shadowed copy is skipped rather than run."""
    real = os.path.join(SCRIPTS_DIRECTORY, "additional_provider.py")
    shadowed = os.path.join(BROKEN_SCRIPTS_DIRECTORY, "additional_provider.py")

    scripts = TriageScript.discover_all([SCRIPTS_DIRECTORY, BROKEN_SCRIPTS_DIRECTORY])

    # The real one is kept; the copy is dropped instead of quietly replacing it.
    assert real in scripts
    assert shadowed not in scripts

    # Asking for the shadowed copy by path says exactly what happened, rather than running the
    # other file under this path.
    with pytest.raises(ValueError, match="already resolves to"):  # allow-pytest.raises: no expect_error fixture
        TriageScript.load(shadowed)


def test_module_that_exits_at_import_does_not_end_the_run():
    """Discovery imports every .py in a directory the user named; SystemExit is not an Exception."""
    scripts = TriageScript.discover_all([BROKEN_SCRIPTS_DIRECTORY])

    names = {os.path.basename(path) for path in scripts}
    assert "cli_tool.py" not in names
    # The run survives it: the other scripts in the same directory are still discovered.
    assert "broken_dependency_check.py" in names


def test_directories_are_validated_deduplicated_and_importable():
    directories = resolve_scripts_directories(make_args(SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY + os.sep))

    # One list, built-in first, each directory once - so no caller has to prepend the built-in one.
    assert directories == [os.path.realpath(triage_home), os.path.realpath(SCRIPTS_DIRECTORY)]
    assert directories[1] in sys.path


def test_missing_directory_is_reported_with_the_spelling_the_user_gave():
    missing = "/no/such/scripts/dir"
    expected = f"{missing} is not a directory"

    with pytest.raises(TTTriageError, match=expected):  # allow-pytest.raises: no expect_error fixture
        resolve_scripts_directories(make_args(missing))


def test_no_flag_still_yields_the_built_in_directory():
    assert resolve_scripts_directories(ScriptArguments({})) == [os.path.realpath(triage_home)]


def test_naming_the_built_in_directory_with_the_flag_is_a_no_op():
    """Dedupe covers it, so there is no special case to keep in sync."""
    assert resolve_scripts_directories(make_args(triage_home)) == [os.path.realpath(triage_home)]
    assert resolve_scripts_directories(make_args(triage_home + os.sep)) == [os.path.realpath(triage_home)]
