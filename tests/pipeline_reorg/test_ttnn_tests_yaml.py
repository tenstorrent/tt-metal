import pytest
import yaml
import pathlib
import re


YAML_PATH = pathlib.Path(__file__).parent / "ttnn-tests.yaml"


@pytest.fixture
def yaml_content():
    """Load the ttnn-tests.yaml file."""
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def test_yaml_is_valid(yaml_content):
    """Test that ttnn-tests.yaml is valid YAML."""
    assert yaml_content is not None
    assert isinstance(yaml_content, list)


def test_core_ttnn_unit_test_group_exists(yaml_content):
    """Test that 'core ttnn unit test group' entry exists."""
    group_names = [entry.get("name") for entry in yaml_content]
    assert "core ttnn unit test group" in group_names


def test_core_ttnn_group_command_is_merged(yaml_content):
    """Test that the core ttnn unit test group command is a single merged pytest call."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    # Should not contain && (chained commands)
    assert "&&" not in cmd, f"Command still contains && chaining: {cmd}"


def test_core_ttnn_group_includes_all_test_directories(yaml_content):
    """Test that the merged command includes all four test directories."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    test_dirs = [
        "tests/ttnn/unit_tests/base_functionality",
        "tests/ttnn/unit_tests/benchmarks",
        "tests/ttnn/unit_tests/tensor",
        "tests/ttnn/unit_tests/per_core_allocation",
    ]

    for test_dir in test_dirs:
        assert test_dir in cmd, f"Command missing test directory: {test_dir}"


def test_core_ttnn_group_has_marker_filter(yaml_content):
    """Test that the command has the marker filter."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    assert '-m "not disable_fast_runtime_mode"' in cmd or "-m 'not disable_fast_runtime_mode'" in cmd


def test_core_ttnn_group_has_timeout_flag(yaml_content):
    """Test that the command has the timeout flag."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    assert "--timeout 300" in cmd


def test_core_ttnn_group_has_verbose_and_exitfirst_flags(yaml_content):
    """Test that the command has the verbose and exitfirst flags."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    assert "-xv" in cmd or "-x" in cmd and "-v" in cmd


def test_core_ttnn_group_single_pytest_invocation(yaml_content):
    """Test that there is only a single pytest invocation."""
    core_ttnn = next(
        entry for entry in yaml_content if entry.get("name") == "core ttnn unit test group"
    )
    cmd = core_ttnn.get("cmd", "")

    # Count pytest invocations
    pytest_count = cmd.count("pytest")
    assert pytest_count == 1, f"Expected 1 pytest invocation, found {pytest_count}"
