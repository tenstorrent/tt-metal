# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ttrun command-line utility."""

import os
import tempfile
import yaml
import importlib
from pathlib import Path
from unittest.mock import patch
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from ttnn.distributed.ttrun import (
    main,
    parse_binding_config,
    resolve_path,
    get_rank_environment,
    build_mpi_command,
    RankBinding,
    TTRunConfig,
    ORIGINAL_CWD,
)

# Import the module directly to avoid conflicts with distributed.py
ttrun_module = importlib.import_module("ttnn.distributed.ttrun")


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_rank_binding_yaml(temp_dir):
    """Create a sample rank binding YAML file."""
    yaml_content = {
        "rank_bindings": [
            {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "env_overrides": {"TEST_VAR": "value0"}},
            {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1},
        ],
        "global_env": {"GLOBAL_VAR": "global_value"},
        "mesh_graph_desc_path": str(temp_dir / "mesh_graph.yaml"),
    }
    yaml_file = temp_dir / "rank_binding.yaml"
    mesh_graph_file = temp_dir / "mesh_graph.yaml"

    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)
    mesh_graph_file.touch()

    return yaml_file


@pytest.fixture
def sample_mesh_graph_descriptor(temp_dir):
    """Create a sample mesh graph descriptor file."""
    mesh_file = temp_dir / "mesh_graph.yaml"
    mesh_file.touch()
    return mesh_file


class TestCommandLineArguments:
    """Test command-line argument parsing and validation."""

    def test_mutually_exclusive_options(self, runner, sample_rank_binding_yaml, sample_mesh_graph_descriptor):
        """Test that --rank-binding and --mesh-graph-descriptor are mutually exclusive."""
        result = runner.invoke(
            main,
            [
                "--rank-binding",
                str(sample_rank_binding_yaml),
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_no_mode_specified(self, runner):
        """Test that at least one mode must be specified."""
        result = runner.invoke(main, ["echo", "test"])
        assert result.exit_code != 0
        assert "must be specified" in result.output.lower()

    def test_new_mode_not_implemented(self, runner, sample_mesh_graph_descriptor):
        """Test that new mode raises not implemented error."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "not yet implemented" in result.output.lower()

    def test_new_mode_requires_hosts(self, runner, sample_mesh_graph_descriptor):
        """Test that new mode requires --hosts unless --mock-cluster-rank-binding is provided."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "--hosts is required" in result.output.lower()

    def test_new_mode_with_hosts(self, runner, sample_mesh_graph_descriptor):
        """Test that new mode accepts --hosts."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "--hosts",
                "node1,node2",
                "echo",
                "test",
            ],
        )
        # Should fail with "not yet implemented" but not with "--hosts is required"
        assert result.exit_code != 0
        assert "--hosts is required" not in result.output.lower()
        assert "not yet implemented" in result.output.lower()

    def test_new_mode_with_mock_cluster_no_hosts(self, runner, sample_mesh_graph_descriptor, temp_dir):
        """Test that new mode doesn't require --hosts when --mock-cluster-rank-binding is provided."""
        mock_file = temp_dir / "mock.yaml"
        mock_file.touch()

        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "--mock-cluster-rank-binding",
                str(mock_file),
                "echo",
                "test",
            ],
        )
        # Should fail with "not yet implemented" but not with "--hosts is required"
        assert result.exit_code != 0
        assert "--hosts is required" not in result.output.lower()
        assert "not yet implemented" in result.output.lower()

    def test_legacy_mode_ignores_hosts(self, runner, sample_rank_binding_yaml):
        """Test that legacy mode ignores --hosts option."""
        import subprocess

        with patch.object(subprocess, "run"):
            result = runner.invoke(
                main,
                [
                    "--rank-binding",
                    str(sample_rank_binding_yaml),
                    "--hosts",
                    "node1,node2",
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
            # Should succeed (legacy mode ignores --hosts, may log warning)
            assert result.exit_code == 0


class TestYAMLParsing:
    """Test YAML configuration parsing and validation."""

    def test_valid_rank_binding_yaml(self, sample_rank_binding_yaml):
        """Test parsing a valid rank binding YAML."""
        config = parse_binding_config(sample_rank_binding_yaml)
        assert len(config.rank_bindings) == 2
        assert config.rank_bindings[0].rank == 0
        assert config.rank_bindings[1].rank == 1
        assert config.global_env["GLOBAL_VAR"] == "global_value"
        assert config.mesh_graph_desc_path.exists()

    def test_invalid_rank_binding_duplicate_ranks(self, temp_dir):
        """Test that duplicate ranks are rejected."""
        yaml_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0},
                {"rank": 0, "mesh_id": 0},  # Duplicate rank
            ],
            "mesh_graph_desc_path": str(temp_dir / "mesh_graph.yaml"),
        }
        yaml_file = temp_dir / "invalid.yaml"
        mesh_file = temp_dir / "mesh_graph.yaml"
        mesh_file.touch()

        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(ValueError, match="Duplicate ranks"):
            parse_binding_config(yaml_file)


class TestPathResolution:
    """Test path resolution functions."""

    def test_resolve_absolute_path(self, temp_dir):
        """Test resolving an absolute path."""
        test_file = temp_dir / "test.txt"
        test_file.touch()

        resolved = resolve_path(test_file, must_exist=True)
        assert resolved.is_absolute()
        assert resolved.exists()

    def test_resolve_path_not_found(self, temp_dir):
        """Test that missing paths raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            resolve_path(temp_dir / "nonexistent.txt", must_exist=True)


class TestEnvironmentVariables:
    """Test environment variable building."""

    def test_get_rank_environment(self, sample_rank_binding_yaml):
        """Test building environment variables for a rank."""
        config = parse_binding_config(sample_rank_binding_yaml)
        binding = config.rank_bindings[0]

        env = get_rank_environment(binding, config)

        assert env["TT_MESH_ID"] == "0"
        assert env["TT_MESH_GRAPH_DESC_PATH"] == str(config.mesh_graph_desc_path)
        assert env["TT_MESH_HOST_RANK"] == "0"
        assert env["TEST_VAR"] == "value0"  # From env_overrides
        assert env["GLOBAL_VAR"] == "global_value"  # From global_env


class TestMPICommandBuilding:
    """Test MPI command building."""

    def test_build_mpi_command_basic(self, sample_rank_binding_yaml):
        """Test building a basic MPI command."""
        config = parse_binding_config(sample_rank_binding_yaml)
        program = ["echo", "hello"]

        cmd = build_mpi_command(config, program)

        assert "mpirun" in cmd[0] or cmd[0].endswith("mpirun")
        assert "--tag-output" in cmd
        assert "--bind-to" in cmd


class TestLegacyFlow:
    """Test legacy flow function - minimal smoke tests."""

    def test_legacy_flow_dry_run(self, runner, sample_rank_binding_yaml):
        """Test legacy flow works with dry-run flag."""
        import subprocess

        with patch.object(subprocess, "run") as mock_subprocess:
            result = runner.invoke(
                main,
                [
                    "--rank-binding",
                    str(sample_rank_binding_yaml),
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
            # Should not call subprocess.run in dry-run mode
            mock_subprocess.assert_not_called()
            # Dry-run should succeed
            assert result.exit_code == 0
