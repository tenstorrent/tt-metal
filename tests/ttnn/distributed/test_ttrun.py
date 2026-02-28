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
    get_mpi_launcher,
    RankfileSyntax,
    build_rankfile_args,
    detect_rankfile_syntax,
    inject_rankfile_mpi_args,
    get_generate_rank_bindings_output_paths,
    build_generate_rank_bindings_mpi_cmd,
    run_phase1_generate_rank_bindings,
    new_mode_flow,
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
        """Test that new mode requires --hosts or --mock-cluster-rank-binding."""
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
        """Test that new mode accepts --hosts and attempts Phase 1."""
        import subprocess

        # Mock subprocess.run to avoid actually running MPI
        with patch.object(subprocess, "run") as mock_run:
            # Mock Phase 1 failure (executable not found) - this is expected in test env
            mock_run.side_effect = FileNotFoundError("generate_rank_bindings not found")

            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(sample_mesh_graph_descriptor),
                    "--hosts",
                    "node1,node2",
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
            # Should fail because generate_rank_bindings not found, but not because "--hosts is required"
            assert result.exit_code != 0
            assert "--hosts is required" not in result.output.lower()
            # Should attempt Phase 1 (find executable)
            assert "generate_rank_bindings" in result.output.lower() or "Phase 1" in result.output.lower()

    def test_new_mode_with_mock_cluster_no_hosts(self, runner, sample_mesh_graph_descriptor, temp_dir):
        """Test that new mode doesn't require --hosts when --mock-cluster-rank-binding is provided."""
        import subprocess
        import yaml

        # Create valid mock mapping file
        mock_file = temp_dir / "mock.yaml"
        mock_desc1 = temp_dir / "mock_desc_0.yaml"
        mock_desc2 = temp_dir / "mock_desc_1.yaml"
        mock_desc1.touch()
        mock_desc2.touch()

        mock_data = {
            "rank_to_cluster_mock_cluster_desc": {
                0: str(mock_desc1),
                1: str(mock_desc2),
            }
        }
        with open(mock_file, "w") as f:
            yaml.dump(mock_data, f)

        # Mock subprocess.run to avoid actually running MPI
        with patch.object(subprocess, "run") as mock_run:
            # Mock Phase 1 failure (executable not found) - this is expected in test env
            mock_run.side_effect = FileNotFoundError("generate_rank_bindings not found")

            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(sample_mesh_graph_descriptor),
                    "--mock-cluster-rank-binding",
                    str(mock_file),
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
            # Should fail because generate_rank_bindings not found, but not because "--hosts is required"
            assert result.exit_code != 0
            assert "--hosts is required" not in result.output.lower()
            # Should attempt Phase 1
            assert "generate_rank_bindings" in result.output.lower() or "Phase 1" in result.output.lower()


class TestPhase2Helpers:
    """Test Phase 2 helper functions for generate_rank_bindings."""

    def test_get_generate_rank_bindings_output_paths(self, temp_dir):
        """Test get_generate_rank_bindings_output_paths returns correct paths."""
        output_dir = temp_dir / "output"
        rank_bindings_path, rankfile_path = get_generate_rank_bindings_output_paths(output_dir)

        assert rank_bindings_path == output_dir / "rank_bindings.yaml"
        assert rankfile_path == output_dir / "rankfile"

    def test_build_generate_rank_bindings_mpi_cmd_hosts(self, temp_dir):
        """Test build_generate_rank_bindings_mpi_cmd with hosts."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"
        hosts = ["node1", "node2", "node3"]

        cmd = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, hosts, output_dir)

        # Check MPI launcher
        assert cmd[0] in ["mpirun", "mpirun-ulfm"] or "mpirun" in cmd[0]
        # Check --host
        assert "--host" in cmd
        hosts_idx = cmd.index("--host")
        assert cmd[hosts_idx + 1] == "node1,node2,node3"
        # Check -np
        assert "-np" in cmd
        np_idx = cmd.index("-np")
        assert cmd[np_idx + 1] == "3"
        # Check executable and args
        assert str(executable.resolve()) in cmd
        assert "--mesh-graph-descriptor" in cmd
        assert str(mgd_path.resolve()) in cmd
        # Note: generate_rank_bindings doesn't accept --output-dir, it hardcodes output to "tt-run-generated/"

    def test_build_generate_rank_bindings_mpi_cmd_mock(self, temp_dir):
        """Test build_generate_rank_bindings_mpi_cmd with mock cluster."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"
        mock_desc0 = temp_dir / "mock0.yaml"
        mock_desc1 = temp_dir / "mock1.yaml"
        mock_desc0.touch()
        mock_desc1.touch()
        mock_rank_to_desc = {0: mock_desc0, 1: mock_desc1}

        cmd = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, None, output_dir, mock_rank_to_desc)

        # Check --oversubscribe flag (mock clusters don't use --host)
        assert "--oversubscribe" in cmd
        # Should NOT have --host (MPI defaults to localhost)
        assert "--host" not in cmd
        # Check per-rank -np 1 segments (not single -np 2)
        np_indices = [i for i, arg in enumerate(cmd) if arg == "-np"]
        assert len(np_indices) == 2  # One per rank
        assert cmd[np_indices[0] + 1] == "1"
        assert cmd[np_indices[1] + 1] == "1"
        # Check : separator between ranks
        assert ":" in cmd
        # Check per-rank TT_METAL_MOCK_CLUSTER_DESC_PATH (one per rank segment)
        env_vars = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-x"]
        mock_env_vars = [e for e in env_vars if "TT_METAL_MOCK_CLUSTER_DESC_PATH" in e]
        assert len(mock_env_vars) == 2
        # Each rank segment should have its own env var
        # Rank 0 env var should be before the : separator
        colon_idx = cmd.index(":")
        rank0_env_idx = next(i for i, arg in enumerate(cmd[:colon_idx]) if arg == "-x")
        assert str(mock_desc0.resolve()) in cmd[rank0_env_idx + 1]
        # Rank 1 env var should be after the : separator
        rank1_env_idx = next(i for i, arg in enumerate(cmd[colon_idx:]) if arg == "-x")
        assert str(mock_desc1.resolve()) in cmd[colon_idx + rank1_env_idx + 1]

    def test_build_generate_rank_bindings_mpi_cmd_no_hosts_no_mock(self, temp_dir):
        """Test build_generate_rank_bindings_mpi_cmd raises ValueError if neither hosts nor mock provided."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"

        with pytest.raises(ValueError, match="Either hosts or mock_rank_to_desc must be provided"):
            build_generate_rank_bindings_mpi_cmd(executable, mgd_path, None, output_dir, None)

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


class TestRankfileInjection:
    """Test rankfile injection functions."""

    def test_get_mpi_launcher(self):
        """Test get_mpi_launcher returns mpirun or mpirun-ulfm (may be full path)."""
        launcher = get_mpi_launcher()
        # Should be a string, not None
        assert isinstance(launcher, str)
        # shutil.which may return full path, so check basename
        launcher_basename = Path(launcher).name
        assert launcher_basename in ["mpirun", "mpirun-ulfm"]

    def test_build_rankfile_args_map_by(self, temp_dir):
        """Test build_rankfile_args with MAP_BY_RANKFILE_FILE syntax."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.MAP_BY_RANKFILE_FILE, rankfile)

        assert len(args) == 2
        assert args[0] == "--map-by"
        assert args[1] == f"rankfile:file={rankfile.resolve()}"

    def test_build_rankfile_args_rankfile(self, temp_dir):
        """Test build_rankfile_args with RANKFILE syntax."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.RANKFILE, rankfile)

        assert len(args) == 2
        assert args[0] == "--rankfile"
        assert args[1] == str(rankfile.resolve())

    def test_build_rankfile_args_mca(self, temp_dir):
        """Test build_rankfile_args with MCA_RMAPS_RANKFILE_PATH syntax."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.MCA_RMAPS_RANKFILE_PATH, rankfile)

        assert len(args) == 3
        assert args[0] == "--mca"
        assert args[1] == "rmaps_rankfile_path"
        assert args[2] == str(rankfile.resolve())

    def test_build_rankfile_args_invalid_syntax(self, temp_dir):
        """Test build_rankfile_args raises ValueError for invalid syntax."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        # Create a fake enum value
        class FakeSyntax:
            pass

        fake_syntax = FakeSyntax()

        with pytest.raises(ValueError, match="Unknown rankfile syntax"):
            build_rankfile_args(fake_syntax, rankfile)  # type: ignore

    def test_inject_rankfile_mpi_args(self, temp_dir):
        """Test inject_rankfile_mpi_args prepends rankfile args."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()
        base_args = ["--host", "node1,node2"]

        # Mock detect_rankfile_syntax to return MAP_BY_RANKFILE_FILE
        def mock_detect(launcher, subprocess_run=None):
            return RankfileSyntax.MAP_BY_RANKFILE_FILE

        result = inject_rankfile_mpi_args(rankfile, base_args, "mpirun", detect_fn=mock_detect)

        # Should prepend rankfile args
        assert result[0] == "--map-by"
        assert result[1] == f"rankfile:file={rankfile.resolve()}"
        assert result[2:] == base_args

    def test_legacy_flow_rankfile_conflict(self, runner, sample_rank_binding_yaml, temp_dir):
        """Test legacy_flow skips rankfile injection if already in mpi_args."""
        import subprocess

        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        with patch.object(subprocess, "run"):
            result = runner.invoke(
                main,
                [
                    "--rank-binding",
                    str(sample_rank_binding_yaml),
                    "--mpi-args",
                    "--rankfile /tmp/other_rankfile",
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
            # Should succeed but warn about conflict (if rankfile param was used)
            # Since we're not passing --rankfile param here, no conflict expected
            assert result.exit_code == 0
