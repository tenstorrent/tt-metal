# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ttrun command-line utility."""

import os
import shlex
import shutil
import sys
import yaml
import importlib
from pathlib import Path
from unittest.mock import patch
import pytest
from click.testing import CliRunner

from ttnn.distributed.ttrun import (
    main,
    parse_binding_config,
    resolve_path,
    get_rank_environment,
    build_mpi_command,
    get_mpi_launcher,
    RankfileSyntax,
    build_rankfile_args,
    detect_rankfile_syntax,
    mpi_args_specify_bind_to,
    normalize_rankfile_mpi_args,
    inject_rankfile_mpi_args,
    get_generate_rank_bindings_output_paths,
    build_generate_rank_bindings_mpi_cmd,
    host_csv_needs_oversubscribe,
    ensure_oversubscribe_for_multi_proc_per_host,
    run_phase1_generate_rank_bindings,
    find_generate_rank_bindings_executable,
    compute_phase1_cache_id,
    compute_phase1_cache_fingerprint_full,
    write_phase1_openmpi_hostfile,
    phase1_outputs_ready,
    phase1_cache_hit_valid,
    write_phase1_cache_key_file,
    read_stored_phase1_cache_key,
    PHASE2_MOCK_MAPPING_FILENAME,
    PHASE1_CACHE_ID_HEX_LEN,
    RANK_BINDINGS_MAPPING_FILENAME,
    FACTORY_SYSTEM_DESCRIPTOR_ENV_VAR,
)

# Import the module directly to avoid conflicts with distributed.py
ttrun_module = importlib.import_module("ttnn.distributed.ttrun")

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _no_rank_wrapper(monkeypatch):
    """Launch lines here are compared verbatim, so they never depend on whether tt-run-rank happens to be built; the
    wrapper's own tests set TT_RUN_RANK_WRAPPER themselves."""
    monkeypatch.setenv("TT_RUN_RANK_WRAPPER", "none")


@pytest.fixture
def unreaped_orphans():
    """The CI container's shape: its PID 1 (tail -f /dev/null) never reaps orphans, so a job's leftover that dies stays
    a zombie with its /proc entry. Making this process a child subreaper that never waits does the same on any host; on
    teardown the subreaper is turned off and the zombies are reaped."""
    import ctypes

    PR_SET_CHILD_SUBREAPER = 36
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    yield
    libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
    try:
        while os.waitpid(-1, os.WNOHANG)[0] > 0:
            pass
    except ChildProcessError:
        pass


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
        assert "do not combine" in result.output.lower()

    def test_no_mode_specified(self, runner):
        """Test that at least one mode must be specified."""
        result = runner.invoke(main, ["echo", "test"])
        assert result.exit_code != 0
        assert "specify exactly one" in result.output.lower()

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

    def test_hosts_rejects_duplicates(self, runner, sample_mesh_graph_descriptor):
        """Duplicate hostnames in --hosts are rejected."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "--hosts",
                "node1,node2,node1",
                "--dry-run",
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "duplicate" in result.output.lower()

    def test_hosts_all_empty_segments_rejected(self, runner, sample_mesh_graph_descriptor):
        """--hosts with only commas / empty entries is rejected."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "--hosts",
                ",,,",
                "--dry-run",
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "no valid hostname" in result.output.lower()

    def test_hosts_rejects_embedded_spaces(self, runner, sample_mesh_graph_descriptor):
        """Host tokens must not contain spaces."""
        result = runner.invoke(
            main,
            [
                "--mesh-graph-descriptor",
                str(sample_mesh_graph_descriptor),
                "--hosts",
                "node1,node 2",
                "--dry-run",
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0
        assert "space" in result.output.lower()

    def test_hosts_empty_segments_between_commas_ok(self, runner, sample_mesh_graph_descriptor):
        """Extra commas act as empty segments and are ignored."""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(sample_mesh_graph_descriptor),
                    "--hosts",
                    "node1,,node2",
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )
        assert result.exit_code == 0
        mock_run.assert_not_called()

    def test_new_mode_with_hosts(self, runner, sample_mesh_graph_descriptor):
        """New mode with --hosts and --dry-run: Phase 1 must not invoke subprocess.run."""
        import subprocess

        with patch.object(subprocess, "run") as mock_run:
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
            assert result.exit_code == 0
            mock_run.assert_not_called()

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

        # Mock subprocess.run; --dry-run skips Phase 1 so MPI is never invoked.
        with patch.object(subprocess, "run") as mock_run:
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
            assert result.exit_code == 0
            mock_run.assert_not_called()


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
        assert "--output-dir" in cmd
        assert str(output_dir.resolve()) in cmd

    def test_build_generate_rank_bindings_mpi_cmd_fsd_sets_env(self, temp_dir):
        """FSD path is exported to generate_rank_bindings via TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        fsd_path = temp_dir / "fsd.textproto"
        fsd_path.touch()
        output_dir = temp_dir / "output"
        hosts = ["node1", "node2"]

        cmd = build_generate_rank_bindings_mpi_cmd(
            executable, mgd_path, hosts, output_dir, factory_system_descriptor=fsd_path
        )

        assert f"{FACTORY_SYSTEM_DESCRIPTOR_ENV_VAR}={fsd_path}" in cmd
        # It is exported as an env var, not passed as a CLI flag to the tool.
        assert "--factory-system-descriptor" not in cmd

    def test_build_generate_rank_bindings_mpi_cmd_no_fsd_no_env(self, temp_dir):
        """Without an FSD, the env var is not injected."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"

        cmd = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, ["node1"], output_dir)

        assert not any(tok.startswith(f"{FACTORY_SYSTEM_DESCRIPTOR_ENV_VAR}=") for tok in cmd)

    def test_build_generate_rank_bindings_mpi_cmd_duplicate_hosts_adds_oversubscribe(self, temp_dir):
        """Phase 1: same hostname listed more than once requires --oversubscribe."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"
        hosts = ["node1", "node1", "node2", "node2"]

        cmd = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, hosts, output_dir)

        assert "--oversubscribe" in cmd
        hosts_idx = cmd.index("--host")
        assert cmd[hosts_idx + 1] == "node1,node1,node2,node2"
        np_idx = cmd.index("-np")
        assert cmd[np_idx + 1] == "4"

    def test_build_generate_rank_bindings_mpi_cmd_no_hosts_no_mock(self, temp_dir, expect_error):
        """Test build_generate_rank_bindings_mpi_cmd raises ValueError if neither hosts nor mock provided."""
        executable = temp_dir / "generate_rank_bindings"
        executable.touch()
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"

        with expect_error(ValueError, "Either hosts or mock_rank_to_desc must be provided"):
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

    def test_invalid_rank_binding_duplicate_ranks(self, temp_dir, expect_error):
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

        with expect_error(ValueError, "Duplicate ranks"):
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

    def test_resolve_path_not_found(self, temp_dir, expect_error):
        """Test that missing paths raise ValueError."""
        with expect_error(ValueError, "not found"):
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

    def test_cluster_id_comes_from_rank_binding_env_override(self, sample_rank_binding_yaml, monkeypatch):
        """Phase 2 stamps TT_METAL_CLUSTER_ID from the binding env override and drops a parent export."""
        monkeypatch.setenv("TT_METAL_CLUSTER_ID", "from-the-parent")
        config = parse_binding_config(sample_rank_binding_yaml)
        binding = config.rank_bindings[0]
        binding.env_overrides["TT_METAL_CLUSTER_ID"] = "bh-glx-110-c01u02"

        env = get_rank_environment(binding, config)

        assert env["TT_METAL_CLUSTER_ID"] == "bh-glx-110-c01u02"

    def test_cluster_id_not_taken_from_parent_without_override(self, sample_rank_binding_yaml, monkeypatch):
        """Without a rank-binding override, a parent TT_METAL_CLUSTER_ID is not forwarded."""
        monkeypatch.setenv("TT_METAL_CLUSTER_ID", "from-the-parent")
        config = parse_binding_config(sample_rank_binding_yaml)

        env = get_rank_environment(config.rank_bindings[0], config)

        assert "TT_METAL_CLUSTER_ID" not in env

    def test_get_rank_environment_fsd(self, sample_rank_binding_yaml, temp_dir):
        """FSD path on the config is exported to every rank via TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH."""
        config = parse_binding_config(sample_rank_binding_yaml)
        fsd_path = temp_dir / "fsd.textproto"
        config.factory_system_descriptor_path = fsd_path
        binding = config.rank_bindings[0]

        env = get_rank_environment(binding, config)

        assert env[FACTORY_SYSTEM_DESCRIPTOR_ENV_VAR] == str(fsd_path)

    def test_get_rank_environment_no_fsd(self, sample_rank_binding_yaml):
        """Without an FSD on the config, the env var is absent."""
        config = parse_binding_config(sample_rank_binding_yaml)
        binding = config.rank_bindings[0]

        env = get_rank_environment(binding, config)

        assert FACTORY_SYSTEM_DESCRIPTOR_ENV_VAR not in env


class TestOversubscribeHelpers:
    """--oversubscribe when multiple processes map to one host."""

    def test_host_csv_needs_oversubscribe_bare_duplicates(self):
        assert host_csv_needs_oversubscribe("a,b,c") is False
        assert host_csv_needs_oversubscribe("a,a,b") is True

    def test_host_csv_ignores_explicit_slots(self):
        assert host_csv_needs_oversubscribe("a:4,b:4") is False

    def test_ensure_oversubscribe_prepends_from_host_csv(self):
        args = ["--mca", "btl", "self,tcp", "--host", "h,h", "-np", "2"]
        out = ensure_oversubscribe_for_multi_proc_per_host(args)
        assert out[0] == "--oversubscribe"
        assert "--oversubscribe" not in out[1:] or out.count("--oversubscribe") == 1

    def test_ensure_oversubscribe_rankfile_hint(self, temp_dir):
        rf = temp_dir / "rf"
        rf.write_text("rank 0=host0 slot=0\nrank 1=host0 slot=1\n")
        args = ["--bind-to", "hwt"]
        out = ensure_oversubscribe_for_multi_proc_per_host(args, rankfile_hint=rf)
        assert out[0] == "--oversubscribe"


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

    def test_build_mpi_command_skips_bind_to_none_with_rankfile(self, sample_rank_binding_yaml, temp_dir):
        """Rankfile / map-by rankfile must not get default --bind-to none (Open MPI 5 + PRRTE conflict)."""
        config = parse_binding_config(sample_rank_binding_yaml)
        program = ["echo", "hello"]
        rankfile = temp_dir / "rf"
        rankfile.write_text("rank 0=h slot=0\n")
        mpi_args = ["--rankfile", str(rankfile)]
        cmd = build_mpi_command(config, program, mpi_args, rankfile_syntax=RankfileSyntax.RANKFILE)
        pairs = list(zip(cmd, cmd[1:]))
        assert ("--bind-to", "none") not in pairs


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

    def test_legacy_flow_single_process_works(self, runner, temp_dir):
        """Test legacy flow with single process works (multihost args are safe for single-process)."""
        import subprocess
        import yaml

        # Create rank binding with single rank
        rank_binding = temp_dir / "rank_binding.yaml"
        mgd_path = temp_dir / "mesh_graph.yaml"
        mgd_path.touch()

        yaml_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0},
            ],
            "global_env": {},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_binding, "w") as f:
            yaml.dump(yaml_content, f)

        with patch.object(subprocess, "run") as mock_subprocess:
            result = runner.invoke(
                main,
                [
                    "--rank-binding",
                    str(rank_binding),
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )

            assert result.exit_code == 0
            # Single-process runs should work fine even with multihost args
            # MPI handles single-process runs correctly regardless

    def test_legacy_mode_missing_fsd_errors(self, runner, sample_rank_binding_yaml, temp_dir):
        """Legacy mode validates the FSD path exists (regression: raw path was stored unvalidated)."""
        result = runner.invoke(
            main,
            [
                "--rank-binding",
                str(sample_rank_binding_yaml),
                "--factory-system-descriptor",
                str(temp_dir / "does_not_exist.textproto"),
                "echo",
                "test",
            ],
        )
        assert result.exit_code != 0


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
        """Test build_rankfile_args with MAP_BY_RANKFILE_FILE syntax (relpath from cwd)."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.MAP_BY_RANKFILE_FILE, rankfile, cwd=temp_dir)

        assert len(args) == 2
        assert args[0] == "--map-by"
        assert args[1] == "rankfile:file=rankfile"

    def test_build_rankfile_args_rankfile(self, temp_dir):
        """Test build_rankfile_args with RANKFILE syntax (relpath from cwd)."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.RANKFILE, rankfile, cwd=temp_dir)

        assert len(args) == 2
        assert args[0] == "--rankfile"
        assert args[1] == "rankfile"

    def test_build_rankfile_args_mca(self, temp_dir):
        """Test build_rankfile_args with MCA_RMAPS_RANKFILE_PATH syntax (relpath from cwd)."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        args = build_rankfile_args(RankfileSyntax.MCA_RMAPS_RANKFILE_PATH, rankfile, cwd=temp_dir)

        assert len(args) == 3
        assert args[0] == "--mca"
        assert args[1] == "rmaps_rankfile_path"
        assert args[2] == "rankfile"

    def test_build_rankfile_args_relpath_when_not_under_cwd(self, temp_dir):
        """Rankfile outside cwd uses relpath (avoids PRRTE map-by issues with absolute paths)."""
        rankfile = Path("/other/path/rankfile")

        args = build_rankfile_args(RankfileSyntax.RANKFILE, rankfile, cwd=temp_dir)

        assert args[0] == "--rankfile"
        assert args[1] == os.path.relpath("/other/path/rankfile", str(temp_dir.resolve()))

    def test_build_rankfile_args_invalid_syntax(self, temp_dir, expect_error):
        """Test build_rankfile_args raises ValueError for invalid syntax."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()

        # Create a fake enum value
        class FakeSyntax:
            pass

        fake_syntax = FakeSyntax()

        with expect_error(ValueError, "Unknown rankfile syntax"):
            build_rankfile_args(fake_syntax, rankfile, cwd=temp_dir)  # type: ignore

    def test_inject_rankfile_mpi_args(self, temp_dir):
        """Test inject_rankfile_mpi_args prepends rankfile args (relpath from cwd)."""
        rankfile = temp_dir / "rankfile"
        rankfile.touch()
        base_args = ["--host", "node1,node2"]

        # Mock detect_rankfile_syntax to return MAP_BY_RANKFILE_FILE
        def mock_detect(launcher, subprocess_run=None):
            return RankfileSyntax.MAP_BY_RANKFILE_FILE

        result = inject_rankfile_mpi_args(rankfile, base_args, "mpirun", detect_fn=mock_detect, cwd=temp_dir)

        # Should prepend rankfile args with relative path
        assert result[0] == "--map-by"
        assert result[1] == "rankfile:file=rankfile"
        assert result[2:] == base_args


class TestDetectRankfileSyntax:
    """Rankfile CLI form follows detected Open MPI version."""

    def test_detect_rankfile_syntax_ompi4_uses_rankfile(self):
        def _fake_run(cmd, **kwargs):
            return type("R", (), {"stdout": "mpirun (Open MPI) 4.1.4\n", "stderr": ""})()

        assert detect_rankfile_syntax("mpirun", _subprocess_run=_fake_run) == RankfileSyntax.RANKFILE

    def test_detect_rankfile_syntax_ompi5_uses_map_by(self):
        def _fake_run(cmd, **kwargs):
            return type("R", (), {"stdout": "mpirun (Open MPI) 5.0.0\n", "stderr": ""})()

        assert detect_rankfile_syntax("mpirun", _subprocess_run=_fake_run) == RankfileSyntax.MAP_BY_RANKFILE_FILE

    def test_detect_rankfile_syntax_unknown_falls_back_to_rankfile(self):
        def _fake_run(cmd, **kwargs):
            return type("R", (), {"stdout": "not open mpi\n", "stderr": ""})()

        assert detect_rankfile_syntax("mpirun", _subprocess_run=_fake_run) == RankfileSyntax.RANKFILE

    def test_detect_rankfile_syntax_explicit_path(self):
        def _fake_run(cmd, **kwargs):
            assert cmd[0] == "/usr/bin/mpirun"
            return type("R", (), {"stdout": "mpirun (Open MPI) 4.1.0\n", "stderr": ""})()

        assert detect_rankfile_syntax("/usr/bin/mpirun", _subprocess_run=_fake_run) == RankfileSyntax.RANKFILE

    def test_mpi_args_specify_bind_to(self):
        assert mpi_args_specify_bind_to(None) is False
        assert mpi_args_specify_bind_to([]) is False
        assert mpi_args_specify_bind_to(["--bind-to", "hwt:overload-allowed"]) is True
        assert mpi_args_specify_bind_to(["-bind-to", "core"]) is True
        assert mpi_args_specify_bind_to(["--bind-to=socket"]) is True

    def test_normalize_map_by_rankfile_file_becomes_rankfile(self, temp_dir):
        rf = temp_dir / "rankfile"
        rf.write_text("rank 0=h slot=0\n")
        out = normalize_rankfile_mpi_args(
            ["--map-by", f"rankfile:file={rf}"],
            RankfileSyntax.RANKFILE,
            cwd=temp_dir,
        )
        assert out[0] == "--rankfile"
        assert "rankfile:file=" not in shlex.join(out)


class TestFindGenerateRankBindingsExecutable:
    """Test finding generate_rank_bindings executable."""

    def test_find_generate_rank_bindings_executable_tt_metal_home(self, temp_dir, monkeypatch):
        """Test finding executable via TT_METAL_HOME."""
        executable_path = temp_dir / "build" / "tools" / "scaleout" / "generate_rank_bindings"
        executable_path.parent.mkdir(parents=True)
        executable_path.touch()

        monkeypatch.setenv("TT_METAL_HOME", str(temp_dir))
        result = find_generate_rank_bindings_executable()
        assert result == executable_path.resolve()


class TestPhase1CacheId:
    """Tests for Phase 1 cache fingerprint (content-based MGD, order-invariant hosts)."""

    def test_cache_id_is_short_hex(self, temp_dir):
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"x")
        cid = compute_phase1_cache_id(mgd, sorted(["single-host"]), None)
        assert len(cid) == PHASE1_CACHE_ID_HEX_LEN
        assert all(c in "0123456789abcdef" for c in cid)

    def test_fingerprint_full_is_sha256_hex(self, temp_dir):
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"x")
        fp = compute_phase1_cache_fingerprint_full(mgd, sorted(["h"]), None)
        assert len(fp) == 64
        assert fp[:PHASE1_CACHE_ID_HEX_LEN] == compute_phase1_cache_id(mgd, sorted(["h"]), None)

    def test_cache_id_same_bytes_and_sorted_hosts_stable(self, temp_dir):
        mgd_src = REPO_ROOT / "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto"
        if not mgd_src.is_file():
            pytest.skip(f"Mesh graph descriptor not present: {mgd_src}")
        mgd_a = temp_dir / "a.textproto"
        mgd_b = temp_dir / "subdir" / "b.proto"
        mgd_b.parent.mkdir(parents=True, exist_ok=True)
        data = mgd_src.read_bytes()
        mgd_a.write_bytes(data)
        mgd_b.write_bytes(data)
        hosts = ["zebra-0", "alpha-1", "moon-2"]
        assert compute_phase1_cache_id(mgd_a, sorted(hosts), None) == compute_phase1_cache_id(
            mgd_b, sorted(hosts), None
        )
        assert compute_phase1_cache_id(mgd_a, sorted(hosts), None) == compute_phase1_cache_id(
            mgd_a, sorted(reversed(hosts)), None
        )

    def test_cache_id_hosts_change_changes_id(self, temp_dir):
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"mesh-content-xyz")
        a = compute_phase1_cache_id(mgd, sorted(["h1", "h2"]), None)
        b = compute_phase1_cache_id(mgd, sorted(["h1", "h3"]), None)
        assert a != b

    def test_cache_id_mgd_content_change_changes_id(self, temp_dir):
        mgd1 = temp_dir / "m1.textproto"
        mgd2 = temp_dir / "m2.textproto"
        mgd1.write_bytes(b"aaa")
        mgd2.write_bytes(b"bbb")
        hosts = sorted(["n1"])
        assert compute_phase1_cache_id(mgd1, hosts, None) != compute_phase1_cache_id(mgd2, hosts, None)

    def test_fingerprint_changes_when_fsd_supplied_or_changed(self, temp_dir):
        """Supplying an FSD (and changing its content) invalidates the Phase 1 cache fingerprint."""
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"mesh")
        hosts = sorted(["n1"])
        fsd_a = temp_dir / "fsd_a.textproto"
        fsd_b = temp_dir / "fsd_b.textproto"
        fsd_a.write_bytes(b"fsd-a")
        fsd_b.write_bytes(b"fsd-b")

        no_fsd = compute_phase1_cache_fingerprint_full(mgd, hosts, None)
        with_a = compute_phase1_cache_fingerprint_full(mgd, hosts, None, factory_system_descriptor=fsd_a)
        with_b = compute_phase1_cache_fingerprint_full(mgd, hosts, None, factory_system_descriptor=fsd_b)

        assert no_fsd != with_a  # adding an FSD changes the fingerprint
        assert with_a != with_b  # different FSD content changes the fingerprint

    def test_cache_fingerprint_includes_all_mapped_mgd_contents(self, temp_dir):
        """Multi-MGD mapping fingerprint changes when any referenced .textproto changes."""
        mgd_a = temp_dir / "a.textproto"
        mgd_b = temp_dir / "b.textproto"
        mgd_a.write_bytes(b"mgd-a")
        mgd_b.write_bytes(b"mgd-b")
        mapping = temp_dir / "map.yaml"
        mapping.write_text("subcontext_id_to_mesh_graph_descriptor:\n" f"  0: {mgd_a.name}\n" f"  1: {mgd_b.name}\n")
        hosts = sorted(["n1"])
        fp0 = compute_phase1_cache_fingerprint_full(mapping, hosts, None)
        cid0 = compute_phase1_cache_id(mapping, hosts, None)
        mgd_a.write_bytes(b"mgd-a-v2")
        assert compute_phase1_cache_fingerprint_full(mapping, hosts, None) != fp0
        assert compute_phase1_cache_id(mapping, hosts, None) != cid0
        mgd_a.write_bytes(b"mgd-a")
        mgd_b.write_bytes(b"mgd-b-v2")
        assert compute_phase1_cache_fingerprint_full(mapping, hosts, None) != fp0
        assert compute_phase1_cache_id(mapping, hosts, None) != cid0

    def test_cache_id_mock_same_desc_bytes_different_paths(self, temp_dir):
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"mgd")
        body = b"mock:\n  chip: 0\n"
        p1 = temp_dir / "d1.yaml"
        p2 = temp_dir / "other" / "d2.yaml"
        p2.parent.mkdir(parents=True, exist_ok=True)
        p1.write_bytes(body)
        p2.write_bytes(body)
        a = compute_phase1_cache_id(mgd, None, {0: p1, 1: p2})
        b = compute_phase1_cache_id(mgd, None, {0: p2, 1: p1})
        assert a == b

    def test_cache_id_mock_desc_content_matters(self, temp_dir):
        mgd = temp_dir / "m.textproto"
        mgd.write_bytes(b"mgd")
        p1 = temp_dir / "d1.yaml"
        p2 = temp_dir / "d2.yaml"
        p1.write_bytes(b"a")
        p2.write_bytes(b"b")
        assert compute_phase1_cache_id(mgd, None, {0: p1}) != compute_phase1_cache_id(mgd, None, {0: p2})


class TestPhase1CacheArtifacts:
    """Hostfile and cache completeness helpers."""

    def test_write_phase1_openmpi_hostfile(self, temp_dir):
        hf = temp_dir / "hostfile"
        write_phase1_openmpi_hostfile(hf, ["b", "a"])
        assert hf.read_text().strip().splitlines() == ["b slots=1", "a slots=1"]

    def test_phase1_outputs_ready_real_cluster(self, temp_dir):
        run_dir = temp_dir / "r"
        run_dir.mkdir()
        rb, rf = get_generate_rank_bindings_output_paths(run_dir)
        rb.write_text("x")
        rf.write_text("y")
        assert phase1_outputs_ready(run_dir, mock_mode=False)

    def test_phase1_outputs_ready_multi_mgd_uses_rank_bindings_mapping_yaml(self, temp_dir):
        run_dir = temp_dir / "r"
        run_dir.mkdir()
        mapping = run_dir / RANK_BINDINGS_MAPPING_FILENAME
        _, rf = get_generate_rank_bindings_output_paths(run_dir)
        mapping.write_text("subcontext_id_to_rank_bindings:\n  0: rank_bindings_subctx_0.yaml\n")
        rf.write_text("y")
        rb_single = run_dir / "rank_bindings.yaml"
        assert not rb_single.exists()
        assert phase1_outputs_ready(run_dir, mock_mode=False)

    def test_phase1_outputs_ready_mock_needs_phase2_mapping(self, temp_dir):
        run_dir = temp_dir / "r"
        run_dir.mkdir()
        rb, rf = get_generate_rank_bindings_output_paths(run_dir)
        rb.write_text("x")
        rf.write_text("y")
        assert not phase1_outputs_ready(run_dir, mock_mode=True)
        (run_dir / PHASE2_MOCK_MAPPING_FILENAME).write_text("m: 1\n")
        assert phase1_outputs_ready(run_dir, mock_mode=True)

    def test_phase1_cache_hit_valid_requires_key_file(self, temp_dir):
        run_dir = temp_dir / "r"
        run_dir.mkdir()
        rb, rf = get_generate_rank_bindings_output_paths(run_dir)
        rb.write_text("x")
        rf.write_text("y")
        fp = "a" * 64
        assert not phase1_cache_hit_valid(run_dir, fp, mock_mode=False)
        write_phase1_cache_key_file(run_dir, fp)
        assert read_stored_phase1_cache_key(run_dir) == fp
        assert phase1_cache_hit_valid(run_dir, fp, mock_mode=False)

    def test_phase1_cache_hit_valid_wrong_fingerprint(self, temp_dir):
        run_dir = temp_dir / "r"
        run_dir.mkdir()
        rb, rf = get_generate_rank_bindings_output_paths(run_dir)
        rb.write_text("x")
        rf.write_text("y")
        write_phase1_cache_key_file(run_dir, "b" * 64)
        assert not phase1_cache_hit_valid(run_dir, "c" * 64, mock_mode=False)


class TestRunPhase1GenerateRankBindings:
    """Test Phase 1 generate_rank_bindings orchestration."""

    def test_run_phase1_generate_rank_bindings_success(self, temp_dir):
        """Test successful Phase 1 execution with file creation."""
        from unittest.mock import MagicMock

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        hosts = ["node1", "node2"]
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create output files after subprocess.run
        rank_bindings_path = output_dir / "rank_bindings.yaml"
        rankfile_path = output_dir / "rankfile"

        def mock_run(cmd, cwd=None, **kwargs):
            # Create output files to simulate successful execution
            rank_bindings_path.write_text("rank_bindings:\n  - rank: 0\n")
            rankfile_path.write_text("rank 0=node1 slot=0\n")
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        result_rank_bindings, result_rankfile = run_phase1_generate_rank_bindings(
            mgd_path, hosts, output_dir, subprocess_run=mock_run
        )

        assert result_rank_bindings == rank_bindings_path
        assert result_rankfile == rankfile_path

    def test_run_phase1_generate_rank_bindings_multi_mgd_returns_mapping_path(self, temp_dir):
        """Rank bindings mapping YAML is Phase 2 entry when Phase 1 used -M (MGD mapping)."""
        from unittest.mock import MagicMock

        mgd_path = temp_dir / "mapping.yaml"
        mgd_path.write_text("subcontext_id_to_mesh_graph_descriptor:\n  0: m.textproto\n")
        (temp_dir / "m.textproto").touch()
        hosts = ["node1"]
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        mapping_path = output_dir / RANK_BINDINGS_MAPPING_FILENAME
        rankfile_path = output_dir / "rankfile"

        def mock_run(cmd, cwd=None, **kwargs):
            mapping_path.write_text("subcontext_id_to_rank_bindings:\n  0: rank_bindings_subctx_0.yaml\n")
            rankfile_path.write_text("rank 0=node1 slot=0\n")
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        rb_out, rf_out = run_phase1_generate_rank_bindings(mgd_path, hosts, output_dir, subprocess_run=mock_run)
        assert rb_out == mapping_path
        assert rf_out == rankfile_path

    def test_run_phase1_generate_rank_bindings_failure(self, temp_dir, expect_error):
        """Test Phase 1 failure handling."""
        from unittest.mock import MagicMock

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        hosts = ["node1", "node2"]
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        def mock_run(cmd, cwd=None, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            return mock_result

        with expect_error(RuntimeError, "generate_rank_bindings failed"):
            run_phase1_generate_rank_bindings(mgd_path, hosts, output_dir, subprocess_run=mock_run)


class TestNewModeFlow:
    """Test new_mode_flow orchestration."""

    def test_new_mode_flow_success(self, runner, temp_dir, monkeypatch):
        """Test successful new_mode_flow execution."""
        from unittest.mock import patch

        monkeypatch.setattr(ttrun_module, "ORIGINAL_CWD", temp_dir)
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()

        # Mock run_phase1_generate_rank_bindings
        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        rank_bindings_path.write_text("rank_bindings:\n  - rank: 0\n")
        rankfile_path.write_text("rank 0=node1 slot=0\n")

        with patch.object(
            ttrun_module, "run_phase1_generate_rank_bindings", return_value=(rank_bindings_path, rankfile_path)
        ) as mock_phase1, patch.object(ttrun_module, "legacy_flow") as mock_legacy, patch.object(
            ttrun_module, "resolve_path", return_value=mgd_path
        ), patch.object(
            ttrun_module, "find_generate_rank_bindings_executable"
        ):
            runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(mgd_path),
                    "--hosts",
                    "node1,node2",
                    "echo",
                    "test",
                ],
            )

            # Verify Phase 1 was called
            assert mock_phase1.called
            # Verify legacy_flow was called with generated files
            assert mock_legacy.called
            call_args = mock_legacy.call_args
            assert call_args.kwargs["rank_binding"] == rank_bindings_path
            assert call_args.kwargs["rankfile"] == rankfile_path
            assert call_args.kwargs.get("phase2_failure_hint") is None

    def test_new_mode_flow_phase1_cache_hit(self, runner, temp_dir, monkeypatch):
        """When cache dir has rank_bindings and rankfile, Phase 1 is skipped."""
        from unittest.mock import patch

        monkeypatch.setattr(ttrun_module, "ORIGINAL_CWD", temp_dir)
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.write_bytes(b"mgd-for-cache-test")
        cache_id = compute_phase1_cache_id(mgd_path, sorted(["node1", "node2"]), None)
        run_dir = temp_dir / "generated" / "ttrun" / cache_id
        run_dir.mkdir(parents=True)
        mesh_graph_file = temp_dir / "mesh_graph.yaml"
        mesh_graph_file.touch()
        rank_bindings_content = {
            "rank_bindings": [{"rank": 0, "mesh_id": 0, "mesh_host_rank": 0}],
            "global_env": {},
            "mesh_graph_desc_path": str(mesh_graph_file),
        }
        with open(run_dir / "rank_bindings.yaml", "w") as f:
            yaml.dump(rank_bindings_content, f)
        (run_dir / "rankfile").write_text("rank 0=node1 slot=0\n")
        write_phase1_cache_key_file(
            run_dir, compute_phase1_cache_fingerprint_full(mgd_path, sorted(["node1", "node2"]), None)
        )

        with patch.object(ttrun_module, "run_phase1_generate_rank_bindings") as mock_p1, patch.object(
            ttrun_module, "legacy_flow"
        ) as mock_legacy, patch.object(ttrun_module, "resolve_path", return_value=mgd_path), patch.object(
            ttrun_module, "find_generate_rank_bindings_executable"
        ):
            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(mgd_path),
                    "--hosts",
                    "node2,node1",
                    "--dry-run",
                    "echo",
                    "test",
                ],
            )

        assert result.exit_code == 0
        assert not mock_p1.called
        assert mock_legacy.called
        assert mock_legacy.call_args.kwargs["rank_binding"] == run_dir / "rank_bindings.yaml"
        assert mock_legacy.call_args.kwargs["rankfile"] == run_dir / "rankfile"
        assert mock_legacy.call_args.kwargs.get("phase2_failure_hint") == "stale_phase1_cache"

    def test_new_mode_flow_force_rediscovery_runs_phase1_despite_cache_hit(self, runner, temp_dir, monkeypatch):
        """--force-rediscovery skips cache hit and invokes generate_rank_bindings."""
        from unittest.mock import patch

        monkeypatch.setattr(ttrun_module, "ORIGINAL_CWD", temp_dir)
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.write_bytes(b"mgd-for-force-rediscovery")
        cache_id = compute_phase1_cache_id(mgd_path, sorted(["node1", "node2"]), None)
        run_dir = temp_dir / "generated" / "ttrun" / cache_id
        run_dir.mkdir(parents=True)
        mesh_graph_file = temp_dir / "mesh_graph.yaml"
        mesh_graph_file.touch()
        rank_bindings_content = {
            "rank_bindings": [{"rank": 0, "mesh_id": 0, "mesh_host_rank": 0}],
            "global_env": {},
            "mesh_graph_desc_path": str(mesh_graph_file),
        }
        with open(run_dir / "rank_bindings.yaml", "w") as f:
            yaml.dump(rank_bindings_content, f)
        (run_dir / "rankfile").write_text("rank 0=node1 slot=0\n")
        write_phase1_cache_key_file(
            run_dir, compute_phase1_cache_fingerprint_full(mgd_path, sorted(["node1", "node2"]), None)
        )

        with patch.object(
            ttrun_module,
            "run_phase1_generate_rank_bindings",
            return_value=(run_dir / "rank_bindings.yaml", run_dir / "rankfile"),
        ) as mock_p1, patch.object(ttrun_module, "legacy_flow") as mock_legacy, patch.object(
            ttrun_module, "resolve_path", return_value=mgd_path
        ), patch.object(
            ttrun_module, "find_generate_rank_bindings_executable"
        ):
            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(mgd_path),
                    "--hosts",
                    "node1,node2",
                    "--force-rediscovery",
                    "echo",
                    "test",
                ],
            )

        assert result.exit_code == 0
        assert mock_p1.called
        assert mock_legacy.called
        assert mock_legacy.call_args.kwargs.get("phase2_failure_hint") is None

    def test_new_mode_flow_mock_cluster(self, runner, temp_dir, monkeypatch):
        """Test new_mode_flow with mock cluster rank binding."""
        from unittest.mock import patch

        monkeypatch.setattr(ttrun_module, "ORIGINAL_CWD", temp_dir)
        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()

        mock_binding_file = temp_dir / "mock_binding.yaml"
        mock_desc_0 = temp_dir / "mock0.yaml"
        mock_desc_1 = temp_dir / "mock1.yaml"
        mock_desc_0.touch()
        mock_desc_1.touch()

        mock_binding_content = {
            "rank_to_cluster_mock_cluster_desc": {
                "0": str(mock_desc_0),
                "1": str(mock_desc_1),
            }
        }
        with open(mock_binding_file, "w") as f:
            yaml.dump(mock_binding_content, f)

        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        rank_bindings_path.write_text("rank_bindings:\n  - rank: 0\n")
        rankfile_path.write_text("rank 0=localhost slot=0\n")

        with patch.object(
            ttrun_module, "run_phase1_generate_rank_bindings", return_value=(rank_bindings_path, rankfile_path)
        ) as mock_phase1, patch.object(ttrun_module, "legacy_flow") as mock_legacy, patch.object(
            ttrun_module, "resolve_path", side_effect=lambda p, **kwargs: Path(p) if Path(p).exists() else mgd_path
        ), patch.object(
            ttrun_module, "find_generate_rank_bindings_executable"
        ):
            result = runner.invoke(
                main,
                [
                    "--mesh-graph-descriptor",
                    str(mgd_path),
                    "--mock-cluster-rank-binding",
                    str(mock_binding_file),
                    "echo",
                    "test",
                ],
            )
            assert result.exit_code == 0
            assert not result.exception

            # Verify Phase 1 was called with mock_rank_to_desc
            assert mock_phase1.called
            call_args = mock_phase1.call_args
            assert call_args.kwargs["mock_rank_to_desc"] is not None
            assert len(call_args.kwargs["mock_rank_to_desc"]) == 2


class TestLauncherSelection:
    """tt-run must launch with the plain ``mpirun`` of the ULFM-capable install, never a ``--with-ft ulfm`` wrapper.

    Regression tests for #57698: under ``--with-ft ulfm`` a rank
    killed by a signal is "recoverable" and mpirun can exit 0 although every rank aborted.
    """

    @staticmethod
    def _make_executable(path: Path, content: str = "#!/bin/bash\nexit 0\n") -> Path:
        path.write_text(content)
        path.chmod(0o755)
        return path

    def test_wrapper_script_resolves_to_sibling_mpirun(self, temp_dir, monkeypatch):
        """Docker image layout: mpirun-ulfm is a bash wrapper beside mpirun that injects --with-ft ulfm."""
        bin_dir = temp_dir / "openmpi" / "bin"
        bin_dir.mkdir(parents=True)
        plain = self._make_executable(bin_dir / "mpirun")
        wrapper = self._make_executable(
            bin_dir / "mpirun-ulfm", '#!/bin/bash\nexec "$(dirname "$0")/mpirun" --with-ft ulfm "$@"\n'
        )
        monkeypatch.setattr(ttrun_module.shutil, "which", lambda name: str(wrapper) if name == "mpirun-ulfm" else None)

        assert get_mpi_launcher() == str(plain.resolve())

    def test_symlink_resolves_to_target_mpirun(self, temp_dir, monkeypatch):
        """Debian package layout: /usr/local/bin/mpirun-ulfm is a symlink to <prefix>/bin/mpirun."""
        bin_dir = temp_dir / "openmpi" / "bin"
        bin_dir.mkdir(parents=True)
        plain = self._make_executable(bin_dir / "mpirun")
        link_dir = temp_dir / "usr_local_bin"
        link_dir.mkdir()
        link = link_dir / "mpirun-ulfm"
        link.symlink_to(plain)
        monkeypatch.setattr(ttrun_module.shutil, "which", lambda name: str(link) if name == "mpirun-ulfm" else None)

        assert get_mpi_launcher() == str(plain.resolve())

    def test_wrapper_without_sibling_is_used_directly(self, temp_dir, monkeypatch):
        """No plain mpirun next to the wrapper: keep working, but with the wrapper itself."""
        wrapper = self._make_executable(temp_dir / "mpirun-ulfm")
        monkeypatch.setattr(ttrun_module.shutil, "which", lambda name: str(wrapper) if name == "mpirun-ulfm" else None)

        assert get_mpi_launcher() == str(wrapper)

    def test_falls_back_to_mpirun_when_ulfm_launcher_missing(self, monkeypatch):
        monkeypatch.setattr(ttrun_module.shutil, "which", lambda name: None)

        assert get_mpi_launcher() == "mpirun"

    def test_dangling_symlink_falls_back_to_launcher_name(self, temp_dir, monkeypatch):
        link = temp_dir / "mpirun-ulfm"
        link.symlink_to(temp_dir / "missing-mpirun")
        monkeypatch.setattr(ttrun_module.shutil, "which", lambda name: str(link) if name == "mpirun-ulfm" else None)

        assert get_mpi_launcher() == str(link)

    def test_user_with_ft_in_mpi_args_is_forwarded(self, runner, sample_rank_binding_yaml):
        """Opting into ULFM stays possible through --mpi-args."""
        import subprocess

        with patch.object(ttrun_module, "run_launcher", return_value=0) as mock_launch:
            result = runner.invoke(
                main,
                [
                    "--rank-binding",
                    str(sample_rank_binding_yaml),
                    "--bare",
                    "--mpi-args",
                    "--with-ft ulfm",
                    "echo",
                    "x",
                ],
            )
            assert result.exit_code == 0, result.output
            launched = mock_launch.call_args[0][0]
            assert Path(launched[0]).name != "mpirun-ulfm"
            assert launched[launched.index("--with-ft") + 1] == "ulfm"


@pytest.mark.timeout(60)  # under the regression a surviving rank waits forever; each case normally takes ~1s
@pytest.mark.skipif(
    shutil.which("mpirun-ulfm") is None and shutil.which("mpirun") is None,
    reason="requires an MPI launcher on PATH",
)
class TestRankFailurePropagation:
    """CPU-only end-to-end check: crashed ranks must surface as a non-zero tt-run exit code (#57698)."""

    @staticmethod
    def _mpi_args() -> str:
        args = ["--oversubscribe"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            args.append("--allow-run-as-root")
        return " ".join(args)

    def _run(self, runner, rank_binding_yaml, python_code):
        return runner.invoke(
            main,
            [
                "--rank-binding",
                str(rank_binding_yaml),
                "--bare",
                "--mpi-args",
                self._mpi_args(),
                sys.executable,
                "-c",
                python_code,
            ],
        )

    def test_all_ranks_aborted_by_signal_fails(self, runner, sample_rank_binding_yaml):
        result = self._run(runner, sample_rank_binding_yaml, "import os; os.abort()")
        assert result.exit_code != 0, "tt-run reported success although every rank was killed by SIGABRT"

    def test_one_rank_nonzero_exit_fails(self, runner, sample_rank_binding_yaml):
        code = "import os, sys; sys.exit(7 if os.environ.get('OMPI_COMM_WORLD_RANK') == '1' else 0)"
        result = self._run(runner, sample_rank_binding_yaml, code)
        assert result.exit_code != 0

    def test_all_ranks_succeed_passes(self, runner, sample_rank_binding_yaml):
        result = self._run(runner, sample_rank_binding_yaml, "pass")
        assert result.exit_code == 0, result.output


class TestJobOwnership:
    """tt-run owns the job (teardown plan P5, P6, P7, P19): the wall clock on the mpirun line when asked for, signals
    forwarded to the launcher, leftovers ended and reported."""

    def test_ttrun_adds_no_timeout_or_signal_forwarding_of_its_own(self, sample_rank_binding_yaml, tmp_path):
        from ttnn.distributed.ttrun import (
            build_generate_rank_bindings_mpi_cmd,
            build_mpi_command,
            parse_binding_config,
        )

        # The overall wall clock is PRRTE's own --timeout, reached only through --mpi-args; tt-run never adds one.
        # A stop reaches the ranks through PRRTE job control (tt-run-rank --uri), so neither launch line asks PRRTE to
        # forward a signal (the SIGUSR1 relay is gone).
        config = parse_binding_config(sample_rank_binding_yaml)
        cmd = build_mpi_command(config, ["echo"], mpi_args=["--oversubscribe"])
        assert "--timeout" not in cmd
        executable, mgd_path = tmp_path / "generate_rank_bindings", tmp_path / "mesh.textproto"
        executable.touch()
        mgd_path.touch()
        discovery = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, ["node1"], tmp_path / "output")
        for line in (cmd, discovery):
            assert not any("forward_signals" in a for a in line), line

    def test_user_timeout_in_mpi_args_is_passed_through(self, sample_rank_binding_yaml):
        from ttnn.distributed.ttrun import build_mpi_command, parse_binding_config

        # --mpi-args "--timeout N" reaches the mpirun line verbatim, exactly once, with nothing tt-run invents.
        config = parse_binding_config(sample_rank_binding_yaml)
        cmd = build_mpi_command(config, ["echo"], mpi_args=["--oversubscribe", "--timeout", "90"])
        assert cmd.count("--timeout") == 1
        assert cmd[cmd.index("--timeout") + 1] == "90"

    # prterun's own banners, captured from Open MPI 5.0.7 / PRRTE 3.0.8 (line wrapping as printed).
    _BANNER_SIGKILL = "prterun noticed that process rank 1 with PID 4242 on node nodeB exited on\nsignal 9 (Killed).\n"
    _BANNER_EXIT_CODE = (
        "prterun detected that one or more processes exited with non-zero status,\n"
        "thus causing the job to be terminated. The first process to do so was:\n\n"
        "   Process name: [prterun-nodeA-897012@1,1]\n   Exit code:    7\n"
    )
    _BANNER_MPI_ABORT = (
        "MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD\n  Proc: [[20522,1],1]\n  Errorcode: 7\n"
        'prterun has exited due to process rank 1 with PID 0 on node nodeB calling\n"abort".\n'
    )
    _BANNER_NO_FINALIZE = (
        "prterun has exited due to process rank 1 with PID 4242 on node nodeB exiting\nimproperly. There are three "
        'reasons this could occur:\n2. this process called "init", but exited without calling "finalize".\n'
    )

    @pytest.mark.parametrize(
        "banner,code,expected",
        [
            (_BANNER_SIGKILL, 137, r"rank 1 on nodeB \(pid 4242\) was killed by signal 9 \(SIGKILL\)"),
            (_BANNER_EXIT_CODE, 7, r"rank 1 exited with code 7"),
            (_BANNER_MPI_ABORT, 7, r"rank 1 on nodeB called MPI_Abort \(error code 7\)"),
            (_BANNER_NO_FINALIZE, 1, r"rank 1 on nodeB \(pid 4242\) exited without calling MPI_Finalize"),
            ("", 7, r"a rank failed \(launcher exit 7\)"),
        ],
        ids=["signal", "exit_code", "mpi_abort", "no_finalize", "no_banner"],
    )
    def test_run_launcher_passes_a_rank_failure_code_through_and_names_the_rank(self, banner, code, expected):
        import re

        from ttnn.distributed.ttrun import run_launcher

        # A launcher that reports a rank's failure with nothing left: the code is the truth, returned as is. tt-run
        # reads prterun's banner and says which rank failed (and where, when prterun says), how, and what happened
        # to the others; with no banner it still says a rank failed and points at the launcher's output.
        script = f"printf %s {shlex.quote(banner)} >&2; exit {code}"
        text, rc = self._logged_rc(lambda: run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == code
        assert re.search(expected, text), text
        if banner:
            assert "the launcher then killed the other ranks" in text and "tt-smi -r" in text, text
        assert f"job end: outcome=failed exit={code}" in text, text

    # The line a rank (DistributedContext::request_job_stop) or its wrapper (tt-run-rank) prints when it stops the
    # job, as prterun forwards it with --tag-output.
    @pytest.mark.parametrize(
        "reason,code",
        [
            ("sys.exit(3)", 3),
            ("uncaught Python exception RuntimeError: boom on rank 1", 1),
            ("uncaught C++ exception: TT_FATAL @ mesh_device.cpp: bad shape", 1),
            ("exited with code 7", 7),
            ("was killed by signal 9 (SIGKILL)", 137),
            ("received SIGTERM from pid 999 (not from the launcher)", 143),
        ],
        ids=["sys_exit", "py_exception", "cpp_exception", "exit_code", "sigkill_via_wrapper", "outside_sigterm"],
    )
    def test_run_launcher_names_the_rank_that_stopped_the_job_and_returns_its_code(self, reason, code):
        from ttnn.distributed.ttrun import run_launcher

        # Every rank exits after the stop (143 for the ones told to stop), but the job's code is the stopping rank's
        # and tt-run names it, even when prterun reports another rank first.
        line = f"[1,1]<stderr>: [tt-stop] rank 1 on nodeB (pid 4242): {reason}; stopping the job (exit {code})"
        script = f"echo {shlex.quote(line)} >&2; sleep 0.3; echo {shlex.quote(self._BANNER_EXIT_CODE)} >&2; exit 143"
        text, rc = self._logged_rc(lambda: run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == code, text
        assert f"rank 1 on nodeB stopped the job ({reason})" in text, text
        assert f"rank 1 on nodeB (pid 4242): {reason}; the other ranks were sent SIGTERM and exited" in text, text
        assert f"job end: outcome=failed exit={code} cause=rank failure" in text and "origin=rank 1 on nodeB" in text

    def test_run_launcher_names_the_dead_rank_not_the_peer_that_saw_it_fail(self):
        from ttnn.distributed.ttrun import run_launcher

        # Across hosts the survivor's MPI can report the dead peer before the dead rank's wrapper line arrives. A stop
        # whose reason is a failed peer is a symptom: the dead rank's own line replaces it, whatever the order.
        first = (
            "[1,0]<stderr>: [tt-stop] rank 0 on nodeA (pid 11): Python exception RuntimeError: MPI_Test failed: "
            "MPI_ERR_PROC_FAILED: Process Failure (closing the mesh); stopping the job (exit 1)"
        )
        second = "[1,1]<stderr>: [tt-stop] rank 1 on nodeB (pid 22): was killed by signal 11 (SIGSEGV); stopping the job (exit 139)"
        script = f"echo {shlex.quote(first)} >&2; sleep 0.3; echo {shlex.quote(second)} >&2; exit 1"
        text, rc = self._logged_rc(lambda: run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == 139, text
        assert "origin=rank 1 on nodeB" in text and "was killed by signal 11 (SIGSEGV)" in text, text

    def test_run_launcher_forces_a_job_a_rank_stopped_when_the_ranks_do_not_exit(self, monkeypatch):
        import time

        from ttnn.distributed import ttrun as m

        # A rank stopped the job but another rank never exits (stuck in a device wait): the stop grace applies as
        # for a stop at tt-run, then tt-run forces through the launcher. The code is still the stopping rank's.
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "1")
        line = "[1,0]<stderr>: [tt-stop] rank 0 on nodeA (pid 11): sys.exit(5); stopping the job (exit 5)"
        script = f"echo {shlex.quote(line)} >&2; trap 'exit 143' INT TERM; while :; do sleep 0.1; done"
        t0 = time.time()
        text, rc = self._logged_rc(lambda: m.run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == 5, text
        assert time.time() - t0 < 10
        assert "job forced down" in text and "ranks did not exit within 1 s" in text and "tt-smi -r" in text, text
        assert "the other ranks did not all exit in time and were killed" in text, text
        assert "job end: outcome=forced exit=5" in text, text

    def test_build_mpi_command_runs_each_rank_in_the_wrapper(self, sample_rank_binding_yaml, tmp_path, monkeypatch):
        # Each rank's program runs under tt-run-rank, which stops the job if the program dies; gdbserver launches
        # and TT_RUN_RANK_WRAPPER=none run the program bare.
        wrapper = tmp_path / "tt-run-rank"
        wrapper.write_text("#!/bin/sh\n")
        wrapper.chmod(0o755)
        config = parse_binding_config(sample_rank_binding_yaml)
        monkeypatch.setenv("TT_RUN_RANK_WRAPPER", str(wrapper))
        cmd = build_mpi_command(config, ["python3", "train.py"])
        n = len(config.rank_bindings)
        assert cmd.count(str(wrapper)) == n, cmd
        i = cmd.index(str(wrapper))
        assert cmd[i : i + 4] == [str(wrapper), "--", "python3", "train.py"], cmd
        assert str(wrapper) not in build_mpi_command(config, ["python3", "train.py"], debug_gdbserver=True)
        monkeypatch.setenv("TT_RUN_RANK_WRAPPER", "none")
        assert str(wrapper) not in build_mpi_command(config, ["python3", "train.py"])

    # A fake prterun for the stop tests: it publishes the PMIx tool rendezvous file the way prterun does
    # (<TMPDIR>/prterun.<host>.<pid>.<uid>/pmix.<host>.tool.<pid>, URI on the first line), then runs until its
    # "ranks" are gone (the fake helper creates REC/ranks_gone) or it is signalled. Every signal it gets is
    # recorded in REC/launcher.signals; IGNORE_TERM=1 makes it ignore the force (SIGINT) like a wedged launcher.
    _FAKE_PRTERUN = (
        'd="$TMPDIR/prterun.fakehost.$$.$(id -u)"; mkdir -p "$d"; '
        'printf "fake-uri-%s\\n5.0.5\\n%s\\n" $$ $$ > "$d/pmix.fakehost.tool.$$"; '
        'if [ "$IGNORE_TERM" = 1 ]; then trap "echo INT >> $REC/launcher.signals" INT; '
        'else trap "echo INT >> $REC/launcher.signals; exit 130" INT; fi; '
        'while [ ! -e "$REC/ranks_gone" ]; do sleep 0.1; done; exit 143'
    )

    @pytest.mark.parametrize("ranks_exit", [True, False], ids=["graceful", "forced_after_grace"])
    def test_run_launcher_stops_the_job_gracefully_at_its_timeout(self, ranks_exit, tmp_path, monkeypatch):
        import tempfile

        from ttnn.distributed import ttrun as m

        # --timeout 1: at 1 s tt-run sends SIGTERM to every rank through job control (never the launcher). If the
        # ranks exit, the job ends "stopped" with 241; if not, tt-run forces after the stop grace, with the warning.
        rec = tmp_path / "rec"
        rec.mkdir()
        helper = tmp_path / "tt-run-rank"
        helper.write_text(
            f'#!/bin/bash\necho "$@" >> "{rec}/helper.args"\n' + (f'touch "{rec}/ranks_gone"\n' if ranks_exit else "")
        )
        helper.chmod(0o755)
        monkeypatch.setenv("REC", str(rec))
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setenv("TT_RUN_SIGNAL_HELPER", str(helper))
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "1")
        monkeypatch.setenv("IGNORE_TERM", "0")
        text, rc = self._logged_rc(
            lambda: m.run_launcher(["bash", "-c", self._FAKE_PRTERUN], cwd=Path.cwd(), job_timeout=1.0)
        )
        assert rc == 241, text
        assert "the job reached its --job-timeout of 1 s: sent SIGTERM to all ranks" in text, text
        assert "--signal 15" in (rec / "helper.args").read_text()
        if ranks_exit:
            assert (
                "job stopped at its --job-timeout of 1 s" in text and "job end: outcome=stopped exit=241" in text
            ), text
            assert not (rec / "launcher.signals").exists(), "a graceful timeout must not signal the launcher"
        else:
            assert "job forced down" in text and "tt-smi -r" in text and "job end: outcome=forced exit=241" in text

    @pytest.mark.parametrize(
        "case",
        ["graceful", "no_helper", "grace_expired", "second_signal", "launcher_ignores_force"],
    )
    def test_run_launcher_stops_the_ranks_with_sigterm(self, case, tmp_path, monkeypatch):
        import signal
        import tempfile
        import threading
        import time

        from ttnn.distributed import ttrun as m

        # A stop at tt-run (here SIGTERM) sends SIGTERM to every rank through PRRTE job control (the
        # signal tool (tt-run-rank --uri), given the URI from prterun's rendezvous file) and never signals the launcher, so
        # PRRTE's 250 ms SIGKILL clock is not started. If the helper cannot reach prterun, the ranks do not exit
        # within the stop grace, or a second signal arrives, tt-run forces: SIGTERM to the launcher (PRRTE kills
        # every rank), then SIGKILL if the launcher ignores that too. tt-run reports 128 + the original signal.
        rec = tmp_path / "rec"
        rec.mkdir()
        helper = tmp_path / "tt-run-rank"
        delivers = case == "graceful"  # the "ranks" exit on the SIGTERM only in the graceful case
        helper.write_text(
            "#!/bin/bash\n" f'echo "$@" >> "{rec}/helper.args"\n' + (f'touch "{rec}/ranks_gone"\n' if delivers else "")
        )
        helper.chmod(0o755)
        monkeypatch.setenv("REC", str(rec))
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setenv(
            "TT_RUN_SIGNAL_HELPER",
            "/nonexistent/tt-run-rank" if case in ("no_helper", "launcher_ignores_force") else str(helper),
        )
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "1" if case == "grace_expired" else "60")
        monkeypatch.setenv("IGNORE_TERM", "1" if case == "launcher_ignores_force" else "0")
        monkeypatch.setattr(m, "KILL_GRACE_S", 1.0)

        running = threading.Event()  # signals go to this process only while run_launcher has its handlers installed

        def stop():
            for delay in (0.5, 1.0) if case == "second_signal" else (0.5,):
                time.sleep(delay)
                if running.is_set():
                    os.kill(os.getpid(), signal.SIGTERM)

        def launch():
            running.set()
            try:
                return m.run_launcher(["bash", "-c", self._FAKE_PRTERUN], cwd=Path.cwd())
            finally:
                running.clear()

        threading.Thread(target=stop, daemon=True).start()
        t0 = time.time()
        text, rc = self._logged_rc(launch)
        elapsed = time.time() - t0
        args = (rec / "helper.args").read_text() if (rec / "helper.args").exists() else ""
        launcher_signals = (rec / "launcher.signals").read_text() if (rec / "launcher.signals").exists() else ""

        assert rc == 128 + signal.SIGTERM, text
        assert elapsed < 10, f"stop took {elapsed:.1f}s\n{text}"
        if case in ("graceful", "grace_expired", "second_signal"):
            assert "--signal 15" in args and "--uri fake-uri-" in args, f"helper not called with SIGTERM: {args!r}"
            assert "sent SIGTERM to all ranks" in text, text
        if case == "graceful":
            assert launcher_signals == "", "a graceful stop must not signal the launcher (PRRTE's kill path)"
            assert "job end: outcome=stopped exit=143" in text, text
            return
        # every other case ends in a force: the launcher is signalled and the user is told the devices may be unhealthy
        assert "INT" in launcher_signals, "the force path did not signal the launcher (SIGINT, its kill path)"
        assert "job forced down" in text and "tt-smi -r" in text, text
        assert "job end: outcome=forced exit=143" in text, text
        why = {
            "no_helper": "could not reach prterun",
            "grace_expired": "ranks did not exit within 1 s",
            "second_signal": "second SIGTERM",
            "launcher_ignores_force": "could not reach prterun",
        }[case]
        assert why in text, text
        if case == "launcher_ignores_force":
            assert "did not exit" in text and "SIGKILL" in text, text

    def _fake_job(self, tmp_path, monkeypatch, stop_ends_ranks: bool = True):
        """A fake prterun (rendezvous file, ends once the "ranks" are gone) and a fake signal tool that records its
        arguments and, for SIGTERM, ends the ranks. Returns the record directory."""
        import tempfile

        rec = tmp_path / "rec"
        rec.mkdir()
        helper = tmp_path / "tt-run-rank"
        helper.write_text(
            f'#!/bin/bash\necho "$@" >> "{rec}/helper.args"\n'
            + (f'[ "$4" = 15 ] && touch "{rec}/ranks_gone"\n' if stop_ends_ranks else "")
            + "exit 0\n"
        )
        helper.chmod(0o755)
        monkeypatch.setenv("REC", str(rec))
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setenv("TT_RUN_SIGNAL_HELPER", str(helper))
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "60")
        monkeypatch.setenv("IGNORE_TERM", "0")
        return rec

    def test_run_launcher_ends_leftovers_on_a_ctrl_c_during_the_wait(self, monkeypatch, unreaped_orphans):
        import signal
        import threading
        import time

        from ttnn.distributed.ttrun import run_launcher

        # After a clean finish tt-run waits up to the stop grace for a process still exiting. A Ctrl-C during that wait
        # is still tt-run's (its handlers stay installed through the sweep): the leftover is ended at once, nothing
        # raises out of tt-run, and the job returns 130.
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "60")
        running = threading.Event()

        def interrupt():
            time.sleep(1.5)  # the launcher exited after 0.2 s; tt-run is in the sweep's wait
            if running.is_set():
                os.kill(os.getpid(), signal.SIGINT)

        def launch():
            running.set()
            try:
                return run_launcher(["bash", "-c", "sleep 60 & disown; sleep 0.2; exit 0"], cwd=Path.cwd())
            finally:
                running.clear()

        threading.Thread(target=interrupt, daemon=True).start()
        t0 = time.time()
        text, rc = self._logged_rc(launch)
        assert rc == 130, text
        assert time.time() - t0 < 15, f"the Ctrl-C did not end the wait: {time.time() - t0:.0f} s\n{text}"
        assert "straggler" in text and "outcome=stopped exit=130" in text, text

    @pytest.mark.parametrize("sig,reaches_launcher", [("SIGUSR1", True), ("SIGQUIT", False)], ids=["usr1", "quit"])
    def test_run_launcher_passes_signals_through_the_launcher_without_a_signal_tool(
        self, sig, reaches_launcher, tmp_path, monkeypatch
    ):
        import signal
        import threading
        import time

        from ttnn.distributed import ttrun as m

        # tt-run-rank built without PMIx: no job control. A signal prterun forwards to every rank itself (SIGUSR1,
        # SIGUSR2, SIGALRM, SIGTSTP, SIGCONT) is sent to the launcher instead; SIGQUIT, which would end prterun and
        # orphan the ranks, never is (tt-run says it could not pass it on).
        rec = tmp_path / "rec"
        rec.mkdir()
        monkeypatch.setenv("TT_RUN_SIGNAL_HELPER", "/nonexistent/tt-run-rank")
        fake = (
            f'trap "echo USR1 >> {rec}/launcher.signals" USR1; trap "echo QUIT >> {rec}/launcher.signals" QUIT; '
            "trap 'exit 130' INT; while :; do sleep 0.1; done"
        )
        running = threading.Event()

        def send():
            time.sleep(0.5)
            if running.is_set():
                os.kill(os.getpid(), getattr(signal, sig))
            time.sleep(1.0)
            if running.is_set():
                os.kill(os.getpid(), signal.SIGINT)  # no tool: the stop is forced (SIGINT to the launcher)

        def launch():
            running.set()
            try:
                return m.run_launcher(["bash", "-c", fake], cwd=Path.cwd())
            finally:
                running.clear()

        threading.Thread(target=send, daemon=True).start()
        text, rc = self._logged_rc(launch)
        got = (rec / "launcher.signals").read_text().split() if (rec / "launcher.signals").exists() else []
        assert rc == 130, text
        if reaches_launcher:
            assert got == ["USR1"] and "passed it on to all ranks through the launcher" in text, (got, text)
        else:
            assert got == [] and "could not pass it on to the ranks" in text, (got, text)

    def test_run_launcher_keeps_the_job_on_a_hangup(self, tmp_path, monkeypatch):
        import signal
        import threading
        import time

        from ttnn.distributed import ttrun as m

        # SIGHUP (the terminal closed, an ssh session dropped): the job keeps running with tt-run still supervising it
        # (nothing is sent to the ranks, tt-run says how to stop it); a later stop works as usual.
        rec = self._fake_job(tmp_path, monkeypatch)
        running = threading.Event()

        def send():
            time.sleep(0.5)
            if running.is_set():
                os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(1.0)
            if running.is_set():
                os.kill(os.getpid(), signal.SIGTERM)

        def launch():
            running.set()
            try:
                return m.run_launcher(["bash", "-c", self._FAKE_PRTERUN], cwd=Path.cwd())
            finally:
                running.clear()

        threading.Thread(target=send, daemon=True).start()
        text, rc = self._logged_rc(launch)
        calls = (rec / "helper.args").read_text().splitlines()
        assert rc == 128 + signal.SIGTERM, text
        assert [c.split()[-1] for c in calls] == ["15"], f"a hangup must not reach the ranks: {calls}"
        assert "SIGHUP received" in text and "the job keeps running" in text, text

    def test_run_launcher_drains_stderr_that_is_not_utf8(self):
        from ttnn.distributed.ttrun import run_launcher

        # One byte that is not UTF-8 on the launcher's stderr must not stop the pump: it keeps draining (300 KB more,
        # past any pipe buffer; a stopped drain blocks the launcher and `timeout` ends it with 124) and still reads a
        # rank's [tt-stop] line after it.
        script = (
            "printf 'bad \\377 byte\\n' >&2; head -c 300000 /dev/zero | tr '\\0' x >&2; echo >&2; "
            "printf '[tt-stop] rank 1 on h1 (pid 5): was killed by signal 9 (SIGKILL); stopping the job (exit 137)\\n' >&2; "
            "exit 137"
        )
        text, rc = self._logged_rc(lambda: run_launcher(["timeout", "20", "bash", "-c", script], cwd=Path.cwd()))
        assert rc == 137, f"the launcher blocked on a stderr nobody drained (124) or the code was lost: {rc}"
        assert "origin=rank 1 on h1" in text, text[-2000:]

    @pytest.mark.parametrize("sig", ["SIGUSR1", "SIGUSR2", "SIGALRM", "SIGQUIT", "SIGWINCH", "SIGCONT", "SIGTSTP"])
    def test_run_launcher_passes_other_signals_on_to_the_ranks(self, sig, tmp_path, monkeypatch):
        import signal
        import threading
        import time

        from ttnn.distributed import ttrun as m

        # A signal at tt-run that is not a stop goes to every rank unchanged through job control (the helper with that
        # signal number, never the launcher) and the job goes on; after SIGTSTP tt-run pauses itself too. A stop
        # (SIGTERM) then ends the job as usual.
        rec = self._fake_job(tmp_path, monkeypatch)
        paused = []
        monkeypatch.setattr(m.LaunchedJob, "_pause_self", staticmethod(lambda: paused.append(True)))
        signum = getattr(signal, sig)
        running = threading.Event()

        def send():
            time.sleep(0.5)
            if running.is_set():
                os.kill(os.getpid(), signum)
            time.sleep(0.7)
            if running.is_set():
                os.kill(os.getpid(), signal.SIGTERM)

        def launch():
            running.set()
            try:
                return m.run_launcher(["bash", "-c", self._FAKE_PRTERUN], cwd=Path.cwd())
            finally:
                running.clear()

        threading.Thread(target=send, daemon=True).start()
        text, rc = self._logged_rc(launch)
        calls = (rec / "helper.args").read_text().splitlines()

        assert rc == 128 + signal.SIGTERM, text
        assert [c.split()[-1] for c in calls] == [str(int(signum)), "15"], f"helper calls: {calls}\n{text}"
        assert all(c.startswith("--uri fake-uri-") for c in calls), calls
        assert f"{sig} received: passed it on to all ranks" in text, text
        assert not (rec / "launcher.signals").exists(), "a passed signal must not reach the launcher (its kill path)"
        assert paused == ([True] if sig == "SIGTSTP" else []), paused
        assert "job end: outcome=stopped exit=143" in text, text

    @pytest.mark.parametrize("guarded", [True, False], ids=["guard", "no_guard"])
    def test_a_tt_run_error_hands_the_running_job_to_its_guard(self, guarded, monkeypatch):
        import subprocess

        from ttnn.distributed import ttrun as m

        # tt-run leaves its with-block by an error of its own while the job runs. With a guard, tt-run closes the
        # guard's stdin without "done" and leaves the job alone: the guard stops it gracefully (SIGTERM to every rank,
        # the grace, the force), as for a killed tt-run. Without one, tt-run forces the launcher itself.
        monkeypatch.setattr(m, "KILL_GRACE_S", 1.0)
        job = m.LaunchedJob(["true"], Path("."))
        job.proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        read_fd = None
        if guarded:
            read_fd, job._guard_fd = os.pipe()
        try:
            text = self._logged(lambda: job.__exit__(RuntimeError, RuntimeError("a tt-run bug"), None))
            if guarded:
                assert os.read(read_fd, 16) == b"", "the guard was told the job is over; it must stop it instead"
                assert job.proc.poll() is None, "with a guard, tt-run must leave the running job to it"
                assert "its guard stops the job" in text and "RuntimeError" in text, text
            else:
                assert job.proc.poll() is not None, "without a guard, tt-run must force the launcher down"
        finally:
            if read_fd is not None:
                os.close(read_fd)
            if job.proc.poll() is None:
                job.proc.kill()
            job.proc.wait()

    def test_rank_wrapper_stops_its_rank_when_its_daemon_dies(self, monkeypatch, tmp_path):
        import signal
        import subprocess
        import time

        from ttnn.distributed import ttrun as m

        # P19: the launcher's daemon on a host is killed. The rank under it would run on forever (nothing can stop it
        # any more); its tt-run-rank wrapper sees it was reparented and sends its program SIGTERM.
        monkeypatch.delenv("TT_RUN_RANK_WRAPPER")  # the built one, not the autouse "none"
        wrapper = m.find_rank_wrapper()
        if wrapper is None:
            pytest.skip("tt-run-rank is not built")
        out = tmp_path / "out"
        program = 'trap "echo got TERM; exit 7" TERM; sleep 60 & wait'
        # the "daemon": a shell that starts the wrapper and waits on it
        daemon = subprocess.Popen(
            ["bash", "-c", f'"{wrapper}" -- bash -c \'{program}\' >"{out}" 2>&1 & echo $!; wait'],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        wrapper_pid = int(daemon.stdout.readline())
        try:
            time.sleep(0.5)
            os.kill(daemon.pid, signal.SIGKILL)
            daemon.wait()
            end = time.time() + 5
            while time.time() < end and m.LaunchedJob._still_running(wrapper_pid):
                time.sleep(0.1)
            assert not m.LaunchedJob._still_running(wrapper_pid), "the wrapper (and its rank) outlived its daemon"
            text = out.read_text()
            assert "is gone; stopping the rank (SIGTERM)" in text and "got TERM" in text, text
        finally:
            if m.LaunchedJob._still_running(wrapper_pid):
                os.killpg(daemon.pid, signal.SIGKILL)

    def test_run_launcher_fails_a_job_whose_launcher_lost_a_daemon(self):
        from ttnn.distributed.ttrun import run_launcher

        # prterun ends the job after losing a remote daemon but exits 0: tt-run fails the job and names the host.
        banner = (
            "PRTE has lost communication with a remote daemon.\n"
            "  HNP daemon   : [prterun-h0-1@0,0] on node h0\n"
            "  Remote daemon: [prterun-h0-1@0,4] on node h4\n"
        )
        script = f"printf %s {shlex.quote(banner)} >&2; exit 0"
        text, rc = self._logged_rc(lambda: run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == 1, text
        assert "lost its daemon on h4" in text and "job end: outcome=failed exit=1" in text, text

    @pytest.mark.parametrize("case", ["done", "ttrun_died", "ranks_ignore_the_stop", "no_helper"])
    def test_rank_wrapper_guard_stops_the_job_when_tt_run_dies(self, case, tmp_path, monkeypatch):
        import glob
        import subprocess
        import time

        from ttnn.distributed import ttrun as m

        # tt-run-rank --guard, started by tt-run next to the launcher: while tt-run lives (its "done" at the end) it
        # does nothing. When its stdin closes without "done" (tt-run was SIGKILLed, aborted, crashed) and the launcher
        # still runs, it stops the job as tt-run would: SIGTERM to every rank through job control, then, if the ranks
        # do not exit within the stop grace (or it cannot reach prterun), the force through the launcher.
        monkeypatch.delenv("TT_RUN_RANK_WRAPPER")  # the built one, not the autouse "none"
        wrapper = m.find_rank_wrapper()
        if wrapper is None:
            pytest.skip("tt-run-rank is not built")
        rec = self._fake_job(tmp_path, monkeypatch, stop_ends_ranks=case == "ttrun_died")
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "1")
        launcher = subprocess.Popen(["bash", "-c", self._FAKE_PRTERUN])
        try:
            end = time.time() + 10
            while not glob.glob(f"{tmp_path}/prterun.*.{launcher.pid}.*/pmix.*.tool.{launcher.pid}"):
                assert time.time() < end, "the fake launcher did not publish its rendezvous file"
                time.sleep(0.05)
            read_fd, write_fd = os.pipe()
            helper = [] if case == "no_helper" else [os.environ["TT_RUN_SIGNAL_HELPER"]]
            guard = subprocess.Popen(
                [wrapper, "--guard", str(launcher.pid)] + helper, stdin=read_fd, stderr=subprocess.PIPE, text=True
            )
            os.close(read_fd)
            if case == "done":
                os.write(write_fd, b"done\n")
            os.close(write_fd)
            _, said = guard.communicate(timeout=30)
            if case == "done":
                assert guard.returncode == 0 and said == "", said
                assert launcher.poll() is None, "the guard must leave a job whose tt-run said done alone"
                assert not (rec / "helper.args").exists()
                return
            launcher.wait(timeout=15)
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait()
        calls = (rec / "helper.args").read_text() if (rec / "helper.args").exists() else ""
        launcher_signals = (rec / "launcher.signals").read_text() if (rec / "launcher.signals").exists() else ""
        assert f"tt-run (pid {os.getpid()}) ended while its job was still running" in said, said
        if case == "ttrun_died":
            assert guard.returncode == 0, said
            assert "--signal 15" in calls and "--uri fake-uri-" in calls, calls
            assert "sent SIGTERM to all ranks" in said and "the job ended after the stop" in said, said
            assert launcher_signals == "", "a graceful stop must not signal the launcher (PRRTE's kill path)"
            return
        assert guard.returncode == 1, said
        assert "INT" in launcher_signals, "the force did not signal the launcher (SIGINT, its kill path)"
        assert "job forced down" in said and "tt-smi -r" in said, said
        why = {"ranks_ignore_the_stop": "ranks did not exit within the stop grace", "no_helper": "no signal tool"}
        assert why[case] in said, said

    def test_run_launcher_ends_leftovers_and_fails_the_job(self, monkeypatch, unreaped_orphans):
        import time

        from ttnn.distributed.ttrun import run_launcher

        # like PRRTE's ranks: a child that stays in the launcher's session (only the process group differs) and
        # OUTLASTS the stop grace -> warned about, then SIGTERM, then (here unneeded) SIGKILL; a leak, the job fails.
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "2")
        t0 = time.time()
        text, rc = self._logged_rc(
            lambda: run_launcher(["bash", "-c", "sleep 30 & disown; sleep 0.2; exit 0"], cwd=Path.cwd())
        )
        assert rc != 0
        assert time.time() - t0 < 20
        assert "still running" in text and "waiting up to 2 s (TT_RUN_STOP_GRACE_S)" in text, text
        assert "a leak" in text and "job end: outcome=failed" in text, text
        # the leftover dies on the SIGTERM; as an unreaped zombie it must count as ended at once, not after the SIGTERM
        # grace and a SIGKILL to a corpse (2 s stop grace + well under the 10 s SIGTERM grace)
        assert time.time() - t0 < 2 + 8, f"waited {time.time() - t0:.0f} s on a dead leftover\n{text}"

    @pytest.mark.parametrize("exits_after", [1, 12])
    def test_run_launcher_waits_for_a_rank_still_exiting_after_the_launcher(self, exits_after):
        import time

        from ttnn.distributed.ttrun import run_launcher

        # The SC4 case: every rank finished, but one is still in its exit path (driver teardown) when the launcher
        # returns. A clean finish is never killed: tt-run waits for it up to the stop grace (120 s by default; 12 s is
        # longer than the 10 s tt-run used to allow) and the clean code is forwarded.
        t0 = time.time()
        text, rc = self._logged_rc(
            lambda: run_launcher(["bash", "-c", f"sleep {exits_after} & disown; exit 0"], cwd=Path.cwd())
        )
        assert rc == 0, text
        assert exits_after <= time.time() - t0 < exits_after + 8
        assert "job end: outcome=clean exit=0" in text and "a leak" not in text, text

    def test_run_launcher_ends_a_forced_jobs_leftovers_without_another_grace(self, monkeypatch, unreaped_orphans):
        import signal
        import threading
        import time

        from ttnn.distributed.ttrun import run_launcher

        # The CI shape: tt-run forced the job (no signal tool) and the launcher returned with a rank still alive. The
        # stop grace was already spent, so the leftover is ended after the short settle, not after another grace.
        monkeypatch.setenv("TT_RUN_SIGNAL_HELPER", "/nonexistent/tt-run-rank")
        running = threading.Event()

        def stop():
            time.sleep(1.0)
            if running.is_set():
                os.kill(os.getpid(), signal.SIGTERM)

        def launch():
            running.set()
            try:
                return run_launcher(
                    ["bash", "-c", "sleep 300 & disown; trap 'exit 143' INT TERM; while :; do sleep 0.1; done"],
                    cwd=Path.cwd(),
                )
            finally:
                running.clear()

        threading.Thread(target=stop, daemon=True).start()
        t0 = time.time()
        text, rc = self._logged_rc(launch)
        assert rc == 128 + signal.SIGTERM, text
        assert time.time() - t0 < 30, f"took {time.time() - t0:.0f} s\n{text}"
        assert "job forced down" in text and "straggler" in text, text

    def test_run_launcher_does_not_wait_for_a_zombie(self, unreaped_orphans):
        import time

        from ttnn.distributed.ttrun import run_launcher

        # The CI shape: a rank of the job has exited but its parent has not reaped it yet (state Z). It is not a
        # survivor: the clean job ends at once instead of waiting the stop grace for it, and is not called a leak.
        script = "python3 -c 'import os, time; os.fork() or os._exit(0); time.sleep(4)' & disown; sleep 0.5; exit 0"
        t0 = time.time()
        text, rc = self._logged_rc(lambda: run_launcher(["bash", "-c", script], cwd=Path.cwd()))
        assert rc == 0, text
        assert time.time() - t0 < 8, f"waited {time.time() - t0:.0f} s for a zombie\n{text}"
        assert "a leak" not in text and "job end: outcome=clean exit=0" in text, text

    @pytest.mark.parametrize("site_include", [None, "eth3"], ids=["no_site_choice", "site_include"])
    def test_default_tcp_exclude_yields_to_the_environment(self, site_include, sample_rank_binding_yaml, monkeypatch):
        from click.testing import CliRunner

        from ttnn.distributed.ttrun import main

        # tt-run's default (exclude lo,docker0) applies only when nothing chose the TCP interface. A site profile that
        # exports OMPI_MCA_btl_tcp_if_include (the exabox hosts) keeps its choice: replacing it made every rank on SC4
        # pick an interface the others could not reach (MPI_Comm_dup: MPI_ERR_PROC_FAILED).
        monkeypatch.delenv("OMPI_MCA_btl_tcp_if_exclude", raising=False)
        if site_include:
            monkeypatch.setenv("OMPI_MCA_btl_tcp_if_include", site_include)
        else:
            monkeypatch.delenv("OMPI_MCA_btl_tcp_if_include", raising=False)
        result = CliRunner().invoke(main, ["--rank-binding", str(sample_rank_binding_yaml), "--dry-run", "echo", "x"])
        assert result.exit_code == 0, result.output
        assert ("btl_tcp_if_exclude" in result.output) == (site_include is None), result.output

    def test_the_prterun_address_is_read_only_from_this_users_own_files(self, tmp_path, monkeypatch):
        import tempfile

        from ttnn.distributed.ttrun import LaunchedJob

        # The tmpdir is shared: a decoy another user plants (their uid in the name) or a symlink with this user's name
        # must not point tt-run's stop at another server. Only this user's own directory and file count.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        pid, uid = 4242, os.getuid()
        decoy = tmp_path / f"prterun.h.{pid}.{uid + 1}"
        decoy.mkdir()
        (decoy / f"pmix.h.tool.{pid}").write_text("tcp4://decoy:1\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / f"pmix.h.tool.{pid}").write_text("tcp4://symlinked:1\n")
        (tmp_path / f"prterun.h.{pid}.{uid}").symlink_to(elsewhere)
        assert LaunchedJob.prterun_tool_uri(pid) is None
        real = tmp_path / f"prterun.h2.{pid}.{uid}"
        real.mkdir()
        (real / f"pmix.h2.tool.{pid}").write_text("tcp4://real:1\n")
        assert LaunchedJob.prterun_tool_uri(pid) == "tcp4://real:1"

    def test_launch_env_drops_a_tcp_interface_variable_the_line_contradicts(self, monkeypatch):
        from ttnn.distributed.ttrun import LaunchedJob

        launch_env = LaunchedJob.launch_env

        # A site profile exports an exclude list; --tcp-interface (or --mpi-args) puts an include list on the line.
        # Open MPI aborts on both, so the environment's one is dropped; an unrelated environment is left alone.
        monkeypatch.setenv("OMPI_MCA_btl_tcp_if_exclude", "lo,docker0")
        line = ["mpirun", "--mca", "btl", "self,tcp", "--mca", "btl_tcp_if_include", "eth0", "prog"]
        assert "OMPI_MCA_btl_tcp_if_exclude" not in launch_env(line)
        assert launch_env(["mpirun", "prog"])["OMPI_MCA_btl_tcp_if_exclude"] == "lo,docker0"
        monkeypatch.delenv("OMPI_MCA_btl_tcp_if_exclude")
        monkeypatch.setenv("OMPI_MCA_btl_tcp_if_include", "eth0")
        assert "OMPI_MCA_btl_tcp_if_include" not in launch_env(["mpirun", "--mca", "btl_tcp_if_exclude", "lo", "p"])

    def test_launch_presets_the_variables_mpi_init_would_add(self):
        from ttnn.distributed.ttrun import MPI_INIT_PRESET_ENV, LaunchedJob

        launch_env, with_mpi_init_preset_env = LaunchedJob.launch_env, LaunchedJob.with_mpi_init_preset_env

        # tt-metal #64717: MPI_Init adding these to the environment races Tracy's getenv. Every rank gets them already
        # set, through -x on the launch line and in the launcher's own environment (CI crashed in a Phase 1 rank with
        # the -x copy alone), and a value the user set is left alone.
        line = with_mpi_init_preset_env(["mpirun", "--tag-output", "-np", "1", "prog"])
        assert line[:7] == [
            "mpirun",
            "-x",
            "PMIX_MCA_psec=native",
            "-x",
            "IPATH_NO_BACKTRACE=1",
            "-x",
            "HFI_NO_BACKTRACE=1",
        ], line
        assert line[-4:] == ["--tag-output", "-np", "1", "prog"]
        mine = ["mpirun", "-x", "HFI_NO_BACKTRACE=0", "prog"]
        assert with_mpi_init_preset_env(mine).count("-x") == 3 and "HFI_NO_BACKTRACE=0" in with_mpi_init_preset_env(
            mine
        )
        env = launch_env(["mpirun", "prog"])
        assert all(env[k] == v for k, v in MPI_INIT_PRESET_ENV.items())

    @pytest.mark.parametrize("where", ["tt_run", "mpi_args"])
    def test_a_timeout_is_tt_runs_graceful_one_or_prrtes_own(self, where, sample_rank_binding_yaml, monkeypatch):
        from click.testing import CliRunner

        from ttnn.distributed import ttrun as m

        # tt-run's --job-timeout is its graceful clock (handed to the launcher, never on the mpirun line). A --timeout in
        # --mpi-args is PRRTE's own: it reaches mpirun unchanged, with a warning that it kills every rank at once. A
        # rank program's own --timeout (pytest's) is never read.
        seen = {}
        monkeypatch.setattr(
            m, "run_launcher", lambda cmd, cwd, job_timeout=None: seen.update(cmd=cmd, t=job_timeout) or 0
        )
        args = ["--job-timeout", "30"] if where == "tt_run" else ["--mpi-args", "--timeout 30"]
        result = CliRunner().invoke(
            m.main, ["--rank-binding", str(sample_rank_binding_yaml), *args, "pytest", "x.py", "--timeout", "600"]
        )
        assert result.exit_code == 0, result.output
        line = seen["cmd"]
        values = [line[i + 1] for i, x in enumerate(line[:-1]) if x == "--timeout"]
        assert line[-2:] == ["--timeout", "600"] and "600" in values, line  # pytest's own, on every rank, untouched
        if where == "tt_run":
            assert seen["t"] == 30.0 and "30" not in values, line
        else:
            assert seen["t"] is None and values.count("30") == 1, line
        assert m.mpi_args_timeout(["--oversubscribe", "--timeout", "20"]) == "20"
        assert m.mpi_args_timeout(["--timeout=7"]) == "7" and m.mpi_args_timeout(["-np", "2"]) is None

    def test_run_launcher_preserves_prrtes_timeout_exit_code(self):
        from ttnn.distributed.ttrun import PRTE_TIMEOUT_EXIT_CODE, run_launcher

        # PRRTE's own --timeout (from --mpi-args) killed every rank: tt-run keeps its 241 and says the job was forced
        # down and the devices may need a reset.
        text, rc = self._logged_rc(
            lambda: run_launcher(["bash", "-c", f"exit {PRTE_TIMEOUT_EXIT_CODE}"], cwd=Path.cwd())
        )
        assert rc == PRTE_TIMEOUT_EXIT_CODE
        assert "job forced down" in text and "PRRTE --timeout" in text and "tt-smi -r" in text, text
        assert f"job end: outcome=forced exit={PRTE_TIMEOUT_EXIT_CODE}" in text, text

    @staticmethod
    def _logged(fn):
        """Run fn and return what tt-run logged (loguru, so caplog does not see it)."""
        from loguru import logger

        lines = []
        sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
        try:
            fn()
        finally:
            logger.remove(sink)
        return "".join(lines)

    @classmethod
    def _logged_rc(cls, fn):
        """Like _logged, but also return fn's result (run_launcher's exit code)."""
        out = {}
        text = cls._logged(lambda: out.setdefault("rc", fn()))
        return text, out.get("rc")

    def test_log_phase2_exit_reason_names_a_signal_death_and_skips_the_generic_hint(self):
        import signal

        from ttnn.distributed.ttrun import PRTE_TIMEOUT_EXIT_CODE, log_phase2_exit_reason

        # A rank died by a signal: one line names the signal and the code; the cache / descriptor hint is not logged,
        # because a dead rank is not a stale cache.
        text = self._logged(lambda: log_phase2_exit_reason(128 + signal.SIGKILL, [], "legacy"))
        assert "died by signal 9 (SIGKILL)" in text
        assert "returns 137" in text
        assert "golden" not in text.lower()
        # The wall clock: run_launcher already said so; nothing more is logged.
        assert self._logged(lambda: log_phase2_exit_reason(PRTE_TIMEOUT_EXIT_CODE, [], "legacy")) == ""

    def test_log_phase2_exit_reason_names_a_missing_tcp_interface(self):
        from ttnn.distributed.ttrun import log_phase2_exit_reason

        ttrun_module = sys.modules["ttnn.distributed.ttrun"]
        # The ranks were pinned to an interface this host does not have (a typo in --tcp-interface, or a hand-written
        # --mca btl_tcp_if_include): MPI_Init fails on every rank and the launcher returns 14. tt-run says which
        # interface, which ones exist, and what to change, instead of the generic hint.
        with patch.object(ttrun_module, "get_local_network_interfaces", return_value={"eth0", "lo"}):
            text = self._logged(
                lambda: log_phase2_exit_reason(14, ["--mca", "btl_tcp_if_include", "nosuchif"], "legacy")
            )
            assert "TCP interface 'nosuchif'" in text
            assert "eth0, lo" in text
            assert "golden" not in text.lower()
            # The same code with an interface that exists is not an interface problem: the generic hint stays.
            text = self._logged(lambda: log_phase2_exit_reason(14, ["--mca", "btl_tcp_if_include", "eth0"], "legacy"))
            assert "TCP interface" not in text
            assert "golden" in text.lower()

    def test_sweep_removes_a_stale_shm_segment_but_keeps_a_mapped_one(self):
        import mmap
        import socket
        import time

        from ttnn.distributed.ttrun import LaunchedJob

        remove_stale_launcher_leftovers = LaunchedJob.remove_stale_launcher_leftovers

        # The third piece of P11: before launching, tt-run removes this user's Open MPI shared-memory segments
        # that no live process maps (what a killed job leaves behind), and keeps the ones a running job still maps.
        host, uid = socket.gethostname(), os.getuid()
        stale = Path(f"/dev/shm/sm_segment.{host}.{uid}.ttrun_pytest_stale_{os.getpid()}")
        live = Path(f"/dev/shm/sm_segment.{host}.{uid}.ttrun_pytest_live_{os.getpid()}")
        fresh = Path(f"/dev/shm/sm_segment.{host}.{uid}.ttrun_pytest_fresh_{os.getpid()}")
        mm = fd = None
        try:
            stale.write_bytes(b"\0" * 4096)  # a leftover nobody maps
            os.utime(stale, (time.time() - 3600, time.time() - 3600))  # aged past SHM_MIN_STALE_AGE_S: a genuine leak
            fresh.write_bytes(b"\0" * 4096)  # unmapped but just created, like a concurrent job before it maps it
            fd = os.open(str(live), os.O_RDWR | os.O_CREAT, 0o600)  # a segment this process maps, like a live job
            os.ftruncate(fd, 4096)
            mm = mmap.mmap(fd, 4096)

            remove_stale_launcher_leftovers()

            assert not stale.exists(), "the aged, unmapped segment was not removed"
            assert live.exists(), "a segment a live process maps must be kept"
            assert fresh.exists(), "a just-created unmapped segment (uncertain ownership) must be preserved"
        finally:
            if mm is not None:
                mm.close()
            if fd is not None:
                os.close(fd)
            for p in (stale, live, fresh):
                try:
                    p.unlink()
                except OSError:
                    pass

    def test_sweep_removes_a_stale_prterun_session_dir_but_keeps_a_live_one(self):
        import tempfile

        from ttnn.distributed.ttrun import LaunchedJob

        remove_stale_launcher_leftovers = LaunchedJob.remove_stale_launcher_leftovers

        # P25: a killed launcher never runs prterun's exit path, so its <tmp>/prterun.<host>.<pid>.<uid> directory
        # stays. The same sweep removes the ones whose launcher pid is dead and keeps a live job's directory.
        uid = os.getuid()
        tmp = Path(tempfile.gettempdir())
        dead = tmp / f"prterun.ttrun-pytest.2147483646.{uid}"  # a pid past pid_max: never alive
        live = tmp / f"prterun.ttrun-pytest.{os.getpid()}.{uid}"  # this process: alive
        try:
            dead.mkdir()
            (dead / "hwloc.sm").write_bytes(b"\0")
            live.mkdir()
            (live / "hwloc.sm").write_bytes(b"\0")

            remove_stale_launcher_leftovers()

            assert not dead.exists(), "a session dir whose launcher pid is dead was not removed"
            assert live.exists(), "a session dir whose launcher pid is alive must be kept"
        finally:
            for p in (dead, live):
                shutil.rmtree(p, ignore_errors=True)

    def test_wait_does_not_force_down_a_running_job_with_no_forwarded_signal(self, monkeypatch, tmp_path):
        import subprocess

        from ttnn.distributed import ttrun as m

        # The safety property: with no stop requested, wait() never force-kills and never sends a SIGTERM to the
        # ranks — it just returns the launcher's own code. A healthy job (multi-host included) that nobody asked
        # to stop is never touched, even past the stop grace.
        calls = tmp_path / "helper.args"
        helper = tmp_path / "tt-run-rank"
        helper.write_text(f'#!/bin/bash\necho "$@" >> "{calls}"\n')
        helper.chmod(0o755)
        monkeypatch.setenv("TT_RUN_SIGNAL_HELPER", str(helper))
        monkeypatch.setenv("TT_RUN_STOP_GRACE_S", "1")
        monkeypatch.setattr(m, "KILL_GRACE_S", 1.0)
        job = m.LaunchedJob(["true"], Path("."))
        job.proc = subprocess.Popen(["sleep", "3"], start_new_session=True)  # exits on its own; no stop requested
        try:
            rc = job.wait()
            assert rc == 0, "a job nobody asked to stop must return the launcher's own code"
            assert not calls.exists(), "the ranks were sent a SIGTERM although nobody asked to stop the job"
        finally:
            if job.proc.poll() is None:
                job.proc.kill()
                job.proc.wait()

    def test_wait_ends_a_launcher_that_lingers_after_an_mpi_abort(self, monkeypatch):
        import time

        from ttnn.distributed import ttrun as m

        # prrte#2815 guard: a launcher that printed the MPI_ABORT banner and then hangs (here: ignores SIGTERM and
        # never exits) must be ended by wait(), which returns the rank's errorcode — not left to block forever.
        monkeypatch.setattr(m, "KILL_GRACE_S", 1.0)
        fake = (
            "import sys,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stderr.write('MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD\\n'); "
            "sys.stderr.flush(); time.sleep(600)"
        )
        with m.LaunchedJob([sys.executable, "-c", fake], Path(".")) as job:
            t0 = time.time()
            rc = job.wait()
            assert rc != 0, f"a job whose launcher aborted-then-hung must fail, got {rc}"
            assert job.proc.poll() is not None, "the lingering launcher was not ended"
            assert time.time() - t0 < 20, "the abort guard did not fire promptly"

    def test_wait_does_not_end_a_launcher_without_an_mpi_abort(self, monkeypatch):
        import time

        from ttnn.distributed import ttrun as m

        # The safety property: no MPI_Abort banner → the abort guard never arms. A launcher that just runs a while
        # and exits on its own (here past the grace, writing ordinary stderr) is returned, never force-ended.
        monkeypatch.setattr(m, "KILL_GRACE_S", 1.0)
        fake = (
            "import sys,time; sys.stderr.write('ordinary diagnostic, no abort\\n'); sys.stderr.flush(); time.sleep(3)"
        )
        with m.LaunchedJob([sys.executable, "-c", fake], Path(".")) as job:
            rc = job.wait()
            assert rc == 0, "a launcher that never aborted must return its own code, not be force-ended"
