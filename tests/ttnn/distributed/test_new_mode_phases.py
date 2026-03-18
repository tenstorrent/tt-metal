# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for new mode Phase 1 and Phase 2 MPI commands."""

import yaml
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from ttnn.distributed.ttrun import (
    main,
    run_phase1_generate_rank_bindings,
)


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class TestNewModePhase1Phase2:
    """Test that Phase 1 and Phase 2 use correct MPI commands."""

    def test_phase1_real_cluster_correct_np_and_hosts(self, temp_dir):
        """Test Phase 1 command has correct -np and --host for real cluster."""
        import subprocess

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        hosts = ["node1", "node2", "node3"]
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Mock subprocess.run to capture the command
        captured_cmd = []

        def mock_run(cmd, cwd=None, **kwargs):
            captured_cmd.extend(cmd)
            # Create mock output files
            rank_bindings_path = output_dir / "rank_bindings.yaml"
            rankfile_path = output_dir / "rankfile"
            rank_bindings_path.touch()
            rankfile_path.touch()
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch.object(subprocess, "run", side_effect=mock_run):
            with patch("time.sleep"):  # Skip sleep in tests
                run_phase1_generate_rank_bindings(
                    mgd_path, hosts, output_dir, subprocess_run=subprocess.run, sleep_secs=0
                )

        # Verify Phase 1 command structure
        assert "--host" in captured_cmd
        hosts_idx = captured_cmd.index("--host")
        assert captured_cmd[hosts_idx + 1] == "node1,node2,node3"

        assert "-np" in captured_cmd
        np_idx = captured_cmd.index("-np")
        assert captured_cmd[np_idx + 1] == "3"  # len(hosts)

        # Should have executable and args
        assert "--mesh-graph-descriptor" in captured_cmd
        # Note: generate_rank_bindings accepts --output-dir (default: generated/ttrun)

    def test_phase1_mock_cluster_per_rank_segments(self, temp_dir):
        """Test Phase 1 command uses per-rank -np 1 segments for mock cluster."""
        import subprocess

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        mock_desc0 = temp_dir / "mock0.yaml"
        mock_desc1 = temp_dir / "mock1.yaml"
        mock_desc2 = temp_dir / "mock2.yaml"
        mock_desc0.touch()
        mock_desc1.touch()
        mock_desc2.touch()
        mock_rank_to_desc = {0: mock_desc0, 1: mock_desc1, 2: mock_desc2}

        # Mock subprocess.run to capture the command
        captured_cmd = []

        def mock_run(cmd, cwd=None, **kwargs):
            captured_cmd.extend(cmd)
            # Create mock output files
            rank_bindings_path = output_dir / "rank_bindings.yaml"
            rankfile_path = output_dir / "rankfile"
            rank_bindings_path.touch()
            rankfile_path.touch()
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch.object(subprocess, "run", side_effect=mock_run):
            with patch("time.sleep"):  # Skip sleep in tests
                run_phase1_generate_rank_bindings(
                    mgd_path,
                    None,
                    output_dir,
                    subprocess_run=subprocess.run,
                    sleep_secs=0,
                    mock_rank_to_desc=mock_rank_to_desc,
                )

        # Verify Phase 1 mock command structure
        # Mock clusters should use --oversubscribe, not --host
        assert "--oversubscribe" in captured_cmd
        # Should NOT have --host (MPI defaults to localhost)
        assert "--host" not in captured_cmd

        # Should have per-rank -np 1 segments (not single -np 3)
        np_indices = [i for i, arg in enumerate(captured_cmd) if arg == "-np"]
        assert len(np_indices) == 3  # One per rank
        assert all(captured_cmd[i + 1] == "1" for i in np_indices)

        # Should have : separators between ranks
        assert captured_cmd.count(":") == 2  # 3 ranks = 2 separators

        # Each rank should have its own TT_METAL_MOCK_CLUSTER_DESC_PATH
        env_vars = [captured_cmd[i + 1] for i, arg in enumerate(captured_cmd) if arg == "-x"]
        mock_env_vars = [e for e in env_vars if "TT_METAL_MOCK_CLUSTER_DESC_PATH" in e]
        assert len(mock_env_vars) == 3

    def test_phase2_uses_generated_rankfile_and_rank_bindings(self, runner, temp_dir):
        """Test Phase 2 command uses generated rankfile and rank bindings correctly."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        # Create generated rank_bindings.yaml (from Phase 1 output)
        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        mgd_path = temp_dir / "mesh_graph.yaml"

        # Create rank_bindings.yaml with 4 ranks (different from Phase 1 hosts)
        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "env_overrides": {"RANK0_VAR": "value0"}},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1, "env_overrides": {"RANK1_VAR": "value1"}},
                {"rank": 2, "mesh_id": 1, "mesh_host_rank": 0, "env_overrides": {"RANK2_VAR": "value2"}},
                {"rank": 3, "mesh_id": 1, "mesh_host_rank": 1, "env_overrides": {"RANK3_VAR": "value3"}},
            ],
            "global_env": {"GLOBAL_VAR": "global_value"},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()
        rankfile_path.touch()

        # Write a sample rankfile
        with open(rankfile_path, "w") as f:
            f.write("rank 0=nodeA slot=0\n")
            f.write("rank 1=nodeB slot=0\n")
            f.write("rank 2=nodeC slot=0\n")
            f.write("rank 3=nodeD slot=0\n")

        # Test Phase 2 command building directly (dry-run doesn't call subprocess.run)
        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]

        # Build MPI args with rankfile injection (as legacy_flow does)
        mpi_args = []
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, mpi_args, mpi_launcher)

        # Build Phase 2 MPI command
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Phase 2 should have rankfile injection (auto-detected syntax)
        # Check for rankfile-related args (could be --map-by rankfile:FILE=, --rankfile, or -mca rmaps_rankfile_path)
        cmd_str = " ".join(phase2_cmd)
        has_rankfile = "--rankfile" in phase2_cmd or "--map-by" in phase2_cmd or "rmaps_rankfile_path" in cmd_str
        assert has_rankfile, f"Phase 2 command should include rankfile args. Command: {phase2_cmd}"

        # Phase 2 should have per-rank -np 1 segments (one per rank in rank_bindings)
        np_indices = [i for i, arg in enumerate(phase2_cmd) if arg == "-np"]
        assert len(np_indices) == 4  # One per rank in rank_bindings

        # Phase 2 should have : separators between ranks
        assert phase2_cmd.count(":") == 3  # 4 ranks = 3 separators

        # Phase 2 should have per-rank env vars from rank_bindings
        env_vars = [phase2_cmd[i + 1] for i, arg in enumerate(phase2_cmd) if arg == "-x"]
        assert "RANK0_VAR=value0" in env_vars
        assert "RANK1_VAR=value1" in env_vars
        assert "RANK2_VAR=value2" in env_vars
        assert "RANK3_VAR=value3" in env_vars
        assert "GLOBAL_VAR=global_value" in env_vars

    def test_phase1_vs_phase2_different_commands(self, runner, temp_dir):
        """Test that Phase 1 and Phase 2 produce different MPI commands."""
        import subprocess
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )
        import os

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()

        # Use temp_dir for output (tests use temp to avoid writing to generated/ttrun)
        output_dir = temp_dir / "generated" / "ttrun"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create generated rank_bindings.yaml with 3 ranks (different from Phase 1)
        rank_bindings_path = output_dir / "rank_bindings.yaml"
        rankfile_path = output_dir / "rankfile"
        mesh_graph_file = temp_dir / "mesh_graph.yaml"

        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1},
                {"rank": 2, "mesh_id": 0, "mesh_host_rank": 2},  # 3 ranks, not 2
            ],
            "global_env": {},
            "mesh_graph_desc_path": str(mesh_graph_file),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mesh_graph_file.touch()
        rankfile_path.touch()

        # Capture Phase 1 command
        captured_phase1_cmd = []

        def mock_phase1_run(cmd, cwd=None, **kwargs):
            captured_phase1_cmd.extend(cmd)
            # Create output files in the correct location
            rank_bindings_path.touch()
            rankfile_path.touch()
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        # Run Phase 1
        call_count = 0

        def mock_run(cmd, cwd=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Phase 1
                return mock_phase1_run(cmd, cwd=cwd, **kwargs)
            else:
                # Phase 2 (should not be called in dry-run)
                mock_result = MagicMock()
                mock_result.returncode = 0
                return mock_result

        with patch.object(subprocess, "run", side_effect=mock_run):
            with patch("time.sleep"):  # Skip sleep
                result = runner.invoke(
                    main,
                    [
                        "--mesh-graph-descriptor",
                        str(mgd_path),
                        "--hosts",
                        "node1,node2",
                        "--dry-run",
                        "echo",
                        "test",
                    ],
                )
        # Verify CLI invocation succeeded
        assert result.exit_code == 0
        assert result.exception is None

        # Phase 1 should have been called
        assert call_count >= 1

        # Phase 1: should have -np 2 (len(hosts))
        phase1_np_idx = captured_phase1_cmd.index("-np")
        assert captured_phase1_cmd[phase1_np_idx + 1] == "2"

        # Phase 1: should have --host node1,node2
        phase1_host_idx = captured_phase1_cmd.index("--host")
        assert captured_phase1_cmd[phase1_host_idx + 1] == "node1,node2"

        # Now test Phase 2 command building directly (since dry-run doesn't execute it)
        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Phase 2: should have per-rank -np 1 segments (3 ranks from rank_bindings)
        phase2_np_indices = [i for i, arg in enumerate(phase2_cmd) if arg == "-np"]
        assert len(phase2_np_indices) == 3  # One per rank in rank_bindings

        # Phase 2: should use rankfile (not --host with host list)
        phase2_cmd_str = " ".join(phase2_cmd)
        has_rankfile = "--rankfile" in phase2_cmd or "--map-by" in phase2_cmd or "rmaps_rankfile_path" in phase2_cmd_str
        assert has_rankfile, "Phase 2 should use rankfile, not --host with host list"

        # Phase 2 should NOT have --host node1,node2 (uses rankfile instead)
        if "--host" in phase2_cmd:
            phase2_host_idx = phase2_cmd.index("--host")
            # If --host exists, it should not be the Phase 1 host list
            phase2_host_value = phase2_cmd[phase2_host_idx + 1]
            assert phase2_host_value != "node1,node2", "Phase 2 should not use Phase 1 host list"

    def test_phase2_asymmetrical_hosts_different_slots(self, runner, temp_dir):
        """Test Phase 2 correctly handles asymmetrical hosts with different slot counts."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        # Create rank_bindings.yaml with asymmetrical distribution:
        # - node1: ranks 0, 1, 2 (3 slots)
        # - node2: ranks 3, 4 (2 slots)
        # Total: 5 ranks
        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        mgd_path = temp_dir / "mesh_graph.yaml"

        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "env_overrides": {"RANK": "0"}},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1, "env_overrides": {"RANK": "1"}},
                {"rank": 2, "mesh_id": 0, "mesh_host_rank": 2, "env_overrides": {"RANK": "2"}},
                {"rank": 3, "mesh_id": 1, "mesh_host_rank": 0, "env_overrides": {"RANK": "3"}},
                {"rank": 4, "mesh_id": 1, "mesh_host_rank": 1, "env_overrides": {"RANK": "4"}},
            ],
            "global_env": {"GLOBAL_VAR": "global_value"},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()

        # Create asymmetrical rankfile:
        # node1 has 3 slots (ranks 0, 1, 2)
        # node2 has 2 slots (ranks 3, 4)
        with open(rankfile_path, "w") as f:
            f.write("rank 0=node1 slot=0\n")
            f.write("rank 1=node1 slot=1\n")
            f.write("rank 2=node1 slot=2\n")
            f.write("rank 3=node2 slot=0\n")
            f.write("rank 4=node2 slot=1\n")

        # Test Phase 2 command building
        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Phase 2 should have rankfile injection
        cmd_str = " ".join(phase2_cmd)
        has_rankfile = "--rankfile" in phase2_cmd or "--map-by" in phase2_cmd or "rmaps_rankfile_path" in cmd_str
        assert has_rankfile, f"Phase 2 command should include rankfile args. Command: {phase2_cmd}"

        # Phase 2 should have 5 per-rank -np 1 segments (one per rank in rank_bindings)
        np_indices = [i for i, arg in enumerate(phase2_cmd) if arg == "-np"]
        assert len(np_indices) == 5, f"Expected 5 -np segments for 5 ranks, got {len(np_indices)}"

        # Phase 2 should have 4 : separators (5 ranks = 4 separators)
        assert phase2_cmd.count(":") == 4, f"Expected 4 separators for 5 ranks, got {phase2_cmd.count(':')}"

        # Phase 2 should have per-rank env vars from rank_bindings
        env_vars = [phase2_cmd[i + 1] for i, arg in enumerate(phase2_cmd) if arg == "-x"]
        assert "RANK=0" in env_vars
        assert "RANK=1" in env_vars
        assert "RANK=2" in env_vars
        assert "RANK=3" in env_vars
        assert "RANK=4" in env_vars
        assert "GLOBAL_VAR=global_value" in env_vars

        # Verify rankfile path is correct in the command
        rankfile_path_str = str(rankfile_path.resolve())
        assert rankfile_path_str in cmd_str, f"Rankfile path should be in command: {rankfile_path_str}"

    def test_phase1_vs_phase2_asymmetrical_different_counts(self, runner, temp_dir):
        """Test Phase 1 and Phase 2 with asymmetrical slot distribution."""
        import subprocess
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        mgd_path = temp_dir / "mesh.textproto"
        mgd_path.touch()

        # Use temp_dir for output (tests use temp to avoid writing to generated/ttrun)
        output_dir = temp_dir / "generated" / "ttrun"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create generated rank_bindings.yaml with asymmetrical distribution:
        # node1: 3 ranks (slots 0, 1, 2)
        # node2: 2 ranks (slots 0, 1)
        # Total: 5 ranks (different from Phase 1's 2 hosts)
        rank_bindings_path = output_dir / "rank_bindings.yaml"
        rankfile_path = output_dir / "rankfile"
        mesh_graph_file = temp_dir / "mesh_graph.yaml"

        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1},
                {"rank": 2, "mesh_id": 0, "mesh_host_rank": 2},  # node1, 3 slots
                {"rank": 3, "mesh_id": 1, "mesh_host_rank": 0},  # node2, slot 0
                {"rank": 4, "mesh_id": 1, "mesh_host_rank": 1},  # node2, slot 1
            ],
            "global_env": {},
            "mesh_graph_desc_path": str(mesh_graph_file),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mesh_graph_file.touch()

        # Create asymmetrical rankfile
        with open(rankfile_path, "w") as f:
            f.write("rank 0=node1 slot=0\n")
            f.write("rank 1=node1 slot=1\n")
            f.write("rank 2=node1 slot=2\n")
            f.write("rank 3=node2 slot=0\n")
            f.write("rank 4=node2 slot=1\n")

        # Capture Phase 1 command
        captured_phase1_cmd = []

        def mock_phase1_run(cmd, cwd=None, **kwargs):
            captured_phase1_cmd.extend(cmd)
            # Create output files
            rank_bindings_path.touch()
            rankfile_path.touch()
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        call_count = 0

        def mock_run(cmd, cwd=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_phase1_run(cmd, cwd=cwd, **kwargs)
            else:
                mock_result = MagicMock()
                mock_result.returncode = 0
                return mock_result

        with patch.object(subprocess, "run", side_effect=mock_run):
            with patch("time.sleep"):
                runner.invoke(
                    main,
                    [
                        "--mesh-graph-descriptor",
                        str(mgd_path),
                        "--hosts",
                        "node1,node2",
                        "--dry-run",
                        "echo",
                        "test",
                    ],
                )

        # Phase 1: should have -np 2 (len(hosts))
        phase1_np_idx = captured_phase1_cmd.index("-np")
        assert captured_phase1_cmd[phase1_np_idx + 1] == "2"

        # Phase 1: should have --host node1,node2
        phase1_host_idx = captured_phase1_cmd.index("--host")
        assert captured_phase1_cmd[phase1_host_idx + 1] == "node1,node2"

        # Phase 2: Build command directly (dry-run doesn't execute)
        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Phase 2: should have 5 per-rank -np 1 segments (5 ranks from rank_bindings, not 2 from Phase 1)
        phase2_np_indices = [i for i, arg in enumerate(phase2_cmd) if arg == "-np"]
        assert len(phase2_np_indices) == 5, f"Expected 5 ranks in Phase 2, got {len(phase2_np_indices)}"

        # Phase 2: should use rankfile (not --host with host list)
        phase2_cmd_str = " ".join(phase2_cmd)
        has_rankfile = "--rankfile" in phase2_cmd or "--map-by" in phase2_cmd or "rmaps_rankfile_path" in phase2_cmd_str
        assert has_rankfile, "Phase 2 should use rankfile for asymmetrical distribution"

        # Verify rankfile contains the asymmetrical slot assignments
        rankfile_content = rankfile_path.read_text()
        assert "rank 0=node1 slot=0" in rankfile_content
        assert "rank 1=node1 slot=1" in rankfile_content
        assert "rank 2=node1 slot=2" in rankfile_content
        assert "rank 3=node2 slot=0" in rankfile_content
        assert "rank 4=node2 slot=1" in rankfile_content

    def test_phase2_env_vars_mesh_id_and_host_rank(self, runner, temp_dir):
        """Test Phase 2 correctly sets TT_MESH_ID and TT_MESH_HOST_RANK from rank_bindings."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        mgd_path = temp_dir / "mesh_graph.yaml"

        # Create rank_bindings with different mesh_id and mesh_host_rank per rank
        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1},
                {"rank": 2, "mesh_id": 1, "mesh_host_rank": 0},
                {"rank": 3, "mesh_id": 1, "mesh_host_rank": 1},
            ],
            "global_env": {},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()
        rankfile_path.touch()

        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Split command into rank segments (separated by :)
        rank_segments = []
        current_segment = []
        for arg in phase2_cmd:
            if arg == ":":
                if current_segment:
                    rank_segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(arg)
        if current_segment:
            rank_segments.append(current_segment)

        # Extract env vars for each rank segment
        rank_envs = []
        for segment in rank_segments:
            env_vars = {}
            for i, arg in enumerate(segment):
                if arg == "-x" and i + 1 < len(segment):
                    env_pair = segment[i + 1]
                    if "=" in env_pair:
                        key, value = env_pair.split("=", 1)
                        env_vars[key] = value
            rank_envs.append(env_vars)

        # Verify TT_MESH_ID for each rank (segments are ordered by rank)
        assert rank_envs[0]["TT_MESH_ID"] == "0"  # Rank 0
        assert rank_envs[1]["TT_MESH_ID"] == "0"  # Rank 1
        assert rank_envs[2]["TT_MESH_ID"] == "1"  # Rank 2
        assert rank_envs[3]["TT_MESH_ID"] == "1"  # Rank 3

        # Verify TT_MESH_HOST_RANK for each rank
        assert rank_envs[0]["TT_MESH_HOST_RANK"] == "0"
        assert rank_envs[1]["TT_MESH_HOST_RANK"] == "1"
        assert rank_envs[2]["TT_MESH_HOST_RANK"] == "0"
        assert rank_envs[3]["TT_MESH_HOST_RANK"] == "1"

        # Verify TT_MESH_GRAPH_DESC_PATH is set (same for all ranks)
        mgd_path_str = str(mgd_path.resolve())
        assert rank_envs[0]["TT_MESH_GRAPH_DESC_PATH"] == mgd_path_str
        assert rank_envs[1]["TT_MESH_GRAPH_DESC_PATH"] == mgd_path_str
        assert rank_envs[2]["TT_MESH_GRAPH_DESC_PATH"] == mgd_path_str
        assert rank_envs[3]["TT_MESH_GRAPH_DESC_PATH"] == mgd_path_str

    def test_phase2_env_vars_passthrough_and_blocked(self, runner, temp_dir, monkeypatch):
        """Test Phase 2 correctly passes through env vars and blocks managed ones."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        # Set up pass-through env vars
        monkeypatch.setenv("TT_CUSTOM_VAR", "passthrough_value")
        monkeypatch.setenv("ARCH_NAME", "grayskull")
        monkeypatch.setenv("WH_ARCH_YAML", "/path/to/wh.yaml")
        monkeypatch.setenv("TTNN_CONFIG_OVERRIDES", "some_config")

        # Set up blocked vars (should NOT pass through, but set from rank_bindings)
        monkeypatch.setenv("TT_MESH_ID", "blocked_value")  # Should be blocked and overridden
        monkeypatch.setenv("TT_MESH_HOST_RANK", "blocked_value")  # Should be blocked and overridden
        monkeypatch.setenv("TT_MESH_GRAPH_DESC_PATH", "blocked_value")  # Should be blocked and overridden
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "blocked_value")  # Should be blocked

        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        mgd_path = temp_dir / "mesh_graph.yaml"

        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "env_overrides": {"TT_VISIBLE_DEVICES": "0,1"}},
            ],
            "global_env": {"GLOBAL_VAR": "global_value"},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()
        rankfile_path.touch()

        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Extract all env vars
        env_vars = [phase2_cmd[i + 1] for i, arg in enumerate(phase2_cmd) if arg == "-x"]
        env_dict = {}
        for env_pair in env_vars:
            if "=" in env_pair:
                key, value = env_pair.split("=", 1)
                env_dict[key] = value

        # Verify pass-through vars are present
        assert "TT_CUSTOM_VAR=passthrough_value" in env_vars
        assert "ARCH_NAME=grayskull" in env_vars
        assert "WH_ARCH_YAML=/path/to/wh.yaml" in env_vars
        assert "TTNN_CONFIG_OVERRIDES=some_config" in env_vars

        # Verify blocked vars are NOT from parent env (they're set from rank_bindings)
        assert env_dict["TT_MESH_ID"] == "0"  # From rank_bindings, not "blocked_value"
        assert env_dict["TT_MESH_HOST_RANK"] == "0"  # From rank_bindings, not "blocked_value"
        mgd_path_str = str(mgd_path.resolve())
        assert env_dict["TT_MESH_GRAPH_DESC_PATH"] == mgd_path_str  # From config, not "blocked_value"

        # Verify TT_VISIBLE_DEVICES comes from env_overrides, not parent env
        assert env_dict["TT_VISIBLE_DEVICES"] == "0,1"  # From env_overrides, not "blocked_value"

        # Verify global env var
        assert "GLOBAL_VAR=global_value" in env_vars

    def test_phase2_env_vars_per_rank_overrides(self, runner, temp_dir):
        """Test Phase 2 correctly applies per-rank env_overrides."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
        )

        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        rankfile_path = temp_dir / "rankfile"
        mgd_path = temp_dir / "mesh_graph.yaml"

        rank_bindings_content = {
            "rank_bindings": [
                {
                    "rank": 0,
                    "mesh_id": 0,
                    "mesh_host_rank": 0,
                    "env_overrides": {"RANK0_VAR": "value0", "TT_VISIBLE_DEVICES": "0"},
                },
                {
                    "rank": 1,
                    "mesh_id": 0,
                    "mesh_host_rank": 1,
                    "env_overrides": {"RANK1_VAR": "value1", "TT_VISIBLE_DEVICES": "1"},
                },
            ],
            "global_env": {"GLOBAL_VAR": "global_value"},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()
        rankfile_path.touch()

        config = parse_binding_config(rank_bindings_path)
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Find rank segments (separated by :)
        rank_segments = []
        current_segment = []
        for arg in phase2_cmd:
            if arg == ":":
                if current_segment:
                    rank_segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(arg)
        if current_segment:
            rank_segments.append(current_segment)

        # Extract env vars for each rank segment
        rank_envs = []
        for segment in rank_segments:
            env_vars = {}
            for i, arg in enumerate(segment):
                if arg == "-x" and i + 1 < len(segment):
                    env_pair = segment[i + 1]
                    if "=" in env_pair:
                        key, value = env_pair.split("=", 1)
                        env_vars[key] = value
            rank_envs.append(env_vars)

        # Verify rank 0 has its specific env vars
        assert rank_envs[0]["RANK0_VAR"] == "value0"
        assert rank_envs[0]["TT_VISIBLE_DEVICES"] == "0"
        assert "RANK1_VAR" not in rank_envs[0]  # Should not be in rank 0

        # Verify rank 1 has its specific env vars
        assert rank_envs[1]["RANK1_VAR"] == "value1"
        assert rank_envs[1]["TT_VISIBLE_DEVICES"] == "1"
        assert "RANK0_VAR" not in rank_envs[1]  # Should not be in rank 1

        # Verify both ranks have global env var
        assert rank_envs[0]["GLOBAL_VAR"] == "global_value"
        assert rank_envs[1]["GLOBAL_VAR"] == "global_value"

        # Verify both ranks have TT_MESH_ID and TT_MESH_HOST_RANK
        assert rank_envs[0]["TT_MESH_ID"] == "0"
        assert rank_envs[0]["TT_MESH_HOST_RANK"] == "0"
        assert rank_envs[1]["TT_MESH_ID"] == "0"
        assert rank_envs[1]["TT_MESH_HOST_RANK"] == "1"

    def test_phase2_mock_cluster_desc_path_mapping(self, runner, temp_dir):
        """Test Phase 2 correctly sets TT_METAL_MOCK_CLUSTER_DESC_PATH based on Phase 1 mapping."""
        from ttnn.distributed.ttrun import (
            parse_binding_config,
            build_mpi_command,
            inject_rankfile_mpi_args,
            get_mpi_launcher,
            build_phase2_mock_mapping,
        )
        from pathlib import Path

        # Create Phase 1 mock cluster mapping: 2 hosts, 2 ranks
        phase1_hosts = ["node1", "node2"]
        phase1_mock_desc = {
            0: temp_dir / "mock_desc_node1.yaml",
            1: temp_dir / "mock_desc_node2.yaml",
        }
        for desc_path in phase1_mock_desc.values():
            desc_path.touch()

        # Create Phase 2 rankfile: 4 ranks, 2 on node1, 2 on node2
        rankfile_path = temp_dir / "rankfile"
        rankfile_content = """rank 0=node1 slot=0
rank 1=node1 slot=1
rank 2=node2 slot=0
rank 3=node2 slot=1
"""
        rankfile_path.write_text(rankfile_content)

        # Build Phase 2 mock mapping
        phase2_mock_mapping = build_phase2_mock_mapping(rankfile_path, phase1_hosts, phase1_mock_desc)
        assert phase2_mock_mapping is not None
        assert len(phase2_mock_mapping) == 4

        # Verify mapping: ranks on node1 get Phase 1 rank 0's descriptor, ranks on node2 get Phase 1 rank 1's
        assert phase2_mock_mapping[0] == phase1_mock_desc[0]  # Rank 0 on node1 -> Phase 1 rank 0
        assert phase2_mock_mapping[1] == phase1_mock_desc[0]  # Rank 1 on node1 -> Phase 1 rank 0
        assert phase2_mock_mapping[2] == phase1_mock_desc[1]  # Rank 2 on node2 -> Phase 1 rank 1
        assert phase2_mock_mapping[3] == phase1_mock_desc[1]  # Rank 3 on node2 -> Phase 1 rank 1

        # Create Phase 2 rank_bindings.yaml
        rank_bindings_path = temp_dir / "rank_bindings.yaml"
        mgd_path = temp_dir / "mesh_graph.yaml"
        rank_bindings_content = {
            "rank_bindings": [
                {"rank": 0, "mesh_id": 0, "mesh_host_rank": 0},
                {"rank": 1, "mesh_id": 0, "mesh_host_rank": 1},
                {"rank": 2, "mesh_id": 0, "mesh_host_rank": 2},
                {"rank": 3, "mesh_id": 0, "mesh_host_rank": 3},
            ],
            "global_env": {},
            "mesh_graph_desc_path": str(mgd_path),
        }
        with open(rank_bindings_path, "w") as f:
            yaml.dump(rank_bindings_content, f)
        mgd_path.touch()

        # Create Phase 2 mock binding file
        # Note: parse_binding_config converts string keys to int, so we can use either
        phase2_mock_binding_path = temp_dir / "phase2_mock_mapping.yaml"
        phase2_mock_data = {
            "rank_to_cluster_mock_cluster_desc": {
                rank: str(desc_path) for rank, desc_path in phase2_mock_mapping.items()
            }
        }
        with open(phase2_mock_binding_path, "w") as f:
            yaml.dump(phase2_mock_data, f)

        # Parse config with Phase 2 mock binding
        config = parse_binding_config(rank_bindings_path, phase2_mock_binding_path)
        assert config.mock_cluster_rank_binding is not None
        assert len(config.mock_cluster_rank_binding) == 4

        # Build Phase 2 MPI command
        program = ["echo", "test"]
        mpi_launcher = get_mpi_launcher()
        effective_mpi_args = inject_rankfile_mpi_args(rankfile_path, [], mpi_launcher)
        phase2_cmd = build_mpi_command(config, program, effective_mpi_args)

        # Split command into rank segments
        rank_segments = []
        current_segment = []
        for arg in phase2_cmd:
            if arg == ":":
                if current_segment:
                    rank_segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(arg)
        if current_segment:
            rank_segments.append(current_segment)

        # Extract env vars for each rank segment
        rank_envs = []
        for segment in rank_segments:
            env_vars = {}
            for i, arg in enumerate(segment):
                if arg == "-x" and i + 1 < len(segment):
                    env_pair = segment[i + 1]
                    if "=" in env_pair:
                        key, value = env_pair.split("=", 1)
                        env_vars[key] = value
            rank_envs.append(env_vars)

        # Verify TT_METAL_MOCK_CLUSTER_DESC_PATH is set correctly for each rank
        assert rank_envs[0]["TT_METAL_MOCK_CLUSTER_DESC_PATH"] == str(phase1_mock_desc[0].resolve())
        assert rank_envs[1]["TT_METAL_MOCK_CLUSTER_DESC_PATH"] == str(phase1_mock_desc[0].resolve())
        assert rank_envs[2]["TT_METAL_MOCK_CLUSTER_DESC_PATH"] == str(phase1_mock_desc[1].resolve())
        assert rank_envs[3]["TT_METAL_MOCK_CLUSTER_DESC_PATH"] == str(phase1_mock_desc[1].resolve())

    def test_phase2_mock_cluster_desc_path_no_hosts(self, runner, temp_dir):
        """Test Phase 2 mock mapping when Phase 1 used mock clusters (no hosts)."""
        from ttnn.distributed.ttrun import build_phase2_mock_mapping
        from pathlib import Path

        # Phase 1: Mock cluster (all localhost), 3 ranks
        phase1_hosts = None
        phase1_mock_desc = {
            0: temp_dir / "mock_desc_0.yaml",
            1: temp_dir / "mock_desc_1.yaml",
            2: temp_dir / "mock_desc_2.yaml",
        }
        for desc_path in phase1_mock_desc.values():
            desc_path.touch()

        # Phase 2: 4 ranks, all on localhost
        rankfile_path = temp_dir / "rankfile"
        rankfile_content = """rank 0=localhost slot=0
rank 1=localhost slot=1
rank 2=localhost slot=2
rank 3=localhost slot=3
"""
        rankfile_path.write_text(rankfile_content)

        # Build Phase 2 mock mapping
        phase2_mock_mapping = build_phase2_mock_mapping(rankfile_path, phase1_hosts, phase1_mock_desc)
        assert phase2_mock_mapping is not None
        assert len(phase2_mock_mapping) == 4

        # Verify mapping: Phase 2 ranks map directly to Phase 1 ranks (modulo)
        assert phase2_mock_mapping[0] == phase1_mock_desc[0]  # Rank 0 -> Phase 1 rank 0
        assert phase2_mock_mapping[1] == phase1_mock_desc[1]  # Rank 1 -> Phase 1 rank 1
        assert phase2_mock_mapping[2] == phase1_mock_desc[2]  # Rank 2 -> Phase 1 rank 2
        assert phase2_mock_mapping[3] == phase1_mock_desc[0]  # Rank 3 -> Phase 1 rank 0 (modulo)
