#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Launch script for multi-host Galaxy jobs via tt-run.

This module can be used programmatically or as a CLI tool.
"""
import argparse
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a specific Galaxy mode (2x or 4x)."""

    rank_binding_yaml: str
    hosts: tuple[str, ...]
    rankfile: str

    @property
    def hosts_str(self) -> str:
        """Return hosts as a comma-separated string."""
        return ",".join(self.hosts)


@dataclass(frozen=True)
class HostConfig:
    """Configuration for a specific host, containing 2x and 4x modes."""

    dual: ModeConfig  # 2x config
    quad: ModeConfig  # 4x config
    env: tuple[tuple[str, str], ...] = ()  # Environment variables as (key, value) pairs

    def get_mode(self, config_type: str) -> ModeConfig:
        """Get the ModeConfig for the given config type ('2x' or '4x')."""
        if config_type == "2x":
            return self.dual
        elif config_type == "4x":
            return self.quad
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def env_exports(self) -> str:
        """Return env vars as shell export statements."""
        if not self.env:
            return ""
        return " && ".join(f"export {k}={shlex.quote(v)}" for k, v in self.env)


# Mapping from config type to MESH_DEVICE value
MESH_DEVICE_MAP = {"2x": "DUAL", "4x": "QUAD"}

# Special alias commands that bypass normal job setup
COMMAND_ALIASES: dict[str, str] = {"reset": "tt-smi -glx_reset --snapshot_no_tty; sudo rm -rf /dev/shm/*"}

# Common environment variables for all hosts
COMMON_ENV: tuple[tuple[str, str], ...] = (
    ("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"),
    ("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev"),
)

CONFIGS: dict[str, HostConfig] = {
    "g05glx01": HostConfig(
        dual=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02"),
            rankfile="/etc/mpirun/rankfile_g05glx01_g05glx02",
        ),
        quad=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02", "g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile",
        ),
        env=COMMON_ENV,
    ),
    "g05glx02": HostConfig(
        dual=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02"),
            rankfile="/etc/mpirun/rankfile_g05glx01_g05glx02",
        ),
        quad=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02", "g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile",
        ),
        env=COMMON_ENV,
    ),
    "g05glx03": HostConfig(
        dual=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
            hosts=("g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile_g05glx03_g05glx04",
        ),
        quad=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02", "g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile",
        ),
        env=COMMON_ENV,
    ),
    "g05glx04": HostConfig(
        dual=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
            hosts=("g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile_g05glx03_g05glx04",
        ),
        quad=ModeConfig(
            rank_binding_yaml="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml",
            hosts=("g05glx01", "g05glx02", "g05glx03", "g05glx04"),
            rankfile="/etc/mpirun/rankfile",
        ),
        env=COMMON_ENV,
    ),
}


def get_galaxy_config(config_type: str, hostname: str | None = None) -> tuple[ModeConfig, HostConfig, str]:
    """
    Get Galaxy configuration based on hostname and config type.

    Args:
        config_type: "2x" or "4x"
        hostname: Override hostname (defaults to socket.gethostname())

    Returns:
        Tuple of (ModeConfig, HostConfig, hostname)

    Raises:
        ValueError: If hostname is not configured or config_type is invalid
    """
    if hostname is None:
        hostname = socket.gethostname()

    if hostname not in CONFIGS:
        available = ", ".join(sorted(CONFIGS.keys())) or "(none)"
        raise ValueError(f"No configuration found for hostname '{hostname}'. " f"Available hosts: {available}")

    host_config = CONFIGS[hostname]
    return host_config.get_mode(config_type), host_config, hostname


def build_mpi_args(cfg: ModeConfig) -> str:
    """Build the MPI arguments string."""
    return (
        f"--host {cfg.hosts_str} "
        f"--map-by rankfile:file={cfg.rankfile} "
        "--bind-to none "
        "--output-filename logs/mpi_job "
    )


def tcp_interface() -> str:
    return "cnx1"


def build_inner_command(
    command: list[str],
    config_type: str,
    host_cfg: HostConfig,
    tt_metal_home: str,
) -> str:
    """
    Build the inner bash command that will be executed via tt-run.

    Args:
        command: List of command arguments
        config_type: "2x" or "4x"
        host_cfg: Host configuration
        tt_metal_home: Path to TT_METAL_HOME

    Returns:
        The inner bash command string
    """
    venv_path = os.path.join(tt_metal_home, "python_env", "bin", "activate")
    cmd_str = shlex.join(command)
    mesh_device = MESH_DEVICE_MAP[config_type]

    parts = [f"source {shlex.quote(venv_path)}"]
    parts.append(f"export MESH_DEVICE={mesh_device}")
    if host_cfg.env:
        parts.append(host_cfg.env_exports())
    parts.append(cmd_str)

    return " && ".join(parts)


def build_tt_run_command(
    command: list[str],
    config_type: str,
    hostname: str | None = None,
    tt_metal_home: str | None = None,
) -> tuple[list[str], ModeConfig, HostConfig, str]:
    """
    Build the full tt-run command.

    Args:
        command: List of command arguments (single-word commands in COMMAND_ALIASES
                 are expanded to their shell equivalents)
        config_type: "2x" or "4x"
        hostname: Override hostname
        tt_metal_home: Override TT_METAL_HOME

    Returns:
        Tuple of (tt_run_command_list, ModeConfig, HostConfig, hostname)

    Raises:
        ValueError: If configuration is invalid
    """
    if tt_metal_home is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME", os.getcwd())

    assert os.path.exists(tt_metal_home), f"TT_METAL_HOME not found: {tt_metal_home}"

    cfg, host_cfg, hostname = get_galaxy_config(config_type, hostname)

    # Resolve rank binding path
    if not os.path.isabs(cfg.rank_binding_yaml):
        rank_binding_path = os.path.join(tt_metal_home, cfg.rank_binding_yaml)
    else:
        rank_binding_path = cfg.rank_binding_yaml

    assert os.path.exists(rank_binding_path), f"Rank binding file not found: {rank_binding_path}"

    mpi_args = build_mpi_args(cfg)

    # Check for alias commands (e.g., "reset", "nuke")
    if len(command) == 1 and command[0] in COMMAND_ALIASES:
        venv_path = os.path.join(tt_metal_home, "python_env", "bin", "activate")
        inner_bash_cmd = f"source {shlex.quote(venv_path)} && {COMMAND_ALIASES[command[0]]}"
    else:
        inner_bash_cmd = build_inner_command(command, config_type, host_cfg, tt_metal_home)

    tt_run_cmd = [
        "tt-run",
        "--tcp-interface",
        tcp_interface(),
        "--rank-binding",
        rank_binding_path,
        "--mpi-args",
        mpi_args,
        "bash",
        "-c",
        inner_bash_cmd,
    ]

    return tt_run_cmd, cfg, host_cfg, hostname


def format_config_summary(
    cfg: ModeConfig,
    host_cfg: HostConfig,
    hostname: str,
    config_type: str,
    dryrun: bool = False,
) -> str:
    """Format the configuration summary for display."""
    prefix = "[DRY RUN] " if dryrun else ""
    title = f"{prefix}{config_type} Galaxy Configuration"
    mesh_device = MESH_DEVICE_MAP[config_type]

    lines = [
        "",
        "=" * 60,
        f"  {title}",
        "=" * 60,
        f"  {'Hostname:':<18} {hostname}",
        f"  {'Mode:':<18} {mesh_device}",
        f"  {'Hosts:':<18} {cfg.hosts_str}",
        f"  {'Rankfile:':<18} {cfg.rankfile}",
        f"  {'Rank binding:':<18} {cfg.rank_binding_yaml}",
    ]

    if host_cfg.env:
        lines.append(f"  {'-' * 56}")
        lines.append(f"  {'Environment:':<18}")
        for key, value in host_cfg.env:
            display_value = value if len(value) <= 75 else value[:72] + "..."
            lines.append(f"    {key}={display_value}")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def validate_files(rank_binding_path: str, rankfile: str) -> list[str]:
    """
    Validate that required files exist.

    Returns:
        List of missing file descriptions (empty if all exist)
    """
    missing = []
    if not os.path.exists(rank_binding_path):
        missing.append(f"Rank binding YAML: {rank_binding_path}")
    if not os.path.exists(rankfile):
        missing.append(f"Rankfile: {rankfile}")
    return missing


def run_galaxy_command(
    command: list[str],
    config_type: str,
    dryrun: bool = False,
    hostname: str | None = None,
    tt_metal_home: str | None = None,
) -> int:
    """
    Run a command on the Galaxy cluster.

    Args:
        command: List of command arguments (single-word commands in COMMAND_ALIASES
                 are expanded to their shell equivalents)
        config_type: "2x" or "4x"
        dryrun: If True, only print the command without executing
        hostname: Override hostname
        tt_metal_home: Override TT_METAL_HOME

    Returns:
        Exit code (0 for success)
    """
    if tt_metal_home is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME", os.getcwd())

    tt_run_cmd, cfg, host_cfg, hostname = build_tt_run_command(command, config_type, hostname, tt_metal_home)

    # Print configuration summary
    print(format_config_summary(cfg, host_cfg, hostname, config_type, dryrun))

    # Resolve rank binding path for validation
    if not os.path.isabs(cfg.rank_binding_yaml):
        rank_binding_path = os.path.join(tt_metal_home, cfg.rank_binding_yaml)
    else:
        rank_binding_path = cfg.rank_binding_yaml

    # Validate files
    missing_files = validate_files(rank_binding_path, cfg.rankfile)
    if missing_files:
        print("Error: Required files not found:")
        for f in missing_files:
            print(f"  - {f}")
        return 1

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Print command
    printable_cmd = shlex.join(tt_run_cmd)
    print(f"Command:\n{printable_cmd}\n")

    if dryrun:
        return 0

    try:
        result = subprocess.run(tt_run_cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except FileNotFoundError:
        print("Error: 'tt-run' not found. Make sure it is in your PATH (e.g. activate python_env).")
        return 127


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch a job on multi-host Galaxy via tt-run.",
        usage="%(prog)s [-d] {2x,4x} -- command [args...]",
        epilog="Special commands: 'reset' (reset devices and clear shared memory)",
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        action="store_true",
        help="Print the command that would be run, without executing it",
    )
    parser.add_argument(
        "config",
        choices=["2x", "4x"],
        help="Galaxy configuration to use (2x=Dual, 4x=Quad)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (after --)",
    )

    parsed = parser.parse_args(args)

    # Remove leading '--' if present
    remainder = parsed.command
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]

    if not remainder:
        parser.error("Missing command to run. Provide it after '--'.")

    try:
        return run_galaxy_command(
            command=remainder,
            config_type=parsed.config,
            dryrun=parsed.dryrun,
        )
    except ValueError as e:
        parser.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
