# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load JIRA/SFTP credentials.

Credential loading is one of the two spots that diverge between deployments, so
it dispatches on ``launch_mode``:

    orchestration (k8s)  environment variables only (injected from K8s Secrets)
    slurm (default)      environment variables, falling back to KEY=VALUE files
                         in the log directory (.jira-creds / .sftp-creds)

Environment variables read in both modes:

    JIRA_BEARER_TOKEN      (slurm fallback: .jira-creds)
    SFTP_USER / SFTP_HOST  (slurm fallback: .sftp-creds)
"""

from __future__ import annotations

import os
from pathlib import Path


def _source_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE shell credentials file."""
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.replace("export ", "").strip()
            env[key] = value.strip().strip("'\"")
    return env


# --- JIRA -----------------------------------------------------------------


def _load_jira_orchestration() -> str:
    return os.environ.get("JIRA_BEARER_TOKEN", "")


def _load_jira_slurm(log_dir: str) -> str:
    token = os.environ.get("JIRA_BEARER_TOKEN")
    if token:
        return token
    creds_file = Path(log_dir) / ".jira-creds"
    if not creds_file.is_file():
        return ""
    return _source_env_file(creds_file).get("JIRA_BEARER_TOKEN", "")


def load_jira_secrets(log_dir: str, launch_mode: str = "slurm") -> str:
    """Load the JIRA bearer token for the given launch mode."""
    if launch_mode == "orchestration":
        return _load_jira_orchestration()
    return _load_jira_slurm(log_dir)


# --- SFTP user/host -------------------------------------------------------


def _load_sftp_orchestration() -> tuple[str, str]:
    return os.environ.get("SFTP_USER", ""), os.environ.get("SFTP_HOST", "")


def _load_sftp_slurm(log_dir: str) -> tuple[str, str]:
    user = os.environ.get("SFTP_USER")
    host = os.environ.get("SFTP_HOST")
    if user and host:
        return user, host
    creds_file = Path(log_dir) / ".sftp-creds"
    if creds_file.is_file():
        env = _source_env_file(creds_file)
        user = user or env.get("SFTP_USER", "")
        host = host or env.get("SFTP_HOST", "")
    return user or "", host or ""


def load_sftp_secrets(log_dir: str, launch_mode: str = "slurm") -> tuple[str, str]:
    """Load the SFTP user/host for the given launch mode."""
    if launch_mode == "orchestration":
        return _load_sftp_orchestration()
    return _load_sftp_slurm(log_dir)
