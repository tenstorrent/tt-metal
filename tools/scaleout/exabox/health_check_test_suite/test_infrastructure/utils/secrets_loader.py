# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load JIRA/SFTP credentials from KEY=VALUE files in the log directory."""

from __future__ import annotations

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


def load_jira_secrets(log_dir: str) -> str:
    """Load JIRA bearer token from credentials file."""
    creds_file = Path(log_dir) / ".jira-creds"
    if not creds_file.is_file():
        return ""
    env = _source_env_file(creds_file)
    return env.get("JIRA_BEARER_TOKEN", "")


def load_sftp_secrets(log_dir: str) -> tuple[str, str]:
    """Load SFTP user and host from credentials file."""
    creds_file = Path(log_dir) / ".sftp-creds"
    if not creds_file.is_file():
        return "", ""
    env = _source_env_file(creds_file)
    return env.get("SFTP_USER", ""), env.get("SFTP_HOST", "")
