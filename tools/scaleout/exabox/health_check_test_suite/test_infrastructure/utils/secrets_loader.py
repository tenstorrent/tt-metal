# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load JIRA/SFTP credentials.

Environment variables take precedence (the containerized flow injects them via
``docker --env-file``), falling back to KEY=VALUE files in the log directory for
bare-metal / local invocations:

    JIRA_BEARER_TOKEN      <- .jira-creds
    SFTP_USER / SFTP_HOST  <- .sftp-creds
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


def load_jira_secrets(log_dir: str) -> str:
    """Load JIRA bearer token: env JIRA_BEARER_TOKEN, else .jira-creds file."""
    token = os.environ.get("JIRA_BEARER_TOKEN")
    if token:
        return token
    creds_file = Path(log_dir) / ".jira-creds"
    if not creds_file.is_file():
        return ""
    return _source_env_file(creds_file).get("JIRA_BEARER_TOKEN", "")


def load_sftp_secrets(log_dir: str) -> tuple[str, str]:
    """Load SFTP user/host: env SFTP_USER/SFTP_HOST, else .sftp-creds file."""
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
