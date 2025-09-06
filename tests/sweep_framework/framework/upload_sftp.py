# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import tempfile
import shutil
from typing import Optional
from framework.sweeps_logger import sweeps_logger as logger


def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    return value


def upload_run_sftp(local_path: pathlib.Path) -> bool:
    """
    Upload a single oprun_*.json to an SFTP server.

    Configuration is read from environment variables:
      - SFTP_USERNAME: username on the server (required)
      - SFTP_HOSTNAME: hostname (required)
      - SFTP_PORT: port (optional; defaults to 22)
      - SFTP_PRIVATE_KEY: private key contents (PEM format) for key-based auth (required)
      - SFTP_REMOTE_DIR: remote directory to upload into (optional; defaults to user's home)

    Returns True on success, False if configuration is missing or upload fails.
    """
    try:
        local_path = pathlib.Path(local_path)
        if not local_path.exists():
            logger.warning(f"SFTP upload skipped: file does not exist: {local_path}")
            return False

        # Check CLI availability first
        if shutil.which("sftp") is None:
            logger.warning("SFTP upload skipped: 'sftp' command not found in PATH")
            return False

        username = _read_env("SFTP_USERNAME")
        hostname = _read_env("SFTP_HOSTNAME")
        private_key = _read_env("SFTP_PRIVATE_KEY")
        port = _read_env("SFTP_PORT") or "22"
        remote_dir = _read_env("SFTP_REMOTE_DIR")  # May be None

        if not username or not hostname or not private_key:
            # Not configured in this environment; do not treat as error
            logger.info("SFTP credentials not provided; skipping upload.")
            return False

        # Write the private key to a temp file with restricted permissions
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = pathlib.Path(tmpdir) / "id_key"
            key_path.write_text(private_key)
            os.chmod(key_path, 0o600)

            # Create a batchfile to drive sftp non-interactively
            batch_path = pathlib.Path(tmpdir) / "batch.txt"
            lines = []
            if remote_dir:
                # Try to create and then cd into the directory; ignore mkdir errors
                lines.append(f"mkdir {remote_dir}")
                lines.append(f"cd {remote_dir}")
            # Put the file
            # Quote the local path by escaping spaces; sftp batch does not support shell quoting, so use absolute path without quotes if possible
            lines.append(f"put {str(local_path)}")
            # List to verify
            lines.append("ls -hal")
            batch_path.write_text("\n".join(lines) + "\n")

            # Execute the sftp command
            cmd = [
                "sftp",
                "-oStrictHostKeyChecking=no",
                "-P",
                str(port),
                "-i",
                str(key_path),
                "-b",
                str(batch_path),
                f"{username}@{hostname}",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except Exception as e:
                logger.error(f"SFTP upload failed to execute: {e}")
                return False

            if result.returncode != 0:
                logger.error(
                    "SFTP upload failed (non-zero exit). Stdout: %s Stderr: %s",
                    result.stdout,
                    result.stderr,
                )
                return False

            logger.info(f"SFTP uploaded file '{local_path.name}' to {hostname}:{remote_dir or '~'}")
            return True
    except Exception as e:
        logger.error(f"Unexpected error during SFTP upload: {e}")
        return False
