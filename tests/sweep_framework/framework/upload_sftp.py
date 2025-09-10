# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import tempfile
import shutil
from typing import Optional

try:
    from framework.sweeps_logger import sweeps_logger as logger
except ModuleNotFoundError:
    # Allow direct execution of this file from root: python tests/sweep_framework/framework/upload_sftp.py
    import sys as _sys

    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from sweeps_logger import sweeps_logger as logger  # type: ignore


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
      - SFTP_PRIVATE_KEY: private key contents (PEM format) for key-based auth (optional)
      - SFTP_PRIVATE_KEY_PATH: path to a private key file inside the environment (optional)
      - SFTP_REMOTE_DIR: remote directory to upload into (optional; defaults to user's home)

    Authentication order of preference:
      1) ssh-agent forwarding (if SSH_AUTH_SOCK is present)
      2) SFTP_PRIVATE_KEY (inline key contents)
      3) SFTP_PRIVATE_KEY_PATH (path to key file)

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
        private_key_path_env = _read_env("SFTP_PRIVATE_KEY_PATH")
        port = _read_env("SFTP_PORT") or "22"
        remote_dir = _read_env("SFTP_REMOTE_DIR")

        if not username or not hostname:
            # Not configured in this environment; do not treat as error
            logger.info("SFTP username/hostname not provided; skipping upload.")
            return False

        # Determine authentication method
        auth_mode = None
        if os.getenv("SSH_AUTH_SOCK"):
            auth_mode = "ssh_agent"
        elif private_key:
            auth_mode = "inline_key"
        elif private_key_path_env:
            auth_mode = "key_path"
        else:
            logger.info(
                "SFTP auth not provided; set SFTP_PRIVATE_KEY, SFTP_PRIVATE_KEY_PATH, or use ssh-agent (SSH_AUTH_SOCK). Skipping upload."
            )
            return False

        # Write the private key to a temp file with restricted permissions
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path: Optional[pathlib.Path] = None
            if auth_mode == "inline_key":
                key_path = pathlib.Path(tmpdir) / "id_key"
                key_path.write_text(private_key)  # type: ignore[arg-type]
                os.chmod(key_path, 0o600)
                logger.info("SFTP will authenticate using an inline private key (temp file).")
            elif auth_mode == "key_path":
                candidate = pathlib.Path(private_key_path_env)  # type: ignore[arg-type]
                if not candidate.exists():
                    logger.error(f"SFTP key path does not exist: {candidate}")
                    return False
                if not candidate.is_file():
                    logger.error(f"SFTP key path is not a file: {candidate}")
                    return False
                key_path = candidate
                logger.info(f"SFTP will authenticate using key file at path: {key_path}")
            else:
                # ssh-agent
                logger.info("SFTP will authenticate using ssh-agent (SSH_AUTH_SOCK detected).")

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
                "-oBatchMode=yes",
                "-P",
                str(port),
            ]
            if key_path is not None:
                cmd.extend(["-i", str(key_path)])
            cmd.extend(["-b", str(batch_path), f"{username}@{hostname}"])
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a single oprun_*.json to an SFTP server.")
    parser.add_argument("local_path_positional", type=str, nargs="?", help="Path to the oprun_*.json file to upload")
    parser.add_argument(
        "--local-path", dest="local_path_flag", type=str, help="Path to the oprun_*.json file to upload"
    )
    args = parser.parse_args()
    selected_path = args.local_path_flag or args.local_path_positional
    if not selected_path:
        parser.print_usage()
        print("error: local_path is required. Provide a positional path or use --local-path.")
        raise SystemExit(2)
    upload_run_sftp(selected_path)
