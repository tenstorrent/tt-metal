# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload the health-check CSV results to the Data-team SFTP endpoint.

Private-key resolution is the SFTP half of the launch-mode divergence:

    orchestration (k8s)  the ``SFTP_KEY_PATH`` env var points at a mounted Secret
                         file, falling back to ``~/.ssh/sftp_upload_key``.
    slurm (default)      ``SFTP_PRIVATE_KEY_B64`` (base64 PEM/OpenSSH) or
                         ``SFTP_PRIVATE_KEY`` (raw text) env vars, then
                         ``{log_dir}/.sftp_upload_key`` / ``~/.ssh/sftp_upload_key``.
                         The env paths let the Slurm flow inject the key via
                         ``docker --env-file`` without writing it to a
                         bind-mounted (world-readable) directory.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import paramiko

log = logging.getLogger(__name__)


def resolve_sftp_key_path(log_dir: str) -> Optional[Path]:
    """Prefer {log_dir}/.sftp_upload_key; fall back to ~/.ssh/sftp_upload_key."""
    candidates = (
        Path(log_dir) / ".sftp_upload_key",
        Path.home() / ".ssh" / "sftp_upload_key",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _pkey_from_text(key_text: str):
    """Load an RSA or Ed25519 private key from PEM/OpenSSH text."""
    for key_cls in (paramiko.RSAKey, paramiko.Ed25519Key):
        try:
            return key_cls.from_private_key(io.StringIO(key_text))
        except Exception:
            continue
    return None


def _pkey_from_file(path: Path):
    """Load an RSA or Ed25519 private key from a file."""
    for key_cls in (paramiko.RSAKey, paramiko.Ed25519Key):
        try:
            return key_cls.from_private_key_file(str(path))
        except Exception:
            continue
    return None


def _sftp_pkey_orchestration(log_dir: str):
    """k8s: key comes from the SFTP_KEY_PATH mounted Secret file."""
    key_env = os.environ.get("SFTP_KEY_PATH")
    if key_env:
        path = Path(key_env)
        if path.is_file():
            pkey = _pkey_from_file(path)
            if pkey is None:
                log.warning("SFTP key %s could not be parsed as RSA/Ed25519", path)
            return pkey
        log.warning("SFTP_KEY_PATH %s does not exist", key_env)
    fallback = Path.home() / ".ssh" / "sftp_upload_key"
    if fallback.is_file():
        return _pkey_from_file(fallback)
    return None


def _sftp_pkey_slurm(log_dir: str):
    """slurm: env (b64/raw) then a key file in log_dir / ~/.ssh."""
    b64 = os.environ.get("SFTP_PRIVATE_KEY_B64")
    raw = os.environ.get("SFTP_PRIVATE_KEY")
    key_text = None
    if b64:
        try:
            key_text = base64.b64decode(b64).decode("utf-8")
        except Exception as exc:
            log.warning("Could not decode SFTP_PRIVATE_KEY_B64: %s", exc)
    elif raw:
        key_text = raw

    if key_text:
        pkey = _pkey_from_text(key_text)
        if pkey is not None:
            return pkey
        log.warning("SFTP key from env could not be parsed as RSA/Ed25519")

    key_path = resolve_sftp_key_path(log_dir)
    if key_path is None:
        return None
    pkey = _pkey_from_file(key_path)
    if pkey is None:
        log.warning("SFTP key file %s could not be parsed as RSA/Ed25519", key_path)
    return pkey


def _load_sftp_pkey(log_dir: str, launch_mode: str = "slurm"):
    """Resolve the SFTP private key for the launch mode. Returns a paramiko PKey
    or None if no key could be loaded."""
    if launch_mode == "orchestration":
        return _sftp_pkey_orchestration(log_dir)
    return _sftp_pkey_slurm(log_dir)


def upload_csv_sftp(
    csv_dir: str,
    sftp_user: str,
    sftp_host: str,
    log_dir: str = "",
    launch_mode: str = "slurm",
) -> None:
    """Upload all CSV files in csv_dir to the SFTP server."""
    pkey = _load_sftp_pkey(log_dir, launch_mode)
    if pkey is None:
        log.warning(
            "No SFTP private key found (launch_mode=%s); skipping CSV upload",
            launch_mode,
        )
        return

    csv_files = list(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        log.info("No CSV files to upload")
        return

    log.info("Uploading %d CSV file(s) to %s@%s ...", len(csv_files), sftp_user, sftp_host)
    transport = None
    try:
        transport = paramiko.Transport((sftp_host, 22))
        transport.connect(username=sftp_user, pkey=pkey)
        sftp = paramiko.SFTPClient.from_transport(transport)

        for csv_file in csv_files:
            remote_path = csv_file.name
            log.info("Uploading %s", csv_file.name)
            sftp.put(str(csv_file), remote_path)

        sftp.close()
        log.info("SFTP upload complete")
    except Exception as exc:
        log.warning("SFTP upload failed: %s", exc)
    finally:
        if transport:
            transport.close()
