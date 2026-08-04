# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload the health-check CSV results to the Data-team SFTP endpoint."""

from __future__ import annotations

import logging
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


def upload_csv_sftp(
    csv_dir: str, sftp_user: str, sftp_host: str, log_dir: str = ""
) -> None:
    """Upload all CSV files in csv_dir to the SFTP server."""
    sftp_key_path = resolve_sftp_key_path(log_dir)
    if sftp_key_path is None:
        log.warning(
            "SFTP private key not found under %s or ~/.ssh/sftp_upload_key; "
            "skipping CSV upload",
            log_dir or "<log_dir>",
        )
        return

    csv_files = list(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        log.info("No CSV files to upload")
        return

    log.info(
        "Uploading %d CSV file(s) to %s@%s ...", len(csv_files), sftp_user, sftp_host
    )
    try:
        pkey = paramiko.RSAKey.from_private_key_file(str(sftp_key_path))
    except Exception:
        try:
            pkey = paramiko.Ed25519Key.from_private_key_file(str(sftp_key_path))
        except Exception as exc:
            log.warning("Failed to load SFTP key: %s", exc)
            return

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
