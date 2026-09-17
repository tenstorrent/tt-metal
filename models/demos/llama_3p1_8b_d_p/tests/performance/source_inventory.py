# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exact full-file hashes with a bounded number of independent shared-storage readers."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths):
    names = list(paths)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate source paths are not a valid inventory")
    # map preserves input order and propagates the first input-order error; no partial map is returned.
    with ThreadPoolExecutor(max_workers=8) as readers:
        values = list(readers.map(file_sha, names))
    return dict(zip(names, values))
