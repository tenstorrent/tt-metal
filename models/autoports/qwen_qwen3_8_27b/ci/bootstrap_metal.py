#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""Restore the exact inherited native rotary fix before the CI source build."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    here = Path(__file__).resolve().parent
    manifest = json.loads((here/'metal-runtime.json').read_text())
    patch = here/'metal-runtime.patch'
    if digest(patch) != manifest['patch_sha256']:
        raise ValueError('Native source patch checksum mismatch')
    target = root/manifest['file']
    actual = digest(target)
    if actual == manifest['after_sha256']:
        print('Native rotary source already matches measured execution')
        return
    if actual != manifest['before_sha256']:
        raise ValueError('Unexpected native rotary source; refusing to patch another version')
    subprocess.run(['git','-C',str(root),'apply','--check',str(patch)],check=True)
    subprocess.run(['git','-C',str(root),'apply',str(patch)],check=True)
    if digest(target) != manifest['after_sha256']:
        raise ValueError('Native rotary source does not match measured execution')
    print('Native rotary source matches measured execution')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
