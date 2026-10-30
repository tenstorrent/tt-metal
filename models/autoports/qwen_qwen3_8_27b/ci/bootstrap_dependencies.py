#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""Prepare the measured autoport dependency sources before the plugin installer.

Usage: python3 bootstrap_dependencies.py /path/to/vllm-tt-plugin
No packages are installed here. The caller subsequently sources the plugin's
existing docs/install-vllm-tt.sh, retaining its dependency overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(repo: Path) -> dict:
    resources = Path(__file__).resolve().parent
    manifest = json.loads((resources / 'dependency-source.json').read_text())
    patch = resources / 'vllm-tt-runtime.patch'
    if digest(patch) != manifest['patch_sha256']:
        raise RuntimeError('Packaged runtime patch checksum mismatch')
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    if head != manifest['parent_sha']:
        raise RuntimeError(f'Plugin HEAD {head} must equal pinned parent {manifest["parent_sha"]}')
    paths = manifest['runtime_sha256']
    def runtime_matches() -> bool:
        return all((repo / p).is_file() and digest(repo / p) == h for p, h in paths.items())
    installer = repo / 'docs/install-vllm-tt.sh'
    original = git(repo, 'show', 'HEAD:docs/install-vllm-tt.sh').decode()
    old_requirements = 'https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/requirements/common.txt'
    old_install = 'VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm==0.30.0'
    if original.count(old_requirements) != 1 or original.count(old_install) != 1:
        raise RuntimeError('Pinned installer no longer has the expected source locations')
    exact = original.replace(old_requirements,
        f'https://raw.githubusercontent.com/vllm-project/vllm/{manifest["vllm_sha"]}/requirements/common.txt')
    exact = exact.replace(old_install,
        'VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm '
        f'"vllm @ git+https://github.com/vllm-project/vllm.git@{manifest["vllm_sha"]}"')
    if installer.read_text() not in (original, exact):
        raise RuntimeError('Installer has unexpected edits; refusing to overwrite')
    # Repeat invocation is allowed only if the complete measured runtime matches.
    if not runtime_matches():
        if git(repo, 'status', '--porcelain', '--', 'src/').strip():
            raise RuntimeError('Plugin runtime has unexpected edits; refusing to overwrite')
        subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(repo), 'apply', str(patch)], check=True)
        if not runtime_matches():
            raise RuntimeError('Patched runtime differs from the measured source manifest')
    installer.write_text(exact)
    result = {
        'parent_sha': head,
        'measured_plugin_sha': manifest['measured_plugin_sha'],
        'runtime_files_verified': len(paths),
        'vllm_sha': manifest['vllm_sha'],
        'installer_sha256': digest(installer),
        'dependency_overrides_sha256': digest(repo / 'docs/vllm-overrides.txt'),
        'packages_installed': False,
    }
    print('QWEN38_DEPENDENCY_SOURCE ' + json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plugin_checkout', type=Path)
    args = parser.parse_args()
    prepare(args.plugin_checkout.resolve())
