# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the offline-rounded gpt-oss weights named by environment variables come from.

GPT_OSS_DECODE_SPLIT_ROUND_DIR (a directory of layer_<L>.pt) and GPT_OSS_DENSE_ROUNDED_DECODE (one file) take either
a local path or a Hugging Face Hub location at a pinned revision:

    hf://<owner>/<repo>@<revision>[/<path in the repo>]

A Hub location is downloaded once into the Hub cache (huggingface_hub; HF_TOKEN for private repos) and its local path
returned: the directory's files for a directory, the file for a path ending in a file suffix. The revision is required,
so a later upload cannot change what a run loads.
"""

import functools
import os

HUB_PREFIX = "hf://"


def _parse(value):
    spec = value[len(HUB_PREFIX) :]
    owner, _, rest = spec.partition("/")
    name_rev, _, path = rest.partition("/")
    name, at, revision = name_rev.partition("@")
    if not (owner and name and at and revision):
        raise ValueError(f"{value!r}: expected hf://<owner>/<repo>@<revision>[/<path>]")
    return f"{owner}/{name}", revision, path.strip("/")


@functools.lru_cache(maxsize=None)
def resolve(value):
    """The local path of a weight location (see the module docstring); "" and local paths are returned as given."""
    value = (value or "").strip()
    if not value.startswith(HUB_PREFIX):
        return value
    repo, revision, path = _parse(value)
    from huggingface_hub import hf_hub_download, snapshot_download

    if path and os.path.splitext(path)[1]:  # one file
        return hf_hub_download(repo, path, revision=revision)
    root = snapshot_download(repo, revision=revision, allow_patterns=[f"{path}/*"] if path else None)
    return os.path.join(root, path) if path else root


def env_path(name):
    """resolve() of environment variable `name` ("" when unset)."""
    return resolve(os.environ.get(name, ""))


def dense_rounded_decode_tag():
    """Cache-name tag of the decode copies made from GPT_OSS_DENSE_ROUNDED_DECODE: "_drnd" plus a hash of the source as
    given (a pinned hf:// location includes its revision, so each revision gets its own files), "" when unset."""
    import hashlib

    spec = os.environ.get("GPT_OSS_DENSE_ROUNDED_DECODE", "").strip()
    return "_drnd" + hashlib.sha1(spec.encode()).hexdigest()[:10] if spec else ""
