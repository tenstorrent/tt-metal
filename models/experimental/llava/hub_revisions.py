# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face Hub revisions for MoE-LLaVA.

Loading MoE-LLaVA-Qwen needs ``trust_remote_code=True``: its config's ``auto_map`` points at
modeling/tokenizer code in ``Qwen/Qwen-1_8B``. Remote code runs with the caller's
privileges, so every load goes through ``remote_code_kwargs()``. It pins both the checkpoint
repo (``revision``) and the repo that serves the code (``code_revision``) to fixed commits,
so a later push to either repo cannot change what runs.

Set the two SHAs below to the commits that were validated. ``MOE_LLAVA_REVISION`` and
``QWEN_CODE_REVISION`` environment variables override them (e.g. to test a newer snapshot).
"""

import os

from loguru import logger

MOE_LLAVA_CKPT = "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"

# TODO(before merge): fill in the validated 40-char commit SHAs from
#   https://huggingface.co/LanguageBind/MoE-LLaVA-Qwen-1.8B-4e/commits/main
#   https://huggingface.co/Qwen/Qwen-1_8B/commits/main
_PINNED_MOE_LLAVA_REVISION = None
_PINNED_QWEN_CODE_REVISION = None

_warned = False


def moe_llava_revision():
    return os.environ.get("MOE_LLAVA_REVISION") or _PINNED_MOE_LLAVA_REVISION


def qwen_code_revision():
    return os.environ.get("QWEN_CODE_REVISION") or _PINNED_QWEN_CODE_REVISION


def _warn_if_unpinned(revision, code_revision):
    global _warned
    if _warned or (revision and code_revision):
        return
    _warned = True
    logger.warning(
        "MoE-LLaVA Hub revisions are not pinned; trust_remote_code will run whatever is on 'main' of "
        f"{MOE_LLAVA_CKPT} and Qwen/Qwen-1_8B. Set the SHAs in models/experimental/llava/hub_revisions.py "
        "or MOE_LLAVA_REVISION / QWEN_CODE_REVISION."
    )


def checkpoint_kwargs():
    """kwargs for snapshot_download / hf_hub_download of the MoE-LLaVA checkpoint files."""
    revision = moe_llava_revision()
    return {"revision": revision} if revision else {}


def remote_code_kwargs():
    """kwargs for AutoConfig / AutoTokenizer / AutoModel .from_pretrained(MOE_LLAVA_CKPT, ...)."""
    revision = moe_llava_revision()
    code_revision = qwen_code_revision()
    _warn_if_unpinned(revision, code_revision)
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    if code_revision:
        kwargs["code_revision"] = code_revision
    return kwargs
