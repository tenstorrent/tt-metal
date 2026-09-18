# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Marker package: installed into the serving image via tt-model.yaml ``runtime.extension`` so that ``blobfile`` (needed by
``tiktoken.load.load_tiktoken_bpe`` for the local Kimi vocab file) and ``tiktoken`` are present. No code."""
