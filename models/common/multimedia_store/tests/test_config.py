# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The media embedding store's configuration from PREFILL_MEDIA_STORE_* and MEDIA_STORE_KEY_ENCODING; needs no
Mooncake."""

import pytest

from models.common.multimedia_store.config import (
    KEY_ENCODING_ENV,
    MediaEmbeddingStoreConfig,
    key_encoding_from_env,
    parse_bytes,
)

ME = "10.0.0.1:17800"


@pytest.mark.parametrize(
    "text, want", [("1G", 1 << 30), ("2GiB", 2 << 30), ("512M", 512 << 20), ("4096", 4096), ("1t", 1 << 40)]
)
def test_parse_bytes(text, want):
    assert parse_bytes(text) == want


def test_config_off_without_master():
    assert MediaEmbeddingStoreConfig.from_env({}) is None


def test_config_from_env():
    env = {
        "PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051",
        "PREFILL_MEDIA_STORE_SEGMENT": ME,
        "PREFILL_MEDIA_STORE_METADATA": "http://10.0.0.9:9000/metadata",
        "PREFILL_MEDIA_STORE_DEVICE": "rocep201s0f0",
    }
    c = MediaEmbeddingStoreConfig.from_env(env)
    assert (c.master, c.segment, c.segment_bytes, c.device) == ("10.0.0.9:50051", ME, 1 << 30, "rocep201s0f0")
    env["PREFILL_MEDIA_STORE_BYTES"] = "2G"
    assert MediaEmbeddingStoreConfig.from_env(env).segment_bytes == 2 << 30
    assert (c.metadata, c.protocol, c.key_encoding) == ("http://10.0.0.9:9000/metadata", "rdma", "")
    env[KEY_ENCODING_ENV] = "hex"
    assert MediaEmbeddingStoreConfig.from_env(env).key_encoding == "hex"


def test_config_pull_mode_has_no_store():
    env = {
        "PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051",
        "PREFILL_MEDIA_STORE_SEGMENT": ME,
        "PREFILL_MEDIA_STORE_METADATA": "http://10.0.0.9:9000/metadata",
        "PREFILL_MEDIA_STORE_BYTES": "0",
    }
    assert MediaEmbeddingStoreConfig.from_env(env).segment_bytes == 0


def test_config_needs_segment_and_metadata(expect_error):
    with expect_error(ValueError, "SEGMENT and PREFILL_MEDIA_STORE_METADATA must be set"):
        MediaEmbeddingStoreConfig.from_env({"PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051"})


def test_key_encoding_from_env(expect_error):
    assert key_encoding_from_env({}) == key_encoding_from_env({KEY_ENCODING_ENV: " "}) == ""
    assert key_encoding_from_env({KEY_ENCODING_ENV: "hex"}) == "hex"
    with expect_error(ValueError, "want 'hex' or unset"):
        key_encoding_from_env({KEY_ENCODING_ENV: "base64"})
