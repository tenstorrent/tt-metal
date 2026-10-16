# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID


@pytest.fixture
def batched_env(monkeypatch):
    monkeypatch.setenv("PREFILL_SP", "4")
    monkeypatch.setenv("PREFILL_TP", "1")
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "128")
    monkeypatch.setenv("PREFILL_MAX_CHUNKS_PER_BATCH", "4")
    monkeypatch.delenv("PREFILL_MTP_LEVELS", raising=False)
    producer._load_env_config()
    yield
    monkeypatch.undo()
    producer._load_env_config()


def test_each_chunk_lands_in_its_column_of_every_row(batched_env):
    sp, chunk, k_max = 4, 128, 4
    stride = chunk // sp
    a = list(range(1000, 1000 + chunk))
    b = list(range(2000, 2000 + chunk))
    rows = producer._h2d_rows_batch([(a, 49), (b, 0)])
    assert rows.shape == (sp, 1, k_max * stride)
    rot_a = rotate_chunk_tokens(a, 49, sp)
    rot_b = rotate_chunk_tokens(b, 0, sp)
    for r in range(sp):
        assert rows[r, 0, 0:stride].tolist() == rot_a[r * stride : (r + 1) * stride]
        assert rows[r, 0, stride : 2 * stride].tolist() == rot_b[r * stride : (r + 1) * stride]
        assert set(rows[r, 0, 2 * stride :].tolist()) == {MTP_PAD_TOKEN_ID}


def test_metadata_follows_the_batched_layout(batched_env):
    assert producer.CHUNK_METADATA_SIZE_BYTES == 200
    meta = producer._pack_metadata((1, 49, 89), (2, 0, 128))
    assert len(meta) == 200
    assert meta[:8] == (2).to_bytes(4, "little") + (0).to_bytes(4, "little")


def test_single_chunk_push_is_unchanged(monkeypatch):
    monkeypatch.delenv("PREFILL_MAX_CHUNKS_PER_BATCH", raising=False)
    producer._load_env_config()
    assert producer.CHUNK_METADATA_SIZE_BYTES == 12
    assert producer._pack_metadata((3, 10, 20)) == b"".join(v.to_bytes(4, "little") for v in (3, 10, 20))
