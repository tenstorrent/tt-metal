# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A Hugging Face cache snapshot (shards are relative symlinks into ../../blobs) must read like a plain directory."""

import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint, Qwen38CheckpointRowReader

ROWS = 64
HEAD_DIM = 8
NAME = "table.shard_0.weight"
SHARD = "part-0.safetensors"


class _SyntheticCheckpoint(Qwen38Checkpoint):
    """Real header/open/pread code over a tiny corpus without the pinned index."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._file_guard = None
        self.weight_map = {NAME: SHARD}
        self.shards = (SHARD,)


def test_snapshot_symlink_shards_read_like_real_files(tmp_path) -> None:
    torch.manual_seed(1)
    plain = tmp_path / "plain"
    plain.mkdir()
    save_file({NAME: torch.randn(ROWS, HEAD_DIM, dtype=torch.bfloat16)}, str(plain / SHARD))
    (plain / "config.json").write_text("{}")

    repo = tmp_path / "hub" / "models--x--y"
    snapshot = repo / "snapshots" / "rev0"
    (repo / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, entry in enumerate(sorted(plain.iterdir())):
        shutil.copy(entry, repo / "blobs" / f"sha{index}")
        (snapshot / entry.name).symlink_to(f"../../blobs/sha{index}")
    assert (snapshot / SHARD).is_symlink()

    indices = torch.tensor([[0, 5, 5], [ROWS - 1, 3, 0]], dtype=torch.long)
    real, linked = _SyntheticCheckpoint(plain), _SyntheticCheckpoint(snapshot)
    expected = real.tensor_rows(NAME, indices)
    assert torch.equal(linked.tensor_rows(NAME, indices).view(torch.int16), expected.view(torch.int16))

    reader_real, reader_linked = Qwen38CheckpointRowReader(real, NAME), Qwen38CheckpointRowReader(linked, NAME)
    try:
        assert reader_linked.read_rows([0, 5, 5, ROWS - 1]) == reader_real.read_rows([0, 5, 5, ROWS - 1])
        assert torch.equal(reader_linked.read(indices).view(torch.int16), expected.view(torch.int16))
    finally:
        reader_real.close()
        reader_linked.close()
