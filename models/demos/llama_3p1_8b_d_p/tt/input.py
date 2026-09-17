# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host token preparation for the fixed SP4/TP8, 1024-token prefill input."""

import torch

CHUNK_SIZE = 1024
LOCAL_CHUNK_SIZE = 256
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 128256


def validate_chunk_range(actual_start, actual_end):
    """Validate logical bounds; a padded physical chunk never extends the context limit."""
    if type(actual_start) is not int or type(actual_end) is not int:
        raise TypeError("actual_start and actual_end must be eager Python ints")
    if actual_start < 0 or actual_start % 32:
        raise ValueError("actual_start must be nonnegative and aligned to 32 tokens")
    if not actual_start < actual_end <= MAX_SEQ_LEN:
        raise ValueError("a chunk must satisfy 0 <= actual_start < actual_end <= 2048")
    if actual_end - actual_start > CHUNK_SIZE:
        raise ValueError("a chunk can contain at most 1024 valid tokens")


def pack_token_ids(token_ids, *, actual_start, actual_end, pad_id=0):
    """Return CPU [1,1,1,1024] IDs in SP-row order, ready to shard on dimension 3.

    ``token_ids`` contains exactly the true input interval, without a padded suffix. Absolute
    256-token blocks belong to successive SP rows. Each row keeps encounter order, including for
    an overlapping restart such as [32,65). The runtime must not call this on an already reshuffled
    H2D input: that input goes directly to the model's device-token entry point.
    """
    validate_chunk_range(actual_start, actual_end)
    if type(pad_id) is not int or not 0 <= pad_id < VOCAB_SIZE:
        raise ValueError("pad_id must be an integer inside the model vocabulary")
    if not isinstance(token_ids, torch.Tensor):
        if not isinstance(token_ids, (tuple, list)) or any(type(token) is not int for token in token_ids):
            raise ValueError("token_ids must be a one-dimensional integer tensor or list of Python ints")
        token_ids = torch.tensor(token_ids, dtype=torch.int64)
    if token_ids.device.type != "cpu" or token_ids.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional CPU tensor")
    if token_ids.dtype not in (torch.int32, torch.int64, torch.uint32):
        raise ValueError("token_ids must use int32, int64 or uint32")
    if token_ids.numel() != actual_end - actual_start:
        raise ValueError("token_ids must contain exactly actual_end - actual_start valid tokens")
    ids = token_ids.to(torch.int64)
    if torch.any(ids < 0) or torch.any(ids >= VOCAB_SIZE):
        raise ValueError("token ID is outside the model vocabulary")
    padded = torch.full((CHUNK_SIZE,), pad_id, dtype=torch.int64)
    padded[: ids.numel()] = ids
    positions = torch.arange(actual_start, actual_start + CHUNK_SIZE)
    packed = torch.cat([padded[(positions // LOCAL_CHUNK_SIZE) % 4 == row] for row in range(4)])
    return packed.reshape(1, 1, 1, CHUNK_SIZE)


def upload_token_chunk(mesh_device, token_ids, *, actual_start, actual_end, pad_id=0):
    """Allocate caller-owned UINT32 row-major device IDs; this is an explicit host input step."""
    import ttnn

    if tuple(mesh_device.shape) != (4, 8) or mesh_device.get_num_devices() != 32:
        raise ValueError("token upload requires the full SP4/TP8 Galaxy")
    packed = pack_token_ids(token_ids, actual_start=actual_start, actual_end=actual_end, pad_id=pad_id)
    return ttnn.from_torch(
        packed,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(4, 8), dims=(3, None)),
    )
