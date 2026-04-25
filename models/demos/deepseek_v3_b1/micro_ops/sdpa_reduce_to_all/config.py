# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

DEFAULT_SDPA_REDUCE_NUM_L_CHUNKS = 2
DEFAULT_SDPA_REDUCE_COMPUTE_BLOCK_SIZE = 8
SDPA_REDUCE_MAX_COMPUTE_BLOCK_SIZE = 8
SDPA_REDUCE_DEFAULT_NUM_LINKS = 2
SDPA_REDUCE_L1_ALIGNMENT = 16


@dataclass(frozen=True)
class SdpaReduceConfig:
    out_tiles: int
    num_l_chunks: int
    tiles_per_l_chunk: int
    l_chunk_size_bytes: int
    compute_block_size: int
    input_page_size_bytes: int
    aligned_page_size: int
    ms_tile_size_bytes: int
    slots_per_worker: int
    slots_per_round: int
    slot_size: int
    r2_buffer_offset: int
    brisc_buffer_size: int
    ncrisc_buffer_offset: int


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def compute_sdpa_out_tiles(batch_size: int, l_width: int, tile_height: int, tile_width: int) -> int:
    input_l_num_pages = (batch_size // tile_height) * (l_width // tile_width)

    pnh = 8
    dh = input_l_num_pages * tile_width
    dht = dh // tile_width
    pnht = pnh // tile_height
    return pnht * dht


def resolve_sdpa_reduce_config(
    *,
    batch_size: int,
    l_width: int,
    num_cores: int,
    tile_height: int = 8,
    tile_width: int = 32,
    bytes_per_element: int = 2,
    num_links: int = SDPA_REDUCE_DEFAULT_NUM_LINKS,
    packet_header_size_bytes: int = 0,
    l1_alignment: int = SDPA_REDUCE_L1_ALIGNMENT,
    max_payload_size_bytes: int | None = None,
    num_l_chunks_override: int | None = None,
    compute_block_size_override: int | None = None,
) -> SdpaReduceConfig:
    input_page_size_bytes = tile_height * tile_width * bytes_per_element
    out_tiles = compute_sdpa_out_tiles(batch_size, l_width, tile_height, tile_width)

    if out_tiles <= 0:
        raise ValueError(f"out_tiles must be > 0, got {out_tiles}")
    if input_page_size_bytes <= 0:
        raise ValueError(f"input_page_size_bytes must be > 0, got {input_page_size_bytes}")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be > 0, got {num_cores}")
    if num_links <= 0:
        raise ValueError(f"num_links must be > 0, got {num_links}")
    if num_cores % num_links != 0:
        raise ValueError(f"num_cores={num_cores} must be divisible by num_links={num_links}")

    num_l_chunks = DEFAULT_SDPA_REDUCE_NUM_L_CHUNKS if num_l_chunks_override is None else num_l_chunks_override
    if num_l_chunks <= 0:
        raise ValueError(f"num_l_chunks must be > 0, got {num_l_chunks}")
    if out_tiles % num_l_chunks != 0:
        raise ValueError(f"out_tiles={out_tiles} must be divisible by num_l_chunks={num_l_chunks}")

    tiles_per_l_chunk = out_tiles // num_l_chunks
    l_chunk_size_bytes = tiles_per_l_chunk * input_page_size_bytes
    if max_payload_size_bytes is not None and l_chunk_size_bytes > max_payload_size_bytes:
        raise ValueError(
            "SDPA L chunk payload exceeds the configured fabric payload limit: "
            f"{l_chunk_size_bytes} > {max_payload_size_bytes}"
        )

    compute_block_size = (
        DEFAULT_SDPA_REDUCE_COMPUTE_BLOCK_SIZE if compute_block_size_override is None else compute_block_size_override
    )
    if compute_block_size <= 0:
        raise ValueError(f"compute_block_size must be > 0, got {compute_block_size}")
    if compute_block_size > SDPA_REDUCE_MAX_COMPUTE_BLOCK_SIZE:
        raise ValueError(
            f"compute_block_size={compute_block_size} exceeds supported maximum "
            f"{SDPA_REDUCE_MAX_COMPUTE_BLOCK_SIZE}"
        )
    if out_tiles % compute_block_size != 0:
        raise ValueError(f"out_tiles={out_tiles} must be divisible by compute_block_size={compute_block_size}")

    num_workers_per_link = num_cores // num_links
    if num_workers_per_link % 2 != 0:
        raise ValueError(f"num_workers_per_link={num_workers_per_link} must be even for Type A/B worker split")
    workers_per_type = num_workers_per_link // 2
    slots_per_worker = 1 + num_l_chunks
    slots_per_round = workers_per_type * slots_per_worker
    if slots_per_round > 32:
        raise ValueError(f"slots_per_round={slots_per_round} exceeds 32-bit forwarder semaphore capacity")

    aligned_page_size = round_up(input_page_size_bytes, l1_alignment)
    slot_size = round_up(packet_header_size_bytes + l_chunk_size_bytes, l1_alignment)
    r2_buffer_offset = slots_per_round * slot_size
    brisc_buffer_size = 2 * slots_per_round * slot_size

    return SdpaReduceConfig(
        out_tiles=out_tiles,
        num_l_chunks=num_l_chunks,
        tiles_per_l_chunk=tiles_per_l_chunk,
        l_chunk_size_bytes=l_chunk_size_bytes,
        compute_block_size=compute_block_size,
        input_page_size_bytes=input_page_size_bytes,
        aligned_page_size=aligned_page_size,
        ms_tile_size_bytes=aligned_page_size,
        slots_per_worker=slots_per_worker,
        slots_per_round=slots_per_round,
        slot_size=slot_size,
        r2_buffer_offset=r2_buffer_offset,
        brisc_buffer_size=brisc_buffer_size,
        ncrisc_buffer_offset=brisc_buffer_size,
    )
