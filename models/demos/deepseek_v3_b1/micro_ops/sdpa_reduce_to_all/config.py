# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass

SDPA_REDUCE_MAX_COMPUTE_BLOCK_SIZE = 8
SDPA_REDUCE_DEFAULT_NUM_LINKS = 2
SDPA_REDUCE_L1_ALIGNMENT = 16


@dataclass(frozen=True)
class SdpaReduceTuning:
    num_l_chunks: int
    compute_block_size: int


SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES = {
    # Derived from the SDPA reduce perf matrix. Explicit overrides bypass this
    # table for profiling sweeps, but default op setup must use a known tuning.
    2048: SdpaReduceTuning(num_l_chunks=4, compute_block_size=8),
    4096: SdpaReduceTuning(num_l_chunks=2, compute_block_size=4),
    8192: SdpaReduceTuning(num_l_chunks=2, compute_block_size=8),
    15232: SdpaReduceTuning(num_l_chunks=2, compute_block_size=8),
}


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


def find_min_num_l_chunks_for_payload(
    *,
    out_tiles: int,
    input_page_size_bytes: int,
    max_payload_size_bytes: int,
) -> int | None:
    if max_payload_size_bytes <= 0:
        raise ValueError(f"max_payload_size_bytes must be > 0, got {max_payload_size_bytes}")
    if input_page_size_bytes <= 0:
        raise ValueError(f"input_page_size_bytes must be > 0, got {input_page_size_bytes}")

    max_tiles_per_l_chunk = max_payload_size_bytes // input_page_size_bytes
    if max_tiles_per_l_chunk == 0:
        return None

    for num_l_chunks in range(1, out_tiles + 1):
        if out_tiles % num_l_chunks != 0:
            continue
        tiles_per_l_chunk = out_tiles // num_l_chunks
        if tiles_per_l_chunk <= max_tiles_per_l_chunk:
            return num_l_chunks
    return None


def find_largest_valid_compute_block_size(out_tiles: int) -> int:
    for candidate in range(min(out_tiles, SDPA_REDUCE_MAX_COMPUTE_BLOCK_SIZE), 0, -1):
        if out_tiles % candidate == 0:
            return candidate
    raise ValueError(f"Could not derive a valid compute block size for out_tiles={out_tiles}")


def derive_safe_sdpa_reduce_tuning(
    *,
    out_tiles: int,
    input_page_size_bytes: int,
    max_payload_size_bytes: int,
) -> SdpaReduceTuning:
    num_l_chunks = find_min_num_l_chunks_for_payload(
        out_tiles=out_tiles,
        input_page_size_bytes=input_page_size_bytes,
        max_payload_size_bytes=max_payload_size_bytes,
    )
    if num_l_chunks is None:
        raise ValueError(
            "Could not derive a safe SDPA reduce config because max_payload_size_bytes="
            f"{max_payload_size_bytes} cannot fit one L tile of {input_page_size_bytes} bytes"
        )

    return SdpaReduceTuning(
        num_l_chunks=num_l_chunks,
        compute_block_size=find_largest_valid_compute_block_size(out_tiles),
    )


def get_sdpa_reduce_tuning(max_payload_size_bytes: int) -> SdpaReduceTuning | None:
    return SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES.get(max_payload_size_bytes)


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
    if max_payload_size_bytes is not None and max_payload_size_bytes <= 0:
        raise ValueError(f"max_payload_size_bytes must be > 0, got {max_payload_size_bytes}")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be > 0, got {num_cores}")
    if num_links <= 0:
        raise ValueError(f"num_links must be > 0, got {num_links}")
    if num_cores % num_links != 0:
        raise ValueError(f"num_cores={num_cores} must be divisible by num_links={num_links}")

    use_num_l_chunks_override = num_l_chunks_override is not None
    use_compute_block_size_override = compute_block_size_override is not None
    tuning = None
    if not (use_num_l_chunks_override and use_compute_block_size_override):
        if max_payload_size_bytes is None:
            raise ValueError(
                "max_payload_size_bytes must be provided when SDPA reduce tuning overrides are not fully specified"
            )
        tuning = get_sdpa_reduce_tuning(max_payload_size_bytes)
        if tuning is None:
            tuning = derive_safe_sdpa_reduce_tuning(
                out_tiles=out_tiles,
                input_page_size_bytes=input_page_size_bytes,
                max_payload_size_bytes=max_payload_size_bytes,
            )
            supported_payloads = ", ".join(
                str(payload) for payload in sorted(SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES)
            )
            warnings.warn(
                "Using untuned SDPA reduce config for max_payload_size_bytes="
                f"{max_payload_size_bytes}: num_l_chunks={tuning.num_l_chunks}, "
                f"compute_block_size={tuning.compute_block_size}. Known tuned payloads are "
                f"{{{supported_payloads}}}. Benchmark this payload and add an entry to "
                "SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES if it will be used in production.",
                RuntimeWarning,
                stacklevel=2,
            )

    num_l_chunks = num_l_chunks_override if use_num_l_chunks_override else tuning.num_l_chunks
    if num_l_chunks <= 0:
        raise ValueError(f"num_l_chunks must be > 0, got {num_l_chunks}")
    if out_tiles % num_l_chunks != 0:
        raise ValueError(f"out_tiles={out_tiles} must be divisible by num_l_chunks={num_l_chunks}")

    tiles_per_l_chunk = out_tiles // num_l_chunks
    l_chunk_size_bytes = tiles_per_l_chunk * input_page_size_bytes
    if max_payload_size_bytes is not None and l_chunk_size_bytes > max_payload_size_bytes:
        min_num_l_chunks = find_min_num_l_chunks_for_payload(
            out_tiles=out_tiles,
            input_page_size_bytes=input_page_size_bytes,
            max_payload_size_bytes=max_payload_size_bytes,
        )
        update_hint = (
            f"Increase num_l_chunks to at least {min_num_l_chunks} for this shape/payload"
            if min_num_l_chunks is not None
            else "No valid num_l_chunks can fit even one L tile in this payload"
        )
        raise ValueError(
            "SDPA reduce config mismatch: "
            f"num_l_chunks={num_l_chunks} gives tiles_per_l_chunk={tiles_per_l_chunk} and "
            f"l_chunk_size_bytes={l_chunk_size_bytes}, but max_payload_size_bytes={max_payload_size_bytes}. "
            f"{update_hint}. If this is a new tuned payload, update "
            "SDPA_REDUCE_TUNING_BY_MAX_PAYLOAD_SIZE_BYTES or pass explicit overrides for profiling."
        )

    compute_block_size = compute_block_size_override if use_compute_block_size_override else tuning.compute_block_size
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
