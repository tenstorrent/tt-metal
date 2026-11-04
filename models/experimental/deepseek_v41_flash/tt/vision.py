# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash image encoder on one 1x4 stage: the ViT (32 blocks, 2D RoPE, full attention
inside one image) and the 3x3 aligner of the checkpoint's ``inference/vision.py``.

Tensor parallel over the stage's 4 chips: each chip holds 4 of the 16 heads, 1/4 of the MLP and 1/4
of the aligner's hidden width, and the row-parallel projections end in a TP all-reduce. Activations
are bf16 ``[1, 1, n, dim]``, replicated. The weights cost about 245 MB per chip in bf16.

One image at a time: its patches are padded to a bucket length (a multiple of the SDPA chunk), and
``cu_window_seqlens = [0, n, n_pad]`` keeps the padded tail its own attention window. After the
final norm the tail rows are zeroed, so the 3x3 unfold -- a ``ttnn.embedding`` row gather over an
index table -- reads the zero row ``n`` wherever a window leaves the image, as the reference's
``F.pad`` does. ``aligner.w1`` is stored in the gather's (kernel position, channel) order.

``encode(patches, n_h, n_w)`` returns the aligner rows ``[n_llm_h * n_llm_w, text_hidden_size]``
(bf16, host); ``image_rows`` adds the learned start / newline / end vectors for a whole image span.

Traced: ``capture`` records one trace per ViT bucket (ViT and aligner, the aligner padded to the most tokens an
image of that bucket can make, :func:`aligner_pad`), and :class:`EmbeddingRowWriter` one for the row write into
the embedding table. Their device buffers are allocated at capture, so an image then only replays: no program
compiles (no new CCL semaphores in L1_SMALL) and nothing is allocated. A traced image's data moves over host
sockets only (:class:`ImageSockets`), as a decode step's does: each trace takes its inputs with
``recv_async_h2d`` and the tower's sends the aligner rows back with ``send_async_d2h``.
"""

import functools
import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tt.config import dequantized
from models.experimental.deepseek_v41_flash.tt.decode.engram import open_rows_sockets
from models.experimental.deepseek_v4_flash.tt.decode.moe import _tp_all_reduce
from models.experimental.deepseek_v4_flash.tt.model import _d2h_page_plan
from models.experimental.deepseek_v4_flash.tt.prefill.attention import _rot_transformation_mat
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache, _load_weight, _materialize
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

SDPA_CHUNK = 128
# Padded patch counts of the ViT and token counts of the aligner: a few shapes instead of every
# multiple of the chunk, so ``warmup`` can compile every ViT program at startup. Past the last
# bucket the length rounds up to a multiple of the chunk (or tile), which only the eager path takes:
# an image of at most 1024 tokens has at most 9 * 1021 patches, inside the last ViT bucket.
VIT_BUCKETS = (256, 512, 1024, 2048, 3072, 4096, 6144, 9216)
ALIGN_BUCKETS = (64, 128, 192, 256, 384, 512, 640, 768, 896, 1024)
# Token types of ``image_processor.image_token_types`` (TEXT is -1).
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
# An image span reaches the LLM as token ids past the vocabulary, rows of an extended embedding table: the three
# learned delimiters at ``vocab + DELIMITER_ROWS[type]``, then the IMAGE rows at ``vocab + FIRST_IMAGE_ROW + slot``.
DELIMITER_ROWS = {IMAGE_START: 0, IMAGE_NEW_LINE: 1, IMAGE_END: 2}
FIRST_IMAGE_ROW = 3
# Rows per ``write_embedding_rows`` write: one fixed shape, so one program.
EMBEDDING_ROW_BLOCK = 32

# Receiver / sender cores of the tower's host sockets, clear of the pipeline's (0,0)-(0,4) (see ``decode/engram.py``).
TOWER_IN_CORE = (0, 5)  # its packed inputs
TOWER_OUT_CORE = (0, 6)  # D2H: the aligner rows
# A socket's FIFO takes its size of L1 on every worker core of its stage (the allocator runs in lockstep) and the
# decode stages have next to none to spare beside decode's circular buffers, so the tower's inputs share one socket a
# chip of two small pages; the table's stage opens none (EmbeddingRowWriter rides on sockets it already has).
TOWER_PAGE_BYTES = 256
SOCKET_FIFO_PAGES = 2


def tower_row_width(vision_config) -> int:
    """bf16 columns of one packed tower input row: the patch (padded to tiles), its cos and sin, and a tile column
    whose first element is the mask (see :meth:`DeepSeekV41Vision._packed_inputs`)."""
    patch_k = -(-3 * vision_config.patch_size**2 // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    head_dim = vision_config.hidden_size // vision_config.num_attention_heads
    return patch_k + 2 * head_dim + ttnn.TILE_SIZE


@dataclass
class ImageSockets:
    """The host sockets every traced image transfer goes over. An H2D socket feeds one chip, so a replicated input
    goes over one per chip of its stage (in mesh order); the D2H socket sends from chip 0 of the vision stage."""

    tower_in: list  # H2D on the vision stage, TOWER_PAGE_BYTES pages: an image's packed rows, then its ids
    tower_out: ttnn.D2HSocket  # the aligner rows, a page a fixed slice of one row
    # H2D on the table's stage, shared with what else writes them (the Engram rows of a decode step or a prefill
    # chunk), so written only with no step in flight; EmbeddingRowWriter pads its blocks to their page.
    table: list


def open_image_sockets(vision_config, vision_device: ttnn.MeshDevice, table_sockets: list) -> ImageSockets:
    """Open the tower's :class:`ImageSockets` on ``vision_device``, beside the already open ``table_sockets`` (one
    H2D socket per chip of the embedding table's stage, of any page size) the row writes take.

    The FIFO and configs are long-lived L1 buffers: open them before the stage's decode rings are built (as
    :func:`~.decode.engram.open_rows_sockets` says), so they sit above the rings and clear of what a prefill replay
    overwrites, and before any trace capture on the stage.
    """
    pipeline = active_system_config().pipeline
    alignment = pipeline.pcie_alignment
    row_bytes = 2 * tower_row_width(vision_config)
    if TOWER_PAGE_BYTES % alignment or row_bytes % TOWER_PAGE_BYTES:
        raise ValueError(
            f"a {TOWER_PAGE_BYTES} B tower page must be {alignment} B-aligned and divide a {row_bytes} B packed row"
        )
    out = ttnn.D2HSocket(
        vision_device,
        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*TOWER_OUT_CORE)),
        pipeline.d2h_fifo_bytes,
    )
    # A fixed page whatever the bucket: a slice of one row, so every bucket's rows split into whole pages.
    _, cols = _d2h_page_plan(vision_config.text_hidden_size, 2, pipeline.d2h_fifo_bytes, alignment)
    out.set_page_size(2 * cols)
    return ImageSockets(
        tower_in=open_rows_sockets(vision_device, TOWER_PAGE_BYTES, SOCKET_FIFO_PAGES, TOWER_IN_CORE),
        tower_out=out,
        table=table_sockets,
    )


def span_ids(types: torch.Tensor, slots: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """The token ids of one image span: ``types`` (``image_token_types``) -> int64 ids of the same length, the
    span's IMAGE positions taking the image-row ``slots`` (one per IMAGE position) in reading order."""
    image = types == IMAGE
    if int(image.sum()) != slots.numel():
        raise ValueError(f"{slots.numel()} slots for a span of {int(image.sum())} IMAGE tokens")
    ids = torch.empty(types.numel(), dtype=torch.int64)
    for kind, row in DELIMITER_ROWS.items():
        ids[types == kind] = vocab_size + row
    ids[image] = vocab_size + FIRST_IMAGE_ROW + slots.long()
    return ids


def image_mask(ids: ttnn.Tensor, vocab_size: int) -> ttnn.Tensor:
    """Token ids (uint32 ROW_MAJOR ``[1, T]``, on device) -> uint32 ROW_MAJOR ``[1, T]``, 1 where the id is an image
    span's row (``>= vocab_size``), else 0: the reference's ``image_mask``, from the ids alone, inside a trace."""
    tiled = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
    as_float = ttnn.typecast(tiled, ttnn.float32)  # exact: the ids are below 2^24
    ttnn.deallocate(tiled)
    flags = ttnn.ge(as_float, float(vocab_size))
    ttnn.deallocate(as_float)
    as_int = ttnn.typecast(flags, ttnn.uint32)
    ttnn.deallocate(flags)
    mask = ttnn.to_layout(as_int, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(as_int)
    return mask


def extend_embedding(table: ttnn.Tensor, image_rows: int, vocab_size: int) -> ttnn.Tensor:
    """The bf16 ROW_MAJOR embedding ``table`` (``vocab_size`` rows, on device) with ``image_rows`` zero rows appended,
    for :func:`write_embedding_rows` to fill. Deallocates ``table``."""
    if image_rows <= 0 or image_rows % ttnn.TILE_SIZE:
        raise ValueError(f"image_rows {image_rows} must be a positive multiple of {ttnn.TILE_SIZE}")
    vocab, hidden = table.shape[-2], table.shape[-1]
    if vocab != vocab_size:
        raise ValueError(f"the embedding table has {vocab} rows, the vocabulary {vocab_size}")
    zeros = ttnn.zeros(
        [image_rows, hidden],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=table.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    extended = ttnn.concat([ttnn.reshape(table, [vocab, hidden]), zeros], dim=0)
    ttnn.deallocate(zeros)
    ttnn.deallocate(table)
    return extended


def _row_blocks(table: ttnn.Tensor, rows: torch.Tensor, ids, vocab_size: int):
    """``rows`` (``[n, hidden]``) at token ids ``ids`` (each one of the ``table``'s image rows, ``vocab_size ..``) as
    blocks of :data:`EMBEDDING_ROW_BLOCK`: ``(rows [1, 1, block, hidden] bf16, ids [block] int32)``, a short block
    padded with zero rows at the id one past the table, which the write skips."""
    ids = torch.as_tensor(list(ids), dtype=torch.int64)
    total, hidden = table.shape[-2], table.shape[-1]
    if rows.dim() != 2 or rows.shape[0] != ids.numel() or rows.shape[1] != hidden:
        raise ValueError(f"expected rows [{ids.numel()}, {hidden}], got {tuple(rows.shape)}")
    if ids.numel() and (int(ids.min()) < vocab_size or int(ids.max()) >= total):
        raise ValueError(f"embedding row ids must lie in the image rows [{vocab_size}, {total})")
    n = EMBEDDING_ROW_BLOCK
    for begin in range(0, ids.numel(), n):
        block_rows = torch.zeros(n, hidden, dtype=torch.bfloat16)
        block_ids = torch.full((n,), total, dtype=torch.int32)
        chunk = ids[begin : begin + n]
        block_rows[: chunk.numel()] = rows[begin : begin + n].to(torch.bfloat16)
        block_ids[: chunk.numel()] = chunk.to(torch.int32)
        yield block_rows.reshape(1, 1, n, hidden), block_ids


def write_embedding_rows(table: ttnn.Tensor, rows: torch.Tensor, ids, vocab_size: int) -> None:
    """Write ``rows`` (``[n, hidden]``) into the :func:`extend_embedding` ``table`` at token ids ``ids`` (each one of
    its image rows, ``vocab_size ..``), in blocks of :data:`EMBEDDING_ROW_BLOCK`, so one program serves every write.

    Eager and synchronous. Call it with no decode or prefill in flight: it allocates (and frees) device tensors,
    which a captured trace may also use. Once traces are captured, write through :class:`EmbeddingRowWriter`.
    """
    device = table.device()
    view = ttnn.reshape(table, [1, 1, table.shape[-2], table.shape[-1]])
    replicate = ttnn.ReplicateTensorToMesh(device)
    for block_rows, block_ids in _row_blocks(table, rows, ids, vocab_size):
        new = ttnn.from_torch(
            block_rows,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        at = ttnn.from_torch(
            block_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        ttnn.transformer.deepseek_v4_prefill_write_rows(new, view, at)
        ttnn.deallocate(new)
        ttnn.deallocate(at)
    ttnn.synchronize_device(device)


def _page_bytes(*tensors: torch.Tensor) -> torch.Tensor:
    """``tensors`` back to back as raw bytes (uint8), the stream of socket pages their receives take in order."""
    return torch.cat([t.contiguous().reshape(-1).view(torch.uint8) for t in tensors])


def _over_sockets(dispatch, sockets: list, data: torch.Tensor, read=None) -> None:
    """Call ``dispatch``, which queues the programs that receive ``data`` (:func:`_page_bytes`) on every one of
    ``sockets`` and send ``read``, while ``data`` goes into each socket and ``read`` = ``(D2H socket, host tensor)``
    fills, one thread a socket; return once they are all done.

    The transfers start first: a socket write blocks while its FIFO is full and a D2H FIFO holds barely a page, so
    neither can wait for the dispatch, nor the dispatch for them (a program-cache miss may sync the host while the
    device sits in a receive). A socket copies raw bytes: bf16 is read as an int16 view, which DLPack takes.
    """
    jobs = [(socket.write_tensor, data) for socket in sockets]
    if read is not None:
        socket, out = read
        jobs.append((socket.read_tensor, out.view(torch.int16) if out.dtype == torch.bfloat16 else out))
    with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="image-sockets") as pool:
        pending = [pool.submit(job, t) for job, t in jobs]
        dispatch()
        for job in pending:
            job.result()


class EmbeddingRowWriter:
    """:func:`write_embedding_rows` as one captured trace on the table's device, allocating nothing once built.

    The trace takes one block's rows, then its ids, off :class:`ImageSockets`' ``table`` sockets (one per chip, the
    table being replicated) into buffers of its own, then writes them into the table; :meth:`write` streams every
    block in and replays once a block. Those sockets are another layer's, so their page is whatever it is: a block's
    rows are padded with zero rows to whole pages and its ids fill one page. The buffers are scratch, rewritten by
    every replay, so another trace's replay may overwrite them in between. Build it last, after every other
    allocation and capture on the device: the trace's addresses must not land on a buffer allocated after it.
    """

    def __init__(self, table: ttnn.Tensor, vocab_size: int, sockets: ImageSockets):
        self.table = table
        self.vocab_size = vocab_size
        self.sockets = sockets
        self.device = table.device()
        n, hidden = EMBEDDING_ROW_BLOCK, table.shape[-1]
        page = sockets.table[0].get_page_size()
        if page % 4 or page // 4 < n:
            raise ValueError(f"a {page} B socket page cannot carry a block's {n} uint32 ids")
        # Rows a whole number of pages holds, and the block padded to a multiple of them.
        per = math.lcm(2 * hidden, page) // (2 * hidden)
        self.padded_rows = -(-n // per) * per
        replicate = ttnn.ReplicateTensorToMesh(self.device)
        self.view = ttnn.reshape(table, [1, 1, table.shape[-2], hidden])
        # Row-major, so a tensor page is one socket page.
        self.rows = ttnn.from_torch(
            torch.zeros(1, 1, self.padded_rows * 2 * hidden // page, page // 2),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.ids = ttnn.from_torch(
            torch.zeros(1, 1, 1, page // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        # The compile run takes a block of zero rows at every id past the table, so it writes nothing.
        blank = (torch.zeros(n, hidden, dtype=torch.bfloat16), torch.full((n,), table.shape[-2], dtype=torch.int32))
        _over_sockets(self._write_graph, sockets.table, self._block_pages([blank]))
        self.trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._write_graph()
        ttnn.end_trace_capture(self.device, self.trace, cq_id=0)
        ttnn.synchronize_device(self.device)

    def _block_pages(self, blocks) -> torch.Tensor:
        """The socket pages of ``blocks`` (:func:`_row_blocks`), in replay order: each one's rows padded to
        :attr:`padded_rows`, then its ids padded to a page."""
        n, hidden = EMBEDDING_ROW_BLOCK, self.table.shape[-1]
        rows = torch.zeros(self.padded_rows, hidden, dtype=torch.bfloat16)
        ids = torch.zeros(self.ids.shape[-1], dtype=torch.int32)
        pages = []
        for block_rows, block_ids in blocks:
            rows[:n] = block_rows.reshape(n, hidden)
            ids[:n] = block_ids
            pages.append(_page_bytes(rows, ids))
        return torch.cat(pages)

    def _write_graph(self) -> None:
        """One block: its rows and ids off the sockets, then into the table."""
        for socket in self.sockets.table:
            ttnn.experimental.recv_async_h2d(self.rows, socket)
        for socket in self.sockets.table:
            ttnn.experimental.recv_async_h2d(self.ids, socket)
        n, hidden = EMBEDDING_ROW_BLOCK, self.table.shape[-1]
        padded = ttnn.reshape(self.rows, [1, 1, self.padded_rows, hidden])
        rows = ttnn.slice(padded, [0, 0, 0, 0], [1, 1, n, hidden])
        ids = ttnn.slice(self.ids, [0, 0, 0, 0], [1, 1, 1, n])
        flat_ids = ttnn.reshape(ids, [n])
        ttnn.transformer.deepseek_v4_prefill_write_rows(rows, self.view, flat_ids)
        for part, whole in ((rows, padded), (padded, self.rows), (flat_ids, ids), (ids, self.ids)):
            if part.buffer_address() != whole.buffer_address():
                ttnn.deallocate(part)

    def write(self, rows: torch.Tensor, ids) -> None:
        """:func:`write_embedding_rows` of ``rows`` at ``ids``, by trace replays fed over the sockets. Synchronous;
        call it with no decode or prefill in flight (the sockets are shared with them)."""
        blocks = list(_row_blocks(self.table, rows, ids, self.vocab_size))
        if not blocks:
            return

        def replay_all() -> None:
            for _ in blocks:
                ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=False)

        _over_sockets(replay_all, self.sockets.table, self._block_pages(blocks))
        ttnn.synchronize_device(self.device)


def _bucket(n: int, buckets, multiple: int) -> int:
    """The first bucket >= ``n``, else ``n`` rounded up to a multiple of ``multiple``."""
    for b in buckets:
        if n <= b:
            return b
    return -(-n // multiple) * multiple


def aligner_pad(n_pad: int, ratio: int, max_tokens: int) -> int:
    """The aligner rows a traced ViT bucket of ``n_pad`` rows pads to: room for the most tokens an image of up to
    ``n_pad - 1`` patches makes (one a patch or two high makes the most), at most ``max_tokens``, as a bucket."""
    n = n_pad - 1
    most = max(-(-h // ratio) * -(-(n // h) // ratio) for h in range(1, n + 1))
    return _bucket(min(most, max_tokens), ALIGN_BUCKETS, ttnn.TILE_SIZE)


# (dtype, layout) of each device input of the tower; all replicated in DRAM.
_INPUT_FORMATS = {
    "x": (ttnn.bfloat16, ttnn.TILE_LAYOUT),  # patches [1, 1, n_pad, patch_k]
    "cos": (ttnn.bfloat16, ttnn.TILE_LAYOUT),  # [1, 1, n_pad, head_dim]
    "sin": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "mask": (ttnn.bfloat16, ttnn.TILE_LAYOUT),  # [1, 1, n_pad, 1], 1 on the image's rows
    "cu": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),  # cu_window_seqlens [0, n, n_pad]
    "index": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),  # the aligner's gather index [1, nt_pad * ratio**2]
}


@dataclass
class _VisionTrace:
    tid: int | None  # None until captured
    rows: ttnn.Tensor  # [1, 1, n_pad * tower_row_width / cols, cols] bf16 (cols a socket page), the packed rows
    ids: ttnn.Tensor  # [1, 1, 1 + nt_pad * ratio**2 / w, w] uint32, cu_window_seqlens then the gather index
    nt_pad: int


def image_processor(snapshot_dir):
    """The checkpoint's ``inference/image_processor.py`` (resize plan, patchify, span token types)."""
    path = Path(snapshot_dir) / "inference" / "image_processor.py"
    spec = importlib.util.spec_from_file_location("deepseek_v41_image_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interleave_perm(head_dim: int) -> torch.Tensor:
    """Channel order that puts pair ``j`` of the checkpoint's split-half rotation (channels ``j`` and
    ``j + head_dim/2``) at channels ``2j, 2j+1``, the pairs ``rotary_embedding_llama`` rotates."""
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


def rope_tables(n_h: int, n_w: int, head_dim: int, theta: float):
    """cos, sin ``[n_h * n_w, head_dim]`` of ``vision.get_vision_cos_sin`` (row frequencies, then
    column frequencies), each value repeated for its interleaved pair."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1)
    return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)


def gather_index(n_h: int, n_w: int, ratio: int, nt_pad: int) -> torch.Tensor:
    """``[nt_pad * ratio**2]`` ViT rows of the unfold: for LLM token ``(i, j)`` and kernel position
    ``a * ratio + b`` the row ``(ratio*i + a) * n_w + (ratio*j + b)``; ``n_h * n_w`` (the zero row)
    outside the image and for the padded tokens."""
    n = n_h * n_w
    th, tw = -(-n_h // ratio), -(-n_w // ratio)
    rr = ratio * torch.arange(th).reshape(th, 1, 1, 1) + torch.arange(ratio).reshape(1, 1, ratio, 1)
    cc = ratio * torch.arange(tw).reshape(1, tw, 1, 1) + torch.arange(ratio).reshape(1, 1, 1, ratio)
    idx = torch.where((rr < n_h) & (cc < n_w), rr * n_w + cc, torch.full((), n))
    out = torch.full((nt_pad, ratio * ratio), n, dtype=torch.long)
    out[: th * tw] = idx.expand(th, tw, ratio, ratio).reshape(th * tw, ratio * ratio)
    return out.reshape(-1)


def _pack_qkv(t: torch.Tensor, n_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    """``wqkv`` weight ``[3D, D]`` or bias ``[3D]`` (q | k | v, head-major) -> rank-major blocks of
    (q | k | v) of that rank's heads, q and k channels in :func:`_interleave_perm` order, so a
    column shard over ``tp`` ranks hands each rank what ``nlp_create_qkv_heads`` splits."""
    rest = t.shape[1:]
    perm = _interleave_perm(head_dim)
    q, k, v = t.chunk(3, dim=0)
    q, k = (x.reshape(n_heads, head_dim, *rest)[:, perm].reshape(n_heads * head_dim, *rest) for x in (q, k))
    per_rank = n_heads // tp * head_dim
    return torch.cat([x[r * per_rank : (r + 1) * per_rank] for r in range(tp) for x in (q, k, v)])


class DeepSeekV41Vision:
    """ViT + aligner of DeepSeek-V4.1-Flash, tensor parallel over one 1xTP submesh (see the module docstring)."""

    def __init__(
        self,
        vision_config,
        loader: DeepseekV4WeightLoader,
        device: ttnn.MeshDevice,
        cache: WeightCache | None = None,
    ):
        """Upload the tower's weights to ``device`` (bf16, DRAM; cached under ``vision.*``).

        ``vision_config`` is :func:`~.config.load_vision_config`'s namespace.
        """
        v = vision_config
        self.config = v
        self.device = device
        self.tp = device.get_num_devices()
        self.head_dim = v.hidden_size // v.num_attention_heads
        self.local_heads = v.num_attention_heads // self.tp
        if v.num_attention_heads % self.tp or v.intermediate_size % (self.tp * ttnn.TILE_SIZE):
            raise ValueError(f"the vision tower does not split over TP{self.tp}")
        self.eps = 1e-6
        self.patch_k = -(-3 * v.patch_size**2 // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.row_width = tower_row_width(v)
        r2 = v.downsample_ratio**2
        tp, d, inter, heads, hd = self.tp, v.hidden_size, v.intermediate_size, v.num_attention_heads, self.head_dim

        replicate = ttnn.ReplicateTensorToMesh(device)
        cols = ttnn.ShardTensorToMesh(device, dim=-1)
        rows = ttnn.ShardTensorToMesh(device, dim=-2)

        def get(name):
            return dequantized(loader, name)()

        def patch_weight():
            w = get("vision.patch_embed.proj.weight")  # [D, 3 * p * p]
            padded = torch.zeros(self.patch_k, d)
            padded[: w.shape[1]] = w.t()
            return padded.reshape(1, 1, self.patch_k, d)

        def aligner_w1():
            # [dim, D * r2] in F.unfold's (channel, kernel position) order -> [r2 * D, dim] in the gather's order.
            w = get("aligner.w1.weight")
            w = w.view(w.shape[0], d, r2).permute(0, 2, 1).reshape(w.shape[0], r2 * d)
            return w.t().contiguous().reshape(1, 1, r2 * d, -1)

        # name -> (thunk of the host tensor, mesh mapper); the cache name carries the TP width.
        specs = {
            "patch_embed.weight": (patch_weight, replicate),
            "patch_embed.bias": (lambda: get("vision.patch_embed.proj.bias").reshape(1, 1, 1, -1), replicate),
            "norm.weight": (lambda: get("vision.norm.weight").reshape(1, 1, 1, -1), replicate),
            "aligner.w1": (aligner_w1, cols),
            "aligner.b1": (lambda: get("aligner.w1.bias").reshape(1, 1, 1, -1), cols),
            "aligner.w2": (lambda: get("aligner.w2.weight").t().reshape(1, 1, v.text_hidden_size, -1), rows),
            "aligner.b2": (lambda: get("aligner.w2.bias").reshape(1, 1, 1, -1), replicate),
        }
        for b in range(v.num_hidden_layers):
            p = f"vision.blocks.{b}."
            specs.update(
                {
                    f"blocks.{b}.norm1": (lambda p=p: get(p + "norm1.weight").reshape(1, 1, 1, -1), replicate),
                    f"blocks.{b}.norm2": (lambda p=p: get(p + "norm2.weight").reshape(1, 1, 1, -1), replicate),
                    f"blocks.{b}.wqkv": (
                        lambda p=p: _pack_qkv(get(p + "attn.wqkv.weight"), heads, hd, tp).t().reshape(1, 1, d, 3 * d),
                        cols,
                    ),
                    f"blocks.{b}.bqkv": (
                        lambda p=p: _pack_qkv(get(p + "attn.wqkv.bias"), heads, hd, tp).reshape(1, 1, 1, -1),
                        cols,
                    ),
                    f"blocks.{b}.wo": (lambda p=p: get(p + "attn.wo.weight").t().reshape(1, 1, d, d), rows),
                    f"blocks.{b}.bo": (lambda p=p: get(p + "attn.wo.bias").reshape(1, 1, 1, -1), replicate),
                    f"blocks.{b}.w_gate": (
                        lambda p=p: get(p + "mlp.w1.weight")[:inter].t().reshape(1, 1, d, inter),
                        cols,
                    ),
                    f"blocks.{b}.w_up": (
                        lambda p=p: get(p + "mlp.w1.weight")[inter:].t().reshape(1, 1, d, inter),
                        cols,
                    ),
                    f"blocks.{b}.w2": (lambda p=p: get(p + "mlp.w2.weight").t().reshape(1, 1, inter, d), rows),
                }
            )
        cache = _as_cache(cache).sub("vision")
        self.weights = {}
        for name, (thunk, mapper) in specs.items():
            file = cache.file(f"{name}.tp{tp}")
            self.weights[name] = _load_weight(
                _materialize(thunk, file, ttnn.bfloat16), device, cache_file_name=file, mesh_mapper=mapper
            )
        self.delimiters = {
            kind: loader.get_tensor(name, translate=False).to(torch.bfloat16)
            for kind, name in (
                (IMAGE_START, "image_start"),
                (IMAGE_NEW_LINE, "image_newline"),
                (IMAGE_END, "image_end"),
            )
        }
        self.trans_mat = ttnn.from_torch(
            _rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.matmul_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.traces: dict[int, _VisionTrace] = {}  # by ViT bucket, once captured
        self.sockets: ImageSockets | None = None  # what the traces take their inputs on and send their rows out of

    def _upload(self, t: torch.Tensor, dtype, layout) -> ttnn.Tensor:
        """A host tensor replicated over the stage, DRAM-interleaved."""
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def _block(self, x: ttnn.Tensor, b: int, cos, sin, cu_window) -> ttnn.Tensor:
        """``vision.Block``: ``x + attn(norm1(x))``, then ``+ mlp(norm2(x))``, on ``[1, 1, n_pad, D]``."""
        w = self.weights
        h = ttnn.rms_norm(x, weight=w[f"blocks.{b}.norm1"], epsilon=self.eps, compute_kernel_config=self.matmul_config)
        qkv = ttnn.linear(
            h, w[f"blocks.{b}.wqkv"], bias=w[f"blocks.{b}.bqkv"], compute_kernel_config=self.matmul_config
        )
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.local_heads,
            num_kv_heads=self.local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_rot = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_config
        )
        k_rot = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_config
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=self.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=SDPA_CHUNK,
                k_chunk_size=SDPA_CHUNK,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.sdpa_config,
            cu_window_seqlens=cu_window,
        )
        for t in (q_rot, k_rot, v):
            ttnn.deallocate(t)
        heads = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        partial = ttnn.linear(heads, w[f"blocks.{b}.wo"], compute_kernel_config=self.matmul_config)
        ttnn.deallocate(heads)
        attn = _tp_all_reduce(partial, self.device)
        ttnn.deallocate(partial)
        attn_out = ttnn.add(attn, w[f"blocks.{b}.bo"])
        ttnn.deallocate(attn)
        x_attn = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        h = ttnn.rms_norm(
            x_attn, weight=w[f"blocks.{b}.norm2"], epsilon=self.eps, compute_kernel_config=self.matmul_config
        )
        gate = ttnn.linear(h, w[f"blocks.{b}.w_gate"], compute_kernel_config=self.matmul_config)
        up = ttnn.linear(h, w[f"blocks.{b}.w_up"], compute_kernel_config=self.matmul_config)
        ttnn.deallocate(h)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        partial = ttnn.linear(act, w[f"blocks.{b}.w2"], compute_kernel_config=self.matmul_config)
        ttnn.deallocate(act)
        mlp = _tp_all_reduce(partial, self.device)
        ttnn.deallocate(partial)
        out = ttnn.add(x_attn, mlp)
        ttnn.deallocate(x_attn)
        ttnn.deallocate(mlp)
        return out

    def _vit_inputs(self, patches: torch.Tensor, n_h: int, n_w: int, n_pad: int) -> dict[str, torch.Tensor]:
        """The ViT's host inputs for one image padded to ``n_pad`` rows (see :data:`_INPUT_FORMATS`)."""
        n = n_h * n_w
        x = torch.zeros(1, 1, n_pad, self.patch_k)
        x[0, 0, :n, : patches[0].numel()] = patches.reshape(n, -1).float()
        cos, sin = torch.ones(1, 1, n_pad, self.head_dim), torch.zeros(1, 1, n_pad, self.head_dim)
        cos[0, 0, :n], sin[0, 0, :n] = rope_tables(n_h, n_w, self.head_dim, float(self.config.rope_theta))
        mask = torch.zeros(1, 1, n_pad, 1)
        mask[0, 0, :n] = 1.0
        cu = torch.tensor([0, n, n_pad], dtype=torch.int32)
        return {"x": x, "cos": cos, "sin": sin, "mask": mask, "cu": cu}

    def _index_input(self, n_h: int, n_w: int, nt_pad: int) -> torch.Tensor:
        """The aligner's gather index (host) for ``nt_pad`` tokens."""
        return gather_index(n_h, n_w, self.config.downsample_ratio, nt_pad).to(torch.int32).reshape(1, -1)

    def _vit_graph(self, inputs: dict[str, ttnn.Tensor], n_pad: int) -> ttnn.Tensor:
        """The ViT on device inputs (:meth:`_vit_inputs` uploaded), which it leaves allocated -> its features
        ``[n_pad, D]`` (ROW_MAJOR, replicated)."""
        v = self.config
        x = ttnn.linear(
            inputs["x"],
            self.weights["patch_embed.weight"],
            bias=self.weights["patch_embed.bias"],
            compute_kernel_config=self.matmul_config,
        )
        for b in range(v.num_hidden_layers):
            x = self._block(x, b, inputs["cos"], inputs["sin"], inputs["cu"])
        normed = ttnn.rms_norm(
            x, weight=self.weights["norm.weight"], epsilon=self.eps, compute_kernel_config=self.matmul_config
        )
        ttnn.deallocate(x)
        features = ttnn.multiply(normed, inputs["mask"])
        ttnn.deallocate(normed)
        out = ttnn.reshape(ttnn.to_layout(features, ttnn.ROW_MAJOR_LAYOUT), [n_pad, v.hidden_size])
        ttnn.deallocate(features)
        return out

    def _aligner_graph(self, features: ttnn.Tensor, index: ttnn.Tensor, nt_pad: int) -> ttnn.Tensor:
        """The aligner on the :meth:`_vit_graph` features and a device gather index (both left allocated) ->
        ``[1, 1, nt_pad, text_hidden_size]`` (TILE, replicated)."""
        v, r = self.config, self.config.downsample_ratio
        gathered = ttnn.embedding(index, features, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, nt_pad * r2, D]
        unfolded = ttnn.to_layout(ttnn.reshape(gathered, [1, 1, nt_pad, v.hidden_size * r * r]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(gathered)
        hidden = ttnn.linear(
            unfolded,
            self.weights["aligner.w1"],
            bias=self.weights["aligner.b1"],
            compute_kernel_config=self.matmul_config,
        )
        ttnn.deallocate(unfolded)
        act = ttnn.gelu(hidden, fast_and_approximate_mode=False)
        ttnn.deallocate(hidden)
        partial = ttnn.linear(act, self.weights["aligner.w2"], compute_kernel_config=self.matmul_config)
        ttnn.deallocate(act)
        reduced = _tp_all_reduce(partial, self.device)
        ttnn.deallocate(partial)
        out = ttnn.add(reduced, self.weights["aligner.b2"])
        ttnn.deallocate(reduced)
        return out

    def _graph(self, inputs: dict[str, ttnn.Tensor], n_pad: int, nt_pad: int) -> ttnn.Tensor:
        """ViT then aligner on device inputs (left allocated): what one trace of :meth:`capture` records."""
        features = self._vit_graph(inputs, n_pad)
        out = self._aligner_graph(features, inputs["index"], nt_pad)
        ttnn.deallocate(features)
        return out

    def _bucket_image(self, n_pad: int) -> tuple[torch.Tensor, int, int]:
        """``(patches, n_h, n_w)`` of a blank image filling the ViT bucket ``n_pad``, for the compile and capture runs."""
        n_h = 24
        n_w = (n_pad - 1) // n_h
        p = self.config.patch_size
        return torch.zeros(n_h * n_w, 3, p, p, dtype=torch.bfloat16), n_h, n_w

    def _bucket_inputs(self, n_pad: int, nt_pad: int) -> dict[str, torch.Tensor]:
        """Host inputs of :meth:`_bucket_image`."""
        patches, n_h, n_w = self._bucket_image(n_pad)
        return {**self._vit_inputs(patches, n_h, n_w, n_pad), "index": self._index_input(n_h, n_w, nt_pad)}

    def _packed_inputs(
        self, patches: torch.Tensor, n_h: int, n_w: int, n_pad: int, nt_pad: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The :data:`_INPUT_FORMATS` inputs of one image as the socket pages a trace receives: ``rows``
        ``[n_pad, tower_row_width]`` bf16, row ``i`` patch ``i`` (``patch_k``), its cos and sin (``head_dim`` each),
        then its mask and zeros (a tile); and ``ids`` ``[1 + nt_pad * ratio**2 / w, w]`` int32 (``w`` words a page),
        ``cu_window_seqlens`` and zeros, then the gather index."""
        vit = self._vit_inputs(patches, n_h, n_w, n_pad)
        pk, hd = self.patch_k, self.head_dim
        rows = torch.zeros(n_pad, self.row_width, dtype=torch.bfloat16)
        rows[:, :pk] = vit["x"][0, 0]
        rows[:, pk : pk + hd] = vit["cos"][0, 0]
        rows[:, pk + hd : pk + 2 * hd] = vit["sin"][0, 0]
        rows[:, pk + 2 * hd] = vit["mask"][0, 0, :, 0]
        w = self.sockets.tower_in[0].get_page_size() // 4
        index = self._index_input(n_h, n_w, nt_pad).reshape(-1, w)
        head = torch.zeros(1, w, dtype=torch.int32)
        head[0, : vit["cu"].numel()] = vit["cu"]
        return rows, torch.cat([head, index])

    def _socket_graph(self, trace: _VisionTrace, n_pad: int) -> None:
        """What one trace of :meth:`capture` records: the packed inputs off the sockets, split into
        :data:`_INPUT_FORMATS`' tensors, :meth:`_graph`, and its rows out of the D2H socket."""
        sockets, nt_pad, r = self.sockets, trace.nt_pad, self.config.downsample_ratio
        for socket in sockets.tower_in:
            ttnn.experimental.recv_async_h2d(trace.rows, socket)
        for socket in sockets.tower_in:
            ttnn.experimental.recv_async_h2d(trace.ids, socket)
        pk, hd = self.patch_k, self.head_dim
        rows = ttnn.reshape(trace.rows, [1, 1, n_pad, self.row_width])
        tiled = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
        if rows.buffer_address() != trace.rows.buffer_address():
            ttnn.deallocate(rows)
        inputs = {
            name: ttnn.slice(tiled, [0, 0, 0, start], [1, 1, n_pad, start + width])
            for name, start, width in (("x", 0, pk), ("cos", pk, hd), ("sin", pk + hd, hd), ("mask", pk + 2 * hd, 1))
        }
        ttnn.deallocate(tiled)
        w = trace.ids.shape[-1]
        cu = ttnn.slice(trace.ids, [0, 0, 0, 0], [1, 1, 1, 3])
        inputs["cu"] = ttnn.reshape(cu, [3])
        index = ttnn.slice(trace.ids, [0, 0, 1, 0], [1, 1, trace.ids.shape[2], w])
        inputs["index"] = ttnn.reshape(index, [1, nt_pad * r * r])
        for part, whole in ((cu, inputs["cu"]), (index, inputs["index"])):
            if part.buffer_address() != whole.buffer_address():
                ttnn.deallocate(part)
        out = self._graph(inputs, n_pad, nt_pad)
        for t in inputs.values():
            ttnn.deallocate(t)
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out)
        # Row-major, so a tensor page is one socket page: the rows as slices of the D2H socket's page.
        cols = sockets.tower_out.get_page_size() // 2
        paged = ttnn.reshape(out_rm, [nt_pad * self.config.text_hidden_size // cols, cols])
        ttnn.experimental.send_async_d2h(paged, sockets.tower_out)
        if paged.buffer_address() != out_rm.buffer_address():
            ttnn.deallocate(paged)
        ttnn.deallocate(out_rm)

    def _exchange(self, dispatch, nt_pad: int, rows: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Run ``dispatch`` (a :meth:`_socket_graph` run or replay) on the packed inputs ``rows`` / ``ids`` -> its
        aligner rows ``[nt_pad, text_hidden_size]`` (bf16, host), off the D2H socket."""
        hidden, socket = self.config.text_hidden_size, self.sockets.tower_out
        cols = socket.get_page_size() // 2
        out = torch.empty(nt_pad * hidden // cols, cols, dtype=torch.bfloat16)
        _over_sockets(dispatch, self.sockets.tower_in, _page_bytes(rows, ids), (socket, out))
        return out.reshape(nt_pad, hidden)

    def _upload_inputs(self, host: dict[str, torch.Tensor]) -> dict[str, ttnn.Tensor]:
        return {name: self._upload(t, *_INPUT_FORMATS[name]) for name, t in host.items()}

    def vit(self, patches: torch.Tensor, n_h: int, n_w: int, n_pad: int) -> ttnn.Tensor:
        """``vision.ViT`` of one image, eagerly -> its final features ``[n_pad, D]`` (ROW_MAJOR, replicated), rows
        ``>= n_h * n_w`` zero. ``patches`` is ``[n_h * n_w, 3, p, p]``."""
        inputs = self._upload_inputs(self._vit_inputs(patches, n_h, n_w, n_pad))
        out = self._vit_graph(inputs, n_pad)
        for t in inputs.values():
            ttnn.deallocate(t)
        return out

    def aligner(self, features: ttnn.Tensor, n_h: int, n_w: int) -> ttnn.Tensor:
        """``vision.Aligner`` on the :meth:`vit` features, eagerly -> ``[1, 1, nt_pad, text_hidden_size]`` (TILE,
        replicated); the first ``ceil(n_h/r) * ceil(n_w/r)`` rows are the image's tokens."""
        r = self.config.downsample_ratio
        nt_pad = _bucket(-(-n_h // r) * -(-n_w // r), ALIGN_BUCKETS, ttnn.TILE_SIZE)
        index = self._upload(self._index_input(n_h, n_w, nt_pad), *_INPUT_FORMATS["index"])
        out = self._aligner_graph(features, index, nt_pad)
        ttnn.deallocate(index)
        return out

    def encode(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """One image's patches ``[n_h * n_w, 3, p, p]`` -> its aligner rows ``[n_llm_h * n_llm_w,
        text_hidden_size]`` (bf16, host), in reading order. Once :meth:`capture` has run, by replaying its
        bucket's trace, its inputs and rows over the sockets, so it allocates nothing on the device; eagerly before."""
        r = self.config.downsample_ratio
        nt = -(-n_h // r) * -(-n_w // r)
        n_pad = _bucket(n_h * n_w + 1, VIT_BUCKETS, SDPA_CHUNK)  # + 1: the zero row the gather reads
        if not self.traces:
            features = self.vit(patches, n_h, n_w, n_pad)
            out = self.aligner(features, n_h, n_w)
            ttnn.deallocate(features)
            rows = ttnn.to_torch(ttnn.get_device_tensors(out)[0])[0, 0, :nt].to(torch.bfloat16)
            ttnn.deallocate(out)
            return rows
        trace = self.traces.get(n_pad)
        if trace is None or nt > trace.nt_pad:
            raise ValueError(f"no vision trace for a {n_h}x{n_w}-patch image ({nt} tokens)")
        rows = self._exchange(
            lambda: ttnn.execute_trace(self.device, trace.tid, cq_id=0, blocking=False),
            trace.nt_pad,
            *self._packed_inputs(patches, n_h, n_w, n_pad, trace.nt_pad),
        )
        return rows[:nt]

    def image_rows(self, patches: torch.Tensor, n_h: int, n_w: int, types: torch.Tensor) -> torch.Tensor:
        """The embedding rows of a whole image span ``[len(types), text_hidden_size]`` (bf16, host): the
        aligner rows at the IMAGE slots, the learned vectors at start / newline / end. ``types`` is
        ``image_processor.image_token_types(n_llm_h, n_llm_w)``."""
        out = torch.empty(types.numel(), self.config.text_hidden_size, dtype=torch.bfloat16)
        for kind, row in self.delimiters.items():
            out[types == kind] = row
        out[types == IMAGE] = self.encode(patches, n_h, n_w)
        return out

    def warmup(self):
        """Compile every program :meth:`capture` records: each ViT bucket with its :func:`aligner_pad` (a new shape
        compiles for seconds, and every all-reduce program keeps its semaphores in L1_SMALL for good)."""
        r, cap = self.config.downsample_ratio, self.config.max_image_tokens
        for n_pad in VIT_BUCKETS:
            nt_pad = aligner_pad(n_pad, r, cap)
            inputs = self._upload_inputs(self._bucket_inputs(n_pad, nt_pad))
            ttnn.deallocate(self._graph(inputs, n_pad, nt_pad))
            for t in inputs.values():
                ttnn.deallocate(t)
        ttnn.synchronize_device(self.device)

    def capture(self, sockets: ImageSockets):
        """Capture one trace per ViT bucket, ViT and aligner (padded to :func:`aligner_pad`), over buffers allocated
        here, taking its inputs off ``sockets`` and sending its rows out of them (see :meth:`_socket_graph`); from
        then on :meth:`encode` only streams an image in, replays and reads the rows back.

        The buffers are scratch, rewritten by every encode, so another trace's replay may overwrite them in
        between. Call it last, after every other allocation and capture on the device: a trace's addresses must not
        land on a buffer allocated after it.
        """
        self.sockets = sockets
        r, cap = self.config.downsample_ratio, self.config.max_image_tokens
        page = sockets.tower_in[0].get_page_size()
        w = page // 4
        for n_pad in VIT_BUCKETS:
            nt_pad = aligner_pad(n_pad, r, cap)
            if nt_pad * r * r % w:
                raise ValueError(f"the {nt_pad * r * r}-entry gather index does not fill {w}-word socket pages")
            # Row-major, so a tensor page is one socket page: the packed rows as page slices.
            trace = _VisionTrace(
                None,
                self._upload(
                    torch.zeros(1, 1, n_pad * 2 * self.row_width // page, page // 2),
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                self._upload(
                    torch.zeros(1, 1, 1 + nt_pad * r * r // w, w, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
                nt_pad,
            )
            # The compile run: programs at these buffers' addresses.
            packed = self._packed_inputs(*self._bucket_image(n_pad), n_pad, nt_pad)
            self._exchange(functools.partial(self._socket_graph, trace, n_pad), nt_pad, *packed)
            trace.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._socket_graph(trace, n_pad)
            ttnn.end_trace_capture(self.device, trace.tid, cq_id=0)
            self.traces[n_pad] = trace
        ttnn.synchronize_device(self.device)
