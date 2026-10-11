# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental packed-byte TT cache transport; scheduler leases remain external.

No serving capability is enabled here. The caller MUST hold exclusive request,
page and device-execution leases, publish resident decode state and fence before
capture/restore. Writes target private pages and a quarantined recurrent slot.
The caller publishes the restored frontier only after codec validation and fence.
"""

from dataclasses import dataclass

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import MAX_CHUNK_BYTES, Segment


@dataclass(frozen=True)
class Window:
    logical_offset: int
    size: int
    start: int
    length: int


def page_windows(pages, page_bytes, *, limit=MAX_CHUNK_BYTES):
    """Coalesce adjacent logical pages only when their physical IDs are adjacent."""
    if type(page_bytes) is not int or page_bytes < 1 or type(limit) is not int or limit < page_bytes:
        raise ValueError("A transfer window must contain at least one complete physical page")
    if not pages or any(type(p) is not int or p < 0 for p in pages) or len(set(pages)) != len(pages):
        raise ValueError("A private prefix mapping requires distinct nonnegative physical pages")
    count = limit // page_bytes
    first = 0
    windows = []
    while first < len(pages):
        end = first + 1
        while end < len(pages) and end - first < count and pages[end] == pages[end - 1] + 1:
            end += 1
        windows.append(Window(first * page_bytes, (end - first) * page_bytes, pages[first], end - first))
        first = end
    return tuple(windows)


def overlapping(windows, offset, size):
    if type(offset) is not int or type(size) is not int or offset < 0 or size < 1:
        raise ValueError("Transfer range must have a nonnegative offset and positive size")
    if not windows or offset + size > windows[-1].logical_offset + windows[-1].size:
        raise ValueError("Transfer range exceeds its checkpoint segment")
    for window in windows:
        begin = max(offset, window.logical_offset)
        end = min(offset + size, window.logical_offset + window.size)
        if begin < end:
            yield window, begin - window.logical_offset, begin - offset, end - begin


class PackedCacheTransfer:
    """Bounded synchronous transport under an already-held external exclusive lease.

    The qualified default retains one host window at a time. The opt-in
    batched path retains at most 1 MiB of windows until each completion fence.
    No tensor address changes or BFP8 conversions occur. Full convolution rank
    buffers use read/modify/write; neighbouring slots retain their exact bytes.
    """

    def __init__(self, mesh, cache, checkpoint, *, slot, pages, batched=False):
        import ttnn

        self.ttnn = ttnn
        self.mesh, self.cache, self.checkpoint = mesh, cache, checkpoint
        self.slot, self.pages = slot, tuple(pages)
        if type(batched) is not bool:
            raise ValueError("Batched transport must be explicitly enabled or disabled")
        self.batched = batched
        if tuple(mesh.shape) != (1, checkpoint.layout.tp_size) or checkpoint.layout.tp_size != 4:
            raise ValueError("Packed cache transport requires the complete physical TP4 mesh")
        if type(slot) is not int or not 0 <= slot < cache.batch_size or not 1 <= cache.batch_size <= 32:
            raise ValueError("Recurrent slot is outside the supported fixed cache")
        if len(cache.layers) != len(checkpoint.layout.layer_types):
            raise ValueError("Cache layers do not match the complete checkpoint layout")
        if len(self.pages) != checkpoint.consumed // checkpoint.layout.page_tokens or any(
            type(p) is not int or not 0 <= p < cache.num_pages for p in self.pages
        ):
            raise ValueError("Physical pages do not cover exactly the consumed prefix")
        self.kv_windows = page_windows(self.pages, checkpoint.layout.kv_page_bytes)
        self.tensors = {}
        self.addresses = {}
        for layer, (state, kind) in enumerate(zip(cache.layers, checkpoint.layout.layer_types)):
            names = ("key", "value") if kind == "full_attention" else ("recurrent", "conv")
            for name in names:
                tensor = getattr(state, name)
                expected = {
                    "key": ((cache.num_pages, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
                    "value": ((cache.num_pages, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
                    "recurrent": ((cache.batch_size, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
                    "conv": ((cache.batch_size, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                }[name]
                if (tuple(tensor.shape), tensor.dtype, tensor.layout) != expected:
                    raise ValueError("Cache shape/precision/layout does not match packed TP4 ABI")
                if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    raise ValueError("Cache transfer requires canonical interleaved DRAM tensors")
                shards = ttnn.get_device_tensors(tensor)
                if len(shards) != 4:
                    raise ValueError("Missing physical tensor ranks")
                for rank, shard in enumerate(shards):
                    self.tensors[rank, layer, name] = shard
                    self.addresses[rank, layer, name] = shard.buffer_address()
        actual = checkpoint.layout
        if (actual.page_tokens, actual.kv_page_bytes, actual.recurrent_bytes, actual.conv_bytes) != (
            32,
            8 * 1088,
            12 * 128 * 128 * 4,
            3 * 2560 * 2,
        ):
            raise ValueError("Checkpoint byte geometry differs from the physical TT ABI")
        self.current = self.host = self.raw = self.view = None
        self.dirty = False
        self.max_host_window_bytes = 0
        self.bytes_read = self.bytes_written = 0
        self.windows_read = self.windows_written = 0
        self.max_host_staging_bytes = self.synchronizations = 0
        # Retain asynchronous buffers even when a fence raises. The exclusive
        # lease must quarantine the device before disposing of this transport.
        self.pending = []

    def _windows(self, segment):
        expected = next((s for s in self.checkpoint.segments() if s == segment), None)
        if expected is None:
            raise ValueError("Transfer segment is not part of this checkpoint")
        if segment.kind in ("key", "value"):
            return self.kv_windows
        return (Window(0, segment.size, self.slot, 1),)

    def _flush(self):
        if self.dirty:
            self.ttnn.copy_host_to_device_tensor(self.host, self.view)
            # Retain every host reference until the asynchronous write is done.
            self._sync()
            self.bytes_written += self.raw.numel()
            self.windows_written += 1
            self.dirty = False

    def _view(self, segment, window):
        tensor = self.tensors[segment.rank, segment.layer, segment.kind]
        return tensor if segment.kind == "conv" else self.ttnn.narrow(tensor, 0, window.start, window.length)

    def _raw(self, segment, window, host):
        import torch

        buffer = host.host_buffer().get_shard(self.ttnn.MeshCoordinate(0, segment.rank))
        if buffer is None:
            raise ValueError(f"Downloaded rank {segment.rank} is missing its parent mesh coordinate")
        raw = torch.from_dlpack(buffer)
        expected = self._physical_bytes(segment, window)
        if raw.dtype != torch.uint8 or raw.numel() != expected or expected > MAX_CHUNK_BYTES:
            raise ValueError("Downloaded packed buffer has unexpected physical size or dtype")
        return raw

    def _physical_bytes(self, segment, window):
        return self.cache.batch_size * segment.size if segment.kind == "conv" else window.size

    def _batches(self, segment, offset, size):
        batch, total = [], 0
        for overlap in overlapping(self._windows(segment), offset, size):
            count = self._physical_bytes(segment, overlap[0])
            if count > MAX_CHUNK_BYTES:
                raise ValueError("A physical transfer exceeds the bounded host staging budget")
            if total + count > MAX_CHUNK_BYTES:
                yield batch
                batch, total = [], 0
            batch.append(overlap)
            total += count
        if batch:
            yield batch

    def _sync(self):
        self.ttnn.synchronize_device(self.mesh)
        self.synchronizations += 1

    def _batched_transfer(self, segment, offset, size, data=None):
        """Submit a bounded group of windows before waiting for completion.

        Preserve physical bytes, including neighbours in convolution buffers.
        Both directions retain host/view references through their final fence.
        Reads still precede writes; eliminating those requires a separately
        qualified rank-local host allocation API. No device gather allocation
        or tensor address change is introduced here.
        """
        import torch

        if self.pending:
            raise RuntimeError("A previous asynchronous transfer did not finish; quarantine the destination")
        output = bytearray(size) if data is None else None
        for overlaps in self._batches(segment, offset, size):
            submitted = False
            try:
                for window, within, codec_offset, length in overlaps:
                    view = self._view(segment, window)
                    submitted = True
                    host = view.cpu(blocking=False)
                    self.pending.append((window, within, codec_offset, length, view, host))
                self.max_host_staging_bytes = max(
                    self.max_host_staging_bytes,
                    sum(self._physical_bytes(segment, value[0]) for value in self.pending),
                )
                self._sync()
                submitted = False
                for window, within, codec_offset, length, view, host in self.pending:
                    raw = self._raw(segment, window, host)
                    self.max_host_window_bytes = max(self.max_host_window_bytes, raw.numel())
                    self.bytes_read += raw.numel()
                    self.windows_read += 1
                    skip = self.slot * segment.size if segment.kind == "conv" else 0
                    target = raw[skip + within : skip + within + length]
                    if data is None:
                        output[codec_offset : codec_offset + length] = target.numpy().tobytes()
                    else:
                        value = torch.frombuffer(
                            bytearray(data[codec_offset : codec_offset + length]), dtype=torch.uint8
                        )
                        target.copy_(value)
                        # Mark before submission: an enqueue failure can still
                        # have put earlier work on the device command queue.
                        submitted = True
                        self.ttnn.copy_host_to_device_tensor(host, view)
                        self.bytes_written += raw.numel()
                        self.windows_written += 1
            finally:
                if submitted:
                    self._sync()
                # If the fence above fails, keep buffers pinned by self.pending.
                self.pending.clear()
        return bytes(output) if output is not None else None

    def _window(self, segment, window):
        key = (segment, window)
        if self.current != key:
            self._flush()
            self.raw = self.host = self.view = None
            view = self._view(segment, window)
            # Rank views retain their parent mesh coordinates. cpu() uses the
            # view's active-coordinate subset and allocates only that shard;
            # allocating with view.device() would stage the entire TP4 mesh.
            host = view.cpu(blocking=True)
            self.synchronizations += 1
            raw = self._raw(segment, window, host)
            self.view, self.host, self.raw, self.current = view, host, raw, key
            self.max_host_window_bytes = max(self.max_host_window_bytes, raw.numel())
            self.max_host_staging_bytes = max(self.max_host_staging_bytes, raw.numel())
            self.bytes_read += raw.numel()
            self.windows_read += 1
        skip = self.slot * segment.size if segment.kind == "conv" else 0
        return self.raw[skip : skip + window.size]

    def read(self, segment: Segment, offset: int, size: int) -> bytes:
        if size > MAX_CHUNK_BYTES:
            raise ValueError("Codec read exceeds bounded staging")
        if self.batched:
            return self._batched_transfer(segment, offset, size)
        output = bytearray(size)
        for window, source, destination, length in overlapping(self._windows(segment), offset, size):
            raw = self._window(segment, window)
            output[destination : destination + length] = raw[source : source + length].numpy().tobytes()
        return bytes(output)

    def write(self, segment: Segment, offset: int, data: bytes):
        import torch

        if not isinstance(data, bytes) or len(data) > MAX_CHUNK_BYTES:
            raise ValueError("Codec write requires bounded opaque bytes")
        if self.batched:
            self._batched_transfer(segment, offset, len(data), data)
            return
        for window, destination, source, length in overlapping(self._windows(segment), offset, len(data)):
            raw = self._window(segment, window)
            # bytearray supplies writable owned storage; no numerical conversion.
            value = torch.frombuffer(bytearray(data[source : source + length]), dtype=torch.uint8)
            raw[destination : destination + length].copy_(value)
            self.dirty = True

    def fence(self):
        self._flush()
        self._sync()
        self.pending.clear()
        for key, tensor in self.tensors.items():
            if tensor.buffer_address() != self.addresses[key]:
                raise ValueError("Cache destination address changed during transfer")

    def close(self, *, discard=False):
        if discard:
            # Previously flushed writes remain PRIVATE. Only the caller's
            # abort/quarantine can make the destination safe for another use.
            self.dirty = False
        self.fence()
        self.current = self.host = self.raw = self.view = None
