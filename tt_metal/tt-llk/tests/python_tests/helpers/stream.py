# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional

from ttexalens.tt_exalens_lib import (
    read_from_device,
    read_word_from_device,
    write_to_device,
    write_words_to_device,
)


class Stream:
    """

    SINGLE PRODUCER, SINGLE CONSUMER

    If you need bidirectional traffic, use two separate unidirectional streams

    Memory layout:
        [0:4]   write_idx  (std::uint32_t, producer-owned)
        [4:8]   read_idx   (std::uint32_t, consumer-owned)
        [8:8+N] buffer[N]  (char array, circular)
    """

    _WRITE_IDX_OFFSET = 0
    _READ_IDX_OFFSET = 4
    _BUFFER_OFFSET = 8

    def __init__(self, address: int, buffer_size: int, location: str = "0,0"):
        """Attach to an existing Stream laid out at *address* on the device.

        Args:
            address: L1 device address of the stream.
            buffer_size: Size of the circular buffer in bytes.
            location: Tensix core coordinate.
        """
        self._address = address
        self._buffer_size = buffer_size
        self._location = location

        # sync with device, even if it might be garbage, just in case.
        self._load_read_idx()
        self._load_write_idx()

    def _capacity(self) -> int:
        """Usable capacity of the ring buffer (one slot reserved as sentinel)."""
        return self._buffer_size - 1

    # ── low-level index helpers ──────────────────────────────────────

    def _load_read_idx(self) -> int:
        self._read_idx = read_word_from_device(
            self._location, self._address + self._READ_IDX_OFFSET
        )
        return self._read_idx

    def _load_write_idx(self) -> int:
        self._write_idx = read_word_from_device(
            self._location, self._address + self._WRITE_IDX_OFFSET
        )
        return self._write_idx

    def _store_read_idx(self, value: int) -> None:
        self._read_idx = value
        write_words_to_device(
            location=self._location,
            addr=self._address + self._READ_IDX_OFFSET,
            data=[self._read_idx],
        )

    def _store_write_idx(self, value: int) -> None:
        self._write_idx = value
        write_words_to_device(
            location=self._location,
            addr=self._address + self._WRITE_IDX_OFFSET,
            data=[self._write_idx],
        )

    def _producer_poll_free(self) -> int:
        """Poll for number of free bytes."""

        return (
            self._capacity() - self._write_idx + self._load_read_idx()
        ) % self._buffer_size

    def _producer_wait_free(self, timeout: float = 0.1) -> int:
        """Spin until at least one byte of free space is available."""

        deadline = time.monotonic() + timeout
        while True:
            free = self._producer_poll_free()
            if free > 0:
                return free

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Stream: timed out after {timeout}s waiting for free space"
                )
            time.sleep(0.001)

    def _consumer_poll_avail(self) -> int:
        """Poll for number of available bytes."""

        return (
            self._load_write_idx() - self._read_idx + self._buffer_size
        ) % self._buffer_size

    def _consumer_wait_avail(self, timeout: float = 0.1) -> int:
        """Spin until at least one byte of data is available."""

        deadline = time.monotonic() + timeout
        while True:
            avail = self._consumer_poll_avail()
            if avail > 0:
                return avail

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Stream: timed out after {timeout}s waiting for data"
                )
            time.sleep(0.001)

    def init(self) -> None:
        """Initialize the stream. Should be called only once, before all operations."""
        self._store_write_idx(0)
        self._store_read_idx(0)

    def push(self, data: bytes, timeout: float = 0.1) -> None:
        """Push *data* into the stream, blocking until space is available.

        Args:
            data: Raw bytes to send to the device.
            timeout: Maximum seconds to wait for free space per chunk.
        """
        remaining = len(data)
        offset = 0

        while remaining > 0:
            free = self._producer_wait_free(timeout)
            chunk = min(remaining, free)

            head = min(chunk, self._buffer_size - self._write_idx)
            if head > 0:
                write_to_device(
                    self._location,
                    self._address + self._BUFFER_OFFSET + self._write_idx,
                    data[offset : offset + head],
                )

            tail = chunk - head
            if tail > 0:
                write_to_device(
                    self._location,
                    self._address + self._BUFFER_OFFSET,
                    data[offset + head : offset + head + tail],
                )

            self._store_write_idx((self._write_idx + chunk) % self._buffer_size)

            offset += chunk
            remaining -= chunk

    def peek(self) -> Optional[int]:
        """Peek at the next byte in the stream without consuming it.

        Returns:
            The next byte if available, otherwise None.
        """
        avail = self._consumer_poll_avail()
        if avail > 0:
            data = read_from_device(
                self._location,
                self._address + self._BUFFER_OFFSET + self._read_idx,
                num_bytes=1,
            )
            return data[0]

        return None

    def pop(self, size: int, timeout: float = 2.0) -> bytes:
        """Pop *size* bytes from the stream, blocking until available.

        Args:
            size: Number of bytes to consume.
            timeout: Maximum seconds to wait for data per chunk.

        Returns:
            A ``bytes`` object of *size* bytes.
        """
        result = bytearray()
        remaining = size

        while remaining > 0:
            avail = self._consumer_wait_avail(timeout)
            chunk = min(remaining, avail)

            head = min(chunk, self._buffer_size - self._read_idx)
            if head > 0:
                result.extend(
                    read_from_device(
                        self._location,
                        self._address + self._BUFFER_OFFSET + self._read_idx,
                        num_bytes=head,
                    )
                )

            tail = chunk - head
            if tail > 0:
                result.extend(
                    read_from_device(
                        self._location,
                        self._address + self._BUFFER_OFFSET,
                        num_bytes=tail,
                    )
                )

            self._store_read_idx((self._read_idx + chunk) % self._buffer_size)

            remaining -= chunk

        return bytes(result)
