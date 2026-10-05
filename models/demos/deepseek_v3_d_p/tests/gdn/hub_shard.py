# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HTTP range reads of single tensors (or rows of one) from a safetensors shard on the Hugging Face hub.

Lets GDN preparation steps fetch one layer's weights or a few embedding rows without downloading whole shards. Used
by the GDN baseline harness (``qwen36/tests/gdn_baseline/prepare.py``) and the GDN layer-checkpoint fetch
(``tests/gdn/prepare.py``).
"""

from __future__ import annotations

import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class HubSafetensorsShard:
    """Range reads of one safetensors shard on the Hugging Face hub."""

    DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}

    def __init__(self, repo: str, revision: str, filename: str):
        from huggingface_hub import get_session, hf_hub_url

        self.session = get_session()
        self.filename = filename
        response = self.session.get(
            hf_hub_url(repo, filename, revision=revision), headers={"Range": "bytes=0-7"}, follow_redirects=True
        )
        response.raise_for_status()
        self.url = str(response.url)
        header_size = struct.unpack("<Q", response.content)[0]
        self.header = json.loads(self.read(8, header_size))
        self.data_start = 8 + header_size
        self.bytes_read = 8 + header_size

    def read(self, offset: int, size: int) -> bytes:
        for attempt in range(5):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
                response.raise_for_status()
                if len(response.content) != size:
                    raise IOError(f"short read {len(response.content)} != {size}")
                return response.content
            except Exception:  # noqa: BLE001 - transient CDN errors
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    def tensor(self, key: str) -> torch.Tensor:
        meta = self.header[key]
        if meta["dtype"] not in self.DTYPES:
            raise ValueError(f"{key}: unsupported checkpoint dtype {meta['dtype']}")
        begin, end = meta["data_offsets"]
        raw = self.read(self.data_start + begin, end - begin)
        self.bytes_read += end - begin
        return torch.frombuffer(bytearray(raw), dtype=self.DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()

    def rows(self, key: str, ids: list[int]) -> torch.Tensor:
        meta = self.header[key]
        dtype = self.DTYPES[meta["dtype"]]
        width = meta["shape"][1]
        row_bytes = width * torch.tensor([], dtype=dtype).element_size()
        base = self.data_start + meta["data_offsets"][0]
        spans, start, last = [], ids[0], ids[0]
        for i in ids[1:]:
            if i - last > 8:
                spans.append((start, last))
                start = i
            last = i
        spans.append((start, last))

        def fetch(span):
            lo, hi = span
            raw = self.read(base + lo * row_bytes, (hi - lo + 1) * row_bytes)
            return lo, torch.frombuffer(bytearray(raw), dtype=dtype).reshape(hi - lo + 1, width)

        out = {}
        with ThreadPoolExecutor(16) as pool:
            for lo, block in pool.map(fetch, spans):
                for r in range(block.shape[0]):
                    out[lo + r] = block[r].clone()
                self.bytes_read += block.numel() * block.element_size()
        log(f"  {key}: {len(ids)} rows in {len(spans)} range reads")
        return torch.stack([out[i] for i in ids])
