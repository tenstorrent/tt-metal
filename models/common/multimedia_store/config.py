# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration of the prefill runner's media embedding store, from PREFILL_MEDIA_STORE_* and
MEDIA_STORE_KEY_TRANSFORM. Kept apart from the reader so a runner can check whether the store is on without importing
Mooncake."""

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

ENV_PREFIX = "PREFILL_MEDIA_STORE_"
_DEFAULT_SEGMENT_BYTES = "1G"

KEY_TRANSFORM_ENV = "MEDIA_STORE_KEY_TRANSFORM"
"""How a media key becomes its store key. The encoder service, which writes each value, and the prefill runner, which
reads it, each transform keys this way themselves, so both must see the same value:

- unset, empty or ``hex`` (the default): the store key is the media key's lowercase hex;
- ``none``: the store key is the media key itself."""
KEY_TRANSFORMS = ("hex", "none")


def key_transform_from_env(environ: Mapping[str, str] = os.environ) -> str:
    """MEDIA_STORE_KEY_TRANSFORM: 'hex' (the default, when unset) or 'none'."""
    transform = environ.get(KEY_TRANSFORM_ENV, "").strip() or "hex"
    if transform not in KEY_TRANSFORMS:
        raise ValueError(f"{KEY_TRANSFORM_ENV}={transform!r}: want 'hex' (the default) or 'none'")
    return transform


def parse_bytes(text: str) -> int:
    """'1G', '1GiB', '512M', '1048576' -> bytes. Suffixes are binary (G = GiB)."""
    m = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)(?:i?B)?\s*", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a byte size: {text!r} (examples: 1G, 512M, 1048576)")
    return int(m.group(1)) << {"": 0, "K": 10, "M": 20, "G": 30, "T": 40}[m.group(2).upper()]


@dataclass(frozen=True)
class MediaEmbeddingStoreConfig:
    master: str
    """Mooncake master address, host:port."""
    segment: str
    """This host's segment name, 'host:port', which writers pass as preferred_segment. A bare 'host' gets a port
    Mooncake picks, so the full name is known only once the store starts (MooncakeEmbeddingReader.segment)."""
    segment_bytes: int
    """Host memory this host gives the store. More than 0 is push mode: writers put values here, read in place. 0 is
    pull mode: this host holds nothing and reads every value from the host that does, such as the encoder's."""
    metadata: str
    """Mooncake metadata server."""
    protocol: str = "rdma"
    device: str = ""
    """RDMA device(s), e.g. 'rocep201s0f0'; empty lets Mooncake pick."""
    key_transform: str = "hex"
    """How media keys become store keys (KEY_TRANSFORM_ENV), from MEDIA_STORE_KEY_TRANSFORM."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["MediaEmbeddingStoreConfig"]:
        """The store's config from PREFILL_MEDIA_STORE_*, or None when PREFILL_MEDIA_STORE_MASTER is unset.

        PREFILL_MEDIA_STORE_MASTER    master address, host:port (turns the store on)
        PREFILL_MEDIA_STORE_SEGMENT   this host's address, host:port or host (required)
        PREFILL_MEDIA_STORE_BYTES     segment size (default 1G); 0 for pull mode, with no store on this host
        PREFILL_MEDIA_STORE_METADATA  metadata server, e.g. http://<master host>:<port>/metadata (required)
        PREFILL_MEDIA_STORE_PROTOCOL  rdma (default) or tcp
        PREFILL_MEDIA_STORE_DEVICE    RDMA device(s) (default: Mooncake's choice)
        MEDIA_STORE_KEY_TRANSFORM     'hex' (default) or 'none'; shared with the encoder
        """

        def get(name: str, default: Optional[str] = None) -> Optional[str]:
            value = environ.get(ENV_PREFIX + name, "").strip()
            return value or default

        master = get("MASTER")
        if master is None:
            return None
        if ":" not in master:
            raise ValueError(f"{ENV_PREFIX}MASTER={master!r}: want host:port")
        missing = [ENV_PREFIX + n for n in ("SEGMENT", "METADATA") if get(n) is None]
        if missing:
            raise ValueError(f"{ENV_PREFIX}MASTER is set, so {' and '.join(missing)} must be set too")
        return cls(
            master=master,
            segment=get("SEGMENT"),
            segment_bytes=parse_bytes(get("BYTES", _DEFAULT_SEGMENT_BYTES)),
            metadata=get("METADATA"),
            protocol=get("PROTOCOL", "rdma"),
            device=get("DEVICE", ""),
            key_transform=key_transform_from_env(environ),
        )
