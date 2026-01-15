import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

from core.ops.llm_ops import StoreKVCacheOp
from backends.Tenstorrent.tenstorrent_op_base import TenstorrentOpMixin

OP_MAPPING = {}


class TenstorrentStoreKVCacheOp(TenstorrentOpMixin, StoreKVCacheOp):
    """
    Tenstorrent-specific storekvcache operation.

    Uses CPU execution to avoid device lock contention in multi-process benchmark scenarios.
    The parent StoreKVCacheOp class provides correct CPU implementation.
    """

    pass


OP_MAPPING["torch"] = TenstorrentStoreKVCacheOp
