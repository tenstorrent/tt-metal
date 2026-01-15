import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

from core.ops.vector_reduction_ops import TopkOp
from backends.Tenstorrent.tenstorrent_op_base import TenstorrentOpMixin

OP_MAPPING = {}


class TenstorrentTopkOp(TenstorrentOpMixin, TopkOp):
    """
    Tenstorrent-specific topk operation.

    Uses CPU execution to avoid device lock contention in multi-process benchmark scenarios.
    The parent TopkOp class provides correct CPU implementation.
    """

    pass


OP_MAPPING["torch"] = TenstorrentTopkOp
