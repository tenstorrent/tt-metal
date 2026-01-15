import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

from core.ops.vector_index_ops import IndexAddOp
from backends.Tenstorrent.tenstorrent_op_base import TenstorrentOpMixin

OP_MAPPING = {}


class TenstorrentIndexAddOp(TenstorrentOpMixin, IndexAddOp):
    """
    Tenstorrent-specific indexadd operation.

    Uses CPU execution to avoid device lock contention in multi-process benchmark scenarios.
    The parent IndexAddOp class provides correct CPU implementation.
    """

    pass


OP_MAPPING["torch"] = TenstorrentIndexAddOp
