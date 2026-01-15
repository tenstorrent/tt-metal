import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

from core.ops.llm_ops import ScaleDynamicQuantOp
from backends.Tenstorrent.tenstorrent_op_base import TenstorrentOpMixin

OP_MAPPING = {}


class TenstorrentScaleDynamicQuantOp(TenstorrentOpMixin, ScaleDynamicQuantOp):
    """
    Tenstorrent-specific scaledynamicquant operation.

    Uses CPU execution to avoid device lock contention in multi-process benchmark scenarios.
    The parent ScaleDynamicQuantOp class provides correct CPU implementation.
    """

    pass


OP_MAPPING["torch"] = TenstorrentScaleDynamicQuantOp
