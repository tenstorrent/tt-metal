"""DeepSeek V4 Flash ttnn model — re-exports from submodules.

The implementations live in the sibling modules (``model.py``, ``attention.py``, ``moe.py``,
``layers.py``, ...); this module is the stable import surface over them, mirroring ``__all__``.
"""

from .attention import (
    DeepSeekV4Attention,
    _StaticLayerCache,
    build_static_layer_cache,
    int32_pos_tensor,
    make_rope_table,
)
from .attention_csa import DeepSeekV4CSACompressor
from .attention_hca import DeepSeekV4HCACompressor
from .common import (
    DeepSeekV4Module,
    _MASK_NEG,
    _region,
    _trace_capture_guard,
)
from .decoder_layer import DeepSeekV4DecoderLayer
from .embedding import DeepSeekV4Embedding
from .hyperconnection import DeepSeekV4HyperConnection, DeepSeekV4HyperHead
from .layers import DeepSeekV4RMSNorm, Linear, to_ttnn_device
from .model import DeepSeekV4Model
from .moe import (
    DeepSeekV4HashRouter,
    DeepSeekV4MLP,
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
    DeepSeekV4TopKRouter,
)
from .weight_cache import WeightCache

__all__ = [
    "DeepSeekV4Attention",
    "DeepSeekV4CSACompressor",
    "DeepSeekV4DecoderLayer",
    "DeepSeekV4Embedding",
    "DeepSeekV4HCACompressor",
    "DeepSeekV4HashRouter",
    "DeepSeekV4HyperConnection",
    "DeepSeekV4HyperHead",
    "DeepSeekV4MLP",
    "DeepSeekV4Model",
    "DeepSeekV4Module",
    "DeepSeekV4PreloadedExperts",
    "DeepSeekV4RMSNorm",
    "DeepSeekV4SparseMoeBlock",
    "DeepSeekV4TopKRouter",
    "Linear",
    "WeightCache",
    "_StaticLayerCache",
    "_MASK_NEG",
    "_region",
    "_trace_capture_guard",
    "build_static_layer_cache",
    "int32_pos_tensor",
    "make_rope_table",
    "to_ttnn_device",
]
