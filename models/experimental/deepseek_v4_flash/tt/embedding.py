from typing import Optional

import ttnn

from .common import DeepSeekV4Module
from .layers import to_ttnn_device
from .weight_cache import WeightCache, _as_cache
from .weight_loader import DeepseekV4WeightLoader


class DeepSeekV4Embedding(DeepSeekV4Module):
    def __init__(
        self,
        weight_loader: DeepseekV4WeightLoader,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
    ):
        self.weight_loader = weight_loader
        self.device = device
        cache = _as_cache(cache)
        # ``ttnn.embedding`` expects a row-major weight table.
        cfn = cache.file("embed_tokens")
        hit = cache.hit("embed_tokens", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if cache.require_cache and not hit:
            raise RuntimeError("weight cache miss for 'embed_tokens' with require_cache=True")
        embed = None if hit else weight_loader.get_tensor("embed_tokens.weight")
        self.embedding_weight = to_ttnn_device(embed, device, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=cfn)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(input_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT)
