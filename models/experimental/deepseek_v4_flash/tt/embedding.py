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


class DeepSeekV4Flash(DeepSeekV4Module):
    """V4-Flash model wired up to the safetensors weight loader.

    Only the embedding table is materialised; the rest of the model is a
    placeholder.
    """

    def __init__(
        self,
        config: dict,
        weights_dir: str,
        device: ttnn.MeshDevice,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir
        self.device = device
        self.weight_loader = DeepseekV4WeightLoader(weights_dir)
        # Converted-weight cache (opt-in): pass ``cache_dir`` to dump/reuse the
        # tilized ttnn tensors across runs (skipping re-conversion / dequant).
        # ``None`` keeps caching off, so weights are converted every time.
        self.cache = WeightCache(cache_dir)
        self.embed_tokens = DeepSeekV4Embedding(self.weight_loader, device, cache=self.cache)

    def forward(self, input_ids: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        input_embeddings = self.embed_tokens(input_ids)
        return input_embeddings
