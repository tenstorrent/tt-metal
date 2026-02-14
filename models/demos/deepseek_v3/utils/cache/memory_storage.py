import ttnn
from models.demos.deepseek_v3.utils.cache.cache import CacheKey


class InMemoryCacheStorage:
    """
    A cache backed by host memory. Does not persist across runs.
    We could make this more advanced by implementing LRU caching. Injected into TensorCache to facilitate unit testing.
    """

    def __init__(self):
        self.cache = {}

    def set(self, key: CacheKey, tensor: ttnn.Tensor):
        self.cache[key.fingerprint] = tensor

    def has(self, key: CacheKey) -> bool:
        return key.fingerprint in self.cache

    def get(self, key: CacheKey) -> ttnn.Tensor:
        return self.cache[key.fingerprint]
