from typing import TYPE_CHECKING, Protocol

import ttnn

if TYPE_CHECKING:
    from models.demos.deepseek_v3.utils.cache.cache import CacheKey


class CacheStorage(Protocol):
    def set(self, key: "CacheKey", tensor: ttnn.Tensor) -> None:
        ...

    def has(self, key: "CacheKey") -> bool:
        ...

    def get(self, key: "CacheKey") -> ttnn.Tensor:
        ...
