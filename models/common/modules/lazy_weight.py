import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import ttnn


@dataclass
class LazyWeight:
    """
    Lazy loading of weights with caching.
    """

    device: Any
    create_func: Callable[[], Any]
    file_name: Optional[str] = None
    _value: Optional[Any] = None

    def get_weight(self) -> Any:
        if self._value is not None:
            return self._value

        if self.file_name and os.path.exists(self.file_name):
            self._value = ttnn.load_tensor(self.file_name, device=self.device)
            return self._value

        tensor = self.create_func()

        if self.file_name:
            ttnn.dump_tensor(self.file_name, tensor)

        self._value = tensor
        return self._value
