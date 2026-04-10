import functools, operator
from typing import Iterable
import ttml


def prod(x: Iterable[int]) -> int:
    return functools.reduce(operator.mul, x, 1)


class Mesh:
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    _axis_map: dict[str, int]

    def __init__(self, shape: tuple[int, ...], axis_names: tuple[str, ...]):
        if len(shape) != len(axis_names):
            raise ValueError("each axis in a mesh must have an assigned name")

        self.shape = shape
        self.axis_names = axis_names
        self._axis_map = {name: i for i, name in enumerate(axis_names)}

    def num_devices(self) -> int:
        return prod(self.shape)

    def has_axis(self, name: str) -> bool:
        return name in self._axis_map

    def axis_size(self, name: str) -> int:
        return self.shape[idx] if (idx := self._axis_map.get(name)) is not None else 0


_current_mesh: Mesh | None = None


def open_device_mesh(mesh: tuple[int, ...] | Mesh, physical_ids: tuple[int, ...] | None = None):
    if not isinstance(mesh, Mesh):
        mesh = Mesh(mesh, tuple(f"_{i}" for i in range(len(mesh))))
    if physical_ids is None:
        physical_ids = ()

    if mesh.num_devices() > 1:
        ttml.core.distributed.enable_fabric(mesh.num_devices())

    ttnn_shape = list(mesh.shape) + [1] * max(0, 2 - len(mesh.shape))
    ttml.autograd.AutoContext.get_instance().open_device(ttnn_shape, list(physical_ids))

    global _current_mesh
    _current_mesh = mesh


def current_mesh() -> Mesh | None:
    global _current_mesh
    return _current_mesh
