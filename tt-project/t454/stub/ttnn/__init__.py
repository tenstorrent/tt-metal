# CPU-only stand-in for t454's host emulation: names only, no device.
import enum


class _Any:
    def __getattr__(self, n):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


class Layout(enum.Enum):
    ROW_MAJOR = 0
    TILE = 1


ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE
bfloat16 = "bf16"
bfloat8_b = "bf8"
float32 = "f32"
uint32 = "u32"
int32 = "i32"
uint16 = "u16"


class Tensor:
    pass


class MeshDevice:
    pass


def __getattr__(n):
    return _Any()
