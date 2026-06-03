from models.experimental.opt_transfer.schema import MemoryPlacement

_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "bf8": 1,
    "bfp8": 1,
    "bfloat8_b": 1,
}
_OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def tensor_bytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * _DTYPE_BYTES[dtype]


class L1Budget:
    """Aggregate L1 capacity model. A tensor 'fits' if its total bytes are within a safety
    fraction of num_cores * per_core_bytes (interleaved across the grid)."""

    def __init__(self, per_core_bytes: int, num_cores: int, safety: float = 0.5):
        self.aggregate = int(per_core_bytes * num_cores * safety)

    def fits(self, total_bytes: int) -> bool:
        return total_bytes <= self.aggregate


def eval_condition(condition, dims: dict) -> bool:
    if condition is None:
        return True
    if condition.get("op") not in _OPS or condition.get("var") not in dims:
        return False
    return _OPS[condition["op"]](dims[condition["var"]], condition["value"])


def decide_placement(observations, size_bytes, dims, l1_budget, default_buffer="DRAM") -> MemoryPlacement:
    """Pick L1 vs DRAM for a tensor. Order: (1) budget backstop forces DRAM if it can't fit L1;
    (2) else honor a donor observation that prefers L1 whose size-condition holds; (3) else default."""
    if not l1_budget.fits(size_bytes):
        return MemoryPlacement(buffer="DRAM", layout="interleaved")
    for obs in observations:
        mc = obs.memory_config or {}
        if mc.get("buffer") == "L1" and eval_condition(obs.condition, dims):
            # interleaved only in this plan; sharded shard-spec instantiation is a follow-on
            return MemoryPlacement(buffer="L1", layout="interleaved")
    return MemoryPlacement(buffer=default_buffer, layout="interleaved")
