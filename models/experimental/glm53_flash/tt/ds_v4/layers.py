"""The plain-ttnn projection of the DeepSeek-V4-Flash layer library.

Only :class:`Linear` is kept: the decode projections (``LinearDecode``, ``LinearDecodeLargeK``,
``BatchedLinearDecode``) run through ``matmul_decode`` and the DRAM prefetcher, which Wormhole
does not support.
"""

from typing import Optional

import ttnn

from .common import DeepSeekV4Module, _HIFI4
from .weight_cache import _load_weight, _materialize


class Linear(DeepSeekV4Module):
    """``nn.Linear`` (bias-free) as ``x @ Wᵀ`` for ttnn.

    ttnn ``linear`` computes ``a @ b`` with ``b`` shaped ``[in, out]``, so the torch
    ``[out=N, in=K]`` weight is stored transposed as ``[K, N]`` (TILE, DRAM-interleaved).
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        mesh_mapper=None,
    ):
        """Store the transposed weight ``[K, N]`` as TILE DRAM, ready for ``ttnn.linear``.

        ``weight`` is the torch ``[N, K]`` (out, in) tensor or a thunk returning it; ``dtype``
        is used for the host materialization and the device copy alike, and ``mesh_mapper``
        shards or replicates the device tensor across the mesh.
        """
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
            mesh_mapper=mesh_mapper,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x [..., K]`` (TILE) -> ``[..., N]`` (TILE), HiFi4 accumulation."""
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)
