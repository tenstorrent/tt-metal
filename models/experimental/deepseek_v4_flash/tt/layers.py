from typing import Optional

import ttnn

from .common import DeepSeekV4Module, _HIFI4
from .weight_cache import _load_weight, _materialize
import torch


def _num_cores_to_rectangle_core_range_set(num_cores: int, grid) -> ttnn.CoreRangeSet:
    """A single rectangular ``CoreRangeSet`` of exactly ``num_cores`` cores.

    Finds the widest ``x`` that divides ``num_cores`` and fits ``grid.x``, giving a
    ``(x, num_cores // x)`` rectangle. Raises if no such rectangle fits the device grid.
    """
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def to_ttnn_device(
    tensor: torch.Tensor,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    cache_file_name: Optional[str] = None,
) -> ttnn.Tensor:
    return _load_weight(tensor, device, cache_file_name=cache_file_name, layout=layout)


class Linear(DeepSeekV4Module):
    """``nn.Linear`` (bias-free) as ``x @ Wᵀ`` for ttnn.

    ttnn ``linear`` computes ``a @ b`` with ``b`` shaped ``[in, out]``, so we
    store the torch ``[out, in]`` weight transposed.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class LinearDecode(DeepSeekV4Module):
    """Bias-free ``x @ Wᵀ`` backed by ``ttnn.experimental.matmul_decode``.

    Both operands stay L1 width-sharded and resident on the core grid, which is
    the layout the decode-optimized matmul kernel expects. The (static) weight is
    prepared and loaded once in the constructor; ``forward`` only reshards the
    incoming activation into the matching width-sharded L1 config before the op.

    Two weight layouts are supported, selected by ``partial_width_sharded``:

    - ``False`` (fully width-sharded): the torch ``[out, in]`` weight is stored as
      ``[K, N]`` and width(N)-sharded across ``N // 64`` cores (shard ``[K, N/cores]``),
      matching ``test_matmul_decode``. The full activation is gathered onto every core.
    - ``True`` (partial width-sharded): ``[K, N]`` is reshaped/permuted so a 2D
      ``(K_blocks x N_blocks)`` grid of ``[Kc, Nc]`` blocks maps across
      ``K_blocks * N_blocks`` cores (``Kc = K/K_blocks``, ``Nc = N/N_blocks``), and the
      K-partials are reduced across cores. Requires ``k_blocks`` and ``n_blocks``.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        partial_width_sharded: bool = False,
        num_inputA_cores: int = 32,
        k_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
        N: Optional[int] = None,
    ):
        self.partial_width_sharded = partial_width_sharded
        self.num_inputA_cores = num_inputA_cores
        self.dtype = dtype
        self._grid = device.compute_with_storage_grid_size()

        self.N = N
        if partial_width_sharded:
            self.n_blocks = n_blocks
            self.k_blocks = k_blocks

        # The decode op wants the weight as [K, N]; torch nn.Linear stores [out=N, in=K].
        w = _materialize(weight, cache_file_name, dtype)
        if w is None:
            # Cache hit: the tilized, width-sharded weight is already on disk and its
            # serialized spec carries the real (width-sharded) layout, so none of the
            # K/N-derived shard-config or torch reshape work below is needed. ``as_tensor``
            # requires a ``memory_config`` when a device is given but ignores it on a
            # cache-hit load, so pass a throwaway config just to satisfy that guard.
            self.weight = ttnn.as_tensor(
                None,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
            )
            return

        w = w.t().contiguous()
        k, n = int(w.shape[0]), int(w.shape[1])
        self.K, self.N = k, n

        if partial_width_sharded:
            if k_blocks is None or n_blocks is None:
                raise ValueError("partial_width_sharded=True requires k_blocks and n_blocks")
            kc, nc = k // k_blocks, n // n_blocks
            num_inputB_cores = k_blocks * n_blocks
            # Fold the K-blocks into the width so a width-sharded [Kc, Nc] block lands on
            # core c = kb * n_blocks + nb (row-major), matching the op's expected geometry.
            w = w.reshape(k_blocks, kc, n).permute(1, 0, 2).reshape(kc, n * k_blocks)
            shard_shape = (kc, nc)
        else:
            num_inputB_cores = n // 64
            shard_shape = (k, n // num_inputB_cores)

        b_core_range_set = _num_cores_to_rectangle_core_range_set(num_inputB_cores, self._grid)
        b_memory_config = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        print("Weight dtype: ", dtype)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=b_memory_config,
            cache_file_name=cache_file_name,
        )

    def get_input_memory_config(self, m: int, k: int) -> ttnn.MemoryConfig:
        a_core_range_set = _num_cores_to_rectangle_core_range_set(self.num_inputA_cores, self._grid)
        a_memory_config = ttnn.create_sharded_memory_config(
            (32, k // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return a_memory_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.partial_width_sharded:
            # The partial layout reduces the K-partials onto n_blocks output cores, so shard the
            # output WIDTH_SHARDED across a rectangular grid of n_blocks cores (shard
            # [padded_m, N / n_blocks]).
            m = x.shape[-2]
            m_padded = ((m + 31) // 32) * 32
            output_core_range_set = _num_cores_to_rectangle_core_range_set(self.n_blocks, self._grid)
            output_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    output_core_range_set,
                    [m_padded, self.N // self.n_blocks],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        else:
            output_memory_config = None
        return ttnn.experimental.matmul_decode(
            x, self.weight, partial_width_sharded=self.partial_width_sharded, output_mem_config=output_memory_config
        )


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(self, weight, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None):
        w = _materialize(weight, cache_file_name, ttnn.bfloat16)
        self.weight = _load_weight(
            w.reshape(1, 1, 1, -1) if w is not None else None, device, cache_file_name=cache_file_name
        )
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


def _rms_norm_unweighted(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim (matches ``DeepseekV4UnweightedRMSNorm``)."""
    return ttnn.rms_norm(x, epsilon=eps)


# ---------------------------------------------------------------------------- #
# DRAM-sharded weight prep for the fused-QKV decode op
# (``ttnn.experimental.deepseek_fused_qkv``)
# ---------------------------------------------------------------------------- #
def dram_num_banks(device: ttnn.MeshDevice) -> int:
    """Number of DRAM banks on ``device`` (the WIDTH-shard fan-out for weights)."""
    g = device.dram_grid_size()
    return g.x * g.y


def dram_width_sharded_weight(
    weight: torch.Tensor,
    device: ttnn.MeshDevice,
    *,
    num_banks: Optional[int] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    cache_file_name: Optional[str] = None,
    already_kn: bool = False,
) -> ttnn.Tensor:
    """Build a DRAM ``WIDTH_SHARDED`` weight for the fused-QKV op.

    ``weight`` is a torch ``nn.Linear`` weight ``[out, in]`` (transposed to the
    ``[K=in, N=out]`` layout ttnn matmul wants), unless ``already_kn`` is set (then
    it is taken as ``[K, N]`` directly). ``N`` is padded up to ``TILE * num_banks``
    and width-sharded ROW_MAJOR so bank ``b`` holds columns ``[b*Nc, (b+1)*Nc)``
    (``Nc = padded_N / num_banks``). Each per-bank shard is the full ``[K, Nc]``.

    ``weight`` may be ``None`` only on a verified cache hit (the caller passed a
    ``cache_file_name`` whose tile already exists): the serialized spec carries the
    width-sharded layout, so the K/N shard-config work below is skipped and the
    tensor is loaded straight from disk (``as_tensor`` needs a ``memory_config``
    when a device is given but ignores it on a cache-hit load).
    """
    if weight is None:
        return ttnn.as_tensor(
            None,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    grid = device.dram_grid_size()
    if num_banks is None:
        num_banks = grid.x * grid.y

    w = weight if already_kn else weight.t().contiguous()
    k, n = int(w.shape[0]), int(w.shape[1])
    tile = ttnn.TILE_SIZE
    step = tile * num_banks
    padded_n = ((n + step - 1) // step) * step
    if padded_n != n:
        w = torch.nn.functional.pad(w, (0, padded_n - n))
    nc = padded_n // num_banks

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    shard_spec = ttnn.ShardSpec(dram_grid, [k, nc], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        cache_file_name=cache_file_name,
    )
