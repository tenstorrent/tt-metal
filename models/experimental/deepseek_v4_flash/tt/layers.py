from typing import Callable, Optional

import ttnn

from .common import DeepSeekV4Module, _HIFI4, rectangular_core_range_set, width_sharded_l1_config
from .weight_cache import _load_weight, _materialize
import torch


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
        K: int = -1,
        N: int = -1,
        partial_width_sharded: bool = False,
        num_inputA_cores: int = 32,
        k_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
    ):
        self.partial_width_sharded = partial_width_sharded
        self.num_inputA_cores = num_inputA_cores
        self.dtype = dtype
        self.device = device
        self.l1_weights = None

        assert K != -1 and N != -1, "K and N must be set"
        self.K = K
        self.N = N
        if partial_width_sharded:
            self.n_blocks = n_blocks
            self.k_blocks = k_blocks

        if partial_width_sharded:
            if k_blocks is None or n_blocks is None:
                raise ValueError("partial_width_sharded=True requires k_blocks and n_blocks")
            kc, nc = self.K // k_blocks, self.N // n_blocks
            num_inputB_cores = k_blocks * n_blocks
            shard_shape = (kc, nc)
        else:
            if n_blocks is None:
                num_inputB_cores = self.N // 64
            else:
                num_inputB_cores = n_blocks
            shard_shape = (self.K, self.N // num_inputB_cores)

        b_core_range_set = rectangular_core_range_set(num_inputB_cores, self.device)
        self.weights_memory_config = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
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
        if partial_width_sharded:
            # Fold the K-blocks into the width so a width-sharded [Kc, Nc] block lands on
            # core c = kb * n_blocks + nb (row-major), matching the op's expected geometry.
            w = w.reshape(k_blocks, kc, self.N).permute(1, 0, 2).reshape(kc, self.N * k_blocks)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def fetch_weights(self):
        self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        # self.weight.deallocate()

    def get_input_memory_config(self, m: int, k: int) -> ttnn.MemoryConfig:
        a_core_range_set = rectangular_core_range_set(self.num_inputA_cores, self.device)
        a_memory_config = ttnn.create_sharded_memory_config(
            (32, k // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return a_memory_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.l1_weights is None or not self.l1_weights.is_allocated():
            self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        m = x.shape[-2]
        m_padded = ((m + 31) // 32) * 32
        if self.partial_width_sharded:
            # The partial layout reduces the K-partials onto n_blocks output cores, so shard the
            # output WIDTH_SHARDED across a rectangular grid of n_blocks cores (shard
            # [padded_m, N / n_blocks]).
            output_core_range_set = rectangular_core_range_set(self.n_blocks, self.device)
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
            output_memory_config = width_sharded_l1_config(m_padded, self.N, self.device)
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(x.shape[-2], x.shape[-1]))
        result = ttnn.experimental.matmul_decode(
            x, self.l1_weights, partial_width_sharded=self.partial_width_sharded, output_mem_config=output_memory_config
        )
        self.l1_weights.deallocate()
        self.l1_weights = None
        return result


class BatchedLinearDecode(DeepSeekV4Module):
    """Batched (block-diagonal) ``x[b] @ W[b]`` via ``ttnn.experimental.matmul_decode``.

    A rank-4 activation ``[d0, d1, M, K]`` (batch = ``d0*d1``) is matmul'd against a
    per-batch weight ``[batch, K, N]`` that is folded along BOTH batch and N into a
    width-sharded ``[1, 1, Bc*K, b_blocks*N]`` tensor (``Bc = batch / b_blocks``,
    ``Nc = N / n_blocks``) laid across a ``b_blocks x n_blocks`` core grid -- the layout
    the batched matmul_decode factory expects. The op infers ``b_blocks`` / ``n_blocks``
    from the operand shapes and emits a DRAM-interleaved ``[d0, d1, M, N]`` result.

    As in :class:`LinearDecode`, the (static) weight is prepared once here and only the
    activation is resharded per call. ``preprocess`` (optional) is applied to the raw
    torch weight on a cache MISS, *before* the batch/N fold, to normalize it to
    ``[batch, K, N]`` (e.g. the o_a reshape from ``[batch*N, K]``); it is skipped on a
    cache hit (the folded, tilized weight is already on disk).
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        batch: int,
        K: int,
        N: int,
        b_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
        num_inputA_cores: int = 32,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.K = K
        self.N = N
        self.num_inputA_cores = num_inputA_cores

        # One batch per core row (Bc = 1) by default; widen N across as many cores as the grid
        # allows while keeping each N-shard tile-aligned.
        self.b_blocks = b_blocks if b_blocks is not None else batch
        if n_blocks is None:
            device_grid = device.compute_with_storage_grid_size()
            max_cores = device_grid.x * device_grid.y
            n_blocks = max(1, max_cores // self.b_blocks)
            while n_blocks > 1 and (N % n_blocks != 0 or (N // n_blocks) % ttnn.TILE_SIZE != 0):
                n_blocks -= 1
        self.n_blocks = n_blocks

        assert batch % self.b_blocks == 0, "b_blocks must divide batch"
        assert N % self.n_blocks == 0, "n_blocks must divide N"
        self.bc = batch // self.b_blocks
        self.nc = N // self.n_blocks

        b_core_range_set = rectangular_core_range_set(self.b_blocks * self.n_blocks, device)
        self.weights_memory_config = ttnn.create_sharded_memory_config(
            (self.bc * K, self.nc),
            core_grid=b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        w = _materialize(weight, cache_file_name, dtype)
        if w is not None:
            if preprocess is not None:
                w = preprocess(w)
            # w: [batch, K, N] -> [b_blocks, Bc, K, N] -> [Bc, K, b_blocks, N] -> [1, 1, Bc*K, b_blocks*N].
            w = (
                w.reshape(self.b_blocks, self.bc, K, N)
                .permute(1, 2, 0, 3)
                .reshape(1, 1, self.bc * K, self.b_blocks * N)
                .contiguous()
            )
        self.weight = _load_weight(w, device, cache_file_name=cache_file_name, dtype=dtype)

    def deallocate(self):
        pass

    def get_input_memory_config(self, m: int) -> ttnn.MemoryConfig:
        # Activation A is width(K)-sharded: shard [batch * m_padded, K / num_inputA_cores].
        m_padded = ((m + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        a_core_range_set = rectangular_core_range_set(self.num_inputA_cores, self.device)
        return ttnn.create_sharded_memory_config(
            (self.batch * m_padded, self.K // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: rank-4 [d0, d1, M, K] with d0*d1 == batch. Reshard to the width(K)-sharded L1 layout,
        # then run the batched matmul_decode (b_blocks / n_blocks are inferred from the shapes).
        m = x.shape[-2]
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(m))
        l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        y = ttnn.experimental.matmul_decode(x, l1_weights)  # DRAM-interleaved [d0, d1, M, N]
        l1_weights.deallocate()
        return y


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(
        self, weight, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None, sharded: bool = False
    ):
        w = _materialize(weight, cache_file_name, ttnn.bfloat16)
        self.weight = _load_weight(
            w.reshape(1, 1, 1, -1) if w is not None else None, device, cache_file_name=cache_file_name
        )
        self.eps = eps
        self.sharded = sharded

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.sharded:
            b, s, t, d = x.shape
            x_mem_config = width_sharded_l1_config(b * s * t, d, x.device())
            x = ttnn.to_memory_config(x, x_mem_config)
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


def _rms_norm_unweighted(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim (matches ``DeepseekV4UnweightedRMSNorm``)."""
    # if not x.is_sharded():
    #     b, s, t, d = x.shape
    #     x_mem_config = width_sharded_l1_config(b * s * t, d, x.device())
    #     x = ttnn.to_memory_config(x, x_mem_config)
    return ttnn.rms_norm(x, epsilon=eps)
