# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-bounded causal QSA scorer used by the mesh decoder.

``causal_indexer_scores(owner, query, compressed_keys, positions)`` substitutes
for the accurate-SFPU fused scorer plus its runtime causal bound. Query and keys
are BF16 DRAM tiled tensors [1,4,Q,128] and [1,1,K,128]. Absolute positions are
FP32 DRAM tiled [1,1,Q,1]; each row has (position+1)//4 complete compressed keys.
Q is 1..32 and K is tile aligned, 32..65536. All live-loop bounds are device
values: advancing persistent position inputs does not change the program.

The native op emits FP32 [1,4,Q,K] dots, or packed [1,1,4,K] for Q1;
every invalid/padded cell is zeroed each invocation. Valid dots retain separate exact BF16 products, four sequential
FP32 tile folds, then the same SFPU row reduction as accurate ttnn.sum. The
public function retains the original TTNN ReLU/head-sum/scale operations.
There is no TF32 dot, host read, mutable helper cache, or persistent allocation
inside a replay. Prepare this signature and the terminal ops before capture.

Validation contract: compare valid per-head dots and
final scores bitwise with fused_indexer_scores(..., accurate_sum=True), then
compare selected IDs for positions 0,2,3,127,128,2047,2048 and near capacity,
including Q=1,3,17,32 and changed positions on the same trace. Run allocation
tracking with program-cache allocations included, watcher separately, and time
warmed replay separately. The implementation still writes complete padded
outputs, while score computation visits only the live compressed prefix.
"""

from pathlib import Path

_KERNEL_ROOT = Path(__file__).parent / "kernels"


def _causal_indexer_dots(query, compressed_keys, positions, *, output=None, num_cores=None, packed_q1=False):
    """Return causal per-head FP32 dots, leaving final head arithmetic to TTNN."""
    import ttnn as t

    if not all(isinstance(value, t.Tensor) for value in (query, compressed_keys, positions)):
        raise ValueError("Expected device tensors")
    count, capacity = query.shape[-2], compressed_keys.shape[-2]
    if tuple(query.shape) != (1, 4, count, 128) or not 1 <= count <= 32:
        raise ValueError("Expected BF16 query [1,4,1..32,128]")
    if tuple(compressed_keys.shape) != (1, 1, capacity, 128) or not 32 <= capacity <= 65536 or capacity % 32:
        raise ValueError("Expected BF16 compressed keys [1,1,K,128], tile-aligned K in 32..65536")
    if tuple(positions.shape) != (1, 1, count, 1):
        raise ValueError("Expected absolute FP32 positions [1,1,Q,1]")
    if packed_q1 and count != 1:
        raise ValueError("Packed-head dots require Q=1")
    output_shape = (1, 1, 4, capacity) if packed_q1 else (1, 4, count, capacity)
    output_padded = (1, 1, 32, capacity) if packed_q1 else (1, 4, 32, capacity)
    device = query.device()

    def validate(value, padded_shape, dtype):
        tile = value.get_tile()
        if (
            not t.is_tensor_storage_on_device(value)
            or not value.is_allocated()
            or value.dtype != dtype
            or value.layout != t.TILE_LAYOUT
            or value.memory_config() != t.DRAM_MEMORY_CONFIG
            or tuple(value.padded_shape) != padded_shape
            or value.device() != device
            or tuple(tile.tile_shape) != (32, 32)
            or tile.transpose_within_face
            or tile.transpose_of_faces
        ):
            raise ValueError("Expected exact interleaved DRAM standard untransposed32x32 tiles on one device/mesh")

    validate(query, (1, 4, 32, 128), t.bfloat16)
    validate(compressed_keys, (1, 1, capacity, 128), t.bfloat16)
    validate(positions, (1, 1, 32, 32), t.float32)
    if output is None:
        output = t.empty(
            output_shape,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            device=device,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
    if not isinstance(output, t.Tensor) or tuple(output.shape) != output_shape:
        raise ValueError(f"Expected output {output_shape}")
    validate(output, output_padded, t.float32)
    if output.buffer_address() in {value.buffer_address() for value in (query, compressed_keys, positions)}:
        raise ValueError("Output must not alias inputs")
    grid = device.compute_with_storage_grid_size()
    key_tiles = capacity // 32
    available = min(grid.x * grid.y, key_tiles if packed_q1 else key_tiles * 4)
    num_cores = available if num_cores is None else num_cores
    if type(num_cores) is not int or not 1 <= num_cores <= available:
        raise ValueError("num_cores must be a positive integer within available compute cores")
    cores = [t.CoreCoord(index % grid.x, index // grid.x) for index in range(num_cores)]
    ranges = t.CoreRangeSet([t.CoreRange(core, core) for core in cores])
    reader, writer, compute = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    for index, core in enumerate(cores):
        reader[core.x][core.y] = [
            query.buffer_address(),
            compressed_keys.buffer_address(),
            positions.buffer_address(),
            index,
            num_cores,
            key_tiles,
            count,
        ]
        writer[core.x][core.y] = [
            output.buffer_address(),
            positions.buffer_address(),
            index,
            num_cores,
            key_tiles,
            count,
        ]
        compute[core.x][core.y] = [index, num_cores, key_tiles, count]
    read_ct, write_ct = [], []
    for value in (query, compressed_keys, positions):
        read_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    for value in (output, positions):
        write_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    config = t.ComputeConfigDescriptor(
        math_fidelity=t.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
        math_approx_mode=False,
    )
    kernel_prefix = "causal_score_packed_q1" if packed_q1 else "causal_score"
    specs = [
        (f"{kernel_prefix}_reader.cpp", read_ct, reader, t.ReaderConfigDescriptor()),
        (f"{kernel_prefix}_writer.cpp", write_ct, writer, t.WriterConfigDescriptor()),
        (f"{kernel_prefix}_compute.cpp", [], compute, config),
    ]
    kernels = [
        t.KernelDescriptor(
            kernel_source=str(_KERNEL_ROOT / name),
            source_type=t.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        for name, ct, rt, cfg in specs
    ]
    cbs = []
    for index, tiles, dtype in (
        (0, 1, t.float32),
        (1, 4, t.bfloat16),
        (2, 8, t.bfloat16),
        (3, 8, t.bfloat16),
        (4, 2, t.float32),
        (5, 1, t.float32),
        (6, 1, t.float32),
    ):
        size = 4096 if dtype == t.float32 else 2048
        cbs.append(
            t.CBDescriptor(
                total_size=tiles * size,
                core_ranges=ranges,
                format_descriptors=[t.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
        )
    t.generic_op(
        [query, compressed_keys, positions, output],
        t.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]),
    )
    return output


def causal_indexer_dots(query, compressed_keys, positions, *, output=None, num_cores=None):
    """Original [1,4,Q,K] per-head dot contract, unchanged for all Q=1..32."""
    return _causal_indexer_dots(query, compressed_keys, positions, output=output, num_cores=num_cores)


def causal_indexer_packed_q1_dots(query, compressed_keys, positions, *, output=None, num_cores=None):
    """Q1 dots [1,1,4,K]: four heads share one physical row tile.

    Each worker owns entire key-column tiles and assembles all four head rows,
    so there are no partial-write races. Four key tiles are retained across
    all four heads; each head uses exactly the original dot arithmetic. The
    output has one quarter of the original physical dot storage. No runtime
    position is read on the host. Warm this distinct program before capture.
    """
    return _causal_indexer_dots(query, compressed_keys, positions, output=output, num_cores=num_cores, packed_q1=True)


def causal_indexer_scores(owner, query, compressed_keys, positions, *, num_cores=None, packed_decode=False):
    """Same causal score contract; opt-in packed_decode changes only Q1 storage.

    Q>1 keeps the existing path even when packed_decode is enabled. The Q1
    head reduction becomes dim=2 over four logical rows, matching the H-axis
    reduction used after the original dim=1 head transpose. Exact score and
    selected-ID equivalence are validated by the stage's component probes.
    """
    import ttnn as t

    if type(packed_decode) is not bool:
        raise ValueError("packed_decode must be bool")
    if packed_decode and query.shape[-2] == 1:
        dots = causal_indexer_packed_q1_dots(query, compressed_keys, positions, num_cores=num_cores)
        return t.multiply(t.sum(t.relu(dots), dim=2, keepdim=True), 128**-0.5)
    dots = causal_indexer_dots(query, compressed_keys, positions, num_cores=num_cores)
    return t.multiply(t.sum(t.relu(dots), dim=1, keepdim=True), 128**-0.5)
