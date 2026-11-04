# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab's dense linears: the wide grid and the resident fused siblings.

Inside a slab every dense linear runs as one 2D-multicast matmul over the rows, on a per-slab interleaved copy of its
decode weight in the weight's format, on a grid with more columns for narrow N. The QSA K/V and the MoE shared
gate/up run fused on resident copies, built only when the builder is told the slab's rows before the build. The
**exact set** (MoE router, QSA ``index_q`` / ``index_k``) keeps ``decode_matmul.prefill_matmul_program_config``
because its outputs select experts and blocks; call sites ask for it with ``exact=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, SLAB_ROWS_RULE, is_slab_rows
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import prefill_matmul_program_config

# Bytes of one 32 x 32 tile: bf16; bfloat8_b (1024 mantissa bytes + 64 shared exponents); bfloat4_b (512 + 64).
TILE_BYTES = {
    "bf16": 2 * ttnn.TILE_SIZE * ttnn.TILE_SIZE,
    "bf8": ttnn.TILE_SIZE * ttnn.TILE_SIZE + 64,
    "bf4": ttnn.TILE_SIZE * ttnn.TILE_SIZE // 2 + 64,
}
# A resident weight's format: the module's dense weight format (decode_matmul's QWEN38_DENSE_WEIGHT_DTYPE, whose ttnn
# dtypes these names stand for).
SPEC_DTYPES = ("bf16", "bf8", "bf4")
WEIGHT_DTYPE_NAMES = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8", ttnn.bfloat4_b: "bf4"}
TTNN_DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
CACHE_TAGS = {"bf16": "bf16", "bf8": "bf8b", "bf4": "bf4b"}


def weight_dtype_name(dtype) -> str:
    """The spec-dtype name of a module's dense weight dtype (ttnn.bfloat16 / bfloat8_b / bfloat4_b)."""

    if dtype not in WEIGHT_DTYPE_NAMES:
        raise ValueError(f"dense weight dtype must be one of {tuple(WEIGHT_DTYPE_NAMES)}, got {dtype!r}")
    return WEIGHT_DTYPE_NAMES[dtype]


# The DRAM the builder keeps free beyond the resident prefill weights and the context-scaled state (the traces, the
# slab state, the transients inside a layer).
DRAM_MARGIN_BYTES = 512 << 20
# The 2048-row slab's post-build DRAM need per device beyond the resident weights, the context state and the margin:
# the free DRAM after the build at the upper end of a device-measured pass / OOM band at a 32k context, less that
# context's state (Qwen38ResidentContext(32768).context_state_bytes_per_device) and the margin.
SLAB_BAND_FREE_AFTER_BUILD_BYTES = 2_010_176_000
SLAB_BAND_CONTEXT_STATE_BYTES = 436_797_440
SLAB_WORKING_SET_BYTES = SLAB_BAND_FREE_AFTER_BUILD_BYTES - SLAB_BAND_CONTEXT_STATE_BYTES - DRAM_MARGIN_BYTES


def slab_working_set_bytes(rows: int) -> int:
    """The working-set term for a slab of ``rows``: the measured 2048-row band for every slab, 2048 rows being the
    largest (its constant parts, the traces and the fragmentation, are not separable from the rows-linear ones)."""

    if not is_slab_rows(rows):
        raise ValueError(f"a prefill slab takes {SLAB_ROWS_RULE}, got {rows!r}")
    return SLAB_WORKING_SET_BYTES


# The fused siblings of the slab path, local [K, N] per device (the shapes the modules' decode weights have).
# ``shard_dim`` is the mesh shard axis of the [1, 1, K_total, N_total] host block (3: the columns over the devices),
# ``columns`` the sibling's whole-tile column slices.  Layer counts: 36 GDN, 12 QSA, 48 gated residuals per block
# (attn, mlp), 48 MoE.
HIDDEN = 2560
QSA_HEAD_DIM = 256
MOE_LOCAL_INTERMEDIATE = 160
KV_COLUMNS = (("k", 0, QSA_HEAD_DIM), ("v", QSA_HEAD_DIM, QSA_HEAD_DIM))
SHARED_COLUMNS = (
    ("shared_gate", 0, MOE_LOCAL_INTERMEDIATE),
    ("shared_up", MOE_LOCAL_INTERMEDIATE, MOE_LOCAL_INTERMEDIATE),
)
MODULES = ("gr", "gdn", "qsa", "moe")
LAYER_COUNTS = {"gr": 96, "gdn": 36, "qsa": 12, "moe": 48}


@dataclass(frozen=True)
class ResidentSpec:
    """One resident prefill weight: its name in the module's table, local K x N, mesh shard axis, dtype tag and, for a
    fused sibling, the (name, first column, width) slices its output is cut into."""

    name: str
    k: int
    n: int
    shard_dim: int
    dtype: str
    columns: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in SPEC_DTYPES:
            raise ValueError(f"resident prefill weight dtype must be one of {SPEC_DTYPES}, got {self.dtype!r}")

    @property
    def tiles(self) -> int:
        return math.ceil(self.k / ttnn.TILE_SIZE) * math.ceil(self.n / ttnn.TILE_SIZE)

    @property
    def bytes(self) -> int:
        return self.tiles * TILE_BYTES[self.dtype]

    @property
    def ttnn_dtype(self):
        return TTNN_DTYPES[self.dtype]

    @property
    def cache_name(self) -> str:
        """A new tensorbin name per layout and dtype: no cached tensorbin of another layout can be loaded into it."""

        return f"{self.name}_prefill_interleaved.{CACHE_TAGS[self.dtype]}"


def resident_plan(module: str, weight_dtype: str = "bf16") -> tuple[ResidentSpec, ...]:
    """The resident prefill weights of ``module``: the fused [k | v] (QSA) and [gate | up] (MoE) siblings in
    ``weight_dtype``, so the fused linear reads the same tiles; none for the gated residual and the GDN."""

    if module not in MODULES:
        raise ValueError(f"prefill dense module must be one of {MODULES}, got {module!r}")
    if weight_dtype not in SPEC_DTYPES:
        raise ValueError(f"the module's dense weight dtype must be one of {SPEC_DTYPES}, got {weight_dtype!r}")
    if module == "qsa":
        return (ResidentSpec("kv", HIDDEN, 2 * QSA_HEAD_DIM, 3, weight_dtype, KV_COLUMNS),)
    if module == "moe":
        return (ResidentSpec("shared_gate_up", HIDDEN, 2 * MOE_LOCAL_INTERMEDIATE, 3, weight_dtype, SHARED_COLUMNS),)
    return ()


def plan_bytes_per_device(weight_dtypes: Mapping[str, str] | None = None) -> int:
    """Padded-tile bytes of every resident prefill weight of the 48 layers per device (DERIVED); ``weight_dtypes``
    names the modules' dense weight formats for the fused siblings (bf16 unless named)."""

    weight_dtypes = weight_dtypes or {}
    return sum(
        LAYER_COUNTS[module] * sum(spec.bytes for spec in resident_plan(module, weight_dtypes.get(module, "bf16")))
        for module in MODULES
    )


def wide_prefill_matmul_program_config(mesh_device, rows: int, k: int, n: int):
    """The 2D-multicast config that runs the most columns for a ``[rows, k] x [k, n]`` slab linear.

    Ranks the program factory's column count first among configs whose output subblock is at least two tiles wide (a
    one-tile subblock stalls the matmul); a one-tile N falls back to decode_matmul's rule. Every other field is
    decode_matmul's, so only the block order over the cores changes.
    """

    grid = mesh_device.compute_with_storage_grid_size()
    m_tiles, k_tiles, n_tiles = rows // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, math.ceil(n / ttnn.TILE_SIZE)
    if rows % ttnn.TILE_SIZE or k % ttnn.TILE_SIZE:
        raise ValueError(f"prefill linear needs whole row and K tiles, got rows={rows} k={k}")
    best = None
    for cols in range(1, min(int(grid.x), n_tiles) + 1):
        per_core_n = math.ceil(n_tiles / cols)
        running = math.ceil(n_tiles / per_core_n)
        subblock = decode_matmul._largest_divisor(per_core_n, 4)
        if subblock < 2:
            continue
        key = (running, subblock)
        if best is None or key > best[0]:
            best = (key, running, per_core_n, subblock)
    if best is None:
        return prefill_matmul_program_config(mesh_device, rows, k, n)
    _key, running, per_core_n, subblock = best
    grid_rows = min(int(grid.y), m_tiles)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(running, grid_rows),
        in0_block_w=decode_matmul._largest_divisor(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=subblock,
        per_core_M=math.ceil(m_tiles / grid_rows),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class Qwen38TTNNPrefillDenseWeights:
    """The resident prefill weights of one module: name -> (device tensor, spec).  Empty in every module but the QSA
    and the MoE."""

    def __init__(self, entries: Mapping[str, tuple[Any, ResidentSpec]] | None = None) -> None:
        self._entries: dict[str, tuple[Any, ResidentSpec]] = dict(entries or {})

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def items(self):
        return self._entries.items()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def bytes(self) -> int:
        return sum(spec.bytes for _tensor, spec in self._entries.values())

    def tensor(self, name: str):
        return self._entries[name][0]

    def spec(self, name: str) -> ResidentSpec:
        return self._entries[name][1]

    def deallocate(self) -> None:
        for name, (tensor, _spec) in list(self._entries.items()):
            ttnn.deallocate(tensor)
            del self._entries[name]


class Qwen38TTNNPrefillDense:
    """One module's slab linears: the wide program configs (cached per shape), the resident prefill weights (attached
    by the builder after every other resident allocation) and the fused-sibling linear.  Construction touches no
    device: ``mesh_device`` is read when a config is first asked for."""

    def __init__(self, mesh_device) -> None:
        self.mesh_device = mesh_device
        self.weights = Qwen38TTNNPrefillDenseWeights()
        self._program_configs: dict[tuple[int, int, int], Any] = {}

    @classmethod
    def resolve(cls, given: "Qwen38TTNNPrefillDense | None", mesh_device) -> "Qwen38TTNNPrefillDense":
        """The module's ``prefill_dense`` argument, or a fresh one."""

        if given is None:
            return cls(mesh_device)
        if not isinstance(given, Qwen38TTNNPrefillDense):
            raise TypeError(f"prefill_dense must be a Qwen38TTNNPrefillDense or None, got {given!r}")
        return given

    def program_config(self, rows: int, k: int, n: int):
        """The wide 2D-multicast config for one shape, built once."""

        key = (rows, k, n)
        if key not in self._program_configs:
            self._program_configs[key] = wide_prefill_matmul_program_config(self.mesh_device, rows, k, n)
        return self._program_configs[key]

    def resident(self, name: str):
        """The resident prefill weight ``name`` holds, or None (the per-slab copy)."""

        return self.weights.tensor(name) if name in self.weights else None

    def attach(self, weights: Qwen38TTNNPrefillDenseWeights) -> None:
        for name, entry in weights.items():
            if name in self.weights:
                raise ValueError(f"prefill dense weight {name!r} is attached twice")
            self.weights._entries[name] = entry

    def fused_linear(self, activation, name: str, rows: int, *, compute_kernel_config):
        """One linear on the fused sibling ``name`` over the slab rows: the ``[1, 1, rows, N]`` interleaved DRAM output
        the caller cuts into the spec's whole-tile column slices with its own literal bounds (a captured body admits no
        host-integer shape op) and hands to :meth:`retag_sharded`."""

        weight, spec = self.weights.tensor(name), self.weights.spec(name)
        if not spec.columns:
            raise ValueError(f"prefill dense weight {name!r} is not a fused sibling")
        return ttnn.linear(
            activation,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.program_config(rows, spec.k, spec.n),
            compute_kernel_config=compute_kernel_config,
        )

    def retag_sharded(self, *parts, reference, shard_dim: int) -> None:
        """Mesh metadata of a fused sibling's column slices: ``reference``'s distribution and coordinates, sharded on
        ``shard_dim`` (the slices are per-device values already; no bytes move)."""

        topology = reference.tensor_topology()
        for part in parts:
            part.update_tensor_topology(
                ttnn.TensorTopology(
                    topology.distribution_shape(),
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
                    topology.mesh_coords(),
                )
            )

    def deallocate(self) -> None:
        self.weights.deallocate()


def admit_prefill_dense_dram(
    mesh_device,
    *,
    plan_bytes: int,
    context_state_bytes: int,
    slab_working_set: int = SLAB_WORKING_SET_BYTES,
    margin: int = DRAM_MARGIN_BYTES,
) -> dict:
    """Refuse the resident prefill weights when the free DRAM per device (the allocator's view, after every other
    resident allocation) would not hold them plus what the slab process allocates after the build: the context-scaled
    state (the QSA caches and RoPE tables the model allocates next), the 2048-row slab's working set (the measured
    band, ``SLAB_WORKING_SET_BYTES``) and the margin; returns the numbers it admitted on."""

    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    free = int(view.total_bytes_free_per_bank) * banks
    needed = plan_bytes + context_state_bytes + slab_working_set + margin
    numbers = {
        "free_bytes_per_device": free,
        "plan_bytes_per_device": plan_bytes,
        "context_state_bytes_per_device": context_state_bytes,
        "slab_working_set_bytes_per_device": slab_working_set,
        "margin_bytes": margin,
        "needed_bytes_per_device": needed,
        "num_banks": banks,
    }
    if free < needed:

        def mib(value: int) -> str:
            return f"{value / 2**20:.0f} MiB"

        raise ValueError(
            "prefill slab refused: the resident prefill weights need "
            f"{mib(plan_bytes)} per device plus {mib(context_state_bytes)} of context state, the 2048-row slab's "
            f"{mib(slab_working_set)} working set and a {mib(margin)} margin = {mib(needed)}, but {mib(free)} are free "
            f"after the resident build: {(free - plan_bytes) / banks / 1e6:.1f} MB per bank would remain after the "
            "weights, and the slab is measured to run at 251.3 MB per bank and to fail at 141.0; use a smaller "
            "context or a process that reserves less DRAM (the device profiler's program count)"
        )
    return numbers


def _upload(mesh_device, host, spec: ResidentSpec, cache_dir: Path):
    """One resident DRAM-interleaved TILE tensor in the spec's dtype, cached under the spec's new tensorbin name."""

    import torch

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, spec.shard_dim))
    expected = (
        1,
        1,
        spec.k * (MESH_SHAPE[1] if spec.shard_dim == 2 else 1),
        spec.n * (MESH_SHAPE[1] if spec.shard_dim == 3 else 1),
    )
    if tuple(host.shape) != expected:
        raise RuntimeError(f"prefill dense {spec.name}: host block {tuple(host.shape)} is not {expected}")
    tensor = ttnn.as_tensor(
        host.to(torch.bfloat16).contiguous(),
        dtype=spec.ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        cache_file_name=cache_dir / spec.cache_name,
    )
    local = tuple(int(value) for value in tensor.shape)
    if local != (1, 1, spec.k, spec.n):
        raise RuntimeError(f"prefill dense {spec.name}: local block {local} is not {(1, 1, spec.k, spec.n)}")
    return tensor


def _host_blocks(module: str, checkpoint, placement, *, layer_index: int) -> dict:
    """The [1, 1, K_total, N_total] host blocks of a module's fused siblings, from the decode loaders' tt/ sources."""

    import torch

    if module == "qsa":
        from models.demos.blackhole.qwen38_flash_next.ttnn import qsa

        prefix = f"model.language_model.layers.{layer_index}.self_attn."
        k = qsa._expanded_pair_kv(checkpoint.tensor(prefix + "k_proj.weight")).transpose(0, 1)
        v = qsa._expanded_pair_kv(checkpoint.tensor(prefix + "v_proj.weight")).transpose(0, 1)
        kv = torch.cat(
            [
                torch.cat(
                    [k[:, d * QSA_HEAD_DIM : (d + 1) * QSA_HEAD_DIM], v[:, d * QSA_HEAD_DIM : (d + 1) * QSA_HEAD_DIM]],
                    dim=1,
                )
                for d in range(MESH_SHAPE[1])
            ],
            dim=1,
        )
        return {"kv": kv.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * 2 * QSA_HEAD_DIM)}
    if module == "moe":
        from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoEWeights

        source = Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)
        gate = source.shared_gate_proj.transpose(0, 1)
        up = source.shared_up_proj.transpose(0, 1)
        local = MOE_LOCAL_INTERMEDIATE
        gate_up = torch.cat(
            [
                torch.cat([gate[:, d * local : (d + 1) * local], up[:, d * local : (d + 1) * local]], dim=1)
                for d in range(MESH_SHAPE[1])
            ],
            dim=1,
        )
        return {"shared_gate_up": gate_up.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * 2 * local)}
    raise ValueError(f"prefill dense fused siblings belong to the QSA and the MoE, got {module!r}")


def _cache_dir(module: str, checkpoint, mesh_contract, cache_root, *, layer_index: int, tt_metal_sha: str) -> Path:
    """The module's own per-layer cache directory (the decode weights' tensorbins live there under other names)."""

    if module == "qsa":
        from models.demos.blackhole.qwen38_flash_next.ttnn import qsa

        return qsa._cache_dir(cache_root, checkpoint, mesh_contract, tt_metal_sha, "backbone", layer_index)
    from models.demos.blackhole.qwen38_flash_next.ttnn import moe

    return moe._layer_cache_dir(cache_root, checkpoint, mesh_contract, tt_metal_sha, "backbone", layer_index)


def build_prefill_dense_weights(
    module: str,
    mesh_device,
    mesh_contract,
    checkpoint,
    placement,
    cache_root,
    *,
    layer_index: int,
    tt_metal_sha: str,
    weight_dtype: str = "bf16",
) -> Qwen38TTNNPrefillDenseWeights:
    """The resident prefill weights of one backbone module (its fused siblings in ``weight_dtype``, the module's dense
    weight format): nothing is read or allocated for a module without one (the gated residual and the GDN)."""

    plan = resident_plan(module, weight_dtype)
    if not plan:
        return Qwen38TTNNPrefillDenseWeights()
    mesh_contract.validate_mesh(mesh_device)
    hosts = _host_blocks(module, checkpoint, placement, layer_index=layer_index)
    cache_dir = _cache_dir(
        module, checkpoint, mesh_contract, cache_root, layer_index=layer_index, tt_metal_sha=tt_metal_sha
    )
    entries = {}
    for spec in plan:
        entries[spec.name] = (_upload(mesh_device, hosts[spec.name], spec, cache_dir), spec)
    return Qwen38TTNNPrefillDenseWeights(entries)
