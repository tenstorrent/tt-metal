# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Production Galaxy collective adapters injected into 2D module configs.

``MLP2D`` and ``RMSNorm2D`` consume Galaxy resources directly through their
``tt_ccl`` collaborator and ``collective_resource_selector``. ``Attention2D``
and ``LMHead2D`` instead take narrow callables because their collective
signatures are not stable; those adapters live here so both reconstructed
Galaxy models share one implementation without importing model code.

This is the deliberately separate Galaxy CCL owner described in the plan. It is
not merged into ``models/common/modules/tt_ccl.py``; see the CCL follow-up note
in ``MILESTONE_A_STATUS.md``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

import ttnn
from models.common.models.galaxy.plans import select_galaxy_resource
from models.common.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
    GalaxyDecodePlacements,
    GalaxyDenseGeometry,
)
from models.common.modules.attention.attention_2d import Attention2DLowLevelCallables, PrefillRowMode


def deallocate_if_allocated(tensor: Any) -> None:
    if tensor is None:
        return
    is_allocated = getattr(tensor, "is_allocated", None)
    if callable(is_allocated) and not is_allocated():
        return
    deallocate = getattr(tensor, "deallocate", None)
    if callable(deallocate):
        deallocate(True)


def _buffer_address(tensor: Any) -> int | None:
    address = getattr(tensor, "buffer_address", None)
    if not callable(address):
        return None
    try:
        return int(address())
    except BaseException:
        return None


def _aliases_borrowed_buffer(tensor: Any, resource: Any) -> bool:
    """Report whether a collective result aliases a borrowed persistent buffer."""

    address = _buffer_address(tensor)
    for buffer in getattr(resource, "persistent_output_buffers", ()) or ():
        if tensor is buffer:
            return True
        if address is not None and address == _buffer_address(buffer):
            return True
    return False


class GalaxyColumnAllReduce:
    """Column (axis-1) all-reduce satisfying the LMHead2D collective contract.

    The LM head reduces the hidden dimension over the four mesh columns. The
    collective borrows its input and returns a distinct owned output, which is
    exactly what ``LMHead2D`` validates before it will accept the callable.
    """

    cluster_axis = 1
    consumes_input = False
    returns_owned_output = True

    def __init__(
        self,
        mesh_device: Any,
        *,
        num_links: int = 1,
        topology: Any = ttnn.Topology.Linear,
        memory_config: Any = None,
    ):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config

    def __call__(self, tensor: Any) -> Any:
        return ttnn.all_reduce(
            tensor,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config or tensor.memory_config(),
        )


class GalaxyColumnUserSelector:
    """Select each mesh column's user slice from a column-replicated tensor.

    Decode logits leave ``LMHead2D`` with the whole physical batch present on
    every device — the vocabulary is sharded over mesh rows and the column
    all-reduce leaves the four columns identical — while ``Sampling2D`` consumes
    one column's ``users_per_column`` rows, with its top-k, top-p, temperature
    and seed buffers sharded the same way. TTNN has no per-column slice, so the
    selection is a matmul against a one-hot selector whose *rows differ per
    column*: the host source is ``I(32)`` sharded over columns on the user axis,
    so column ``c`` holds rows ``8c .. 8c + 7``. The product is an exact row
    gather, not an arithmetic mix.

    Placement:

    - selector host source ``[1, 1, 32, 32]``, sharded ``(None, 2)``, so each
      device owns ``[1, 1, 8, 32]``;
    - input ``[1, 1, 32, local_vocab]``, replicated over columns;
    - output ``[1, 1, 8, local_vocab]``, which is exactly the layout the
      Milestone A ``Sampling2D`` hardware qualification staged by hand.

    **Unqualified.** This composition has never run on a Galaxy mesh. Qualify it
    with the focused selector test before trusting a device sampling path built
    on it; the alternative is composing the logits to host and calling
    ``Sampling2D.sample_host``.
    """

    def __init__(
        self,
        mesh_device: Any,
        *,
        max_batch_size: int = GALAXY_PHYSICAL_BATCH,
        users_per_column: int = GALAXY_USERS_PER_COLUMN,
        dtype: Any = ttnn.bfloat16,
        memory_config: Any = None,
        compute_kernel_config: Any = None,
    ):
        if max_batch_size != GALAXY_PHYSICAL_BATCH or users_per_column * GALAXY_COLUMNS != max_batch_size:
            raise ValueError("the Galaxy column user selector requires batch 32 as 8 users per column")
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.users_per_column = users_per_column
        self.dtype = dtype
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.compute_kernel_config = compute_kernel_config
        self._selector: Any = None

    def selector(self) -> Any:
        if self._selector is None:
            identity = torch.eye(self.max_batch_size, dtype=torch.float32).reshape(1, 1, self.max_batch_size, -1)
            self._selector = ttnn.from_torch(
                identity,
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 2), mesh_shape=GALAXY_MESH_SHAPE),
            )
        return self._selector

    def __call__(self, tensor: Any) -> Any:
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != 4 or shape[-2] != self.max_batch_size:
            raise ValueError(f"column user selection expects [1, 1, {self.max_batch_size}, W], got {shape}")
        return ttnn.matmul(
            self.selector(),
            tensor,
            memory_config=self.memory_config,
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

    def release(self) -> None:
        deallocate_if_allocated(self._selector)
        self._selector = None


def galaxy_runtime_tensor_factory(
    offsets: tuple[int, ...],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    mesh_device: Any,
) -> tuple[Any, Any, Any]:
    """Materialize Attention2D's replicated batch-offset and prefix-bound tensors."""

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def make(values: tuple[int, ...]) -> Any:
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return make(offsets), make(lower), make(upper)


class GalaxyAttentionCollectives:
    """Mode-aware Attention2D collectives backed only by Galaxy resources.

    One instance is shared by every layer of one model: it holds no per-layer
    state, only the fused batch-offset tensor and the injected resource owner.
    """

    def __init__(
        self,
        *,
        resources: Any,
        mesh_device: Any,
        geometry: GalaxyDenseGeometry,
        decode_placements: GalaxyDecodePlacements,
        transformation_matrices: Callable[[], dict[str, Any]] | None = None,
        use_fused_qk_rotary: bool = False,
        collective_dtype: Any = ttnn.bfloat8_b,
        head_dtype: Any = ttnn.bfloat16,
    ):
        self.resources = resources
        self.mesh_device = mesh_device
        self.geometry = geometry
        self.decode_placements = decode_placements
        self._transformation_matrices = transformation_matrices
        self.use_fused_qk_rotary = use_fused_qk_rotary
        self.collective_dtype = collective_dtype
        self.head_dtype = head_dtype
        self._batch_offsets: Any = None
        self._closed = False

    # Lifecycle

    def bind_transformation_matrices(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Borrow the model's RoPE transformation matrices through a provider."""

        if not callable(provider):
            raise TypeError("transformation-matrix provider must be callable")
        self._transformation_matrices = provider

    def _rope_matrices(self) -> dict[str, Any]:
        if self._transformation_matrices is None:
            raise RuntimeError("RoPE transformation matrices have not been bound to the Galaxy collectives")
        return self._transformation_matrices()

    def cleanup(self) -> None:
        if self._closed:
            return
        deallocate_if_allocated(self._batch_offsets)
        self._batch_offsets = None
        self._closed = True

    def callables(self) -> Attention2DLowLevelCallables:
        return Attention2DLowLevelCallables(
            rotary=self.rotary,
            reduce_qkv=self.reduce_qkv,
            gather_heads=self.gather_heads,
            reduce_output=self.reduce_output,
            is_borrowed_output=self.is_borrowed_output,
            reduce_create_qkv_heads=self.reduce_create_qkv_heads,
            gather_users=self.gather_users,
        )

    # Attention2D low-level contract

    def is_borrowed_output(self, tensor: Any) -> bool:
        del tensor
        return False

    def rotary(self, q: Any, k: Any, rot_mats: Any, *, mode: str, **_: Any) -> tuple[Any, Any]:
        """Apply Llama-style RoPE to the created Q/K heads."""

        if rot_mats is None or len(rot_mats) < 2:
            raise ValueError("Galaxy attention rotary requires resolved (cos, sin) tensors")
        cos, sin = rot_mats[0], rot_mats[1]
        matrices = self._rope_matrices()
        if mode == "decode":
            if self.use_fused_qk_rotary:
                return ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, matrices["decode"])
            return (
                ttnn.experimental.rotary_embedding_llama(q, cos, sin, matrices["decode"], is_decode_mode=True),
                ttnn.experimental.rotary_embedding_llama(k, cos, sin, matrices["decode"], is_decode_mode=True),
            )
        # Prefill RoPE requires bfloat16 inputs; the projections may be bf8.
        rotated = []
        for tensor in (q, k):
            staged = tensor
            if staged.dtype != ttnn.bfloat16:
                staged = ttnn.typecast(tensor, dtype=ttnn.bfloat16)
            rotated.append(
                ttnn.experimental.rotary_embedding_llama(staged, cos, sin, matrices["prefill"], is_decode_mode=False)
            )
            if staged is not tensor:
                deallocate_if_allocated(staged)
        return rotated[0], rotated[1]

    def reduce_qkv(self, tensor: Any, *, mode: str, recipe: Any = None, **_: Any) -> Any:
        """Reduce the fused QKV projection over the four mesh columns.

        Concatenated physical-batch-32 prefill arrives as one token stream and
        leaves as one row per user, which is the layout ``nlp_create_qkv_heads``,
        the per-row causal SDPA, and the per-user paged cache fill all require.
        """

        reduced = self._all_reduce(tensor, mode=mode, cluster_axis=1)
        return self._split_rows(reduced, recipe)

    def reduce_output(self, tensor: Any, *, mode: str, recipe: Any = None, **_: Any) -> Any:
        """Reduce the attention output projection over the eight mesh rows.

        The residual stream is always one token stream, so a concatenated
        prefill merges its per-user rows back before the reduction.
        """

        # ``_merge_rows`` returns a view over the caller's buffer, which
        # Attention2D still owns and releases; it must not be deallocated here.
        return self._all_reduce(self._merge_rows(tensor, recipe), mode=mode, cluster_axis=0)

    def gather_heads(self, tensor: Any, **_: Any) -> Any:
        """Return the concatenated heads unchanged.

        Decode gathers users before the head concat, and prefill keeps the
        row-local K shard the WO matmul expects, so no collective is required.
        """

        return tensor

    def gather_users(self, tensor: Any, *, mode: str, **_: Any) -> Any:
        """Gather the per-column user group before the decode head concat."""

        if mode != "decode":
            raise ValueError("Galaxy attention user gather is decode-only")
        context = self.resources.context(mode)
        resource = select_galaxy_resource(context, "all_gather", 1, tensor)
        output = ttnn.all_gather(
            tensor,
            1,
            cluster_axis=1,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=self.decode_placements.attention_gather_users_memcfg,
            subdevice_id=context.worker_sub_device_id,
        )
        self.resources.synchronize(mode)
        return output

    def reduce_create_qkv_heads(self, tensor: Any, *, mode: str, config: Any, **_: Any) -> tuple[Any, Any, Any]:
        """Run the production fused column reduction and head creation."""

        if mode != "decode":
            raise ValueError("fused QKV head creation is decode-only")
        context = self.resources.context(mode)
        resource = select_galaxy_resource(context, "all_reduce_create_qkv_heads", 1, tensor)
        key = resource.key
        placements = self.decode_placements
        collective_input = ttnn.to_memory_config(
            tensor, placements.attention_qkv_collective_input_memcfg, dtype=self.collective_dtype
        )
        try:
            reduced, q, k, v = ttnn.experimental.all_reduce_create_qkv_heads(
                collective_input,
                resource.persistent_output_buffers[0],
                cluster_axis=1,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=context.next_semaphore_handles(
                    key.operation, key.cluster_axis, key.geometry, key.sequence_key
                ),
                num_heads=config.n_heads // GALAXY_MESH_SHAPE[0],
                memory_config=placements.attention_qkv_reduced_memcfg,
                topology=resource.topology,
                num_links=resource.num_links,
                subdevice_id=context.worker_sub_device_id,
                num_kv_heads=config.n_kv_heads // GALAXY_MESH_SHAPE[0],
                final_memory_config=placements.attention_heads_memcfg,
                batch_offset=self._fused_batch_offsets(config),
                slice_size=config.users_per_column,
                dtype=self.head_dtype,
            )
        finally:
            deallocate_if_allocated(collective_input)
        self.resources.synchronize(mode)
        # The fused reduction result is scratch. Release it unless the op
        # returned the borrowed persistent buffer itself.
        if not _aliases_borrowed_buffer(reduced, resource):
            deallocate_if_allocated(reduced)
        return q, k, v

    # Internals

    @staticmethod
    def _is_concat32(recipe: Any) -> bool:
        return getattr(recipe, "row_mode", None) is PrefillRowMode.CONCAT_32

    def _split_rows(self, tensor: Any, recipe: Any) -> Any:
        """View ``[1, 1, 32 * S, W]`` as ``[32, 1, S, W]``.

        Every batched prefill row length is a multiple of the 128-token chunk
        alignment, so the split always falls on a tile boundary and the reshape
        is a view rather than a copy.
        """

        if not self._is_concat32(recipe):
            return tensor
        shape = tuple(int(value) for value in tensor.shape)
        tokens, width = shape[-2], shape[-1]
        rows = GALAXY_PHYSICAL_BATCH
        if tokens % rows:
            raise ValueError(f"concat-32 prefill needs {rows} equal rows, got {tokens} tokens")
        return ttnn.reshape(tensor, ttnn.Shape((rows, 1, tokens // rows, width)))

    def _merge_rows(self, tensor: Any, recipe: Any) -> Any:
        """View ``[32, 1, S, W]`` back as the single ``[1, 1, 32 * S, W]`` stream."""

        if not self._is_concat32(recipe):
            return tensor
        shape = tuple(int(value) for value in tensor.shape)
        rows, width = shape[0], shape[-1]
        return ttnn.reshape(tensor, ttnn.Shape((1, 1, rows * shape[-2], width)))

    def _fused_batch_offsets(self, config: Any) -> Any:
        if self._batch_offsets is None:
            self._batch_offsets = ttnn.as_tensor(
                torch.tensor(config.batch_offsets, dtype=torch.int32).reshape(GALAXY_MESH_SHAPE[1], 1),
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=self.mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._batch_offsets

    def _all_reduce(self, tensor: Any, *, mode: str, cluster_axis: int) -> Any:
        context = self.resources.context(mode)
        resource = select_galaxy_resource(context, "all_reduce", cluster_axis, tensor)
        if not resource.persistent_output_buffers:
            raise ValueError(f"{mode} attention all-reduce axis {cluster_axis} requires a persistent output")
        output_memcfg = self._all_reduce_output_memory_config(mode, cluster_axis)
        reduced = ttnn.reduce_scatter(
            tensor,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=context.worker_sub_device_id,
        )
        try:
            output = ttnn.all_gather(
                reduced,
                3,
                cluster_axis=cluster_axis,
                num_links=resource.num_links,
                topology=resource.topology,
                memory_config=output_memcfg,
                subdevice_id=context.worker_sub_device_id,
            )
        finally:
            self.resources.synchronize(mode)
            deallocate_if_allocated(reduced)
        return output

    def _all_reduce_output_memory_config(self, mode: str, cluster_axis: int) -> Any:
        if mode != "decode":
            return ttnn.DRAM_MEMORY_CONFIG
        if cluster_axis == 0:
            # The reduced attention output feeds the fused residual norm.
            return self.decode_placements.residual_memcfg
        return self.decode_placements.attention_qkv_reduced_memcfg
