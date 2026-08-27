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

import os
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


def _ccl_tracing() -> bool:
    return bool(os.getenv("TTTV2_GALAXY_CCL_TRACE"))


def _ccl_shape_note(name: str, tensor: Any) -> str:
    """Describe a tensor the way a CCL shard-fill fault needs it described.

    The two numbers that matter are the tiles the tensor *has* and the tiles its
    shard spec *claims*, because `all_reduce_async`'s reduction kernel waits for a
    full shard on every output core:

        cb_in.wait_front(num_blocks * block_num_tiles);   // ring_size * shard

    so a tensor whose tile count is less than `cores * shard_tiles` leaves the last
    core waiting forever, with no abort and no traceback (D-B19). Printing both
    side by side makes that visible at a glance instead of after a gdb dump: the
    line that named D-B19 read `tiles=501 spec_tiles=504`.

    `tensor.buffer().num_pages()` is not reachable from Python, so the tile counts
    are derived from the logical shape and the shard spec rather than read off the
    buffer.
    """

    tile = 32
    shape = tuple(tensor.shape)
    tiles = (shape[-1] // tile) * (shape[-2] // tile) if len(shape) >= 2 else 0
    try:
        spec = tensor.memory_config().shard_spec
        cores = spec.grid.num_cores()
        shard_shape = tuple(spec.shape)
        spec_tiles = cores * (shard_shape[-1] // tile) * (shard_shape[-2] // tile)
        placement = f"shard={shard_shape} cores={cores} spec_tiles={spec_tiles}"
        fill = "" if spec_tiles == tiles else "  <- SHARDS NOT FULL"
    except BaseException:
        placement = "shard=interleaved"
        fill = ""
    return f"{name}: logical={shape} tiles={tiles} {placement}{fill}"


def _ccl_trace(message: str) -> None:
    """Print and flush a CCL step name when TTTV2_GALAXY_CCL_TRACE is set.

    A device-side CCL hang leaves the host blocked in
    `FDMeshCommandQueue::wait_for_outstanding_reads` with no Python traceback and
    no further log output, so the only way to say *which* enqueued op never
    completed is to name each one before entering it. Off by default; this costs
    nothing when the variable is unset.
    """

    if os.getenv("TTTV2_GALAXY_CCL_TRACE"):
        print(f"[ccl] {message}", flush=True)


def _relocate_sharded(tensor: Any, memory_config: Any) -> Any:
    """Move a sharded tensor to another sharded placement, inside the partition.

    The explicit pair, for the reason `llama33_70b_galaxy/model.py::_relocate`
    documents at length: a direct `to_memory_config` between two shard specs that
    differ in grid *and* width resolves to `reshard_program_factory_generic`,
    which builds over the full compute grid and is illegal under a loaded
    sub-device manager. `sharded_to_interleaved` runs on its input's
    `shard_spec.grid` and `interleaved_to_sharded` on its output shard's cores,
    and both of those are worker-confined here.

    Returns `tensor` itself when it is already in the requested placement, so
    callers must compare identity before deallocating.
    """

    if tensor.memory_config() == memory_config:
        return tensor
    staged = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)
    try:
        return ttnn.interleaved_to_sharded(staged, memory_config)
    finally:
        deallocate_if_allocated(staged)


class GalaxyColumnAllReduce:
    """Column (axis-1) all-reduce satisfying the LMHead2D collective contract.

    The LM head reduces the hidden dimension over the four mesh columns. The
    collective borrows its input and returns a distinct owned output, which is
    exactly what ``LMHead2D`` validates before it will accept the callable.

    ``subdevice_id`` is a *callable*, resolved at call time rather than at
    construction. ``ttnn.all_reduce`` forwards straight to
    ``ttnn::experimental::all_reduce_async`` (see
    ``ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp``), which places its
    workers on the named sub-device and, given none, on the whole compute grid.
    Under the Galaxy decode partition that is illegal - the prefetcher's worker
    sub-device owns 50 of the 70 compute cores - and the LM head reduction
    aborts with

        TT_FATAL ... Kernel group cores do not match sub device cores
                     for programmable core type TENSIX

    It is a callable and not a value because the sub-device id belongs to the
    live operation-boundary context, which is created after this collective is
    constructed and differs between prefill and decode.
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
        subdevice_id: Any = None,
        resources: Any = None,
        placements: Any = None,
        dtype: Any = None,
    ):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config
        self.subdevice_id = subdevice_id
        self.resources = resources
        self.placements = placements
        self.dtype = dtype

    def _resolved_subdevice_id(self) -> Any:
        return self.subdevice_id() if callable(self.subdevice_id) else self.subdevice_id

    def __call__(self, tensor: Any) -> Any:
        if self.resources is not None and self.placements is not None:
            return self._persistent_all_reduce(tensor)
        return ttnn.all_reduce(
            tensor,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config or tensor.memory_config(),
            subdevice_id=self._resolved_subdevice_id(),
        )

    def _persistent_all_reduce(self, tensor: Any) -> Any:
        """Reduce the decode logits against a keyed persistent buffer.

        `ttnn.all_reduce` cannot be used here. It forwards to the
        `all_reduce_async` overload that takes no persistent buffer and no
        semaphores, which falls back to
        `composite_common::composite_all_gather`; that calls `ttnn::concat` with
        no `sub_core_grids`, so the concat builds over the full compute grid and
        the decode sub-device manager rejects it:

            TT_FATAL ... Kernel group cores do not match sub device cores
                         for programmable core type TENSIX
            (from ttnn::prim::concat, under ttnn::all_reduce)

        The persistent-buffer overload takes the fused path, which honours
        `subdevice_id`. This is what the production
        `tt_ccl.line_all_reduce(..., lm_head=True, buffer_key="LM_HEAD")` does.

        Three details are not free choices:

        * **The staging to 32 cores.** `all_reduce_async` validates
          `buffer_shard_volume >= output_shard_volume * ring_size`. The matmul
          leaves the logits on the 24-core ring, where a 4-device reduction would
          need a buffer shard a third larger than the 32-core layout needs. The
          production code reshards for the same reason
          (`LM_HEAD_OUT_RING_RESHARD_MEMCFG`, `num_cores_after_lm_head = 32`).
        * **`fp32_dest_acc=True`.** The production comment on this line is worth
          quoting: "fp32 dest accumulation for the LM-head all_reduce only: its
          bf16 cross-device sum was order-dependent (ETH ring arrival order) ->
          per-row logit non-determinism -> greedy flips". A bfloat16 cross-device
          sum here is not reproducible across runs, which is exactly the class of
          defect this project distrusts a single passing run for.
        * **The result is placed back into the caller's placement.** `LMHead2D`
          uses one `output_memcfg` for the matmul, the collective, the optional
          concat and the mask add, so the tensor this returns must be laid out the
          way the matmul's output was. The two DRAM round trips that costs are a
          real decode-latency cost and belong on the performance follow-up list;
          they are not the reason any of this is correct.
        """

        context = self.resources.context("decode")

        def trace_step(message: str) -> None:
            """Name a step, and under the trace flag *wait* for it.

            Enqueues are asynchronous, so naming the three ops was not enough to
            find D-B19: attempt 3's `logs3/a3_14_step2_gate.log` shows all three
            printing and the block landing on the collective's own final
            `synchronize`, which says only "one of these device programs never
            completed". Synchronising after each one under the trace flag turns
            that into a name. Off by default; a decode step must not synchronise
            three extra times per token in production.
            """

            _ccl_trace(message)
            if _ccl_tracing():
                self.resources.synchronize("decode")
                _ccl_trace(f"{message} -- completed on device")

        _ccl_trace(f"lm_head stage input from {tensor.memory_config().shard_spec.grid.num_cores()} cores")
        _ccl_trace(_ccl_shape_note("lm_head in", tensor))
        staged = _relocate_sharded(tensor, self.placements.lm_head_all_reduce_input_memcfg)
        trace_step(f"lm_head staged, shape={tuple(staged.shape)}")
        _ccl_trace(_ccl_shape_note("lm_head staged", staged))
        resource = select_galaxy_resource(context, "all_reduce", self.cluster_axis, staged)
        if not resource.persistent_output_buffers:
            raise ValueError("decode LM head all-reduce requires a persistent output buffer")
        key = resource.key
        reduced = None
        # The buffer is kept in DRAM between tokens and brought into L1 only for
        # this call. `interleaved_to_sharded` runs on its output shard's cores, so
        # it stays inside the partition, and freeing the L1 copy straight after
        # keeps the largest allocation of the decode step off the worker cores for
        # all but one op.
        _ccl_trace("lm_head buffer DRAM -> L1")
        buffer_l1 = ttnn.interleaved_to_sharded(
            resource.persistent_output_buffers[0], self.placements.lm_head_all_reduce_buffer_memcfg
        )
        trace_step(f"lm_head buffer in L1, shape={tuple(buffer_l1.shape)}")
        _ccl_trace(_ccl_shape_note("lm_head buffer", buffer_l1))
        try:
            reduced = ttnn.experimental.all_reduce_async(
                staged,
                buffer_l1,
                cluster_axis=self.cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=context.next_semaphore_handles(
                    key.operation, key.cluster_axis, key.geometry, key.sequence_key
                ),
                num_links=resource.num_links,
                memory_config=self.placements.lm_head_all_reduce_input_memcfg,
                dtype=self.dtype,
                topology=resource.topology,
                subdevice_id=self._resolved_subdevice_id(),
                fp32_dest_acc=True,
            )
            _ccl_trace(_ccl_shape_note("lm_head reduced", reduced))
            trace_step(f"lm_head all_reduce_async returned, shape={tuple(reduced.shape)}")
            placed = _relocate_sharded(reduced, tensor.memory_config())
            trace_step("lm_head reduced placed back")
            # `all_reduce_async` may hand back the L1 buffer view itself rather
            # than a fresh tensor; that one is released once, below, in `finally`.
            if placed is not reduced and reduced is not buffer_l1:
                deallocate_if_allocated(reduced)
            if placed is reduced:
                raise RuntimeError(
                    "decode LM head all-reduce returned its input placement; LMHead2D requires a distinct tensor"
                )
            return placed
        except BaseException:
            if reduced is not None and reduced is not buffer_l1:
                deallocate_if_allocated(reduced)
            raise
        finally:
            _ccl_trace("lm_head synchronize")
            self.resources.synchronize("decode")
            _ccl_trace("lm_head synchronized")
            deallocate_if_allocated(buffer_l1)
            if staged is not tensor:
                deallocate_if_allocated(staged)


def compose_galaxy_logits(tensor: Any, *, mesh_device: Any = None, vocab_size: int | None = None) -> torch.Tensor:
    """Compose Galaxy LM head logits to `[rows, vocab]` on host.

    **`to_torch_auto_compose` cannot be used here, and gets it wrong silently.**
    It infers a composer from the tensor's own `tensor_topology()`, and a matmul
    output inherits its *activation's* topology, not its weight's. The LM head's
    in0 is replicated over mesh rows and sharded over mesh columns on its last
    axis (the reduced hidden dim), so the logits are labelled the same way -
    while the vocabulary is in fact sharded over mesh *rows* by the weight mapper
    (`[PlacementShard(-1), PlacementShard(-2)]`) and replicated over columns by
    the column all-reduce.

    Auto-composing therefore concatenates the four *columns* along the vocabulary
    axis and takes one row, which on Llama-3.3-70B produces a 64128-wide tensor
    holding four copies of mesh row 0's 16032-token vocabulary slice. Measured on
    `(8, 4)`: the step-2 gate's `comp_pcc` failed with

        RuntimeError: The size of tensor a (16416768) must match the size of
                      tensor b (8208384) at non-singleton dimension 0

    - 128 x 128256 against 128 x 64128. **A caller that slices `[:, :vocab_size]`
    gets no error at all**, just a truncated tensor of the wrong tokens, which is
    why this is a defect and not a shape nuisance: `GalaxyDirectRunner` did
    exactly that, so every step-3 logit, argmax and accuracy number would have
    been silently wrong.

    The composition is the one the production LM head's host reference uses -
    `ConcatMesh2dToTensor(dims=(0, 3))` then `[:1]` in
    `models/demos/llama3_70b_galaxy/tt/lm_head.py::forward_on_host` - with the
    axes swapped because here it is the rows that carry the vocabulary.

    `_compose_kv` in `test_model_wh_galaxy.py` already refuses auto-compose for
    the same reason, one tensor earlier in the same graph.
    """

    device = mesh_device or tensor.device()
    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(3, 0), mesh_shape=GALAXY_MESH_SHAPE),
    ).float()
    # dims=(3, 0) is (mesh-row-target, mesh-column-target): rows concatenate on
    # the vocabulary axis, and the four columns - identical after the all-reduce -
    # stack on the free leading axis. Column 0 is the authoritative copy.
    first_column = composed[0]
    flat = first_column.reshape(-1, first_column.shape[-1])
    return flat[:, :vocab_size] if vocab_size is not None else flat


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
        """Reduce the attention output over a mesh axis, inside the partition.

        Split by mode, exactly as the qualified ``MLP2D._all_reduce_tg`` is:

        **Decode** uses ``ttnn.experimental.all_reduce_async`` against the keyed
        resource's persistent buffer. That is the same call MLP2D makes for the
        identical axis-0 hidden reduction, against the same shared resource -
        ``build_galaxy_decode_collectives`` says so: "Attention and MLP finish
        decode with the same axis-0 hidden reduction, so they share one keyed
        resource and one persistent buffer". The buffer's ``(8, 4, TILE, W)``
        row-sharded shape is a ``buffer_tensor``, not a scatter output, which is
        why the previous ``reduce_scatter`` + ``all_gather`` pair here validated
        it and then never passed it anywhere.

        That pair cannot run under the decode partition at all. The Galaxy decode
        worker sub-device is **not contiguous** - ``worker_cores()`` is ``x=1..3``
        plus ``x=5..6``, split by the ``x=4`` prefetch sender column - and
        ``ttnn.reduce_scatter``'s program factory lays its workers out from
        ``worker_cores(TENSIX, sub_device_id).bounding_box()``, which spans
        straight across that sender column:

            TT_FATAL ... Kernel group cores do not match sub device cores
                         for programmable core type TENSIX

        (``reduce_scatter_program_factory.cpp`` carries its own note that
        "interaction with subdevice needs to be investigated".)

        ``all_reduce_async`` additionally requires a ``WIDTH_SHARDED`` input,
        while the ``wo`` projection lands in interleaved DRAM, so the input is
        first placed into the reduction's own output placement with
        ``interleaved_to_sharded`` - which runs on that placement's cores and so
        stays inside the partition.

        The reduction runs at ``head_dtype``, the decode activation dtype, not at
        ``collective_dtype``. That is not a free choice: this axis-0 resource is
        shared with ``MLP2D``, which hands ``all_reduce_async`` its
        ``decode_ccl_dtype`` - the residual dtype, bfloat16 - and the op sizes its
        circular buffer from the data and checks it against the persistent
        buffer's L1 bank. The two consumers must therefore agree with each other
        *and* with the buffer, which ``build_galaxy_decode_collectives`` now
        allocates at the residual dtype for exactly this reason. Disagreeing
        gives

            TT_FATAL ... Cannot set circular buffer size to 65536. This is
                         larger than the associated dynamically allocated L1
                         buffer bank size of 34816 B

        ``collective_dtype`` still governs the axis-1 fused QKV collective, which
        has its own buffer.

        **Prefill** keeps the plain ``reduce_scatter`` + ``all_gather`` pair,
        which is what MLP2D uses for prefill too (its ``persistent=False``
        branch). The prefill mode plan is not partitioned the way decode is.
        """

        context = self.resources.context(mode)
        resource = select_galaxy_resource(context, "all_reduce", cluster_axis, tensor)
        if not resource.persistent_output_buffers:
            raise ValueError(f"{mode} attention all-reduce axis {cluster_axis} requires a persistent output")
        output_memcfg = self._all_reduce_output_memory_config(mode, cluster_axis)

        if mode != "decode":
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
                return ttnn.all_gather(
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

        key = resource.key
        staged = tensor
        reduced = None
        if not tensor.memory_config().is_sharded():
            staged = ttnn.interleaved_to_sharded(tensor, output_memcfg, output_dtype=self.head_dtype)
        elif tensor.dtype != self.head_dtype:
            staged = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=self.head_dtype)
            staged = ttnn.interleaved_to_sharded(staged, output_memcfg)
        try:
            reduced = ttnn.experimental.all_reduce_async(
                staged,
                resource.persistent_output_buffers[0],
                cluster_axis=cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=context.next_semaphore_handles(
                    key.operation, key.cluster_axis, key.geometry, key.sequence_key
                ),
                num_links=resource.num_links,
                memory_config=output_memcfg,
                dtype=self.head_dtype,
                topology=resource.topology,
                subdevice_id=context.worker_sub_device_id,
                use_optimal_ccl_for_llama=True,
            )
            # Attention2D's decode-output contract is the activation dtype. The
            # reduction already produces it, so this is a guard rather than a
            # step; if a future recipe reduces at a different dtype it converts
            # through the placement-preserving pair, which stays inside the
            # partition where ttnn.typecast would not.
            if reduced.dtype != self.head_dtype:
                recast = ttnn.sharded_to_interleaved(reduced, ttnn.DRAM_MEMORY_CONFIG, output_dtype=self.head_dtype)
                deallocate_if_allocated(reduced)
                reduced = ttnn.interleaved_to_sharded(recast, output_memcfg)
                deallocate_if_allocated(recast)
            return reduced
        finally:
            self.resources.synchronize(mode)
            if staged is not tensor:
                deallocate_if_allocated(staged)

    def _all_reduce_output_memory_config(self, mode: str, cluster_axis: int) -> Any:
        if mode != "decode":
            return ttnn.DRAM_MEMORY_CONFIG
        if cluster_axis == 0:
            # The reduced attention output feeds the fused residual norm.
            return self.decode_placements.residual_memcfg
        return self.decode_placements.attention_qkv_reduced_memcfg
