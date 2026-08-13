# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Muse-Glimmer-30B **multichip** (tensor-parallel) decoder.

``MultichipDecoder`` keeps the single-chip :class:`OptimizedDecoder` public
contract -- same paged KV semantics, same arbitrary logical prefill lengths, same
131072-token capability, same ``sliding_kv_tail`` hand-off -- with every tensor
crossing the boundary now a **replicated** mesh tensor on a ``1x4`` Blackhole
mesh.  So this module re-runs the whole inherited correctness surface (both layer
kinds, the nine non-aligned prefill lengths, caller-chunked continuation prefill,
batched paged prefill+decode at batch 4/13/32, the full 131072 context, an FP32
HF control, determinism, host-fallback trapping, traced replay, a 64-step soak,
the released checkpoint) against the fractured path, and adds what **only** this
stage can assert:

* the comparison the whole stage rests on -- fractured TTNN against single-chip
  TTNN at 0.999 -- lives in its sibling module ``test_multichip_vs_single_chip.py``
  because it needs two meshes in sequence rather than this module's one.  Every
  PCC number *here* compares TTNN against HF, and a tensor-parallel bug worth
  ~1e-3 (a mis-assigned KV head on one device, a collective that reduces the wrong
  axis, a lost padding column) hides comfortably under the precision policy's
  floor in that comparison;
* ``test_plan_matches_the_hardware`` / ``test_per_device_weight_shapes`` /
  ``test_qkv_weight_is_gqa_assigned`` -- the fracturing is what it claims: 8 query
  heads and 1 GQA-assigned KV head per device, ``kv_head_of_device == [0,0,1,1]``,
  and the four norms plus the RoPE tables bit-identical across the mesh;
* ``test_mlp_padding_is_inert`` -- the ``4992 -> 5120`` zero padding really is
  zero on every device, which is what makes ``silu(0) * 0 = 0`` an argument rather
  than a hope;
* ``test_decode_uses_dram_sharded_matmuls`` -- all six decode projections are
  ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` dispatches on the
  single 16-core boundary grid with the **re-swept** per-device ``in0_block_w``.
  The per-device K of every projection moved, so several single-chip entries are
  illegal here, not merely slow -- and a silent fall back to an auto-configured
  ``ttnn.linear`` would keep every PCC test in this file green;
* ``test_decode_has_exactly_two_collectives`` -- the residual stream stays
  replicated and exactly the two ``ROW_PARALLEL_ROLES`` reduce, once per sublayer,
  on ``ttnn.Topology.Ring``.  A third collective would mean the layer had started
  paying for a boundary conversion the design says it does not;
* ``test_qkv_output_shard_is_padded_not_wrong`` -- the one width 16 cores does not
  divide is the 40-tile QKV *output*, and the test pins that the padding is on an
  output and never on a matmul ``in0``;
* ``test_replicas_are_bit_identical`` / ``test_kv_cache_holds_the_expected_head``
  -- the layout contract in both directions: what comes out of the collective and
  what went into each device's single-head cache.

Two PCC bars
------------

Exactly the single-chip stage's two, and for the same reason: this layer inherits
that stage's precision policy unchanged (BFP8 attention weights, **BFP4** MLP
weights, BFP8 KV cache, BF16 activations), and the BFP4 MLP policy costs the
i.i.d.-Gaussian synthetic harness several times more accuracy than it costs the
released checkpoint.  So the released-checkpoint tests hold the layer to the
functional bar ``PCC_THRESHOLD = 0.995`` and the synthetic harness gets the
documented looser ``SYNTHETIC_PCC_THRESHOLD = 0.99``; see
``test_optimized_decoder``'s module docstring for the measurement that bar
records ($optimize OPT-012).

Neither bar is what pins the *parallelisation*, and that is the point of the
split: an HF comparison at 0.99 cannot see a 1e-3 tensor-parallel fault, so the
multichip-vs-single-chip comparison is asserted separately at 0.999 in
``test_multichip_vs_single_chip.py``, where the shared precision policy cancels
out of both sides.  The collective payload dtype is checked the same way:
``DEFAULT_PREFILL_CCL_DTYPE`` reduces prefill in BFP8, whose measured cost is
~1.2e-4 of PCC, which is inside the synthetic bar's headroom and is exercised
explicitly by ``test_decode_ccl_dtype_override`` at a looser bar.
"""

from __future__ import annotations

import gc
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests.test_functional_decoder import (  # noqa: F401
    CONTINUATION_SPLITS,
    HF_ADVERTISED_CONTEXT,
    LAYER_KINDS,
    PAGE_BLOCK_SIZE,
    PCC_THRESHOLD,
    PREFILL_CHUNK_SIZE,
    PREFILL_SEQ_LENS,
    SHORT_MAX_SEQ,
    _DecoderCache,
    _FallbackGuard,
    _reference_prefill_cache_only,
    capture_decode_trace,
    layer_idx_for,
    reference_layers,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import LAYER_KIND_SLIDING, TILE_SIZE
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_DECODE_CCL_DTYPE,
    DEFAULT_MESH_SHAPE,
    DEFAULT_PREFILL_CCL_DTYPE,
    MULTICHIP_BOUNDARY_CORES,
    MULTICHIP_DECODE_MATMUL,
    MULTICHIP_DECODE_SDPA,
    MULTICHIP_PREFILL_MCAST2D,
    MULTICHIP_PREFILL_MINIMAL_BLOCKS,
    ROW_PARALLEL_ROLES,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import DEFAULT_PRECISION
from models.common.utility_functions import comp_pcc

#: Bar for the i.i.d.-Gaussian synthetic-weight harness under the inherited BFP4
#: MLP policy.  Same value and same justification as the single-chip stage's.
SYNTHETIC_PCC_THRESHOLD = 0.99

#: Bar for **multichip TTNN against single-chip TTNN** on identical weights and
#: inputs.  Both sides run the same precision policy on the same part, so the
#: BF16/BFP floor that forces the HF bars down to 0.99 cancels and only the
#: parallelisation is left: the fracture of six weights, the GQA-assigned KV
#: replication, the two reductions and the re-swept per-device matmul geometry.
#: That is the difference this bar is sensitive to and no other test in the file
#: The multichip-against-single-chip bar (0.999) lives in
#: ``test_multichip_vs_single_chip.py`` with the test that asserts it.

#: Bar for the **two-layer** chain against two chained HF layers.
#:
#: Two layers compose two layers' error: on this synthetic harness a single layer
#: is ~0.9936 and the chain measures 0.972, so the single-layer bar would be
#: arithmetically wrong here rather than merely strict.  0.96 is the measured
#: value plus ~40 % of the gap to 1.0, which catches a real regression in the
#: stacking while not asserting something the precision policy cannot deliver.
STACKED_PCC_THRESHOLD = 0.96

#: Bar for a decoder built with ``ccl_dtype=bfloat8_b``, i.e. with the **decode**
#: reduction also moved off the activation dtype.  The shipped decode payload is
#: BF16 precisely because the measured BFP8 decode cost (~1.8e-4) is larger than
#: the released checkpoint's remaining accuracy budget; this looser bar is what
#: lets the knob be exercised for correctness without asserting a policy the
#: stage rejected.
CCL_OVERRIDE_PCC_THRESHOLD = 0.98

#: Bar for a per-device tensor read back through a lossy on-device dtype: the
#: BFP8 KV cache and the BFP8/BFP4 projection weights.  These tests are checking
#: *which* slice of the checkpoint landed on a device, not the precision of the
#: encoding, so they only need to be tight enough to separate "head 0" from
#: "head 1".
QUANTISED_TENSOR_PCC = 0.99

#: Trace region for the session mesh.  Non-zero because ``test_traced_decode_pcc``
#: and the real-weight traced replay capture a decode graph, and a mesh opened
#: with ``trace_region_size=0`` fails at ``begin_trace_capture``.
TRACE_REGION_SIZE = 90112 * 12

#: Per-device weight and cache shapes the plan implies, as literals.  Deriving
#: them from ``decoder.plan`` would make the test agree with the implementation by
#: construction, which is the one thing it must not do.
PER_DEVICE_WEIGHT_SHAPES = {
    "wqkv": (1, 1, 6656, 1280),
    "attn_gate": (1, 1, 6656, 1024),
    "o_proj": (1, 1, 1024, 6656),
    "mlp_gate": (1, 1, 6656, 5120),
    "mlp_up": (1, 1, 6656, 5120),
    "mlp_down": (1, 1, 5120, 6656),
}

#: Per-device ``(K, N)`` of each projection, for the geometry tables.
PER_DEVICE_KN = {
    "wqkv": (6656, 1280),
    "attn_gate": (6656, 1024),
    "o_proj": (1024, 6656),
    "mlp_gate": (6656, 5120),
    "mlp_up": (6656, 5120),
    "mlp_down": (5120, 6656),
}

#: Per-device MLP intermediate before padding: ``19968 / 4``.
UNPADDED_LOCAL_INTERMEDIATE = 4992

#: Released-checkpoint prefill lengths, matching the single-chip stage's list.
REAL_PREFILL_SEQ_LENS = (1, 100, 2049, 4097, 8193, 12345)

SOAK_STEPS = int(os.environ.get("MG_MULTICHIP_SOAK_STEPS", "64"))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def multichip_mesh():
    """One ``1x4`` ``FABRIC_1D_RING`` mesh for the whole session.

    Session-scoped for two reasons.  Opening a mesh is expensive, but more to the
    point the fabric config has to be set *before* ``open_mesh_device`` and torn
    down after ``close_mesh_device``, so a per-test mesh would re-enter fabric
    bring-up dozens of times; and only one mesh can own these four dies at a
    time.  That second reason is also why the single-chip comparison is a
    *separate module* with its own meshes rather than a test here: carving a 1x1
    submesh out of this mesh makes every later collective on the parent hang.
    """
    devices = ttnn.get_num_devices()
    required = DEFAULT_MESH_SHAPE[0] * DEFAULT_MESH_SHAPE[1]
    if devices < required:  # pragma: no cover - depends on the host
        pytest.skip(f"the multichip stage needs {required} devices, this host has {devices}")
    mesh = open_multichip_mesh(DEFAULT_MESH_SHAPE, trace_region_size=TRACE_REGION_SIZE)
    logger.info(
        f"opened mesh {mesh.shape} devices={mesh.get_num_devices()} grid={mesh.compute_with_storage_grid_size()} "
        f"dram_grid={mesh.dram_grid_size()} trace_region={TRACE_REGION_SIZE}"
    )
    try:
        yield mesh
    finally:
        close_multichip_mesh(mesh)


#: Bounds on the session's CCL semaphore residue.
#:
#: Every CCL dispatch creates a global semaphore, it belongs to the *cached
#: program*, and it is only released by ``clear_program_cache()`` -- measured:
#: six distinct ``reduce_scatter`` shapes take 1536 B of the mesh's 4096 B
#: ``L1_SMALL`` region (256 B each), and clearing the cache returns all of it.
#: So a mesh holds **24 distinct CCL programs** at a time, and a session that
#: builds more without clearing gets
#:
#:     Out of Memory: Not enough space to allocate 1760 B L1_SMALL buffer across
#:     110 banks, where each bank needs to store 16 B, but bank size is 6144 B
#:
#: (Without an ``L1_SMALL`` region the same semaphores land in the main L1 pool
#: and fragment it instead, which breaks the 256-row sharded prefill norm; see
#: ``DEFAULT_L1_SMALL_SIZE``.)
#:
#: This is a property of a *suite*, not of the layer: a stacked model dispatches
#: two CCL shapes per layer kind for decode and one per prefill chunk size, not
#: hundreds.  A test session sweeping nine prefill lengths, ragged batches and
#: two layer kinds does build hundreds, so it clears on either trigger below --
#: whichever fires first -- at the cost of a recompile.
PROGRAM_CACHE_LIMIT = 120
#: Clear once fewer than this many bytes of the ``L1_SMALL`` region are free.
#:
#: Raised from 1536 B by the optimized stage.  Its async prefill collective holds
#: **seven** global semaphores (1,792 B) for the life of the mesh, and
#: ``clear_program_cache()`` does *not* release those -- they are the layer's, not
#: a program's.  So the region a session actually has to play with is 4,352 B
#: rather than 6,144, and a floor of 1536 leaves less headroom than it used to
#: name.  Measured: at 1536 the 256-row prefill in
#: ``test_collective_implementation_is_split_by_payload`` failed under watcher
#: with *"Statically allocated circular buffers ... clash with L1 buffers"* while
#: passing in isolation; at 2560 the whole watcher list passes.
#: The trigger is session position, not the layer -- which is the same thing
#: ``PROGRAM_CACHE_LIMIT`` above is about.
L1_SMALL_FREE_FLOOR = 2560


@pytest.fixture(autouse=True)
def _bound_ccl_semaphores(multichip_mesh):
    """Keep the session's CCL semaphore residue bounded; see PROGRAM_CACHE_LIMIT."""
    yield
    entries = multichip_mesh.num_program_cache_entries()
    free = ttnn.get_memory_view(multichip_mesh, ttnn.BufferType.L1_SMALL).total_bytes_free_per_bank
    if entries > PROGRAM_CACHE_LIMIT or free < L1_SMALL_FREE_FLOOR:
        multichip_mesh.clear_program_cache()
        logger.info(
            f"cleared the program cache at {entries} entries / {free} B of L1_SMALL free " "to release CCL semaphores"
        )


@pytest.fixture(scope="session")
def decoder_cache():
    """LRU of built decoders: one ``MultichipDecoder`` uploads ~320 MB of weights."""
    cache = _DecoderCache(capacity=3)
    yield cache
    cache.clear()


# ----------------------------------------------------------------------------- utils


def _page_table_rows(batch: int, max_seq_len: int, *, seed: int) -> torch.Tensor:
    """The ``[batch, blocks]`` logical-to-physical block map, on the host.

    Kept separate from :func:`make_page_table` so a test that has to interpret the
    paged cache (``test_kv_cache_holds_the_expected_head``) can regenerate exactly
    the mapping the device was given.
    """
    blocks_per_seq = (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(batch * blocks_per_seq, generator=generator)
    return permutation.reshape(batch, blocks_per_seq).to(torch.int32)


def make_page_table(mesh, batch: int, max_seq_len: int, *, seed: int = 7) -> ttnn.Tensor:
    """Randomly permuted block assignment, **replicated** across the mesh.

    The page table is replicated rather than fractured because every device holds
    the same *logical* sequence: KV replication splits the head dimension, never
    the blocks, so all four devices index their own cache with the same rows.
    """
    return ttnn.from_torch(
        _page_table_rows(batch, max_seq_len, seed=seed),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_device_hidden(mesh, hidden: torch.Tensor) -> ttnn.Tensor:
    """``[batch, seq, hidden]`` torch -> replicated ``[1, 1, batch*seq, hidden]`` mesh tensor."""
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def decode_position_tensors(mesh, positions: torch.Tensor):
    """Replicated ``(current_pos, rope_pos_ids)`` for a decode step."""
    current_pos = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rope_pos_ids = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return current_pos, rope_pos_ids


def first_device(tensor: ttnn.Tensor) -> torch.Tensor:
    """Device 0's copy of a replicated mesh tensor, as torch.

    ``ttnn.to_torch`` on a multi-device tensor needs a mesh composer and raises
    without one, so every read in this file goes through ``get_device_tensors``.
    For a replicated tensor any device would do and
    ``test_replicas_are_bit_identical`` is what pins that.
    """
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def device_tensor(tensor: ttnn.Tensor, device: int) -> torch.Tensor:
    """Device ``device``'s copy of a mesh tensor, as torch.

    Weights are DRAM **width-sharded** (one shard per bank), which is what the
    DRAM-sharded matmul requires.  ``to_torch`` handles that layout directly on
    every build this has been run on; the interleaved fallback is here so a build
    that cannot read a sharded buffer host-side reports the *contents* test's
    result rather than an unrelated read error.
    """
    shard = ttnn.get_device_tensors(tensor)[device]
    try:
        return ttnn.to_torch(shard)
    except Exception as error:  # pragma: no cover - layout-dependent host read
        logger.warning(f"direct host read of a width-sharded weight failed ({error}); going via interleaved")
        interleaved = ttnn.sharded_to_interleaved(shard, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_torch(interleaved)
        ttnn.deallocate(interleaved)
        return out


def ccl_mode_of(decoder, phase: str) -> str:
    """The decoder's configured reducer for ``phase`` (``"prefill"`` / ``"decode"``).

    ``ccl_mode`` selects between the fused ``ttnn.all_reduce`` and an explicit
    ``reduce_scatter`` + ``all_gather`` pair.  A ring all-reduce *is* that pair, so
    the two spellings move the same bytes and which one is configured is a
    measurement rather than a contract -- the tests read the knob and assert the
    number of *reductions*, and fall back to the fused op if the knob is absent.

    ``ccl_impl="async"`` (the optimized default for both phases) always dispatches
    the pair, whatever ``ccl_mode`` says, because the async primitives have no
    fused form; the mode knob only chooses between the two *wrapper* spellings.
    """
    if getattr(decoder, f"{phase}_ccl_impl", "wrapper") == "async":
        return "rs_ag"
    return getattr(decoder, f"{phase}_ccl_mode", "all_reduce")


def assert_pcc(label: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = SYNTHETIC_PCC_THRESHOLD):
    passed, message = comp_pcc(expected.float(), actual.float(), threshold)
    logger.info(f"{label}: {message}")
    assert passed, f"{label} below PCC {threshold}: {message}"
    return message


def build_multichip(
    mesh,
    decoder_cache,
    kind: str,
    *,
    max_seq_len: int = SHORT_MAX_SEQ,
    max_batch_size: int = 1,
    real_weights: bool = False,
    chunk: int = PREFILL_CHUNK_SIZE,
    **build_kwargs,
) -> MultichipDecoder:
    layer_idx = layer_idx_for(kind)
    # ``MG_MULTICHIP_DECODE_CCL_DTYPE=bfloat8_b`` re-runs this whole surface with
    # the **decode** collective payload flipped, which is how
    # ``logs/real_weight_decode_bfp8_experiment.log`` was produced.  That
    # measurement is what rejects the BFP8 decode payload (it clears the 0.995
    # real-weight bar by 2.8e-6 against 1.05e-4 for BF16), so it has to be
    # reproducible from committed code rather than from a source edit.
    payload = os.environ.get("MG_MULTICHIP_DECODE_CCL_DTYPE")
    if payload and "decode_ccl_dtype" not in build_kwargs:
        build_kwargs["decode_ccl_dtype"] = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[payload]
    # ``MG_MULTICHIP_CCL_IMPL=wrapper`` / ``MG_MULTICHIP_SHARDED_DECODE_IO=0`` put
    # the layer back on the multichip stage's collectives and layer boundary.
    # They exist so the optimized stage's watcher and correctness evidence can be
    # re-run against the *pre-stage* configuration as a control, from committed
    # code rather than a source edit -- which is how
    # ``logs/watcher_run_wrapper_control.log`` was produced.
    impl = os.environ.get("MG_MULTICHIP_CCL_IMPL")
    if impl and "ccl_impl" not in build_kwargs:
        build_kwargs["ccl_impl"] = impl
    sharded_io = os.environ.get("MG_MULTICHIP_SHARDED_DECODE_IO")
    if sharded_io is not None and "sharded_decode_io" not in build_kwargs:
        build_kwargs["sharded_decode_io"] = sharded_io not in ("0", "false", "no")
    # ``MG_MULTICHIP_CCL_AG_BARRIER=0`` drops the all-gather's barrier semaphore.
    # It is never a shipped configuration, and it exists so that the arm without
    # the barrier is reproducible from committed code:
    # ``MG_MULTICHIP_CCL_IMPL=async MG_MULTICHIP_CCL_AG_BARRIER=0 bash
    # doc/optimized_multichip_decoder/bench/run_watcher.sh`` is what produced
    # ``logs/watcher_no_ag_barrier.log``.
    ag_barrier = os.environ.get("MG_MULTICHIP_CCL_AG_BARRIER")
    if ag_barrier is not None and "ccl_ag_barrier" not in build_kwargs:
        build_kwargs["ccl_ag_barrier"] = ag_barrier not in ("0", "false", "no")
    # Values, not just names: keying on the kwarg names alone would let two builds
    # that differ only in a value (e.g. ``ccl_dtype``) share a cached decoder.
    key = (
        "multichip",
        layer_idx,
        max_seq_len,
        max_batch_size,
        real_weights,
        chunk,
        tuple(sorted((name, repr(value)) for name, value in build_kwargs.items())),
    )

    def factory():
        state_dict = R.real_state_dict(layer_idx) if real_weights else R.synthetic_state_dict(layer_idx)
        return MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            prefill_chunk_size=chunk,
            **build_kwargs,
        )

    return decoder_cache.get(key, factory)


def require_real_weights(layer_idx: int):
    """The released checkpoint's tensors for one layer, or a skip.

    Same policy as the single-chip stage's suite: the real-weight surface is what
    the precision and collective-payload policies are selected on, but an operator
    without the checkpoint cached should still get the rest of the file.
    """
    try:
        return R.real_state_dict(layer_idx)
    except FileNotFoundError as error:  # pragma: no cover - weights not cached
        pytest.skip(f"released checkpoint not cached: {error}")


class _MatmulSpy:
    """Record every ``ttnn.linear`` dispatch's config, and its **output** layout.

    The output matters here in a way it did not on the single-chip layer: 16 cores
    do not divide the 40-tile per-device QKV output, so that one matmul writes a
    padded shard.  ``test_qkv_output_shard_is_padded_not_wrong`` reads these
    records to prove the padding is only ever on an output.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._saved = None

    def __enter__(self):
        self._saved = ttnn.linear

        def traced(a, b, **kwargs):
            out = self._saved(a, b, **kwargs)
            in_spec = a.memory_config().shard_spec
            out_spec = out.memory_config().shard_spec
            self.calls.append(
                {
                    "program_config": kwargs.get("program_config"),
                    "in_memory_layout": a.memory_config().memory_layout,
                    "in_cores": None if in_spec is None else in_spec.grid.num_cores(),
                    "in_shard_width": None if in_spec is None else in_spec.shape[1],
                    "in_width": int(a.shape[-1]),
                    "weight_layout": b.memory_config().memory_layout,
                    "weight_dtype": b.dtype,
                    "k": int(b.shape[-2]),
                    "n": int(b.shape[-1]),
                    "out_width": int(out.shape[-1]),
                    "out_dtype": out.dtype,
                    "out_cores": None if out_spec is None else out_spec.grid.num_cores(),
                    "out_shard_width": None if out_spec is None else out_spec.shape[1],
                }
            )
            return out

        ttnn.linear = traced
        return self

    def __exit__(self, *exc):
        ttnn.linear = self._saved
        return False

    def by_shape(self) -> dict[tuple[int, int], dict]:
        return {(call["k"], call["n"]): call for call in self.calls}


class _CollectiveSpy:
    """Count the reducing collectives and record their topology.

    Both CCL modes are counted, because the layer ships ``all_reduce`` for prefill
    and the explicit ``reduce_scatter`` + ``all_gather`` pair for decode
    (``DEFAULT_PREFILL_CCL_MODE`` / ``DEFAULT_DECODE_CCL_MODE``).  A ring
    all-reduce *is* that pair, so both forms count as **one reduction** and
    :meth:`reductions` is what the tests assert on -- otherwise the test would be
    pinning which of two equivalent spellings is faster, which is a measurement,
    not a contract.
    """

    #: The composite wrappers, on ``ttnn``.
    OPS = ("all_reduce", "reduce_scatter", "all_gather")
    #: The async primitives those wrappers lower to, on ``ttnn.experimental``,
    #: which the optimized stage calls directly so it can own the semaphores, the
    #: staging buffers and the all-gather worker count.  Counted as the wrapper
    #: they replace, because the contract these tests pin is *how many reductions
    #: the layer performs*, not which spelling dispatches them.
    ASYNC_OPS = {"reduce_scatter_minimal_async": "reduce_scatter", "all_gather_async": "all_gather"}

    def __init__(self):
        self.calls: list[dict] = []
        self._saved: dict = {}
        self._saved_async: dict = {}

    def __enter__(self):
        for name in self.OPS:
            original = getattr(ttnn, name)
            self._saved[name] = original
            setattr(ttnn, name, self._trap(name, original))
        for name, counts_as in self.ASYNC_OPS.items():
            original = getattr(ttnn.experimental, name)
            self._saved_async[name] = original
            setattr(ttnn.experimental, name, self._trap(counts_as, original))
        return self

    def _trap(self, name, original):
        def trap(tensor, *args, **kwargs):
            self.calls.append(
                {
                    "op": name,
                    "topology": kwargs.get("topology"),
                    "dtype": tensor.dtype,
                    "width": int(tensor.shape[-1]),
                    "rows": int(tensor.shape[-2]),
                }
            )
            return original(tensor, *args, **kwargs)

        return trap

    def __exit__(self, *exc):
        for name, original in self._saved.items():
            setattr(ttnn, name, original)
        for name, original in self._saved_async.items():
            setattr(ttnn.experimental, name, original)
        return False

    #: A distributed RMSNorm's statistics gather moves two scalars per row, so its
    #: payload is one tile wide against the hidden size's 208.  It is a real op and
    #: it is counted -- as a *statistics* gather, not as half of a reduction --
    #: because the contract these tests pin is how many times the layer reduces
    #: the hidden state across the mesh.  See DEFAULT_PREFILL_FRACTURED_NORM.
    STATS_GATHER_MAX_WIDTH = 2 * 32

    def stats_gathers(self) -> int:
        return sum(1 for c in self.calls if c["op"] == "all_gather" and c["width"] <= self.STATS_GATHER_MAX_WIDTH)

    def reductions(self, mode: str) -> int:
        """Number of full reductions performed, under CCL mode ``mode``."""
        counts = {name: sum(1 for call in self.calls if call["op"] == name) for name in self.OPS}
        counts["all_gather"] -= self.stats_gathers()
        if mode == "rs_ag":
            assert (
                counts["reduce_scatter"] == counts["all_gather"]
            ), f"a reduce-scatter/all-gather reduction must dispatch both halves, saw {counts}"
            assert counts["all_reduce"] == 0, f"ccl_mode='rs_ag' must not dispatch ttnn.all_reduce: {counts}"
            return counts["reduce_scatter"]
        assert (
            counts["reduce_scatter"] == counts["all_gather"] == 0
        ), f"ccl_mode='all_reduce' must not dispatch the scatter/gather pair: {counts}"
        return counts["all_reduce"]

    def assert_ring_topology(self):
        """Every collective that takes a topology must be told ``Ring``.

        ``ttnn.all_gather`` ignores and deprecates its ``topology`` argument on
        this build, so it is exempt; the two reducing ops are not.
        """
        for call in self.calls:
            if call["op"] == "all_gather":
                continue
            assert call["topology"] == CCL_TOPOLOGY, (
                f"{call['op']} was dispatched with topology={call['topology']}, not the "
                f"{CCL_TOPOLOGY} the 4-die ring needs"
            )


# ------------------------------------------------------------------- prefill / decode


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_prefill_pcc(multichip_mesh, decoder_cache, reference_layers, kind, seq_len):
    """The nine inherited lengths, including the non-aligned ones.

    Non-aligned lengths matter more here than on the single-chip layer: the
    row-parallel reduction runs on whatever the *padded* chunk produced, so a
    length that is neither tile- nor page- nor chunk-aligned is the case where a
    collective over a partially-padded tensor would show up.  ``seq_len = 1`` also
    pads to a single 32-row tile, which is the only prefill row count that takes
    the DRAM-sharded decode matmul (and therefore reduces from the width-sharded
    L1 layout rather than from DRAM interleaved).
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=101 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ)
    tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0)
    assert tuple(tt_out.shape) == (1, 1, seq_len, decoder.config.hidden_size)
    assert_pcc(
        f"multichip prefill[{kind}] seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("prompt_len", (100, 2048, 3000))
def test_decode_pcc(multichip_mesh, decoder_cache, reference_layers, kind, prompt_len):
    """Prefill, then eight consecutive decode steps off the paged BFP8 cache.

    Multi-step rather than single-step because each step writes one row into each
    device's *single-head* cache and then reads it back through the capped
    ``max_cores_per_head_batch`` SDPA; a fault in either compounds across steps
    and a single decode off a freshly filled cache would not see it.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=202 + prompt_len)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))

    for step in range(8):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=909 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([position]))
        tt_out = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        assert_pcc(
            f"multichip decode[{kind}] prompt={prompt_len} step={step} pos={position}",
            expected,
            first_device(tt_out).reshape(1, 1, -1),
        )
        ttnn.deallocate(tt_out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("first_len,second_len", CONTINUATION_SPLITS)
def test_continuation_prefill(multichip_mesh, decoder_cache, reference_layers, kind, first_len, second_len):
    """Caller-chunked prefill: two ``start_pos``-separated calls == one call.

    The hand-off is the multichip-specific part.  ``sliding_kv_tail`` is now a
    *per-device* mesh tensor carrying that device's one KV head,
    ``[1, 1, tail, 128]`` instead of the single-chip ``[1, 2, tail, 128]``, and it
    is handed straight back to the next ``prefill_forward`` -- never round-tripped
    through torch, which would silently replicate device 0's head onto all four.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    total = first_len + second_len
    hidden = R.synthetic_hidden_states(1, total, seed=90210 + first_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=4242)
    first_out, tail = decoder.prefill_forward(
        to_device_hidden(multichip_mesh, hidden[:, :first_len]),
        page_table=page_table,
        user_id=0,
        return_sliding_kv_tail=True,
    )
    assert (tail is not None) == (kind == LAYER_KIND_SLIDING)
    if tail is not None:
        expected_tail = (1, 1, decoder.sliding_kv_tail_len(first_len), decoder.config.head_dim)
        for name, tensor in zip(("k", "v"), tail):
            assert (
                tuple(tensor.shape) == expected_tail
            ), f"the multichip sliding tail carries one local KV head; {name} is {tuple(tensor.shape)}"
            assert tensor.shape[1] == decoder.config.num_key_value_heads == 1

    second_out = decoder.prefill_forward(
        to_device_hidden(multichip_mesh, hidden[:, first_len:]),
        page_table=page_table,
        user_id=0,
        start_pos=first_len,
        sliding_kv_tail=tail,
    )
    actual = torch.cat(
        [
            first_device(first_out).reshape(1, first_len, -1),
            first_device(second_out).reshape(1, second_len, -1),
        ],
        dim=1,
    )
    assert_pcc(f"multichip continuation prefill[{kind}] {first_len}+{second_len}", expected, actual)
    ttnn.deallocate(first_out)
    ttnn.deallocate(second_out)

    # ...and a decode past the join, so the cache the two calls wrote is read.
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    token = R.synthetic_hidden_states(1, 1, seed=555)
    ref = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([total]))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([total]))
    tt_out = decoder.decode_forward(
        to_device_hidden(multichip_mesh, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"multichip decode after continuation[{kind}] {first_len}+{second_len} pos={total}",
        ref,
        first_device(tt_out).reshape(1, 1, -1),
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("batch", (4, 13, 32))
def test_batched_prefill_decode_pcc(multichip_mesh, decoder_cache, reference_layers, kind, batch):
    """Independent users share one paged cache: ragged per-user prefill, batched decode.

    ``batch=13`` is prime and wider than the 11-core grid, so it has no
    ``batch``-core rectangle and takes the decode head-concat's shape-agnostic
    fallback.  Batch is also where the padded-row hazard lives: every batch from 1
    to 32 tile-pads to the same 32 activation rows, the DRAM-sharded matmul's
    ``per_core_M`` is derived from those padded rows, and on this layer those
    padded rows are additionally what the two collectives move -- so a padded row
    that became an active user would corrupt the reduction, not just one user's
    output.
    """
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 4096
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch)
    page_table = make_page_table(multichip_mesh, batch, max_seq_len, seed=31 + batch)

    # Ragged per-user lengths, but drawn from a bounded set of *distinct* ones.
    # Every distinct prefill row count is a distinct CCL program, and a CCL
    # program holds a semaphore until the program cache is cleared -- 16 fit at a
    # time (see PROGRAM_CACHE_LIMIT), and a clear can only happen between tests.
    # Eight distinct lengths across 32 users keeps every property this test is
    # for (ragged lengths, ragged positions, per-user cache slots, lengths that
    # straddle the 2048 sliding window and are not tile multiples) inside that
    # bound (24 CCL programs); 32 distinct ones exhaust the region mid-test.
    prompt_lens = [2000 + 37 * (user % 8) for user in range(batch)]
    assert (decoder._decode_concat_grid_width(batch) is None) == (batch == 13)
    caches = []
    for user, prompt_len in enumerate(prompt_lens):
        hidden = R.synthetic_hidden_states(1, prompt_len, seed=4000 + user)
        expected, cache = R.reference_prefill(layer, layer_idx, hidden)
        caches.append(cache)
        tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=user)
        assert_pcc(
            f"multichip prefill[{kind}] batch={batch} user={user} seq_len={prompt_len}",
            expected,
            first_device(tt_out).reshape(1, prompt_len, -1),
        )
        ttnn.deallocate(tt_out)

    positions = torch.tensor(prompt_lens, dtype=torch.int32)
    tokens = R.synthetic_hidden_states(batch, 1, seed=8123)
    expected = torch.cat(
        [
            R.reference_decode(
                layer,
                layer_idx,
                tokens[user : user + 1],
                past_key_values=caches[user],
                positions=positions[user : user + 1],
            )
            for user in range(batch)
        ],
        dim=0,
    )
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, positions)
    tt_out = decoder.decode_forward(
        to_device_hidden(multichip_mesh, tokens),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"multichip decode[{kind}] batch={batch} ragged positions",
        expected,
        first_device(tt_out).reshape(batch, 1, -1),
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_nonzero_user(multichip_mesh, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill into a non-zero cache slot, then decode that slot.

    The two things the batched test does not combine: an internal chunk boundary
    (``seq_len > prefill_chunk_size``, so the sliding window is carried across a
    reduction) *and* ``user_id > 0``, so the page-table row slicing and the
    chunked paged read both run off row 2 on all four devices.
    """
    layer_idx, layer = reference_layers[kind]
    seq_len = 12345
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, max_seq_len=SHORT_MAX_SEQ, max_batch_size=4)
    page_table = make_page_table(multichip_mesh, 4, SHORT_MAX_SEQ, seed=555)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=13579)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=2)
    assert_pcc(
        f"multichip multi-chunk prefill[{kind}] user_id=2 seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    positions = torch.tensor([0, 0, seq_len, 0], dtype=torch.int32)
    tokens = R.synthetic_hidden_states(4, 1, seed=13580)
    expected = R.reference_decode(layer, layer_idx, tokens[2:3], past_key_values=cache, positions=positions[2:3])
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, positions)
    tt_out = decoder.decode_forward(
        to_device_hidden(multichip_mesh, tokens),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"multichip decode[{kind}] user_id=2 pos={seq_len}",
        expected,
        first_device(tt_out).reshape(4, 1, -1)[2:3],
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_decode_pcc_vs_fp32_reference(multichip_mesh, decoder_cache, kind):
    """FP32 HF control: rules out an error common-mode to two BF16 graphs.

    Worth keeping on this stage rather than inheriting it in spirit only: the
    collective's payload is BFP8 in prefill, and a reduction error that happened
    to look like BF16 rounding would be invisible against a BF16 reference.
    """
    layer_idx = layer_idx_for(kind)
    layer = R.reference_layer(layer_idx, R.synthetic_state_dict(layer_idx), dtype=torch.float32)
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)

    seq_len = 2049
    hidden_bf16 = R.synthetic_hidden_states(1, seq_len, seed=606060)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden_bf16.float())

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=6060)
    tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden_bf16), page_table=page_table, user_id=0)
    assert_pcc(
        f"multichip prefill[{kind}] vs FP32 HF reference seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    token_bf16 = R.synthetic_hidden_states(1, 1, seed=606061)
    expected = R.reference_decode(
        layer, layer_idx, token_bf16.float(), past_key_values=cache, positions=torch.tensor([seq_len])
    )
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
    tt_out = decoder.decode_forward(
        to_device_hidden(multichip_mesh, token_bf16),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"multichip decode[{kind}] vs FP32 HF reference pos={seq_len}",
        expected,
        first_device(tt_out).reshape(1, 1, -1),
    )
    ttnn.deallocate(tt_out)


# ------------------------------------------------------------------- full 128k context


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (HF_ADVERTISED_CONTEXT, HF_ADVERTISED_CONTEXT - 999))
def test_full_context_prefill(multichip_mesh, decoder_cache, reference_layers, kind, seq_len):
    """HF-vs-TTNN prefill PCC at (and just under) the advertised 131072 context.

    Same reduced reference harness the single-chip suite uses: the HF cache is
    filled with the prefix's K/V without running attention over 131072 queries,
    then the real HF layer runs over 32 query positions against that full prefix.
    """
    from transformers.cache_utils import DynamicCache

    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
    tail = 32
    hidden = R.synthetic_hidden_states(1, seq_len, seed=555 + seq_len)

    page_table = make_page_table(multichip_mesh, 1, HF_ADVERTISED_CONTEXT, seed=99)
    tt_hidden = to_device_hidden(multichip_mesh, hidden)
    tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.deallocate(tt_hidden)

    # The last 32 rows, plus an interior block at an internal chunk boundary: for a
    # sliding layer the tail rows only see the last ~2080 tokens, so the interior
    # block is what validates the mid-prompt window hand-offs.
    offsets = [seq_len - tail]
    if seq_len == HF_ADVERTISED_CONTEXT:
        offsets.insert(0, 8 * PREFILL_CHUNK_SIZE)
    for offset in offsets:
        tt_rows = ttnn.slice(tt_out, [0, 0, offset, 0], [1, 1, offset + tail, decoder.config.hidden_size])
        actual = first_device(tt_rows).reshape(1, tail, -1)
        ttnn.deallocate(tt_rows)

        cache = DynamicCache(config=R.text_config())
        _reference_prefill_cache_only(layer, layer_idx, hidden[:, :offset], cache)
        expected, _ = R.reference_prefill(
            layer, layer_idx, hidden[:, offset : offset + tail], past_key_values=cache, start_pos=offset
        )
        where = "last" if offset == seq_len - tail else f"interior @{offset}"
        assert_pcc(f"multichip prefill[{kind}] full-context seq_len={seq_len} ({where} {tail} rows)", expected, actual)
    ttnn.deallocate(tt_out)


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("prompt_len", (HF_ADVERTISED_CONTEXT - 1, HF_ADVERTISED_CONTEXT - 999))
def test_full_context_decode(multichip_mesh, decoder_cache, reference_layers, kind, prompt_len):
    """Decode at the last valid position of the advertised context, and off a non-aligned prompt.

    Also the depth at which each device's *single-head* cache is read in full: the
    ``full`` (NoPE) layer's decode SDPA touches every cached position, under the
    ``max_cores_per_head_batch`` cap that one local KV head makes load-bearing.
    """
    from transformers.cache_utils import DynamicCache

    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=777 + prompt_len)

    page_table = make_page_table(multichip_mesh, 1, HF_ADVERTISED_CONTEXT, seed=123)
    tt_hidden = to_device_hidden(multichip_mesh, hidden)
    ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.deallocate(tt_hidden)

    cache = DynamicCache(config=R.text_config())
    _reference_prefill_cache_only(layer, layer_idx, hidden, cache)

    position = prompt_len
    token = R.synthetic_hidden_states(1, 1, seed=778)
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position]))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([position]))
    tt_out = decoder.decode_forward(
        to_device_hidden(multichip_mesh, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"multichip decode[{kind}] full-context pos={position}",
        expected,
        first_device(tt_out).reshape(1, 1, -1),
    )
    ttnn.deallocate(tt_out)


# --------------------------------------------------------------------- traced / soak


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_pcc(multichip_mesh, decoder_cache, reference_layers, kind):
    """Decode PCC measured from a warmed *trace replay*, which is the measured perf path.

    A trace on a mesh captures the collectives too, so this is also the only test
    that proves the two reductions replay correctly from a captured graph rather
    than only from an eager dispatch.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    prompt_len = 2048
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=1357)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=246)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))

    positions = torch.tensor([prompt_len])
    token = R.synthetic_hidden_states(1, 1, seed=2468)
    tt_token = to_device_hidden(multichip_mesh, token)
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, positions)
    # The warm-up call inside capture_decode_trace already consumed ``prompt_len``;
    # re-running it during capture and replay writes the same K/V to the same slot,
    # so the reference stays a single decode step.
    trace_id, tt_out = capture_decode_trace(decoder, multichip_mesh, tt_token, current_pos, page_table, rope_pos_ids)
    ttnn.execute_trace(multichip_mesh, trace_id, cq_id=0, blocking=True)
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=positions)
    assert_pcc(
        f"multichip traced decode replay[{kind}] pos={prompt_len}",
        expected,
        first_device(tt_out).reshape(1, 1, -1),
    )
    # Release the trace *and* drop every tensor it captured: a trace's L1 working
    # set stays reserved while any of its tensors is alive, and the next decode in
    # the session needs that L1 back for its circular buffers.
    ttnn.release_trace(multichip_mesh, trace_id)
    for tensor in (tt_out, tt_token, current_pos, rope_pos_ids, page_table):
        ttnn.deallocate(tensor)


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_soak(multichip_mesh, decoder_cache, reference_layers, kind):
    """``SOAK_STEPS`` consecutive decode steps, PCC checked along the way and at the end.

    The soak this stage needs rather than the one it inherits: every step runs two
    collectives over the Ethernet fabric and writes one row into each device's
    single-head BFP8 cache, so a leaked CCL buffer, a fabric-credit fault or a
    cache repack error compounds here where the single-shot tests would miss it.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    prompt_len = 1024
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=515)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=616)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    last = None
    for step in range(SOAK_STEPS):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=2000 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([position]))
        tt_out = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        actual = first_device(tt_out).reshape(1, 1, -1)
        ttnn.deallocate(tt_out)
        assert torch.isfinite(actual.float()).all(), f"non-finite decode output at soak step {step}"
        last = step
        if step % 16 == 0 or step == SOAK_STEPS - 1:
            assert_pcc(f"multichip soak decode[{kind}] step={step} pos={position}", expected, actual)
    assert last == SOAK_STEPS - 1


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_determinism(multichip_mesh, decoder_cache, kind):
    """Three repeats, bit-identical prefill and decode.

    Non-trivial on a mesh: a ring reduction sums four partials, and a
    non-deterministic arrival order would make the sum differ in the last bit
    between runs while every PCC test in the file still passed.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    seq_len = 1024
    hidden = R.synthetic_hidden_states(1, seq_len, seed=246)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=17)

    runs = []
    for _ in range(3):
        tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0)
        runs.append(first_device(tt_out).clone())
        ttnn.deallocate(tt_out)
    assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]), "multichip prefill is not deterministic"

    token = R.synthetic_hidden_states(1, 1, seed=247)
    decode_runs = []
    for _ in range(3):
        current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
        tt_out = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        decode_runs.append(first_device(tt_out).clone())
        ttnn.deallocate(tt_out)
    assert torch.equal(decode_runs[0], decode_runs[1]) and torch.equal(
        decode_runs[1], decode_runs[2]
    ), "multichip decode is not deterministic"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (3000, 12345))
def test_no_host_fallback_in_forward(multichip_mesh, decoder_cache, kind, seq_len):
    """No torch / host round-trip in a measured prefill or decode.

    The mesh raises the stakes: a host fallback on a replicated tensor would have
    to pick *one* device's copy, so a fallback here is a correctness bug and not
    only a performance one.  12345 rows also runs the multi-chunk paths (chunked
    paged SDPA, sliding-tail carry, page-table slicing) inside the guard.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=91)
    tt_hidden = to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, seq_len, seed=92))
    token = to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=93))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))

    with _FallbackGuard() as guard:
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )
    assert not guard.violations, f"host fallback in the multichip path: {sorted(set(guard.violations))}"


# ----------------------------------------------------------------------- real weights


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", REAL_PREFILL_SEQ_LENS)
def test_real_weights_prefill_pcc(multichip_mesh, decoder_cache, kind, seq_len):
    """Released-checkpoint prefill at the functional bar, ``PCC_THRESHOLD``.

    Wide on purpose.  This layer inherits the BFP4 MLP policy *and* adds a BFP8
    prefill collective payload, and both are selected on real-weight evidence, so
    the real-weight surface has to cover the conditions a synthetic result cannot
    settle: sub-tile, non-aligned, multi-chunk, and the tile-only DRAM-sharded
    prefill branch.
    """
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, real_weights=True)

    hidden = R.synthetic_hidden_states(1, seq_len, seed=31337 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=808)
    tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0)
    assert_pcc(
        f"multichip real-weight prefill[{kind}] seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
        threshold=PCC_THRESHOLD,
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_decode_pcc(multichip_mesh, decoder_cache, kind):
    """Released-checkpoint decode: eight steps across the paged BFP8 cache.

    This is the check with the least headroom in the whole programme -- the
    single-chip stage's worst real-weight decode is 0.995079 against the 0.995 bar
    -- which is exactly why the decode collective still reduces in BF16 while
    prefill's reduces in BFP8.
    """
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, real_weights=True)

    prompt_len = 3000
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=1717)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=909)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))

    for step in range(8):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=1800 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([position]))
        tt_out = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        assert_pcc(
            f"multichip real-weight decode[{kind}] step={step} pos={position}",
            expected,
            first_device(tt_out).reshape(1, 1, -1),
            threshold=PCC_THRESHOLD,
        )
        ttnn.deallocate(tt_out)


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_traced_decode_and_batch(multichip_mesh, decoder_cache, kind):
    """Real weights through the traced replay, at batch 8.

    The two remaining conditions the precision and collective policies have to be
    evidenced under: trace replay (the measured performance path, which captures
    the collectives) and batch > 1.
    """
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    batch = 8
    max_seq_len = 4096
    decoder = build_multichip(
        multichip_mesh, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch, real_weights=True
    )
    page_table = make_page_table(multichip_mesh, batch, max_seq_len, seed=414)

    prompt_lens = [1000 + 61 * user for user in range(batch)]
    caches = []
    for user, prompt_len in enumerate(prompt_lens):
        hidden = R.synthetic_hidden_states(1, prompt_len, seed=5000 + user)
        _, cache = R.reference_prefill(layer, layer_idx, hidden)
        caches.append(cache)
        ttnn.deallocate(
            decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=user)
        )

    positions = torch.tensor(prompt_lens, dtype=torch.int32)
    tokens = R.synthetic_hidden_states(batch, 1, seed=6000)
    expected = torch.cat(
        [
            R.reference_decode(
                layer,
                layer_idx,
                tokens[user : user + 1],
                past_key_values=caches[user],
                positions=positions[user : user + 1],
            )
            for user in range(batch)
        ],
        dim=0,
    )
    tt_token = to_device_hidden(multichip_mesh, tokens)
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, positions)
    trace_id, tt_out = capture_decode_trace(decoder, multichip_mesh, tt_token, current_pos, page_table, rope_pos_ids)
    ttnn.execute_trace(multichip_mesh, trace_id, cq_id=0, blocking=True)
    assert_pcc(
        f"multichip real-weight traced decode[{kind}] batch={batch}",
        expected,
        first_device(tt_out).reshape(batch, 1, -1),
        threshold=PCC_THRESHOLD,
    )
    ttnn.release_trace(multichip_mesh, trace_id)
    for tensor in (tt_out, tt_token, current_pos, rope_pos_ids, page_table):
        ttnn.deallocate(tensor)


# ------------------------------------------------------- multichip vs single chip


# The multichip-against-single-chip comparison lives in its own module,
# ``test_multichip_vs_single_chip.py``, and in its own pytest invocation, for two
# separate reasons.  Carving a 1x1 submesh out of this module's open 1x4 mesh
# makes every subsequent collective on the parent mesh hang (minimal repro in that
# module's docstring), and appending that module to *this* invocation is no better:
# the session mesh below is torn down last, so the 1x1 mesh it opens finds the four
# dies still owned and times out on an Ethernet core.  Both cost a ``tt-smi -r``.


# --------------------------------------------------------------- the fractured plan


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_plan_matches_the_hardware(multichip_mesh, decoder_cache, kind):
    """The derived ``MeshPlan`` is the plan the docstring's table claims.

    Two head-count concepts coexist deliberately and this pins which is which:
    ``decoder.plan`` carries the **global** counts from the checkpoint, while
    ``decoder.config`` carries the **local** ones, so every inherited forward
    computes per-device shapes without a second head-count concept in the runtime
    path.  Swapping them would give a layer that builds, runs, and is wrong.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    plan = decoder.plan

    assert plan.tp == 4 == multichip_mesh.get_num_devices()
    # Global, from the checkpoint.
    assert plan.num_attention_heads == 32
    assert plan.num_key_value_heads == 2
    assert plan.head_dim == 128
    assert plan.hidden_size == 6656
    assert plan.intermediate_size == 19968
    # Per device.
    assert plan.local_heads == 8
    assert plan.local_kv_heads == 1
    assert plan.kv_replicated is True, "2 KV heads cannot be split four ways, so they must be replicated"
    assert plan.local_qkv_width == 1280 == (8 + 2 * 1) * 128
    assert plan.local_attn_width == 1024
    assert plan.local_intermediate == 5120, "19968/4 = 4992 is padded up to one shard per DRAM bank"
    assert plan.local_intermediate % (TILE_SIZE * multichip_mesh.dram_grid_size().x) == 0

    # GQA assignment: group size is 32/2 = 16 query heads, and 8 local heads fall
    # inside one group, so devices 0-1 read KV head 0 and devices 2-3 read head 1.
    assert [plan.kv_head_of_device(device) for device in range(plan.tp)] == [0, 0, 1, 1]

    # The layer config carries the *local* counts.
    assert decoder.config.num_attention_heads == 8
    assert decoder.config.num_key_value_heads == 1
    assert decoder.config.hidden_size == 6656, "the residual stream is replicated, not fractured"
    assert decoder.config.intermediate_size == plan.local_intermediate

    # The decode SDPA cap is load-bearing with one local KV head: an uncapped grid
    # would exceed the op's 6-round tree-reduction bound.
    assert MULTICHIP_DECODE_SDPA[4] == 32
    assert decoder.max_cores_per_head_batch == MULTICHIP_DECODE_SDPA[4]
    assert decoder.boundary_cores == MULTICHIP_BOUNDARY_CORES == 16


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_per_device_weight_shapes(multichip_mesh, decoder_cache, kind):
    """Every per-device weight and cache shape, and what stays replicated.

    The shapes are literals rather than expressions over ``decoder.plan``: a test
    that recomputed them from the plan would agree with a wrong plan.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    tensors = {
        "wqkv": decoder.wqkv,
        "attn_gate": decoder.w_attn_gate,
        "o_proj": decoder.wo,
        "mlp_gate": decoder.mlp.gate,
        "mlp_up": decoder.mlp.up,
        "mlp_down": decoder.mlp.down,
    }
    for role, expected in PER_DEVICE_WEIGHT_SHAPES.items():
        tensor = tensors[role]
        assert tuple(tensor.shape) == expected, f"{role} per-device shape is {tuple(tensor.shape)}, expected {expected}"
        assert tensor.dtype == DEFAULT_PRECISION.weight_dtype(
            role
        ), f"{role} is {tensor.dtype}, not the inherited policy's {DEFAULT_PRECISION.weight_dtype(role)}"
        assert tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        assert tensor.memory_config().buffer_type == ttnn.BufferType.DRAM

    blocks = SHORT_MAX_SEQ // PAGE_BLOCK_SIZE
    for name, cache in (("k_cache", decoder.k_cache), ("v_cache", decoder.v_cache)):
        assert tuple(cache.shape) == (
            blocks,
            1,
            PAGE_BLOCK_SIZE,
            128,
        ), f"{name} per-device shape is {tuple(cache.shape)}; KV replication gives each device **one** head"
        assert cache.dtype == DEFAULT_PRECISION.kv_cache_dtype

    # Per-device cache bytes halve against the single-chip layer (2 heads -> 1);
    # they do not quarter, and the reason is the model's GQA ratio, not the mesh.
    logger.info(f"multichip[{kind}] per-device K+V cache bytes: {decoder.local_kv_cache_bytes}")

    # Replicated: the four norms (both forms) and, on a sliding layer, the RoPE tables.
    replicated = []
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        norm = getattr(decoder, name)
        replicated.append((f"{name}.weight", norm.weight))
        replicated.append((f"{name}.weight_rm", norm.weight_rm))
    for name in ("cos_cache", "sin_cache", "cos_cache_tile", "sin_cache_tile"):
        tensor = getattr(decoder, name)
        assert (tensor is not None) == decoder.config.uses_rope, f"{name} presence must follow uses_rope"
        if tensor is not None:
            replicated.append((name, tensor))

    for name, tensor in replicated:
        shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
        assert len(shards) == decoder.plan.tp
        for device in range(1, len(shards)):
            assert torch.equal(
                shards[0], shards[device]
            ), f"{name} must be replicated bit-identically; device {device} differs from device 0"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_qkv_weight_is_gqa_assigned(multichip_mesh, decoder_cache, kind):
    """Device ``d``'s fused ``[q | k | v]`` is the checkpoint slice GQA says it is.

    Rebuilt here from the torch state dict rather than from
    ``multichip_decoder.fused_qkv_weight``, which is the code under test.  The
    interesting half is the KV replication: 2 KV heads over 4 devices, group size
    32/2 = 16 query heads, 8 local heads per device, so devices 0-1 must carry
    global KV head 0 and devices 2-3 head 1.  The last two assertions are what
    would fail on the plausible wrong answers -- ``d % 2`` (which would give
    0,1,0,1) or "device d takes head d // 1" (out of range) -- because they check
    the K/V columns are *shared within a pair* and *different across pairs*.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    layer_idx = layer_idx_for(kind)
    state_dict = R.synthetic_state_dict(layer_idx)
    prefix = f"{R.layer_prefix(layer_idx)}."
    text_config = R.text_config()
    n_heads, n_kv, head_dim = (
        int(text_config.num_attention_heads),
        int(text_config.num_key_value_heads),
        int(text_config.head_dim),
    )
    tp = decoder.plan.tp
    local_heads = n_heads // tp
    q_width = local_heads * head_dim

    # HF stores ``[out, in]``; the device weight is ``[in, out]``.
    wq = state_dict[prefix + "self_attn.q_proj.weight"].float().transpose(0, 1)
    wk = state_dict[prefix + "self_attn.k_proj.weight"].float().transpose(0, 1)
    wv = state_dict[prefix + "self_attn.v_proj.weight"].float().transpose(0, 1)

    kv_blocks = []
    for device in range(tp):
        # Derived from the checkpoint's head counts alone, independently of MeshPlan.
        kv_head = (device * local_heads) * n_kv // n_heads
        assert kv_head == decoder.plan.kv_head_of_device(device)
        expected = torch.cat(
            [
                wq[:, device * q_width : (device + 1) * q_width],
                wk[:, kv_head * head_dim : (kv_head + 1) * head_dim],
                wv[:, kv_head * head_dim : (kv_head + 1) * head_dim],
            ],
            dim=-1,
        )
        actual = device_tensor(decoder.wqkv, device).reshape(expected.shape)
        # BFP8 on device, so PCC rather than equality: this test is about *which*
        # slice landed where, not about the encoding.
        assert_pcc(
            f"multichip wqkv[{kind}] device={device} (q heads {device * local_heads}..{(device + 1) * local_heads}, "
            f"kv head {kv_head})",
            expected,
            actual,
            threshold=QUANTISED_TENSOR_PCC,
        )
        kv_blocks.append(actual[:, q_width:].clone())

    assert torch.equal(kv_blocks[0], kv_blocks[1]), "devices 0 and 1 share GQA KV head 0, so their K/V must match"
    assert torch.equal(kv_blocks[2], kv_blocks[3]), "devices 2 and 3 share GQA KV head 1, so their K/V must match"
    assert not torch.equal(
        kv_blocks[0], kv_blocks[2]
    ), "device pairs 0-1 and 2-3 must hold *different* KV heads; identical K/V means the head assignment collapsed"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_mlp_padding_is_inert(multichip_mesh, decoder_cache, kind):
    """The ``4992 -> 5120`` MLP padding is exactly zero on every device.

    This is what makes the padding an argument rather than a hope: ``silu(0) * 0 =
    0`` in the gate/up product, and the matching zero *rows* of ``mlp_down`` mean
    the padded intermediate columns contribute nothing to the reduced output.  If
    either side were non-zero the padding would be 2.6 % of the MLP weight bytes
    quietly changing the answer -- and note the padding has to be per-device: a
    mesh mapper splits the whole tensor evenly, so padding only the end of the
    full 19968 would hand device 3 all of it.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    pad_from, pad_to = UNPADDED_LOCAL_INTERMEDIATE, decoder.plan.local_intermediate
    assert (pad_from, pad_to) == (4992, 5120)

    for device in range(decoder.plan.tp):
        for role, tensor, axis in (
            ("mlp_gate", decoder.mlp.gate, -1),
            ("mlp_up", decoder.mlp.up, -1),
            ("mlp_down", decoder.mlp.down, -2),
        ):
            values = device_tensor(tensor, device)
            padding = values[..., pad_from:pad_to] if axis == -1 else values[..., pad_from:pad_to, :]
            non_zero = int(torch.count_nonzero(padding))
            assert non_zero == 0, (
                f"{role} device={device}: {non_zero} of {padding.numel()} padded "
                f"{'columns' if axis == -1 else 'rows'} ({pad_from}..{pad_to}) are non-zero, so the "
                "zero-padding is not inert"
            )
            del values, padding
        gc.collect()


# ------------------------------------------------------------------ the decode graph


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_uses_dram_sharded_matmuls(multichip_mesh, decoder_cache, kind):
    """All six decode projections are DRAM-sharded matmuls with the re-swept geometry.

    The stage's whole performance claim rests on this, and on this layer the
    geometry table is not merely tuning: every per-device K moved, so several of
    the single-chip stage's ``in0_block_w`` values are **illegal** here
    (``o_proj``'s K is 1024, i.e. 2 tiles per core, against the single-chip 4).
    A silent fall back to an auto-configured ``ttnn.linear`` would keep every PCC
    test in this file green, so the program config, the width-sharded L1
    activation, the 16-core grid, the width-sharded DRAM weight, the per-role
    weight dtype and the tuned ``in0_block_w`` are all read off the real dispatch.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=71)
    ttnn.deallocate(
        decoder.prefill_forward(
            to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 256, seed=72)),
            page_table=page_table,
            user_id=0,
        )
    )
    token = to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=73))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([256]))

    with _MatmulSpy() as spy:
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )

    assert len(spy.calls) == 6, f"expected 6 decode projections, saw {len(spy.calls)}"
    for call in spy.calls:
        assert isinstance(
            call["program_config"], ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        ), f"decode projection K={call['k']} N={call['n']} is not a DRAM-sharded matmul: {call['program_config']}"
        assert call["in_memory_layout"] == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        assert call["weight_layout"] == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    by_shape = spy.by_shape()
    # gate and up share a per-device shape, so the five distinguishable roles are
    # checked against the table and all six against the invariants above.
    for role in ("wqkv", "attn_gate", "o_proj", "mlp_gate", "mlp_down"):
        k, n = PER_DEVICE_KN[role]
        assert (k, n) in by_shape, f"no decode dispatch with the per-device {role} shape ({k}, {n})"
        call = by_shape[(k, n)]
        cores, in0_block_w = decoder.decode_matmul[role]
        assert cores == MULTICHIP_BOUNDARY_CORES, (
            f"{role} runs on {cores} cores; the whole multichip decode step shares one "
            f"{MULTICHIP_BOUNDARY_CORES}-core grid, which is what removes the single-chip reshards"
        )
        assert call["in_cores"] == cores, f"{role}: activation on {call['in_cores']} cores, expected {cores}"
        assert (
            call["program_config"].in0_block_w == in0_block_w
        ), f"{role}: in0_block_w={call['program_config'].in0_block_w}, expected the re-swept {in0_block_w}"
        assert call["program_config"].per_core_M == 1, "a decode step must be exactly one M tile"
        assert call["program_config"].per_core_N == math.ceil(n / (TILE_SIZE * cores))
        assert call["weight_dtype"] == DEFAULT_PRECISION.weight_dtype(role), (
            f"{role}: weight dtype {call['weight_dtype']} is not the inherited policy's "
            f"{DEFAULT_PRECISION.weight_dtype(role)}"
        )
        # The activation is an ``in0`` and must never be shard-padded: a padded in0
        # would feed reduction columns the weight does not have.
        assert call["in_shard_width"] * cores == call["in_width"], (
            f"{role}: the activation shard is padded ({cores} x {call['in_shard_width']} for "
            f"{call['in_width']} columns), which would corrupt the reduction"
        )
        assert (k // TILE_SIZE) % cores == 0
        assert (k // TILE_SIZE // cores) % in0_block_w == 0


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_collective_implementation_is_split_by_payload(multichip_mesh, decoder_cache, kind):
    """Prefill reduces with the async primitives; decode reduces with the wrappers.

    The split is a measurement.  With the barrier semaphore both primitives ship
    with, the async pair is **15.2 %** faster than ``ttnn.all_reduce`` at the
    107 MB prefill payload (1348.0 against 1588.7 us) and **0.2 % slower** than
    the wrappers at the 40 KB decode payload (0.4555 / 0.4244 against 0.4545 /
    0.4238 ms/token), where the collective is pure fixed cost and one more
    synchronization round outweighs the async op's tuning surface.

    Both halves of that are asserted here, at dispatch level, because either one
    silently flipping still computes the right answer.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    assert decoder.prefill_ccl_impl == "async"
    assert decoder.decode_ccl_impl == "wrapper"
    # ``_bound_ccl_semaphores`` releases accumulated CCL semaphores *after* each
    # test, which is one test too late for this one: it runs a prefill and a decode
    # against one decoder, and run late in a watcher session it failed with
    # "Statically allocated circular buffers ... clash with L1 buffers" while
    # passing in isolation.  Ask for the region up front rather than depending on
    # session position.
    multichip_mesh.clear_program_cache()
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=515151)

    saved = {name: getattr(ttnn, name) for name in _CollectiveSpy.OPS}
    saved_async = {name: getattr(ttnn.experimental, name) for name in _CollectiveSpy.ASYNC_OPS}
    seen: list[str] = []

    def note(name, original):
        def trap(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return trap

    def watch():
        for name, original in saved.items():
            setattr(ttnn, name, note(f"wrapper:{name}", original))
        for name, original in saved_async.items():
            setattr(ttnn.experimental, name, note(f"async:{name}", original))

    def unwatch():
        for name, original in saved.items():
            setattr(ttnn, name, original)
        for name, original in saved_async.items():
            setattr(ttnn.experimental, name, original)

    # ---- prefill must be async, and must not touch a wrapper
    watch()
    try:
        ttnn.deallocate(
            decoder.prefill_forward(
                # Above PREFILL_FRACTURED_NORM_MIN_ROWS, so the shipped fractured
                # prefill norm is the path under test; at or below it the layer
                # deliberately finishes the reduction before the norm instead.
                to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 512, seed=515152)),
                page_table=page_table,
                user_id=0,
            )
        )
    finally:
        unwatch()
    assert not [
        s for s in seen if s.startswith("wrapper:")
    ], f"prefill must reduce with the async primitives, saw {seen}"
    assert sum(1 for s in seen if s == "async:reduce_scatter_minimal_async") == 2, seen
    # Two reductions, and **four** gathers: each reduction's own gather plus the
    # statistics gather of the distributed norm that now sits inside it
    # (DEFAULT_PREFILL_FRACTURED_NORM).  The stats gather is one tile wide against
    # the hidden size's 208, which is what makes it worth having.
    assert sum(1 for s in seen if s == "async:all_gather_async") == (4 if decoder.prefill_fractured_norm else 2), seen

    # ---- decode must be the wrappers, and must not touch an async primitive
    seen.clear()
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([64]))
    watch()
    try:
        ttnn.deallocate(
            decoder.decode_forward(
                to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=515153)),
                current_pos=current_pos,
                page_table=page_table,
                rope_pos_ids=rope_pos_ids,
            )
        )
    finally:
        unwatch()
    assert not [s for s in seen if s.startswith("async:")], f"decode must reduce with the wrappers, saw {seen}"
    assert sum(1 for s in seen if s == "wrapper:reduce_scatter") == 2, seen
    assert sum(1 for s in seen if s == "wrapper:all_gather") == 2, seen

    ttnn.deallocate(page_table)
    for tensor in (current_pos, rope_pos_ids):
        ttnn.deallocate(tensor)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_boundary_layout_is_a_fixed_point(multichip_mesh, decoder_cache, kind):
    """A sharded decode input comes back as the same layout, and is not freed.

    The inter-layer residual contract: the decode residual stays width-sharded in
    L1 across the layer *boundary*, so a stack pays no ``sharded_to_interleaved``
    /``interleaved_to_sharded`` pair per join (measured at 4.7 us on ``sliding``
    and 10.7 us on ``full`` by ``bench/boundary_probe.py``).  Two things have to
    hold for a stack to be able to use it: the layout is a fixed point, and the
    layer does not deallocate a tensor its caller still owns.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=616161)
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([64]))
    boundary = decoder.boundary_memcfg(TILE_SIZE, decoder.config.hidden_size)

    interleaved = to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=616162))
    from_dram = decoder.decode_forward(
        interleaved, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
    )
    assert from_dram.memory_config() == boundary, "an interleaved input must still produce the boundary layout"

    sharded_in = ttnn.interleaved_to_sharded(interleaved, boundary)
    from_sharded = decoder.decode_forward(
        sharded_in, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
    )
    assert from_sharded.memory_config() == boundary
    assert ttnn.is_tensor_storage_on_device(sharded_in), "the layer must not free a tensor its caller owns"
    # Same computation either way: the boundary layout changes conversions, not math.
    assert torch.equal(first_device(from_dram), first_device(from_sharded))

    for tensor in (interleaved, sharded_in, from_dram, from_sharded, page_table, current_pos, rope_pos_ids):
        ttnn.deallocate(tensor)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_has_exactly_two_collectives(multichip_mesh, decoder_cache, kind):
    """Two reductions per decode step and two per prefill chunk, on a ring. No more.

    The design keeps the residual stream **replicated** and reduces once per
    sublayer, so a stacked model pays nothing to hand one layer's output to the
    next.  A third collective would mean the layer had started paying for a
    boundary conversion, which is precisely the fractured-residual contract the
    stage measured and rejected -- and it would not show up in any PCC test.

    Counted per *reduction*, not per dispatch: a ring all-reduce is a
    reduce-scatter plus an all-gather, both modes move the same bytes, and which
    spelling is faster is a measurement rather than a contract.  The mode the
    layer is configured with is what the count is checked against.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=1234)
    # Two row-parallel roles, one reduction each: the count below is not a magic
    # number, it is ``len(ROW_PARALLEL_ROLES)``.
    assert ROW_PARALLEL_ROLES == ("o_proj", "mlp_down")
    per_pass = len(ROW_PARALLEL_ROLES)

    # One prefill chunk -> two reductions.
    with _CollectiveSpy() as spy:
        ttnn.deallocate(
            decoder.prefill_forward(
                to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 256, seed=1235)),
                page_table=page_table,
                user_id=0,
            )
        )
    spy.assert_ring_topology()
    assert (
        spy.reductions(ccl_mode_of(decoder, "prefill")) == per_pass
    ), f"a single-chunk prefill must reduce exactly {per_pass} times {ROW_PARALLEL_ROLES}, saw {spy.calls}"

    # Two prefill chunks -> four.  12345 rows pad to 12352 = 8192 + 4160.
    with _CollectiveSpy() as spy:
        ttnn.deallocate(
            decoder.prefill_forward(
                to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 12345, seed=1236)),
                page_table=page_table,
                user_id=0,
            )
        )
    spy.assert_ring_topology()
    assert (
        spy.reductions(ccl_mode_of(decoder, "prefill")) == 2 * per_pass
    ), f"a two-chunk prefill must reduce {per_pass} times per chunk, saw {spy.calls}"

    # One decode step -> two.
    token = to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=1237))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([12345]))
    with _CollectiveSpy() as spy:
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )
    spy.assert_ring_topology()
    assert (
        spy.reductions(ccl_mode_of(decoder, "decode")) == per_pass
    ), f"a decode step must reduce exactly {per_pass} times {ROW_PARALLEL_ROLES}, saw {spy.calls}"
    # The decode payload is the activation dtype by policy; prefill's is BFP8.
    for call in spy.calls:
        if call["op"] == "all_gather":
            continue
        expected_dtype = decoder.decode_ccl_dtype or decoder.activation_dtype
        assert (
            call["dtype"] == expected_dtype
        ), f"the decode collective moved {call['dtype']}, not the configured {expected_dtype}"
        assert (
            call["width"] == decoder.config.hidden_size
        ), "a row-parallel partial is full width; a narrower payload means the residual was fractured"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_qkv_output_shard_is_padded_not_wrong(multichip_mesh, decoder_cache, reference_layers, kind):
    """The one padded shard in the decode step is an **output**, and it is harmless.

    16 cores divide every width that is a matmul ``in0`` -- 208 hidden tiles, 32
    gated-attention tiles, 160 padded-intermediate tiles -- but not the 40-tile
    per-device QKV *output* (2.5 tiles per core), so that matmul rounds
    ``per_core_N`` up to 3.

    What it then does with the core count is the part worth pinning, because it is
    not what the ``memory_config`` asks for: the DRAM-sharded matmul writes on its
    own storage-core layout (the single-chip stage's TTNN finding 3), so 40 tiles
    at 3 per core come back on **14** cores, not on the 16 the layer requested.
    The logical width is still 1280 and the following ``sharded_to_interleaved``
    drops the pad, so the only cost is 2 of 42 tiles of wasted output work -- and
    the padding never appears on an ``in0``, which is the property that would
    actually corrupt a reduction.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    prompt_len = 256
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=515151)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=5152)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))

    token = R.synthetic_hidden_states(1, 1, seed=5153)
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([prompt_len]))
    with _MatmulSpy() as spy:
        tt_out = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )

    qkv = spy.by_shape()[PER_DEVICE_KN["wqkv"]]
    cores = MULTICHIP_BOUNDARY_CORES
    assert qkv["out_width"] == 1280, f"the QKV projection's logical width is {qkv['out_width']}, expected 1280"
    assert qkv["out_width"] // TILE_SIZE == 40
    assert qkv["program_config"].per_core_N == 3, "40 tiles over 16 cores is 2.5, so per_core_N rounds up to 3"
    assert (
        qkv["out_shard_width"] == 3 * TILE_SIZE
    ), f"the output shard must be per_core_N tiles wide (96 columns); saw {qkv['out_shard_width']}"
    # The op ignores the requested output grid and uses as many cores as its own
    # per_core_N needs: ceil(40 / 3) = 14, not the 16 the memory_config asks for.
    assert qkv["out_cores"] == math.ceil(40 / qkv["program_config"].per_core_N) == 14, (
        f"the QKV output landed on {qkv['out_cores']} cores; the DRAM-sharded matmul writes on its own "
        "storage-core layout, which for 40 tiles at 3 per core is 14"
    )
    assert qkv["out_cores"] * qkv["out_shard_width"] >= 1280, "the padded shard must still cover 1280 columns"

    # ...and not one matmul in this step is handed a padded ``in0``.
    for call in spy.calls:
        assert (
            call["in_shard_width"] * call["in_cores"] == call["in_width"]
        ), f"K={call['k']} N={call['n']}: the activation shard is padded, which is never legal for an in0"

    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([prompt_len]))
    assert_pcc(
        f"multichip decode[{kind}] with the padded QKV output shard",
        expected,
        first_device(tt_out).reshape(1, 1, -1),
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_layer_output_dtype_is_the_activation_dtype(multichip_mesh, decoder_cache, kind):
    """A BFP8 prefill collective payload must not leak into the layer's output dtype.

    The reduced payload is bought by asking the row-parallel *matmul* for the
    lower dtype, so there is no typecast on either side of the collective and no
    extra op.  The residual add that consumes the reduced tensor then takes its
    dtype from the BF16 residual, which is what keeps the layer's output contract
    -- and therefore a stacked model's layer-to-layer hand-off -- unchanged.  If
    that ever stopped holding, the next layer would silently be fed BFP8.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    assert decoder.prefill_ccl_dtype == DEFAULT_PREFILL_CCL_DTYPE == ttnn.bfloat8_b
    assert decoder.decode_ccl_dtype is DEFAULT_DECODE_CCL_DTYPE is None, "decode reduces in the activation dtype"
    assert decoder.activation_dtype == ttnn.bfloat16

    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=6161)
    seq_len = 512
    tt_out = decoder.prefill_forward(
        to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, seq_len, seed=6162)),
        page_table=page_table,
        user_id=0,
    )
    assert tt_out.dtype == ttnn.bfloat16, (
        f"the prefill collective moves {decoder.prefill_ccl_dtype} but the layer output is {tt_out.dtype}, "
        "not the bfloat16 activation dtype the next layer expects"
    )
    ttnn.deallocate(tt_out)

    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
    tt_dec = decoder.decode_forward(
        to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=6163)),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert tt_dec.dtype == ttnn.bfloat16, f"the decode layer output is {tt_dec.dtype}, not bfloat16"
    ttnn.deallocate(tt_dec)


@pytest.mark.timeout(3600)
def test_two_layers_stack(multichip_mesh):
    """One layer's output is the next layer's input, with no conversion in between.

    The whole point of a replicated residual is that a stacked model hands layer
    *n*'s output straight to layer *n+1*; this is the test that says so rather
    than the design saying so.  It uses the two layer *kinds* in the order the
    model has them (three sliding then one full, so a sliding->full boundary is a
    real adjacency) and chains prefill and then a decode step through both, on
    their own paged caches, against the same two HF layers chained the same way.

    Nothing is reshaped, gathered, sliced or re-replicated between the two calls:
    the tensor that comes out of the first ``prefill_forward`` is the tensor that
    goes into the second.  If the layer's output contract drifted from its input
    contract -- a fractured residual, a different dtype, a different memory config
    -- this is where it would show up, and it would not show up anywhere else.

    The PCC bar here is ``STACKED_PCC_THRESHOLD``, not the single-layer one, and
    the difference is not slack: two layers compose two layers' precision error,
    and this model's precision policy is BFP4 MLP weights measured on an
    i.i.d.-Gaussian harness that costs ~2.6x what the real checkpoint does.  A
    single layer lands at ~0.9936 here and the two-layer chain at ~0.972, against
    the *same* HF math -- so composing the single-layer bar would be asserting
    something arithmetically false.  The numerical equivalence of the fracture is
    pinned where the precision policy cancels instead, at 0.999, in
    ``test_multichip_vs_single_chip.py``; what this test is for is the layout
    contract and the semantics of chaining.
    """
    seq_len = 2049
    positions = torch.tensor([seq_len])
    first_idx, second_idx = layer_idx_for(LAYER_KIND_SLIDING), layer_idx_for("full")
    first_layer = R.reference_layer(first_idx, R.synthetic_state_dict(first_idx))
    second_layer = R.reference_layer(second_idx, R.synthetic_state_dict(second_idx))

    hidden = R.synthetic_hidden_states(1, seq_len, seed=31337)
    token = R.synthetic_hidden_states(1, 1, seed=31338)
    reference_first, cache_first = R.reference_prefill(first_layer, first_idx, hidden)
    reference, cache_second = R.reference_prefill(second_layer, second_idx, reference_first)
    reference_decode_first = R.reference_decode(
        first_layer, first_idx, token, past_key_values=cache_first, positions=positions
    )
    reference_decode = R.reference_decode(
        second_layer, second_idx, reference_decode_first, past_key_values=cache_second, positions=positions
    )

    # The session mesh, not a new one: only one mesh can own these four dies, and
    # the two decoders coexisting on it is exactly the stacked-model arrangement.
    mesh = multichip_mesh
    page_table = make_page_table(mesh, 1, 4096, seed=31339)
    try:
        build = dict(
            hf_config=R.hf_config(),
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=4096,
            page_block_size=PAGE_BLOCK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
        )
        first = MultichipDecoder.from_state_dict(R.synthetic_state_dict(first_idx), layer_idx=first_idx, **build)
        second = MultichipDecoder.from_state_dict(R.synthetic_state_dict(second_idx), layer_idx=second_idx, **build)
        # ---- prefill straight through both layers
        tt_hidden = to_device_hidden(mesh, hidden)
        stage_one = first.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
        assert stage_one.dtype == tt_hidden.dtype, "the layer's output dtype must be its input dtype"
        assert stage_one.memory_config() == tt_hidden.memory_config(), (
            f"output memory config {stage_one.memory_config()} differs from the input's "
            f"{tt_hidden.memory_config()}, so a stacked model would need a conversion per layer"
        )
        assert tuple(stage_one.shape) == tuple(tt_hidden.shape)
        stage_two = second.prefill_forward(stage_one, page_table=page_table, user_id=0)
        assert_pcc(
            "multichip two-layer stack prefill",
            reference,
            first_device(stage_two).reshape(1, seq_len, -1),
            threshold=STACKED_PCC_THRESHOLD,
        )
        ttnn.deallocate(stage_one)
        ttnn.deallocate(stage_two)

        # ---- and one decode step, likewise
        current_pos, rope_pos_ids = decode_position_tensors(mesh, positions)
        tt_token = to_device_hidden(mesh, token)
        decode_one = first.decode_forward(
            tt_token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
        )
        assert decode_one.dtype == tt_token.dtype
        # The decode boundary contract is the *optimized* one: width-sharded L1 on
        # the boundary grid, not the DRAM-interleaved layout a caller may hand in.
        # What a stacked model needs is that the contract is a fixed point -- layer
        # n's output is exactly what layer n+1 returns -- so no conversion is
        # needed at any join.  See doc/optimized_multichip_decoder/README.md.
        expected_boundary = first.boundary_memcfg(TILE_SIZE, first.config.hidden_size)
        assert decode_one.memory_config() == expected_boundary, (
            f"decode output memory config {decode_one.memory_config()} is not the boundary contract "
            f"{expected_boundary}, so a stacked model would need a conversion per layer"
        )
        decode_two = second.decode_forward(
            decode_one, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
        )
        assert decode_two.memory_config() == decode_one.memory_config(), (
            "the decode boundary layout must be a fixed point across a layer, or a stack would "
            "drift to a different layout at every join"
        )
        assert tuple(decode_two.shape) == tuple(decode_one.shape)
        assert_pcc(
            "multichip two-layer stack decode",
            reference_decode,
            first_device(decode_two).reshape(1, 1, -1),
            threshold=STACKED_PCC_THRESHOLD,
        )
        ttnn.deallocate(decode_one)
        ttnn.deallocate(decode_two)
        del first, second
        gc.collect()
    finally:
        ttnn.deallocate(page_table)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_replicas_are_bit_identical(multichip_mesh, decoder_cache, kind):
    """Every device's copy of the output is bit-identical, in prefill and decode.

    Not a nicety: the layout contract says ``hidden_states`` in and out are
    replicated, and the *only* reason that is free is that the collective is what
    produces the output on every device.  If the replicas diverged by even one
    bit, a stacked model would drift apart across layers and no single-layer PCC
    test could see it -- which is why this is checked with ``torch.equal`` and not
    with PCC.
    """
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    seq_len = 1024
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=7171)
    tt_out = decoder.prefill_forward(
        to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, seq_len, seed=7172)),
        page_table=page_table,
        user_id=0,
    )
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_out)]
    ttnn.deallocate(tt_out)
    assert len(shards) == decoder.plan.tp == 4
    for device in range(1, len(shards)):
        assert torch.equal(
            shards[0], shards[device]
        ), f"prefill output on device {device} is not bit-identical to device 0's"

    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
    tt_dec = decoder.decode_forward(
        to_device_hidden(multichip_mesh, R.synthetic_hidden_states(1, 1, seed=7173)),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    decode_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_dec)]
    ttnn.deallocate(tt_dec)
    for device in range(1, len(decode_shards)):
        assert torch.equal(
            decode_shards[0], decode_shards[device]
        ), f"decode output on device {device} is not bit-identical to device 0's"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_kv_cache_holds_the_expected_head(multichip_mesh, decoder_cache, reference_layers, kind):
    """Device ``d``'s K *and* V caches hold global KV head ``kv_head_of_device(d)``, and nothing else.

    The other side of ``test_qkv_weight_is_gqa_assigned``: that test checks the
    *weight* slice landed on the right device, this one checks what the forward
    pass then wrote into that device's single-head paged cache, unpicked through
    the same randomly permuted page table the device was given.  A cache holding
    the wrong head still produces a plausible layer output, because SDPA would
    read a self-consistent (wrong) head -- so end-to-end PCC is a weak detector
    and this is a direct one.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind)
    seq_len = 128
    seed = 8181
    hidden = R.synthetic_hidden_states(1, seq_len, seed=seed)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    reference_keys = cache.layers[layer_idx].keys  # [1, 2, seq_len, 128]
    reference_values = cache.layers[layer_idx].values
    assert tuple(reference_keys.shape) == (1, 2, seq_len, decoder.config.head_dim)
    assert tuple(reference_values.shape) == tuple(reference_keys.shape)

    page_rows = _page_table_rows(1, SHORT_MAX_SEQ, seed=seed)
    page_table = ttnn.from_torch(
        page_rows,
        device=multichip_mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(multichip_mesh),
    )
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0))

    blocks = seq_len // PAGE_BLOCK_SIZE
    # Both caches, not just K: they are written by separate ``paged_fill_cache``
    # calls off separate slices of the fused QKV output, so a head-assignment
    # mistake can land in one and not the other.
    for label, cache_tensor, expected in (
        ("K", decoder.k_cache, reference_keys),
        ("V", decoder.v_cache, reference_values),
    ):
        for device in range(decoder.plan.tp):
            head = decoder.plan.kv_head_of_device(device)
            cached = ttnn.to_torch(ttnn.get_device_tensors(cache_tensor)[device]).float()
            assert cached.shape[1] == 1, "each device's cache holds one KV head"
            gathered = torch.cat(
                [cached[int(page_rows[0, block]), 0] for block in range(blocks)], dim=0
            )  # [seq_len, head_dim]
            for candidate in range(int(expected.shape[1])):
                _, message = comp_pcc(expected[0, candidate].float(), gathered, 0.0)
                logger.info(f"multichip {label} cache[{kind}] device={device} vs global KV head {candidate}: {message}")
            # BFP8 cache, so a loose bar -- but far tighter than the wrong head scores.
            assert_pcc(
                f"multichip {label} cache[{kind}] device={device} holds global KV head {head}",
                expected[0, head],
                gathered,
                threshold=QUANTISED_TENSOR_PCC,
            )
            del cached, gathered
            gc.collect()


# ---------------------------------------------------------------- geometry / knobs


@pytest.mark.timeout(600)
def test_geometry_table_is_legal():
    """Every multichip geometry table is legal for the *per-device* shapes, host-side.

    Re-swept rather than inherited, and this is the cheap guard on the re-sweep:
    an illegal ``in0_block_w`` fails at op-compile time inside a trace capture,
    which is a much worse place to find out.  Three rules, all of them
    consequences of the fracture:

    * the decode activation shard must not be padded, so the core count has to
      divide the per-device K in tiles, and ``in0_block_w`` has to divide the
      resulting per-core tile count;
    * the same core count has to serve all three ``in0`` widths -- 208 hidden
      tiles, 32 gated-attention tiles, 160 padded-intermediate tiles -- which is
      what lets the whole decode step share one grid;
    * the prefill activation is DRAM interleaved, not sharded, so there
      ``in0_block_w`` (and ``minimal_matmul``'s ``K_block_size``) must divide the
      whole per-device K in tiles.
    """
    # The three widths the single boundary grid has to serve as an ``in0``.
    for width_tiles, what in ((208, "hidden"), (32, "gated attention"), (160, "padded intermediate")):
        assert width_tiles % MULTICHIP_BOUNDARY_CORES == 0, (
            f"{MULTICHIP_BOUNDARY_CORES} cores do not divide the {width_tiles} {what} tiles, so that tensor "
            "would be shard-padded as a matmul in0"
        )
    assert 6656 // TILE_SIZE == 208 and 1024 // TILE_SIZE == 32 and 5120 // TILE_SIZE == 160

    for (role, dtype), (cores, in0_block_w) in MULTICHIP_DECODE_MATMUL.items():
        k, n = PER_DEVICE_KN[role]
        k_tiles = k // TILE_SIZE
        assert cores == MULTICHIP_BOUNDARY_CORES, f"{role}/{dtype}: the decode step shares one grid, got {cores}"
        assert k_tiles % cores == 0, f"{role}/{dtype}: {cores} cores do not divide {k_tiles} per-device K tiles"
        per_core = k_tiles // cores
        assert (
            per_core % in0_block_w == 0
        ), f"{role}/{dtype}: in0_block_w={in0_block_w} does not divide the {per_core}-tile activation shard"
        assert in0_block_w > 0
    for role in PER_DEVICE_KN:
        assert (
            role,
            DEFAULT_PRECISION.weight_dtype(role),
        ) in MULTICHIP_DECODE_MATMUL, f"{role} has no re-swept decode geometry at its shipped dtype"

    dram_banks = 8  # one shard per bank; asserted against the device in the shape test
    for (role, dtype), entries in MULTICHIP_PREFILL_MCAST2D.items():
        k, n = PER_DEVICE_KN[role]
        thresholds = [max_rows for max_rows, _ in entries]
        assert thresholds == sorted(thresholds), f"{role}/{dtype} 2D-multicast bands are not ascending"
        assert (
            n // TILE_SIZE
        ) % dram_banks == 0, f"{role}: {n // TILE_SIZE} per-device N tiles do not divide {dram_banks} DRAM banks"
        for max_rows, (grid_y, in0_block_w) in entries:
            assert (k // TILE_SIZE) % in0_block_w == 0, (
                f"{role}/{dtype}@{max_rows}: in0_block_w={in0_block_w} does not divide the "
                f"{k // TILE_SIZE} per-device K tiles"
            )
            assert grid_y > 0

    for (role, dtype), entries in MULTICHIP_PREFILL_MINIMAL_BLOCKS.items():
        k, _ = PER_DEVICE_KN[role]
        thresholds = [min_rows for min_rows, _ in entries]
        assert thresholds == sorted(thresholds, reverse=True), f"{role}/{dtype} thresholds are not descending"
        assert thresholds[-1] == TILE_SIZE, f"{role}/{dtype} has no entry covering the smallest prefill"
        for min_rows, blocks in entries:
            if blocks is None:
                continue
            m_block, k_block, n_block = blocks
            assert all(block > 0 for block in blocks), f"{role}/{dtype}@{min_rows} has a non-positive block"
            assert (k // TILE_SIZE) % k_block == 0, (
                f"{role}/{dtype}@{min_rows}: K_block_size={k_block} does not divide the "
                f"{k // TILE_SIZE} per-device K tiles"
            )
    for role in PER_DEVICE_KN:
        assert (
            role,
            DEFAULT_PRECISION.weight_dtype(role),
        ) in MULTICHIP_PREFILL_MINIMAL_BLOCKS, f"{role} has no re-swept prefill blocking at its shipped dtype"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_ccl_dtype_override(multichip_mesh, decoder_cache, reference_layers, kind):
    """``ccl_dtype=bfloat8_b`` moves *both* reductions to BFP8 and still computes.

    The shipped policy splits the payload by mode -- BFP8 in prefill, the
    activation dtype in decode -- because the measured BFP8 decode cost (~1.8e-4
    of PCC) is larger than the released checkpoint's remaining headroom against
    the 0.995 bar.  That is a policy decision, not a capability limit, so the knob
    that the A/B harness sweeps has to keep working: this exercises the rejected
    setting at the looser ``CCL_OVERRIDE_PCC_THRESHOLD``, which is exactly the
    shape of claim the rejection rests on.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_multichip(multichip_mesh, decoder_cache, kind, ccl_dtype=ttnn.bfloat8_b)
    assert decoder.prefill_ccl_dtype == decoder.decode_ccl_dtype == ttnn.bfloat8_b

    seq_len = 2048
    hidden = R.synthetic_hidden_states(1, seq_len, seed=9191)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=9192)
    tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0)
    assert tt_out.dtype == ttnn.bfloat16, "a BFP8 payload must not change the layer's output dtype"
    assert_pcc(
        f"multichip prefill[{kind}] ccl_dtype=bfloat8_b seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
        threshold=CCL_OVERRIDE_PCC_THRESHOLD,
    )
    ttnn.deallocate(tt_out)

    token = R.synthetic_hidden_states(1, 1, seed=9193)
    ref = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([seq_len]))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
    with _CollectiveSpy() as spy:
        tt_dec = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
    assert spy.reductions(ccl_mode_of(decoder, "decode")) == 2
    for call in spy.calls:
        if call["op"] == "all_gather":
            continue
        assert call["dtype"] == ttnn.bfloat8_b, f"the overridden decode payload is {call['dtype']}, not bfloat8_b"
    assert_pcc(
        f"multichip decode[{kind}] ccl_dtype=bfloat8_b pos={seq_len}",
        ref,
        first_device(tt_dec).reshape(1, 1, -1),
        threshold=CCL_OVERRIDE_PCC_THRESHOLD,
    )
    ttnn.deallocate(tt_dec)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("ccl_mode", ("all_reduce", "rs_ag"))
def test_ccl_mode_override(multichip_mesh, decoder_cache, reference_layers, ccl_mode):
    """Both reducer spellings compute the same layer, and dispatch what they claim.

    The mode knob chooses between the two *wrapper* spellings, so this test pins
    ``ccl_impl="wrapper"``: the optimized default calls the async primitives
    directly and they have no fused form, so ``ccl_mode`` is inert there.  Both
    wrapper spellings stay live -- the A/B harness sweeps them, and the async
    rejection in ``doc/optimized_multichip_decoder/README.md`` is measured against
    them -- so both still have to be correct.
    """
    kind = LAYER_KIND_SLIDING
    layer_idx, layer = reference_layers[kind]
    try:
        decoder = build_multichip(multichip_mesh, decoder_cache, kind, ccl_mode=ccl_mode, ccl_impl="wrapper")
    except TypeError as error:  # pragma: no cover - only if the knob is withdrawn
        pytest.skip(f"this build has no ccl_mode knob, so there is one reducer spelling to test: {error}")
    assert ccl_mode_of(decoder, "prefill") == ccl_mode_of(decoder, "decode") == ccl_mode

    seq_len = 1024
    hidden = R.synthetic_hidden_states(1, seq_len, seed=10101)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(multichip_mesh, 1, SHORT_MAX_SEQ, seed=10102)

    with _CollectiveSpy() as spy:
        tt_out = decoder.prefill_forward(to_device_hidden(multichip_mesh, hidden), page_table=page_table, user_id=0)
    spy.assert_ring_topology()
    assert spy.reductions(ccl_mode) == 2
    assert_pcc(
        f"multichip prefill[{kind}] ccl_mode={ccl_mode} seq_len={seq_len}",
        expected,
        first_device(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    token = R.synthetic_hidden_states(1, 1, seed=10103)
    ref = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([seq_len]))
    current_pos, rope_pos_ids = decode_position_tensors(multichip_mesh, torch.tensor([seq_len]))
    with _CollectiveSpy() as spy:
        tt_dec = decoder.decode_forward(
            to_device_hidden(multichip_mesh, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
    spy.assert_ring_topology()
    assert spy.reductions(ccl_mode) == 2
    assert_pcc(
        f"multichip decode[{kind}] ccl_mode={ccl_mode} pos={seq_len}",
        ref,
        first_device(tt_dec).reshape(1, 1, -1),
    )
    ttnn.deallocate(tt_dec)
