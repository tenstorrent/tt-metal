# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fake MoE traffic helpers — reproduce the wire-level behavior of a
deepseek_v3_b1 MoE decoder stage without invoking MoeOp.

See FAKE_MOE_PLAN.md for the design.

Public surface (grows across phases):

  Phase 2 (standalone CCL chain):
    - run_fake_moe_chain(mesh_device, sender_coord, ...)
        Chains broadcast → all_reduce → a2a_dispatch → a2a_combine →
        reduce-to-one on a (4, 2) per-rank submesh, returns the per-device
        outputs and a torch reference for verification.

  Phase 3 (pipeline-stage integration, not yet implemented):
    - FakeMoeDecoderStage  — drop-in replacement for MoEDecoderStage
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

# Default shapes mirror what the real demo uses on a per-pod configuration.
# See FAKE_MOE_PLAN.md §"What MoE actually does" for the source of these.
DEFAULT_BATCH = 32
DEFAULT_SEQ_LEN = 2
DEFAULT_HIDDEN = 1024
DEFAULT_K = 4  # select_experts_k; smaller than the real K=8 for faster smoke tests
DEFAULT_EXPERTS_PER_DEVICE = 4  # so total experts = 4 * 8 = 32 on a (4,2) submesh


@dataclass
class FakeMoeShapes:
    """Sizes used by the fake MoE chain. Defaults are smoke-test sized,
    not production-sized (see DEFAULT_* above).
    """

    batch: int = DEFAULT_BATCH
    seq_len: int = DEFAULT_SEQ_LEN
    hidden: int = DEFAULT_HIDDEN
    select_experts_k: int = DEFAULT_K
    experts_per_device: int = DEFAULT_EXPERTS_PER_DEVICE

    @property
    def activation_shape(self) -> list[int]:
        # `[1, batch, seq_len, hidden]` — the canonical token activation in
        # the demo (leading 1 is the placeholder for "device" dim that
        # CCL ops shard along).
        return [1, self.batch, self.seq_len, self.hidden]

    def total_experts(self, mesh_devices: int) -> int:
        return self.experts_per_device * mesh_devices


# ---------------------------------------------------------------------------
# Phase 2 — standalone CCL chain
# ---------------------------------------------------------------------------
#
# The chain reproduces steps 2–7 of an MoE iteration (see FAKE_MOE_PLAN.md
# §"What MoE actually does"):
#
#     bcast (1 → 8) → all_reduce (axis 0) → a2a_dispatch (axis 1) →
#         a2a_combine (axis 1) → reduce-to-one substitute (8 → 1)
#
# Each step has a torch reference; the chain returns the per-device tensors
# at the end so the test can verify the final state.


def _setup_sub_devices(mesh_device):
    """Create a worker sub-device covering the full compute grid.
    Required by ttnn.broadcast (and most CCL ops) to drive workers.
    """
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    return worker_sub_device_id, sub_device_stall_group, sub_device_manager


def _teardown_sub_devices(mesh_device, sub_device_manager):
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(sub_device_manager)


# ---------------------------------------------------------------------------
# Step builders. Each takes an input tt tensor on the (4, 2) submesh and
# returns the output tt tensor after that step. Per-device input must be
# the same across all devices on the source axis (for steps that broadcast
# or all-reduce).
# ---------------------------------------------------------------------------


def step_all_reduce(tt_input, mesh_device, cluster_axis: int, mem_config, num_links, topology):
    """All-reduce sum along `cluster_axis`. Input on every device of that
    axis is summed and replicated back to every device.
    """
    return ttnn.all_reduce(
        tt_input,
        cluster_axis=cluster_axis,
        topology=topology,
        memory_config=mem_config,
        num_links=num_links,
    )


def step_reduce_to_one_substitute(tt_input, mesh_device, mem_config, num_links, topology):
    """Substitute for the demo's ReduceToOneB1 (3-level tree, 8 → 1
    device). We chain two ttnn.all_reduce ops (axis 0 then axis 1) so every
    device ends up with the full sum across the (4, 2) submesh; the caller
    then reads from one device to obtain the "reduced-to-one" result.

    Compared to the real reduce-to-one tree this uses ~2× the bandwidth,
    but the final result on the chosen root device is identical.
    """
    tt_partial = ttnn.all_reduce(
        tt_input,
        cluster_axis=0,
        topology=topology,
        memory_config=mem_config,
        num_links=num_links,
    )
    tt_full = ttnn.all_reduce(
        tt_partial,
        cluster_axis=1,
        topology=topology,
        memory_config=mem_config,
        num_links=num_links,
    )
    return tt_full


# ---------------------------------------------------------------------------
# Phase 3 — FakeMoeDecoderStage
# ---------------------------------------------------------------------------
#
# A pipeline-stage drop-in for `MoEDecoderStage` whose `setup` and
# `launch_compute` do not invoke the broken `MoeOp` synthetic-weight path.
# Implemented as `PassthroughStage(ACTIVATION)` so the pipeline framework
# (sockets, FIFO sizing, kernel scaffolding) runs end-to-end while the MoE
# slot does no compute. This matches the "wire-level" intent of the fake-MoE
# (Phase 2 already validated MoE-shaped CCL traffic in isolation).
#
# We keep this as a thin wrapper rather than a copy of MoEDecoderStage's
# socket scaffolding (full activation FIFO, MOE_SENDER_CORE entry, exit upstream
# cores, attention + moe + reduce semaphores) because:
#
#   * The existing `PassthroughStage(PassthroughPayload.ACTIVATION)` already
#     sizes its sockets to ACTIVATION_FIFO_SIZE / ACTIVATION_PAGE_SIZE_BYTES,
#     which is what every decoder stage uses.
#   * Wiring CCL primitives (ttnn.all_reduce, etc.) into the pipeline-block
#     socket lifecycle would require integrating them with kernel-side
#     send/recv (the way DecoderBlock and LMHeadSampling do) — that is
#     non-trivial and out of scope for the fake-MoE.
#
# The standalone tests in test_fake_moe_traffic.py validate the CCL chain
# directly on a (4, 2) submesh, which is a stronger guarantee for the CCL
# stack than embedding it inside the pipeline.


def make_fake_moe_decoder_stage_factory():
    """Return a stage factory (mesh_device → StageKind) that produces a
    PassthroughStage(ACTIVATION) — used to swap out MoEDecoderStage in the
    single-pod pipeline configuration. Imported lazily so this helper module
    has no demo-package import-time side effects.
    """
    from models.demos.deepseek_v3_b1.demo.stage import PassthroughPayload, PassthroughStage

    return lambda mesh_device: PassthroughStage(PassthroughPayload.ACTIVATION)


# ---------------------------------------------------------------------------
# Phase 4 — FakeLMHeadStage
# ---------------------------------------------------------------------------
#
# Drop-in for LMHeadStage at slot 14 in the single-pod pipeline. The real
# LMHead stage:
#   * allocates substantial L1 / DRAM tensors and global semaphores in setup()
#   * dispatches LMHeadSampling.op() in launch_compute(), which builds device
#     kernels that block waiting for activation input from the upstream socket
#     (and never exit cleanly when no input is ever driven)
#
# In the fake-MoE pipeline test we never drive tokens through, so the LMHead
# kernels deadlock — and pipeline.terminate() can't tear them down,
# stranding rank 14 and hanging mpirun (verified 2026-04-30).
#
# This FakeLMHeadStage keeps the *socket signature* of LMHeadStage
# (ACTIVATION in → TOKEN out, so adjacent stages 13 → 14 → 15 still match
# their FIFO sizes) but does no work in setup or launch_compute. The
# PipelineBlock scaffolding handles the socket lifecycle, and there are no
# compute kernels left to deadlock teardown.


def make_fake_lm_head_stage_factory():
    """Return a stage factory producing a no-compute LMHead stub.

    The returned stage has the same upstream/downstream socket sizes as
    LMHeadStage (so it slots in at index 14 without breaking adjacent
    FIFO contracts) but its setup and launch_compute are empty.
    """
    from models.demos.deepseek_v3_b1.demo.stage import (
        ACTIVATION_FIFO_SIZE,
        ACTIVATION_PAGE_SIZE_BYTES,
        PIPELINE_CORE_COORD,
        TOKEN_FIFO_SIZE,
        TOKEN_PAGE_SIZE_BYTES,
        StageKind,
    )
    from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock

    class _FakeLMHeadStage(StageKind):
        def create_pipeline_block(self, ctx):
            return PipelineBlock(
                ctx.mesh_device,
                PIPELINE_CORE_COORD,
                upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
                downstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
                upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
                downstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            )

        # setup and launch_compute fall back to StageKind's no-op defaults.

    return lambda mesh_device: _FakeLMHeadStage()
