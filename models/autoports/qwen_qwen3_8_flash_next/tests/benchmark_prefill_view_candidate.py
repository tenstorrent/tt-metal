# SPDX-License-Identifier: Apache-2.0
"""Unqualified, prefill-only decoder view for a fixed-batch serving owner.

This module does not open devices or execute on import. Construction allocates
private execution scratch; execution intentionally updates the owner's public
request state. Callers must serialize owner/view work on CQ0 and prepare before
capture. No model weights, context capacity, sampling, or server slots change.
"""

import copy

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    AttentionOutputProjection,
    MeshOps,
    MultichipDecoder,
)


class _PrefillOps(MeshOps):
    def norm(self, value, weight, eps, *, reuse_output=False):
        # B32 uses FP32 distributed statistics even for a one-token suffix.
        # The ordinary B1 singleton optimization instead communicates BF16.
        stats = ttnn.rms_norm_pre_all_gather(value, dtype=ttnn.float32, compute_kernel_config=self.compute)
        gathered = self.gather(stats)
        if reuse_output:
            del stats
        normalized = ttnn.rms_norm_post_all_gather(value, gathered, epsilon=eps, compute_kernel_config=self.compute)
        return ttnn.multiply(normalized, weight, output_tensor=normalized if reuse_output else None)


class _RowOutputProjection(AttentionOutputProjection):
    def __call__(self, x, *, concatenated=False):
        # Preserve the owner's row-sharded weight and reduction. B32 serving
        # intentionally has no B1 column-sharded singleton weight allocation.
        return self.ops.scatter_reduce(self.ops.linear(x, self.row_weight, compute_kernel_config=self.compute))


class PrefillDecoderView(MultichipDecoder):
    def decode_forward(self, *args, **kwargs):
        raise RuntimeError("Benchmark candidate is prefill-only; decode belongs to the fixed-batch owner")


def _clone_with_ops(owner, ops, cls=None):
    clone = copy.copy(owner)
    if cls is not None:
        clone.__class__ = cls
    clone.ops = ops
    return clone


def _private_zero(reference, shape, mesh):
    return ttnn.zeros(
        shape,
        dtype=reference.dtype,
        layout=reference.layout,
        memory_config=reference.memory_config(),
        device=mesh,
    )


def make_prefill_decoder_view(owner, *, chunk_size=128, lengths=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Clone execution metadata only; share weights and slot/page-owned state.

    The supplied owner must already carry the selected precision policy. Public
    slot IDs retain their original range, including nonzero/remapped slots. No
    checkpoint loading, policy application, cache reset, or sampling occurs.
    """
    if owner.batch_size <= 1:
        raise ValueError("The candidate requires a fixed-batch owner with more than one row")
    if not 1 <= chunk_size <= 512:
        raise ValueError("Candidate chunks must lie within [1,512]")
    if owner.kind != "linear_attention" and not owner.attention.paged_pooled:
        raise ValueError("QSA view requires the serving paged-pool contract")

    ops = copy.copy(owner.ops)
    ops.__class__ = _PrefillOps
    ops.gather_outputs = {}
    ops.reduce_outputs = {}
    ops.large_prefill = False
    ops.l1_prefill = False
    view = _clone_with_ops(owner, ops, PrefillDecoderView)
    view.batch_size = 1
    view.chunk_size = chunk_size
    view.lengths = {}
    view.attn_hyper = _clone_with_ops(owner.attn_hyper, ops)
    view.mlp_hyper = _clone_with_ops(owner.mlp_hyper, ops)
    view.mlp = _clone_with_ops(owner.mlp, ops)
    view.mlp.capacity = _private_zero(owner.mlp.capacity, tuple(owner.mlp.capacity.shape), owner.mesh)
    attention = _clone_with_ops(owner.attention, ops)
    view.attention = attention
    attention.batch_size = 1
    attention.chunk_size = chunk_size
    attention.output_projection = _clone_with_ops(owner.attention.output_projection, ops, _RowOutputProjection)

    if owner.kind == "linear_attention":
        for name in ("recurrent_state", "zero_recurrent", "conv_state", "zero_conv"):
            original = getattr(owner.attention, name)
            setattr(attention, name, _private_zero(original, (1, *tuple(original.shape)[1:]), owner.mesh))
    else:
        attention._decode_phase = False
        # These legacy dense-update buffers are not consumed by ParallelQSA's
        # paged native path; the canonical B1 constructor removes them too.
        for name in ("_physical_update_indices", "_index_core"):
            if hasattr(attention, name):
                delattr(attention, name)
        attention.chunk_offsets = ops.upload(torch.arange(chunk_size).float().reshape(1, 1, -1, 1), ttnn.float32)
        attention.indexer_prefix = ops.upload(torch.zeros(1, 4, 31, 128))
        attention.indexer_weights = ops.upload(torch.ones(1, 1, chunk_size + 32, 4))
        attention.pooled_keys = _private_zero(
            owner.attention.pooled_keys, tuple(owner.attention.pooled_keys.shape), owner.mesh
        )
    if owner.ple is not None:
        view.ple = _clone_with_ops(owner.ple, ops)
        view.ple.workspaces = {}
        view.ple.flat_norm_weights = {}
        # Immutable host norm parts and device weights remain shared. The
        # owner's ple_cache is deliberately retained for request-slot handoff.
    view.prepare(lengths)
    assert_prefill_view_contract(owner, view)
    return view


def assert_prefill_view_contract(owner, view):
    """Host metadata assertions; this is not a numerical/device validation."""
    assert view is not owner and view.batch_size == 1 and owner.batch_size > 1
    assert view.context == owner.context
    assert view.attention.max_requests == owner.attention.max_requests
    assert view.ops is not owner.ops and view.ops.precision_ccl == owner.ops.precision_ccl
    assert view.ops.ccl is owner.ops.ccl  # one serialized CQ0 semaphore stream
    assert view.mlp.capacity is not owner.mlp.capacity
    assert view.attn_hyper.read_weight is owner.attn_hyper.read_weight
    assert view.mlp.router is owner.mlp.router
    assert view.attention.output_projection.row_weight is owner.attention.output_projection.row_weight
    shared = (
        ("recurrent_bank", "conv_bank", "packed_weight", "out_weight")
        if owner.kind == "linear_attention"
        else ("packed_cache", "raw_key_cache", "pooled_cache", "packed_projection", "wo", "cos", "sin")
    )
    private = (
        ("recurrent_state", "conv_state", "zero_recurrent", "zero_conv")
        if owner.kind == "linear_attention"
        else ("pooled_keys", "indexer_prefix", "indexer_weights", "chunk_offsets")
    )
    for name in shared:
        assert getattr(view.attention, name) is getattr(owner.attention, name), name
    for name in private:
        assert getattr(view.attention, name) is not getattr(owner.attention, name), name
    if owner.ple is not None:
        assert view.ple is not owner.ple and view.ple_cache is owner.ple_cache
        assert view.ple.tables is owner.ple.tables
        assert view.ple.workspaces is not owner.ple.workspaces
    return {"shared_attention": list(shared), "private_attention": list(private), "context": view.context}
