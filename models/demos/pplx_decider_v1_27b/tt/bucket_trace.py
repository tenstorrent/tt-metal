# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One prefill trace per prompt-length bucket, with the last-token select, final norm and readout fused in.

A prompt is padded with token 0 up to the smallest bucket that holds it. The padding sits after the real tokens, so
the causal attention and GDN outputs at the last real token are unaffected and no GDN mask is needed. The GDN
recurrent state and conv carry are zeroed inside the trace, and each replay overwrites the KV blocks it uses, so
requests need no other device work than staging the tokens and the one-hot row select.
"""

import os
import time
from dataclasses import dataclass

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import load_readout
from models.tt_transformers.tt.common import Mode, get_block_size

# 4096 is the largest bucket: longer prompts use the 2048-chunk trace path.
BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)
# The readout matmul output is padded to a multiple of the tile width.
READOUT_PAD = 256
PAD_TOKEN_ID = 0
# Passes per bucket before capture: the first compiles every program, the second takes the steady-state path.
WARM_PASSES = 2


def bucket_sizes():
    """BUCKETS, or the comma-separated PPLX_BUCKETS, ascending."""
    text = os.environ.get("PPLX_BUCKETS")
    sizes = tuple(int(b) for b in text.split(",")) if text else BUCKETS
    return tuple(sorted(set(sizes)))


@dataclass
class _Bucket:
    size: int
    tokens: ttnn.Tensor
    chunk_start_idx: ttnn.Tensor
    full_page_table: ttnn.Tensor
    chunk_page_table: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    sel: ttnn.Tensor
    trace_id: int = None
    out: ttnn.Tensor = None


class BucketPrefillTraces:
    def __init__(self, buckets=None):
        self.sizes = tuple(sorted(set(buckets))) if buckets else bucket_sizes()
        self.buckets = {}
        self.mesh = None
        self.readout_w = None
        self.compute_cfg = None
        self.prepared = False

    @property
    def max_bucket(self):
        return self.sizes[-1]

    def capture(self, model, mesh, page_table):
        """prepare + record. With other traces on the device, call prepare before capturing them."""
        self.prepare(model, mesh, page_table)
        self.record(model, mesh)

    def prepare(self, model, mesh, page_table):
        """Allocate every persistent buffer and run every bucket's trace body eagerly, before any capture.

        Nothing may be allocated or compiled once a trace is parked: its replay overwrites that memory. So this must
        run before the first begin_trace_capture on the device, including other traces' (e.g. the chunk trace)."""
        if self.prepared:
            return
        self.mesh = mesh
        assert model.num_devices > 1, "bucket traces use the TP prefill chunk forward"
        assert model._paged_kv_caches is not None, "Call allocate_kv_caches first"
        block_size = get_block_size(model._paged_kv_caches)
        for size in self.sizes:
            assert size % 128 == 0 and size % block_size == 0, f"bucket {size} must be a multiple of 128"
            assert page_table.shape[-1] >= size // block_size, f"page table too short for bucket {size}"
        args = model.args
        rep = ttnn.ReplicateTensorToMesh(mesh)

        # The vision-splice buffers are allocated once, at the first size asked for, and every consumer slices them
        # down: allocate them for the largest bucket (or chunk) before anything else does.
        vision_len = max(self.max_bucket, model._chunked_chunk_size or 2048)
        model._alloc_vision_merge_buffers(mesh, vision_len)
        assert model._vis_buf.shape[-2] >= vision_len, "vision buffers were allocated for a smaller size"

        # Readout weight [1, 1, dim, READOUT_PAD]: the readout rows transposed and zero padded.
        token_ids = args.decision_config.token_ids
        readout = load_readout(args.CKPT_DIR, len(token_ids), args.dim)
        weight = torch.zeros(args.dim, READOUT_PAD, dtype=torch.bfloat16)
        weight[:, : len(token_ids)] = readout.T.to(torch.bfloat16)
        self.readout_w = ttnn.from_torch(
            weight.reshape(1, 1, args.dim, READOUT_PAD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

        # cos/sin are the plain 1D RoPE at positions 0..size-1: drop any staged multimodal table first.
        model._build_request_rope(torch.zeros(1, 1, dtype=torch.long), None)
        pt = page_table.to(torch.int32)

        def upload(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep)

        # Every bucket's inputs exist before the first warm pass, so no later allocation can land in memory a parked
        # trace writes. Tokens and sel are rewritten per request, the rest are constant.
        for size in self.sizes:
            cos_t, sin_t = model._rope_tp_cos_sin_torch(0, size)
            self.buckets[size] = _Bucket(
                size=size,
                tokens=upload(torch.zeros(1, size, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_start_idx=upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                full_page_table=upload(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_page_table=upload(pt[:, : size // block_size].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                cos=upload(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                sin=upload(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                sel=upload(torch.zeros(1, 1, 1, size, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            )

        # Warm every bucket before the first capture: later buckets' programs and lazily allocated buffers must not
        # appear behind an earlier bucket's parked trace.
        start = time.perf_counter()
        for size, bucket in self.buckets.items():
            t0 = time.perf_counter()
            for _ in range(WARM_PASSES):
                out = self._body(model, mesh, bucket)
                ttnn.synchronize_device(mesh)
                ttnn.deallocate(out)
            logger.info(f"Bucket {size} warmed in {time.perf_counter() - t0:.2f}s")
        logger.info(f"All {len(self.sizes)} buckets warmed in {time.perf_counter() - start:.2f}s")
        self.prepared = True

    def record(self, model, mesh):
        """Capture one trace per bucket on prepare's buffers; allocates nothing outside the capture."""
        assert self.prepared, "Call prepare first"
        start = time.perf_counter()
        for size, bucket in self.buckets.items():
            if bucket.trace_id is not None:
                ttnn.release_trace(mesh, bucket.trace_id)
            t0 = time.perf_counter()
            # The output is read right after each replay, before any other trace can run.
            with trace_allocation_tracker.corruptible_allocation_scope(mesh):
                bucket.trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                bucket.out = self._body(model, mesh, bucket)
                ttnn.end_trace_capture(mesh, bucket.trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            logger.info(f"Bucket {size} prefill trace captured in {time.perf_counter() - t0:.2f}s{_trace_region(mesh)}")
        logger.info(f"All {len(self.sizes)} bucket prefill traces captured in {time.perf_counter() - start:.2f}s")
        # The warm and capture passes advanced the GDN state; start the next eager prefill from zero.
        model._reset_gdn_state_for_new_sequence()

    def _body(self, model, mesh, bucket):
        """GDN state reset, prefill, last-token select, final norm and readout. Returns [1, 1, 32, READOUT_PAD]."""
        # Zero what the forward reads: the recurrent state and the cross-chunk conv carry. conv_states are the
        # decode state, which this prefill does not feed (gdn_capture_state=False).
        for layer in model.layers:
            if not layer.is_full_attention:
                gdn = layer.attention
                ttnn.copy(gdn._zero_rec, gdn.rec_state)
                ttnn.copy(gdn._zero_conv_carry, gdn.conv_carry)
        hidden = model._forward_prefill_chunk_tp(
            bucket.tokens,
            bucket.cos,
            bucket.sin,
            bucket.chunk_start_idx,
            bucket.full_page_table,
            bucket.chunk_page_table,
            seq_len=bucket.size,
            gdn_capture_state=False,
        )
        assert hidden.dtype == bucket.sel.dtype, f"hidden is {hidden.dtype}, one-hot select is {bucket.sel.dtype}"
        # Same select + norm as Qwen36Model._masked_bucket_logits_tp; the norm output is the gathered full-dim row.
        x = ttnn.matmul(bucket.sel, hidden)  # [1, 1, 1, dim/tp]
        ttnn.deallocate(hidden)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = model.norm(x, mode=Mode.PREFILL)  # [1, 1, 1, dim]
        out = ttnn.matmul(x, self.readout_w, compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)
        ttnn.deallocate(x)
        return out

    def run(self, model, input_ids):
        """Readout logits fp32 [num_tokens] of the last prompt token, or None if it is longer than the largest bucket."""
        length = len(input_ids)
        size = next((s for s in self.sizes if s >= length), None)
        if size is None:
            return None
        bucket = self.buckets[size]
        # PPLX_BUCKET_EAGER=1 runs the trace body eagerly instead of replaying the trace (for Tracy device profiling,
        # where trace replays of the GDN kernels fail). Same persistent inputs, same readout.
        eager = os.environ.get("PPLX_BUCKET_EAGER", "0") == "1"
        assert eager or bucket.trace_id is not None, "Call record first"
        mesh = self.mesh
        rep = ttnn.ReplicateTensorToMesh(mesh)
        tokens = torch.full((1, size), PAD_TOKEN_ID, dtype=torch.int32)
        tokens[0, :length] = torch.tensor(input_ids, dtype=torch.int32)
        sel = torch.zeros(1, 1, 1, size, dtype=torch.bfloat16)
        sel[0, 0, 0, length - 1] = 1.0
        tokens_host = ttnn.from_torch(
            tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
        )
        sel_host = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(tokens_host, bucket.tokens)
        ttnn.copy_host_to_device_tensor(sel_host, bucket.sel)
        if eager:
            # Same as a warm pass: the body allocates a fresh output, which is freed after the read.
            eager_out = self._body(model, mesh, bucket)
            ttnn.synchronize_device(mesh)
            out = ttnn.to_torch(ttnn.get_device_tensors(eager_out)[0])
            ttnn.deallocate(eager_out)
        else:
            ttnn.execute_trace(mesh, bucket.trace_id, cq_id=0, blocking=False)
            # The readout is replicated; device 0's copy is enough. The blocking read also keeps the host inputs
            # alive until the replay is done.
            out = ttnn.to_torch(ttnn.get_device_tensors(bucket.out)[0])
        n = len(model.args.decision_config.token_ids)
        return out.reshape(-1)[:n].float()

    def release(self):
        for bucket in self.buckets.values():
            if bucket.trace_id is not None:
                ttnn.release_trace(self.mesh, bucket.trace_id)
                bucket.trace_id = None
        self.buckets.clear()
        self.prepared = False


def _trace_region(mesh):
    """Trace region usage for the capture log; empty when unavailable."""
    try:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return f", trace region {view.total_bytes_allocated_per_bank * view.num_banks / 1e6:.1f} MB"
    except Exception:  # noqa: BLE001 - purely informational
        return ""
