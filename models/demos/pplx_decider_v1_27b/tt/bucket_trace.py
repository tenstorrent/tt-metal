# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One prefill trace per prompt-length bucket, with the last-token select, final norm and readout fused in.

A prompt is padded with token 0 up to the smallest bucket that holds it. The padding sits after the real tokens, so
the causal attention and GDN outputs at the last real token are unaffected and no GDN mask is needed. The GDN
recurrent state and conv carry are never read or written (the GDN layers start from zero and drop their final state;
PPLX_GDN_NO_CARRY=0 zeroes the persistent state inside the trace instead), and each replay overwrites the KV blocks it
uses, so requests need no other device work than staging the tokens and the one-hot row select. On a 2D mesh,
sequence parallelism (PPLX_SP, default on; PPLX_SP=0 turns it off) splits each bucket's tokens over the mesh rows instead
of replicating them on every row; only buckets of at least PPLX_SP_MIN (default 256, so every bucket) tokens use it, the
smaller ones stay replicated.
"""

import os
import time
from dataclasses import dataclass

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import load_readout
from models.tt_transformers.tt.ccl import tt_all_gather
from models.tt_transformers.tt.common import Mode, get_block_size

# 4096 is the largest bucket: on a 1xN mesh longer prompts take the 2048-chunk trace path; a 2D mesh has none, so
# PplxDecider caps its max_seq_len at the largest bucket (max_seq_len_for).
BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)
# The readout matmul output is padded to a multiple of the tile width.
READOUT_PAD = 256
PAD_TOKEN_ID = 0
# Passes per bucket before capture: the first compiles every program, the second takes the steady-state path.
WARM_PASSES = 2


def _gdn_no_carry():
    """True (default): the prefill trace skips the GDN state zeroing and writeback; PPLX_GDN_NO_CARRY=0 restores them."""
    return os.environ.get("PPLX_GDN_NO_CARRY", "1") == "1"


def _sp_enabled(mesh):
    """On a 2D mesh PPLX_SP (default "1") splits each bucket's tokens over the mesh rows; PPLX_SP=0 turns it off.

    Default on: replicated rows only waste the extra rows. 1xN meshes are unaffected (tpc.is_2d_mesh is False)."""
    return tpc.is_2d_mesh(mesh) and os.environ.get("PPLX_SP", "1") == "1"


def _sp_min():
    """PPLX_SP_MIN (default 256): with PPLX_SP=1 only buckets of at least this many tokens use sequence parallelism.

    Measured on Galaxy with the short-length projection path (PPLX_SMALL_M_MAX=128): SP is faster from 256 up
    (256: 39.2 vs 43.1 ms replicated; 512: 45.2 vs 54.8 ms), so by default every bucket splits over the rows."""
    return int(os.environ.get("PPLX_SP_MIN", "256"))


def bucket_sizes():
    """BUCKETS, or the comma-separated PPLX_BUCKETS, ascending."""
    text = os.environ.get("PPLX_BUCKETS")
    sizes = tuple(int(b) for b in text.split(",")) if text else BUCKETS
    return tuple(sorted(set(sizes)))


def bucket_sizes_for(max_seq_len, capacity):
    """The bucket_sizes() a model with this max_seq_len and KV capacity (tokens) can use: every bucket below
    max_seq_len plus the first one at or above it, none beyond capacity. A max_seq_len past the largest bucket
    leaves the longer prompts to the 1xN chunk trace (max_seq_len_for caps it on a 2D mesh)."""
    sizes = [s for s in bucket_sizes() if s <= capacity]
    above = [s for s in sizes if s >= max_seq_len]
    return tuple(s for s in sizes if s < max_seq_len) + tuple(above[:1])


def max_seq_len_for(max_seq_len, two_d):
    """The longest prompt a mesh can serve: a 2D mesh has no chunked prefill, so its largest bucket is the limit."""
    return min(max_seq_len, bucket_sizes()[-1]) if two_d else max_seq_len


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
    sp: bool = False
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
        self.sp = None
        self.rows = 1

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

        sp_on = _sp_enabled(mesh)
        sp_sizes = [s for s in self.sizes if s >= _sp_min()] if sp_on else []
        if sp_sizes:
            # Imported here: the module is only needed (and may only exist) when sequence parallelism is on.
            from models.demos.blackhole.qwen36.tt.sp import SeqParallel

            self.sp = SeqParallel(mesh, args, model.tt_ccl)
            self.rows = mesh.shape[0]
            for size in sp_sizes:
                assert size % (self.rows * 32) == 0, f"bucket {size} must split into 32-aligned row slices"
            # Per-local-length constants, built before any warm pass or capture.
            self.sp.prepare([s // self.rows for s in sp_sizes])
            rep_sizes = [s for s in self.sizes if s not in sp_sizes]
            logger.info(f"Sequence parallel over {self.rows} rows for buckets {sp_sizes}; replicated for {rep_sizes}")
        elif sp_on:
            logger.info(f"PPLX_SP=1 but no bucket reaches PPLX_SP_MIN={_sp_min()}; all buckets replicated")

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

        def upload(t, dtype, layout, mapper=rep):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)

        # Every bucket's inputs exist before the first warm pass, so no later allocation can land in memory a parked
        # trace writes. Tokens and sel are rewritten per request, the rest are constant.
        for size in self.sizes:
            cos_t, sin_t = model._rope_tp_cos_sin_torch(0, size)
            bucket_sp = self.sp is not None and size >= _sp_min()
            if not bucket_sp:
                tok_map = cos_map = sel_map = rep
                start_t, start_map = torch.zeros(1, dtype=torch.int32), rep
            else:
                # Each mesh row gets its own contiguous size/rows slice of the tokens, RoPE rows and select; its
                # chunk_start_idx is the slice's first position. The page tables stay replicated: every row fills
                # the whole sequence's KV.
                tok_map, cos_map, sel_map = self.sp.rows_mapper(1), self.sp.rows_mapper(2), self.sp.rows_mapper(3)
                start_t = torch.arange(self.rows, dtype=torch.int32) * (size // self.rows)
                start_map = self.sp.rows_mapper(0)
            self.buckets[size] = _Bucket(
                size=size,
                tokens=upload(torch.zeros(1, size, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, tok_map),
                chunk_start_idx=upload(start_t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, start_map),
                full_page_table=upload(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_page_table=upload(pt[:, : size // block_size].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                cos=upload(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT, cos_map),
                sin=upload(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT, cos_map),
                sel=upload(torch.zeros(1, 1, 1, size, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, sel_map),
                sp=bucket_sp,
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
        no_carry = _gdn_no_carry()
        if not no_carry:
            # Zero what the forward reads: the recurrent state and the cross-chunk conv carry. conv_states are the
            # decode state, which this prefill does not feed (gdn_capture_state=False).
            for layer in model.layers:
                if not layer.is_full_attention:
                    gdn = layer.attention
                    ttnn.copy(gdn._zero_rec, gdn.rec_state)
                    ttnn.copy(gdn._zero_conv_carry, gdn.conv_carry)
        # no_carry: the GDN layers start from zero state and drop their final state, so none of the above is needed.
        if not bucket.sp:
            seq_len, sp_kwargs = bucket.size, {}
        else:
            # Sequence parallel: each mesh row runs its own size/rows tokens and needs the zero-state GDN start.
            assert no_carry, "PPLX_SP=1 requires PPLX_GDN_NO_CARRY=1"
            seq_len, sp_kwargs = bucket.size // self.rows, {"sp": self.sp}
        hidden = model._forward_prefill_chunk_tp(
            bucket.tokens,
            bucket.cos,
            bucket.sin,
            bucket.chunk_start_idx,
            bucket.full_page_table,
            bucket.chunk_page_table,
            seq_len=seq_len,
            gdn_capture_state=False,
            gdn_carry_state=not no_carry,
            **sp_kwargs,
        )
        assert hidden.dtype == bucket.sel.dtype, f"hidden is {hidden.dtype}, one-hot select is {bucket.sel.dtype}"
        # Same select + norm as Qwen36Model._masked_bucket_logits_tp; the norm output is the gathered full-dim row.
        x = ttnn.matmul(bucket.sel, hidden)  # [1, 1, 1, dim/tp]
        ttnn.deallocate(hidden)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if tpc.is_2d_mesh(mesh):
            # The distributed final norm gathers over the whole mesh on 2D; gather the TP axis and norm the full row.
            x = tt_all_gather(
                x,
                mesh,
                model.tt_ccl,
                cluster_axis=1,
                dim=3,
                topology=model.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            norm = model.norm.norm
            x = ttnn.rms_norm(
                x, epsilon=norm.eps, weight=norm.weight, compute_kernel_config=norm.compute_kernel_config_hifi2
            )
        else:
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
        tokens_map, sel_map = (self.sp.rows_mapper(1), self.sp.rows_mapper(3)) if bucket.sp else (rep, rep)
        tokens = torch.full((1, size), PAD_TOKEN_ID, dtype=torch.int32)
        tokens[0, :length] = torch.tensor(input_ids, dtype=torch.int32)
        sel = torch.zeros(1, 1, 1, size, dtype=torch.bfloat16)
        sel[0, 0, 0, length - 1] = 1.0
        tokens_host = ttnn.from_torch(
            tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=tokens_map
        )
        sel_host = ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=sel_map)
        ttnn.copy_host_to_device_tensor(tokens_host, bucket.tokens)
        ttnn.copy_host_to_device_tensor(sel_host, bucket.sel)
        # The readout is replicated, so device 0's copy is enough. Under SP only the mesh row holding the last real
        # token has a meaningful readout (the others select nothing); device order is row-major, index row * cols.
        read_idx = 0
        if bucket.sp:
            read_idx = ((length - 1) // (size // self.rows)) * mesh.shape[1]
        if eager:
            # Same as a warm pass: the body allocates a fresh output, which is freed after the read.
            eager_out = self._body(model, mesh, bucket)
            ttnn.synchronize_device(mesh)
            out = ttnn.to_torch(ttnn.get_device_tensors(eager_out)[read_idx])
            ttnn.deallocate(eager_out)
        else:
            ttnn.execute_trace(mesh, bucket.trace_id, cq_id=0, blocking=False)
            # The blocking read also keeps the host inputs alive until the replay is done.
            out = ttnn.to_torch(ttnn.get_device_tensors(bucket.out)[read_idx])
        n = len(model.args.decision_config.token_ids)
        return out.reshape(-1)[:n].float()

    def release(self):
        for bucket in self.buckets.values():
            if bucket.trace_id is not None:
                ttnn.release_trace(self.mesh, bucket.trace_id)
                bucket.trace_id = None
        self.buckets.clear()
        self.prepared = False
        self.sp = None
        self.rows = 1


def _trace_region(mesh):
    """Trace region usage for the capture log; empty when unavailable."""
    try:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return f", trace region {view.total_bytes_allocated_per_bank * view.num_banks / 1e6:.1f} MB"
    except Exception:  # noqa: BLE001 - purely informational
        return ""
