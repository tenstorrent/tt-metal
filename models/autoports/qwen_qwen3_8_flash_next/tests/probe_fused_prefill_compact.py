# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-capacity compact union of all top10 routes in one token tile.

Prototype only. Uses10*T compact slots, unique selected IDs in ascending order,
then zero-weight duplicate padding. Compares full MoE outputs with expanded512.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import SparseMoe, WeightLoader, linear
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    FusedPreciseProjection,
    FusedSparseMoe,
    fused_compact_union_routes,
    fused_packed_silu,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import aten_router_topk

SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const uint32_t tokens = get_arg_val<uint32_t>(5), slots = tokens * 10;
    const uint32_t stride = ((slots*2+63)/64)*32; // aligned uint16 row stride
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(0), 4096);
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(1), 2048);
    const auto out_ids = TensorAccessor(oa, get_arg_val<uint32_t>(2), slots*2);
    const auto out_scores = TensorAccessor(wa, get_arg_val<uint32_t>(3), slots*2);
    const auto reduce_ids = TensorAccessor(ra, get_arg_val<uint32_t>(4), slots*2);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(ids.get_noc_addr(0), scratch, 4096);
    noc_async_read(scores.get_noc_addr(0), scratch + 4096, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ip = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 4096);
    volatile tt_l1_ptr uint16_t* op = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 6144);
    volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 7168);
    volatile tt_l1_ptr uint16_t* wp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 8192);
    for (uint32_t e=0;e<512;++e) map[e]=0;
    for (uint32_t t=0;t<tokens;++t) {
        uint32_t row = t<16 ? t*16 : 512+(t-16)*16;
        for (uint32_t j=0;j<10;++j) { ASSERT(ip[row+j]<512);map[ip[row+j]]=1; }
    }
    uint32_t used=0;
    for (uint32_t e=0;e<512;++e) if(map[e]) { op[used]=e;map[e]=++used; }
    ASSERT(used<=slots);
    for(uint32_t j=used;j<slots;++j)op[j]=0;
    for(uint32_t j=0;j<tokens*stride;++j)wp[j]=0;
    for(uint32_t t=0;t<tokens;++t) {
        uint32_t row=t<16?t*16:512+(t-16)*16;
        for(uint32_t j=0;j<10;++j)wp[t*stride+map[ip[row+j]]-1]=sp[row+j];
        noc_async_write(scratch+6144,reduce_ids.get_noc_addr(t),slots*2);
        noc_async_write(scratch+8192+t*stride*2,out_scores.get_noc_addr(t),slots*2);
    }
    noc_async_write(scratch+6144,out_ids.get_noc_addr(0),slots*2);
    noc_async_write_barrier();
}
"""


def union_routes(indices, scores):
    tokens = indices.shape[-2]
    slots = 10 * tokens
    assert tuple(indices.shape) == tuple(scores.shape) == (1, 1, tokens, 10) and 1 <= tokens <= 32
    device = indices.device()
    ids = ttnn.empty((1, 1, 1, slots), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    weights = ttnn.empty((tokens, 1, 1, slots), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reduction = ttnn.empty((tokens, 1, 1, slots), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tensors = [indices, scores, ids, weights, reduction]
    core = ttnn.CoreCoord(0, 0)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in tensors] + [tokens]
    ct = []
    for t in tensors:
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=32768,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=32768)],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op(tensors, program)
    return ids, weights, reduction


class UnionMoe(FusedSparseMoe):
    def __call__(self, x):
        tokens = x.shape[-2]
        if tokens > 32:
            raise ValueError("SparseMoe physical chunk is at most one tile; caller must chunk")
        logits = self.router_logits(x)
        scores, indices = aten_router_topk(logits)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        compact = True
        if compact:
            compact_ids, compact_scores, reduction_ids = union_routes(indices, ttnn.typecast(scores, ttnn.bfloat16))
            active = self.compact_sparsity
        else:
            dense = ttnn.scatter(
                ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16)),
                dim=3,
                index=indices,
                src=ttnn.typecast(scores, ttnn.bfloat16),
            )
            # Sparse matmul chooses experts per token tile. Union the selected experts
            # across rows, retaining the independent per-token scores for reduction.
            active = ttnn.to_layout(ttnn.max(dense, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(
            sparsity=active,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
        )
        if compact:
            kwargs["indices"] = compact_ids
        if hasattr(self, "expert_packed"):
            program = self.program.get_decode_gate_up_config(tokens, 2 * self.intermediate, k=self.width)
            packed = ttnn.sparse_matmul(x, self.expert_packed, program_config=program, **kwargs)
            if getattr(self, "fuse_packed_activation", False):
                activated = fused_packed_silu(packed)
            else:
                gate, up = packed[..., : self.intermediate], packed[..., self.intermediate :]
            del packed
        else:
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            if getattr(self, "matmul_expert_activation", False):
                program.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            gate = ttnn.sparse_matmul(x, self.gate, program_config=program, **kwargs)
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            up = ttnn.sparse_matmul(x, self.up, program_config=program, **kwargs)
        if not (hasattr(self, "expert_packed") and getattr(self, "fuse_packed_activation", False)):
            if getattr(self, "matmul_expert_activation", False):
                activated = ttnn.multiply(gate, up, output_tensor=gate if self.reuse_private_buffers else None)
            elif self.fuse_expert_activation:
                activated = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    output_tensor=gate if self.reuse_private_buffers else None,
                )
            elif self.reuse_private_buffers:
                # The expert gate has no other consumer. Preserve both BF16
                # boundaries while avoiding two private 20 MiB temporaries.
                activated = ttnn.multiply(ttnn.silu(gate, output_tensor=gate), up, output_tensor=gate)
            else:
                activated = ttnn.multiply(ttnn.silu(gate), up)
            # These private intermediates have no further consumers. Release their
            # references before the 80 MiB per-expert output allocation on PLE.
            del gate, up
        down = ttnn.sparse_matmul(
            activated,
            self.down,
            is_input_a_sparse=True,
            program_config=self.program.get_decode_down_config(tokens, self.width, k=self.intermediate),
            **kwargs,
        )
        del activated
        if hasattr(self, "reduce_indices"):
            scores_rm = (
                compact_scores if compact else ttnn.to_layout(ttnn.permute(dense, (2, 0, 1, 3)), ttnn.ROW_MAJOR_LAYOUT)
            )
            reduction_ids = reduction_ids if compact else self.reduce_indices[tokens]
            routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                down,
                reduction_ids,
                self.reduce_mapping,
                reduce_dim=1,
                split_size=self.width,
                cluster_axis=0,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
        else:
            weights = ttnn.permute(ttnn.typecast(dense, ttnn.bfloat16), (0, 3, 2, 1))
            # The private per-expert output has no other consumer. Weight it in
            # place so the resident PLE table need not coexist with an 80 MiB copy.
            ttnn.multiply(down, weights, output_tensor=down)
            if self.reuse_private_buffers and tokens < 32:
                # Generic sum clones a padded input before zeroing its padding.
                # This weighted down buffer is private, so zero it in place and
                # expose its existing physical height to the same reduction.
                # Explicit padded shapes make both reshapes metadata-only views.
                ttnn.fill_implicit_tile_padding(down, 0.0)
                down = ttnn.reshape(down, (1, self.experts, 32, self.width), (1, self.experts, 32, self.width))
                routed = ttnn.sum(down, dim=1, keepdim=True)
                routed = ttnn.reshape(routed, (1, 1, tokens, self.width), (1, 1, 32, self.width))
            else:
                routed = ttnn.sum(down, dim=1, keepdim=True)
        del down
        if hasattr(self, "shared_packed"):
            packed = linear(x, self.shared_packed, self.compute)
            gate = packed[..., : self.shared_width]
            up = packed[..., self.shared_width : 2 * self.shared_width]
            route = packed[..., 2 * self.shared_width :]
            del packed
        else:
            gate = (
                ttnn.linear(
                    x,
                    self.shared_gate,
                    activation="silu",
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if getattr(self, "matmul_shared_activation", False)
                else linear(x, self.shared_gate, self.compute)
            )
            up = linear(x, self.shared_up, self.compute)
            route = linear(x, self.shared_route, self.compute)
        shared = (
            ttnn.multiply(gate, up)
            if getattr(self, "matmul_shared_activation", False)
            else (
                ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                if self.fuse_shared_activation
                else ttnn.multiply(ttnn.silu(gate), up)
            )
        )
        shared = linear(shared, self.shared_down, self.compute)
        shared = ttnn.multiply(shared, ttnn.sigmoid(route))
        return ttnn.add(routed, shared)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clean-timing", action="store_true")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert-dtype", default="bfloat16")
    p.add_argument(
        "--repeated-rows", action="store_true", help="Identical routes across rows stress zero-score padding"
    )
    p.add_argument("--integrated", action="store_true", help="Exercise production FusedSparseMoe union routing")
    p.add_argument("--counts", type=int, nargs="+", default=[1, 2, 7, 16, 17, 31, 32])
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(618)
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    weights = load_weights(a.snapshot, a.layer, "mlp.")
    assert bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING) != a.clean_timing
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    report = dict(
        rows=[],
        passed=False,
        layer=a.layer,
        expert_dtype=a.expert_dtype,
        clean_timing=a.clean_timing,
        integrated=a.integrated,
        repeated_rows=a.repeated_rows,
        watcher=os.environ.get("TT_METAL_WATCHER"),
    )
    try:
        loader = WeightLoader(weights, mesh)
        original = SparseMoe(loader, cfg, expert_dtype=getattr(ttnn, a.expert_dtype))
        owner = FusedSparseMoe.__new__(FusedSparseMoe)
        owner.__dict__.update(original.__dict__)
        owner.compact_experts = True
        owner.fuse_expert_activation = owner.fuse_shared_activation = owner.fuse_packed_activation = True
        owner.pack_projections(shared=True, experts=True, packed_gate_cores=(10, 4))
        owner.prepare_fused_reduce(mesh)
        owner.precise_router = FusedPreciseProjection(owner.precise_router, owner.compute, 64, widen_in_unpack=True)
        if a.integrated:
            owner.early_packed_activation = True
            owner.fuse_moe_cleanup = True
        candidate_type = FusedSparseMoe if a.integrated else UnionMoe
        candidate = candidate_type.__new__(candidate_type)
        candidate.__dict__.update(owner.__dict__)
        if a.integrated:
            candidate.compact_prefill_limit = 32
        route_packet = fused_compact_union_routes if a.integrated else union_routes
        for count in a.counts:
            host_x = (torch.randn(1, 1, 1 if a.repeated_rows else count, 2560) * 0.02).bfloat16()
            if a.repeated_rows:
                host_x = host_x.expand(1, 1, count, 2560).contiguous()
            x = loader.tensor(host_x)
            logits = owner.router_logits(x)
            scores, indices = aten_router_topk(logits)
            scores = ttnn.typecast(
                ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=owner.compute), ttnn.bfloat16
            )
            gather, packed_scores, reduction = route_packet(indices, scores)
            host_ids = ttnn.to_torch(indices).reshape(count, 10).long()
            host_scores = ttnn.to_torch(scores).reshape(count, 10)
            chosen = sorted(set(host_ids.flatten().tolist()))
            slots = 10 * count
            wanted_ids = torch.tensor(chosen + [0] * (slots - len(chosen)), dtype=torch.int32)
            wanted_scores = torch.zeros(count, slots, dtype=torch.bfloat16)
            mapping = {e: i for i, e in enumerate(chosen)}
            for t in range(count):
                for j in range(10):
                    wanted_scores[t, mapping[int(host_ids[t, j])]] = host_scores[t, j]
            packet_exact = (
                torch.equal(ttnn.to_torch(gather).flatten().int(), wanted_ids)
                and torch.equal(ttnn.to_torch(reduction).reshape(count, slots).int(), wanted_ids.expand(count, slots))
                and torch.equal(ttnn.to_torch(packed_scores).reshape(count, slots), wanted_scores)
            )
            assert packet_exact
            del logits, scores, indices, gather, packed_scores, reduction
            expected = ttnn.to_torch(owner(x))
            actual = ttnn.to_torch(candidate(x))
            pcc = float(torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1])
            row = dict(
                count=count,
                packet_exact=packet_exact,
                unique_experts=len(chosen),
                slots=slots,
                exact=torch.equal(expected, actual),
                pcc=pcc,
                max_abs=float((expected.float() - actual.float()).abs().max()),
                times={},
            )
            for name, model in [("expanded", owner), ("compact_union", candidate)]:
                model(x)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = model(x)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                    trace_allocation_tracker.acknowledge_corruptible(out)
                samples = []
                for _ in range(8):
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - start) * 1000)
                assert torch.equal(ttnn.to_torch(out), expected if name == "expanded" else actual)
                row["times"][name] = statistics.median(samples[2:])
                ttnn.release_trace(mesh, trace)
                trace = None
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = all(r["pcc"] >= 0.995 for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
