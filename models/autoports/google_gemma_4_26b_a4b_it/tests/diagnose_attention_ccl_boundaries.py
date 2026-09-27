# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Locate the first changing boundary in duplicate TP4 decoder replays.

This is a diagnostic companion to run_multichip_decoder. Retaining intermediate
tensors changes allocator lifetimes; use --output-only as the matching control.
Host reads and comparisons occur only after blocking trace execution.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    GeneralizedRouter,
    OptimizedAttention,
    OptimizedExperts,
)


class BoundaryRouter(GeneralizedRouter):
    """Retain the selected router's logits, gate outputs and expert IDs."""

    def __call__(self, x, normalized=None):
        if x.shape[-2] != 1:
            return self.original(x, normalized=normalized)
        retain = self.diagnostic_decoder.retain
        source = self.original.source
        router = source.source
        if normalized is None:
            normalized = self.original.normalize(x, source.epsilon)
        scaled = ttnn.mul(
            normalized,
            source.scale,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, router.scalar_root_size)],
        )
        if scaled.is_sharded():
            scaled = ttnn.to_memory_config(scaled, ttnn.L1_MEMORY_CONFIG)
        scores = retain(
            "router_scores",
            ttnn.linear(
                scaled,
                self.projection_weight,
                dtype=ttnn.float32,
                program_config=self.projection_program,
                compute_kernel_config=self.projection_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
        )
        scores = retain("router_centered_scores", ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True)))
        rounded = retain("router_rounded_scores", ttnn.typecast(scores, ttnn.bfloat16))
        padded = ttnn.pad(rounded, [(0, 0), (0, 0), (0, 0), (0, 128)], float("-inf"))
        face = ttnn.to_memory_config(ttnn.reshape(padded, (1, 16, 16)), self.memory)
        values, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            face,
            bias_tensor=self.bias,
            input_indices_tensor=self.indices,
            output_tensor=self.output,
            output_indices_tensor=self.output_indices,
            eps=1e-20,
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=8,
            output_softmax=True,
            grouped=False,
        )
        values = ttnn.to_memory_config(values, ttnn.L1_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.L1_MEMORY_CONFIG)
        values = retain("router_gate_values", ttnn.view(values[:, 0, :8], (1, 1, 1, 8)))
        indices = retain("router_gate_indices", ttnn.view(indices[:, 0, :8], (1, 1, 1, 8)))
        if self.retain_decode_indices:
            self.last_decode_indices = indices
        routing = ttnn.scatter(ttnn.zeros_like(rounded), dim=-1, index=indices, src=values)
        return ttnn.mul(routing, router.per_expert_scale)


class BoundaryExperts(OptimizedExperts):
    def record_address(self, name, value):
        callback = getattr(self, "address_callback", None)
        if callback is not None:
            callback("expert_" + name, value)

    def retain(self, name, value):
        self.record_address(name, value)
        selected = getattr(self, "retained_expert_names", None)
        if selected is not None and name not in selected:
            return value
        self.boundaries[name] = value
        callback = getattr(self, "boundary_callback", None)
        return callback("expert_" + name, value) if callback is not None else value

    def _chunk(self, x, routing, decode):
        assert decode and self.indexed_router is not None and not self.expert_split
        self.record_address("chunk_entry_input", x)
        self.record_address("previous_chunk", getattr(self, "boundaries", {}))
        self.boundaries = {}
        if self.decode_activation_dtype is not None:
            x = self.retain("activation_cast", ttnn.typecast(x, self.decode_activation_dtype))
        sparsity = self.retain("routing_row_major", ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT))
        common = dict(
            sparsity=sparsity,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.decode_compute,
        )
        indices = self.indexed_router.decode_indices()
        self.record_address("indices", indices)
        common["indices"] = indices
        slots = self.config.top_k
        selected = self.retain(
            "mix_weights_row_major", ttnn.gather(sparsity, dim=-1, index=indices, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        mix_weights = self.retain("mix_weights_tiled", ttnn.to_layout(selected, ttnn.TILE_LAYOUT))
        gu = self.retain("gate_up_raw", ttnn.sparse_matmul(x, self.gate_up, program_config=self.gate_config, **common))
        gu = ttnn.reshape(gu, (1, slots, 1, 2 * self.width))
        self.record_address("gate_up_reshaped", gu)
        gate, up = gu[..., : self.width], gu[..., self.width :]
        self.retain("gate", gate)
        self.retain("up", up)
        if self.decode_gelu_activations is not None:
            hidden = self.retain("hidden", ttnn.mul(gate, up, input_tensor_a_activations=self.decode_gelu_activations))
        else:
            hidden = self.retain("hidden", ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up))
        down = self.retain(
            "down_raw",
            ttnn.sparse_matmul(hidden, self.down, program_config=self.down_config, is_input_a_sparse=True, **common),
        )
        down = ttnn.reshape(down, (1, slots, 1, self.config.hidden_size))
        return self.retain(
            "output",
            ttnn.matmul(
                mix_weights,
                self.retain("down_permuted", ttnn.permute(down, (0, 2, 1, 3))),
                dtype=ttnn.bfloat16,
                memory_config=self.mix_memory,
                program_config=self.mix_program,
                compute_kernel_config=self.mix_compute,
            ),
        )


class BoundaryDecoder(MultichipDecoder):
    """Same replicated forward operations, retaining their tensor handles."""

    def record_address(self, name, value):
        phase = getattr(self, "address_phase", None)
        if phase is None:
            return
        if isinstance(value, dict):
            # Recursion keeps temporary handles in this short-lived frame,
            # not in the model forward after its dictionaries are cleared.
            for child_name, child in value.items():
                self.record_address(name + ":" + child_name, child)
            return
        self.address_records.append(
            dict(
                phase=phase,
                name=name,
                address=int(value.buffer_address()),
                shape=list(value.shape),
                padded_shape=list(value.padded_shape),
                dtype=str(value.dtype),
                memory=str(value.memory_config()),
            )
        )

    def retain(self, name, value):
        self.record_address(name, value)
        if name == "output" or any(name.startswith(prefix) for prefix in self.retained_prefixes):
            self.boundaries[name] = value
        return value

    def install_attention_probe(self, dtype):
        attention = self.layer.self_attn

        def mesh_allreduce(value):
            scattered = self.retain(
                "attention_rs",
                ttnn.experimental.reduce_scatter_minimal_async(
                    value,
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                    num_links=self.ccl.num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl.topology,
                    cluster_axis=1,
                    barrier_semaphore=self.ccl.get_barrier_semaphore(),
                ),
            )
            return self.retain(
                "attention_ag",
                ttnn.experimental.all_gather_async(
                    scattered,
                    dim=3,
                    cluster_axis=1,
                    mesh_device=self.ccl.mesh_device,
                    topology=self.ccl.topology,
                    multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                    num_links=self.ccl.num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=self.ccl.get_barrier_semaphore(),
                ),
            )

        def allreduce(value):
            value = self.retain("attention_dram", ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))
            return mesh_allreduce(value)

        def reduce(value):
            return allreduce(self.retain("attention_cast", ttnn.typecast(value, dtype)))

        def project(value, decode):
            self.retain("attention_sdpa", value)
            return reduce(self.retain("attention_wo_fp32", OptimizedAttention.project(attention, value, decode)))

        attention.project = project
        router = self.layer.moe.router
        assert isinstance(router, GeneralizedRouter) and router.direct_projection and router.center_logits
        router.__class__ = BoundaryRouter
        router.diagnostic_decoder = self

    def post_attention_norm(self, value, epsilon):
        if value.shape[-2] != 1:
            return self.retain("post_attention_norm", self.normalize(value, epsilon, self.post_attention_norm_weight))
        # Match OptimizedDecoder.normalize, including its decode sharding path.
        memory, program = self.layer.post_feedforward_layernorm_1._sharded_cfg
        value = self.retain(
            "attention_sharded_fp32",
            ttnn.to_memory_config(self.retain("attention_promoted_fp32", ttnn.typecast(value, ttnn.float32)), memory),
        )
        result = self.retain(
            "post_attention_norm_sharded",
            ttnn.rms_norm(
                value,
                epsilon=epsilon,
                program_config=program,
                compute_kernel_config=self.layer.self_attn.compute,
                memory_config=memory,
            ),
        )
        result = self.retain("post_attention_norm_unweighted", ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG))
        return self.retain(
            "post_attention_norm",
            ttnn.mul(result, self.post_attention_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG),
        )

    def _forward(self, x, **attention_kwargs):
        # This matches MultichipDecoder._forward's replicated/grouped/fused path.
        self.record_address("forward_entry_input", x)
        self.record_address("previous_forward", getattr(self, "boundaries", {}))
        self.boundaries = {}
        eps = self.config.rms_norm_eps
        normed = self.retain("input_norm", self.normalize(x, eps, self.input_norm_weight))
        attention = self.layer.self_attn(normed, **attention_kwargs)
        residual = self.retain(
            "residual",
            ttnn.add(ttnn.typecast(x, ttnn.float32), self.post_attention_norm(attention, eps)),
        )
        normalized = self.retain("residual_norm", self.normalize(residual, eps))
        routes = self.retain("routes", self.layer.moe.router(residual, normalized=normalized))
        if x.shape[-2] == 1:
            self.retain("route_indices", self.layer.moe.router.decode_indices())
        expert_input = self.retain("expert_input", ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16))
        routed = self.retain("routed_local", self.layer.moe.experts(expert_input, routes))
        shared_input = self.retain("shared_input", ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16))
        shared = self.retain("shared_local", self.layer.shared_mlp(shared_input, reduce_output=False))
        shared, routed = self._reduce_moe_pair(shared, routed)
        self.retain("shared_reduced", shared)
        self.retain("routed_reduced", routed)
        return self.retain("output", self._fused_tail(residual, shared, routed, x.shape[-2] <= 32))


def snapshot(tensors, include_padding=False):
    snapshots = {
        name: [ttnn.to_torch(part).float().clone() for part in ttnn.get_device_tensors(tensor)]
        for name, tensor in tensors.items()
    }
    if include_padding:
        for name, tensor in tensors.items():
            if tensor.layout != ttnn.TILE_LAYOUT or tuple(tensor.shape) == tuple(tensor.padded_shape):
                continue
            # Two Shape arguments select the metadata-only view overload. This
            # runs after blocking replay and exposes physical padding to the read.
            shape = ttnn.Shape(tensor.padded_shape)
            physical = ttnn.reshape(tensor, shape, shape)
            snapshots[name + ":physical"] = [
                ttnn.to_torch(part).float().clone() for part in ttnn.get_device_tensors(physical)
            ]
    return snapshots


def compare(reference, actual):
    comparisons = {}
    for name, wanted_parts in reference.items():
        # Unused physical rows may contain stable NaNs. Compare their FP32
        # snapshot bits, while keeping logical NaNs a correctness failure.
        physical = name.endswith(":physical")

        def comparable(tensor):
            return tensor.view(torch.int32) if physical else tensor

        ranks = []
        for rank, (wanted, got) in enumerate(zip(wanted_parts, actual[name])):
            changed = comparable(wanted) != comparable(got)
            finite_pair = torch.isfinite(wanted) & torch.isfinite(got)
            ranks.append(
                dict(
                    rank=rank,
                    equal=torch.equal(comparable(wanted), comparable(got)),
                    changed_elements=int(changed.sum()),
                    reference_nonfinite=int((~torch.isfinite(wanted)).sum()),
                    actual_nonfinite=int((~torch.isfinite(got)).sum()),
                    finite_max_abs_diff=(
                        float((wanted[finite_pair] - got[finite_pair]).abs().max()) if finite_pair.any() else None
                    ),
                    first_changed_index=changed.nonzero()[0].tolist() if changed.any() else None,
                )
            )
        comparisons[name] = dict(
            equal=all(item["equal"] for item in ranks),
            comparison="snapshot_bits" if physical else "logical_values",
            reference_replicas_equal=all(
                torch.equal(comparable(wanted_parts[0]), comparable(p)) for p in wanted_parts[1:]
            ),
            actual_replicas_equal=all(
                torch.equal(comparable(actual[name][0]), comparable(p)) for p in actual[name][1:]
            ),
            ranks=ranks,
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--duplicates", type=int, default=1, help="Duplicate comparisons per input/position.")
    parser.add_argument("--dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat8_b")
    parser.add_argument("--output-only", action="store_true")
    parser.add_argument(
        "--check-replicated-boundaries",
        action="store_true",
        help="Stop at a boundary expected to be equal across ranks.",
    )
    parser.add_argument("--expert-boundaries", action="store_true", help="Retain indexed expert internal operations.")
    parser.add_argument(
        "--expert-retain",
        nargs="+",
        choices=[
            "none",
            "gate_up_raw",
            "gate",
            "up",
            "hidden",
            "down_raw",
            "down_permuted",
            "mix_weights_row_major",
            "mix_weights_tiled",
        ],
        help="Restrict expert-internal lifetime retention to these tensors.",
    )
    parser.add_argument("--gate-k-block", type=int, choices=[44, 88], help="Override only indexed gate/up K blocking.")
    parser.add_argument(
        "--gate-n-tiles", type=int, choices=[1, 2], default=1, help="Gate/up worker N tiles; N2 uses grid 6x1."
    )
    parser.add_argument(
        "--expert-input-dram",
        action="store_true",
        help="Copy indexed expert input tiles to DRAM before the original chunk.",
    )
    parser.add_argument(
        "--router-core-x",
        type=int,
        choices=[0, 1],
        default=None,
        help="Place the one-core generalized gate on this x coordinate.",
    )
    parser.add_argument(
        "--log-addresses", action="store_true", help="Record host buffer metadata during warmup/capture."
    )
    parser.add_argument(
        "--retain",
        nargs="+",
        choices=["all", "none", "attention", "router", "moe"],
        default=["all"],
        help="Boundary groups kept alive by the diagnostic subclass; output is always retained.",
    )
    parser.add_argument(
        "--read-output-only", action="store_true", help="Retain chosen boundaries but read only output."
    )
    parser.add_argument("--replay-delay-ms", type=float, default=0, help="Host delay before the duplicate replay.")
    parser.add_argument("--read-padding", action="store_true", help="Also snapshot metadata views of physical tiles.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.length <= 4096 or not 1 <= args.steps <= 128:
        parser.error("Fixture supports length1..4096 and steps1..128")
    if not 1 <= args.duplicates <= 128:
        parser.error("duplicates must be between 1 and 128")
    if not 0 <= args.replay_delay_ms <= 1000:
        parser.error("Replay delay must be between0 and1000ms")
    if len(args.retain) > 1 and ("all" in args.retain or "none" in args.retain):
        parser.error("all and none must be the only retention selection")
    if args.expert_boundaries and args.output_only:
        parser.error("expert boundaries require the diagnostic decoder")
    if args.expert_retain and not args.expert_boundaries:
        parser.error("expert-retain requires expert-boundaries")
    if args.expert_retain and "none" in args.expert_retain and len(args.expert_retain) != 1:
        parser.error("none must be the only expert retention selection")
    if args.log_addresses and args.output_only:
        parser.error("address logging requires the diagnostic subclass")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(__file__).parents[1]
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    fixture = torch.load(root / f"doc/optimized_decoder/actual_text_layer{args.layer}_4096_128.pt", weights_only=True)
    x = fixture["prefill"][:, : args.length]
    decode = fixture["decode"][:, : args.steps]
    extent = (args.length + args.steps + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    block = 32
    pages = extent // block
    table = torch.randperm(pages, dtype=torch.int32)[None]
    result = dict(
        command=sys.argv,
        runtime_sha256=hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dtype=args.dtype,
        experts_sha256=hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        expert_boundaries=args.expert_boundaries,
        expert_retained_names=args.expert_retain,
        gate_k_block=args.gate_k_block,
        gate_n_tiles=args.gate_n_tiles,
        log_addresses=args.log_addresses,
        expert_input_dram=args.expert_input_dram,
        router_core_x=args.router_core_x,
        layer=args.layer,
        output_only=args.output_only,
        check_replicated_boundaries=args.check_replicated_boundaries,
        retained_groups=args.retain if not args.output_only else [],
        read_output_only=args.read_output_only,
        replay_delay_ms=args.replay_delay_ms,
        read_padding=args.read_padding,
        length=args.length,
        steps=args.steps,
        completed_steps=0,
        duplicates=args.duplicates,
        completed_duplicate_checks=0,
        passed=False,
        reference_read_host_ms=[],
        physical_variation_steps=[],
        limitation="Retained tensor handles change allocator lifetimes; this run does not measure performance.",
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16777216)
    trace_id = None
    try:
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

        cls = MultichipDecoder if args.output_only else BoundaryDecoder
        dtype = getattr(ttnn, args.dtype)
        decoder = cls.from_state_dict(
            hf.state_dict(),
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            fused_tail=True,
            hybrid_experts=True,
            optimized_shared=True,
            shared_geometry=1,
            grouped_moe_reduce=True,
            attention_ccl_dtype=dtype,
        )
        attention = decoder.layer.self_attn
        if args.router_core_x is not None:
            router = decoder.layer.moe.router
            core = ttnn.CoreCoord(args.router_core_x, 0)
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
            shard = ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
            router.memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
            for field in ("bias", "indices", "output", "output_indices"):
                setattr(router, field, ttnn.to_memory_config(getattr(router, field), router.memory))
        result["router_memory"] = str(decoder.layer.moe.router.memory)
        if not args.output_only:
            prefixes = {
                "all": ("",),
                "none": (),
                "attention": ("input_norm", "attention_", "post_attention_", "residual"),
                "router": ("router_", "routes", "route_indices"),
                "moe": ("expert_", "routed_", "shared_"),
            }
            decoder.retained_prefixes = tuple(prefix for group in args.retain for prefix in prefixes[group])
            decoder.install_attention_probe(dtype)
            decoder.address_records = []
            decoder.address_phase = None
        experts = decoder.layer.moe.experts.decode
        if args.gate_k_block is not None:
            experts.gate_config.in0_block_w = args.gate_k_block
        if args.gate_n_tiles == 2:
            experts.gate_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(6, 1),
                in0_block_w=experts.gate_config.in0_block_w,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=1,
                out_block_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=False,
                mcast_in0=True,
            )
        if args.expert_boundaries:
            assert type(experts) is OptimizedExperts
            experts.__class__ = BoundaryExperts
            experts.boundary_callback = decoder.retain
            experts.retained_expert_names = [] if args.expert_retain == ["none"] else args.expert_retain
            if args.log_addresses:
                experts.address_callback = decoder.record_address
        if args.expert_input_dram:
            original_chunk = experts._chunk

            def chunk_with_dram_input(value, routing, decode):
                return original_chunk(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), routing, decode)

            experts._chunk = chunk_with_dram_input
        result["expert_gate_program"] = str(experts.gate_config)
        result["expert_down_program"] = str(experts.down_config)
        projection = attention.source.weights.wqkv
        projection = getattr(projection, "projection", projection)
        for owner, field in ((attention, "output_compute"), (projection, "decode_compute")):
            original = getattr(owner, field)
            setattr(
                owner,
                field,
                ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=original.math_approx_mode,
                    fp32_dest_acc_en=original.fp32_dest_acc_en,
                    packer_l1_acc=original.packer_l1_acc,
                    dst_full_sync_en=original.dst_full_sync_en,
                ),
            )
        cfg = attention.config
        cache = [
            upload(torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim), ttnn.bfloat8_b) for _ in range(2)
        ]
        page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rope = tuple(upload(t.unsqueeze(0)) for t in (cos, sin))
        rope_decode = tuple(upload(t.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))
        dx = upload(x.unsqueeze(0))
        # The original runner performs one validation and three timed prefills.
        for _ in range(4):
            with device_only():
                y = decoder.prefill_forward(dx, rope_mats=rope, page_table=page_table, kv_cache=cache)
            ttnn.synchronize_device(mesh)
            del y
        token = upload(decode[:, :1].unsqueeze(0))
        pos = upload(torch.tensor([[args.length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cache_pos = upload(torch.tensor([args.length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def forward():
            with device_only():
                return decoder.decode_forward(
                    token,
                    rope_mats=rope_decode,
                    current_pos=pos,
                    cache_pos=cache_pos,
                    page_table=page_table,
                    kv_cache=cache,
                )

        for warmup in range(2):
            if args.log_addresses:
                decoder.address_phase = f"warm{warmup + 1}"
            y = forward()
        if args.log_addresses:
            decoder.address_phase = "capture"
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        y = forward()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        if args.log_addresses:
            result["buffer_addresses"] = decoder.address_records
        retained_tensors = {"output": y} if args.output_only else decoder.boundaries
        tensors = {"output": y} if args.read_output_only else retained_tensors
        result["boundaries"] = {
            name: dict(
                logical_shape=list(tensor.shape),
                padded_shape=list(tensor.padded_shape),
                dtype=str(tensor.dtype),
                memory=str(tensor.memory_config()),
            )
            for name, tensor in retained_tensors.items()
        }
        for step in range(args.steps):
            for value, dst, dtype, layout in (
                (decode[:, step : step + 1].unsqueeze(0), token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (torch.tensor([[args.length + step]], dtype=torch.int32), pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (torch.tensor([args.length + step], dtype=torch.int32), cache_pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            ):
                host = ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(host, dst)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            read_start = time.perf_counter()
            reference = snapshot(tensors, args.read_padding)
            result["reference_read_host_ms"].append((time.perf_counter() - read_start) * 1000)
            for duplicate in range(args.duplicates):
                if args.replay_delay_ms:
                    time.sleep(args.replay_delay_ms / 1000)
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                actual = snapshot(tensors, args.read_padding)
                comparisons = compare(reference, actual)
                result["completed_duplicate_checks"] += 1
                # Padding is diagnostic evidence, not part of the logical output
                # contract. Continue harmless physical-only changes until the
                # original logical replay failure is observed.
                physical_differences = [
                    name for name, item in comparisons.items() if name.endswith(":physical") and not item["equal"]
                ]
                if physical_differences:
                    result["physical_variation_steps"].append(dict(step=step, boundaries=physical_differences))
                first_difference = next(
                    (
                        name
                        for name, item in comparisons.items()
                        if not name.endswith(":physical") and not item["equal"]
                    ),
                    None,
                )
                replicated = {
                    "input_norm",
                    "attention_ag",
                    "attention_promoted_fp32",
                    "attention_sharded_fp32",
                    "post_attention_norm_sharded",
                    "post_attention_norm_unweighted",
                    "post_attention_norm",
                    "residual",
                    "residual_norm",
                    "routes",
                    "route_indices",
                    "expert_input",
                    "shared_input",
                    "shared_reduced",
                    "routed_reduced",
                    "output",
                }
                first_replica_boundary = (
                    next(
                        (
                            name
                            for name, item in comparisons.items()
                            if not name.endswith(":physical")
                            and (name in replicated or name.startswith("router_"))
                            and not (item["reference_replicas_equal"] and item["actual_replicas_equal"])
                        ),
                        None,
                    )
                    if args.check_replicated_boundaries
                    else None
                )
                output = comparisons["output"]
                nonfinite = any(item["reference_nonfinite"] or item["actual_nonfinite"] for item in output["ranks"])
                replicas_differ = not (output["reference_replicas_equal"] and output["actual_replicas_equal"])
                result["completed_steps"] = step + 1
                if first_difference is not None or first_replica_boundary is not None or replicas_differ or nonfinite:
                    result["failure"] = dict(
                        tp=4,
                        step=step,
                        duplicate=duplicate,
                        absolute_position=args.length + step,
                        first_different_boundary=first_difference,
                        first_replica_contract_difference=first_replica_boundary,
                        first_different_physical_boundary=physical_differences[0] if physical_differences else None,
                        output_replicas_differ=replicas_differ,
                        output_nonfinite=nonfinite,
                        comparisons=comparisons,
                    )
                    tensor_path = args.output.with_suffix(".pt")
                    torch.save(dict(reference=reference, actual=actual, input=decode[:, step : step + 1]), tensor_path)
                    result["failure_tensors"] = str(tensor_path)
                    print("REPLAY_BOUNDARY_FAILURE", json.dumps(result["failure"]), flush=True)
                    break
            if "failure" in result:
                break
        else:
            result["passed"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("boundaries", "failure")}), flush=True)
        assert result["passed"], result.get("failure")
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
