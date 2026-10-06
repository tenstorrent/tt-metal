# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe direct-M sparse expert batching on recorded real-layer activations."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder, PackedExperts, geglu
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config


class DirectMExperts(PackedExperts):
    """Test-only peer batching; keep expert weights and arithmetic unchanged."""

    def __init__(self, source, batch_tokens):
        # Reuse the same packed buffers, avoiding another copy of model weights.
        self.__dict__.update(source.__dict__)
        self.batch_tokens = batch_tokens
        self.gate_config = _build_sparse_matmul_config(batch_tokens, 2 * self.width)
        self.down_config = _build_sparse_matmul_config(batch_tokens, self.config.hidden_size)
        self.observed_shapes = {}

    def _prefill_chunk(self, x, routing):
        cfg = self.config
        length = x.shape[-2]
        assert length == self.batch_tokens
        gu = ttnn.sparse_matmul(
            x,
            self.gate_up,
            sparsity=self.sparsity,
            nnz=cfg.num_experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self.gate_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
        )
        self.observed_shapes["gate_up"] = list(gu.shape)
        gu = ttnn.reshape(gu, (1, cfg.num_experts, length, 2 * self.width))
        gate, up = gu[..., : self.width], gu[..., self.width :]
        hidden = (
            geglu(gate, up) if self.fused_gelu else ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up)
        )
        down = ttnn.sparse_matmul(
            hidden,
            self.down,
            sparsity=self.sparsity,
            nnz=cfg.num_experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self.down_config,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
        )
        self.observed_shapes["down"] = list(down.shape)
        down = ttnn.reshape(down, (1, cfg.num_experts, length, cfg.hidden_size))
        weighted = ttnn.mul(down, ttnn.permute(routing, (0, 3, 2, 1)))
        return ttnn.reshape(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]), (1, 1, length, cfg.hidden_size))

    def __call__(self, x, routing):
        assert x.shape[-2] % self.batch_tokens == 0
        outputs = [
            self._prefill_chunk(
                x[:, :, start : start + self.batch_tokens, :],
                routing[:, :, start : start + self.batch_tokens, :],
            )
            for start in range(0, x.shape[-2], self.batch_tokens)
        ]
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).parents[1] / "doc/functional_decoder/headline_inputs.pt",
    )
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, nargs="+", default=[32, 64, 128, 256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timing-samples", type=int, default=3)
    parser.add_argument("--warmup-replays", type=int, default=3)
    args = parser.parse_args()
    if args.length < 32 or args.length % 32:
        parser.error("length must be a positive multiple of 32")
    if any(value < 32 or value % 32 or args.length % value for value in args.batch_tokens):
        parser.error("batch-tokens must be multiples of 32 which divide length")
    if min(args.repeats, args.timing_samples, args.warmup_replays) < 1:
        parser.error("repeats, timing-samples and warmup-replays must be positive")

    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    cfg._attn_implementation = "eager"
    recorded = torch.load(args.inputs, map_location="cpu", weights_only=True)
    source_x = recorded["x"]
    if source_x.ndim != 3 or source_x.shape[0] != 1 or source_x.shape[-1] != cfg.hidden_size:
        parser.error("recorded x must have shape [1, S, hidden_size]")
    if source_x.shape[1] < args.length:
        parser.error("recorded x is shorter than length")
    host_x = source_x[:, : args.length].contiguous()
    hf = load_layer(cfg, args.layer, True)
    layer_type = cfg.layer_types[args.layer]
    extent = (args.length + 1023) // 1024 * 1024
    rope = Gemma4TextRotaryEmbedding(cfg)
    with torch.no_grad():
        cos, sin = rope(host_x, torch.arange(extent)[None], layer_type=layer_type)

    report = dict(
        layer=args.layer,
        layer_type=layer_type,
        revision=REVISION,
        real_weights=True,
        recorded_inputs=str(args.inputs),
        input_source="expert inputs and routes captured from selected FusedDecoder prefill",
        length=args.length,
        batch_tokens=args.batch_tokens,
        repeats=args.repeats,
        timing_samples=args.timing_samples,
        warmup_replays=args.warmup_replays,
        pcc_threshold=0.995,
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FusedDecoder.from_state_dict(hf.state_dict(), hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)
        selected = decoder.layer.moe.experts
        baseline = copy.copy(selected)
        baseline.prefill_batch_tokens = 32
        assert isinstance(baseline, PackedExperts)
        report["capture_fusion"] = decoder.fusion
        report["expert_fused_gelu"] = baseline.fused_gelu
        captures = []

        class CaptureExperts:
            def __call__(self, x, routing):
                captures.append((ttnn.clone(x), ttnn.clone(routing)))
                return selected(x, routing)

        decoder.layer.moe.experts = CaptureExperts()

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout)

        xt = upload(host_x[None])
        tables = tuple(upload(value[None]) for value in (cos, sin))
        pages = extent // 32
        page_table = upload(torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        attention_cfg = decoder.layer.self_attn.config
        cache_shape = (pages, attention_cfg.num_key_value_heads, 32, attention_cfg.head_dim)
        caches = tuple(upload(torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2))
        with device_only():
            output = decoder.prefill_forward(xt, rope_mats=tables, page_table=page_table, kv_cache=caches)
        output.deallocate(True)
        decoder.layer.moe.experts = selected
        assert captures, "prefill did not reach the expert boundary"
        with device_only():
            expert_input = captures[0][0] if len(captures) == 1 else ttnn.concat([pair[0] for pair in captures], dim=2)
            routes = captures[0][1] if len(captures) == 1 else ttnn.concat([pair[1] for pair in captures], dim=2)
        assert expert_input.shape[-2] == args.length
        report["captured_chunks"] = len(captures)
        report["expert_input_shape"] = list(expert_input.shape)
        report["routing_shape"] = list(routes.shape)
        print("REAL_EXPERT_BOUNDARY_READY", flush=True)

        with device_only():
            output = baseline(expert_input, routes)
        reference = ttnn.to_torch(output).float()
        output.deallocate(True)

        candidates = [("loop32", 32, baseline)] + [
            (f"direct_m{tokens}", tokens, DirectMExperts(baseline, tokens)) for tokens in args.batch_tokens
        ]
        for name, batch_tokens, candidate in candidates:
            row = dict(candidate=name, batch_tokens=batch_tokens)
            try:
                with device_only():
                    output = candidate(expert_input, routes)
                actual = ttnn.to_torch(output).float()
                output.deallocate(True)
                passing, pcc = comp_pcc(reference, actual, 0.995)
                row.update(
                    pcc=float(pcc),
                    passed=bool(passing),
                    exact_equal=torch.equal(reference, actual),
                    max_abs=float((reference - actual).abs().max()),
                )
                if isinstance(candidate, DirectMExperts):
                    row["sparse_output_shapes"] = candidate.observed_shapes
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    output = candidate(expert_input, routes)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    for _ in range(args.warmup_replays):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(args.timing_samples):
                        start = time.perf_counter_ns()
                        for _ in range(args.repeats):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter_ns() - start) / (args.repeats * 1000))
                    replay_equal = torch.equal(ttnn.to_torch(output).float(), actual)
                finally:
                    ttnn.release_trace(mesh, trace)
                    output.deallocate(True)
                row.update(
                    trace_replay_equal=replay_equal,
                    passed=bool(passing) and replay_equal,
                    traced_host_us=statistics.median(samples),
                    traced_host_us_samples=samples,
                )
            except Exception as error:
                row.update(passed=False, error=f"{type(error).__name__}: {error}")
                report["results"].append(row)
                save()
                raise
            report["results"].append(row)
            save()
            print(row, flush=True)
        assert all(row["passed"] for row in report["results"]), report["results"]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
