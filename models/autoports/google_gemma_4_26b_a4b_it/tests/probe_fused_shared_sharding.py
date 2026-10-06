# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the shared-down projection plus norm on live decoder activations."""

import argparse
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_expert_mix import measure
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-fidelity", choices=("default", "LoFi", "HiFi2"), default="default")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-samples", type=int, default=3)
    parser.add_argument("--warmup-replays", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(cfg, args.layer, True)
    inputs = torch.load(Path(__file__).parents[1] / "doc/functional_decoder/headline_inputs.pt", weights_only=True)
    x_host = inputs["x"][:, :64].contiguous()
    rope = Gemma4TextRotaryEmbedding(cfg)
    with torch.no_grad():
        cos, sin = rope(x_host, torch.arange(1024)[None], layer_type=cfg.layer_types[args.layer])
    report = dict(
        layer=args.layer, layer_type=cfg.layer_types[args.layer], revision=REVISION, real_weights=True, results=[]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        fusion = "_".join(
            flag
            for flag in FusedDecoder.DEFAULT_FUSIONS[cfg.layer_types[args.layer]].split("_")
            if not flag.startswith(("sharedshard", "sharedblock"))
        )
        decoder = FusedDecoder.from_state_dict(
            hf.state_dict(), hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, fusion=fusion
        )
        source = decoder.layer.shared_mlp
        original_down = source.down_proj
        captures = []

        def capture_down(value):
            linear = ttnn.linear

            def record_linear(a, b, **kwargs):
                result = linear(a, b, **kwargs)
                if a.shape[-2] == 1:
                    captures.append((ttnn.clone(a), b, ttnn.clone(result), kwargs))
                return result

            with patch.object(ttnn, "linear", record_linear):
                return original_down(value)

        source.down_proj = capture_down

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        tables = tuple(upload(value[None]) for value in (cos, sin))
        page_table = upload(torch.arange(31, -1, -1, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        attention = decoder.layer.self_attn.config
        caches = tuple(
            upload(torch.zeros((32, attention.num_key_value_heads, 32, attention.head_dim))) for _ in range(2)
        )
        x = upload(x_host[None])
        with device_only():
            out = decoder.prefill_forward(x, rope_mats=tables, page_table=page_table, kv_cache=caches)
        out.deallocate(True)
        x = upload(inputs["decode_inputs"][:, :1][None].contiguous())
        position = upload(torch.tensor([64], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        positions = torch.zeros(1, 32, dtype=torch.int32)
        positions[0, 0] = 64
        rope_position = upload(positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        decode_tables = tuple(upload(value.squeeze(0)) for value in (cos, sin))
        with device_only():
            out = decoder.decode_forward(
                x,
                current_pos=rope_position,
                cache_pos=position,
                rope_mats=decode_tables,
                page_table=page_table,
                kv_cache=caches,
            )
        out.deallocate(True)
        source.down_proj = original_down
        assert len(captures) == 1
        activation, weight, captured, kwargs = captures[0]
        assert not kwargs, kwargs
        control = ttnn.linear(activation, weight)
        assert torch.equal(ttnn.to_torch(control), ttnn.to_torch(captured))
        control.deallocate(True)
        captured.deallocate(True)
        norm = decoder.layer.post_feedforward_layernorm_1
        memory, norm_program = norm._sharded_cfg
        end = memory.shard_spec.grid.bounding_box().end
        report.update(
            capture_fusion=decoder.fusion,
            input_shape=list(activation.shape),
            weight_shape=list(weight.shape),
            norm_memory=str(memory),
            baseline_matches_capture=True,
            scope="decode shared down projection, all layout movement, and following norm; BF16",
            baseline_fidelity="HiFi2 (inferred without explicit program config)",
            candidate_fidelity=(
                "LoFi (inferred with explicit program config)"
                if args.candidate_fidelity == "default"
                else args.candidate_fidelity
            ),
        )

        def normalize(value):
            return ttnn.rms_norm(
                ttnn.to_memory_config(value, memory),
                weight=norm.tt_weight,
                epsilon=norm.eps,
                program_config=norm_program,
            )

        def baseline():
            return normalize(ttnn.linear(activation, weight))

        reference = ttnn.to_torch(baseline()).float()
        compute = (
            None
            if args.candidate_fidelity == "default"
            else ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, args.candidate_fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        )
        candidates = [("interleaved_then_shard", baseline)]
        for block in (2, 3, 6, 11, 22, 33, 66):
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(end.x + 1, end.y + 1),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=memory.shard_spec.shape[1] // 32,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            candidates.append(
                (
                    f"direct_shard_k{block}",
                    lambda program=program: normalize(
                        ttnn.linear(
                            activation,
                            weight,
                            program_config=program,
                            memory_config=memory,
                            compute_kernel_config=compute,
                        )
                    ),
                )
            )
        for name, fn in candidates:
            row = dict(candidate=name)
            try:
                actual, repeated, samples = measure(mesh, fn, args)
                passing, pcc = comp_pcc(reference, actual, 0.995)
                row.update(
                    pcc=float(pcc),
                    passed=bool(passing) and repeated,
                    exact_equal=torch.equal(reference, actual),
                    repeated_equal=repeated,
                    traced_host_us=statistics.median(samples),
                    traced_host_us_samples=samples,
                )
            except Exception as error:
                row.update(passed=False, error=f"{type(error).__name__}: {error}")
            report["results"].append(row)
            save()
            print(row, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
