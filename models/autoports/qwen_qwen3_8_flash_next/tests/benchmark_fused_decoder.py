"""Serialized real-weight, identical-input fusion candidate measurements."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import (
    CHECKPOINT,
    no_runtime_host_conversion,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["functional", "fused"], required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--conv-bf16", action="store_true")
    parser.add_argument(
        "--candidate",
        choices=[
            "default",
            "dedicated_moe",
            "native_conv",
            "native_norm",
            "native_frontend",
            "split_hc",
            "packed_gdn",
            "qsa_heads",
            "fast_capacity",
            "concat_heads",
            "merged_hc",
            "packed_ple",
            "decode_rope",
            "no_native_conv",
            "no_native_norm",
            "rms_l2",
            "rms_precise",
            "fold_query_scale",
            "unified_moe",
            "addcmul_write",
            "addcmul_read",
            "l2_activation",
            "softplus_activation",
            "position_mac",
            "signed_sqrt",
            "floor_position",
            "indexed_cache",
            "indexed_decode_cache",
            "paged_decode_cache",
            "weighted_reduce",
            "attn_weighted_reduce",
            "precise_norm",
            "stable_norm",
            "hc_split_all",
            "fp32_norm",
            "packed_router_shared",
            "ple_gate_chain",
            "ple_norm_output",
            "ple_adjacent_both",
            "ple_dot_scale",
            "gdn_query_scale",
            "gdn_decay_exp",
            "gdn_adjacent_both",
            "qsa_mask_output",
            "moe_mask_softmax",
            "moe_topk10",
            "qsa_shared_rope",
            "qsa_adjacent_both",
        ],
        default="default",
    )
    args = parser.parse_args()
    if args.candidate == "fast_capacity":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import FastCapacityMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedMoE = FastCapacityMoE
    if args.candidate == "dedicated_moe":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import DedicatedMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedMoE = DedicatedMoE
    if args.candidate == "qsa_heads":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import NativeQSAHeads
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedQSA = NativeQSAHeads
    if args.candidate.startswith("native_"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import (
            NativeConvGDN,
            NativeFrontEndGDN,
            NativeNormGDN,
        )
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedGatedDeltaNet = dict(
            native_conv=NativeConvGDN, native_norm=NativeNormGDN, native_frontend=NativeFrontEndGDN
        )[args.candidate]
    if args.candidate in ("split_hc", "packed_gdn"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import PackedGDN, SplitPrefillHC
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        if args.candidate == "split_hc":
            fused_decoder.FusedHyperConnection = SplitPrefillHC
        else:
            fused_decoder.FusedGatedDeltaNet = PackedGDN
    if args.candidate in ("concat_heads", "merged_hc", "packed_ple"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import ConcatQSAHeads, MergedHC, PackedPLE
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        name, cls = dict(
            concat_heads=("FusedQSA", ConcatQSAHeads),
            merged_hc=("FusedHyperConnection", MergedHC),
            packed_ple=("FusedPLE", PackedPLE),
        )[args.candidate]
        setattr(fused_decoder, name, cls)
    if args.candidate == "decode_rope":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import DecodeRoPE
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedQSA = DecodeRoPE
    if args.candidate in ("no_native_conv", "no_native_norm"):
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        attribute = "native_conv" if args.candidate == "no_native_conv" else "native_norm"
        fused_decoder.FusedGatedDeltaNet = type(args.candidate, (fused_decoder.FusedGatedDeltaNet,), {attribute: False})
    if args.candidate == "rms_l2":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import RMSL2GDN
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedGatedDeltaNet = RMSL2GDN
    if args.candidate == "rms_precise":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import PreciseRMSL2GDN
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedGatedDeltaNet = PreciseRMSL2GDN
    if args.candidate == "fold_query_scale":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import FoldQueryScaleGDN
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedGatedDeltaNet = FoldQueryScaleGDN
    if args.candidate == "unified_moe":
        from models.autoports.qwen_qwen3_8_flash_next.tests.unified_moe_candidate import UnifiedMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder

        fused_decoder.FusedMoE = UnifiedMoE
    if args.candidate in ("addcmul_write", "addcmul_read", "l2_activation", "softplus_activation", "position_mac"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.late_fusion_candidates import install

        install(args.candidate)
    if args.candidate in ("signed_sqrt", "floor_position"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.review_fusion_candidates import install

        install(args.candidate)
    if args.candidate in ("indexed_cache", "indexed_decode_cache", "paged_decode_cache"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_cache_candidate import install

        install(args.candidate)
    if args.candidate in ("weighted_reduce", "attn_weighted_reduce"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_weighted_reduce_candidate import install

        install(args.candidate)
    if args.candidate in ("precise_norm", "fp32_norm", "stable_norm"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_hc_precision_candidates import install

        install(args.candidate)
    if args.candidate == "hc_split_all":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_hc_split_candidates import install

        install("split_all")
    if args.candidate == "packed_router_shared":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_router_shared_candidate import install

        install(args.candidate)
    if args.candidate in ("ple_gate_chain", "ple_norm_output", "ple_adjacent_both", "ple_dot_scale"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_ple_gate_candidate import install

        install(args.candidate)
    if args.candidate in ("gdn_query_scale", "gdn_decay_exp", "gdn_adjacent_both"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_gdn_adjacent_candidates import install

        install(args.candidate)
    if args.candidate == "qsa_mask_output":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_mask_candidate import install

        install(args.candidate)
    if args.candidate in ("moe_mask_softmax", "moe_topk10"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_moe_mask_softmax_candidate import install

        install(args.candidate)
    if args.candidate == "qsa_shared_rope":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_shared_rope_candidate import install

        install(args.candidate)
    if args.candidate == "qsa_adjacent_both":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_combined_candidate import install

        install(args.candidate)
    if args.conv_bf16:
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_hc_split_candidates import install_conv_bf16

        conv_source = install_conv_bf16()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    dest = root / "doc/fused_decoder" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    if args.conv_bf16:
        (dest / "generated_conv_chunk.py.txt").write_text(conv_source)
        (dest / "conv_candidate.py.txt").write_bytes((root / "tests/fused_hc_split_candidates.py").read_bytes())
    if args.candidate in ("indexed_cache", "indexed_decode_cache", "paged_decode_cache"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_cache_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate in ("weighted_reduce", "attn_weighted_reduce"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_weighted_reduce_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate in ("precise_norm", "fp32_norm", "stable_norm"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_hc_precision_candidates import snapshot_provenance

        snapshot_provenance(dest / "hc_sources")
    if args.candidate == "packed_router_shared":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_router_shared_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate in ("ple_gate_chain", "ple_norm_output", "ple_adjacent_both", "ple_dot_scale"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_ple_gate_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate in ("gdn_query_scale", "gdn_decay_exp", "gdn_adjacent_both"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_gdn_adjacent_candidates import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate == "qsa_mask_output":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_mask_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate in ("moe_mask_softmax", "moe_topk10"):
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_moe_mask_softmax_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate == "qsa_shared_rope":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_shared_rope_candidate import snapshot_provenance

        snapshot_provenance(dest)
    if args.candidate == "qsa_adjacent_both":
        from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_combined_candidate import snapshot_provenance

        snapshot_provenance(dest)
    source = root / "tt/fused_decoder.py"
    source_snapshot = source.read_bytes()
    (dest / "fused_source.py.txt").write_bytes(source_snapshot)
    provenance_paths = sorted((root / "tt").rglob("*.py")) + sorted((root / "tt").rglob("*.cpp"))
    provenance_paths += [
        Path(__file__),
        root / "tests/fusion_candidates.py",
        root / "tests/precise_recurrence_candidate.py",
        root / "tests/unified_moe_candidate.py",
        root / "tests/late_fusion_candidates.py",
        root / "tests/review_fusion_candidates.py",
        root / "tests/fused_cache_candidate.py",
        root / "tests/fused_weighted_reduce_candidate.py",
        root / "tests/fused_router_shared_candidate.py",
        root / "tests/fused_ple_gate_candidate.py",
        root / "tests/fused_gdn_adjacent_candidates.py",
        root / "tests/fused_qsa_mask_candidate.py",
        root / "tests/fused_moe_mask_softmax_candidate.py",
        root / "tests/fused_qsa_shared_rope_candidate.py",
        root / "tests/fused_qsa_combined_candidate.py",
    ]
    if args.impl == "fused":
        provenance_paths += sorted(
            Path("ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu").rglob("*")
        )
    if args.impl == "fused":
        provenance_paths += sorted(
            Path("ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn").rglob("*")
        )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_paths if p.is_file()}
    for name in (
        "fusion_candidates.py",
        "precise_recurrence_candidate.py",
        "unified_moe_candidate.py",
        "late_fusion_candidates.py",
        "review_fusion_candidates.py",
        "fused_cache_candidate.py",
        "fused_weighted_reduce_candidate.py",
        "fused_router_shared_candidate.py",
        "fused_ple_gate_candidate.py",
        "fused_gdn_adjacent_candidates.py",
        "fused_qsa_mask_candidate.py",
        "fused_moe_mask_softmax_candidate.py",
        "fused_qsa_shared_rope_candidate.py",
        "fused_qsa_combined_candidate.py",
    ):
        path = root / "tests" / name
        if path.exists():
            (dest / (name + ".txt")).write_bytes(path.read_bytes())
    (dest / "source_manifest.json").write_text(
        json.dumps(
            dict(
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                checkpoint=str(CHECKPOINT),
                source_hashes=hashes,
            ),
            indent=2,
        )
        + "\n"
    )
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{args.layer}.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = (FusedDecoder if args.impl == "fused" else FunctionalDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            batch_size=1,
            max_requests=3,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        model.prepare([1, 128])
        identities = {
            name: type(getattr(model, name)).__module__ + "." + type(getattr(model, name)).__qualname__
            for name in ("attention", "mlp", "ple", "attn_hyper", "mlp_hyper")
            if hasattr(model, name)
        }

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        rng = torch.Generator().manual_seed(2026)
        x = (
            torch.randn(1, 128, 2560, generator=rng).repeat(1, 1, 4) * 0.05
            + torch.randn(1, 128, 10240, generator=rng) * 0.005
        ).bfloat16()
        tokens = torch.randint(cfg.vocab_size, (1, 128), generator=rng, dtype=torch.int32)
        xd, dd = upload(x.unsqueeze(0)), upload(x[:, :1].unsqueeze(0))
        pd = upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(1, 32, dtype=torch.int32)
        slots[0, 0] = 1
        sd = upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page = (
            upload(
                torch.randperm(cfg.max_position_embeddings // 32, generator=rng).reshape(1, -1).int(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if args.layer == 3
            else None
        )
        pt = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        dt = upload(tokens[:, :1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        common = dict(current_positions=pd, slots=sd, page_table=page)

        def prefill():
            with no_runtime_host_conversion():
                return model.prefill_forward(xd, tokens=pt, **common)

        def decode():
            with no_runtime_host_conversion():
                return model.decode_forward(dd, tokens=dt, **common)

        def drain():
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)

        for _ in range(2):
            prefill()
            decode()
            ttnn.synchronize_device(mesh)
            drain()
        if args.profile:
            from tracy import signpost
        else:
            signpost = lambda _: None
        timings = []
        prefill_host = None
        for _ in range(1 if args.profile else 5):
            signpost("PERF_PREFILL")
            before = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - before) * 1000)
            signpost("PERF_PREFILL_END")
            drain()
            prefill_host = ttnn.to_torch(out)
            del out
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        drain()
        for _ in range(2):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            drain()
        samples = []
        for _ in range(1 if args.profile else 5):
            signpost("PERF_DECODE")
            before = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - before) * 1000 / 10)
            signpost("PERF_DECODE_END")
            drain()
        saved = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/fusion_outputs")
        saved.mkdir(parents=True, exist_ok=True)
        output_path = saved / (args.tag + ".pt")
        torch.save({"prefill": prefill_host, "decode": ttnn.to_torch(out)}, output_path)
        result = dict(
            impl=args.impl,
            layer=args.layer,
            tag=args.tag,
            seed=2026,
            candidate=args.candidate,
            conv_bf16=args.conv_bf16,
            runtime_classes=identities,
            batch=1,
            prefill_length=128,
            position=0,
            decode_replays=50 if not args.profile else 10,
            prefill_host_ms=timings,
            traced_decode_ms=samples,
            median_prefill_host_ms=statistics.median(timings),
            median_traced_decode_ms=statistics.median(samples),
            profiler=args.profile,
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            outputs=str(output_path),
            source_sha256=hashlib.sha256(source_snapshot).hexdigest(),
        )
        (dest / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
