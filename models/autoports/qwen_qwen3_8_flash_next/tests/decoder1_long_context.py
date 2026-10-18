"""Bounded real layer-1 context proof, preserving its coupled persistent state.

For prefix chunks execute PLE -> residual addition -> attention hyper-connection
-> GDN, in both HF and TT. The omitted attention residual write and MoE suffix
are stateless and cannot change this isolated layer's cache for fixed external
inputs. Periodically execute that suffix too and require complete output PCC
>= .995. At the end, use public complete decoder calls for aligned/unaligned
prefill and changed-input traced decode at the last advertised position.

Example (run hardware jobs serially):
TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.decoder1_long_context --warm-reference-pages
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc, pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.ple_long_context import (
    MappedCheckpointTable,
    StreamingPCC,
    copy_input,
    device_buffer,
    lane,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import (
    load_reference,
    no_runtime_host_conversion,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.test_ple import CHECKPOINT, LazyCheckpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context", type=int, default=262144, help="Harness length only; config/capability are never reduced"
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--full-check-every", type=int, default=4096)
    parser.add_argument("--warm-reference-pages", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--weight-cache",
        type=Path,
        default=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/decoder1_long_context.json"),
    )
    args = parser.parse_args()
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    assert cfg.max_position_embeddings == 262144
    assert 2 <= args.chunk_size <= args.context <= cfg.max_position_embeddings
    assert args.context % args.chunk_size == 0, "Use aligned maximum and test its nonaligned predecessor"
    assert args.full_check_every > 0 and args.full_check_every % args.chunk_size == 0
    torch.set_num_threads(8)
    state = CheckpointState(args.checkpoint, "model.language_model.layers.1.")
    reference = load_reference(cfg, state, 1)
    checkpoint = LazyCheckpoint(args.checkpoint)
    reference.ple.ple_embedding.ngram_embedding = MappedCheckpointTable(checkpoint)
    token_generator = torch.Generator().manual_seed(731)
    tokens = torch.randint(0, cfg.vocab_size, (args.context,), generator=token_generator)
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    tokens[100::101] = eos
    records, state_records = [], []
    ple_statistics = StreamingPCC()
    started = time.monotonic()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    page_warmup = None
    prefix_consumed = 0

    def persist(complete=False):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "complete": complete,
                    "layer_idx": 1,
                    "weights": "real",
                    "batch": 1,
                    "hf_advertised_context": cfg.max_position_embeddings,
                    "tested_context": args.context,
                    "capability_reduction": None,
                    "validated_coupled_prefix": prefix_consumed,
                    "prefix_omission": "Only stateless attention residual write and MoE suffix; periodically included",
                    "input_distribution": "integration control: common N(0,.05) vector repeated into4 streams plus independent N(0,.005) noise",
                    "reference_page_warmup": page_warmup,
                    "records": records,
                    "state_records": state_records,
                    "ple_prefix_diagnostic": ple_statistics.result() if ple_statistics.n else None,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )

    def metric(expected, actual, *, mode, position, length, acceptance=True, **extra):
        assert tuple(actual.shape) == tuple(expected.shape), (actual.shape, expected.shape)
        metrics = pcc_metrics(expected, actual)
        score = metrics["pcc"]
        assert math.isfinite(score), (mode, score)
        row = {
            "mode": mode,
            "position": position,
            "length": length,
            **metrics,
            "threshold": 0.995 if acceptance else None,
            **extra,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        persist()
        if acceptance:
            assert score >= 0.995, row
        return score

    try:
        model = FunctionalDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=1,
            mesh_device=mesh,
            batch_size=1,
            max_requests=2,
            chunk_size=args.chunk_size,
            cache_dir=args.weight_cache,
        )
        shapes = (1, args.chunk_size - 1, args.chunk_size)
        model.prepare(shapes)
        if args.warm_reference_pages:
            page_warmup = reference.ple.ple_embedding.ngram_embedding.warm_pages()
            print(json.dumps({"event": "HF_REFERENCE_PAGE_WARMUP_COMPLETE", **page_warmup}), flush=True)
        buffers = {}
        for seq in shapes:
            buffers[seq] = {
                "x": device_buffer(
                    torch.zeros(1, 1, seq, 10240, dtype=torch.bfloat16), ttnn.bfloat16, mesh, tiled=True
                ),
                "tokens": device_buffer(torch.zeros(1, seq, dtype=torch.int32), ttnn.uint32, mesh),
                "positions": device_buffer(lane(0), ttnn.uint32, mesh),
                "slots": device_buffer(lane(1), ttnn.uint32, mesh),
            }

        def activation(ids, position, *, variant=0):
            # Match test_functional_decoder.py's representative input formula.
            # Four streams share a common component but remain unequal.
            generator = torch.Generator().manual_seed(1000 + position + variant)
            seq = ids.shape[-1]
            base = torch.randn(1, seq, 2560, generator=generator) * 0.05
            noise = torch.randn(1, seq, 10240, generator=generator) * 0.005
            return (base.repeat(1, 1, 4) + noise).to(torch.bfloat16)

        def refresh(position, ids, *, variant=0):
            seq = ids.shape[-1]
            x = activation(ids, position, variant=variant)
            active = buffers[seq]
            copy_input(x.unsqueeze(0), active["x"])
            copy_input(ids.to(torch.int32), active["tokens"])
            copy_input(lane(position), active["positions"])
            return x, active

        def full(active, *, decode=False):
            function = model.decode_forward if decode else model.prefill_forward
            with no_runtime_host_conversion():
                return function(
                    active["x"], current_positions=active["positions"], slots=active["slots"], tokens=active["tokens"]
                )

        for seq in shapes:
            warm = full(buffers[seq])
            ttnn.synchronize_device(mesh)
            del warm
        warm = full(buffers[1], decode=True)
        ttnn.synchronize_device(mesh)
        del warm
        hf_cache = DynamicCache(config=cfg)

        def compare_state(position, hf):
            pairs = {
                "gdn_recurrent": (hf.layers[1].recurrent_states[0], ttnn.to_torch(model.attention.recurrent_bank)[1:2]),
                "gdn_conv": (
                    hf.layers[1].conv_states[0][..., -3:].transpose(1, 2),
                    ttnn.to_torch(model.attention.conv_bank)[1:2],
                ),
                "ple_conv": (
                    hf.layers[1].conv_states[1].transpose(1, 2),
                    ttnn.to_torch(model.ple_cache.conv_history)[9:18].unsqueeze(0),
                ),
            }
            for name, (expected, actual) in pairs.items():
                assert actual.shape == expected.shape
                metrics = pcc_metrics(expected, actual)
                row = {
                    "position": position,
                    "state": name,
                    **metrics,
                    "acceptance": "diagnostic; full decoder outputs must pass .995",
                }
                state_records.append(row)
                print(json.dumps(row), flush=True)
            # Raw HF token history may retain a token before EOS whereas TT's
            # equivalent state replaces it with EOS; do not assert byte identity.
            token_history = ttnn.to_torch(model.ple_cache.token_history)
            assert int(token_history[1, 2]) == position

        prefix_end = args.context - args.chunk_size
        for position in range(0, prefix_end, args.chunk_size):
            seq = args.chunk_size
            ids = tokens[position : position + seq].reshape(1, seq)
            x, active = refresh(position, ids)
            hf_ple = reference.ple(x, ids, hf_cache)
            hf_residual = x + hf_ple  # Required PLE injection before HC mixing.
            hf_mixed, hf_hyper, hf_inject = reference.attn_hyper_connection(hf_residual)
            hf_update = reference.linear_attn(hf_mixed, cache_params=hf_cache)
            with no_runtime_host_conversion():
                tt_ple = model.ple.forward(
                    active["x"],
                    tokens=active["tokens"],
                    positions=active["positions"],
                    slots=active["slots"],
                    valid_lengths=model.lengths[seq],
                    cache=model.ple_cache,
                )
                tt_residual = ttnn.add(active["x"], tt_ple)
                tt_mixed, tt_inject = model.attn_hyper.read(tt_residual)
                tt_update = model.attention.prefill_forward(
                    tt_mixed, current_positions=active["positions"], slots=active["slots"]
                )
            actual_ple = ttnn.to_torch(tt_ple).squeeze(0)
            ple_score = pcc(actual_ple, hf_ple)
            ple_statistics.update(actual_ple, hf_ple)
            if ple_score < 0.99:
                row = {
                    "mode": "ple_component_diagnostic_below_0_99",
                    "position": position,
                    "length": seq,
                    "pcc": ple_score,
                    "threshold": None,
                    "interpretation": "Requires comparison with complete layer output; not a relaxed layer gate",
                }
                records.append(row)
                print(json.dumps(row), flush=True)
            # Periodically finish the stateless suffix to qualify complete
            # outputs without recomputing or advancing any cache twice.
            if position % args.full_check_every == 0:
                hf_after_attn = hf_hyper + (hf_update.unsqueeze(-2) * hf_inject.unsqueeze(-1)).flatten(-2)
                hf_mlp_input, hf_mlp_hyper, hf_mlp_inject = reference.mlp_hyper_connection(hf_after_attn)
                hf_mlp = reference.mlp(hf_mlp_input)
                expected = hf_mlp_hyper + (hf_mlp.unsqueeze(-2) * hf_mlp_inject.unsqueeze(-1)).flatten(-2)
                with no_runtime_host_conversion():
                    tt_after_attn = model.attn_hyper.write(tt_residual, tt_update, tt_inject)
                    tt_mlp_input, tt_mlp_inject = model.mlp_hyper.read(tt_after_attn)
                    output = model.mlp_hyper.write(tt_after_attn, model.mlp(tt_mlp_input), tt_mlp_inject)
                metric(
                    expected,
                    ttnn.to_torch(output).squeeze(0),
                    mode="periodic_complete_prefix_chunk",
                    position=position,
                    length=seq,
                )
                del output, tt_after_attn, tt_mlp_input, tt_mlp_inject
            prefix_consumed = position + seq
            if prefix_consumed % args.full_check_every == 0 or prefix_consumed == prefix_end:
                compare_state(prefix_consumed, hf_cache)
                print(
                    json.dumps(
                        {
                            "event": "COUPLED_PREFIX_PROGRESS",
                            "position": prefix_consumed,
                            "ple_diagnostic": ple_statistics.result(),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                persist()
            del tt_ple, tt_residual, tt_mixed, tt_inject, tt_update
        tensors = {
            "ple_tokens": model.ple_cache.token_history,
            "ple_conv": model.ple_cache.conv_history,
            "gdn_recurrent_bank": model.attention.recurrent_bank,
            "gdn_conv_bank": model.attention.conv_bank,
            "gdn_recurrent_work": model.attention.recurrent_state,
            "gdn_conv_work": model.attention.conv_state,
        }

        def snapshot():
            return {name: ttnn.to_torch(tensor) for name, tensor in tensors.items()}

        def restore(saved):
            for name, host in saved.items():
                copy_input(host, tensors[name])

        prefix_tt, prefix_hf = snapshot(), copy.deepcopy(hf_cache)
        # Complete aligned final chunk proves all outputs, not just final token.
        ids = tokens[prefix_end : args.context].reshape(1, args.chunk_size)
        x, active = refresh(prefix_end, ids)
        expected = reference(
            x, position_embeddings=(None, None), past_key_values=copy.deepcopy(prefix_hf), ple_input_ids=ids
        )
        output = full(active)
        metric(
            expected,
            ttnn.to_torch(output).squeeze(0),
            mode="full_context_complete_prefill",
            position=prefix_end,
            length=args.chunk_size,
        )
        del output
        restore(prefix_tt)
        hf_cache = copy.deepcopy(prefix_hf)
        ids = tokens[prefix_end : args.context - 1].reshape(1, args.chunk_size - 1)
        x, active = refresh(prefix_end, ids)
        expected = reference(x, position_embeddings=(None, None), past_key_values=hf_cache, ple_input_ids=ids)
        output = full(active)
        metric(
            expected,
            ttnn.to_torch(output).squeeze(0),
            mode="nonaligned_context_complete_prefill",
            position=prefix_end,
            length=args.chunk_size - 1,
        )
        del output
        prefix_consumed = args.context - 1
        compare_state(prefix_consumed, hf_cache)
        final_tt, final_hf = snapshot(), copy.deepcopy(hf_cache)
        ids = tokens[-1:].reshape(1, 1)
        x, active = refresh(args.context - 1, ids)
        expected = reference(
            x, position_embeddings=(None, None), past_key_values=copy.deepcopy(final_hf), ple_input_ids=ids
        )
        output = full(active)
        metric(
            expected,
            ttnn.to_torch(output).squeeze(0),
            mode="final_token_complete_prefill",
            position=args.context - 1,
            length=1,
        )
        del output
        restore(final_tt)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced_output = full(active, decode=True)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        first = None
        for replay in range(3):
            ids = torch.tensor([[int(tokens[-1]) if replay != 1 else eos]])
            x, active = refresh(args.context - 1, ids)
            restore(final_tt)
            expected = reference(
                x, position_embeddings=(None, None), past_key_values=copy.deepcopy(final_hf), ple_input_ids=ids
            )
            with no_runtime_host_conversion():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(traced_output).squeeze(0)
            metric(
                expected,
                actual,
                mode="full_context_complete_traced_decode",
                position=args.context - 1,
                length=1,
                replay=replay,
                trace_id=str(trace),
                capture_count=1,
                token=ids.item(),
            )
            if replay == 0:
                first = actual.clone()
            elif replay == 2:
                torch.testing.assert_close(actual, first, rtol=0, atol=0)
        persist(complete=True)
        print(json.dumps({"event": "DECODER1_LONG_CONTEXT_COMPLETE", "artifact": str(args.output)}), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
