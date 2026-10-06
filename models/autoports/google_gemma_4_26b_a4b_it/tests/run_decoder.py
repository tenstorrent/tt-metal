# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only real-shape HF parity runner. All host conversion is in this harness."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def load_layer(config, layer_idx, real):
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, layer_idx)
    if real:
        index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)).read_text())
        prefix = f"model.language_model.layers.{layer_idx}."
        state = {}
        for shard in sorted({v for k, v in index["weight_map"].items() if k.startswith(prefix)}):
            path = hf_hub_download(MODEL, shard, revision=REVISION)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = f.get_tensor(key).float()
    else:
        stats_path = Path(__file__).parents[1] / "doc/functional_decoder/weight_stats.json"
        stats = {row["name"]: row for row in json.loads(stats_path.read_text())["tensors"]}
        state = {}
        for name, param in layer.state_dict().items():
            row = stats[f"model.language_model.layers.{layer_idx}.{name}"]
            source_dtype = getattr(torch, row["source_dtype"].removeprefix("torch."))
            state[name] = (torch.randn(param.shape) * row["std"] + row["mean"]).to(source_dtype).float()
    layer.load_state_dict(state, assign=True)
    return layer.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--prefill-timing", action="store_true", help="Three synchronized warmed host-wall prefill samples"
    )
    parser.add_argument("--fusion", help="Explicit fusion ablation; default selects the layer-kind policy")
    parser.add_argument("--group-size", type=int, default=16384)
    parser.add_argument("--decoder", choices=("functional", "fused"), default="functional")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--cache-extent", type=int, help="Explicit paged capacity for tight allocation checks")
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--input-fixture", type=Path, help="Recorded text-derived layer inputs, outside timed execution"
    )
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verify-program-cache", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-output-tensors", type=Path, help="Save already-read outputs for paired decoder checks")
    args = parser.parse_args()
    if args.save_output_tensors and args.profile:
        parser.error("Paired output capture is a correctness check, separate from profiling")
    saved_outputs = {}
    torch.manual_seed(42)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    config._attn_implementation = "eager"
    hf = load_layer(config, args.layer, args.real)
    print("HF_LAYER_READY", flush=True)
    fixture = None
    fixture_sha256 = None
    if args.input_fixture:
        assert args.real, "Recorded activations require real model weights"
        fixture = torch.load(args.input_fixture, map_location="cpu", weights_only=True)
        fixture_sha256 = hashlib.sha256(args.input_fixture.read_bytes()).hexdigest()
        metadata = fixture["metadata"]
        assert metadata["model"] == MODEL and metadata["revision"] == REVISION
        assert metadata["layer"] == args.layer
        assert fixture["prefill"].shape == (1, args.length, config.hidden_size)
        assert fixture["decode"].shape == (1, args.steps, config.hidden_size)
        assert torch.isfinite(fixture["prefill"]).all() and torch.isfinite(fixture["decode"]).all()
        assert torch.equal(fixture["prefill"], fixture["prefill"].bfloat16().float())
        assert torch.equal(fixture["decode"], fixture["decode"].bfloat16().float())
    x = fixture["prefill"].float() if fixture else torch.randn(1, args.length, config.hidden_size).bfloat16().float()
    rope = Gemma4TextRotaryEmbedding(config)
    layer_type = config.layer_types[args.layer]
    extent = (args.length + max(128, args.steps) + 1023) // 1024 * 1024
    cos, sin = rope(x, torch.arange(extent)[None], layer_type=layer_type)
    positions = torch.arange(args.length)
    allowed = positions[:, None] >= positions[None, :]
    if layer_type == "sliding_attention":
        allowed &= positions[:, None] - positions[None, :] < config.sliding_window
    mask = torch.zeros(args.length, args.length).masked_fill(~allowed, float("-inf"))[None, None]
    hf_cache = DynamicCache()
    with torch.no_grad():
        ref = hf(
            x,
            position_embeddings=(cos[:, : args.length], sin[:, : args.length]),
            attention_mask=mask,
            past_key_values=hf_cache,
        )
    print("HF_FORWARD_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder_class = FunctionalDecoder
        if args.decoder == "fused":
            from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder

            decoder_class = FusedDecoder
        decoder = decoder_class.from_state_dict(
            hf.state_dict(),
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({"fusion": args.fusion, "group_size": args.group_size} if args.decoder == "fused" else {}),
        )
        print("TT_LAYER_READY", flush=True)

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        block = 32
        cache_extent = args.cache_extent if args.cache_extent is not None else extent
        if cache_extent < args.length + (args.steps if args.decode else 0) or cache_extent % 128:
            raise ValueError("Cache extent must cover the requested tokens and be a multiple of 128")
        pages = (cache_extent + block - 1) // block
        table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None]
        pt = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        attn = decoder.layer.self_attn.config
        shape = (pages, attn.num_key_value_heads, block, attn.head_dim)
        cache = [device(torch.zeros(shape), getattr(decoder, "kv_cache_dtype", ttnn.bfloat16)) for _ in range(2)]
        xt = device(x[None])
        rt = tuple(device(t[None]) for t in (cos, sin))

        def prefill():
            if not args.prefix_length:
                return decoder.prefill_forward(xt, rope_mats=rt, page_table=pt, kv_cache=cache)
            prefix = args.prefix_length
            assert 0 < prefix < args.length
            first = decoder.prefill_forward(xt[:, :, :prefix, :], rope_mats=rt, page_table=pt, kv_cache=cache)
            rest = decoder.prefill_forward(
                xt[:, :, prefix:, :], rope_mats=rt, page_table=pt, kv_cache=cache, start_pos=prefix
            )
            return ttnn.concat([first, rest], dim=2)

        with device_only():
            out = prefill()
        prefill_cache_entries = mesh.num_program_cache_entries()
        if args.verify_program_cache:
            assert prefill_cache_entries > 0, "Warmup did not populate the program cache"
        if args.profile or args.verify_program_cache:
            ttnn.synchronize_device(mesh)
            out.deallocate(True)
            if args.profile:
                from tracy import signpost

                signpost("PERF_PREFILL")
            if args.verify_program_cache:
                mesh.set_program_cache_misses_allowed(False)
            try:
                with device_only():
                    out = prefill()
            finally:
                mesh.set_program_cache_misses_allowed(True)
            ttnn.synchronize_device(mesh)
            if args.profile:
                signpost("PERF_PREFILL_END")
        prefill_host_us = []
        if args.prefill_timing:
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                out.deallocate(True)
                start = time.perf_counter_ns()
                with device_only():
                    out = prefill()
                ttnn.synchronize_device(mesh)
                prefill_host_us.append((time.perf_counter_ns() - start) / 1000)
        actual = ttnn.to_torch(out).squeeze(0).float()
        if args.save_output_tensors:
            saved_outputs["prefill"] = actual
            saved_outputs["decode"] = []
        passing, pcc = comp_pcc(ref, actual, 0.995)
        result = dict(
            cache_extent=cache_extent,
            cache_pages=pages,
            decoder=args.decoder,
            fusion=decoder.fusion if args.decoder == "fused" else None,
            runtime_prefill_audit="clean",
            program_cache_miss_guard=args.verify_program_cache,
            prefill_cache_entries=prefill_cache_entries,
            prefix_length=args.prefix_length,
            layer_type=layer_type,
            phase="prefill",
            length=args.length,
            real_weights=args.real,
            pcc=float(pcc),
            passed=bool(passing),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if prefill_host_us:
            result["warmed_prefill_host_us"] = prefill_host_us
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        if fixture:
            result["input_fixture"] = str(args.input_fixture)
            result["input_fixture_sha256"] = fixture_sha256
            result["input_source"] = fixture["metadata"]
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        if args.decode:
            hf_stages = {}
            if args.diagnostic:
                for name, module in hf.named_modules():
                    if name in [
                        "self_attn",
                        "post_attention_layernorm",
                        "mlp",
                        "router",
                        "experts",
                        "post_feedforward_layernorm_1",
                        "post_feedforward_layernorm_2",
                        "post_feedforward_layernorm",
                    ]:

                        def hook(module, inp, out, name=name):
                            hf_stages[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()

                        module.register_forward_hook(hook)
            dx = (
                fixture["decode"][:, :1].float()
                if fixture
                else torch.randn(1, 1, config.hidden_size).bfloat16().float()
            )
            pos = args.length
            dmask = torch.zeros(1, 1, 1, pos + 1)
            if layer_type == "sliding_attention":
                dmask[..., : max(0, pos + 1 - config.sliding_window)] = float("-inf")
            with torch.no_grad():
                dref = hf(
                    dx,
                    position_embeddings=(cos[:, pos : pos + 1], sin[:, pos : pos + 1]),
                    attention_mask=dmask,
                    past_key_values=hf_cache,
                )
            dt = device(dx[None])
            decode_rope_layout = getattr(decoder, "decode_rope_layout", ttnn.TILE_LAYOUT)
            dr = tuple(device(t.squeeze(0), layout=decode_rope_layout) for t in (cos, sin))
            result["decode_rope_layout"] = str(decode_rope_layout)
            p = torch.zeros(1, 32, dtype=torch.int32)
            p[0, 0] = pos
            cp = torch.full((1,), -1, dtype=torch.int32)
            cp[0] = pos
            ptpos = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ctpos = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

            def forward():
                return decoder.decode_forward(
                    dt, rope_mats=dr, current_pos=ptpos, cache_pos=ctpos, page_table=pt, kv_cache=cache
                )

            if args.diagnostic:

                class Probe:
                    def __init__(self, wrapped, name):
                        self.wrapped, self.name = wrapped, name

                    def __getattr__(self, name):
                        return getattr(self.wrapped, name)

                    def __call__(self, *a, **kw):
                        out = self.wrapped(*a, **kw)
                        actual = ttnn.to_torch(out).float().reshape(-1)
                        ref_stage = hf_stages[self.name].float().reshape(-1)
                        print("STAGE", self.name, comp_pcc(ref_stage, actual, 0.995), flush=True)
                        return out

                originals = {}
                for name in ["self_attn", "shared_mlp"]:
                    original = getattr(decoder.layer, name)
                    originals[name] = original
                    setattr(decoder.layer, name, Probe(original, "mlp" if name == "shared_mlp" else name))
                norm_originals = {}
                for name in [
                    "post_attention_layernorm",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                    "post_feedforward_layernorm",
                ]:
                    norm = getattr(decoder.layer, name)
                    norm_originals[name] = norm.forward
                    norm.forward = Probe(norm.forward, name)
            warm = forward()
            if args.diagnostic:
                for name, original in originals.items():
                    setattr(decoder.layer, name, original)
                for name, original in norm_originals.items():
                    getattr(decoder.layer, name).forward = original
            ttnn.synchronize_device(mesh)
            warm.deallocate(True)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            if args.verify_program_cache:
                mesh.set_program_cache_misses_allowed(False)
            try:
                with device_only():
                    traced = forward()
            finally:
                mesh.set_program_cache_misses_allowed(True)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            da = ttnn.to_torch(traced).squeeze(0).float()
            dpass, dpcc = comp_pcc(dref, da, 0.995)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            repeat = ttnn.to_torch(traced).squeeze(0).float()
            result["decode"] = dict(
                pcc=float(dpcc),
                passed=bool(dpass),
                traced=True,
                runtime_decode_audit="clean",
                repeated_equal=torch.equal(da, repeat),
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(result, flush=True)
            decode_pccs = [float(dpcc)]
            decode_checks = [dict(position=pos, pcc=float(dpcc), passed=bool(dpass))]
            # HF and input preparation are outside the measured device window.
            decode_inputs, decode_refs = [dx], [dref]
            for step in range(1, args.steps):
                next_x = (
                    fixture["decode"][:, step : step + 1].float()
                    if fixture
                    else torch.randn_like(dx).bfloat16().float()
                )
                step_pos = pos + step
                step_mask = torch.zeros(1, 1, 1, step_pos + 1)
                if layer_type == "sliding_attention":
                    step_mask[..., : max(0, step_pos + 1 - config.sliding_window)] = float("-inf")
                with torch.no_grad():
                    next_ref = hf(
                        next_x,
                        position_embeddings=(cos[:, step_pos : step_pos + 1], sin[:, step_pos : step_pos + 1]),
                        attention_mask=step_mask,
                        past_key_values=hf_cache,
                    )
                decode_inputs.append(next_x)
                decode_refs.append(next_ref)

            def refresh(t, dest, dtype, layout):
                host = ttnn.from_torch(t, dtype=dtype, layout=layout)
                ttnn.copy_host_to_device_tensor(host, dest)

            if args.profile:
                ttnn.synchronize_device(mesh)
                signpost("PERF_DECODE")
            # The profile loop exercises the required successive positions;
            # no readback occurs within the measured loop.
            for step, next_x in enumerate(decode_inputs):
                p[0, 0], cp[0] = pos + step, pos + step
                refresh(next_x[None], dt, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                refresh(p, ptpos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                refresh(cp, ctpos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                if not args.profile:
                    got = ttnn.to_torch(traced).squeeze(0).float()
                    if args.save_output_tensors:
                        saved_outputs["decode"].append(got)
                    ok, value = comp_pcc(decode_refs[step], got, 0.995)
                    decode_pccs.append(float(value))
                    decode_checks.append(dict(position=pos + step, pcc=float(value), passed=bool(ok)))
                    dpass = dpass and ok
            ttnn.synchronize_device(mesh)
            if args.profile:
                signpost("PERF_DECODE_END")
                got = ttnn.to_torch(traced).squeeze(0).float()
                ok, value = comp_pcc(decode_refs[-1], got, 0.995)
                decode_pccs.append(float(value))
                decode_checks.append(dict(position=pos + args.steps - 1, pcc=float(value), passed=bool(ok)))
                dpass = dpass and ok
            if args.timing:
                durations = []
                for repeat_index in range(5):
                    start = time.perf_counter_ns()
                    for _ in range(30):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    durations.append((time.perf_counter_ns() - start) / 30000)
                result["traced_decode_host_us"] = durations
            result["decode"].update(
                steps=args.steps,
                min_pcc=min(decode_pccs),
                passed=bool(dpass),
                positions=[pos, pos + args.steps - 1],
                checks=decode_checks,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(result, flush=True)
            ttnn.release_trace(mesh, trace)
            if args.save_output_tensors:
                torch.save(saved_outputs, args.save_output_tensors)
            assert dpass and torch.equal(da, repeat), result
        if args.save_output_tensors:
            torch.save(saved_outputs, args.save_output_tensors)
        assert passing, result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
