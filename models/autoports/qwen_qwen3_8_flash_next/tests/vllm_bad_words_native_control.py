# SPDX-License-Identifier: Apache-2.0
"""Fresh canonical TT control of a host-compatibility request; no vLLM scheduler.

Prepare only by default. --run requires exclusive serialized TT ownership.
This intentionally uses host logits/token feedback for the original host-only
bad_words request; it is neither the performance path nor a sampling substitute.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
if (
    os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1"
    or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
):
    raise RuntimeError("Trace and program-cache allocations must remain tracked")

import torch
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer
from vllm_tt_plugin.host_sampler import create_host_sampler

from models.autoports.qwen_qwen3_8_flash_next.doc.vllm_integration.bad_words_hf_control import close_sampler, metadata

ROOT = Path(__file__).resolve().parents[1]
BATCH, CONTEXT = 32, 262144


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    prepared_path = ROOT / "doc/vllm_integration/bad_words_hf_prepared.json"
    prepared = json.loads(prepared_path.read_text())
    live = json.loads(args.live_artifact.read_text())
    observed = next(row for row in live["responses"] if row["seed"] == 3)["raw_response"]
    ids = observed["prompt_token_ids"]
    if ids != prepared["prompt_ids"] or len(ids) != 56:
        raise ValueError("Expected the exact captured56-token seed3 prompt")
    tokenizer = get_tokenizer(str(args.checkpoint), local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        prepared["request"]["messages"], tokenize=False, add_generation_prompt=True
    )
    if tokenizer.encode(rendered, add_special_tokens=False) != ids:
        raise ValueError("Current tokenizer/template differs from the observed prompt")
    params = ChatCompletionRequest(**prepared["request"]).to_sampling_params(
        max_tokens=100, default_sampling_params=prepared["generation_config_diff"]
    )
    params.update_from_tokenizer(tokenizer)
    if (params.seed, params.top_k, params.top_p, params.temperature) != (3, 20, 1.0, 1.0):
        raise ValueError("Control sampling differs from the original request")
    live_runtime = json.loads(args.live_runtime.read_text())
    public_pool = CONTEXT + BATCH * 32
    if (live_runtime["batch"], live_runtime["context"], live_runtime["public_pool_tokens"]) != (
        BATCH,
        CONTEXT,
        public_pool,
    ):
        raise ValueError("Live runtime geometry differs from the canonical control")
    expected_chunks = [{"start": start, "length": 8, "slot": args.slot} for start in range(0, 56, 8)]
    slots = [args.slot] + [slot for slot in range(BATCH) if slot != args.slot]
    if sorted(slots) != list(range(BATCH)):
        raise ValueError("Canonical request slots must be a complete unique permutation")
    result = dict(
        status="prepared_not_run",
        prompt=rendered,
        prompt_ids=ids,
        request=prepared["request"],
        resolved_sampling=repr(params),
        scope="Fresh48-layer canonical TT model/generator; host-compatibility correctness only",
        batch=BATCH,
        context=CONTEXT,
        public_pool_tokens=public_pool,
        physical_target_row=0,
        target_state_slot=args.slot,
        request_slot_permutation=slots,
        other_rows="31 inactive, private scratch slots/pages",
        page_mapping="Six reversed public pages; original live physical pages were not recorded",
        slot_mapping_limit="Original seed3 likely occupied slot3, but request-ID to slot evidence is absent",
        expected_native_chunks=expected_chunks,
        observed_tt_response=observed,
        steps=[],
        generated_ids=[],
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                prepared_path,
                args.live_artifact,
                args.live_runtime,
                ROOT / "tt/generator.py",
                ROOT / "tt/model.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/qsa.py",
                ROOT / "tt/paged_qsa.py",
                ROOT / "tt/precision.py",
                ROOT / "doc/vllm_integration/bad_words_hf_control.py",
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")

    persist()
    if not args.run:
        print("NATIVE_CONTROL_PREPARED", args.output, flush=True)
        return
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in (None, "0"):
        raise RuntimeError("Disable profiling for this correctness control")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    if (
        os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise RuntimeError("Trace and program-cache allocations must remain tracked")
    events = args.output.with_suffix(".events.jsonl")
    if events.exists():
        raise FileExistsError(events)
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(events)
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import QwenFlashNextGenerator
    from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenFlashNextModel
    from models.autoports.qwen_qwen3_8_flash_next.tt.precision import runtime_summary

    mesh, generator, sampler = None, None, None
    started = time.perf_counter()
    result["status"] = "running"
    persist()
    try:
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        # build_generator currently lacks attention_pool_tokens forwarding. Use
        # its canonical model/generator classes directly, with the proven public
        # pool geometry, without any adapter or engine request transformation.
        model = QwenFlashNextModel(
            args.checkpoint,
            mesh,
            batch_size=BATCH,
            max_seq_len=CONTEXT,
            attention_pool_tokens=public_pool,
            precision_config=ROOT / "doc/datatype_sweep/selected_precision_config.json",
            cache_dir="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple",
        )
        generator = QwenFlashNextGenerator(
            model,
            AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True),
            sampling_mode="host",
            prefill_trace_max=8,
            last_token_only=True,
            trace_cache_reset=False,
        )
        cache = model.cache
        generator.prepare(kv_cache=cache)
        actual_runtime = runtime_summary(model)
        if actual_runtime != live_runtime["runtime"]:
            raise AssertionError("Native precision/runtime metadata differs from observed server")
        shapes = [
            {name: list(tensor.shape) for name, tensor in row.items() if name != "ple_cache"} for row in cache.layers
        ]
        if shapes != live_runtime["cache_shapes"] or len(model.layers) != 48:
            raise AssertionError("Native layers/cache geometry differs from observed server")
        result.update(
            runtime=actual_runtime,
            cache_shapes=shapes,
            prepared_memory=view(mesh),
            trace_ids=dict(generator.traces),
            model_prepare_seconds=time.perf_counter() - started,
        )
        generator.reset()
        generated = result["generated_ids"]
        sampler = create_host_sampler()
        sampling = metadata(params, generated)
        pages = torch.zeros(BATCH, CONTEXT // 32, dtype=torch.int32)
        pages[0, :6] = torch.arange(public_pool // 32 - 1, public_pool // 32 - 7, -1)
        active = [True] + [False] * (BATCH - 1)
        tokens = torch.zeros(BATCH, len(ids), dtype=torch.int32)
        tokens[0] = torch.tensor(ids, dtype=torch.int32)
        event_offset = len(events.read_text().splitlines())
        before = dict(generator.counters)
        tick = time.perf_counter()
        logits = generator.prefill_forward(
            tokens,
            page_table=pages,
            kv_cache=cache,
            prompt_lens=[56] + [0] * (BATCH - 1),
            start_pos=[0] * BATCH,
            slots=slots,
            active_rows=active,
        )
        rows = [json.loads(line) for line in events.read_text().splitlines()[event_offset:]]
        chunks = [
            {"start": row["start_positions"][0], "length": row["chunk_length"], "slot": row["slots"][0]}
            for row in rows
            if row["event"] == "prefill_chunk" and row["active"][0]
        ]
        result["observed_native_chunks"] = chunks
        if chunks != expected_chunks:
            raise AssertionError("Actual physical prefill chunks differ from the56-token control schedule")
        observed_ids = observed["choices"][0]["token_ids"]
        stop_ids = set(prepared["eos_token_ids"])
        for step in range(100):
            values = logits.reshape(BATCH, -1)[0:1].contiguous()
            if values.shape[-1] != model.cfg.vocab_size or not torch.isfinite(values).all():
                raise AssertionError("Malformed/nonfinite canonical host logits")
            top = torch.topk(values[0].float(), 20)
            token = int(sampler(values.clone(), sampling).sampled_token_ids.item())
            generated.append(token)
            for pattern in params.bad_words_token_ids:
                if generated[-len(pattern) :] == pattern:
                    raise AssertionError("Native control emitted a forbidden token sequence")
            result["steps"].append(
                dict(
                    step=step,
                    token_id=token,
                    text=tokenizer.decode([token]),
                    seconds=time.perf_counter() - tick,
                    raw_top20_ids=top.indices.tolist(),
                    raw_top20_logits=top.values.tolist(),
                    observed_tt_token_id=observed_ids[step] if step < len(observed_ids) else None,
                )
            )
            result["output_text"] = tokenizer.decode(generated, skip_special_tokens=False)
            persist()
            print("NATIVE_TOKEN", step, token, repr(tokenizer.decode([token])), flush=True)
            if token in stop_ids or step == 99:
                result["finish_reason"] = "stop" if token in stop_ids else "length"
                break
            tick = time.perf_counter()
            feedback = torch.zeros(BATCH, dtype=torch.int32)
            feedback[0] = token
            logits = generator.decode_forward(
                feedback,
                [56 + step] + [0] * (BATCH - 1),
                page_table=pages,
                kv_cache=cache,
                slots=slots,
                active_rows=active,
            )
        result["counter_delta"] = {key: value - before.get(key, 0) for key, value in generator.counters.items()}
        delta = result["counter_delta"]
        if delta.get("logits_readbacks", 0) != len(generated):
            raise AssertionError("Expected exactly one explicit host logit read per sampled token")
        if any(delta.get(mode + "_replays", 0) for mode in ("greedy", "sample", "argmax")):
            raise AssertionError("The host-only request unexpectedly consumed a device sampling trace")
        if delta.get("token_refreshes", 0) != len(generated) - 1:
            raise AssertionError("Expected explicit compatibility token feedback on every decode step")
        result["exact_observed_tokens"] = generated == observed_ids
        result["first_divergence"] = next(
            (i for i, (a, b) in enumerate(zip(generated, observed_ids)) if a != b),
            min(len(generated), len(observed_ids)) if len(generated) != len(observed_ids) else None,
        )
        result["classification"] = (
            "Observed token sequence reproduced without serving scheduler or companion requests"
            if result["exact_observed_tokens"]
            else "Difference remains unresolved; inspect row/page/control details and first divergence"
        )
        result["status"] = "completed_needs_qualitative_review"
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        try:
            if sampler is not None:
                close_sampler(sampler)
            if generator is not None:
                ttnn.synchronize_device(mesh)
                generator.close()
                result["shutdown_counters"] = dict(generator.counters)
        finally:
            result["closed_mesh"] = False
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
                    result["closed_mesh"] = True
            finally:
                persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540"),
    )
    parser.add_argument("--live-artifact", type=Path, default=ROOT / "doc/vllm_integration/bad_words_targeted_100.json")
    parser.add_argument(
        "--live-runtime", type=Path, default=ROOT / "readiness_vllm/full32_scratch_tracked_runtime.json"
    )
    parser.add_argument("--slot", type=int, choices=range(32), default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trace-region-size", type=int, default=2500000000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    main(parser.parse_args())
