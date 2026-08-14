# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every full-model evidence run, over **one** 52-layer build.

Each readiness runner calls ``build_generator`` itself, and building the 52-layer
stack takes tens of minutes (~484 M parameters per layer, read, transposed and
packed into BFP4/BFP8 on the host).  ``build_generator`` therefore memoises the
generator per (mesh, config), and this driver runs every stage in one process over
one build.

The runners themselves are the stock ones -- ``run_prefill_check``,
``run_teacher_forcing``, ``run_autoregressive`` -- called through their
programmatic entry points, unmodified.  Two shims are needed and both are narrow:
``AutoModelForCausalLM`` does not know ``MuseGlimmerConfig``, and the default HF
revision for this repo is metadata-only (see ``readiness_cli.py``).

Stages::

    capacity      DRAM footprint and the context contract's measured fields
    prefill       models.common.readiness_check.run_prefill_check
    teacher       models.common.readiness_check.run_teacher_forcing
    autoregress   models.common.readiness_check.run_autoregressive (chat + raw)
    shapes        non-aligned prompt lengths, long prompts, page-table cases
    sampling      split-sampling trace contract + determinism
    perf          TTFT, token-out decode, traced logits-only decode
    fallback      runtime fallback audit over the measured paths

Usage::

    python doc/full_model/bench/evidence.py --stages all
    python doc/full_model/bench/evidence.py --stages perf,sampling --max-seq-len 131072
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    GREEDY,
    build_generator,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.model import dram_capacity_bytes  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/full_model"
STAGES = ("capacity", "prefill", "misses", "teacher", "autoregress", "shapes", "sampling", "perf", "fallback")


def say(*args) -> None:
    print(*args, flush=True)


def register_hf_shims(model_id: str) -> str:
    """``readiness_cli``'s two shims, applied in-process."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

    from models.autoports.meta_models_muse_glimmer_30b.tt.model import weights_snapshot_dir

    AutoModelForCausalLM._model_mapping._extra_content[MuseGlimmerConfig] = MuseGlimmerForConditionalGeneration
    snapshot = str(weights_snapshot_dir(model_id))

    def redirect(args, kwargs):
        args = tuple(snapshot if a == model_id else a for a in args)
        if kwargs.get("pretrained_model_name_or_path") == model_id:
            kwargs["pretrained_model_name_or_path"] = snapshot
        kwargs.setdefault("local_files_only", True)
        return args, kwargs

    original_model = AutoModelForCausalLM.from_pretrained
    original_tok = AutoTokenizer.from_pretrained

    def patched_model(*a, **k):
        a, k = redirect(a, k)
        k.setdefault("dtype", torch.bfloat16)
        return original_model(*a, **k)

    def patched_tok(*a, **k):
        a, k = redirect(a, k)
        return original_tok(*a, **k)

    AutoModelForCausalLM.from_pretrained = patched_model
    AutoTokenizer.from_pretrained = patched_tok
    return snapshot


def capture_stdout(fn, *args, **kwargs):
    """Run ``fn`` and return ``(result, printed_text)``; the runners print their metrics."""
    buffer = io.StringIO()

    class _Tee(io.TextIOBase):
        def write(self, text):
            buffer.write(text)
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
            return len(text)

    with contextlib.redirect_stdout(_Tee()):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


# ------------------------------------------------------------------- stages


def stage_capacity(generator, summary: dict) -> None:
    report = generator.capability_report()
    config = generator.model.config
    per_layer_blocks_bytes = report["per_device_kv_cache_bytes"] / max(config.max_num_blocks, 1)
    capacity = dram_capacity_bytes(generator.mesh_device)
    long_lived = report["per_device_total_bytes"]
    report["per_device_kv_cache_bytes_per_block"] = per_layer_blocks_bytes
    report["per_device_free_after_long_lived_bytes"] = capacity - long_lived
    report["full_context_sequences_that_fit"] = int(
        (capacity - (long_lived - report["per_device_kv_cache_bytes"])) // max(report["per_device_kv_cache_bytes"], 1)
    )
    summary["capacity"] = report
    for key, value in report.items():
        say(f"EV capacity {key}={value}")


def references(args) -> list[str]:
    """``--reference`` is a comma-separated list, so one build can be scored against
    several references -- the bf16 one and the fp32 control, for instance."""
    return [name.strip() for name in args.reference.split(",") if name.strip()]


def stage_prefill(generator, summary: dict, args) -> None:
    from models.common.readiness_check.run_prefill_check import run_prefill_check

    summary["prefill_check_by_reference"] = {}
    for name in references(args):
        say(f"EV prefill reference={name}")
        stats, text = capture_stdout(
            run_prefill_check,
            model_dir=ROOT,
            reference_path=ROOT / name,
            mesh_device=generator.mesh_device,
        )
        summary["prefill_check_by_reference"][name] = {"per_entry": stats, "output": text}
        (OUT / f"logs/run_prefill_check_{pathlib.Path(name).stem}.txt").write_text(text)
    primary = references(args)[0]
    summary["prefill_check"] = summary["prefill_check_by_reference"][primary]
    (OUT / "logs/run_prefill_check.txt").write_text(summary["prefill_check"]["output"])


def stage_misses(generator, summary: dict, args) -> None:
    """Where the prefill check's top-k misses are, and how far off they are.

    ``run_prefill_check`` reports only the rates.  A top-100 miss out of a
    202048-wide vocab is a big miss, so the useful diagnostic is *which* positions
    miss and what HF rank the TT token actually has: a couple of late positions at
    rank 100-500 is a precision-margin story, an early cluster or rank >10000 is a
    wrapper or indexing story.
    """
    from models.common.readiness_check.schema import load_reference

    reference = load_reference(str(ROOT / references(args)[0]))
    entry = reference.entries[0]
    prompt_len = int(entry.tf_prompt_len)
    gen_len = int(entry.generated_tokens.shape[1])
    tokens = torch.cat([entry.prompt_tokens[0], entry.generated_tokens[0]]).unsqueeze(0)

    generator.reset()
    logits = generator.prefill_forward(
        tokens=tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=[int(tokens.shape[1])],
        return_all_logits=True,
    )
    # Same alignment as generate.py and run_prefill_check: logits[0, i] predicts i+1.
    window = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :].float()
    tt_pred = torch.argmax(window, dim=-1)
    topk = entry.topk_tokens  # [gen_len, k], HF's ranked ids
    rows = []
    for index in range(gen_len):
        ref = topk[index].tolist()
        token = int(tt_pred[index])
        rank = ref.index(token) if token in ref else -1
        if rank == 0:
            continue
        tt_top5 = torch.topk(window[index], k=5)
        rows.append(
            {
                "gen_index": index,
                "absolute_position": prompt_len + index,
                "tt_token": token,
                "hf_rank_of_tt_token": rank,
                "hf_top1": int(ref[0]),
                "tt_top5": [int(t) for t in tt_top5.indices.tolist()],
                "tt_top5_values": [round(float(v), 4) for v in tt_top5.values.tolist()],
                "tt_top1_minus_top2": round(float(tt_top5.values[0] - tt_top5.values[1]), 4),
            }
        )
    outside = [row for row in rows if row["hf_rank_of_tt_token"] < 0]
    summary["prefill_misses"] = {
        "gen_len": gen_len,
        "prompt_len": prompt_len,
        "non_top1_positions": len(rows),
        "outside_top_k_positions": len(outside),
        "k": reference.k,
        "rows": rows,
    }
    say(f"EV misses non_top1={len(rows)}/{gen_len} outside_top{reference.k}={len(outside)}")
    for row in rows:
        say(f"EV misses {row}")


def stage_teacher(generator, summary: dict, args) -> None:
    from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

    summary["teacher_forcing_by_reference"] = {}
    for name in references(args):
        say(f"EV teacher reference={name}")
        generator.reset()
        stats, text = capture_stdout(
            run_teacher_forcing,
            model_dir=ROOT,
            reference_path=ROOT / name,
            mesh_device=generator.mesh_device,
        )
        summary["teacher_forcing_by_reference"][name] = {
            "per_entry": stats,
            "output": text,
            "counters": dict(generator.counters),
        }
        (OUT / f"logs/run_teacher_forcing_{pathlib.Path(name).stem}.txt").write_text(text)
    primary = references(args)[0]
    summary["teacher_forcing"] = summary["teacher_forcing_by_reference"][primary]
    (OUT / "logs/run_teacher_forcing.txt").write_text(summary["teacher_forcing"]["output"])


def stage_autoregress(generator, summary: dict, args) -> None:
    from models.common.readiness_check.run_autoregressive import run_autoregressive

    runs = {
        "chat": OUT / "prompts/autoregressive_chat_prompt.txt",
        "raw": REPO / "models/common/readiness_check/autoregressive_prompt.txt",
    }
    summary["autoregressive"] = {}
    for label, prompt_file in runs.items():
        output_dir = ROOT / f"readiness_autoregressive_{label}"
        paths, text = capture_stdout(
            run_autoregressive,
            model_dir=ROOT,
            hf_model_id=args.hf_model,
            prompt_file=prompt_file,
            mesh_device=generator.mesh_device,
            output_dir=output_dir,
            max_new_tokens=args.autoregressive_tokens,
        )
        summary["autoregressive"][label] = {
            "output_dir": str(output_dir.relative_to(ROOT)),
            "prompt_file": str(prompt_file),
            "paths": {k: str(v) for k, v in paths.items()},
        }
        (OUT / f"logs/run_autoregressive_{label}.txt").write_text(text)


def stage_shapes(generator, summary: dict, args) -> None:
    """Prompt lengths that are not divisible by tile, page, chunk or trace sizes."""
    model = generator.model
    vocab = model.config.vocab_size
    torch.manual_seed(11)
    lengths = [int(n) for n in args.shape_lengths.split(",")]
    results = []
    for length in lengths:
        if length > model.config.max_seq_len - 4:
            say(f"EV shapes skip {length}: beyond the supported context")
            continue
        prompt = [int(t) for t in torch.randint(0, vocab, (length,)).tolist()]
        generator.reset()
        started = time.perf_counter()
        tokens = generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
        elapsed = time.perf_counter() - started
        aligned = {
            "tile": length % 32 == 0,
            "page": length % model.config.page_block_size == 0,
            "chunk": length % model.config.prefill_chunk_size == 0,
        }
        results.append({"prompt_len": length, "tokens": tokens, "seconds": round(elapsed, 3), "aligned": aligned})
        say(f"EV shapes prompt_len={length} aligned={aligned} tokens={tokens} in {elapsed:.2f}s")
    summary["prompt_shapes"] = results


def stage_sampling(generator, summary: dict, args) -> None:
    """The split-sampling trace contract, asserted on the tensors themselves."""
    model = generator.model
    vocab = model.config.vocab_size
    torch.manual_seed(13)
    prompt = [int(t) for t in torch.randint(0, vocab, (128,)).tolist()]
    out: dict = {}

    generator.reset()
    generator.reset_counters()
    first = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    generator.reset()
    generator.reset_counters()
    again = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    out["deterministic_across_calls"] = first == again
    out["steady_state_counters"] = dict(generator.counters)
    say(f"EV sampling deterministic={first == again} counters={generator.counters}")

    # tt_out_tok identity: the sampler's output tensor *is* the decode token input.
    slot = next(iter(generator.sampling._trace_states.values()))
    token_input = generator._device_inputs["tokens"]
    out["sampling_trace_captured"] = slot["id"] is not None
    out["tt_out_tok_is_decode_token_input"] = slot["output"][0] is token_input
    out["sampling_trace_logits_is_decode_trace_output"] = slot["input"] is generator._trace_logits
    say(f"EV sampling trace_id={slot['id']} tt_out_tok_is_token_input={out['tt_out_tok_is_decode_token_input']}")
    say(f"EV sampling logits_identity={out['sampling_trace_logits_is_decode_trace_output']}")

    # Two replays with different token/position values must change the inputs the
    # trace reads and the token it produces, with no host staging in between.
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
    generator._stage(tokens=[prompt[-1]] * 32, positions=torch.full((32,), len(prompt), dtype=torch.int64))
    before_tokens = ttnn.to_torch(ttnn.get_device_tensors(token_input)[0]).reshape(-1)[:1].tolist()
    before_pos = (
        ttnn.to_torch(ttnn.get_device_tensors(generator._device_inputs["current_pos"])[0]).reshape(-1)[:1].tolist()
    )
    staged = dict(generator.counters)
    step1 = generator._decode_step_traced(host_sampling=False)
    mid_tokens = ttnn.to_torch(ttnn.get_device_tensors(token_input)[0]).reshape(-1)[:1].tolist()
    mid_pos = (
        ttnn.to_torch(ttnn.get_device_tensors(generator._device_inputs["current_pos"])[0]).reshape(-1)[:1].tolist()
    )
    step2 = generator._decode_step_traced(host_sampling=False)
    after_tokens = ttnn.to_torch(ttnn.get_device_tensors(token_input)[0]).reshape(-1)[:1].tolist()
    after_pos = (
        ttnn.to_torch(ttnn.get_device_tensors(generator._device_inputs["current_pos"])[0]).reshape(-1)[:1].tolist()
    )
    out["two_step_replay"] = {
        "token_before": before_tokens,
        "token_after_step1": mid_tokens,
        "token_after_step2": after_tokens,
        "pos_before": before_pos,
        "pos_after_step1": mid_pos,
        "pos_after_step2": after_pos,
        "sampled_step1": [int(step1[0])],
        "sampled_step2": [int(step2[0])],
        "token_feedback_is_device_side": mid_tokens == [int(step1[0])],
        "position_advanced_on_device": mid_pos[0] == before_pos[0] + 1 and after_pos[0] == before_pos[0] + 2,
        "host_staging_between_replays": {
            key: dict(generator.counters)[key] - staged[key]
            for key in ("token_refreshes", "position_refreshes", "page_table_refreshes")
        },
    }
    for key, value in out["two_step_replay"].items():
        say(f"EV sampling two_step {key}={value}")

    # Greedy is the top-k op path, not force-argmax; record the alternative too.
    out["force_argmax_enabled"] = bool(generator.sampling.tt_sampling.force_argmax_sampling)
    out["sampling_params_greedy"] = {
        "top_k": generator._sampling_params.top_k[0],
        "top_p": generator._sampling_params.top_p[0],
        "temperature_reciprocal": generator._sampling_params.temperature[0],
    }
    say(f"EV sampling greedy_params={out['sampling_params_greedy']} force_argmax={out['force_argmax_enabled']}")

    # A non-greedy sampling mode must run through the same path.
    from models.common.sampling.generator import SamplingParams

    generator.reset()
    sampled = generator.generate(
        prompt_token_ids=prompt,
        max_new_tokens=8,
        enable_trace=True,
        sampling_params=SamplingParams(temperature=0.8, top_k=32, top_p=0.95),
    )
    out["top_k_top_p_tokens"] = sampled
    out["top_k_top_p_differs_from_greedy"] = sampled != first
    say(f"EV sampling top_k_top_p tokens={sampled}")
    # ...and greedy after it must still be greedy (trace keyed correctly).
    generator.reset()
    back_to_greedy = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    out["greedy_after_sampled_matches"] = back_to_greedy == first
    say(f"EV sampling greedy_after_sampled_matches={back_to_greedy == first}")
    summary["split_sampling"] = out


def stage_perf(generator, summary: dict, args) -> None:
    """Batch-1 TTFT and decode, at the vLLM primary single-user profile."""
    model = generator.model
    vocab = model.config.vocab_size
    torch.manual_seed(17)
    prompt_len, gen_len = args.perf_prompt_len, args.perf_gen_len
    prompt = [int(t) for t in torch.randint(0, vocab, (prompt_len,)).tolist()]
    out: dict = {"workload": {"prompt_len": prompt_len, "gen_len": gen_len, "batch": 1}}

    # Warm every program and capture the traces before anything is timed.
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)

    # ---- TTFT: prefill plus the first sampled token, as a caller sees it.
    ttfts = []
    for _ in range(args.perf_rounds):
        generator.reset()
        ttnn.synchronize_device(generator.mesh_device)
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
        ttfts.append((time.perf_counter() - started) * 1e3)
    out["ttft_ms"] = {"min": min(ttfts), "mean": sum(ttfts) / len(ttfts), "rounds": ttfts}
    say(f"EV perf TTFT prompt={prompt_len} min={min(ttfts):.2f} ms mean={sum(ttfts)/len(ttfts):.2f} ms")

    # ---- token-out decode: the whole generate loop minus prefill/TTFT.
    # Difference of two *separately* timed windows, so neither timer contains the
    # cache zeroing that reset() does between them: (prompt + gen_len tokens) minus
    # (prompt + 1 token) is exactly gen_len-1 decode steps.
    token_out = []
    for _ in range(args.perf_rounds):
        generator.reset()
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
        one_token_s = time.perf_counter() - started
        generator.reset()
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=gen_len, enable_trace=True)
        full_s = time.perf_counter() - started
        token_out.append((full_s - one_token_s) / (gen_len - 1) * 1e3)
    out["token_out_decode_ms_per_token"] = {
        "min": min(token_out),
        "mean": sum(token_out) / len(token_out),
        "rounds": token_out,
    }
    out["token_out_decode_tok_s_u"] = 1e3 / min(token_out)
    say(f"EV perf token-out decode min={min(token_out):.3f} ms/token -> {1e3/min(token_out):.2f} t/s/u")

    # ---- traced logits-only decode: replay the decode trace alone, no sampling,
    # no token readback.  This is the fair comparison to a PERF.md-style
    # decoder-stack number and to the teacher-forcing reference.
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
    replays = args.perf_replays
    logits_only = []
    for _ in range(args.perf_rounds):
        ttnn.synchronize_device(generator.mesh_device)
        started = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(generator.mesh_device, generator._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(generator.mesh_device)
        logits_only.append((time.perf_counter() - started) / replays * 1e3)
    out["traced_decode_logits_only_ms_per_token"] = {
        "min": min(logits_only),
        "mean": sum(logits_only) / len(logits_only),
        "replays_per_round": replays,
        "rounds": logits_only,
    }
    out["traced_decode_logits_only_tok_s_u"] = 1e3 / min(logits_only)
    say(f"EV perf traced logits-only min={min(logits_only):.3f} ms/token -> {1e3/min(logits_only):.2f} t/s/u")

    # ---- the sampling trace on its own, so the terminal cost is attributable.
    sampling_only = []
    slot = next(iter(generator.sampling._trace_states.values()))
    for _ in range(args.perf_rounds):
        ttnn.synchronize_device(generator.mesh_device)
        started = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(generator.mesh_device, slot["id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(generator.mesh_device)
        sampling_only.append((time.perf_counter() - started) / replays * 1e3)
    out["sampling_trace_ms_per_token"] = {"min": min(sampling_only), "mean": sum(sampling_only) / len(sampling_only)}
    say(f"EV perf sampling trace min={min(sampling_only):.3f} ms/token")

    # ---- the layer-stack lower bound this has to be compared against.
    out["layer_stack_lower_bound_ms_per_token"] = {
        "sliding_layers": 39,
        "full_layers": 13,
        "sliding_ms_per_layer": 0.4546,
        "full_ms_per_layer": 0.4238,
        "total_ms": 39 * 0.4546 + 13 * 0.4238,
        "source": "doc/optimized_multichip_decoder/README.md traced decode @2048, e2e host timing",
    }
    say(f"EV perf layer-stack lower bound = {out['layer_stack_lower_bound_ms_per_token']['total_ms']:.2f} ms/token")
    summary["performance"] = out


def stage_fallback(generator, summary: dict, args) -> None:
    """Audit the measured paths for host fallback and host-logit boundaries."""
    model = generator.model
    audit = {
        "decode_trace_captured": generator._trace_id is not None,
        "sampling_trace_captured": generator._sampling_captured,
        "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
        "kv_cache_owner": "model layers (generator.model.kv_cache exposes them; set_kv_cache() binds external ones)",
        "host_logit_boundaries": [
            "prefill_forward(): logits are read to host by contract (the readiness prefill check needs them)",
            "generate(host_sampling=True): explicit compatibility mode, gathers logits and argmaxes on host",
            "decode_forward(sample_on_device=False): returns host logits, for callers that sample themselves",
        ],
        "token_out_path_host_work_per_token": [
            "one 32-uint32 readback of the sampled token, which the caller asked for",
        ],
        "reset_behaviour": "zeroes the paged cache, drops the page-table memo and the staging state; keeps weights and traces",
    }
    # Prove the claim rather than assert it: run the steady-state loop and read the
    # counters that would be non-zero if anything host-side were happening.
    vocab = model.config.vocab_size
    torch.manual_seed(19)
    prompt = [int(t) for t in torch.randint(0, vocab, (128,)).tolist()]
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
    generator.reset()
    generator.reset_counters()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=33, enable_trace=True)
    audit["counters_for_33_tokens"] = dict(generator.counters)
    audit["per_token_host_refreshes"] = {
        "token": (generator.counters["token_refreshes"] - 1) / 32,
        "position": (generator.counters["position_refreshes"] - 1) / 32,
        "page_table": generator.counters["page_table_refreshes"] / 32,
        "synchronizations": generator.counters["synchronizations"] / 32,
    }
    for key, value in audit.items():
        say(f"EV fallback {key}={value}")
    summary["fallback_audit"] = audit


# --------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="all")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument(
        "--reference",
        default="readiness_aime24_chat.refpt",
        help="comma-separated; the first is the primary and the rest are controls",
    )
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--lm-head-dtype", default="", help="bfloat4_b | bfloat8_b | bfloat16 (default: the shipped one)"
    )
    parser.add_argument("--lm-head-matmul", default="", help="dram_sharded | mcast1d")
    parser.add_argument("--lm-head-cores", type=int, default=0)
    parser.add_argument("--lm-head-in0-block-w", type=int, default=0)
    parser.add_argument("--lm-head-fidelity", default="", help="LoFi | HiFi2 | HiFi4")
    parser.add_argument("--lm-head-fp32-acc", action="store_true")
    parser.add_argument("--lm-head-output-dtype", default="", help="bfloat16 | float32")
    parser.add_argument(
        "--decoder-kv-cache-dtype",
        default="",
        help="attribution control only: raise the carried-forward KV-cache dtype (bfloat16). Never a shipped config.",
    )
    parser.add_argument(
        "--decoder-weight-dtype",
        default="",
        help="attribution control only: raise the carried-forward weight dtype (bfloat8_b). Never a shipped config.",
    )
    parser.add_argument("--autoregressive-tokens", type=int, default=128)
    parser.add_argument("--shape-lengths", default="1,37,127,129,2049,4097,8193,12345")
    parser.add_argument("--perf-prompt-len", type=int, default=128)
    parser.add_argument("--perf-gen-len", type=int, default=128)
    parser.add_argument("--perf-rounds", type=int, default=3)
    parser.add_argument("--perf-replays", type=int, default=32)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stages = STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(",") if s.strip())
    for stage in stages:
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; choose from {STAGES}")

    register_hf_shims(args.hf_model)
    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {
        "hf_model": args.hf_model,
        "stages": list(stages),
        "max_seq_len": args.max_seq_len,
        "cache_slots": args.max_batch_size,
        "layers": args.layers,
    }
    generator = None
    try:
        started = time.perf_counter()
        build_kwargs: dict = {}
        if args.lm_head_dtype:
            build_kwargs["lm_head_dtype"] = {
                "bfloat4_b": ttnn.bfloat4_b,
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[args.lm_head_dtype]
        if args.lm_head_matmul:
            build_kwargs["lm_head_matmul"] = args.lm_head_matmul
        if args.lm_head_cores:
            build_kwargs["lm_head_cores"] = args.lm_head_cores
        if args.lm_head_in0_block_w:
            build_kwargs["lm_head_in0_block_w"] = args.lm_head_in0_block_w
        dtypes = {
            "bfloat4_b": ttnn.bfloat4_b,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
            "float32": ttnn.float32,
        }
        if args.lm_head_fidelity:
            build_kwargs["lm_head_fidelity"] = {
                "LoFi": ttnn.MathFidelity.LoFi,
                "HiFi2": ttnn.MathFidelity.HiFi2,
                "HiFi4": ttnn.MathFidelity.HiFi4,
            }[args.lm_head_fidelity]
        if args.lm_head_fp32_acc:
            build_kwargs["lm_head_fp32_acc"] = True
        if args.lm_head_output_dtype:
            build_kwargs["lm_head_output_dtype"] = dtypes[args.lm_head_output_dtype]
        decoder_kwargs: dict = {}
        if args.decoder_kv_cache_dtype:
            decoder_kwargs["kv_cache_dtype"] = dtypes[args.decoder_kv_cache_dtype]
        if args.decoder_weight_dtype:
            decoder_kwargs["weight_dtype"] = dtypes[args.decoder_weight_dtype]
        if decoder_kwargs:
            # An attribution control, never a shipped configuration: it changes the
            # precision policy the decoder stage measured and this stage must carry
            # forward unchanged.
            build_kwargs["decoder_kwargs"] = decoder_kwargs
        summary["build_kwargs"] = {k: str(v) for k, v in build_kwargs.items()}
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            layer_indices=layer_indices,
            **build_kwargs,
        )
        summary["build_seconds"] = round(time.perf_counter() - started, 1)
        say(f"EV built in {summary['build_seconds']}s")

        for stage in stages:
            say(f"EV ===== stage {stage} =====")
            started = time.perf_counter()
            if stage == "capacity":
                stage_capacity(generator, summary)
            elif stage == "prefill":
                stage_prefill(generator, summary, args)
            elif stage == "misses":
                stage_misses(generator, summary, args)
            elif stage == "teacher":
                stage_teacher(generator, summary, args)
            elif stage == "autoregress":
                stage_autoregress(generator, summary, args)
            elif stage == "shapes":
                stage_shapes(generator, summary, args)
            elif stage == "sampling":
                stage_sampling(generator, summary, args)
            elif stage == "perf":
                stage_perf(generator, summary, args)
            elif stage == "fallback":
                stage_fallback(generator, summary, args)
            say(f"EV ===== stage {stage} done in {time.perf_counter()-started:.1f}s =====")
        say("EV_OK")
        return 0
    finally:
        name = args.out or f"evidence_{'_'.join(stages)}.json"
        path = OUT / name
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"EV summary -> {path}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
