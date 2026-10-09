# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Phase 0 bring-up: Gemma-4-E2B-it on the LTX DiT handle (full-shape submesh of the 4x8 Galaxy).

Opens the mesh with the LTX distilled ring params, carves ``full`` exactly like the pipeline, loads
Gemma4Generator on ``full`` and reproduces text_demo.py::_run_generation_via_generator with host
greedy sampling and no tracing anywhere. Never asserts on content; records every step's wall-clock
and the outcome to ``bringup_results.json`` under OUT_DIR ($LTX_ENHANCER_EXP_DIR, default ~/ltx_enhancer_experiments) (also on failure).

``test_e2b_decode_timing`` (same file) loads the generator once and times decode across arms: eager vs
traced decode x host vs device sampling. Results go to ``decode_timing_results.json`` in the same dir,
rewritten after every repeat. Knobs (env):

- ``E2B_ARMS``: comma list of ``eager-host,eager-device,trace-host,trace-device`` (default all, in that
  order: eager arms run before any trace is captured).
- ``E2B_REPEATS`` (default 3): prefill + decode per arm; repeat 0 carries the compile / trace capture.
- ``E2B_NEW_TOKENS`` (default 128): decode steps per repeat.
- ``E2B_IGNORE_STOP`` (default 1): decode the full budget past a stop token so every arm times the same
  number of steps; where the reply would have stopped is still recorded.
- ``E2B_TEMPERATURE`` (default 0 = greedy, so token parity across arms is checked); > 0 samples with the
  enhancer's top_k 64 / top_p 0.95 and ``E2B_SEED`` (default 10).
- ``E2B_PROMPT`` (default ``beekeeper``): user prompt, templated with the T2V system prompt.
- ``E2B_PROFILE=1``: for a Tracy run (``python -m tracy -p -r -v -m pytest ...``). Each repeat decodes
  1 + ``E2B_PROFILE_DECODE_STEPS`` (default 3) steps instead of ``E2B_NEW_TOKENS``. The device profiler is
  drained only at window boundaries: after load, after each prefill, after decode step 0 (trace capture on
  the cold repeat) and after the decode loop. On the last repeat, prefill is wrapped in
  ``prefill_start``/``prefill_stop`` signposts and decode steps 1..K in ``start``/``stop``. Every window must
  fit the per-core buffer, sized by ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` (default 1000 programs);
  each drain reads the whole buffer on every core of every chip, so keep that count no larger than needed.

Prefill is eager and host-sampled in every arm (PLI models never trace prefill); only decode varies.
Gemma4 runs its device sampler eagerly even under a decode trace (``_tt_disable_sampling_trace``).
``GEMMA4_CCL_ASYNC`` and ``GEMMA4_DEVICE_PLI`` are read at model construction, so compare them across
processes. Two more timing knobs:

- ``E2B_BASELINE_JSON``: a previous ``decode_timing_results.json``; every repeat's tokens are also
  compared against its first completed repeat (``first_divergence_vs_baseline``).
- ``E2B_STRICT=1``: fail the test (non-zero pytest exit) on a load/setup failure, any recorded error or a
  reference mismatch.
- ``E2B_FEEDBACK_POISON=1``: when decode token feedback is on (``GEMMA4_DEVICE_PLI=1``), pass token 0
  and position 0 from the host on every step that does not reload inputs. Output must not change, which
  shows the step really runs from the device-resident token and positions.
- ``E2B_BASELINE_MIN_MATCH=N`` (with ``E2B_BASELINE_JSON`` and ``E2B_STRICT=1``): also fail when a repeat
  diverges from the baseline stream before token N.

``test_e2b_device_pli_pcc`` (same file) checks device PLI (``GEMMA4_DEVICE_PLI=1``) against the host
reference, eagerly and under a trace, and writes ``device_pli_results.json``.
"""

import json
import math
import os
import statistics
import time
import traceback

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.models.ltx.ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL

# Outputs (videos, logs, JSON) stay out of the repo tree; LTX_ENHANCER_EXP_DIR relocates them.
OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "e2b_on_dit_handle"
)
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUT_DIR, "bringup_results.json")
TIMING_RESULTS_PATH = os.path.join(OUT_DIR, "decode_timing_results.json")
DEVICE_PLI_RESULTS_PATH = os.path.join(OUT_DIR, "device_pli_results.json")
GALAXY_RING = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]

SNAPSHOT = (
    "/mnt/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
REPO_ID = "google/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 64
PAGE_BLOCK_SIZE = 64


def _dump(results, path=RESULTS_PATH):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"results -> {path}")


def _shard_argmax(dev_tensor, vocab_size):
    """Argmax of the last vocab row of one device shard (host read of that shard only)."""
    t = ttnn.to_torch(dev_tensor).float()
    t = t.reshape(-1, t.shape[-1])[-1, :vocab_size]
    return int(t.argmax().item())


def _load_generator(full, results):
    """Gemma4Generator on ``full`` (snapshot, else repo id) with the batch-1 identity page table and the
    generation_config stop ids. Records model path, layout and model args into ``results``.
    Returns (generator, kv_cache, tokenizer, page_table), generator None on load failure."""
    from models.demos.gemma4.demo.text_demo import (
        _create_tt_page_table,
        _install_hybrid_page_tables,
        _resolve_bounded_sliding,
    )
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    page_max_num_blocks = math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)  # batch=1 right-sizing
    paged_attention_config = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=page_max_num_blocks)
    page_table = _create_tt_page_table(1, paged_attention_config)
    results["page_table_shape"] = list(page_table.shape)

    generator = tt_kv_cache = tokenizer = None
    t0 = time.time()
    for candidate in (SNAPSHOT, REPO_ID):
        try:
            bounded_sliding = _resolve_bounded_sliding(MAX_SEQ_LEN, full, candidate)
            results["bounded_sliding"] = bounded_sliding
            generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
                mesh_device=full,
                model_path=candidate,
                max_batch_size=1,
                max_seq_len=MAX_SEQ_LEN,
                paged_attention_config=paged_attention_config,
                bounded_sliding_kv_cache=bounded_sliding,
            )
            results["model_path"] = candidate
            break
        except Exception as e:  # noqa: BLE001 — try the repo id next
            results["errors"].append({"step": "2_load", "model_path": candidate, "traceback": traceback.format_exc()})
            logger.warning(f"load from {candidate} failed: {type(e).__name__}: {e}")
            if generator is not None:
                break
    results["timings_s"]["load_s"] = round(time.time() - t0, 3)
    if generator is None:
        return None, None, None, page_table

    model = generator.model[0]
    model_args = generator.model_args[0]
    mc = getattr(model, "mesh_config", None)
    results["layout"] = {
        "tp": getattr(mc, "tp", None),
        "dp": getattr(mc, "dp", None),
        "mesh_shape": list(getattr(mc, "mesh_shape", full.shape)),
        "tp_axis": getattr(mc, "tp_axis", None),
        "prefill_tp": getattr(getattr(mc, "prefill", None), "tp", None),
        "prefill_sp": getattr(getattr(mc, "prefill", None), "sp", None),
        "generator_data_parallel": generator.data_parallel,
        "num_models": len(generator.model),
    }
    results["model_args"] = {
        "num_hidden_layers": getattr(model_args, "num_hidden_layers", None),
        "vocab_size": getattr(model_args, "vocab_size", None),
        "max_prefill_chunk_size": getattr(model_args, "max_prefill_chunk_size", None),
        "device_name": getattr(model_args, "device_name", None),
        "weight_cache_path": str(getattr(model_args, "model_cache_path", None)),
        "has_per_layer_inputs": bool(getattr(model, "hidden_size_per_layer_input", 0)),
    }
    logger.info(f"[bringup] layout {json.dumps(results['layout'])}")
    if not hasattr(tokenizer, "stop_tokens"):
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
    # The demo's stop set is [eos]; Gemma4's turn closer <turn|> (106) is only in generation_config,
    # so add those ids or an instruct reply never stops within the budget.
    gen_cfg = os.path.join(results["model_path"], "generation_config.json")
    if os.path.isfile(gen_cfg):
        with open(gen_cfg) as f:
            eos_ids = json.load(f).get("eos_token_id", [])
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        tokenizer.stop_tokens = sorted(set(list(tokenizer.stop_tokens) + [int(i) for i in eos_ids]))
    results["stop_tokens"] = list(tokenizer.stop_tokens)
    if bounded_sliding:
        _install_hybrid_page_tables(
            model, model_args, batch_size=1, block_size=PAGE_BLOCK_SIZE, max_seq_len=MAX_SEQ_LEN
        )
    return generator, tt_kv_cache, tokenizer, page_table


def _encode_prompt(prompt, tokenizer, generator, results, max_new_tokens):
    """T2V-templated ``prompt`` -> ([1, L] int32 prefill tokens, prompt ids, decoding_pos, prefill_lens),
    with the template's <bos> kept single. Records the prompt diagnostics into ``results``."""
    from models.tt_dit.pipelines.ltx.prompt_enhancer import build_messages
    from models.tt_transformers.tt.common import preprocess_inputs_prefill

    messages = build_messages(prompt, "t2v")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    results["prompt_chars"] = len(text)
    results["prompt_head"] = text[:200]
    results["prompt_tail"] = text[-200:]
    # The template already emits <bos>. encode_prompt(instruct=False) calls tokenizer.encode with
    # add_special_tokens=True; this tokenizer has add_bos_token=False so that is a no-op, but probe
    # for a double BOS anyway and strip the template's copy if one shows up.
    bos = getattr(tokenizer, "bos_token", None)
    bos_id = tokenizer.bos_token_id
    probe = tokenizer.encode(text, add_special_tokens=True)
    text_for_encode = text
    if bos and text.startswith(bos) and len(probe) > 1 and probe[0] == probe[1] == bos_id:
        text_for_encode = text[len(bos) :]
    results["bos_stripped_for_encode"] = text_for_encode is not text
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [text_for_encode], tokenizer, generator.model_args, False, max_new_tokens, max_prefill_len=MAX_SEQ_LEN
    )
    prompt_ids = list(encoded_prompts[0])
    results["prompt_tokens"] = len(prompt_ids)
    results["prompt_first_ids"] = prompt_ids[:4]
    results["prompt_double_bos"] = len(prompt_ids) > 1 and prompt_ids[0] == prompt_ids[1] == tokenizer.bos_token_id
    results["prefill_lens"] = list(prefill_lens)
    results["decoding_pos"] = [int(p) for p in decoding_pos]
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
    results["prefill_input_shape"] = list(input_tokens_prefill_pt.shape)
    return input_tokens_prefill_pt, prompt_ids, decoding_pos, prefill_lens


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_e2b_bringup(mesh_device, device_params, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load):
    # Host greedy only; no device sampling, no traces (LTX_TRACED=0 is set by the launcher).
    os.environ["GEMMA4_HOST_SAMPLE"] = "1"
    # Writable weight cache away from the shared HF snapshot (model_config.resolve_model_cache_path reads it).
    os.environ.setdefault("TT_CACHE_PATH", os.path.join(os.path.expanduser("~"), ".cache", "tt-gemma4-e2b"))

    parent = mesh_device
    results = {
        "parent_shape": list(parent.shape),
        "device_params": {k: str(v) for k, v in device_params.items()},
        "tt_cache_path": os.environ["TT_CACHE_PATH"],
        "model_path": None,
        "layout": None,
        "timings_s": {},
        "steps_completed": [],
        "errors": [],
        "status": "incomplete",
    }

    def mark(step):
        results["steps_completed"].append(step)
        logger.info(f"[bringup] step done: {step}")

    full = None
    generator = None
    try:
        t0 = time.time()
        full = parent.create_submesh(ttnn.MeshShape(*parent.shape))
        results["full_shape"] = list(full.shape)
        results["full_num_devices"] = full.get_num_devices()
        results["timings_s"]["create_submesh_s"] = round(time.time() - t0, 3)
        mark("1_open_mesh")

        # ---- 2. load ------------------------------------------------------------------------
        from models.demos.gemma4.demo.text_demo import _host_sample_greedy

        generator, tt_kv_cache, tokenizer, page_table = _load_generator(full, results)
        if generator is None:
            results["status"] = "load_failure"
            return
        mark("2_load")
        model_args = generator.model_args[0]

        # ---- 3a. warmup ---------------------------------------------------------------------
        t0 = time.time()
        generator.warmup_model_prefill(
            kv_cache=tt_kv_cache, enable_trace=False, can_sample_on_device=False, greedy_only=True
        )
        results["timings_s"]["warmup_s"] = round(time.time() - t0, 3)
        mark("3a_warmup")

        # ---- 4. prompt ----------------------------------------------------------------------
        input_tokens_prefill_pt, prompt_ids, decoding_pos, prefill_lens = _encode_prompt(
            "beekeeper", tokenizer, generator, results, MAX_NEW_TOKENS
        )
        mark("4_prompt")

        # ---- 3b. prefill --------------------------------------------------------------------
        t0 = time.time()
        prefill_out = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
            enable_trace=False,
            sampling_params=None,
        )
        prefilled_token = _host_sample_greedy(prefill_out)
        results["timings_s"]["prefill_s"] = round(time.time() - t0, 3)
        results["prefill_logits_shape"] = list(prefill_out.shape)
        first_tok = int(prefilled_token.view(-1)[0].item())
        results["prefill_token"] = first_tok
        mark("3b_prefill")

        # ---- 3c. decode ---------------------------------------------------------------------
        all_outputs = prompt_ids[: prefill_lens[0]] + [first_tok]
        new_ids = [first_tok]
        current_pos = torch.tensor([decoding_pos[0]])
        out_tok = prefilled_token.view(1, 1)
        stopped = first_tok in tokenizer.stop_tokens
        results["new_token_ids"] = new_ids
        decode_kwargs = dict(
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        step_times = []
        vocab = model_args.vocab_size
        t_dec = time.time()
        iteration = 0
        while iteration < MAX_NEW_TOKENS and not stopped:
            ts = time.time()
            if iteration == 0:
                # Keep the raw device output on step 0 to answer the DP question: how many shards
                # come back and whether row 0's argmax matches the other rows / the host path.
                raw = generator.decode_forward(out_tok, current_pos, read_from_device=False, **decode_kwargs)
                dev = raw[0][0] if isinstance(raw[0], tuple) else raw[0]
                shards = ttnn.get_device_tensors(dev)
                cols = full.shape[1]
                row0 = _shard_argmax(shards[0], vocab)
                per_row = [_shard_argmax(shards[r * cols], vocab) for r in range(full.shape[0])]
                results["decode_step0"] = {
                    "num_models_returned": len(raw),
                    "device_shards": len(shards),
                    "shard_shape": list(dev.shape),
                    "row0_argmax": row0,
                    "per_row_col0_argmax": per_row,
                    "rows_agree": len(set(per_row)) == 1,
                }
                to_host = generator.read_decode_output(raw)
                decode_out, _ = generator.process_decode_output_host(to_host, is_tokens=False)
                results["decode_step0"]["host_logits_shape"] = list(decode_out.shape)
            else:
                decode_out, _ = generator.decode_forward(out_tok, current_pos, **decode_kwargs)
            out_tok = _host_sample_greedy(decode_out)
            step_times.append(round(time.time() - ts, 4))
            current_pos += 1
            tok = int(out_tok[0, 0].item())
            if iteration == 0:
                results["decode_step0"]["host_argmax"] = tok
                results["decode_step0"]["row0_matches_host"] = tok == row0
            if tok in tokenizer.stop_tokens:
                stopped = True
            else:
                all_outputs.append(tok)
                new_ids.append(tok)
            iteration += 1
            if iteration % 8 == 0:
                logger.info(f"[bringup] decode {iteration}: {tokenizer.decode(new_ids)[-80:]!r}")
        decode_s = time.time() - t_dec
        results["timings_s"]["decode_s"] = round(decode_s, 3)
        results["decode_steps"] = iteration
        results["decode_step_times_s"] = step_times
        results["stopped_on_stop_token"] = stopped
        results["new_tokens"] = len(new_ids)
        results["decode_tok_s"] = round(iteration / decode_s, 3) if decode_s > 0 and iteration else None
        results["output_text"] = tokenizer.decode(new_ids)
        logger.info(f"[bringup] output: {results['output_text']!r}")
        mark("3c_decode")
        results["status"] = "success"
    except Exception:  # noqa: BLE001 — the outcome is the data
        results["errors"].append(
            {"step": f"after:{results['steps_completed'][-1:]}", "traceback": traceback.format_exc()}
        )
        results["status"] = "exception"
        logger.error(results["errors"][-1]["traceback"])
    finally:
        _dump(results)


# Decode timing arms: name -> (decode trace, device sampling). Eager arms first so no trace is live yet.
TIMING_ARMS = {
    "eager-host": (False, False),
    "eager-device": (False, True),
    "trace-host": (True, False),
    "trace-device": (True, True),
}


def _host_sample(logits, temperature, top_k, top_p):
    """Last-position host sampling, same rule as the enhancer: argmax at T<=0, else top-k then top-p."""
    from models.tt_transformers.tt.common import sample_host

    last = logits.reshape(-1, logits.shape[-1])[-1:].float()
    if temperature <= 0:
        return int(last.argmax().item())
    if 0 < top_k < last.shape[-1]:
        kth = torch.topk(last, top_k, dim=-1).values[..., -1:]
        last = last.masked_fill(last < kth, float("-inf"))
    _, tok = sample_host(last, temperature=temperature, top_p=top_p)
    return int(tok.reshape(-1)[0].item())


def _step_stats(step_times):
    """Step 0 (compile / trace capture on a cold arm) apart from the steady-state steps."""
    stats = {"steps": len(step_times), "step0_s": step_times[0] if step_times else None}
    steady = step_times[1:]
    if steady:
        stats.update(
            steady_median_s=round(statistics.median(steady), 5),
            steady_mean_s=round(statistics.fmean(steady), 5),
            steady_p90_s=round(sorted(steady)[int(0.9 * (len(steady) - 1))], 5),
            steady_min_s=round(min(steady), 5),
            steady_max_s=round(max(steady), 5),
            steady_tok_s=round(len(steady) / sum(steady), 3),
        )
    return stats


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_e2b_decode_timing(mesh_device, device_params, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load):
    from models.demos.gemma4.demo.sampling_utils import model_can_sample_on_device
    from models.tt_dit.pipelines.ltx.prompt_enhancer import ENHANCER_TOP_K, ENHANCER_TOP_P, resolve_enhancer_cache_dir

    arms = [a.strip() for a in os.environ.get("E2B_ARMS", ",".join(TIMING_ARMS)).split(",") if a.strip()]
    unknown = [a for a in arms if a not in TIMING_ARMS]
    assert not unknown, f"unknown E2B_ARMS {unknown}; choose from {list(TIMING_ARMS)}"
    repeats = int(os.environ.get("E2B_REPEATS", "3"))
    new_tokens = int(os.environ.get("E2B_NEW_TOKENS", "128"))
    ignore_stop = os.environ.get("E2B_IGNORE_STOP", "1") == "1"
    temperature = float(os.environ.get("E2B_TEMPERATURE", "0"))
    seed = int(os.environ.get("E2B_SEED", "10"))
    prompt = os.environ.get("E2B_PROMPT", "beekeeper")
    profile = os.environ.get("E2B_PROFILE", "0") == "1"
    profile_decode_steps = int(os.environ.get("E2B_PROFILE_DECODE_STEPS", "3"))
    strict = os.environ.get("E2B_STRICT", "0") == "1"
    feedback_poison = os.environ.get("E2B_FEEDBACK_POISON", "0") == "1"
    baseline_min_match = int(os.environ.get("E2B_BASELINE_MIN_MATCH", "0"))
    if profile:
        new_tokens = 1 + profile_decode_steps  # step 0 apart, then the signposted window
    # The warm converted cache the enhancer itself uses (shared tt_cache dir when writable).
    os.environ["TT_CACHE_PATH"] = resolve_enhancer_cache_dir()

    results = {
        "parent_shape": list(mesh_device.shape),
        "device_params": {k: str(v) for k, v in device_params.items()},
        "tt_cache_path": os.environ["TT_CACHE_PATH"],
        "config": {
            "arms": arms,
            "repeats": repeats,
            "new_tokens": new_tokens,
            "ignore_stop": ignore_stop,
            "temperature": temperature,
            "top_k": ENHANCER_TOP_K if temperature > 0 else None,
            "top_p": ENHANCER_TOP_P if temperature > 0 else None,
            "seed": seed if temperature > 0 else None,
            "prompt": prompt,
            "profile": profile,
            "profile_decode_steps": profile_decode_steps if profile else None,
            "profiler_program_support_count": os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"),
            "max_seq_len": MAX_SEQ_LEN,
            "env": {
                k: os.environ.get(k)
                for k in (
                    "GEMMA4_CCL_ASYNC",
                    "GEMMA4_CCL_TOPOLOGY",
                    "GEMMA4_HOST_SAMPLE",
                    "GEMMA4_DEVICE_PLI",
                    "GEMMA4_ALWAYS_REFRESH_DECODE",
                    "LTX_TRACED",
                )
            },
        },
        "model_path": None,
        "layout": None,
        "timings_s": {},
        "arms": {},
        "summary": [],
        "errors": [],
        "status": "incomplete",
    }

    def dump():
        _dump(results, TIMING_RESULTS_PATH)

    try:
        full = mesh_device.create_submesh(ttnn.MeshShape(*mesh_device.shape))
        generator, tt_kv_cache, tokenizer, page_table = _load_generator(full, results)
        if generator is None:
            results["status"] = "load_failure"
            dump()
            assert not strict, f"load failed, see {TIMING_RESULTS_PATH}"
            return
        max_prompt_budget = new_tokens + 1  # prefill token + decode steps must fit max_seq_len
        input_tokens_prefill_pt, _, decoding_pos, _ = _encode_prompt(
            prompt, tokenizer, generator, results, max_prompt_budget
        )
        assert decoding_pos[0] + max_prompt_budget <= MAX_SEQ_LEN, (
            f"prompt {decoding_pos[0]} + {max_prompt_budget} tokens exceeds max_seq_len={MAX_SEQ_LEN}; "
            f"lower E2B_NEW_TOKENS"
        )
        can_sample = model_can_sample_on_device(generator.model[0])
        results["can_sample_on_device"] = can_sample
        results["device_pli"] = bool(getattr(generator.model[0], "_device_pli", False))
        feedback_on = bool(getattr(generator.model[0], "_tt_supports_decode_token_feedback", False))
        results["decode_token_feedback"] = feedback_on
        results["config"]["feedback_poison"] = feedback_poison
        # Count host input staging per repeat: with token feedback a warm traced repeat restages once (step 0).
        host_prep_calls = [0]
        model0 = generator.model[0]
        orig_prep = model0.prepare_decode_inputs_host

        def counted_prep(*args, **kwargs):
            host_prep_calls[0] += 1
            return orig_prep(*args, **kwargs)

        model0.prepare_decode_inputs_host = counted_prep
        results["sampling_dp"] = getattr(generator.model[0], "sampling_dp", None)
        stop_tokens = set(tokenizer.stop_tokens)
        dump()
    except Exception:  # noqa: BLE001 — the outcome is the data
        results["errors"].append({"step": "setup", "traceback": traceback.format_exc()})
        results["status"] = "exception"
        logger.error(results["errors"][-1]["traceback"])
        dump()
        assert not strict, f"setup failed, see {TIMING_RESULTS_PATH}"
        return

    from models.common.sampling.generator import SamplingParams

    if temperature > 0:
        device_sampling_params = SamplingParams(
            temperature=temperature, top_k=ENHANCER_TOP_K, top_p=ENHANCER_TOP_P, seed=seed
        )
    else:  # greedy: top_k=1 routes the device sampler to its force-argmax path
        device_sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    def flush_profiler():
        # Drain on-device profiler buffers to the host; only called when E2B_PROFILE=1.
        ttnn.ReadDeviceProfiler(full)
        ttnn.synchronize_device(full)

    signpost = None
    if profile:
        from tracy import signpost

        flush_profiler()  # load-time ops, so the first window starts from an empty buffer

    reference = None  # first completed repeat's tokens; every other repeat is compared to it
    baseline_ids = None
    baseline_path = os.environ.get("E2B_BASELINE_JSON")
    if baseline_path:
        with open(baseline_path) as f:
            base = json.load(f)
        baseline_ids = next(
            (r["token_ids"] for a in base["arms"].values() for r in a.get("repeats", []) if r.get("token_ids")),
            None,
        )
        results["baseline"] = {"path": baseline_path, "tokens": len(baseline_ids) if baseline_ids else None}
    for arm in arms:
        enable_trace, device_sample = TIMING_ARMS[arm]
        arm_res = results["arms"][arm] = {"enable_trace": enable_trace, "device_sampling": device_sample, "repeats": []}
        if device_sample and not can_sample:
            arm_res["skipped"] = "model has no on-device sampler"
            dump()
            continue
        sampling_params = device_sampling_params if device_sample else None
        # Token feedback: after step 0 the traced step reads the sampled token and the
        # advanced positions straight from its device inputs, no host restage.
        use_feedback = feedback_on and enable_trace and device_sample
        arm_res["token_feedback"] = use_feedback
        for rep in range(repeats):
            rep_res = {"repeat": rep}
            host_prep_calls[0] = 0
            arm_res["repeats"].append(rep_res)
            try:
                if temperature > 0:
                    torch.manual_seed(seed)
                measured = profile and rep == repeats - 1
                if measured:
                    signpost("prefill_start")
                # Prefill is the same in every arm: eager, host-sampled first token.
                t0 = time.perf_counter()
                logits = generator.prefill_forward_text(
                    input_tokens_prefill_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    warmup_prefill=False,
                    enable_trace=False,
                    sampling_params=None,
                )
                tok = _host_sample(logits, temperature, ENHANCER_TOP_K, ENHANCER_TOP_P)
                rep_res["prefill_s"] = round(time.perf_counter() - t0, 4)
                if measured:
                    signpost("prefill_stop")
                if profile:
                    flush_profiler()

                new_ids = [tok]
                stop_at = 0 if tok in stop_tokens else None
                current_pos = torch.tensor([decoding_pos[0]])
                step_times = []
                t_dec = time.perf_counter()
                for step in range(new_tokens):
                    if stop_at is not None and not ignore_stop:
                        break
                    reload = step == 0 or not use_feedback
                    host_tok, host_pos = torch.tensor([[tok]]), current_pos
                    if feedback_poison and not reload:
                        host_tok, host_pos = torch.tensor([[0]]), torch.zeros_like(current_pos)
                    ts = time.perf_counter()
                    out, _ = generator.decode_forward(
                        host_tok,
                        host_pos,
                        enable_trace=enable_trace,
                        page_table=page_table,
                        kv_cache=tt_kv_cache,
                        sampling_params=sampling_params,
                        # Host PLI needs the token on host every step; with token feedback only step 0 restages.
                        reload_inputs=reload,
                        reload_page_table=False,
                        # Prefill sampled on host, so the device sampler gets its params at decode step 0.
                        reload_sampling_params=device_sample and step == 0,
                        reset_sampling_state=device_sample and step == 0,
                    )
                    if device_sample:
                        if step == 0:
                            rep_res["device_token_output_shape"] = list(out.shape)
                            flat = out.reshape(-1)
                            rep_res["device_token_output_head"] = [int(x) for x in flat[:8].tolist()]
                        tok = int(out.reshape(-1)[0].item())
                    else:
                        tok = _host_sample(out, temperature, ENHANCER_TOP_K, ENHANCER_TOP_P)
                    step_times.append(round(time.perf_counter() - ts, 5))
                    if profile and step == 0:
                        flush_profiler()  # step 0 carries the trace capture on the cold repeat
                        if measured:
                            signpost("start")
                    current_pos += 1
                    new_ids.append(tok)
                    if stop_at is None and tok in stop_tokens:
                        stop_at = len(new_ids) - 1
                rep_res["decode_s"] = round(time.perf_counter() - t_dec, 4)
                rep_res["host_prep_calls"] = host_prep_calls[0]
                if measured and step_times:
                    signpost("stop")
                    rep_res["signposted_decode_steps"] = len(step_times) - 1
                if profile:
                    flush_profiler()
                rep_res.update(_step_stats(step_times))
                rep_res["step_times_s"] = step_times
                rep_res["stop_index"] = stop_at
                rep_res["token_ids"] = new_ids
                reply = new_ids[:stop_at] if stop_at is not None else new_ids
                rep_res["text"] = tokenizer.decode(reply, skip_special_tokens=True)
                if reference is None:
                    reference = {"arm": arm, "repeat": rep, "token_ids": new_ids}
                rep_res["first_divergence_vs_reference"] = _first_divergence(new_ids, reference["token_ids"])
                rep_res["matches_reference"] = rep_res["first_divergence_vs_reference"] is None
                if baseline_ids is not None:
                    rep_res["first_divergence_vs_baseline"] = _first_divergence(new_ids, baseline_ids)
                logger.info(
                    f"[timing] {arm} rep {rep}: prefill {rep_res['prefill_s']} s, step0 {rep_res.get('step0_s')} s, "
                    f"steady median {rep_res.get('steady_median_s')} s = {rep_res.get('steady_tok_s')} tok/s, "
                    f"matches ref {rep_res['matches_reference']}"
                )
            except Exception:  # noqa: BLE001 — record and move to the next arm
                rep_res["error"] = traceback.format_exc()
                results["errors"].append({"step": f"{arm}/rep{rep}", "traceback": rep_res["error"]})
                logger.error(rep_res["error"])
                dump()
                break
            dump()

    results["reference"] = {k: v for k, v in (reference or {}).items() if k != "token_ids"}
    for arm, arm_res in results["arms"].items():
        warm = [r for r in arm_res["repeats"][1:] if "steady_tok_s" in r]
        cold = arm_res["repeats"][0] if arm_res["repeats"] else {}
        row = {
            "arm": arm,
            "cold_step0_s": cold.get("step0_s"),
            "warm_step0_s": round(statistics.median([r["step0_s"] for r in warm]), 5) if warm else None,
            "steady_median_s": round(statistics.median([r["steady_median_s"] for r in warm]), 5) if warm else None,
            "steady_tok_s": round(statistics.median([r["steady_tok_s"] for r in warm]), 3) if warm else None,
            "prefill_s": round(statistics.median([r["prefill_s"] for r in warm]), 4) if warm else None,
            "all_match_reference": all(r.get("matches_reference") for r in arm_res["repeats"]) or None,
            "first_divergence_vs_baseline": [r.get("first_divergence_vs_baseline") for r in arm_res["repeats"]],
            "skipped_or_failed": arm_res.get("skipped") or any("error" in r for r in arm_res["repeats"]),
        }
        results["summary"].append(row)
        logger.info(f"[timing] summary {json.dumps(row)}")
    results["status"] = "success" if not results["errors"] else "partial"
    dump()
    if strict:
        assert not results["errors"], f"{len(results['errors'])} errors, see {TIMING_RESULTS_PATH}"
        mismatched = [r["arm"] for r in results["summary"] if not r["all_match_reference"]]
        assert not mismatched, f"arms not matching the reference tokens: {mismatched}"
        if baseline_ids is not None and baseline_min_match > 0:
            early = [
                (arm, r["repeat"], r["first_divergence_vs_baseline"])
                for arm, a in results["arms"].items()
                for r in a["repeats"]
                if r.get("first_divergence_vs_baseline") is not None
                and r["first_divergence_vs_baseline"] < baseline_min_match
            ]
            assert not early, f"diverged from the baseline before token {baseline_min_match}: {early}"


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float(torch.equal(a, b))


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_e2b_device_pli_pcc(mesh_device, device_params, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load):
    """Device PLI (``compute_device_pli``) vs the host reference (``compute_host_pli``).

    Eager over edge, random and real prompt token ids, then the same chain captured in a trace and
    replayed with new ids. Gates: PCC >= ``E2B_PLI_MIN_PCC`` (default 0.999) overall and per layer, all
    chips identical, traced output bit-identical to eager.
    """
    from models.tt_dit.pipelines.ltx.prompt_enhancer import resolve_enhancer_cache_dir

    os.environ["GEMMA4_DEVICE_PLI"] = "1"
    os.environ["TT_CACHE_PATH"] = resolve_enhancer_cache_dir()
    min_pcc = float(os.environ.get("E2B_PLI_MIN_PCC", "0.999"))
    results = {"tt_cache_path": os.environ["TT_CACHE_PATH"], "timings_s": {}, "errors": [], "ids": {}}

    full = mesh_device.create_submesh(ttnn.MeshShape(*mesh_device.shape))
    generator, _, tokenizer, _ = _load_generator(full, results)
    assert generator is not None, "load failed, see device_pli_results.json"
    model = generator.model[0]
    assert model._device_pli, "GEMMA4_DEVICE_PLI=1 did not enable device PLI"
    n_layers = len(model.layers)

    torch.manual_seed(0)
    edge = [0, 1, 2, 50, 106, 262143]
    rand = torch.randint(0, 262144, (24,)).tolist()
    prompt_ids = tokenizer.encode("A beekeeper in a white suit lifts a frame of honeycomb at golden hour.")
    real = (prompt_ids[:10] + prompt_ids[-10:])[:20]
    ids = edge + rand + real
    results["ids"] = {"edge": edge, "random": rand, "real": real}

    replicate = ttnn.ReplicateTensorToMesh(full)

    def stage(tok):
        return ttnn.from_torch(
            torch.tensor([[tok]], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate,
        )

    def read_all(out):
        shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        shards = [t.reshape(1, 1, -1, t.shape[-1])[:, :, :n_layers, :].float() for t in shards]
        return shards[0], all(torch.equal(shards[0], t) for t in shards[1:])

    per_id = []
    eager_out = {}
    t0 = time.time()
    for tok in ids:
        out = model.compute_device_pli(ttnn.to_device(stage(tok), full))
        dev, same = read_all(out)
        out.deallocate(True)
        ref = model.compute_host_pli(tok).float()
        layer_pcc = [_pcc(dev[:, :, i], ref[:, :, i]) for i in range(n_layers)]
        row = {
            "id": tok,
            "pcc": round(_pcc(dev, ref), 6),
            "min_layer_pcc": round(min(layer_pcc), 6),
            "worst_layer": int(torch.tensor(layer_pcc).argmin()),
            "max_abs_diff": round(float((dev - ref).abs().max()), 5),
            "ref_abs_max": round(float(ref.abs().max()), 4),
            "chips_identical": same,
        }
        per_id.append(row)
        eager_out[tok] = dev
    results["timings_s"]["eager_s"] = round(time.time() - t0, 3)
    results["eager"] = per_id

    # Trace: persistent token buffer, compile run, capture, replay with new ids.
    trace_rows = []
    tok_buf = ttnn.to_device(stage(ids[0]), full)
    model.compute_device_pli(tok_buf).deallocate(True)
    tid = ttnn.begin_trace_capture(full, cq_id=0)
    traced_out = model.compute_device_pli(tok_buf)
    ttnn.end_trace_capture(full, tid, cq_id=0)
    try:
        for tok in ids[::3]:
            ttnn.copy_host_to_device_tensor(stage(tok), tok_buf)
            ttnn.execute_trace(full, tid, cq_id=0, blocking=True)
            dev, same = read_all(traced_out)
            trace_rows.append(
                {"id": tok, "equals_eager": bool(torch.equal(dev, eager_out[tok])), "chips_identical": same}
            )
    finally:
        ttnn.release_trace(full, tid)
    results["traced"] = trace_rows

    worst = min(per_id, key=lambda r: r["min_layer_pcc"])
    results["summary"] = {
        "ids": len(ids),
        "min_pcc": min(r["pcc"] for r in per_id),
        "min_layer_pcc": worst["min_layer_pcc"],
        "worst_id": worst["id"],
        "max_abs_diff": max(r["max_abs_diff"] for r in per_id),
        "all_chips_identical": all(r["chips_identical"] for r in per_id + trace_rows),
        "traced_equals_eager": all(r["equals_eager"] for r in trace_rows),
        "min_pcc_gate": min_pcc,
    }
    logger.info(f"[device-pli] summary {json.dumps(results['summary'])}")
    _dump(results, DEVICE_PLI_RESULTS_PATH)
    s = results["summary"]
    assert s["min_layer_pcc"] >= min_pcc, f"device PLI PCC {s['min_layer_pcc']} < {min_pcc} (id {s['worst_id']})"
    assert s["all_chips_identical"], "device PLI differs across chips"
    assert s["traced_equals_eager"], "traced device PLI differs from eager"
