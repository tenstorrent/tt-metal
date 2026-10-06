# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Phase 0 bring-up: Gemma-4-E2B-it on the LTX DiT handle (full-shape submesh of the 4x8 Galaxy).

Opens the mesh with the LTX distilled ring params, carves ``full`` exactly like the pipeline, loads
Gemma4Generator on ``full`` and reproduces text_demo.py::_run_generation_via_generator with host
greedy sampling and no tracing anywhere. Never asserts on content; records every step's wall-clock
and the outcome to ``bringup_results.json`` under OUT_DIR ($LTX_ENHANCER_EXP_DIR, default ~/ltx_enhancer_experiments) (also on failure).
"""

import json
import math
import os
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
GALAXY_RING = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]

SNAPSHOT = (
    "/mnt/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
REPO_ID = "google/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 64
PAGE_BLOCK_SIZE = 64


def _dump(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"results -> {RESULTS_PATH}")


def _shard_argmax(dev_tensor, vocab_size):
    """Argmax of the last vocab row of one device shard (host read of that shard only)."""
    t = ttnn.to_torch(dev_tensor).float()
    t = t.reshape(-1, t.shape[-1])[-1, :vocab_size]
    return int(t.argmax().item())


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
        from models.demos.gemma4.demo.text_demo import (
            _create_tt_page_table,
            _host_sample_greedy,
            _install_hybrid_page_tables,
            _resolve_bounded_sliding,
        )
        from models.demos.gemma4.tt.generator import Gemma4Generator
        from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

        page_max_num_blocks = math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)  # batch=1 right-sizing
        paged_attention_config = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=page_max_num_blocks)
        page_table = _create_tt_page_table(1, paged_attention_config)
        results["page_table_shape"] = list(page_table.shape)

        load_err = None
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
                load_err = traceback.format_exc()
                results["errors"].append({"step": "2_load", "model_path": candidate, "traceback": load_err})
                logger.warning(f"load from {candidate} failed: {type(e).__name__}: {e}")
                if generator is not None:
                    break
        results["timings_s"]["load_s"] = round(time.time() - t0, 3)
        if generator is None:
            results["status"] = "load_failure"
            return
        mark("2_load")

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

        # ---- 3a. warmup ---------------------------------------------------------------------
        t0 = time.time()
        generator.warmup_model_prefill(
            kv_cache=tt_kv_cache, enable_trace=False, can_sample_on_device=False, greedy_only=True
        )
        results["timings_s"]["warmup_s"] = round(time.time() - t0, 3)
        mark("3a_warmup")

        # ---- 4. prompt ----------------------------------------------------------------------
        from models.tt_dit.pipelines.ltx.prompt_enhancer import build_messages

        messages = build_messages("beekeeper", "t2v")
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
            [text_for_encode], tokenizer, generator.model_args, False, MAX_NEW_TOKENS, max_prefill_len=MAX_SEQ_LEN
        )
        prompt_ids = list(encoded_prompts[0])
        results["prompt_tokens"] = len(prompt_ids)
        results["prompt_first_ids"] = prompt_ids[:4]
        results["prompt_double_bos"] = len(prompt_ids) > 1 and prompt_ids[0] == prompt_ids[1] == tokenizer.bos_token_id
        results["prefill_lens"] = list(prefill_lens)
        results["decoding_pos"] = [int(p) for p in decoding_pos]
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        results["prefill_input_shape"] = list(input_tokens_prefill_pt.shape)
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
