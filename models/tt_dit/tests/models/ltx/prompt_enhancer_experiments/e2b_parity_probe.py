# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Phase 3 parity probe: teacher-force the HF greedy sequence through Gemma-4-E2B-it on the LTX handle.

Same mesh fixture, submesh, page table and ``from_pretrained`` call as test_e2b_bringup.py. Instead of
free-running, the device prefills the templated prompt and then consumes the HF reference new tokens
one by one through ``decode_forward``; the device logits captured before each HF token is fed are the
device distribution at that new-token index given the HF prefix, which is what HF's greedy ``generate``
recorded in hf_reference_scores.json. Per position: argmax agreement, top-5 overlap, top-1/top-2
margins; at the contested index the gap between the two disputed token ids on both sides. Never
asserts on parity; writes parity_probe_results.json under OUT_DIR ($LTX_ENHANCER_EXP_DIR, default ~/ltx_enhancer_experiments) (also on failure).
"""

import json
import math
import os
import subprocess
import sys
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
HF_REF_PATH = os.path.join(OUT_DIR, "hf_reference.json")
HF_SCORES_PATH = os.path.join(OUT_DIR, "hf_reference_scores.json")
HF_SCORES_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2b_hf_reference_scores.py")
RESULTS_PATH = os.path.join(OUT_DIR, "parity_probe_results.json")
DEVICE_LOGITS_PATH = os.path.join(OUT_DIR, "parity_probe_device_logits.pt")
GALAXY_RING = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]

SNAPSHOT = (
    "/mnt/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
REPO_ID = "google/gemma-4-E2B-it"
SHARED_TT_CACHE = "/mnt/models/huggingface/tt_cache/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 64
PAGE_BLOCK_SIZE = 64
CONTESTED_INDEX = 2
CONTESTED_TOKENS = (60420, 32165)  # device ' cinematic' vs HF ' documentary' in the Phase 0 free run
TOP_K = 5


def _dump(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"results -> {RESULTS_PATH}")


def _load_hf_scores():
    """Per-position HF logits summary; regenerated on CPU through the helper when absent."""
    if not os.path.isfile(HF_SCORES_PATH):
        logger.info(f"[parity] {HF_SCORES_PATH} missing, running {HF_SCORES_HELPER} on CPU")
        subprocess.run([sys.executable, HF_SCORES_HELPER], check=True)
    with open(HF_SCORES_PATH) as f:
        return json.load(f)


def _summarize(vec, hf_pos, contested_vec=None):
    """Device-vs-HF stats for one position from the device logit vector ``vec`` ([vocab] float)."""
    top = torch.topk(vec, TOP_K)
    dev_top_ids = top.indices.tolist()
    hf_top_ids = hf_pos["top5_ids"]
    dev_arg = dev_top_ids[0]
    hf_arg = hf_pos["argmax"]
    hf_logit_dev_arg = _hf_logit(hf_pos, dev_arg)
    row = {
        "index": hf_pos["index"],
        "hf_token_fed_next": hf_pos["token_id"],
        "device_argmax": dev_arg,
        "hf_argmax": hf_arg,
        "top1_match": dev_arg == hf_arg,
        "device_top5_ids": dev_top_ids,
        "hf_top5_ids": hf_top_ids,
        "device_top5_logits": [round(v, 4) for v in top.values.tolist()],
        "hf_top5_logits": hf_pos["top5_logits"],
        "top5_overlap": len(set(dev_top_ids) & set(hf_top_ids)),
        "hf_argmax_in_device_top5": hf_arg in dev_top_ids,
        "device_top1_minus_top2": round(float(top.values[0] - top.values[1]), 4),
        "hf_top1_minus_top2": hf_pos["top1_minus_top2"],
        # Negative when the device prefers its own argmax over HF's choice; 0 when they agree.
        "device_gap_hf_argmax_minus_device_argmax": round(float(vec[hf_arg] - vec[dev_arg]), 4),
        # Mirror image on the HF side; None when the device argmax is outside the stored HF top-5.
        "hf_gap_device_argmax_minus_hf_argmax": None
        if hf_logit_dev_arg is None
        else round(hf_logit_dev_arg - float(hf_pos["top5_logits"][0]), 4),
    }
    return row


def _hf_logit(hf_pos, token_id):
    """HF logit of ``token_id`` when it is inside the stored HF top-5, else None."""
    if token_id in hf_pos["top5_ids"]:
        return float(hf_pos["top5_logits"][hf_pos["top5_ids"].index(token_id)])
    return None


def _verdict(top1_agreement, dev_gap, hf_gap):
    """near_tie_numerics / systematic_mismatch / inconclusive per the stage rule."""
    if top1_agreement is None or dev_gap is None or hf_gap is None:
        return "inconclusive"
    small_both = abs(dev_gap) < 1.0 and abs(hf_gap) < 1.0
    opposite_small = (dev_gap * hf_gap < 0) and max(abs(dev_gap), abs(hf_gap)) < 1.5
    if top1_agreement >= 0.9 and (small_both or opposite_small):
        return "near_tie_numerics"
    large_flip = (dev_gap * hf_gap < 0) and min(abs(dev_gap), abs(hf_gap)) >= 1.0
    if top1_agreement < 0.75 or large_flip:
        return "systematic_mismatch"
    return "inconclusive"


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_e2b_parity_probe(mesh_device, device_params, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load):
    os.environ["GEMMA4_HOST_SAMPLE"] = "1"
    os.environ.setdefault("TT_CACHE_PATH", SHARED_TT_CACHE)

    parent = mesh_device
    results = {
        "parent_shape": list(parent.shape),
        "device_params": {k: str(v) for k, v in device_params.items()},
        "tt_cache_path": os.environ["TT_CACHE_PATH"],
        "env": {
            k: os.environ.get(k) for k in ("LTX_TRACED", "GEMMA4_HOST_SAMPLE", "TT_CACHE_PATH", "GEMMA4_CCL_TOPOLOGY")
        },
        "model_path": None,
        "timings_s": {},
        "steps_completed": [],
        "errors": [],
        "status": "incomplete",
        "positions_compared": 0,
    }

    def mark(step):
        results["steps_completed"].append(step)
        logger.info(f"[parity] step done: {step}")

    generator = None
    device_logits = []
    try:
        with open(HF_REF_PATH) as f:
            hf_ref = json.load(f)
        hf_scores = _load_hf_scores()
        hf_positions = hf_scores["positions"]
        hf_new_ids = hf_ref["new_token_ids"][:MAX_NEW_TOKENS]
        assert (
            hf_scores["new_token_ids"][: len(hf_new_ids)] == hf_new_ids
        ), "hf_reference_scores.json disagrees with hf_reference.json"
        results["hf_reference"] = {
            "transformers_version": hf_ref["transformers_version"],
            "prompt_token_count": hf_ref["prompt_token_count"],
            "new_token_ids": hf_new_ids,
            "scores_seconds": hf_scores["seconds"],
        }
        mark("0_hf_reference")

        t0 = time.time()
        full = parent.create_submesh(ttnn.MeshShape(*parent.shape))
        results["full_shape"] = list(full.shape)
        results["timings_s"]["create_submesh_s"] = round(time.time() - t0, 3)
        mark("1_open_mesh")

        from models.demos.gemma4.demo.text_demo import (
            _create_tt_page_table,
            _install_hybrid_page_tables,
            _resolve_bounded_sliding,
        )
        from models.demos.gemma4.tt.generator import Gemma4Generator
        from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

        page_max_num_blocks = math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
        paged_attention_config = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=page_max_num_blocks)
        page_table = _create_tt_page_table(1, paged_attention_config)

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
                results["errors"].append(
                    {"step": "2_load", "model_path": candidate, "traceback": traceback.format_exc()}
                )
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
        vocab = model_args.vocab_size
        results["vocab_size"] = vocab
        mc = getattr(model, "mesh_config", None)
        results["layout"] = {
            "tp": getattr(mc, "tp", None),
            "dp": getattr(mc, "dp", None),
            "tp_axis": getattr(mc, "tp_axis", None),
            "generator_data_parallel": generator.data_parallel,
        }
        if bounded_sliding:
            _install_hybrid_page_tables(
                model, model_args, batch_size=1, block_size=PAGE_BLOCK_SIZE, max_seq_len=MAX_SEQ_LEN
            )

        t0 = time.time()
        generator.warmup_model_prefill(
            kv_cache=tt_kv_cache, enable_trace=False, can_sample_on_device=False, greedy_only=True
        )
        results["timings_s"]["warmup_s"] = round(time.time() - t0, 3)
        mark("3_warmup")

        # Same templated text HF saw; the template carries the single <bos>, so no re-wrapping.
        text = hf_ref["templated_prompt"]
        bos = getattr(tokenizer, "bos_token", None)
        bos_id = tokenizer.bos_token_id
        probe = tokenizer.encode(text, add_special_tokens=True)
        if bos and text.startswith(bos) and len(probe) > 1 and probe[0] == probe[1] == bos_id:
            text = text[len(bos) :]
        input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [text], tokenizer, generator.model_args, False, MAX_NEW_TOKENS, max_prefill_len=MAX_SEQ_LEN
        )
        prompt_ids = list(encoded_prompts[0])
        results["prompt_tokens"] = len(prompt_ids)
        results["prompt_ids_match_hf"] = prompt_ids == hf_ref["prompt_token_ids"]
        if not results["prompt_ids_match_hf"]:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(prompt_ids, hf_ref["prompt_token_ids"])) if a != b), None
            )
            results["prompt_first_diff_index"] = first_diff
            logger.warning(f"[parity] prompt ids differ from HF at {first_diff}")
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)
        prompt_len = int(decoding_pos[0])
        mark("4_prompt")

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
        results["timings_s"]["prefill_s"] = round(time.time() - t0, 3)
        device_logits.append(prefill_out.float().reshape(-1)[:vocab].clone())
        mark("5_prefill")

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
        # Teacher forcing: step i feeds HF token i at position prompt_len + i and yields the device
        # distribution for new-token index i + 1. The last HF token is not fed (nothing to compare after it).
        step_times = []
        t_dec = time.time()
        for i in range(len(hf_new_ids) - 1):
            ts = time.time()
            out_tok = torch.tensor([[hf_new_ids[i]]], dtype=torch.int32)
            current_pos = torch.tensor([prompt_len + i])
            decode_out, _ = generator.decode_forward(out_tok, current_pos, **decode_kwargs)
            device_logits.append(decode_out.float().reshape(-1)[:vocab].clone())
            step_times.append(round(time.time() - ts, 4))
            if (i + 1) % 8 == 0:
                agree = sum(int(v.argmax().item()) == p["argmax"] for v, p in zip(device_logits, hf_positions))
                logger.info(f"[parity] fed {i + 1} HF tokens, top-1 agreement so far {agree}/{len(device_logits)}")
        results["timings_s"]["decode_s"] = round(time.time() - t_dec, 3)
        results["decode_step_times_s"] = step_times
        mark("6_teacher_forced_decode")

        torch.save(torch.stack(device_logits), DEVICE_LOGITS_PATH)
        results["device_logits_path"] = DEVICE_LOGITS_PATH

        rows = [_summarize(vec, hf_positions[i]) for i, vec in enumerate(device_logits)]
        n = len(rows)
        results["positions_compared"] = n
        results["positions"] = rows
        top1 = sum(r["top1_match"] for r in rows) / n
        top5 = sum(r["hf_argmax_in_device_top5"] for r in rows) / n
        overlap = sum(r["top5_overlap"] for r in rows) / (TOP_K * n)
        results["top1_agreement"] = round(top1, 4)
        results["top5_agreement"] = round(top5, 4)
        results["mean_top5_overlap_fraction"] = round(overlap, 4)
        results["mismatch_indices"] = [r["index"] for r in rows if not r["top1_match"]]
        results["device_free_run_token_ids"] = [r["device_argmax"] for r in rows]
        results["device_argmax_text"] = tokenizer.decode(results["device_free_run_token_ids"], skip_special_tokens=True)

        a, b = CONTESTED_TOKENS
        c = device_logits[CONTESTED_INDEX]
        hf_c = hf_scores["contested"]
        dev_gap = float(c[a] - c[b])
        hf_gap = float(hf_c["gap_a_minus_b"])
        contested = {
            "index": CONTESTED_INDEX,
            "tokens": [a, b],
            "token_strs": [tokenizer.decode([a]), tokenizer.decode([b])],
            "device_logit_a": float(c[a]),
            "device_logit_b": float(c[b]),
            "device_gap_a_minus_b": round(dev_gap, 4),
            "device_rank_a": int((c > c[a]).sum().item()),
            "device_rank_b": int((c > c[b]).sum().item()),
            "hf_logit_a": hf_c["logit_a"],
            "hf_logit_b": hf_c["logit_b"],
            "hf_gap_a_minus_b": round(hf_gap, 4),
            "hf_rank_a": hf_c["rank_a"],
            "hf_rank_b": hf_c["rank_b"],
        }
        hf_full = hf_scores.get("index2_full_logits")
        if hf_full is not None and len(hf_full) == vocab:
            hf_vec = torch.tensor(hf_full, dtype=torch.float32)
            lp_dev = torch.log_softmax(c, -1)
            lp_hf = torch.log_softmax(hf_vec, -1)
            contested["kl_hf_to_device"] = round(float((lp_hf.exp() * (lp_hf - lp_dev)).sum()), 5)
            contested["max_abs_logit_diff"] = round(float((c - hf_vec).abs().max()), 4)
            contested["mean_abs_logit_diff"] = round(float((c - hf_vec).abs().mean()), 4)
            contested["logit_corr"] = round(float(torch.corrcoef(torch.stack([c, hf_vec]))[0, 1]), 6)
        results["contested"] = contested

        worst = sorted(
            (r for r in rows if not r["top1_match"]),
            key=lambda r: r["device_gap_hf_argmax_minus_device_argmax"],
        )[:8]
        results["worst_positions"] = [
            {
                k: r[k]
                for k in (
                    "index",
                    "device_argmax",
                    "hf_argmax",
                    "device_gap_hf_argmax_minus_device_argmax",
                    "hf_top1_minus_top2",
                    "device_top1_minus_top2",
                    "top5_overlap",
                )
            }
            | {"device_str": tokenizer.decode([r["device_argmax"]]), "hf_str": tokenizer.decode([r["hf_argmax"]])}
            for r in worst
        ]
        results["verdict"] = _verdict(top1, dev_gap, hf_gap)
        logger.info(
            f"[parity] top1={top1:.3f} top5={top5:.3f} overlap={overlap:.3f} "
            f"index2 gap dev={dev_gap:+.4f} hf={hf_gap:+.4f} verdict={results['verdict']}"
        )
        logger.info(f"[parity] mismatches at {results['mismatch_indices']}")
        logger.info(f"[parity] worst {json.dumps(results['worst_positions'])}")
        mark("7_compare")
        results["status"] = "success"
    except Exception:  # noqa: BLE001 — the outcome is the data
        results["errors"].append(
            {"step": f"after:{results['steps_completed'][-1:]}", "traceback": traceback.format_exc()}
        )
        results["status"] = "exception"
        logger.error(results["errors"][-1]["traceback"])
    finally:
        _dump(results)
