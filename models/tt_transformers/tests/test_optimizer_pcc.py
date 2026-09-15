# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model correctness gate for production Llama optimizer runs.

Two checks, both against the Hugging Face bf16 reference on the same fixed English text:

1. PCC of the logits at every teacher-forced position (one prefill + FORCED_TOKENS decode steps),
   held to the absolute floor PCC_THRESHOLD. This is the number the optimizer parses ("PCC: x").
2. Top-1 / top-5 agreement with the reference's argmax over the same positions, held RELATIVE to a
   baseline pinned from the unmodified tree (generated/optimizer_accuracy_baseline.json). A single
   last-position PCC did not catch a 1.6-point top-1 loss from a 4-bit down-projection; token
   agreement over 129 positions does.

The optimizer keeps or reverts a change on the parsed PCC alone (perf_automation/agent/pcc_runner.py
takes the WORST "pcc ... <float>" in the output and compares it with the threshold lifted from this
file), not on the pytest exit code. So when the agreement gate fails this test also reports
"PCC: 0.000000" for that run: the score is the gate's verdict, and a change that loses tokens is
scored as failed, not as 0.99-and-passing.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


# The checkpoint this gate runs. Weights come from HF_MODEL (a local directory); the id is stated so
# the optimizer's roofline can resolve the model from the test it executes.
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Absolute floor for the worst logits PCC over every teacher-forced position. The optimizer lifts
# this constant from the file text as its pass/fail threshold (model_files._extract_pcc_threshold).
PCC_THRESHOLD = 0.90
MAX_SEQ_LEN = 2048
PROMPT_TOKENS = 128
FORCED_TOKENS = 128

# Relative floors against the pinned baseline, in percentage points of agreement and in PCC. Over
# 129 positions one token is 0.78 points, so 1.0 allows a single flip and refuses two.
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))

# See test_optimizer_perf.CACHE_ROOT: a fresh weight cache forces the HF conversion path and avoids
# the warm-cache load that has hung on the embedding file. Gitignored, inside the checkout.
GENERATED = Path(__file__).resolve().parents[3] / "generated"
CACHE_ROOT = GENERATED / "optimizer_cache"
# Pinned on the first run (or when PCC_GATE_PIN_BASELINE=1) and compared against afterwards. Lives
# outside the model directory so the optimizer's reverts never touch it.
BASELINE_PATH = GENERATED / "optimizer_accuracy_baseline.json"

# Public-domain English (Declaration of Independence, 1776). Real text, so the reference's next-token
# distribution is peaked and a flipped argmax means the model changed, not that the prompt was noise.
TEXT = (
    "When in the Course of human events, it becomes necessary for one people to dissolve the "
    "political bands which have connected them with another, and to assume among the powers of the "
    "earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle "
    "them, a decent respect to the opinions of mankind requires that they should declare the causes "
    "which impel them to the separation. We hold these truths to be self-evident, that all men are "
    "created equal, that they are endowed by their Creator with certain unalienable Rights, that "
    "among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, "
    "Governments are instituted among Men, deriving their just powers from the consent of the "
    "governed, That whenever any Form of Government becomes destructive of these ends, it is the "
    "Right of the People to alter or to abolish it, and to institute new Government, laying its "
    "foundation on such principles and organizing its powers in such form, as to them shall seem "
    "most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that "
    "Governments long established should not be changed for light and transient causes; and "
    "accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils "
    "are sufferable, than to right themselves by abolishing the forms to which they are accustomed. "
    "But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a "
    "design to reduce them under absolute Despotism, it is their right, it is their duty, to throw "
    "off such Government, and to provide new Guards for their future security. Such has been the "
    "patient sufferance of these Colonies; and such is now the necessity which constrains them to "
    "alter their former Systems of Government. The history of the present King of Great Britain is "
    "a history of repeated injuries and usurpations, all having in direct object the establishment "
    "of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid "
    "world."
)


def _logits(result) -> torch.Tensor:
    """decode_forward returns (logits, log_probs) when sampling is done on host; keep the logits."""
    return result[0] if isinstance(result, tuple) else result


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    """(top-1 match, reference argmax within the model's top 5) for one position."""
    ref = int(expected.reshape(-1).argmax())
    top5 = actual.reshape(-1).float().topk(5).indices.tolist()
    return top5[0] == ref, ref in top5


def _close(mesh_device):
    if mesh_device is not None:
        for submesh in list(mesh_device.get_submeshes()):
            if submesh is not mesh_device:
                ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _load_baseline() -> dict | None:
    try:
        doc = json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("top1_pct") is not None else None


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(1800)
def test_optimizer_full_model_pcc(monkeypatch):
    """Teacher-forced prefill + FORCED_TOKENS decode steps against Hugging Face, all 32 layers."""
    model_path = os.environ["HF_MODEL"]
    mesh_device = generator = model = model_args = reference = state_dict = tt_kv_cache = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = tempfile.TemporaryDirectory(prefix="pcc-", dir=str(CACHE_ROOT))
    monkeypatch.setenv("TT_CACHE_PATH", cache.name)
    # Under `pytest -s` the first print lands on the same line as the node id, which contains "pcc";
    # the optimizer's parser reads any "pcc ... <float>" on a line as a measurement. End that line first.
    print("", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokens = tokenizer.encode(TEXT)
        needed = PROMPT_TOKENS + FORCED_TOKENS
        assert len(tokens) >= needed, f"TEXT tokenizes to {len(tokens)} tokens; the gate needs {needed}"
        tokens = tokens[:needed]
        assert tokens[0] == tokenizer.bos_token_id, "the tokenizer did not prepend BOS; the prompt must start with it"
        prompt = torch.tensor([tokens[:PROMPT_TOKENS]], dtype=torch.long)
        forced = tokens[PROMPT_TOKENS:]

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 4),
            trace_region_size=52_000_000,
            num_command_queues=1,
        )
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        optimizations = lambda args: DecodersPrecision.performance(args.n_layers, args.model_name)
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=True,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=MAX_SEQ_LEN,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            num_layers=None,
            use_prefetcher=False,
            use_hf_rope=False,
        )
        assert model_args.n_layers == 32
        assert len(model.layers) == 32
        assert mesh_device.get_num_devices() == 4
        state_dict = None
        gc.collect()

        # One reference pass over prompt + forced tokens: logits[p] predicts token p+1, so position
        # PROMPT_TOKENS-1 is what prefill must match and PROMPT_TOKENS+i is decode step i.
        reference = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).eval()
        with torch.no_grad():
            hf_logits = reference(torch.tensor([tokens], dtype=torch.long)).logits[0].float()
        reference = None
        gc.collect()

        generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
        page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            1, paged_attention_config.max_num_blocks
        )
        kv_cache = [tt_kv_cache]

        t0 = time.perf_counter()
        tt_prefill = generator.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[PROMPT_TOKENS],
            enable_trace=False,
            warmup_prefill=False,
        ).reshape(1, -1)
        prefill_pcc = _pcc(tt_prefill, hf_logits[PROMPT_TOKENS - 1])
        top1_hits, top5_hits = [], []
        hit1, hit5 = _agreement(tt_prefill, hf_logits[PROMPT_TOKENS - 1])
        top1_hits.append(hit1)
        top5_hits.append(hit5)

        decode_pccs = []
        for step, token in enumerate(forced):
            position = PROMPT_TOKENS + step
            tt_decode = _logits(
                generator.decode_forward(
                    torch.tensor([[token]], dtype=torch.long),
                    torch.tensor([position], dtype=torch.int64),
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=False,
                    read_from_device=True,
                    sampling_params=None,
                    reset_batch=(step == 0),
                )
            ).reshape(1, -1)
            decode_pccs.append(_pcc(tt_decode, hf_logits[position]))
            hit1, hit5 = _agreement(tt_decode, hf_logits[position])
            top1_hits.append(hit1)
            top5_hits.append(hit5)
        elapsed = time.perf_counter() - t0

        positions = len(top1_hits)
        top1_pct = 100.0 * sum(top1_hits) / positions
        top5_pct = 100.0 * sum(top5_hits) / positions
        mean_pcc = (prefill_pcc + sum(decode_pccs)) / positions
        worst_pcc = min(prefill_pcc, min(decode_pccs))
        worst_step = min(range(len(decode_pccs)), key=decode_pccs.__getitem__)
        print(
            f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} "
            f"mean_corr={mean_pcc:.6f} worst_corr={worst_pcc:.6f} worst_step={worst_step} "
            f"prefill_corr={prefill_pcc:.6f} device_seconds={elapsed:.1f} tree={_git_head()}",
            flush=True,
        )

        baseline = _load_baseline()
        pin = os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None
        if pin:
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_PATH.write_text(
                json.dumps(
                    {
                        "top1_pct": top1_pct,
                        "top5_pct": top5_pct,
                        "mean_corr": mean_pcc,
                        "worst_corr": worst_pcc,
                        "positions": positions,
                        "tree": _git_head(),
                        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=1,
                )
            )
            print(
                f"ACCURACY_BASELINE pinned to {BASELINE_PATH} from tree {_git_head() or '?'}: "
                f"top1={top1_pct:.2f} top5={top5_pct:.2f} mean_corr={mean_pcc:.6f}. Later runs are held "
                f"to top1 >= base-{TOP1_DROP_PTS}, top5 >= base-{TOP5_DROP_PTS}, mean_corr >= base-{MEAN_PCC_DROP}.",
                flush=True,
            )
            failures = []
        else:
            failures = []
            if top1_pct < baseline["top1_pct"] - TOP1_DROP_PTS:
                failures.append(f"top-1 {top1_pct:.2f}% < baseline {baseline['top1_pct']:.2f}% - {TOP1_DROP_PTS}")
            if top5_pct < baseline["top5_pct"] - TOP5_DROP_PTS:
                failures.append(f"top-5 {top5_pct:.2f}% < baseline {baseline['top5_pct']:.2f}% - {TOP5_DROP_PTS}")
            if mean_pcc < baseline["mean_corr"] - MEAN_PCC_DROP:
                failures.append(
                    f"mean logits correlation {mean_pcc:.6f} < baseline {baseline['mean_corr']:.6f} - {MEAN_PCC_DROP}"
                )
            print(
                f"ACCURACY_VS_BASELINE tree={baseline.get('tree') or '?'} "
                f"top1_delta_pts={top1_pct - baseline['top1_pct']:+.2f} "
                f"top5_delta_pts={top5_pct - baseline['top5_pct']:+.2f} "
                f"mean_corr_delta={mean_pcc - baseline['mean_corr']:+.6f}",
                flush=True,
            )

        if failures:
            print("ACCURACY GATE FAILED: " + "; ".join(failures), flush=True)
            # The optimizer reads the worst "PCC: x" in this output as the verdict; a change that
            # loses tokens against the pinned baseline is a failed verdict, whatever its correlation.
            print("PCC: 0.000000 (accuracy gate failed, see ACCURACY GATE FAILED above)", flush=True)
            assert not failures, "; ".join(failures)

        print(
            f"Prefill PCC: {prefill_pcc:.6f} | Decode PCC (worst of {len(decode_pccs)} steps): "
            f"{min(decode_pccs):.6f} | PCC: {worst_pcc:.6f}",
            flush=True,
        )
        assert worst_pcc >= PCC_THRESHOLD
    finally:
        reference = None
        generator = None
        model = None
        model_args = None
        state_dict = None
        tt_kv_cache = None
        gc.collect()
        _close(mesh_device)
        cache.cleanup()
