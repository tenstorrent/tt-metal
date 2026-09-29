# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model correctness gate for optimizer runs on Gemma 4 (the gemma4 demo code, any variant in HF_MODEL).

The Gemma 4 counterpart of models/tt_transformers/tests/test_optimizer_pcc.py, with the same checks
and the same printed lines, so an optimizer reads both gates the same way:

1. PCC of the logits at every teacher-forced position -- the prefill's last position, then
   FORCED_TOKENS decode steps -- against the Hugging Face bf16 reference. A broken model is refused
   by two floors on the per-position PCC (see BROKEN_PCC / LOW_PCC below); "PCC: x" (the worst
   position) is still printed for an optimizer to record.
2. Top-1 / top-5 agreement with the reference's argmax and the mean correlation, held RELATIVE to a
   baseline pinned from the unmodified tree (generated/optimizer_accuracy_baseline_<model dir>.json).

The model is built and run through gemma4's own Gemma4Generator -- from_pretrained, then
prefill_forward_text and decode_forward, untraced, returning logits -- with the same paged-attention
config, page table and bounded sliding-window KV cache as test_optimizer_gemma4_perf.py, so the gate
checks the decode path the perf test times. (It used to call ttnn_decode_forward with page_table=None,
a non-paged KV path the perf test never runs: a paged-SDPA edit passed it and then crashed in the
timed run, 2026-09-29.) Each decode step is fed the reference token (teacher forcing). Weights come
from the converted-weight store (models/tt_transformers/tests/optimizer_weight_cache.py) through a
fresh per-run TT_CACHE_PATH: gemma4's own warm cache reloads tensorbins straight onto the mesh, which
is the pinned-memory path that stalls on this QB2.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache

# The checkpoint, stated so an optimizer's roofline can resolve the model from the test it runs.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "google/gemma-4-26B-A4B-it"
MESH_SHAPE = (1, 4)  # QB2: four Blackhole chips in a row, TP=4

# Floors on the per-position logits PCC that refuse a BROKEN model; the pinned top-1/top-5/mean checks
# below are what judge a change. A single worst position is not a usable floor on this checkpoint: a
# few teacher-forced positions have a near-flat reference distribution, and their PCC swung 0.46-0.70
# across 40 gate runs of healthy trees (steps 31 and 33, independent of the mean scores); a worst-step
# floor of 0.50 refused three changes whose top-1/top-5/mean all beat the pin (2026-09-29). So:
#  - any position below BROKEN_PCC fails (a model that no longer correlates with the reference), and
#  - more than MAX_LOW_POSITIONS positions below LOW_PCC fails (degradation beyond the fragile ones).
# The unmodified tree's worst position is 0.633 (step 79, 2026-09-28): zero positions below LOW_PCC.
BROKEN_PCC = float(os.environ.get("PCC_GATE_BROKEN_PCC", "0.25"))
LOW_PCC = float(os.environ.get("PCC_GATE_LOW_PCC", "0.50"))
MAX_LOW_POSITIONS = int(os.environ.get("PCC_GATE_MAX_LOW_POSITIONS", "2"))
MAX_SEQ_LEN = 1024
PAGE_BLOCK_SIZE = 32  # the perf test's paged-attention block size
PROMPT_TOKENS = 128
FORCED_TOKENS = 128

# Relative floors against the pinned baseline (same as the tt-transformers gate). Over 129 positions
# one token is 0.78 points, so 1.0 allows a single flip and refuses two.
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))

GENERATED = Path(__file__).resolve().parents[4] / "generated"
CACHE_ROOT = GENERATED / "optimizer_cache"
REFERENCE_ROOT = GENERATED / "optimizer_reference"

# Public-domain English (Declaration of Independence, 1776), the text the tt-transformers gate uses.
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


def _model_path() -> str:
    return os.environ["HF_MODEL"]


def _baseline_path() -> Path:
    return GENERATED / f"optimizer_accuracy_baseline_{Path(_model_path()).name}.json"


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    """(top-1 match, reference argmax within the model's top 5) for one position."""
    ref = int(expected.reshape(-1).argmax())
    top5 = actual.reshape(-1).float().topk(5).indices.tolist()
    return top5[0] == ref, ref in top5


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


def encode_text(tokenizer) -> list[int]:
    """TEXT's tokens, starting with BOS: gemma4's tokenizer adds none, and Gemma expects it."""
    tokens = tokenizer.encode(TEXT)
    if tokenizer.bos_token_id is not None and (not tokens or tokens[0] != tokenizer.bos_token_id):
        tokens = [tokenizer.bos_token_id, *tokens]
    return tokens


def _tokens(tokenizer) -> list[int]:
    tokens = encode_text(tokenizer)
    needed = PROMPT_TOKENS + FORCED_TOKENS
    assert len(tokens) >= needed, f"TEXT tokenizes to {len(tokens)} tokens; the gate needs {needed}"
    return tokens[:needed]


def _reference_logits(model_path: str, tokens: list[int]) -> torch.Tensor:
    """HF bf16 logits for these tokens, [len(tokens), vocab], from the on-disk cache when it holds them."""
    key = hashlib.sha256((Path(model_path).name + ":" + ",".join(map(str, tokens))).encode()).hexdigest()[:16]
    path = REFERENCE_ROOT / f"{Path(model_path).name}-{key}.pt"
    if path.is_file():
        return torch.load(path)
    reference = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).eval()
    with torch.no_grad():
        logits = reference(torch.tensor([tokens], dtype=torch.long)).logits[0].float()
    reference = None
    gc.collect()
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(logits, path)
    return logits


def _load_baseline() -> dict | None:
    try:
        doc = json.loads(_baseline_path().read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("top1_pct") is not None else None


def _vocab_logits(tensor, vocab_size: int) -> torch.Tensor:
    """A mesh tensor's logits as one host row per position, [positions, vocab]: gathered when vocab-sharded."""
    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
    if shards[0].shape[-1] >= vocab_size:
        full = shards[0]
    else:
        full = torch.cat(shards, dim=-1)
    return full.reshape(-1, full.shape[-1])[:, :vocab_size]


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_optimizer_gemma4_pcc(monkeypatch):
    """Teacher-forced prefill + FORCED_TOKENS decode steps against Hugging Face, every layer."""
    import math

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = _model_path()
    mesh_device = generator = tt_kv_cache = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = RunCache(CACHE_ROOT, "gemma4-pcc-")
    monkeypatch.setenv("TT_CACHE_PATH", cache.path)
    # Under `pytest -s` the first print lands on the node id's line, which contains "pcc"; the
    # optimizer's parser reads any "pcc ... <float>" on a line as a measurement. End that line first.
    print("", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokens = _tokens(tokenizer)
        prompt = torch.tensor([tokens[:PROMPT_TOKENS]], dtype=torch.long)
        forced = tokens[PROMPT_TOKENS:]
        hf_logits = _reference_logits(model_path, tokens)

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*MESH_SHAPE), l1_small_size=24576, num_command_queues=1
        )
        # The perf test's paged-attention setup, so both tests run one decode path.
        paged_attention_config = PagedAttentionConfig(
            block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
        )
        lc = resolve_gemma4_demo_long_context(MAX_SEQ_LEN, mesh_device, model_path, paged_attention=True)
        generator, tt_kv_cache, _tokenizer = cache.build(
            lambda: Gemma4Generator.from_pretrained(
                mesh_device=mesh_device,
                model_path=model_path,
                max_batch_size=1,
                max_seq_len=MAX_SEQ_LEN,
                paged_attention_config=paged_attention_config,
                bounded_sliding_kv_cache=lc["bounded_sliding"],
            ),
            loaders=[(Gemma4ModelArgs, "load_state_dict")],
        )
        gc.collect()
        cache.loaded()
        vocab = generator.model_args[0].vocab_size
        page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            1, paged_attention_config.max_num_blocks
        )

        def _host_logits(out) -> torch.Tensor:
            """One position's logits as a flat host row, [vocab]."""
            first = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(first, torch.Tensor):
                first = _vocab_logits(first, vocab)
            return first.float().reshape(-1, first.shape[-1])[0, :vocab]

        t0 = time.perf_counter()
        last = _host_logits(
            generator.prefill_forward_text(
                prompt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=[PROMPT_TOKENS],
                warmup_prefill=False,
                enable_trace=False,
                sampling_params=None,
            )
        )
        prefill_pcc = _pcc(last, hf_logits[PROMPT_TOKENS - 1])
        top1_hits, top5_hits = [], []
        hit1, hit5 = _agreement(last, hf_logits[PROMPT_TOKENS - 1])
        top1_hits.append(hit1)
        top5_hits.append(hit5)

        decode_pccs = []
        for step, token in enumerate(forced):
            position = PROMPT_TOKENS + step
            step_logits = _host_logits(
                generator.decode_forward(
                    torch.tensor([[token]], dtype=torch.long),
                    torch.tensor([position], dtype=torch.int64),
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    enable_trace=False,
                    sampling_params=None,
                )
            )
            decode_pccs.append(_pcc(step_logits, hf_logits[position]))
            hit1, hit5 = _agreement(step_logits, hf_logits[position])
            top1_hits.append(hit1)
            top5_hits.append(hit5)
        elapsed = time.perf_counter() - t0

        positions = len(top1_hits)
        top1_pct = 100.0 * sum(top1_hits) / positions
        top5_pct = 100.0 * sum(top5_hits) / positions
        all_pccs = [prefill_pcc, *decode_pccs]
        mean_pcc = sum(all_pccs) / positions
        worst_pcc = min(all_pccs)
        worst_step = min(range(len(decode_pccs)), key=decode_pccs.__getitem__)
        low_positions = sum(1 for value in all_pccs if value < LOW_PCC)
        print(
            f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} "
            f"mean_corr={mean_pcc:.6f} worst_corr={worst_pcc:.6f} worst_step={worst_step} "
            f"low_positions={low_positions} prefill_corr={prefill_pcc:.6f} device_seconds={elapsed:.1f} "
            f"tree={_git_head()}",
            flush=True,
        )

        baseline = _load_baseline()
        pin = os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None
        failures = []
        if worst_pcc < BROKEN_PCC:
            failures.append(f"worst position PCC {worst_pcc:.6f} < {BROKEN_PCC} (step {worst_step})")
        if low_positions > MAX_LOW_POSITIONS:
            # Worded without "PCC" before the threshold: an optimizer reads "pcc ... <number>" in the
            # failure as the failing correlation, and 0.50 would read as a model that no longer
            # correlates at all (a broken build) instead of an accuracy loss.
            failures.append(
                f"{low_positions} positions correlate with the reference below {LOW_PCC}, "
                f"more than the {MAX_LOW_POSITIONS} allowed"
            )
        if pin and failures:
            # Never pin a broken model as the baseline every later change is held to.
            pin = False
        if pin:
            path = _baseline_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
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
                f"ACCURACY_BASELINE pinned to {path} from tree {_git_head() or '?'}: "
                f"top1={top1_pct:.2f} top5={top5_pct:.2f} mean_corr={mean_pcc:.6f}. Later runs are held "
                f"to top1 >= base-{TOP1_DROP_PTS}, top5 >= base-{TOP5_DROP_PTS}, mean_corr >= base-{MEAN_PCC_DROP}.",
                flush=True,
            )
        elif baseline is not None:
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
            # An optimizer reads the worst "PCC: x" in this output as the verdict; a change that loses
            # tokens against the pinned baseline is a failed verdict, whatever its correlation.
            print("PCC: 0.000000 (accuracy gate failed, see ACCURACY GATE FAILED above)", flush=True)
            assert not failures, "; ".join(failures)

        print(
            f"Prefill PCC: {prefill_pcc:.6f} | Decode PCC (worst of {len(decode_pccs)} steps): "
            f"{min(decode_pccs):.6f} | PCC: {worst_pcc:.6f}",
            flush=True,
        )
    finally:
        generator = None
        tt_kv_cache = None
        gc.collect()
        if mesh_device is not None:
            for submesh in list(mesh_device.get_submeshes()):
                if submesh is not mesh_device:
                    ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        cache.cleanup()
