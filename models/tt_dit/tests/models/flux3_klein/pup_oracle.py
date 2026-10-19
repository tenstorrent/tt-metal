# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU bf16 oracle of the FLUX 3 Image Klein prompt upsampler (design ``docs/M9_DESIGN.md`` section 8).

The shipped example records are not a deterministic oracle of the pup: BFL's own base vs distilled records of the
same input (two vLLM greedy runs of the one model) agree exactly on 9 of 13 pairs and sit at token-F1 0.67 / 0.90 /
0.94 / 0.998 on the other four. A device caption that misses the record (exact, else F1 >= 0.9) is therefore judged
against its PROPER reference: the HF ``Qwen3_5ForConditionalGeneration`` in bf16 on the CPU, fed the SAME
``PupInputs`` (token ids and pixel values from ``upsampler.build_pup_inputs``), decoded greedily with vLLM's presence
penalty (a ``LogitsProcessor`` applying the rule of ``upsampler.presence_greedy``: HF ``generate`` has no presence
penalty of its own and its ``repetition_penalty`` is a different, multiplicative operator), stopped on the two stop
ids, at most 3584 new tokens. ``--teacher-force`` also records, per position of the record's caption, the oracle's
top-1 and the gap ``v[top1] - v[record token]`` under the presence penalty of the record prefix: the tie audit of
design section 9.3 (a position with a gap near zero is a bf16 tie, either choice is "the model"). The target is
``upsampler.audit_target_ids``: text-to-image and edit records only (a layout record's caption is
``bbox.reconcile`` output, not a sequence the model emitted, so those cases get no ``gaps``); the first and the
last two positions are boundary positions (``AUDIT_BOUNDARY``: the record is ``.strip()``ped text).

Results are cached under ``$FLUX3_KLEIN_PUP_ORACLE_CACHE`` (else ``$FLUX3_KLEIN_OUT/pup_oracle``), keyed by the pup
checkpoint blob and the inputs' sha256, as ``<key>.json`` plus a readable ``<set>-<case>.txt``. No ttnn import:
it runs on a node CPU through the device lock (minutes per caption at 40 threads) or, for ONE short text-to-image
case, on a login box with enough free RAM (the model is 18.8 GB in bf16).

    python -m models.tt_dit.tests.models.flux3_klein.pup_oracle --case distilled/man_dog [--case ...] \\
        [--threads 40] [--teacher-force] [--max-new-tokens 3584]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger

from ....pipelines.flux3_klein.upsampler import (
    FLUX3_KLEIN_PUP_MAX_NEW_TOKENS,
    FLUX3_KLEIN_PUP_STOP_IDS,
    PupInputs,
    audit_target_ids,
    build_pup_inputs,
    finish_caption,
    first_divergence,
    load_system_prompts,
    presence_greedy,
    token_f1,
)
from .common import PUP_CASES, load_record, pup_case_id, pup_oracle_cache, pup_request_of, variant_checkpoint_dir

ORACLE_VERSION = 1


def checkpoint_blob(snapshot: Path) -> str:
    """The first 16 hex digits of the ``model.safetensors`` blob name (a HuggingFace cache symlink target) or, for a
    plain directory, of the file's size + mtime."""
    path = Path(snapshot) / "model.safetensors"
    real = Path(os.path.realpath(path))
    if real.name != path.name:
        return real.name[:16]
    stat = real.stat()
    return hashlib.sha256(f"{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]


def oracle_key(snapshot: Path, inputs: PupInputs) -> str:
    return hashlib.sha256(
        f"pup-oracle-v{ORACLE_VERSION}|{checkpoint_blob(snapshot)}|{inputs.sha256()}".encode()
    ).hexdigest()


def cached_oracle(snapshot: Path, inputs: PupInputs, *, case: str = "") -> dict | None:
    """The cached oracle dict of ``inputs`` (``None`` when the cache is unset or has no entry: the caller reports
    "run the oracle", never skips)."""
    root = pup_oracle_cache()
    if root is None:
        return None
    path = root / f"{oracle_key(snapshot, inputs)}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    logger.info(f"ORACLE HIT {path.name[:16]} ({case}): {data.get('generated_tokens')} tokens")
    return data


class PresencePenaltyProcessor:
    """vLLM's presence penalty as an HF ``LogitsProcessor``: the SAME rule as ``upsampler.presence_greedy`` (subtract
    ``penalty`` from the score of every id generated so far, prompt excluded, each id once)."""

    def __init__(self, prompt_len: int, penalty: float) -> None:
        self.prompt_len = int(prompt_len)
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        if self.penalty > 0:
            for row in range(scores.shape[0]):
                generated = input_ids[row, self.prompt_len :].tolist()
                if generated:
                    scores[row, torch.tensor(sorted(set(generated)), dtype=torch.long)] -= self.penalty
        return scores


_MODELS: dict = {}


def load_oracle_model(snapshot: Path, *, threads: int = 40):
    """The HF model in bf16 on the CPU, loaded once per process."""
    from transformers import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(int(threads))
    key = str(snapshot)
    if key not in _MODELS:
        t0 = time.time()
        _MODELS[key] = Qwen3_5ForConditionalGeneration.from_pretrained(
            key, dtype=torch.bfloat16, device_map="cpu", local_files_only=True
        ).eval()
        logger.info(f"oracle model loaded in {time.time() - t0:.1f}s ({threads} threads)")
    return _MODELS[key]


def _model_inputs(inputs: PupInputs) -> dict:
    kwargs = {"input_ids": inputs.input_ids, "attention_mask": torch.ones_like(inputs.input_ids)}
    if inputs.pixel_values is not None:
        kwargs["pixel_values"] = inputs.pixel_values
        kwargs["image_grid_thw"] = inputs.image_grid_thw
    return kwargs


def teacher_forced_gaps(model, inputs: PupInputs, target_ids: Sequence[int]) -> list[tuple[int, int, int, float]]:
    """One forward over prompt + target[:-1]: per position ``(pos, top1, target, gap)`` with the presence penalty of
    the target prefix applied before the argmax (the device's ``teacher_forced`` computes the same quantity)."""
    targets = [int(t) for t in target_ids]
    ids = torch.cat([inputs.input_ids, torch.tensor([targets[:-1]], dtype=torch.int64)], dim=1)
    kwargs = _model_inputs(inputs)
    kwargs["input_ids"] = ids
    kwargs["attention_mask"] = torch.ones_like(ids)
    with torch.inference_mode():
        logits = model(**kwargs, use_cache=False).logits[0, inputs.prompt_tokens - 1 :].float()
    out = []
    for i, target in enumerate(targets):
        v = logits[i].clone()
        prefix = targets[:i]
        if inputs.presence_penalty > 0 and prefix:
            v[torch.tensor(sorted(set(prefix)), dtype=torch.long)] -= inputs.presence_penalty
        top = int(torch.argmax(v).item())
        out.append((i, top, target, float(v[top] - v[target])))
    return out


def run_oracle(
    snapshot: Path,
    inputs: PupInputs,
    *,
    max_new_tokens: int = FLUX3_KLEIN_PUP_MAX_NEW_TOKENS,
    threads: int = 40,
    teacher_force: Sequence[int] | None = None,
    tokenizer=None,
    case: str = "",
    model=None,
) -> dict:
    """Greedy bf16 decode with the presence penalty on the CPU (module docstring); returns the dict that is cached:
    ``ids`` (without the stop id), ``caption``, ``generated_tokens``, ``wall_s``, versions, and ``gaps`` for
    ``teacher_force``."""
    import transformers
    from transformers import LogitsProcessorList

    model = model if model is not None else load_oracle_model(snapshot, threads=threads)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **_model_inputs(inputs),
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            logits_processor=LogitsProcessorList(
                [PresencePenaltyProcessor(inputs.prompt_tokens, inputs.presence_penalty)]
            ),
            eos_token_id=list(FLUX3_KLEIN_PUP_STOP_IDS),
            pad_token_id=FLUX3_KLEIN_PUP_STOP_IDS[1],
            repetition_penalty=1.0,
            use_cache=True,
        )
    ids = [int(t) for t in out[0, inputs.prompt_tokens :].tolist()]
    while ids and ids[-1] in FLUX3_KLEIN_PUP_STOP_IDS:
        ids = ids[:-1]
    wall = time.time() - t0
    caption = None
    if tokenizer is not None:
        try:
            caption = finish_caption(ids, inputs.locked, tokenizer=tokenizer)
        except RuntimeError as error:
            caption = f"<{error}>"
    result = {
        "case": case,
        "key": oracle_key(snapshot, inputs),
        "checkpoint_blob": checkpoint_blob(snapshot),
        "inputs_sha256": inputs.sha256(),
        "task": inputs.task,
        "system_key": inputs.system_key,
        "prompt_tokens": inputs.prompt_tokens,
        "image_tokens": inputs.num_image_tokens,
        "presence_penalty": inputs.presence_penalty,
        "ids": ids,
        "caption": caption,
        "generated_tokens": len(ids),
        "wall_s": wall,
        "tok_s": len(ids) / wall if wall > 0 else 0.0,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "threads": int(threads),
        "max_new_tokens": int(max_new_tokens),
        "oracle_version": ORACLE_VERSION,
    }
    logger.info(f"ORACLE {case}: {len(ids)} tokens in {wall:.1f}s ({result['tok_s']:.2f} tok/s)")
    if teacher_force is not None:
        t0 = time.time()
        gaps = teacher_forced_gaps(model, inputs, teacher_force)
        agree = sum(1 for _, top, tgt, _ in gaps if top == tgt)
        result["gaps"] = gaps
        result["teacher_force_ids"] = [int(t) for t in teacher_force]
        result["teacher_agreement"] = agree / max(1, len(gaps))
        logger.info(f"ORACLE {case} teacher-forced: agreement {agree}/{len(gaps)} in {time.time() - t0:.1f}s")
    return result


def write_oracle(result: dict, *, case: str) -> Path | None:
    root = pup_oracle_cache()
    if root is None:
        logger.warning("no oracle cache directory (FLUX3_KLEIN_PUP_ORACLE_CACHE / FLUX3_KLEIN_OUT unset): not cached")
        return None
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{result['key']}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (root / f"{case.replace('/', '-')}.txt").write_text(
        f"{case}\nkey {result['key']}\n{result['generated_tokens']} tokens, {result['wall_s']:.1f}s\n\n{result['caption']}\n"
    )
    return path


def oracle_for_case(case: tuple[str, str], *, snapshot: Path, tokenizer, processor, prompts) -> tuple[dict, PupInputs]:
    """The record, its upsampler inputs and (``record``, ``inputs``) of ``(set, case)``."""
    record = load_record(*case)
    prompt, images, boxes = pup_request_of(record)
    inputs = build_pup_inputs(prompt, images, boxes, prompts=prompts, tokenizer=tokenizer, processor=processor)
    expected = PUP_CASES.get(case)
    if expected is not None and inputs.prompt_tokens != expected:
        raise RuntimeError(f"{pup_case_id(case)}: {inputs.prompt_tokens} prompt tokens, pinned {expected}")
    return record, inputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True, help="<set>/<case> of examples/, repeatable")
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=FLUX3_KLEIN_PUP_MAX_NEW_TOKENS)
    parser.add_argument(
        "--teacher-force", action="store_true", help="also record the per-position gaps of the record caption"
    )
    parser.add_argument("--force", action="store_true", help="recompute even when cached")
    args = parser.parse_args(argv)
    snapshot = variant_checkpoint_dir("pup")
    if snapshot is None:
        print("pup snapshot not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)", file=sys.stderr)
        return 2
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    prompts = load_system_prompts(snapshot)
    for spec in args.case:
        case = tuple(spec.split("/", 1))
        record, inputs = oracle_for_case(
            case, snapshot=snapshot, tokenizer=tokenizer, processor=processor, prompts=prompts
        )
        label = pup_case_id(case)
        record_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
        target_ids = None
        if args.teacher_force:
            try:
                target_ids = audit_target_ids(record["prompt"], tokenizer, locked=inputs.locked)
            except ValueError as error:
                logger.warning(f"{label}: no teacher-forced audit ({error})")
        cached = None if args.force else cached_oracle(snapshot, inputs, case=label)
        if cached is None or (target_ids is not None and "gaps" not in cached):
            result = run_oracle(
                snapshot,
                inputs,
                max_new_tokens=args.max_new_tokens,
                threads=args.threads,
                teacher_force=target_ids,
                tokenizer=tokenizer,
                case=label,
            )
            write_oracle(result, case=label)
        else:
            result = cached
        caption = result["caption"] or ""
        caption_ids = tokenizer.encode(caption, add_special_tokens=False)
        exact = caption == record["prompt"]
        f1 = token_f1(caption_ids, record_ids)
        div = first_divergence(caption_ids, record_ids)
        print(
            f"RESULT oracle[{label}] exact_rec {exact} f1_rec {f1:.4f} first_div {div} len {len(caption_ids)} "
            f"(record {len(record_ids)}) wall {result['wall_s']:.1f}s"
            + (f" teacher_agreement {result['teacher_agreement']:.4f}" if "teacher_agreement" in result else ""),
            flush=True,
        )
        print(f"\n{caption}\n", flush=True)
        # the sanity row of the sampler: the processor's rule and presence_greedy agree on the first step
        assert presence_greedy(torch.zeros(4), [], inputs.presence_penalty) == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
