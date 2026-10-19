# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein prompt upsampler ("pup") gates (design ``docs/M9_DESIGN.md`` sections 9 and 11).

Host rows (no device; the snapshot's tokenizer / processor and the BFL clone ``$FLUX3_KLEIN_REFERENCE`` where
noted, skipped without them):
  U0a ``test_build_inputs_pinned``        ``build_pup_inputs`` token ids == the reference recipe (``pup.py:99-119``)
                                           transcribed here with the vendored constants; prompt token counts pinned
                                           (logs/m9_designer_host_facts.log:10-17), grids and image tokens pinned,
                                           the template tail, the presence penalty per task; the worst case (10
                                           references + a 2000-char instruction) = 3422 tokens (3703 with one
                                           locked box); the refusals.
  U0b ``test_sizing_bit_exact``           vendored ``reference_size`` / ``policy_reference`` vs the clone's on
                                           synthetic sizes, an RGBA-over-white input, the 200:1 aspect error.
  U0c ``test_presence_greedy``            ``presence_greedy`` vs a transcription of vLLM's penalty + argmax.
  U0c' ``test_token_f1_record_pairs``     ``token_f1`` / ``first_divergence`` on BFL's own record pairs reproduce
                                           logs/m9_designer_host_facts.log:56-68 (0.6716 / 0.8973 / 0.9388 / 0.9978).
  U0d ``test_vendored_pup_is_reference``  the vendored ``pup.py`` / ``bbox.py`` are the clone black-formatted;
                                           ``pup_prompts.json`` == the clone's == the snapshot's.
  U0e ``test_refusals_and_caption``       the ``ValueError`` rows (11 images, a prompt of ``max_model_len`` tokens
                                           or more), the stop id, ``RuntimeError("empty generation")``,
                                           ``is_finished`` on the records; ``new_token_budget`` (vLLM's
                                           ``min(max_tokens, max_model_len - prompt)``).
  U0f ``test_reconcile``                  ``bbox.reconcile`` on a synthetic answer that re-plans a locked box.
  U0g ``test_subprocess_staging``         ``stage_reference`` (the subprocess client's PNG hand-over) keeps
                                           ``policy_reference`` pixel-exact for an EXIF-rotated JPEG and an RGBA
                                           image; ``audit_target_ids`` / ``divergence_audit`` rows.
  U0h ``test_split_images``               ``split_images`` / ``vision_padded_rows`` on two references of
                                           different sizes (the per-image tower calls of the wrapper) and on the
                                           two equal mugprint references.
Device rows (``bh_1x1_nofabric``: one chip, the qwen36 single-device parameters, no fabric; one weight load per
row, ``QWEN36_EXACT_PROFILE=1`` set by the wrapper):
  U1 ``test_caption_t2i[case]``, U2 ``test_caption_bbox[case]``, U3 ``test_caption_i2i[case]``: the gate of design
     9.2 per recorded case: the device caption == the record (PASS), else token-F1 >= ``RECORD_F1`` = 0.9 vs the
     record (PASS; PLAN M9's number; calibration: BFL's own base vs distilled pairs of the same input sit at 0.67 /
     0.90 / 0.94 / 0.998 and 1.0 x 9), else the CPU bf16 oracle (``pup_oracle.py``) on the same token ids: no
     oracle on disk -> FAIL "run the oracle"; exact == oracle -> PASS; F1 >= 0.9 vs the oracle AND the
     teacher-forced audit of 9.3 passes (``divergence_audit``: the oracle's gaps on disk, ``TIE_GAP`` frozen, no
     device-only disagreement above it; a layout case has no audit and cannot pass this tier) -> PASS; else FAIL
     with both distances. A pass on an oracle tier is reported as a pytest warning (the warnings summary) and in
     ``record_property`` (junit), never silently.
  U4 ``test_determinism_and_stop``: two runs bit-equal, double_exposure s1 == s2, no stop id in the ids, the
     ``max_new_tokens`` cap, the empty-generation error.
  U5 ``test_teacher_forced_audit[case]``: design 9.3 on the text-to-image / edit cases (a layout record's caption
     is ``reconcile`` output, no audit target); diagnostic until ``TIE_GAP`` is frozen at the first measurement;
     needs the oracle's gaps on disk.
  U6 ``test_timings``: the warm man_dog / bandana RECORD lines into ``$FLUX3_KLEIN_OUT/pup_timings.json``.
  U7 ``test_two_sizes_i2i``: two references of DIFFERENT sizes (``examples/inputs/1.png`` 896x1120 and ``2.png``
     1024x1024 -> 640x800 and 704x704 at the 2-reference budget: 500 + 484 image tokens): the per-image vision
     path produces a caption (the tower's own multi-image concat fails on unequal token counts; M9 fixer finding
     13); the second run is bit-equal.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from ....pipelines.flux3_klein import upsampler as U
from ....pipelines.flux3_klein.upsampler import (
    AUDIT_BOUNDARY,
    FLUX3_KLEIN_PUP_MAX_NEW_TOKENS,
    FLUX3_KLEIN_PUP_STOP_IDS,
    PUP_MAX_PROMPT,
    PUP_MAX_SEQ_LEN,
    audit_target_ids,
    build_pup_inputs,
    divergence_audit,
    finish_caption,
    first_divergence,
    load_system_prompts,
    new_token_budget,
    presence_greedy,
    split_images,
    stage_reference,
    token_f1,
    vision_padded_rows,
)
from ....reference.flux3_klein import bbox
from ....reference.flux3_klein import pup as vend_pup
from . import pup_oracle
from .common import (
    PUP_BBOX_CASES,
    PUP_CASES,
    PUP_I2I_CASES,
    PUP_T2I_CASES,
    REFERENCE_CLONE_ENV,
    REFERENCE_DIR,
    load_record,
    pup_case_id,
    pup_oracle_cache,
    pup_request_of,
    renders_dir,
    require_example_input,
    variant_checkpoint_dir,
)

# The host rows need the tokenizer / processor / prompts of the snapshot, not the 18 GB weights.
PUP_HOST_REQUIRED = ("config.json", "tokenizer.json", "system_prompts.json", "preprocessor_config.json")
RECORD_F1 = 0.9
"""PLAN.md M9: token-F1 vs the record below which the oracle decides (section 9.2; calibration in the docstring)."""
ORACLE_F1 = 0.9
"""The same number against the proper reference (design 9.2 step 3)."""
TIE_GAP: float | None = None
"""Design 9.3: the largest oracle gap among the interior positions where the oracle itself disagrees with the
record (``divergence_audit['tie_scale']``), rounded up to one significant digit, frozen at the FIRST audit (log +
line to be written here); ``None`` = the audit only logs and the oracle-F1 gate tier cannot pass (not measured as
of 2026-10-19: no oracle on disk)."""
TEMPLATE_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PINNED_SYSTEM_KEY = {
    ("distilled", "man_dog"): "t2i",
    ("distilled", "balloons"): "t2i_bbox",
    ("distilled", "3_bandana"): "i2i",
    ("distilled", "cactus"): "i2i_bbox",
}
PINNED_GRID = {
    ("distilled", "3_bandana"): [[1, 64, 64]],
    ("distilled", "2_coffee"): [[1, 64, 64]],
    ("distilled", "1_figurine"): [[1, 70, 56]],
    ("distilled", "2_3_mugprint"): [[1, 44, 44], [1, 44, 44]],
    ("distilled", "cactus"): [[1, 64, 64]],
}
RECORD_PAIRS = {  # logs/m9_designer_host_facts.log:56-62: (f1 of ids, first divergence, len base, len distilled)
    "man_dog": (0.6716, 4, 270, 275),
    "1_figurine": (1.0, 254, 254, 254),
    "2_coffee": (0.9978, 102, 223, 222),
    "3_bandana": (0.8973, 49, 239, 238),
    "2_3_mugprint": (1.0, 237, 237, 237),
    "balloons": (0.9388, 141, 519, 510),
    "cactus": (1.0, 292, 292, 292),
}
ALL_CASES = PUP_T2I_CASES + PUP_BBOX_CASES + PUP_I2I_CASES
AUDIT_CASES = [c for c in ALL_CASES if c not in PUP_BBOX_CASES and c != ("distilled", "cactus")]
"""The teacher-forced audit's cases: those whose record caption is the model's own (stripped) output; balloons
(``t2i_bbox``) and cactus (``i2i_bbox``) are ``bbox.reconcile`` output (``audit_target_ids`` refuses them)."""


# --- fixtures ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pup_snapshot() -> Path:
    found = variant_checkpoint_dir("pup", PUP_HOST_REQUIRED)
    if found is None:
        pytest.skip("pup snapshot not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    return found


@pytest.fixture(scope="module")
def pup_host(pup_snapshot):
    """``(tokenizer, processor, prompts)`` of the snapshot."""
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(pup_snapshot), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(pup_snapshot), local_files_only=True)
    return tokenizer, processor, load_system_prompts(pup_snapshot)


def reference_clone() -> Path | None:
    """The BFL clone (``$FLUX3_KLEIN_REFERENCE``) when it holds ``src/flux3/pup.py``."""
    value = os.environ.get(REFERENCE_CLONE_ENV)
    if value and (Path(value) / "src" / "flux3" / "pup.py").is_file():
        return Path(value)
    return None


def import_reference_module(name: str):
    """``flux3.<name>`` from the clone's ``src`` (skips without the clone)."""
    clone = reference_clone()
    if clone is None:
        pytest.skip(f"{REFERENCE_CLONE_ENV} unset or without src/flux3/pup.py")
    src = str(clone / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(f"flux3.{name}")


def record_inputs(case, pup_host):
    tokenizer, processor, prompts = pup_host
    record = load_record(*case)
    prompt, images, boxes = pup_request_of(record)
    return record, build_pup_inputs(prompt, images, boxes, prompts=prompts, tokenizer=tokenizer, processor=processor)


# --- U0: host ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=[pup_case_id(c) for c in ALL_CASES])
def test_build_inputs_pinned(case, pup_host) -> None:
    """U0a: ``build_pup_inputs`` == the reference recipe transcribed here (``pup.py:99-119`` with the vendored
    constants, the same tokenizer / processor); pinned counts, grids, keys, penalties and the template tail."""
    tokenizer, processor, prompts = pup_host
    record = load_record(*case)
    prompt, images, boxes = pup_request_of(record)
    inputs = build_pup_inputs(prompt, images, boxes, prompts=prompts, tokenizer=tokenizer, processor=processor)

    # the reference recipe, independently of the module under test
    task = "i2i" if images else "t2i"
    system_key, user_text, locked = bbox.plan(prompt, list(boxes) if boxes else None, len(images))
    if images:
        references = [vend_pup.policy_reference(image, len(images)) for image in images]
        scaffold = vend_pup.I2I_SCAFFOLD[locked is not None].format(instruction=user_text)
        content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = processor(text=text, images=references, return_tensors="pt")
        expected_ids = encoded["input_ids"].tolist()
        assert torch.equal(inputs.pixel_values, encoded["pixel_values"])
        assert inputs.image_grid_thw.tolist() == encoded["image_grid_thw"].tolist()
        assert inputs.num_image_tokens == int(encoded["image_grid_thw"].prod(-1).sum()) // 4
        assert tuple(inputs.pixel_values.shape[1:]) == (1536,)
        assert [image.size for image in references] == [
            vend_pup.reference_size(image.width, image.height, len(images)) for image in images
        ]
    else:
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        expected_ids = [tokenizer.encode(text, add_special_tokens=False)]
        assert inputs.pixel_values is None and inputs.image_grid_thw is None and inputs.num_image_tokens == 0
    assert inputs.input_ids.tolist() == expected_ids
    assert inputs.input_ids.dtype == torch.int64 and inputs.text == text
    assert inputs.task == task and inputs.system_key == system_key
    assert inputs.presence_penalty == vend_pup.PRESENCE_PENALTY[task] == (1.5 if task == "t2i" else 0.0)
    assert inputs.text.endswith(TEMPLATE_TAIL)
    assert inputs.input_ids[0, -9:].tolist() == [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
    assert (inputs.locked is None) == (locked is None) and (inputs.locked is None or list(inputs.locked) == locked)
    if case in PINNED_SYSTEM_KEY:
        assert system_key == PINNED_SYSTEM_KEY[case]
    if case in PINNED_GRID:
        assert inputs.image_grid_thw.tolist() == PINNED_GRID[case]
    pinned = PUP_CASES[case]
    if pinned is not None:
        assert inputs.prompt_tokens == pinned, f"{pup_case_id(case)}: {inputs.prompt_tokens} tokens, pinned {pinned}"
    assert inputs.prompt_tokens <= PUP_MAX_PROMPT
    assert len(inputs.sha256()) == 64 and inputs.sha256() == inputs.sha256()
    record_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
    logger.info(
        f"RESULT inputs[{pup_case_id(case)}] key {system_key} prompt_tokens {inputs.prompt_tokens} image_tokens "
        f"{inputs.num_image_tokens} grid {None if inputs.image_grid_thw is None else inputs.image_grid_thw.tolist()} "
        f"record caption tokens {len(record_ids)} presence {inputs.presence_penalty}"
    )


def test_build_inputs_worst_case(pup_host) -> None:
    """U0a: ten 512x512 references and a 2000-char instruction: 3422 prompt tokens through the plain ``i2i``
    scaffold (2560 image tokens; measured 2026-10-19, logs/m9_builder_host.log; the designer's 3575 of
    logs/m9_designer_host_facts.log:17 fed the LOCKED scaffold to the ``i2i_bbox`` system prompt directly, +138
    template tokens) and, with one locked box, the ``i2i_bbox`` input under ``PUP_MAX_PROMPT``; the image
    placeholder count per reference at the per-count budgets."""
    tokenizer, processor, prompts = pup_host
    images = [Image.new("RGB", (512, 512), (i * 20, 100, 200)) for i in range(10)]
    inputs = build_pup_inputs("x" * 2000, images, None, prompts=prompts, tokenizer=tokenizer, processor=processor)
    assert (inputs.prompt_tokens, inputs.num_image_tokens) == (3422, 2560)
    assert inputs.system_key == "i2i"
    box = [{"id": "obj_1", "desc": "a box", "from": "target", "src_bbox": None, "tgt_bbox": [0.1, 0.1, 0.5, 0.5]}]
    boxed = build_pup_inputs("x" * 2000, images, box, prompts=prompts, tokenizer=tokenizer, processor=processor)
    assert boxed.system_key == "i2i_bbox" and boxed.num_image_tokens == 2560 and boxed.locked is not None
    assert inputs.prompt_tokens < boxed.prompt_tokens <= PUP_MAX_PROMPT
    assert boxed.prompt_tokens + FLUX3_KLEIN_PUP_MAX_NEW_TOKENS <= PUP_MAX_SEQ_LEN
    for n_refs, size, tokens in ((1, (1024, 1024), 1024), (2, (1024, 1024), 484), (3, (1024, 1024), 256)):
        one = build_pup_inputs(
            "x", [Image.new("RGB", size)] * n_refs, None, prompts=prompts, tokenizer=tokenizer, processor=processor
        )
        assert one.num_image_tokens == n_refs * tokens, (n_refs, one.num_image_tokens)
    logger.info(
        f"RESULT worst case prompt_tokens {inputs.prompt_tokens} (i2i) / {boxed.prompt_tokens} (i2i_bbox, one box) "
        f"image_tokens {inputs.num_image_tokens} cap {PUP_MAX_PROMPT}"
    )


def test_sizing_bit_exact(expect_error) -> None:
    """U0b: the vendored ``reference_size`` / ``policy_reference`` vs the clone's; pinned sizes
    (logs/m9_designer_host_facts.log:18)."""
    ref = import_reference_module("pup")
    # 35 x 7000 is exactly 200:1 (admitted; 32 x 7000 below is refused)
    sizes = [(4000, 3000), (1024, 1024), (896, 1120), (1536, 1024), (35, 7000), (31, 31), (4096, 4096), (1001, 999)]
    checked = 0
    for width, height in sizes:
        for n_refs in (1, 2, 3, 10):
            assert vend_pup.reference_size(width, height, n_refs) == ref.reference_size(width, height, n_refs)
            checked += 1
    assert vend_pup.reference_size(4000, 3000, 1) == (1152, 864)
    assert vend_pup.reference_size(1024, 1024, 2) == (704, 704)
    assert vend_pup.reference_size(1024, 1024, 3) == (512, 512)
    assert vend_pup.reference_size(896, 1120, 1) == (896, 1120)
    assert (
        vend_pup.MAX_PIXELS_BY_REFS
        == ref.MAX_PIXELS_BY_REFS
        == {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, 11)}}
    )
    with expect_error(ValueError, "aspect ratio"):
        vend_pup.reference_size(32, 7000, 1)
    with expect_error(ValueError, "aspect ratio"):
        ref.reference_size(32, 7000, 1)
    rng = np.random.default_rng(9)
    rgba = Image.fromarray(rng.integers(0, 256, (300, 500, 4), dtype=np.uint8), "RGBA")
    pal = Image.fromarray(rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)).convert("P")
    pal.info["transparency"] = 3
    big = Image.fromarray(rng.integers(0, 256, (1500, 2100, 3), dtype=np.uint8))
    for image, n_refs in ((rgba, 1), (pal, 2), (big, 1), (big, 3), (rgba, 10)):
        ours = vend_pup.policy_reference(image, n_refs)
        theirs = ref.policy_reference(image, n_refs)
        assert ours.size == theirs.size and ours.mode == theirs.mode == "RGB"
        assert np.array_equal(np.asarray(ours), np.asarray(theirs)), (image.mode, n_refs)
        checked += 1
    for name in ("1.png", "2.png", "3.png"):
        found = require_example_input(name)
        with Image.open(found) as image:
            for n_refs in (1, 2):
                assert np.array_equal(
                    np.asarray(vend_pup.policy_reference(image, n_refs)),
                    np.asarray(ref.policy_reference(image, n_refs)),
                )
                checked += 1
    logger.info(f"RESULT sizing bit-exact on {checked} rows vs {reference_clone()}")


def vllm_greedy(logits: torch.Tensor, output_ids: list[int], presence: float) -> int:
    """vLLM's sampler at temperature 0 transcribed: ``output_bin_counts = bincount(output_tokens)``;
    ``logits -= presence * (output_bin_counts > 0)``; argmax."""
    v = logits.float().clone()
    if output_ids:
        counts = torch.bincount(torch.tensor(output_ids, dtype=torch.long), minlength=v.numel())
        v -= presence * (counts > 0).to(v.dtype)
    return int(torch.argmax(v).item())


def test_presence_greedy() -> None:
    """U0c: ``presence_greedy`` == the vLLM transcription on random logits with repeated histories, penalty 0 / 1.5,
    incl. a case where the penalty flips the argmax; bf16 logits are read in fp32."""
    generator = torch.Generator().manual_seed(1234)
    vocab = 4096
    flips = 0
    for step in range(200):
        logits = torch.randn(vocab, generator=generator) * 4
        history = torch.randint(0, vocab, (int(step % 37),), generator=generator).tolist()
        history += history[: len(history) // 2]  # repeats: each id counts once
        if step % 2:  # every other step the current argmax is already in the output: the penalty must act on it
            history.append(int(torch.argmax(logits).item()))
        for penalty in (0.0, 1.5):
            got = presence_greedy(logits, history, penalty)
            assert got == vllm_greedy(logits, history, penalty), (step, penalty)
            bf16 = logits.to(torch.bfloat16)  # the device reads bf16 logits: the rule applies to their fp32 copy
            assert presence_greedy(bf16, history, penalty) == vllm_greedy(bf16.float(), history, penalty)
        plain = presence_greedy(logits, [], 0.0)
        if plain in history and presence_greedy(logits, history, 1.5) != plain:
            flips += 1
    assert flips > 0, "no history ever flipped the argmax: the penalty path was not exercised"
    # a constructed flip: the top logit is 0.5 above the runner-up and already generated
    logits = torch.zeros(10)
    logits[3], logits[7] = 5.0, 4.5
    assert presence_greedy(logits, [], 1.5) == 3 and presence_greedy(logits, [3], 1.5) == 7
    assert presence_greedy(logits, [3, 3, 3], 1.5) == 7 and presence_greedy(logits, [3], 0.0) == 3
    assert presence_greedy(logits, [7], 1.5) == 3
    # a [1, 1, vocab] row (the device's shape) and an int32 history
    assert presence_greedy(logits.reshape(1, 1, -1), [torch.tensor(3)], 1.5) == 7
    logger.info(f"RESULT presence_greedy == vLLM transcription on 400 rows, {flips} argmax flips observed")


def test_token_f1_record_pairs(pup_host) -> None:
    """U0c: ``token_f1`` / ``first_divergence`` on BFL's base vs distilled records of the same input reproduce the
    designer's calibration (logs/m9_designer_host_facts.log:56-62) to 4 decimals."""
    tokenizer = pup_host[0]
    for case, (f1, div, len_a, len_b) in RECORD_PAIRS.items():
        a = tokenizer.encode(load_record("base", case)["prompt"], add_special_tokens=False)
        b = tokenizer.encode(load_record("distilled", case)["prompt"], add_special_tokens=False)
        assert (len(a), len(b)) == (len_a, len_b), case
        assert abs(token_f1(a, b) - f1) < 5e-5, (case, token_f1(a, b))
        assert first_divergence(a, b) == div, case
        assert token_f1(a, a) == 1.0 and token_f1(b, b) == 1.0
    assert token_f1([1, 2, 3], [4, 5]) == 0.0 and token_f1([], [1]) == 0.0
    assert abs(token_f1([1, 1, 2], [1, 2, 2]) - 2 / 3) < 1e-12  # multiset intersection: {1, 2} of 3 and 3
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    s1 = load_record("upsampling", "double_exposure_s1")
    s2 = load_record("upsampling", "double_exposure_s2")
    assert s1["prompt"] == s2["prompt"] and s1["input_prompt"] == s2["input_prompt"] and s1["seed"] != s2["seed"]
    logger.info(f"RESULT token_f1 reproduces {len(RECORD_PAIRS)} record pairs")


def black_source(source: str) -> str:
    import black

    return black.format_str(source, mode=black.Mode(line_length=120))


def strip_header(path: Path) -> str:
    """The vendored file after its SPDX / provenance header (the comment lines before the first blank line)."""
    lines = path.read_text().splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[i:]).lstrip("\n")


def test_vendored_pup_is_reference(pup_snapshot) -> None:
    """U0d: ``bbox.py`` is the clone's file black-formatted; ``pup.py`` keeps the clone's constants and the source
    of ``reference_size`` / ``policy_reference`` / ``Upsampler.__call__`` (black-formatted) apart from the
    documented import rename; ``pup_prompts.json`` == the clone's == the snapshot's ``system_prompts.json`` (H6)."""
    clone = reference_clone()
    if clone is None:
        pytest.skip(f"{REFERENCE_CLONE_ENV} unset or without src/flux3/pup.py")
    ref_pup = import_reference_module("pup")
    assert strip_header(REFERENCE_DIR / "bbox.py") == black_source((clone / "src/flux3/bbox.py").read_text())
    for name in (
        "STOP_TOKENS",
        "MAX_REFERENCES",
        "ENGINE",
        "SAMPLING",
        "PRESENCE_PENALTY",
        "I2I_SCAFFOLD",
        "REFERENCE_MULTIPLE",
        "MAX_ASPECT_RATIO",
        "MAX_PIXELS_BY_REFS",
        "JPEG_QUALITY",
    ):
        assert getattr(vend_pup, name) == getattr(ref_pup, name), name
    for obj in ("reference_size", "policy_reference"):
        assert black_source(inspect.getsource(getattr(vend_pup, obj))) == black_source(
            inspect.getsource(getattr(ref_pup, obj))
        ), obj
    ours = black_source(inspect.getsource(vend_pup.Upsampler.__call__))
    theirs = black_source(inspect.getsource(ref_pup.Upsampler.__call__))
    assert ours == theirs
    vendored = json.loads((REFERENCE_DIR / "pup_prompts.json").read_text())
    assert vendored == json.loads((clone / "src/flux3/pup_prompts.json").read_text())
    assert vendored == json.loads((pup_snapshot / "system_prompts.json").read_text())
    assert vendored == load_system_prompts(pup_snapshot) == load_system_prompts(None)
    assert {k: len(v) for k, v in vendored.items()} == {"t2i": 3404, "i2i": 2677, "t2i_bbox": 3302, "i2i_bbox": 3034}
    assert (FLUX3_KLEIN_PUP_STOP_IDS, FLUX3_KLEIN_PUP_MAX_NEW_TOKENS) == ((248046, 248044), 3584)
    assert json.loads((pup_snapshot / "generation_config.json").read_text())["eos_token_id"] == list(
        FLUX3_KLEIN_PUP_STOP_IDS
    )
    logger.info("RESULT vendored pup.py / bbox.py / pup_prompts.json match the clone and the snapshot")


def test_refusals_and_caption(pup_host, expect_error, tmp_path) -> None:
    """U0e: the ``ValueError`` rows of ``build_pup_inputs`` (11 images, a prompt over ``PUP_MAX_PROMPT``), the stop
    ids of the tokenizer, ``finish_caption`` (stop id stripped, strip, empty -> ``RuntimeError``), the pass-through
    rule ``bbox.is_finished`` on the records (``generate.py:53``)."""
    tokenizer, processor, prompts = pup_host
    assert tuple(tokenizer.convert_tokens_to_ids(t) for t in vend_pup.STOP_TOKENS) == FLUX3_KLEIN_PUP_STOP_IDS
    assert tokenizer.eos_token_id == FLUX3_KLEIN_PUP_STOP_IDS[0]
    with expect_error(ValueError, "at most 10 references"):
        build_pup_inputs(
            "x", [Image.new("RGB", (64, 64))] * 11, None, prompts=prompts, tokenizer=tokenizer, processor=processor
        )
    # the cap is vLLM's: a prompt of max_model_len (8192) tokens or more is refused, a shorter one is generated to
    # min(3584, 8192 - prompt) tokens (new_token_budget); 4700 "x " tokens + the 729-token template = 5429 < 8192
    long_ok = build_pup_inputs("x " * 4700, [], None, prompts=prompts, tokenizer=tokenizer, processor=processor)
    assert 4608 < long_ok.prompt_tokens <= PUP_MAX_PROMPT
    assert new_token_budget(long_ok.prompt_tokens) == min(
        FLUX3_KLEIN_PUP_MAX_NEW_TOKENS, PUP_MAX_SEQ_LEN - long_ok.prompt_tokens
    )
    with expect_error(ValueError, "exceeds the upsampler"):
        build_pup_inputs("x " * 8300, [], None, prompts=prompts, tokenizer=tokenizer, processor=processor)
    assert PUP_MAX_PROMPT == PUP_MAX_SEQ_LEN - 1 == 8191
    assert new_token_budget(737) == 3584 and new_token_budget(4608) == 3584 and new_token_budget(4609) == 3583
    assert new_token_budget(8191) == 1 and new_token_budget(100, max_new_tokens=8) == 8
    with expect_error(ValueError, "no room"):
        new_token_budget(8192)
    with expect_error(ValueError, "aspect ratio"):
        build_pup_inputs(
            "x", [Image.new("RGB", (32, 7000))], None, prompts=prompts, tokenizer=tokenizer, processor=processor
        )
    # a long (not over-long) edit prompt is admitted with a shorter tail: 10 refs + 7000-char instruction
    long_edit = build_pup_inputs(
        "y " * 2600,
        [Image.new("RGB", (512, 512))] * 10,
        None,
        prompts=prompts,
        tokenizer=tokenizer,
        processor=processor,
    )
    assert 4608 < long_edit.prompt_tokens <= PUP_MAX_PROMPT, long_edit.prompt_tokens
    assert 0 < new_token_budget(long_edit.prompt_tokens) < FLUX3_KLEIN_PUP_MAX_NEW_TOKENS
    bad = tmp_path / "snapshot"
    bad.mkdir()
    (bad / "system_prompts.json").write_text(json.dumps({"t2i": "p"}))
    with expect_error(ValueError, "holds the system prompts"):
        load_system_prompts(bad)
    assert load_system_prompts(tmp_path) == prompts  # no system_prompts.json -> the vendored fallback
    ids = tokenizer.encode("  A caption.  ", add_special_tokens=False)
    assert finish_caption(ids + [FLUX3_KLEIN_PUP_STOP_IDS[0]], None, tokenizer=tokenizer) == "A caption."
    assert finish_caption(ids + [FLUX3_KLEIN_PUP_STOP_IDS[1]], None, tokenizer=tokenizer) == "A caption."
    assert finish_caption(ids, None, tokenizer=tokenizer) == "A caption."
    with expect_error(RuntimeError, "empty generation"):
        finish_caption([], None, tokenizer=tokenizer)
    with expect_error(RuntimeError, "empty generation"):
        finish_caption([FLUX3_KLEIN_PUP_STOP_IDS[0]], None, tokenizer=tokenizer)
    locked = bbox.plan("x", load_record("distilled", "balloons")["boxes"], 0)[2]
    answer = tokenizer.encode(
        'A <obj_1> and a <obj_2>. [{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "big"}]',
        add_special_tokens=False,
    )
    reconciled = finish_caption(answer, tuple(locked), tokenizer=tokenizer)
    assert bbox.is_finished(reconciled) and bbox.detect(reconciled)[1][0]["desc"] == "big"
    for case in ALL_CASES:
        record = load_record(*case)
        assert record["upsampled"] is True and not bbox.is_finished(record["input_prompt"]), case
        assert bbox.is_finished(record["prompt"]) == bool(record.get("boxes")), case
    knitted = load_record("base_vs_distilled", "knitted_distilled")
    assert knitted["upsampled"] is False and bbox.is_finished(knitted["prompt"])
    assert (
        knitted["prompt"]
        == f"{knitted['input_prompt'].rstrip()} {json.dumps(bbox.to_canvas(knitted['boxes']), ensure_ascii=False)}"
    )
    logger.info("RESULT refusals, stop ids, finish_caption and is_finished rows exact")


def test_reconcile() -> None:
    """U0f: ``bbox.reconcile`` (``bbox.py:125-163``): a row re-planning a locked box under its own id (IoU 1.0) is
    renamed to the locked id in the rows and the ``<id>`` citations, the locked rows keep their boxes and adopt
    the answer's desc, cited extra rows are appended, uncited ones dropped; the record captions are finished."""
    boxes = load_record("distilled", "balloons")["boxes"]
    key, text, locked = bbox.plan("hot-air balloons over a green valley at sunrise", boxes, 0)
    assert key == "t2i_bbox" and text.startswith("ORIGINAL PROMPT:\nhot-air balloons")
    assert locked == [
        {"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red and yellow striped hot-air balloon"},
        {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a small blue hot-air balloon"},
    ]
    answer = (
        "A <balloon_1> drifts above a <obj_2> under the <sky_1>. "
        '[{"id": "balloon_1", "bbox": [80, 60, 480, 380], "desc": "a striped balloon"}, '
        '{"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a blue balloon"}, '
        '{"id": "sky_1", "bbox": [0, 0, 200, 1000], "desc": "pale sky"}, '
        '{"id": "cloud_1", "bbox": [0, 0, 100, 100], "desc": "uncited"}]'
    )
    out = bbox.reconcile(answer, [dict(row) for row in locked])
    caption, rows = bbox.detect(out)
    assert caption == "A <obj_1> drifts above a <obj_2> under the <sky_1>."
    assert rows == [
        {"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a striped balloon"},
        {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a blue balloon"},
        {"id": "sky_1", "bbox": [0, 0, 200, 1000], "desc": "pale sky"},
    ]
    assert bbox.is_finished(out)
    # an answer without a table keeps the locked rows verbatim
    assert bbox.detect(bbox.reconcile("Just <obj_1> and <obj_2>.", [dict(r) for r in locked]))[1] == locked
    for case in ("balloons", "cactus"):
        record = load_record("distilled", case)
        assert bbox.is_finished(record["prompt"]) and not bbox.is_finished(record["input_prompt"])
    cactus_locked = bbox.plan(
        "add a small potted cactus in the marked spot", load_record("distilled", "cactus")["boxes"], 1
    )
    assert cactus_locked[0] == "i2i_bbox" and cactus_locked[2] == [
        {
            "id": "obj_1",
            "from": "target",
            "src_bbox": None,
            "tgt_bbox": [550, 620, 850, 850],
            "desc": "a small potted cactus",
        }
    ]
    logger.info("RESULT reconcile rows exact")


def test_subprocess_staging(tmp_path, pup_host, expect_error) -> None:
    """U0g: ``stage_reference`` (the subprocess client hands a PIL reference to the CLI child as a PNG) keeps the
    child's ``policy_reference(Image.open(png))`` pixel-equal to ``policy_reference(image)`` for an EXIF-rotated
    JPEG (orientation 6, which a plain PNG save dropped: M9 finding) and an RGBA image; ``audit_target_ids`` refuses
    a layout caption; ``divergence_audit`` on synthetic rows (boundary exclusion, tie scale, suspicious, errors)."""
    rng = np.random.default_rng(19)
    base = Image.fromarray(rng.integers(0, 256, (300, 480, 3), dtype=np.uint8))
    jpeg = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW on load
    base.save(jpeg, format="JPEG", quality=95, exif=exif.tobytes())
    with Image.open(jpeg) as rotated:
        assert rotated.getexif().get(0x0112) == 6
        expected = vend_pup.policy_reference(rotated, 1)
        staged = Image.open(stage_reference(rotated, tmp_path / "staged_0.png"))
    assert staged.size == (300, 480) and expected.size == (288, 480)  # transposed, then the 32-multiple resize
    assert np.array_equal(np.asarray(vend_pup.policy_reference(staged, 1)), np.asarray(expected))
    # the earlier hand-over (a plain PNG save) is NOT faithful: the orientation tag is gone, no transpose
    with Image.open(jpeg) as rotated:
        rotated.save(tmp_path / "plain.png")
    plain = Image.open(tmp_path / "plain.png")
    assert plain.size == (480, 300) and vend_pup.policy_reference(plain, 1).size != expected.size
    rgba = Image.fromarray(rng.integers(0, 256, (200, 260, 4), dtype=np.uint8), "RGBA")
    staged_rgba = Image.open(stage_reference(rgba, tmp_path / "staged_1.png"))
    assert staged_rgba.mode == "RGB"
    for n_refs in (1, 3):
        assert np.array_equal(
            np.asarray(vend_pup.policy_reference(staged_rgba, n_refs)),
            np.asarray(vend_pup.policy_reference(rgba, n_refs)),
        )
    # audit targets: text captions only
    tokenizer = pup_host[0]
    man_dog = load_record("distilled", "man_dog")["prompt"]
    target = audit_target_ids(man_dog, tokenizer)
    assert target[:-1] == tokenizer.encode(man_dog, add_special_tokens=False) and target[-1] == 248046
    with expect_error(ValueError, "layout caption"):
        audit_target_ids(load_record("distilled", "balloons")["prompt"], tokenizer)
    with expect_error(ValueError, "layout caption"):
        audit_target_ids("plain text", tokenizer, locked=({"id": "obj_1"},))
    assert all(c not in PUP_BBOX_CASES and c != ("distilled", "cactus") for c in AUDIT_CASES)
    assert len(AUDIT_CASES) == len(ALL_CASES) - 2
    # divergence_audit: positions 0 and the last two are boundary; interior rows decide
    targets = [10, 11, 12, 13, 14, 15, 248046]
    device = [
        (10, 10, 0.0),
        (99, 11, 0.5),
        (12, 12, 0.0),
        (98, 13, 0.3),
        (14, 14, 0.0),
        (97, 15, 1.0),
        (1, 248046, 2.0),
    ]
    oracle = [
        (0, 10, 10, 0.0),
        (1, 11, 11, 0.02),
        (2, 12, 12, 0.0),
        (3, 96, 13, 0.4),
        (4, 14, 14, 0.0),
        (5, 15, 15, 0.0),
        (6, 248046, 248046, 0.0),
    ]
    report = divergence_audit(device, oracle, tie_gap=None)
    assert AUDIT_BOUNDARY == (1, 2) and (report["positions"], report["interior"]) == (7, 4)
    assert (report["dev_agree"], report["orc_agree"]) == (3, 6)  # device agrees at 0, 2, 4; the oracle everywhere but 3
    assert [s[0] for s in report["suspicious"]] == [1]  # pos 3: the oracle disagrees too; pos 5 is boundary
    assert report["tie_scale"] == 0.4 and report["max_suspicious_oracle_gap"] == 0.02
    assert report["errors"] == [] and report["passed"] is False  # no frozen tie gap: never a pass
    assert divergence_audit(device, oracle, tie_gap=0.01)["errors"] == report["suspicious"]
    assert divergence_audit(device, oracle, tie_gap=0.01)["passed"] is False
    assert divergence_audit(device, oracle, tie_gap=0.1)["passed"] is True
    with expect_error(ValueError, "positions"):
        divergence_audit(device[:-1], oracle, tie_gap=None)
    logger.info("RESULT staging pixel-exact (EXIF orientation 6, RGBA); audit target / divergence rows exact")


def two_sizes_request(pup_host) -> tuple[str, list, U.PupInputs]:
    """The U0h / U7 request: two references of different sizes and the bandana-style edit instruction."""
    tokenizer, processor, prompts = pup_host
    images = [Image.open(require_example_input("1.png")), Image.open(require_example_input("2.png"))]
    prompt = "put the dog from the second image next to the figurine of the first image"
    inputs = build_pup_inputs(prompt, images, None, prompts=prompts, tokenizer=tokenizer, processor=processor)
    return prompt, images, inputs


def test_split_images(pup_host, expect_error) -> None:
    """U0h: ``split_images`` cuts the packed patches per image in attachment order with the per-image grids; the
    two-sizes request has unequal token counts (500 + 484 = 984 image tokens, grids [1, 50, 40] and [1, 44, 44];
    2026-10-19 fixer measurement, logs/m9_fixer_host_split.log), the mugprint request two equal ones (484 + 484)."""
    _, images, inputs = two_sizes_request(pup_host)
    assert [image.size for image in images] == [(896, 1120), (1024, 1024)]
    assert inputs.image_grid_thw.tolist() == [[1, 50, 40], [1, 44, 44]] and inputs.num_image_tokens == 984
    parts = split_images(inputs.pixel_values, inputs.image_grid_thw)
    assert [(int(p.shape[0]), g.tolist()) for p, g in parts] == [(2000, [[1, 50, 40]]), (1936, [[1, 44, 44]])]
    assert torch.equal(torch.cat([p for p, _ in parts]), inputs.pixel_values)
    assert [int(g.prod()) // 4 for _, g in parts] == [500, 484]
    assert sorted({vision_padded_rows(int(g.prod())) for _, g in parts}) == [2048] and vision_padded_rows(2048) == 4096
    with expect_error(ValueError, "patch rows"):
        split_images(inputs.pixel_values[:-1], inputs.image_grid_thw)
    tokenizer, processor, prompts = pup_host
    record = load_record("distilled", "2_3_mugprint")
    prompt, mug_images, boxes = pup_request_of(record)
    mug = build_pup_inputs(prompt, mug_images, boxes, prompts=prompts, tokenizer=tokenizer, processor=processor)
    mug_parts = split_images(mug.pixel_values, mug.image_grid_thw)
    assert [g.tolist() for _, g in mug_parts] == [[[1, 44, 44]], [[1, 44, 44]]] and mug.num_image_tokens == 968
    logger.info("RESULT split_images: two sizes 2000 + 1936 patches (500 + 484 tokens), mugprint 2 x 1936")


# --- device --------------------------------------------------------------------------------------------------------------------------

mesh_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(1, 1), U.PUP_DEVICE_PARAMS]],
    ids=["bh_1x1_nofabric"],
    indirect=["mesh_device", "device_params"],
)


@pytest.fixture
def pup(mesh_device):
    """One ``Flux3KleinUpsampler`` on the test's 1-chip mesh (the mesh fixture is function-scoped, so each row loads
    the weights from the tensor cache); closed after the row."""
    from .device_params import PUP_DEVICE_PARAMS

    assert PUP_DEVICE_PARAMS == U.PUP_DEVICE_PARAMS
    snapshot = variant_checkpoint_dir("pup")
    if snapshot is None:
        pytest.skip("pup snapshot not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    upsampler = U.Flux3KleinUpsampler(snapshot, mesh_device=mesh_device)
    yield upsampler
    upsampler.close()


def timings_file() -> Path | None:
    out = renders_dir()
    return out / "pup_timings.json" if out is not None else None


def save_timings(label: str, timings: U.PupTimings, extra: dict | None = None) -> None:
    path = timings_file()
    if path is None:
        return
    data = json.loads(path.read_text()) if path.is_file() else {}
    from dataclasses import asdict

    data[label] = {**asdict(timings), **(extra or {}), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(data, indent=1))


def run_audit(pup: U.Flux3KleinUpsampler, inputs, oracle: dict | None, label: str) -> dict | None:
    """Design 9.3 on one case: the device's teacher-forced rows against the oracle's cached gaps through
    ``divergence_audit`` (``TIE_GAP`` of this module); ``None`` when the oracle has no gaps on disk or the case has
    no audit target (a layout caption). Logs the RESULT / AUDIT lines."""
    if oracle is None or "gaps" not in oracle or inputs.locked:
        return None
    targets = oracle["teacher_force_ids"]
    device = pup.teacher_forced(inputs, targets)
    report = divergence_audit(device, oracle["gaps"], tie_gap=TIE_GAP)
    logger.info(
        f"RESULT audit[{label}] device agreement {report['dev_agree']}/{report['positions']} oracle agreement "
        f"{report['orc_agree']}/{report['positions']} interior {report['interior']} device-only disagreements "
        f"{len(report['suspicious'])} oracle tie scale {report['tie_scale']:.4f} max oracle gap at device-only "
        f"disagreements {report['max_suspicious_oracle_gap']:.4f} TIE_GAP {TIE_GAP} errors {len(report['errors'])} "
        f"passed {report['passed']}"
    )
    for pos, top, tgt, gap, o_gap in report["suspicious"][:20]:
        logger.info(f"AUDIT[{label}] pos {pos} device {top} record {tgt} device gap {gap:.4f} oracle gap {o_gap:.4f}")
    return report


def gate_caption(pup: U.Flux3KleinUpsampler, case: tuple[str, str], record_property) -> dict:
    """Design 9.2 on one recorded case; returns the facts and asserts the gate. Verdicts: ``exact`` / ``record-f1``
    (PLAN M9's record gate), ``oracle-exact`` / ``oracle-f1`` (the oracle tiers of design 9.2 step 3; ``oracle-f1``
    only with the 9.3 audit passed), ``no-oracle`` / ``no-audit`` / ``fail`` (assert). An oracle-tier pass is
    warned (pytest's warnings summary), so it is never silently green."""
    label = pup_case_id(case)
    record = load_record(*case)
    prompt, images, boxes = pup_request_of(record)
    inputs = pup.inputs(prompt, images, boxes)
    pinned = PUP_CASES.get(case)
    assert pinned is None or inputs.prompt_tokens == pinned, (inputs.prompt_tokens, pinned)
    ids, timings = pup.generate_ids(inputs)
    caption = finish_caption(ids, inputs.locked, tokenizer=pup.tokenizer)
    tokenizer = pup.tokenizer
    record_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)
    exact_rec = caption == record["prompt"]
    f1_rec = token_f1(caption_ids, record_ids)
    div_rec = first_divergence(caption_ids, record_ids)
    assert not any(t in pup.stop_ids for t in ids) and len(ids) <= FLUX3_KLEIN_PUP_MAX_NEW_TOKENS
    oracle = pup_oracle.cached_oracle(pup.snapshot, inputs, case=label)
    exact_orc = f1_orc = f1_orc_rec = None
    if oracle is not None:
        orc_caption = oracle.get("caption") or ""
        orc_ids = tokenizer.encode(orc_caption, add_special_tokens=False)
        exact_orc = caption == orc_caption
        f1_orc = token_f1(caption_ids, orc_ids)
        f1_orc_rec = token_f1(orc_ids, record_ids)
    facts = {
        "case": label,
        "exact_rec": exact_rec,
        "f1_rec": f1_rec,
        "first_div": div_rec,
        "len": len(caption_ids),
        "record_len": len(record_ids),
        "exact_orc": exact_orc,
        "f1_orc": f1_orc,
        "f1_orc_rec": f1_orc_rec,
        "ttft_s": timings.ttft_s,
        "tok_s": timings.decode_tok_s,
        "total_s": timings.total_s,
        "trace_captured": timings.trace_captured,
    }
    logger.info(
        f"RESULT pup[{label}] exact_rec {exact_rec} f1_rec {f1_rec:.4f} exact_orc {exact_orc} f1_orc "
        f"{'None' if f1_orc is None else f'{f1_orc:.4f}'} f1_orc_rec {'None' if f1_orc_rec is None else f'{f1_orc_rec:.4f}'} "
        f"first_div {div_rec} len {len(caption_ids)} ttft {timings.ttft_s:.2f} tok_s {timings.decode_tok_s:.1f} "
        f"total {timings.total_s:.1f}"
    )
    logger.info(f"CAPTION[{label}] {caption}")
    save_timings(label, timings, {"exact_rec": exact_rec, "f1_rec": f1_rec})
    if exact_rec:
        verdict = "exact"
    elif f1_rec >= RECORD_F1:
        verdict = "record-f1"
    elif oracle is None:
        verdict = "no-oracle"
    elif exact_orc:
        verdict = "oracle-exact"
    elif f1_orc >= ORACLE_F1:
        verdict = "oracle-f1"
    else:
        verdict = "fail"
    audit = None
    if verdict == "oracle-f1":
        # design 9.2 step 3: F1 >= 0.9 vs the oracle AND the divergence audit of 9.3 passes
        audit = run_audit(pup, inputs, oracle, label)
        facts["audit"] = None if audit is None else {k: v for k, v in audit.items() if k != "suspicious"}
        if audit is None:
            verdict = "no-audit"
        elif not audit["passed"]:
            verdict = "fail"
    facts["verdict"] = verdict
    record_property("pup_gate", json.dumps(facts))
    cache_dir = pup_oracle_cache()
    assert verdict != "no-oracle", (
        f"{label}: record F1 {f1_rec:.4f} < {RECORD_F1} (first divergence at token {div_rec}) and no CPU oracle on disk "
        f"({cache_dir}); run `python -m models.tt_dit.tests.models.flux3_klein.pup_oracle --case {case[0]}/{case[1]} "
        f"--teacher-force` on a node CPU"
    )
    assert verdict != "no-audit", (
        f"{label}: record F1 {f1_rec:.4f} < {RECORD_F1}, oracle F1 {f1_orc:.4f} >= {ORACLE_F1} but the 9.3 audit cannot "
        f"run ({'a layout caption has no audit target' if inputs.locked else 'no oracle gaps on disk: run pup_oracle --teacher-force'})"
    )
    assert verdict != "fail", (
        f"{label}: record F1 {f1_rec:.4f} < {RECORD_F1} and oracle F1 {f1_orc:.4f} vs {ORACLE_F1} "
        f"(oracle vs record {f1_orc_rec:.4f}); first divergence vs record at {div_rec}"
        + (
            ""
            if audit is None
            else f"; audit: TIE_GAP {TIE_GAP}, {len(audit['errors'])} device-only disagreements above it "
            f"(passed {audit['passed']})"
        )
    )
    if verdict.startswith("oracle"):
        warnings.warn(
            f"pup gate {label}: PASS on the {verdict} tier (record F1 {f1_rec:.4f} < {RECORD_F1}, oracle F1 "
            f"{f1_orc:.4f}, oracle vs record {f1_orc_rec:.4f}); PLAN M9's record gate was not met",
            UserWarning,
            stacklevel=2,
        )
    return facts


@mesh_rows
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", PUP_T2I_CASES, ids=[pup_case_id(c) for c in PUP_T2I_CASES])
def test_caption_t2i(mesh_device, pup, case, record_property) -> None:
    """U1: text-to-image (presence penalty 1.5) against the record / the oracle (design 9.2)."""
    gate_caption(pup, case, record_property)


@mesh_rows
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", PUP_BBOX_CASES, ids=[pup_case_id(c) for c in PUP_BBOX_CASES])
def test_caption_bbox(mesh_device, pup, case, record_property) -> None:
    """U2: layout mode (``t2i_bbox``); the record is a reconciled answer, the device caption goes through the same
    ``reconcile``."""
    facts = gate_caption(pup, case, record_property)
    logger.info(f"RESULT bbox[{facts['case']}] verdict {facts['verdict']}")


@mesh_rows
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", PUP_I2I_CASES, ids=[pup_case_id(c) for c in PUP_I2I_CASES])
def test_caption_i2i(mesh_device, pup, case, record_property) -> None:
    """U3: edits (the vision tower, presence penalty 0); may be BLOCKED by the 9B tower on one chip (design
    decision 6) without blocking U1 / U2."""
    gate_caption(pup, case, record_property)


@mesh_rows
@pytest.mark.timeout(7200)
def test_determinism_and_stop(mesh_device, pup, expect_error) -> None:
    """U4: two runs of man_dog bit-equal; double_exposure s1 / s2 (the same input) bit-equal; no stop id in the ids;
    the ``max_new_tokens`` cap returns exactly that many ids; ``finish_caption([])`` raises."""
    tokenizer = pup.tokenizer
    man_dog = load_record("distilled", "man_dog")
    inputs = pup.inputs(man_dog["input_prompt"], [], None)
    first, t_first = pup.generate_ids(inputs)
    second, t_second = pup.generate_ids(inputs)
    assert first == second, f"non-deterministic: first divergence at {first_divergence(first, second)}"
    assert not any(t in pup.stop_ids for t in first) and 0 < len(first) <= FLUX3_KLEIN_PUP_MAX_NEW_TOKENS
    capped, _ = pup.generate_ids(inputs, max_new_tokens=8)
    assert capped == first[:8] and len(capped) == 8
    s1 = pup.inputs(load_record("upsampling", "double_exposure_s1")["input_prompt"], [], None)
    s2 = pup.inputs(load_record("upsampling", "double_exposure_s2")["input_prompt"], [], None)
    assert torch.equal(s1.input_ids, s2.input_ids)
    ids_s1, _ = pup.generate_ids(s1)
    ids_s2, t_s2 = pup.generate_ids(s2)
    assert ids_s1 == ids_s2
    with expect_error(RuntimeError, "empty generation"):
        finish_caption([], None, tokenizer=tokenizer)
    save_timings("man_dog_warm", t_second)
    save_timings("double_exposure_warm", t_s2)
    logger.info(
        f"RESULT determinism: man_dog {len(first)} ids x2 equal, double_exposure {len(ids_s1)} ids equal; warm "
        f"man_dog ttft {t_second.ttft_s:.2f}s decode {t_second.decode_tok_s:.1f} tok/s total {t_second.total_s:.1f}s "
        f"(first run: trace captured {t_first.trace_captured}, total {t_first.total_s:.1f}s)"
    )


@mesh_rows
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case", AUDIT_CASES, ids=[pup_case_id(c) for c in AUDIT_CASES])
def test_teacher_forced_audit(mesh_device, pup, case) -> None:
    """U5 (design 9.3): the device's top-1 vs the record under teacher forcing, next to the oracle's
    (``divergence_audit``); a device disagreement where the oracle agrees with the record by a gap above
    ``TIE_GAP`` is a numerics error. Logs only until ``TIE_GAP`` is frozen; needs the oracle's gaps on disk
    (``pup_oracle --teacher-force``). Text-to-image / edit cases only (``AUDIT_CASES``)."""
    label = pup_case_id(case)
    record = load_record(*case)
    prompt, images, boxes = pup_request_of(record)
    inputs = pup.inputs(prompt, images, boxes)
    assert not inputs.locked, label
    oracle = pup_oracle.cached_oracle(pup.snapshot, inputs, case=label)
    if oracle is None or "gaps" not in oracle:
        pytest.skip(f"{label}: no oracle gaps on disk; run pup_oracle --case {case[0]}/{case[1]} --teacher-force")
    assert oracle["teacher_force_ids"] == audit_target_ids(record["prompt"], pup.tokenizer)
    report = run_audit(pup, inputs, oracle, label)
    assert report is not None and report["positions"] == len(oracle["teacher_force_ids"])
    if TIE_GAP is not None:
        assert not report[
            "errors"
        ], f"{label}: {len(report['errors'])} device disagreements above TIE_GAP {TIE_GAP}: {report['errors'][:5]}"


@mesh_rows
@pytest.mark.timeout(7200)
def test_timings(mesh_device, pup) -> None:
    """U6: the warm man_dog and bandana RECORD lines into ``$FLUX3_KLEIN_OUT/pup_timings.json``."""
    for case in (("distilled", "man_dog"), ("distilled", "3_bandana")):
        record = load_record(*case)
        prompt, images, boxes = pup_request_of(record)
        inputs = pup.inputs(prompt, images, boxes)
        pup.generate_ids(inputs)
        ids, warm = pup.generate_ids(inputs)
        save_timings(f"{pup_case_id(case)}_warm", warm)
        logger.info(
            f"RESULT timings[{pup_case_id(case)}] warm: load {warm.load_s:.1f}s vision {warm.vision_s:.2f}s prefill "
            f"{warm.prefill_s:.2f}s ttft {warm.ttft_s:.2f}s decode {warm.decode_tok_s:.1f} tok/s new {len(ids)} total "
            f"{warm.total_s:.1f}s {warm.dram}"
        )


@mesh_rows
@pytest.mark.timeout(7200)
def test_two_sizes_i2i(mesh_device, pup, pup_host) -> None:
    """U7: two references of different sizes (the per-image vision path, M9 fixer finding 13): the tower runs once
    per image, the packed rows are 984 (500 + 484), a caption is produced and a second run is bit-equal."""
    prompt, images, _ = two_sizes_request(pup_host)
    inputs = pup.inputs(prompt, images, None)
    assert inputs.image_grid_thw.shape[0] == 2 and inputs.num_image_tokens == 984
    ids, timings = pup.generate_ids(inputs)
    caption = finish_caption(ids, inputs.locked, tokenizer=pup.tokenizer)
    assert 0 < len(ids) <= FLUX3_KLEIN_PUP_MAX_NEW_TOKENS and not any(t in pup.stop_ids for t in ids)
    again, warm = pup.generate_ids(inputs)
    assert again == ids, f"non-deterministic: first divergence at {first_divergence(ids, again)}"
    save_timings("two_sizes_warm", warm, {"image_tokens": inputs.num_image_tokens, "new": len(ids)})
    logger.info(
        f"RESULT two_sizes: grids {inputs.image_grid_thw.tolist()} image_tokens {inputs.num_image_tokens} new "
        f"{len(ids)} finish {timings.finish} vision {warm.vision_s:.2f}s ttft {warm.ttft_s:.2f}s decode "
        f"{warm.decode_tok_s:.1f} tok/s total {warm.total_s:.1f}s (vision compiled {timings.vision_compiled})"
    )
    logger.info(f"CAPTION[two_sizes] {caption}")
