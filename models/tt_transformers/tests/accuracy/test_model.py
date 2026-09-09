# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for tt_transformers: teacher-forced token agreement against a stored reference.

Run it with the same two variables the demos take:

    TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
    pytest models/tt_transformers/tests/accuracy/test_model.py -x -q

It exists so an automated optimizer has something to check a change against. tt-optimization-loop
looks for `tests/accuracy/test_*.py` under the model folder, runs it before any edit and after
every one, reads the `top-1` and `top-5` figures it prints, and undoes any change that fails it
or lowers either figure. Without this file the loop refuses to start on this model: "No accuracy
test was found under tests/accuracy, so there is no correctness gate."

What it measures. The reference file at `reference_outputs/<model>.refpt` holds 1024 tokens of
text scored by the full-precision HuggingFace model: at each position, the token it predicted
and its top five. The first half is the prompt. Over the second half the on-chip model is
*teacher-forced*: at every step it predicts a token, the prediction is recorded and discarded,
and the reference's token is fed in instead. So the score is 512 independent one-step
predictions from a known-good prefix, and one early disagreement cannot cascade into the rest.
Top-1 is the share of positions where the chip's prediction is the reference's most likely
token; top-5 the share where it is among the reference's five most likely. This is the same
measurement simple_text_demo.py's `ci-token-matching` case makes, in a file the loop can find,
that passes on its own, and that prints its figures where a reader -- human or tool -- can see
them.

Floors. `models/model_targets.yaml` is consulted first, so a model with centralised targets is
held to them. A model without an entry -- Qwen3-8B, at the time of writing -- is held to 0.83
top-1 and 0.96 top-5, the floors the loop's own design record uses. Qwen3-8B measures about
0.852 and 0.978 here, so a single flipped prediction (0.002) is noise and eleven are the margin.
The loop is stricter than these floors: it also refuses any change whose figures come out lower
than the unchanged model's, so the numbers printed by the first run become the bar.

Cost. Prefill of 512 tokens plus 512 untraced decode steps at roughly 90 ms each: about a
minute on a warm weight cache, plus the model build. The loop pays this twice per experiment.
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.utils.model_targets import resolve_accuracy_targets
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model, preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_outputs")

DEFAULT_FLOORS = {"top1": 0.83, "top5": 0.96}
"""Applied when model_targets.yaml has no accuracy entry for this model and board."""

MAX_SEQ_LEN = 2048
"""Room for the 1024-token reference with a margin; the reference never exceeds it."""
PAGE_BLOCK_SIZE = 32
OPTIMIZATIONS = DecodersPrecision.performance
"""The preset the optimizer tunes, so the gate judges the configuration that actually ships."""


class TokenAccuracy:
    """The reference split into a prompt half and a scored half, and the running tally."""

    def __init__(self, reference_path):
        if not os.path.exists(reference_path):
            pytest.fail(
                f"No reference at {reference_path}. Generate one with:\n"
                f"  HF_MODEL=$HF_MODEL python models/tt_transformers/tests/generate_reference_outputs.py "
                f"--total_length 1024 --model $HF_MODEL --output_file {reference_path}"
            )
        data = torch.load(reference_path)
        reference_tokens = data["reference_tokens"]
        split = reference_tokens.shape[-1] // 2
        self.input_prompt = reference_tokens[0, :split]
        self.reference_tokens = reference_tokens[0, split:]
        self.top5_tokens = data["top5_tokens"][split - 1 :, :]
        self.maxindex = len(self.reference_tokens) - 1
        self.gt_pos = -1
        self.store_predicted_tokens = []

    def collect_predicted_tokens(self, token):
        """Record what the model predicted; return what it must be fed next."""
        self.store_predicted_tokens.append(token)
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self):
        n = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        top1 = sum(self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i] for i in range(n))
        top5 = sum(self.store_predicted_tokens[i] in self.top5_tokens[i, :] for i in range(n))
        return top1 / n, top5 / n, n


class PassMarks:
    """The pass rule: centralised targets when the model has them, the default floors otherwise.

    Two rules, kept distinct because they round differently. model_targets.yaml stores whole
    percentages (87 means 87%), and simple_text_demo.py compares against them the way CI does:
    the measured fraction is rounded *up* to a whole percent, then must reach the target less
    half a point -- so a measured 0.8612 passes a target of 87. This gate applies exactly that
    rule where an entry exists, so a model that passes CI passes here. Without an entry the
    default floors are fractions and compared directly.
    """

    def __init__(self, model_args, seq_len):
        targets = resolve_accuracy_targets(
            model_name=model_args.base_model_name, sku=model_args.device_name, batch_size=1, seq_len=seq_len
        )
        if targets and "top1" in targets and "top5" in targets:
            self.source = "models/model_targets.yaml"
            self.percent_targets = {"top1": float(targets["top1"]), "top5": float(targets["top5"])}
        else:
            self.source = "default floors (no entry in models/model_targets.yaml)"
            self.percent_targets = None

    def describe(self, name):
        if self.percent_targets is not None:
            return f"{self.percent_targets[name]:.0f}% after rounding up to a whole percent"
        return f"{DEFAULT_FLOORS[name]:.4f}"

    def passes(self, name, value):
        if self.percent_targets is not None:
            return math.ceil(value * 100) >= self.percent_targets[name] - 0.5
        return value >= DEFAULT_FLOORS[name]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_token_accuracy(mesh_device, reset_seeds):
    assert os.getenv("HF_MODEL"), "Set HF_MODEL, e.g. export HF_MODEL=Qwen/Qwen3-8B"
    model_name = os.environ["HF_MODEL"].strip("/").split("/")[-1]
    token_acc = TokenAccuracy(os.path.join(REFERENCE_DIR, f"{model_name}.refpt"))
    scored = len(token_acc.reference_tokens)

    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_SEQ_LEN // PAGE_BLOCK_SIZE,
    )
    t_build = time.perf_counter()
    model_args, model, kv_cache, _state_dict = create_tt_model(
        mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=lambda args: OPTIMIZATIONS(args.n_layers, args.model_name),
        max_seq_len=MAX_SEQ_LEN,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
    )
    logger.info(f"Model built in {time.perf_counter() - t_build:.1f}s ({model_args.n_layers} layers)")
    tokenizer = model_args.tokenizer
    page_table = torch.argsort(torch.randperm(paged_attention_config.max_num_blocks)).reshape(1, -1)
    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    # The reference was scored on raw text with no chat template, so the prompt is handed over
    # as text and re-encoded the same way. A tokenizer that does not round-trip would shift every
    # scored position, so that is checked rather than assumed.
    prompt_text = tokenizer.decode(token_acc.input_prompt.tolist())
    input_tokens_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt_text], tokenizer, [model_args], False, scored, max_prefill_len=MAX_SEQ_LEN
    )
    assert len(encoded_prompts[0]) == len(token_acc.input_prompt), (
        f"reference prompt is {len(token_acc.input_prompt)} tokens but re-encoded to "
        f"{len(encoded_prompts[0])}; the scored positions would be misaligned"
    )
    input_tokens = torch.stack(input_tokens_pt).view(1, -1)
    assert scored + decoding_pos[0] <= MAX_SEQ_LEN

    # Host argmax throughout: sampling_params=None returns logits, so this path does not depend
    # on whether the model has an on-device sampler, and greedy is what the reference scored.
    logger.info(f"Prefill of {decoding_pos[0]} tokens (padded to {prefill_lens[0]})...")
    logits = generator.prefill_forward_text(
        input_tokens,
        page_table=page_table,
        kv_cache=[kv_cache],
        prompt_lens=decoding_pos,
        sampling_params=None,
        enable_trace=False,
        warmup_prefill=False,
    )
    out_tok = torch.argmax(logits, dim=-1)
    current_pos = torch.tensor(decoding_pos)

    logger.info(f"Teacher-forcing {scored} decode steps...")
    t_decode = time.perf_counter()
    for iteration in range(scored):
        out_tok[0] = token_acc.collect_predicted_tokens(out_tok[0].item())
        logits, _log_probs = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=[kv_cache],
            reset_batch=(iteration == 0),
            sampling_params=None,
            prompt_tokens=input_tokens,
            output_tokens=out_tok,
        )
        out_tok = torch.argmax(logits, dim=-1)
        current_pos += 1
    logger.info(f"Decode took {time.perf_counter() - t_decode:.1f}s")

    top1, top5, n = token_acc.compute_accuracy()
    marks = PassMarks(model_args, seq_len=token_acc.input_prompt.shape[-1] + scored)
    # Printed once each, as a fraction, before any mention of the pass marks: the loop reads the
    # first `top-1 ... <number>` and `top-5 ... <number>` it sees as the measurement.
    logger.info(f"top-1 accuracy: {top1:.4f} over {n} teacher-forced predictions")
    logger.info(f"top-5 accuracy: {top5:.4f} over {n} teacher-forced predictions")
    logger.info(f"pass marks from {marks.source}: {marks.describe('top1')} and {marks.describe('top5')}")

    assert marks.passes("top1", top1), (
        f"{model_args.model_name} agreed with the reference's first choice on {top1:.4f} of {n} "
        f"predictions, under the {marks.describe('top1')} mark ({marks.source})"
    )
    assert marks.passes("top5", top5), (
        f"{model_args.model_name} landed in the reference's top five on {top5:.4f} of {n} "
        f"predictions, under the {marks.describe('top5')} mark ({marks.source})"
    )
