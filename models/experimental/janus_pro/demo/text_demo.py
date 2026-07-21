# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Janus-Pro text demo: engages only the LLaMA text-decode path (no vision tower)
and scores greedy next-token predictions against a HuggingFace fp32 reference
stored in ``models/tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt``.

Modeled on ``models/demos/multimodal/gemma3/demo/text_demo.py``: teacher-forced
decode (feed the reference ground-truth token each step) + top-1/top-5 accuracy,
plus warmup and BenchmarkProfiler perf capture. Parametrized over ``enable_trace``
(``notrace`` / ``trace``): use notrace for Top-1/5 (Accuracy table), trace for
Speed/TTFT (Performance table). Manual-run only — the CI save/verify path is
intentionally absent.

Generate the reference first (host job)::

    python3 models/tt_transformers/tests/generate_reference_hf.py \\
        --model deepseek-community/Janus-Pro-7B \\
        --output_file models/tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt \\
        --total_length 1024 --trust-remote-code
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.janus_pro.tt.janus_pro_e2e_model import JanusMultimodalGenerator, TtJanusProModel
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import create_submeshes

# Janus ModelArgs.model_name is a snapshot hash, not a stable label, so the refpt
# filename is pinned here rather than derived from the model.
JANUS_REFPT_NAME = "Janus-Pro-7B"


def create_multimodal_model(mesh_device, max_batch_size, max_seq_len, dtype=ttnn.bfloat8_b):
    model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, cache_hf=True)
    state_dict = model_args.load_state_dict()
    model = TtJanusProModel(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        vision_dtype=ttnn.bfloat16,
    )
    return model_args, model


def _greedy(logits):
    # logits: [B, seq, vocab] or [B, vocab]; take the last position and argmax.
    if logits.dim() == 3:
        logits = logits[:, -1]
    return torch.argmax(logits, dim=-1).reshape(-1)


class TokenAccuracy:
    """Loads a reference .refpt, teacher-forces the ground-truth continuation, and
    scores predicted tokens against the reference top-1/top-5. The reference's first
    half is the prompt; the second half is the scored ground truth."""

    def __init__(self, model_name):
        self.gt_pos = -1
        self.store_predicted_tokens = []
        reference_data_file = os.path.join("models/tt_transformers/tests/reference_outputs/", model_name) + ".refpt"
        assert os.path.exists(reference_data_file), f"Reference file not found: {reference_data_file}"
        logger.info(f"Loading reference data from {reference_data_file}")
        reference_data = torch.load(reference_data_file)
        self.reference_tokens = reference_data["reference_tokens"]
        split_point = self.reference_tokens.shape[-1] // 2
        self.input_prompt = self.reference_tokens[0, :split_point]
        self.gt_tokens = self.reference_tokens[0, split_point:]
        self.top5_tokens = reference_data["top5_tokens"][split_point - 1 :, :]
        self.maxindex = len(self.gt_tokens) - 1

    def collect_predicted_tokens(self, token):
        # Record the prediction, then hand back the ground-truth token to feed next
        # (teacher forcing decouples per-step errors from the scored sequence).
        self.store_predicted_tokens.append(token)
        self.gt_pos += 1
        return self.gt_tokens[min(self.gt_pos, self.maxindex)].reshape(1)

    def compute_accuracy(self):
        count = 0
        count_t5 = 0
        matching_sz = min(len(self.gt_tokens), len(self.store_predicted_tokens))
        for i in range(matching_sz):
            if self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i]:
                count += 1
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_t5 += 1
        return count / matching_sz, count_t5 / matching_sz


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("max_generated_tokens", [256])
@pytest.mark.parametrize("enable_trace", [False, True], ids=["notrace", "trace"])
@pytest.mark.timeout(1200)
def test_demo_text(mesh_device, max_generated_tokens, enable_trace, reset_seeds):
    max_batch_size = 1

    # Load the reference first (host-only) to size the run. The KV cache is then
    # allocated to exactly prefill + scored length: the full context (e.g. 4096)
    # would OOM DRAM on a single device, where the 7B weights already nearly fill it.
    token_acc = TokenAccuracy(model_name=JANUS_REFPT_NAME)
    input_prompt = token_acc.input_prompt
    prefill_len = int(input_prompt.shape[0])
    num_score = min(max_generated_tokens, len(token_acc.gt_tokens))
    # Warmup requires a power-of-two sequence length; round the KV cache up to fit
    # prefill + scored tokens (the full 4096 context would OOM DRAM on a single device).
    max_seq_len = 1 << (prefill_len + num_score - 1).bit_length()

    # Build one model per submesh (data_parallel=1 here); Generator expects lists.
    submeshes = create_submeshes(mesh_device, 1)
    model_args_i, model_i = create_multimodal_model(submeshes[0], max_batch_size, max_seq_len)
    model_args, model = [model_args_i], [model_i]

    generator = JanusMultimodalGenerator(model, model_args, mesh_device)
    tokenizer = model_args[0].tokenizer

    total_len = prefill_len + num_score
    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((max_batch_size, total_len), pad_id, dtype=torch.long)
    tokens[0, :prefill_len] = input_prompt
    prefill_lens = torch.tensor([prefill_len], dtype=torch.long)

    # token-accuracy keeps host logits / host greedy; no on-device sampling.
    # Only prefill is warmed: warmup_model_decode drives the paged-attention path
    # (page_table sized by num_blocks), but this demo decodes non-paged, and num_blocks=0
    # there divides by zero (SIGFPE). The decode loop below runs the non-paged path directly.
    # PERF.md: Top-1/5 from notrace; Speed/TTFT for the Performance table from trace.
    generator.warmup_model_prefill(
        kv_cache=None, enable_trace=enable_trace, can_sample_on_device=False, greedy_only=True
    )
    logger.info("Warmup complete")

    profiler = BenchmarkProfiler()
    profiler.start("run")

    profiler.start("inference_prefill")
    prefill_out = generator.prefill_forward_text(tokens, prompt_lens=prefill_lens, enable_trace=False)
    profiler.end("inference_prefill")

    pred = _greedy(prefill_out)
    next_token = token_acc.collect_predicted_tokens(int(pred[0]))
    tokens[0, prefill_len] = next_token[0]

    profiler.start("inference_decode")
    for gen_idx in range(num_score - 1):
        position_id = prefill_lens + gen_idx
        logits, _ = generator.decode_forward(
            next_token.reshape(max_batch_size, 1), position_id, enable_trace=enable_trace
        )
        pred = _greedy(logits)
        next_token = token_acc.collect_predicted_tokens(int(pred[0]))
        tokens[0, position_id[0] + 1] = next_token[0]
    profiler.end("inference_decode")
    profiler.end("run")

    top1, top5 = token_acc.compute_accuracy()

    prefill_time = profiler.get_duration("inference_prefill")
    decode_time = profiler.get_duration("inference_decode")
    decode_t_s_u = (num_score - 1) / decode_time if decode_time > 0 else 0.0
    logger.info(f"TTFT (prefill): {prefill_time * 1000:.1f} ms")
    logger.info(f"Decode: {decode_t_s_u:.2f} tok/s/user over {num_score - 1} tokens")
    logger.info(f"Janus text accuracy — top1={top1:.4f} top5={top5:.4f} (trace={enable_trace})")
    print(
        f"\n=== Janus-Pro text (trace={enable_trace}) ===\n"
        f"top1={top1 * 100:.2f}%  top5={top5 * 100:.2f}%  (scored {num_score} tokens)\n"
        f"TTFT={prefill_time * 1000:.1f} ms  decode={decode_t_s_u:.2f} tok/s/user\n"
    )
