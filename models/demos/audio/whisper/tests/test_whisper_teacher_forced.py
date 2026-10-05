# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced token agreement with the fp32 Hugging Face reference, on real speech.

Both models see the same tokens at every position: the prompt, then the fp32 reference's own greedy
transcript. At each position the test compares the next-token choice (argmax of the logits) and the
logits themselves, so a precision change shows up per token instead of only in the final text.
The TT side runs the KV-cache path the generator uses: one prefill pass over the prompt, then one
decode step per token.
"""

import glob
import os

import pytest
import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from transformers import WhisperForConditionalGeneration

import ttnn
from models.common.generation_utils import get_logits_processor
from models.demos.audio.whisper.demo.demo import (
    init_conditional_generation_tt_model,
    load_conditional_generation_ref_model,
)
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import get_mesh_mappers

MODEL_NAME = "openai/whisper-large-v3"
# <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>: the generator's prompt for English text
PROMPT = [50258, 50259, 50360, 50364]
EOT = 50257
NUM_CLIPS = 12
MAX_NEW_TOKENS = 224

# Share of positions whose next-token choice may differ from fp32.
MAX_MISMATCH_RATE = 0.05
# A differing choice is only acceptable where fp32 itself nearly tied: its top two logits within this.
MAX_TIE_MARGIN = 1.0
# Floors for the PCC between TT and fp32 logits: averaged over a clip, and at any single position.
# The single-position floor is lower because the end-of-text position is the weakest even for
# plain bfloat16 on CPU.
MIN_CLIP_MEAN_PCC = 0.96
MIN_LOGITS_PCC = 0.85


def load_librispeech_dummy():
    """LibriSpeech's 73-clip validation dummy, as in test_whisper_modules (Arrow cache first, else download)."""
    hf_datasets = os.path.join(os.environ.get("HF_HOME", ""), "datasets")
    arrow_dirs = (
        [
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(hf_datasets, "hf-internal-testing___parquet", "clean-*", "**", "dataset_info.json"),
                recursive=True,
            )
        ]
        if os.path.isdir(hf_datasets)
        else []
    )
    if arrow_dirs:
        return load_from_disk(sorted(arrow_dirs)[-1])
    return load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


def reference_greedy(model, input_features):
    """fp32 greedy transcript with the generator's logits processor -> (tokens, raw logits at each position).

    Row k of the logits predicts token k; the last row predicts what follows the transcript."""
    with torch.no_grad():
        encoder_out = model.model.encoder(input_features).last_hidden_state
        current = torch.tensor([PROMPT])
        processor = get_logits_processor(current, model.config)
        out = model(encoder_outputs=(encoder_out,), decoder_input_ids=current, use_cache=True)
        tokens, rows = [], [out.logits[0, -1]]
        for _ in range(MAX_NEW_TOKENS):
            next_token = int(torch.argmax(processor(current, out.logits[:, -1]), dim=-1))
            if next_token == EOT:
                break
            tokens.append(next_token)
            current = torch.tensor([[next_token]])
            out = model(
                encoder_outputs=(encoder_out,),
                decoder_input_ids=current,
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            rows.append(out.logits[0, -1])
    return tokens, torch.stack(rows).float()


def tt_teacher_forced(config, mesh_device, parameters, lm_head, kv_cache, cross_attn_cache, input_features, tokens):
    """TT logits at the same positions as reference_greedy, feeding the reference's tokens."""
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    input_embeds = ttnn_optimized_functional_whisper.preprocess_encoder_inputs(
        config=config,
        input_features=input_features.unsqueeze(1),
        parameters=parameters.encoder,
        device=mesh_device,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
    )
    encoder_hidden_states = ttnn_optimized_functional_whisper.encoder(
        config, input_embeds, parameters=parameters.encoder
    )
    decode_pos = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
    )

    def decoder_step(input_ids, position, cross_attn_cache_valid):
        hidden_states, attention_mask = ttnn_optimized_functional_whisper.preprocess_decoder_inputs(
            config=config,
            input_ids=input_ids,
            attention_mask=None,
            parameters=parameters.decoder,
            device=mesh_device,
            input_mesh_mapper=input_mesh_mapper,
            decode_pos=position,
            create_attention_mask=False,
        )
        decoder_output = ttnn_optimized_functional_whisper.decoder(
            config,
            hidden_states,
            decoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=cross_attn_cache_valid,
            current_decode_pos=decode_pos,
            parameters=parameters.decoder,
        )
        logits = ttnn.squeeze(decoder_output, 1) @ lm_head
        return ttnn.to_torch(logits, mesh_composer=output_mesh_composer)[0, :, : config.vocab_size].float()

    # Prefill: the whole prompt in one pass, which also fills the cross-attention cache.
    rows = [decoder_step(torch.tensor([PROMPT]), None, cross_attn_cache_valid=False)[len(PROMPT) - 1]]
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.full((1,), len(PROMPT), dtype=torch.int32), dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        ),
        decode_pos,
    )
    # Decode: one step per reference token, at positions len(PROMPT), len(PROMPT) + 1, ...
    for i, token in enumerate(tokens):
        rows.append(decoder_step(torch.tensor([[token]]), len(PROMPT) + i, cross_attn_cache_valid=True)[0])
        ttnn.plus_one(decode_pos)
    return torch.stack(rows)


def logits_pcc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_teacher_forced_tokens_match_fp32(mesh_device):
    reference = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, attn_implementation="eager").eval().float()
    hf_model, config, _, feature_extractor = load_conditional_generation_ref_model(MODEL_NAME, "en", "transcribe")
    _, weights_mesh_mapper, _ = get_mesh_mappers(mesh_device)
    parameters, lm_head, kv_cache, cross_attn_cache = init_conditional_generation_tt_model(
        hf_model, config, mesh_device, weights_mesh_mapper=weights_mesh_mapper
    )
    ds = load_librispeech_dummy()

    positions = mismatches = 0
    worst_pcc = worst_mean_pcc = 1.0
    confident_mismatches = []
    for idx in range(NUM_CLIPS):
        input_features = feature_extractor(
            ds[idx]["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features
        tokens, ref_logits = reference_greedy(reference, input_features)
        tt_logits = tt_teacher_forced(
            config, mesh_device, parameters, lm_head, kv_cache[1], cross_attn_cache[1], input_features, tokens
        )
        ref_choice, tt_choice = ref_logits.argmax(-1), tt_logits.argmax(-1)
        differ = (ref_choice != tt_choice).nonzero().flatten().tolist()
        pccs = [logits_pcc(r, t) for r, t in zip(ref_logits, tt_logits)]
        top2 = ref_logits.topk(2, dim=-1).values
        margins = [round(float(top2[k, 0] - top2[k, 1]), 2) for k in differ]
        mean_pcc = sum(pccs) / len(pccs)
        worst = min(range(len(pccs)), key=lambda k: pccs[k])
        logger.info(
            f"clip {idx}: {len(pccs)} positions, {len(differ)} differ (fp32 top-2 margins there: {margins}), "
            f"logits PCC mean {mean_pcc:.4f}, min {pccs[worst]:.4f} at position {worst} of {len(pccs)}"
        )
        positions += len(pccs)
        mismatches += len(differ)
        worst_pcc = min(worst_pcc, pccs[worst])
        worst_mean_pcc = min(worst_mean_pcc, mean_pcc)
        confident_mismatches += [(idx, k, m) for k, m in zip(differ, margins) if m > MAX_TIE_MARGIN]

    rate = mismatches / positions
    logger.info(
        f"{NUM_CLIPS} clips, {positions} positions: {mismatches} differ ({rate:.2%}); "
        f"logits PCC worst clip mean {worst_mean_pcc:.4f}, worst position {worst_pcc:.4f}"
    )
    assert (
        not confident_mismatches
    ), f"choices differ where fp32 was confident, (clip, position, margin): {confident_mismatches}"
    assert rate <= MAX_MISMATCH_RATE, f"{rate:.2%} of next-token choices differ from fp32 (max {MAX_MISMATCH_RATE:.0%})"
    assert (
        worst_mean_pcc >= MIN_CLIP_MEAN_PCC
    ), f"a clip's mean logits PCC is {worst_mean_pcc:.4f} (min {MIN_CLIP_MEAN_PCC})"
    assert worst_pcc >= MIN_LOGITS_PCC, f"a position's logits PCC fell to {worst_pcc:.4f} (min {MIN_LOGITS_PCC})"
