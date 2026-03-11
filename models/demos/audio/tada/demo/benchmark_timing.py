# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TADA 1B TTS Timing Benchmark: CPU (float32) vs Tenstorrent (TTNN bfloat16).

Runs the same text-to-speech generation on both backends and reports
per-phase and total wall-clock time.

Usage:
    cd tt-metal
    pytest models/demos/audio/tada/demo/benchmark_timing.py -v -s --timeout=1200

Requires:
    - TADA_MODEL_PATH env var (or HumeAI/tada-1b)
    - TADA_CODEC_PATH env var (or HumeAI/tada-codec)
"""

import os
import time
from dataclasses import dataclass, field

import pytest
import torch
from loguru import logger
from scipy.io import wavfile

import ttnn

TADA_L1_SMALL_SIZE = 24576
TADA_SAMPLE_RATE = 24000
TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "HumeAI/tada-1b")
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "HumeAI/tada-codec")
GENERATION_TEXT = "This is a test of text to speech on Tenstorrent hardware."


# ---------------------------------------------------------------------------
# Timing accumulator
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Accumulated wall-clock times for each pipeline phase."""

    model_load_s: float = 0.0
    tokenization_s: float = 0.0
    ar_loop_s: float = 0.0
    ar_embedding_s: float = 0.0
    ar_backbone_s: float = 0.0
    ar_lm_head_s: float = 0.0
    ar_flow_matching_s: float = 0.0
    ar_sampling_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    num_ar_steps: int = 0
    num_fm_steps_per_ar: int = 20
    audio_duration_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: generation_time / audio_duration. <1 means faster than real-time."""
        if self.audio_duration_s > 0:
            return self.total_s / self.audio_duration_s
        return float("inf")

    def summary(self, label: str) -> str:
        lines = [
            f"\n{'=' * 70}",
            f"  TIMING: {label}",
            f"{'=' * 70}",
            f"  Model load:          {self.model_load_s:8.2f}s",
            f"  Tokenization:        {self.tokenization_s:8.2f}s",
            f"  AR loop total:       {self.ar_loop_s:8.2f}s  ({self.num_ar_steps} steps)",
            f"    Embedding:         {self.ar_embedding_s:8.2f}s  ({self.ar_embedding_s / max(self.num_ar_steps, 1) * 1000:.1f}ms/step)",
            f"    Backbone:          {self.ar_backbone_s:8.2f}s  ({self.ar_backbone_s / max(self.num_ar_steps, 1) * 1000:.1f}ms/step)",
            f"    LM head:           {self.ar_lm_head_s:8.2f}s  ({self.ar_lm_head_s / max(self.num_ar_steps, 1) * 1000:.1f}ms/step)",
            f"    Flow matching:     {self.ar_flow_matching_s:8.2f}s  ({self.ar_flow_matching_s / max(self.num_ar_steps, 1) * 1000:.1f}ms/step, {self.num_fm_steps_per_ar} ODE steps)",
            f"    Sampling:          {self.ar_sampling_s:8.2f}s  ({self.ar_sampling_s / max(self.num_ar_steps, 1) * 1000:.1f}ms/step)",
            f"  Waveform decode:     {self.decode_s:8.2f}s",
            f"  ─────────────────────────────────",
            f"  TOTAL (excl. load):  {self.total_s - self.model_load_s:8.2f}s",
            f"  TOTAL (incl. load):  {self.total_s:8.2f}s",
            f"  Audio duration:      {self.audio_duration_s:8.2f}s",
            f"  Real-time factor:    {self.rtf:8.2f}x",
            f"{'=' * 70}",
        ]
        for k, v in self.extra.items():
            lines.insert(-1, f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CPU pipeline (float32)
# ---------------------------------------------------------------------------


def run_cpu_pipeline() -> TimingResult:
    """Run the full TADA pipeline on CPU with float32 and measure timing."""
    import json

    from safetensors.torch import load_file as safetensors_load_file

    timing = TimingResult()
    t_total_start = time.perf_counter()

    # -- Model loading --
    t0 = time.perf_counter()

    from transformers import AutoTokenizer, LlamaConfig, LlamaModel

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    with open(os.path.join(TADA_MODEL_PATH, "config.json")) as f:
        config_dict = json.load(f)

    llama_config = LlamaConfig(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        intermediate_size=config_dict["intermediate_size"],
        max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
        rope_theta=config_dict.get("rope_theta", 500000.0),
    )
    llama_model = LlamaModel(llama_config)
    weights = safetensors_load_file(os.path.join(TADA_MODEL_PATH, "model.safetensors"))
    model_state = {k[6:]: v.float() for k, v in weights.items() if k.startswith("model.")}
    llama_model.load_state_dict(model_state, strict=False)
    llama_model.eval()

    # TADA modules
    HIDDEN_SIZE = 2048
    ACOUSTIC_DIM = 512
    NUM_TIME_BITS = 8
    TIME_DIM = 16
    LATENT_SIZE = 528
    SHIFT_ACOUSTIC = 5
    ACOUSTIC_STD = 1.5
    NUM_EOS_TOKENS = 5

    embed_tokens = llama_model.embed_tokens
    acoustic_proj = torch.nn.Linear(ACOUSTIC_DIM, HIDDEN_SIZE, bias=True)
    acoustic_proj.weight = torch.nn.Parameter(weights["acoustic_proj.weight"].float())
    acoustic_proj.bias = torch.nn.Parameter(weights["acoustic_proj.bias"].float())
    acoustic_mask_emb = torch.nn.Embedding(2, HIDDEN_SIZE)
    acoustic_mask_emb.weight = torch.nn.Parameter(weights["acoustic_mask_emb.weight"].float())
    time_start_embed = torch.nn.Embedding(256, HIDDEN_SIZE)
    time_start_embed.weight = torch.nn.Parameter(weights["time_start_embed.weight"].float())
    time_end_embed = torch.nn.Embedding(256, HIDDEN_SIZE)
    time_end_embed.weight = torch.nn.Parameter(weights["time_end_embed.weight"].float())

    lm_head_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"]).float()
    lm_head = torch.nn.Linear(HIDDEN_SIZE, llama_config.vocab_size, bias=False)
    lm_head.weight = torch.nn.Parameter(lm_head_w)
    lm_head.eval()

    bottleneck_proj = None
    if "bottleneck_proj.weight" in weights:
        bp_w = weights["bottleneck_proj.weight"].float()
        bottleneck_proj = torch.nn.Linear(bp_w.shape[1], bp_w.shape[0], bias=False)
        bottleneck_proj.weight = torch.nn.Parameter(bp_w)
        bottleneck_proj.eval()

    # VibeVoice
    from models.demos.audio.tada.reference.tada_reference import VibeVoiceDiffusionHead

    vv_head = VibeVoiceDiffusionHead(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        head_layers=config_dict.get("prediction_head_num_layers", 6),
        head_ffn_ratio=config_dict.get("prediction_head_ffn_ratio", 4.0),
    )
    vv_state = {k[len("prediction_head.") :]: v.float() for k, v in weights.items() if k.startswith("prediction_head.")}
    vv_head.load_state_dict(vv_state, strict=False)
    vv_head.eval()

    # Decoder
    from pathlib import Path

    from models.demos.audio.tada.reference.tada_reference import Decoder as RefDecoder

    dec_path = os.path.join(TADA_CODEC_PATH, "decoder")
    dec_sd = {}
    for sf in sorted(Path(dec_path).glob("*.safetensors")):
        dec_sd.update(safetensors_load_file(str(sf)))
    dec_sd_clean = {k: v.float() for k, v in dec_sd.items() if "_precomputed_mask" not in k and "rope_freqs" not in k}
    decoder = RefDecoder()
    decoder.load_state_dict(dec_sd_clean, strict=False)
    decoder.eval()

    timing.model_load_s = time.perf_counter() - t0
    logger.info(f"CPU model load: {timing.model_load_s:.2f}s")

    # -- Tokenization --
    t0 = time.perf_counter()

    from models.demos.audio.tada.tt.tada_generator import normalize_text

    text = normalize_text(GENERATION_TEXT)
    prefix = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    eos_token_id = tokenizer.eos_token_id
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    all_tokens = [bos_id] + prefix_tokens + text_tokens + [eot_id] * NUM_EOS_TOKENS
    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    prefix_len = 1 + len(prefix_tokens)

    timing.tokenization_s = time.perf_counter() - t0
    num_steps = input_ids.shape[1]

    # -- AR loop --
    B = 1
    ref_prefix_len = prefix_len - 1
    n_prompt_pad = max(0, ref_prefix_len - SHIFT_ACOUSTIC)

    prompt_acoustic_features = None
    prompt_acoustic_masks = None
    prompt_time_len_before = None
    prompt_time_len_after = None

    if n_prompt_pad > 0:
        prompt_acoustic_features = torch.zeros(B, n_prompt_pad, ACOUSTIC_DIM)
        prompt_acoustic_masks = torch.zeros(B, n_prompt_pad, dtype=torch.long)
        prompt_acoustic_masks[:, -1] = 1
        prompt_time_len_before = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
        prompt_time_len_after = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)

    n_prefix_acoustic = n_prompt_pad + 1

    acoustic_features = torch.zeros(B, ACOUSTIC_DIM)
    acoustic_masks = torch.zeros(B, dtype=torch.long)
    time_len_before = torch.zeros(B, dtype=torch.long)
    time_len_after = torch.zeros(B, dtype=torch.long)
    pos_kv_cache = None
    neg_kv_cache = None
    cache_position = torch.tensor([0], dtype=torch.long)

    use_cfg = True
    acoustic_cfg_scale = 1.6
    duration_cfg_scale = 1.0
    noise_temp = 0.9
    num_fm_steps = 20
    RANDOM_SEED = 42

    all_acoustic_features = []
    all_time_before = []

    # Gray code helpers
    from models.demos.audio.tada.tt.tada_generator import build_time_schedule, decode_gray_code_to_time, scheduled_cfg

    t_ar_start = time.perf_counter()

    for step in range(num_steps):
        input_slice = input_ids[:, step] if step < input_ids.shape[1] else input_ids[:, -1]

        # Embedding
        t_emb = time.perf_counter()
        token_emb = embed_tokens(input_slice.unsqueeze(1))
        ac_emb = acoustic_proj(acoustic_features.unsqueeze(1))
        mask_emb = acoustic_mask_emb(acoustic_masks.unsqueeze(1))
        t_start = time_start_embed(time_len_before.unsqueeze(1))
        t_end = time_end_embed(time_len_after.unsqueeze(1))
        inputs_embeds = token_emb + ac_emb + mask_emb + t_start + t_end
        timing.ar_embedding_s += time.perf_counter() - t_emb

        # Backbone
        t_bb = time.perf_counter()
        with torch.no_grad():
            outputs = llama_model(
                inputs_embeds=inputs_embeds,
                past_key_values=pos_kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )
        pos_kv_cache = outputs.past_key_values
        hidden_cpu = outputs.last_hidden_state

        if use_cfg:
            is_structural = (input_slice == start_header_id) | (input_slice == end_header_id) | (input_slice == eot_id)
            neg_input_slice = torch.where(is_structural, input_slice, torch.full_like(input_slice, pad_token_id))
            neg_embeds = embed_tokens(neg_input_slice.unsqueeze(1)) + ac_emb + mask_emb + t_start + t_end
            with torch.no_grad():
                neg_outputs = llama_model(
                    inputs_embeds=neg_embeds,
                    past_key_values=neg_kv_cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
            neg_kv_cache = neg_outputs.past_key_values
            neg_hidden = neg_outputs.last_hidden_state
        else:
            neg_hidden = hidden_cpu
        timing.ar_backbone_s += time.perf_counter() - t_bb

        # LM head
        t_lm = time.perf_counter()
        with torch.no_grad():
            logits = lm_head(hidden_cpu).squeeze(1)
        timing.ar_lm_head_s += time.perf_counter() - t_lm

        # Sampling
        t_samp = time.perf_counter()
        if step >= input_ids.shape[1] - 1:
            torch.manual_seed(RANDOM_SEED + 10000 + step)
            token_logits = logits.clone()
            token_logits[:, pad_token_id] = float("-inf")
            if 1.1 != 1.0:
                score = torch.gather(token_logits, 1, input_ids)
                score = torch.where(score < 0, score * 1.1, score / 1.1)
                token_logits = token_logits.scatter(1, input_ids, score)
            token_logits = token_logits / 0.6
            sorted_logits, sorted_indices = torch.sort(token_logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            rm = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= 0.9
            rm = rm.scatter(dim=-1, index=sorted_indices, src=rm)
            token_logits = token_logits.masked_fill(rm, float("-inf"))
            next_token = torch.multinomial(torch.softmax(token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() in (eos_token_id, eot_id):
                timing.ar_sampling_s += time.perf_counter() - t_samp
                break
        timing.ar_sampling_s += time.perf_counter() - t_samp

        # Flow matching
        t_fm = time.perf_counter()
        cond = hidden_cpu.squeeze(1)
        neg_cond = neg_hidden.squeeze(1)
        if bottleneck_proj is not None:
            cond_proj = bottleneck_proj(cond)
            neg_cond_proj = bottleneck_proj(neg_cond)
        else:
            cond_proj = cond
            neg_cond_proj = neg_cond

        torch.manual_seed(RANDOM_SEED + step)
        speech = torch.randn(B, LATENT_SIZE) * noise_temp
        t_span = build_time_schedule(num_fm_steps, "logsnr")
        t_curr = t_span[0]

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_curr
            t_val = t_curr.item()
            a_cfg = scheduled_cfg(acoustic_cfg_scale, t_val, "cosine")
            d_cfg = scheduled_cfg(duration_cfg_scale, t_val, "cosine")
            t_torch = t_curr.expand(B)

            if use_cfg:
                speech_doubled = torch.cat([speech, speech], dim=0)
                t_doubled = t_torch.repeat(2)
                cond_combined = torch.cat([cond_proj, neg_cond_proj], dim=0)
                with torch.no_grad():
                    vel = vv_head(speech_doubled, t_doubled, condition=cond_combined)
                vel_pos, vel_neg = vel[:B], vel[B:]
                velocity = torch.cat(
                    [
                        (vel_neg + a_cfg * (vel_pos - vel_neg))[..., :ACOUSTIC_DIM],
                        (vel_neg + d_cfg * (vel_pos - vel_neg))[..., ACOUSTIC_DIM:],
                    ],
                    dim=-1,
                )
            else:
                with torch.no_grad():
                    velocity = vv_head(speech, t_torch, condition=cond_proj)

            speech = speech + dt * velocity
            t_curr = t_span[i]
        timing.ar_flow_matching_s += time.perf_counter() - t_fm

        # Update state
        time_gray = speech[..., -TIME_DIM:]
        predicted_tb = decode_gray_code_to_time(time_gray[..., :NUM_TIME_BITS], NUM_TIME_BITS)
        predicted_ta = decode_gray_code_to_time(time_gray[..., NUM_TIME_BITS:], NUM_TIME_BITS)

        if step >= SHIFT_ACOUSTIC:
            if prompt_acoustic_features is not None and step - SHIFT_ACOUSTIC < prompt_acoustic_features.shape[1]:
                acoustic_features = prompt_acoustic_features[:, step - SHIFT_ACOUSTIC]
                acoustic_masks = prompt_acoustic_masks[:, step - SHIFT_ACOUSTIC]
            else:
                acoustic_features = speech[..., :ACOUSTIC_DIM]
                acoustic_masks = torch.ones(B, dtype=torch.long)
            all_acoustic_features.append(
                acoustic_features.unsqueeze(1) if acoustic_features.dim() == 2 else acoustic_features
            )

            if prompt_time_len_before is not None and step - SHIFT_ACOUSTIC < prompt_time_len_before.shape[1] - 1:
                time_len_before = prompt_time_len_before[:, step - SHIFT_ACOUSTIC + 1]
                time_len_after = prompt_time_len_after[:, step - SHIFT_ACOUSTIC + 1]
            else:
                time_len_before = predicted_tb
                time_len_after = predicted_ta
            all_time_before.append(time_len_before.unsqueeze(1) if time_len_before.dim() == 1 else time_len_before)
        else:
            acoustic_features = torch.zeros(B, ACOUSTIC_DIM)
            acoustic_masks = torch.zeros(B, dtype=torch.long)
            time_len_before = torch.zeros(B, dtype=torch.long)
            time_len_after = torch.zeros(B, dtype=torch.long)

        cache_position = cache_position + 1

    timing.ar_loop_s = time.perf_counter() - t_ar_start
    timing.num_ar_steps = step + 1

    # -- Decode waveform --
    t0 = time.perf_counter()
    if all_acoustic_features:
        pass

        acoustic_cat = torch.cat([f if f.dim() == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1)
        acoustic_cat = acoustic_cat * ACOUSTIC_STD
        time_before_cat = torch.cat([t if t.dim() == 2 else t.unsqueeze(1) for t in all_time_before], dim=1)
        if all_time_before:
            time_before_cat = torch.cat(
                [
                    time_before_cat,
                    all_time_before[-1] if all_time_before[-1].dim() == 2 else all_time_before[-1].unsqueeze(1),
                ],
                dim=1,
            )

        if n_prefix_acoustic > 0 and acoustic_cat.shape[1] > n_prefix_acoustic:
            acoustic_cat = acoustic_cat[:, n_prefix_acoustic:]
            time_before_cat = time_before_cat[:, n_prefix_acoustic:]

        encoded = acoustic_cat[0]
        tb = time_before_cat[0][: encoded.shape[0] + 1]
        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            gap = max(0, int(tb[pos].item()) - 1)
            if gap > 0:
                encoded_expanded.append(torch.zeros(gap, encoded.shape[-1]))
            encoded_expanded.append(encoded[pos].unsqueeze(0))
        trail = int(tb[-1].item())
        if trail > 0:
            encoded_expanded.append(torch.zeros(trail, encoded.shape[-1]))
        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

        with torch.no_grad():
            wav = decoder(encoded_expanded.float(), token_masks)

        timing.audio_duration_s = wav.shape[-1] / TADA_SAMPLE_RATE
    else:
        wav = torch.zeros(1, 1, 0)

    timing.decode_s = time.perf_counter() - t0
    timing.total_s = time.perf_counter() - t_total_start

    # Save audio
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    if wav.numel() > 0:
        audio_np = (wav.squeeze().clamp(-1, 1).float().numpy() * 32767).astype("int16")
        wavfile.write(os.path.join(output_dir, "benchmark_cpu.wav"), TADA_SAMPLE_RATE, audio_np)

    return timing


# ---------------------------------------------------------------------------
# TTNN pipeline
# ---------------------------------------------------------------------------


def run_ttnn_pipeline(mesh_device) -> TimingResult:
    """Run the full TADA pipeline on Tenstorrent and measure timing."""
    import time

    from models.demos.audio.tada.tt.tada_generator import (
        TADA_ACOUSTIC_DIM,
        TADA_ACOUSTIC_STD,
        TADA_NUM_TIME_BITS,
        TADA_SHIFT_ACOUSTIC,
        TADA_TIME_DIM,
        TadaGenerator,
        TadaInferenceOptions,
        decode_gray_code_to_time,
        sample_text_token,
    )
    from models.demos.audio.tada.tt.ttnn_functional_tada import tada_embed_inputs, tada_lm_head
    from models.tt_transformers.tt.common import Mode

    timing = TimingResult()
    t_total_start = time.perf_counter()

    # -- Model loading --
    t0 = time.perf_counter()
    generator = TadaGenerator(
        mesh_device,
        tada_model_id=TADA_MODEL_PATH,
        codec_model_id=TADA_CODEC_PATH,
        max_seq_len=2048,
        max_batch_size=1,
    )
    timing.model_load_s = time.perf_counter() - t0
    logger.info(f"TTNN model load: {timing.model_load_s:.2f}s")

    # -- Tokenization --
    t0 = time.perf_counter()
    opts = TadaInferenceOptions(
        num_flow_matching_steps=20,
        acoustic_cfg_scale=1.6,
        noise_temperature=0.9,
        random_seed=42,
    )
    input_ids, prefix_len = generator._build_input_ids("", GENERATION_TEXT)
    timing.tokenization_s = time.perf_counter() - t0
    num_steps = input_ids.shape[1]

    # -- AR loop (instrumented version of generator.generate) --
    B = 1
    ref_prefix_len = prefix_len - 1
    n_prompt_pad = max(0, ref_prefix_len - TADA_SHIFT_ACOUSTIC)

    prompt_acoustic_features = None
    prompt_acoustic_masks = None
    prompt_time_len_before = None
    prompt_time_len_after = None

    if n_prompt_pad > 0:
        prompt_acoustic_features = torch.zeros(B, n_prompt_pad, TADA_ACOUSTIC_DIM)
        prompt_acoustic_masks = torch.zeros(B, n_prompt_pad, dtype=torch.long)
        prompt_acoustic_masks[:, -1] = 1
        prompt_time_len_before = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
        prompt_time_len_after = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)

    n_prefix_acoustic = n_prompt_pad + 1

    acoustic_features = torch.zeros(B, TADA_ACOUSTIC_DIM)
    acoustic_masks = torch.zeros(B, dtype=torch.long)
    time_len_before = torch.zeros(B, dtype=torch.long)
    time_len_after = torch.zeros(B, dtype=torch.long)

    tada_params = generator._build_tada_parameters_namespace()
    use_cfg = opts.acoustic_cfg_scale != 1.0

    start_header_id = generator.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = generator.tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    all_acoustic_features = []
    all_time_before = []

    t_ar_start = time.perf_counter()

    for step in range(num_steps):
        input_slice = input_ids[:, step] if step < input_ids.shape[1] else input_ids[:, -1]

        # Embedding
        t_emb = time.perf_counter()
        inputs_embeds = tada_embed_inputs(
            input_slice,
            acoustic_features,
            acoustic_masks,
            time_len_before,
            time_len_after,
            parameters=tada_params,
            device=mesh_device,
            input_mesh_mapper=generator.input_mesh_mapper,
        )
        inputs_embeds = ttnn.unsqueeze_to_4D(inputs_embeds)
        timing.ar_embedding_s += time.perf_counter() - t_emb

        # Backbone
        t_bb = time.perf_counter()
        current_pos_tt, rope_idxs = generator._prepare_decode_pos(step)
        hidden_tt = generator._run_llama_step(inputs_embeds, current_pos_tt, rope_idxs, Mode.DECODE)

        if use_cfg:
            is_structural = (
                (input_slice == start_header_id) | (input_slice == end_header_id) | (input_slice == generator.eot_id)
            )
            neg_input_slice = torch.where(
                is_structural, input_slice, torch.full_like(input_slice, generator.pad_token_id)
            )
            neg_embeds = tada_embed_inputs(
                neg_input_slice,
                acoustic_features,
                acoustic_masks,
                time_len_before,
                time_len_after,
                parameters=tada_params,
                device=mesh_device,
                input_mesh_mapper=generator.input_mesh_mapper,
            )
            neg_embeds = ttnn.unsqueeze_to_4D(neg_embeds)
            neg_hidden_tt = generator._run_llama_step(
                neg_embeds, current_pos_tt, rope_idxs, Mode.DECODE, kv_cache=generator.neg_kv_cache
            )

        # Transfer hidden to CPU
        hidden_cpu = ttnn.to_torch(hidden_tt, mesh_composer=generator.output_mesh_composer)
        if hidden_cpu.dim() == 4:
            hidden_cpu = hidden_cpu.squeeze(0).squeeze(0)[:B].unsqueeze(1)
        elif hidden_cpu.dim() == 3:
            hidden_cpu = hidden_cpu[:B]
        ttnn.deallocate(hidden_tt)
        hidden_3d = ttnn.from_torch(
            hidden_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=generator.input_mesh_mapper,
        )

        if use_cfg:
            neg_cpu = ttnn.to_torch(neg_hidden_tt, mesh_composer=generator.output_mesh_composer)
            if neg_cpu.dim() == 4:
                neg_cpu = neg_cpu.squeeze(0).squeeze(0)[:B].unsqueeze(1)
            elif neg_cpu.dim() == 3:
                neg_cpu = neg_cpu[:B]
            ttnn.deallocate(neg_hidden_tt)
            neg_cond_tt = ttnn.from_torch(
                neg_cpu,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=generator.input_mesh_mapper,
            )
        else:
            neg_cond_tt = hidden_3d
        timing.ar_backbone_s += time.perf_counter() - t_bb

        # LM head
        t_lm = time.perf_counter()
        logits_tt = tada_lm_head(hidden_3d, parameters=tada_params)
        logits_cpu = ttnn.to_torch(logits_tt, mesh_composer=generator.output_mesh_composer)
        if logits_cpu.dim() == 3:
            logits_cpu = logits_cpu.squeeze(1)
        ttnn.deallocate(logits_tt)
        timing.ar_lm_head_s += time.perf_counter() - t_lm

        # Sampling
        t_samp = time.perf_counter()
        if step >= input_ids.shape[1] - 1:
            if opts.random_seed is not None:
                torch.manual_seed(opts.random_seed + 10000 + step)
            next_token = sample_text_token(logits_cpu, input_ids, opts, generator.pad_token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() in (generator.eos_token_id, generator.eot_id):
                timing.ar_sampling_s += time.perf_counter() - t_samp
                break
        timing.ar_sampling_s += time.perf_counter() - t_samp

        # Flow matching
        t_fm = time.perf_counter()
        speech = generator._solve_flow_matching(hidden_3d, neg_cond_tt, opts, step_idx=step)
        ttnn.deallocate(hidden_3d)
        if use_cfg:
            ttnn.deallocate(neg_cond_tt)
        timing.ar_flow_matching_s += time.perf_counter() - t_fm

        # Update state
        time_gray = speech[..., -TADA_TIME_DIM:]
        predicted_tb = decode_gray_code_to_time(time_gray[..., :TADA_NUM_TIME_BITS], TADA_NUM_TIME_BITS)
        predicted_ta = decode_gray_code_to_time(time_gray[..., TADA_NUM_TIME_BITS:], TADA_NUM_TIME_BITS)

        if step >= TADA_SHIFT_ACOUSTIC:
            if prompt_acoustic_features is not None and step - TADA_SHIFT_ACOUSTIC < prompt_acoustic_features.shape[1]:
                acoustic_features = prompt_acoustic_features[:, step - TADA_SHIFT_ACOUSTIC]
                acoustic_masks = prompt_acoustic_masks[:, step - TADA_SHIFT_ACOUSTIC]
            else:
                acoustic_features = speech[..., :TADA_ACOUSTIC_DIM]
                acoustic_masks = torch.ones(B, dtype=torch.long)
            all_acoustic_features.append(
                acoustic_features.unsqueeze(1) if acoustic_features.dim() == 2 else acoustic_features
            )

            if prompt_time_len_before is not None and step - TADA_SHIFT_ACOUSTIC < prompt_time_len_before.shape[1] - 1:
                time_len_before = prompt_time_len_before[:, step - TADA_SHIFT_ACOUSTIC + 1]
                time_len_after = prompt_time_len_after[:, step - TADA_SHIFT_ACOUSTIC + 1]
            else:
                time_len_before = predicted_tb
                time_len_after = predicted_ta
            all_time_before.append(time_len_before.unsqueeze(1) if time_len_before.dim() == 1 else time_len_before)
        else:
            acoustic_features = torch.zeros(B, TADA_ACOUSTIC_DIM)
            acoustic_masks = torch.zeros(B, dtype=torch.long)
            time_len_before = torch.zeros(B, dtype=torch.long)
            time_len_after = torch.zeros(B, dtype=torch.long)

    timing.ar_loop_s = time.perf_counter() - t_ar_start
    timing.num_ar_steps = step + 1

    # -- Decode waveform --
    t0 = time.perf_counter()
    if all_acoustic_features:
        pass

        acoustic_cat = torch.cat([f if f.dim() == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1)
        acoustic_cat = acoustic_cat * TADA_ACOUSTIC_STD
        time_before_cat = torch.cat([t if t.dim() == 2 else t.unsqueeze(1) for t in all_time_before], dim=1)
        if all_time_before:
            time_before_cat = torch.cat(
                [
                    time_before_cat,
                    all_time_before[-1] if all_time_before[-1].dim() == 2 else all_time_before[-1].unsqueeze(1),
                ],
                dim=1,
            )

        if n_prefix_acoustic > 0 and acoustic_cat.shape[1] > n_prefix_acoustic:
            acoustic_cat = acoustic_cat[:, n_prefix_acoustic:]
            time_before_cat = time_before_cat[:, n_prefix_acoustic:]

        wav = generator._decode_wav(acoustic_cat[0], time_before_cat[0])
        timing.audio_duration_s = wav.shape[-1] / TADA_SAMPLE_RATE
    else:
        wav = torch.zeros(1, 1, 0)

    timing.decode_s = time.perf_counter() - t0
    timing.total_s = time.perf_counter() - t_total_start

    # Save audio
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    if wav.numel() > 0:
        audio_np = (wav.squeeze().clamp(-1, 1).float().cpu().numpy() * 32767).astype("int16")
        wavfile.write(os.path.join(output_dir, "benchmark_ttnn.wav"), TADA_SAMPLE_RATE, audio_np)

    return timing


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": TADA_L1_SMALL_SIZE}], indirect=True)
def test_benchmark_cpu_vs_ttnn(mesh_device):
    """
    Run TADA TTS on CPU (float32) and Tenstorrent (TTNN bfloat16),
    then compare wall-clock times per phase.
    """
    logger.info("=" * 70)
    logger.info("TADA 1B TTS Timing Benchmark")
    logger.info("=" * 70)

    # Run CPU first (no device needed)
    logger.info("\n--- Running CPU pipeline (float32) ---")
    cpu_timing = run_cpu_pipeline()
    logger.info(cpu_timing.summary("CPU (float32)"))

    # Run TTNN
    logger.info("\n--- Running TTNN pipeline (bfloat16) ---")
    ttnn_timing = run_ttnn_pipeline(mesh_device)
    logger.info(ttnn_timing.summary("TTNN (bfloat16)"))

    # Comparison table
    logger.info(f"\n{'=' * 70}")
    logger.info("  COMPARISON: CPU vs TTNN")
    logger.info(f"{'=' * 70}")
    logger.info(f"  {'Phase':<25s} {'CPU (s)':>10s} {'TTNN (s)':>10s} {'Speedup':>10s}")
    logger.info(f"  {'-' * 55}")

    comparisons = [
        ("Model load", cpu_timing.model_load_s, ttnn_timing.model_load_s),
        ("Tokenization", cpu_timing.tokenization_s, ttnn_timing.tokenization_s),
        ("AR loop (total)", cpu_timing.ar_loop_s, ttnn_timing.ar_loop_s),
        ("  Embedding", cpu_timing.ar_embedding_s, ttnn_timing.ar_embedding_s),
        ("  Backbone", cpu_timing.ar_backbone_s, ttnn_timing.ar_backbone_s),
        ("  LM head", cpu_timing.ar_lm_head_s, ttnn_timing.ar_lm_head_s),
        ("  Flow matching", cpu_timing.ar_flow_matching_s, ttnn_timing.ar_flow_matching_s),
        ("  Sampling", cpu_timing.ar_sampling_s, ttnn_timing.ar_sampling_s),
        ("Waveform decode", cpu_timing.decode_s, ttnn_timing.decode_s),
        (
            "TOTAL (excl. load)",
            cpu_timing.total_s - cpu_timing.model_load_s,
            ttnn_timing.total_s - ttnn_timing.model_load_s,
        ),
        ("TOTAL (incl. load)", cpu_timing.total_s, ttnn_timing.total_s),
    ]

    for name, cpu_t, ttnn_t in comparisons:
        if ttnn_t > 0:
            speedup = cpu_t / ttnn_t
            logger.info(f"  {name:<25s} {cpu_t:>10.2f} {ttnn_t:>10.2f} {speedup:>9.2f}x")
        else:
            logger.info(f"  {name:<25s} {cpu_t:>10.2f} {ttnn_t:>10.2f} {'N/A':>10s}")

    logger.info(
        f"\n  CPU  AR steps: {cpu_timing.num_ar_steps}, Audio: {cpu_timing.audio_duration_s:.2f}s, RTF: {cpu_timing.rtf:.2f}x"
    )
    logger.info(
        f"  TTNN AR steps: {ttnn_timing.num_ar_steps}, Audio: {ttnn_timing.audio_duration_s:.2f}s, RTF: {ttnn_timing.rtf:.2f}x"
    )
    logger.info(f"{'=' * 70}")
