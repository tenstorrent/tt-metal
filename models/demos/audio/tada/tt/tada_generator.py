# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TADA 1B TTS Generator for Tenstorrent hardware.

End-to-end text-to-speech pipeline:
1. Encode prompt audio → acoustic tokens (encoder CNN on host + transformer on TT)
2. Autoregressive generation loop (Llama backbone on TT)
3. Flow matching ODE solver (VibeVoice diffusion head on TT)
4. Decode acoustic features → waveform (decoder transformer on TT + CNN on host)
"""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

import ttnn
from models.demos.audio.tada.tt.ttnn_functional_decoder import decoder_forward
from models.demos.audio.tada.tt.ttnn_functional_encoder import encoder_get_encoder_outputs
from models.demos.audio.tada.tt.ttnn_functional_tada import TADA_MEMORY_CONFIG, tada_embed_inputs, tada_lm_head
from models.demos.audio.tada.tt.ttnn_functional_vibevoice import vibevoice_diffusion_head
from models.demos.utils.common_demo_utils import get_mesh_mappers

# ---------------------------------------------------------------------------
# Inference options
# ---------------------------------------------------------------------------


@dataclass
class TadaInferenceOptions:
    """Controls text generation, acoustic CFG, diffusion, and synthesis."""

    text_do_sample: bool = True
    text_temperature: float = 0.6
    text_top_k: int = 0
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.1
    acoustic_cfg_scale: float = 1.6
    duration_cfg_scale: float = 1.0
    cfg_schedule: Literal["constant", "linear", "cosine"] = "cosine"
    noise_temperature: float = 0.9
    num_flow_matching_steps: int = 20
    time_schedule: Literal["uniform", "cosine", "logsnr"] = "logsnr"
    random_seed: int | None = 42


# ---------------------------------------------------------------------------
# TADA 1B constants
# ---------------------------------------------------------------------------

TADA_HIDDEN_SIZE = 2048
TADA_ACOUSTIC_DIM = 512
TADA_NUM_TIME_BITS = 8
TADA_TIME_DIM = 2 * TADA_NUM_TIME_BITS  # 16
TADA_LATENT_SIZE = TADA_ACOUSTIC_DIM + TADA_TIME_DIM  # 528
TADA_SHIFT_ACOUSTIC = 5
TADA_NUM_TIME_CLASSES = 256
TADA_VOCAB_SIZE = 128256
TADA_NUM_EOS_TOKENS = TADA_SHIFT_ACOUSTIC  # 5
TADA_ACOUSTIC_MEAN = 0.0
TADA_ACOUSTIC_STD = 1.5


# ---------------------------------------------------------------------------
# Gray code utilities (from tada/utils/gray_code.py)
# ---------------------------------------------------------------------------


def _gray_code_to_int(gray: torch.Tensor) -> torch.Tensor:
    binary = gray
    shift = 1
    while shift < 32:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


def decode_gray_code_to_time(gray_bits: torch.Tensor, num_bits: int) -> torch.LongTensor:
    """Convert Gray code bit representation (values in {-1, 1}) back to time values."""
    gray_bits_binary = ((gray_bits + 1.0) / 2.0).round().long()
    gray_code = torch.zeros(*gray_bits_binary.shape[:-1], dtype=torch.long, device=gray_bits.device)
    for i in range(num_bits):
        gray_code += gray_bits_binary[..., num_bits - 1 - i] << i
    return _gray_code_to_int(gray_code)


# ---------------------------------------------------------------------------
# Text normalization (from tada/utils/text.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Replace common Unicode punctuation with ASCII equivalents and normalize."""
    substitutions = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2026": "...",
        "\u2039": "<",
        "\u203a": ">",
        "\u00ab": "<<",
        "\u00bb": ">>",
    }
    pattern = re.compile("|".join(re.escape(char) for char in substitutions))
    text = pattern.sub(lambda m: substitutions[m.group(0)], text)
    text = (
        text.replace("; ", ". ")
        .replace('"', "")
        .replace(":", ",")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "-")
        .replace("-", ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ")
    )
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(r"([.!?]\s*)(\w)", lambda m: m.group(1) + m.group(2).upper(), text.lower())
    if text:
        text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Time & CFG schedules
# ---------------------------------------------------------------------------


def build_time_schedule(num_steps: int, schedule: str) -> torch.Tensor:
    """Build ODE time schedule on CPU. Returns (num_steps+1,) tensor in [0, 1]."""
    if schedule == "cosine":
        u = torch.linspace(0, 1, num_steps + 1)
        return 0.5 * (1 - torch.cos(math.pi * u))
    if schedule == "logsnr":
        log_snr = torch.linspace(5.0, -5.0, num_steps + 1)
        t_span = torch.sigmoid(-log_snr / 2)
        t_span[0] = 0.0
        t_span[-1] = 1.0
        return t_span
    return torch.linspace(0, 1, num_steps + 1)


def scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
    """Compute effective CFG scale at timestep t."""
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    if schedule == "linear":
        return 1.0 + (base_scale - 1.0) * (1.0 - t)
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_scale


# ---------------------------------------------------------------------------
# Text token sampling (CPU)
# ---------------------------------------------------------------------------


def sample_text_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    opts: TadaInferenceOptions,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Sample next text token from logits on CPU.

    Args:
        logits: (B, vocab_size) raw logits
        input_ids: (B, seq_so_far) all token IDs generated so far
        opts: inference options
        pad_token_id: ID to suppress
    Returns:
        (B, 1) sampled token IDs
    """
    token_logits = logits.clone()
    token_logits[:, pad_token_id] = float("-inf")

    if not opts.text_do_sample:
        return token_logits.argmax(dim=-1, keepdim=True)

    # Repetition penalty
    if opts.text_repetition_penalty != 1.0:
        score = torch.gather(token_logits, 1, input_ids)
        score = torch.where(
            score < 0,
            score * opts.text_repetition_penalty,
            score / opts.text_repetition_penalty,
        )
        token_logits = token_logits.scatter(1, input_ids, score)

    # Temperature
    token_logits = token_logits / opts.text_temperature

    # Top-k
    if opts.text_top_k > 0:
        top_k = min(opts.text_top_k, token_logits.size(-1))
        indices_to_remove = token_logits < torch.topk(token_logits, top_k, dim=-1).values[..., -1:]
        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus)
    if 0.0 < opts.text_top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(token_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= opts.text_top_p
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _resolve_model_path(model_id_or_path: str, subfolder: str | None = None) -> str:
    """Resolve a model ID or local path. Downloads from HF if not a local directory."""
    path = model_id_or_path
    if subfolder:
        candidate = os.path.join(path, subfolder)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(path):
        return path
    # Fall back to HF download
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": model_id_or_path}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*"]
    path = snapshot_download(**kwargs)
    if subfolder:
        path = os.path.join(path, subfolder)
    return path


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load all safetensors files from a directory."""
    state_dict = {}
    p = Path(path)
    for sf_file in sorted(p.glob("*.safetensors")):
        state_dict.update(safetensors_load_file(str(sf_file)))
    return state_dict


def _remap_tada_to_meta_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remap HuggingFace TADA key names to Meta/tt_transformers key format for the Llama backbone.

    HF format: model.layers.{i}.self_attn.q_proj.weight
    Meta format: layers.{i}.attention.wq.weight
    """
    remap = {
        "self_attn.q_proj": "attention.wq",
        "self_attn.k_proj": "attention.wk",
        "self_attn.v_proj": "attention.wv",
        "self_attn.o_proj": "attention.wo",
        "mlp.gate_proj": "feed_forward.w1",
        "mlp.up_proj": "feed_forward.w3",
        "mlp.down_proj": "feed_forward.w2",
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm",
    }

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model.layers."):
            # model.layers.{i}.xxx -> layers.{i}.xxx
            new_key = key.replace("model.layers.", "layers.", 1)
            for hf_suffix, meta_suffix in remap.items():
                new_key = new_key.replace(hf_suffix, meta_suffix)
        elif key == "model.norm.weight":
            new_key = "norm.weight"
        elif key == "model.embed_tokens.weight":
            new_key = "tok_embeddings.weight"
            # tie_word_embeddings=true: also create output.weight alias
            if "lm_head.weight" not in state_dict:
                remapped["output.weight"] = value
        elif key == "lm_head.weight":
            new_key = "output.weight"
        else:
            # TADA-specific keys (acoustic_proj, time embeds, prediction_head, etc.)
            # Keep as-is - these are loaded separately
            pass
        remapped[new_key] = value
    return remapped


def _extract_tada_specific_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract TADA-specific weights (non-Llama) from the full state dict."""
    tada_prefixes = [
        "acoustic_proj.",
        "acoustic_mask_emb.",
        "time_start_embed.",
        "time_end_embed.",
        "prediction_head.",
        "bottleneck_proj.",
    ]
    tada_weights = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in tada_prefixes):
            tada_weights[key] = value
    return tada_weights


def _extract_llama_weights(remapped_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract only the Llama backbone weights from the remapped state dict."""
    llama_prefixes = ["layers.", "norm.", "tok_embeddings.", "output."]
    llama_weights = {}
    for key, value in remapped_state_dict.items():
        if any(key.startswith(prefix) for prefix in llama_prefixes):
            llama_weights[key] = value
    return llama_weights


# ---------------------------------------------------------------------------
# TADA Generator
# ---------------------------------------------------------------------------


class TadaGenerator:
    """
    End-to-end TADA 1B TTS generator for Tenstorrent hardware.

    Usage:
        generator = TadaGenerator(mesh_device, "HumeAI/tada-1b", "HumeAI/tada-codec")
        result = generator.generate(
            prompt_audio=audio_tensor,       # (1, 1, T) at 24kHz
            prompt_text="Hello world",
            generation_text="This is a test.",
        )
        waveform = result["audio"]  # (1, 1, T_out)
    """

    def __init__(
        self,
        mesh_device,
        tada_model_id: str = "HumeAI/tada-1b",
        codec_model_id: str = "HumeAI/tada-codec",
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
    ):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
        self.input_mesh_mapper = input_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer

        logger.info("Loading TADA weights...")
        self._load_all_weights(tada_model_id, codec_model_id)
        logger.info("TADA generator ready.")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_all_weights(self, tada_model_id: str, codec_model_id: str):
        """Load all model weights: Llama backbone, TADA heads, encoder, decoder."""
        # 1. Load TADA safetensors (contains Llama + TADA-specific weights)
        tada_path = _resolve_model_path(tada_model_id)
        tada_state_dict = _load_safetensors(tada_path)

        # 2. Separate Llama and TADA-specific weights
        tada_specific = _extract_tada_specific_weights(tada_state_dict)
        # Also pass embed_tokens weight for TADA embeddings (not in tada_specific prefixes)
        tada_specific["model.embed_tokens.weight"] = tada_state_dict["model.embed_tokens.weight"]
        remapped = _remap_tada_to_meta_keys(tada_state_dict)
        llama_weights = _extract_llama_weights(remapped)

        # 3. Load Llama backbone via tt_transformers
        self._load_llama_backbone(llama_weights, tada_path)

        # 4. Preprocess TADA-specific weights for TTNN
        self._load_tada_weights(tada_specific)

        # 5. Load tokenizer
        self._load_tokenizer(tada_path)

        # 6. Load encoder (CNN on host + transformer on TT)
        encoder_path = _resolve_model_path(codec_model_id, subfolder="encoder")
        self._load_encoder(encoder_path)

        # 7. Load decoder (transformer on TT + CNN on host)
        decoder_path = _resolve_model_path(codec_model_id, subfolder="decoder")
        self._load_decoder(decoder_path)

    def _load_llama_backbone(self, llama_weights: dict, model_path: str):
        """Initialize Llama 3.2 1B backbone on TT device."""
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        # ModelArgs requires HF_MODEL env var — set it to our local TADA path
        old_hf_model = os.environ.get("HF_MODEL")
        os.environ["HF_MODEL"] = model_path
        try:
            # Monkey-patch to skip tokenizer/processor (we load our own)
            orig_create_tokenizer = ModelArgs.create_tokenizer
            orig_create_processor = ModelArgs.create_processor
            ModelArgs.create_tokenizer = lambda self: None
            ModelArgs.create_processor = lambda self: None
            try:
                tt_model_args = ModelArgs(
                    self.mesh_device,
                    instruct=False,
                    max_batch_size=self.max_batch_size,
                    optimizations=None,
                    max_seq_len=self.max_seq_len,
                    use_hf_rope=True,
                )
            finally:
                ModelArgs.create_tokenizer = orig_create_tokenizer
                ModelArgs.create_processor = orig_create_processor
        finally:
            if old_hf_model is not None:
                os.environ["HF_MODEL"] = old_hf_model
            else:
                del os.environ["HF_MODEL"]

        self.tt_model_args = tt_model_args

        # Build model with our pre-extracted weights
        # Use bfloat16 for better precision (bfloat8_b causes noticeable quality loss in TTS)
        self.llama_model = Transformer(
            args=tt_model_args,
            dtype=ttnn.bfloat16,
            mesh_device=self.mesh_device,
            state_dict=llama_weights,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat16),
        )

        # Store KV cache handles
        self.kv_cache = [layer.attention.layer_past for layer in self.llama_model.layers]

        # Allocate a separate KV cache for negative conditioning (CFG)
        self.neg_kv_cache = self._allocate_kv_cache()

    def _allocate_kv_cache(self):
        """Allocate a fresh KV cache (same shape as the primary one) for negative conditioning."""
        neg_kv_cache = []
        for layer in self.llama_model.layers:
            attn = layer.attention
            cache_k = torch.zeros(
                attn.batch_size_per_device_group,
                attn.n_local_kv_heads,
                attn.max_seq_len,
                attn.head_dim,
            )
            cache_v = torch.zeros_like(cache_k)
            kv = [
                ttnn.as_tensor(
                    t,
                    dtype=attn.kv_cache_dtype,
                    layout=self.tt_model_args.get_attn_weights_layout(),
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for t in [cache_k, cache_v]
            ]
            neg_kv_cache.append(kv)
        return neg_kv_cache

    def _load_tada_weights(self, tada_specific: dict):
        """Preprocess TADA-specific weights (embeddings, acoustic_proj, VibeVoice) to TTNN."""

        device = self.mesh_device
        wm = self.weights_mesh_mapper

        # acoustic_proj: Linear(512, 2048, bias=True)
        self.tt_acoustic_proj_weight = ttnn.from_torch(
            tada_specific["acoustic_proj.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )
        if "acoustic_proj.bias" in tada_specific:
            self.tt_acoustic_proj_bias = ttnn.from_torch(
                tada_specific["acoustic_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=wm,
            )
        else:
            self.tt_acoustic_proj_bias = None

        # Embedding tables - ROW_MAJOR for ttnn.embedding
        embed_tokens_w = tada_specific.get("model.embed_tokens.weight")
        if embed_tokens_w is None:
            embed_tokens_w = self._get_embed_tokens_weight()
        self.tt_embed_tokens_weight = ttnn.from_torch(
            embed_tokens_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )

        self.tt_acoustic_mask_emb_weight = ttnn.from_torch(
            tada_specific["acoustic_mask_emb.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )

        self.tt_time_start_embed_weight = ttnn.from_torch(
            tada_specific["time_start_embed.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )

        self.tt_time_end_embed_weight = ttnn.from_torch(
            tada_specific["time_end_embed.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )

        # LM head weight (tied with embed_tokens in original, but we load separately)
        lm_head_w = tada_specific.get("lm_head.weight", None)
        if lm_head_w is None:
            # Tied weights - lm_head uses embed_tokens weight
            lm_head_w = embed_tokens_w
        self.tt_lm_head_weight = ttnn.from_torch(
            lm_head_w.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=wm,
        )

        # VibeVoice prediction head
        self._load_vibevoice_weights(tada_specific)

    def _get_embed_tokens_weight(self) -> torch.Tensor:
        """Get embed_tokens weight from the Llama model's embedding layer."""
        # The embedding weights are stored in the Llama model
        emb_weight = ttnn.to_torch(self.llama_model.embd.emb.weight, mesh_composer=self.output_mesh_composer)
        return emb_weight

    def _load_vibevoice_weights(self, tada_specific: dict):
        """Load and preprocess VibeVoice diffusion head weights."""
        from ttnn.model_preprocessing import preprocess_model_parameters

        from models.demos.audio.tada.reference.tada_reference import VibeVoiceDiffusionHead
        from models.demos.audio.tada.tt.ttnn_functional_vibevoice import convert_to_ttnn as vv_convert
        from models.demos.audio.tada.tt.ttnn_functional_vibevoice import (
            create_custom_mesh_preprocessor as vv_preprocessor,
        )

        # Reconstruct the reference VibeVoice model and load weights
        ref_vv = VibeVoiceDiffusionHead(
            hidden_size=TADA_HIDDEN_SIZE,
            head_layers=6,
            head_ffn_ratio=4.0,
            rms_norm_eps=1e-5,
            latent_size=TADA_LATENT_SIZE,
        )
        # Load prediction_head.* weights
        vv_state_dict = {}
        for key, value in tada_specific.items():
            if key.startswith("prediction_head."):
                vv_key = key.replace("prediction_head.", "")
                vv_state_dict[vv_key] = value
        ref_vv.load_state_dict(vv_state_dict, strict=False)
        ref_vv.eval()

        self.vibevoice_parameters = preprocess_model_parameters(
            initialize_model=lambda: ref_vv,
            convert_to_ttnn=vv_convert,
            custom_preprocessor=vv_preprocessor(self.weights_mesh_mapper),
            device=self.mesh_device,
        )

        # bottleneck_proj (Identity for 1B model, but load if present)
        if "bottleneck_proj.weight" in tada_specific:
            self.tt_bottleneck_proj_weight = ttnn.from_torch(
                tada_specific["bottleneck_proj.weight"].T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=self.weights_mesh_mapper,
            )
            self.has_bottleneck_proj = True
        else:
            self.has_bottleneck_proj = False

    def _load_tokenizer(self, model_path: str):
        """Load the Llama tokenizer. TADA uses the Llama 3.2 tokenizer."""
        from transformers import AutoTokenizer

        # TADA uses the Llama 3.2 tokenizer (not bundled in TADA weights)
        tokenizer_id = "meta-llama/Llama-3.2-1B-Instruct"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
        except Exception:
            logger.info(f"Downloading tokenizer from {tokenizer_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def _load_encoder(self, encoder_path: str):
        """Load encoder: WavEncoder CNN (host) + LocalAttentionEncoder (TT)."""
        from ttnn.model_preprocessing import preprocess_model_parameters

        from models.demos.audio.tada.reference.tada_reference import LocalAttentionEncoder as RefLocalAttentionEncoder
        from models.demos.audio.tada.reference.tada_reference import WavEncoder as RefWavEncoder
        from models.demos.audio.tada.tt.ttnn_functional_encoder import convert_to_ttnn as enc_convert
        from models.demos.audio.tada.tt.ttnn_functional_encoder import (
            create_custom_mesh_preprocessor as enc_preprocessor,
        )

        # Load encoder state dict
        enc_state_dict = _load_safetensors(encoder_path)

        # WavEncoder CNN (runs on host)
        self.wav_encoder = RefWavEncoder(
            d_model=64,
            strides=[6, 5, 4, 4],
            d_latent=1024,
        )
        wav_enc_weights = {
            k.replace("wav_encoder.", ""): v for k, v in enc_state_dict.items() if k.startswith("wav_encoder.")
        }
        self.wav_encoder.load_state_dict(wav_enc_weights, strict=False)
        self.wav_encoder.eval()

        # Position embedding (kept on CPU)
        self.encoder_pos_emb_weight = enc_state_dict.get("pos_emb.weight", torch.zeros(2, 1024))

        # hidden_linear projection (1024 -> 512)
        hidden_linear_w = enc_state_dict.get("hidden_linear.weight", None)
        hidden_linear_b = enc_state_dict.get("hidden_linear.bias", None)
        if hidden_linear_w is not None:
            self.encoder_hidden_linear_weight = ttnn.from_torch(
                hidden_linear_w.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=self.weights_mesh_mapper,
            )
            self.encoder_hidden_linear_bias = (
                ttnn.from_torch(
                    hidden_linear_b.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=self.weights_mesh_mapper,
                )
                if hidden_linear_b is not None
                else None
            )
        else:
            self.encoder_hidden_linear_weight = None
            self.encoder_hidden_linear_bias = None

        # LocalAttentionEncoder transformer (runs on TT)
        ref_local_attn = RefLocalAttentionEncoder(
            d_model=1024,
            num_heads=8,
            d_ff=4096,
            num_layers=6,
        )
        local_attn_weights = {
            k.replace("local_attention_encoder.", ""): v
            for k, v in enc_state_dict.items()
            if k.startswith("local_attention_encoder.")
        }
        # Filter out non-parameter keys (precomputed mask, rope freqs)
        local_attn_weights = {
            k: v for k, v in local_attn_weights.items() if "_precomputed_mask" not in k and "rope_freqs" not in k
        }
        ref_local_attn.load_state_dict(local_attn_weights, strict=False)
        ref_local_attn.eval()

        self.encoder_local_attn_params = preprocess_model_parameters(
            initialize_model=lambda: ref_local_attn,
            convert_to_ttnn=enc_convert,
            custom_preprocessor=enc_preprocessor(self.weights_mesh_mapper),
            device=self.mesh_device,
        )

    def _load_decoder(self, decoder_path: str):
        """Load decoder: LocalAttentionEncoder (TT) + DACDecoder CNN (host)."""
        from ttnn.model_preprocessing import preprocess_model_parameters

        from models.demos.audio.tada.reference.tada_reference import DACDecoder as RefDACDecoder
        from models.demos.audio.tada.reference.tada_reference import LocalAttentionEncoder as RefLocalAttentionEncoder
        from models.demos.audio.tada.tt.ttnn_functional_decoder import convert_to_ttnn as dec_convert
        from models.demos.audio.tada.tt.ttnn_functional_decoder import (
            create_custom_mesh_preprocessor as dec_preprocessor,
        )

        dec_state_dict = _load_safetensors(decoder_path)

        # decoder_proj (Linear 512 -> 1024)
        dec_proj_w = dec_state_dict.get("decoder_proj.weight", None)
        dec_proj_b = dec_state_dict.get("decoder_proj.bias", None)
        if dec_proj_w is not None:
            self.decoder_proj_weight = ttnn.from_torch(
                dec_proj_w.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=self.weights_mesh_mapper,
            )
            self.decoder_proj_bias = (
                ttnn.from_torch(
                    dec_proj_b.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=self.weights_mesh_mapper,
                )
                if dec_proj_b is not None
                else None
            )
        else:
            self.decoder_proj_weight = None
            self.decoder_proj_bias = None

        # LocalAttentionEncoder for decoder (on TT)
        ref_dec_local_attn = RefLocalAttentionEncoder(
            d_model=1024,
            num_heads=8,
            d_ff=4096,
            num_layers=6,
        )
        # Decoder uses "local_attention_decoder" prefix in the codec weights
        dec_prefix = "local_attention_decoder."
        if not any(k.startswith(dec_prefix) for k in dec_state_dict):
            dec_prefix = "local_attention_encoder."
        dec_local_weights = {
            k.replace(dec_prefix, ""): v for k, v in dec_state_dict.items() if k.startswith(dec_prefix)
        }
        # Filter out non-parameter keys (precomputed mask, rope freqs)
        dec_local_weights = {
            k: v for k, v in dec_local_weights.items() if "_precomputed_mask" not in k and "rope_freqs" not in k
        }
        ref_dec_local_attn.load_state_dict(dec_local_weights, strict=False)
        ref_dec_local_attn.eval()

        # Keep reference decoder for comparison/debugging (ensure float32)
        self.ref_decoder_proj = torch.nn.Linear(512, 1024)
        if dec_proj_w is not None:
            self.ref_decoder_proj.weight = torch.nn.Parameter(dec_proj_w.float())
            if dec_proj_b is not None:
                self.ref_decoder_proj.bias = torch.nn.Parameter(dec_proj_b.float())
        self.ref_decoder_proj.eval()
        self.ref_dec_local_attn = ref_dec_local_attn.float()
        self.ref_dec_local_attn.eval()

        self.decoder_local_attn_params = preprocess_model_parameters(
            initialize_model=lambda: ref_dec_local_attn,
            convert_to_ttnn=dec_convert,
            custom_preprocessor=dec_preprocessor(self.weights_mesh_mapper),
            device=self.mesh_device,
        )

        # DACDecoder CNN (on host)
        self.wav_decoder = RefDACDecoder(
            input_channel=1024,
            channels=1536,
            rates=[4, 4, 5, 6],
        )
        dac_weights = {
            k.replace("wav_decoder.", ""): v for k, v in dec_state_dict.items() if k.startswith("wav_decoder.")
        }
        self.wav_decoder.load_state_dict(dac_weights, strict=False)
        self.wav_decoder.eval()

    # ------------------------------------------------------------------
    # Llama backbone step
    # ------------------------------------------------------------------

    def _build_tada_parameters_namespace(self):
        """Build a namespace object that mimics the parameter structure expected by tada_embed_inputs."""

        class _Ns:
            pass

        params = _Ns()
        params.model = _Ns()
        params.model.embed_tokens = _Ns()
        params.model.embed_tokens.weight = self.tt_embed_tokens_weight
        params.acoustic_proj = _Ns()
        params.acoustic_proj.weight = self.tt_acoustic_proj_weight
        params.acoustic_proj.bias = self.tt_acoustic_proj_bias
        params.acoustic_mask_emb = _Ns()
        params.acoustic_mask_emb.weight = self.tt_acoustic_mask_emb_weight
        params.time_start_embed = _Ns()
        params.time_start_embed.weight = self.tt_time_start_embed_weight
        params.time_end_embed = _Ns()
        params.time_end_embed.weight = self.tt_time_end_embed_weight
        params.lm_head = _Ns()
        params.lm_head.weight = self.tt_lm_head_weight
        return params

    def _prepare_decode_pos(self, step: int):
        """Prepare current_pos and rope_idxs as TTNN tensors for decode step."""
        current_pos_torch = torch.tensor([step], dtype=torch.int64)
        # Pad to batch size
        current_pos_torch = torch.nn.functional.pad(current_pos_torch, (0, self.max_batch_size - 1), value=0)

        rope_idxs = self.llama_model.rope_setup.get_rot_idxs(
            torch.maximum(current_pos_torch, torch.tensor(0, dtype=torch.int64)),
            on_host=True,
        )

        current_pos_tt = ttnn.from_torch(
            current_pos_torch,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, None),
                mesh_shape=self.tt_model_args.cluster_shape,
            ),
        )
        return current_pos_tt, rope_idxs

    def _run_llama_step(self, x_tt, current_pos_tt, rope_idxs, mode, kv_cache=None):
        """
        Run one Llama decode step, returning normed hidden states (before lm_head).

        Mirrors Transformer.forward() lines 682-723 but stops before lm_head.

        Args:
            kv_cache: optional list of [key, value] per layer. If None, uses self.kv_cache.
        """
        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.model_config import TensorGroup

        if kv_cache is None:
            kv_cache = self.kv_cache

        # Move current_pos to device
        current_pos_device = ttnn.to_device(current_pos_tt, self.mesh_device)
        rope_idxs_device = ttnn.to_device(rope_idxs, self.mesh_device)

        rot_mats_global = self.llama_model.rope_setup.get_rot_mats(rope_idxs_device)
        rot_mats_local = (
            self.llama_model.rope_local_setup.get_rot_mats(rope_idxs_device)
            if hasattr(self.llama_model, "rope_local_setup")
            else None
        )

        for i, layer in enumerate(self.llama_model.layers):
            # Match memory config handling from Transformer.forward()
            activation_dtype = self.tt_model_args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i,
                tensor=TensorGroup.ACTIVATION,
            )
            if mode == Mode.DECODE and not self.tt_model_args.is_galaxy:
                x_tt = ttnn.to_memory_config(
                    x_tt,
                    self.tt_model_args.get_residual_mem_config(mode, None),
                    activation_dtype,
                )
            elif activation_dtype is not None and x_tt.dtype != activation_dtype:
                x_tt = ttnn.typecast(x_tt, activation_dtype)

            x_tt = layer(
                x_tt,
                current_pos_device,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                mode=mode,
                kv_cache=kv_cache[i],
                batch_size=1,
            )

        # Apply final norm (without lm_head)
        x_tt = self.llama_model.norm(
            x_tt,
            mode=mode,
            norm_config=self.tt_model_args.get_norm_config("lm_head", mode, None),
        )
        return x_tt

    # ------------------------------------------------------------------
    # Flow matching ODE solver
    # ------------------------------------------------------------------

    def _solve_flow_matching(
        self,
        cond_tt,
        neg_cond_tt,
        opts: TadaInferenceOptions,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """
        Solve flow matching ODE with Euler method.

        Args:
            cond_tt: (B, 1, hidden_size) conditioning on TT device
            neg_cond_tt: (B, 1, hidden_size) negative conditioning on TT device (or zeros)
            opts: inference options
            step_idx: AR step index, used for per-step random seed
        Returns:
            speech: (B, TADA_LATENT_SIZE) final speech features on CPU
        """
        B = 1  # batch size
        use_cfg = opts.acoustic_cfg_scale != 1.0

        # Seed noise for reproducibility
        if opts.random_seed is not None:
            torch.manual_seed(opts.random_seed + step_idx)

        # Initialize noise on CPU
        speech = torch.randn(B, TADA_LATENT_SIZE) * opts.noise_temperature
        t_span = build_time_schedule(opts.num_flow_matching_steps, opts.time_schedule)
        t_curr = t_span[0]

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_curr
            t_val = t_curr.item()
            a_cfg = scheduled_cfg(opts.acoustic_cfg_scale, t_val, opts.cfg_schedule)
            d_cfg = scheduled_cfg(opts.duration_cfg_scale, t_val, opts.cfg_schedule)
            t_torch = t_curr.expand(B)

            if use_cfg:
                # CFG: double batch [speech, speech] with [cond, neg_cond]
                speech_doubled = speech.repeat(2, 1).unsqueeze(1)  # (2B, 1, 528)
                speech_tt = ttnn.from_torch(
                    speech_doubled,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=self.input_mesh_mapper,
                )

                # Apply bottleneck projection if needed
                if self.has_bottleneck_proj:
                    cond_proj = ttnn.linear(cond_tt, self.tt_bottleneck_proj_weight, memory_config=TADA_MEMORY_CONFIG)
                    neg_cond_proj = ttnn.linear(
                        neg_cond_tt, self.tt_bottleneck_proj_weight, memory_config=TADA_MEMORY_CONFIG
                    )
                else:
                    cond_proj = cond_tt
                    neg_cond_proj = neg_cond_tt

                # Concatenate conditions: [cond, neg_cond]
                cond_combined = ttnn.concat([cond_proj, neg_cond_proj], dim=0, memory_config=TADA_MEMORY_CONFIG)

                t_doubled = t_torch.repeat(2)

                velocity_tt = vibevoice_diffusion_head(
                    speech_tt,
                    t_doubled,
                    cond_combined,
                    parameters=self.vibevoice_parameters,
                )
                velocity_cpu = ttnn.to_torch(velocity_tt, mesh_composer=self.output_mesh_composer)
                if velocity_cpu.dim() == 3:
                    velocity_cpu = velocity_cpu.squeeze(1)
                ttnn.deallocate(velocity_tt)
                ttnn.deallocate(speech_tt)
                if self.has_bottleneck_proj:
                    ttnn.deallocate(cond_proj)
                    ttnn.deallocate(neg_cond_proj)
                ttnn.deallocate(cond_combined)

                # Split pos/neg and apply CFG
                vel_pos = velocity_cpu[:B]
                vel_neg = velocity_cpu[B:]
                velocity = torch.cat(
                    [
                        (vel_neg + a_cfg * (vel_pos - vel_neg))[..., :TADA_ACOUSTIC_DIM],
                        (vel_neg + d_cfg * (vel_pos - vel_neg))[..., TADA_ACOUSTIC_DIM:],
                    ],
                    dim=-1,
                )
            else:
                speech_tt = ttnn.from_torch(
                    speech.unsqueeze(1),  # (B, 1, 528)
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=self.input_mesh_mapper,
                )

                if self.has_bottleneck_proj:
                    cond_proj = ttnn.linear(cond_tt, self.tt_bottleneck_proj_weight, memory_config=TADA_MEMORY_CONFIG)
                else:
                    cond_proj = cond_tt

                velocity_tt = vibevoice_diffusion_head(
                    speech_tt,
                    t_torch,
                    cond_proj,
                    parameters=self.vibevoice_parameters,
                )
                velocity = ttnn.to_torch(velocity_tt, mesh_composer=self.output_mesh_composer)
                if velocity.dim() == 3:
                    velocity = velocity.squeeze(1)
                ttnn.deallocate(velocity_tt)
                ttnn.deallocate(speech_tt)
                if self.has_bottleneck_proj:
                    ttnn.deallocate(cond_proj)

            speech = speech + dt * velocity.float()
            t_curr = t_span[i]

        return speech

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    def _add_bos_eos(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Add BOS token at start and TADA_NUM_EOS_TOKENS EOT tokens at end."""
        eos_tokens = torch.full((input_ids.shape[0], TADA_NUM_EOS_TOKENS), self.eot_id, dtype=torch.long)
        bos = torch.full((input_ids.shape[0], 1), self.bos_token_id, dtype=torch.long)
        return torch.cat([bos, input_ids, eos_tokens], dim=1)

    def _build_input_ids(self, prompt_text: str, generation_text: str) -> tuple[torch.Tensor, int]:
        """Tokenize prompt + generation text with BOS/EOS and chat template.

        Returns:
            input_ids: (1, T) token IDs
            prefix_len: number of prefix tokens (BOS + chat template) before actual text
        """
        # Normalize text (matching reference behavior)
        generation_text = normalize_text(generation_text)
        if prompt_text:
            prompt_text = normalize_text(prompt_text)
        logger.info(f"Normalized text: '{prompt_text}{generation_text}'")

        # Build chat template
        prefix = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        full_text = prompt_text + generation_text
        text_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

        all_tokens = torch.tensor([text_tokens], dtype=torch.long)
        all_tokens = self._add_bos_eos(all_tokens)

        # Insert prefix after BOS
        prefix_tensor = torch.tensor([prefix_tokens], dtype=torch.long)
        input_ids = torch.cat([all_tokens[:, :1], prefix_tensor, all_tokens[:, 1:]], dim=1)
        prefix_len = 1 + len(prefix_tokens)  # BOS + prefix tokens
        return input_ids, prefix_len

    def _encode_prompt(self, prompt_audio: torch.Tensor, prompt_text: str):
        """
        Encode prompt audio to get acoustic features and alignment info.

        Args:
            prompt_audio: (B, 1, T) raw audio at 24kHz
            prompt_text: text spoken in the prompt
        Returns:
            token_values: (B, N, 512) acoustic features
            token_positions: (B, N) frame positions for alignment
            token_masks: (B, T_frames) binary mask
        """
        # Pad audio (reference does F.pad(audio, (0, 960), value=0) before WavEncoder)
        padded_audio = torch.nn.functional.pad(prompt_audio, (0, 960), value=0)

        # For now, use the full encoder pipeline
        enc_out_tt, padded_token_masks = encoder_get_encoder_outputs(
            padded_audio,
            torch.ones(1, padded_audio.shape[-1] // 480, dtype=torch.long),  # Initial mask
            self.wav_encoder,
            self.encoder_pos_emb_weight,
            self.encoder_hidden_linear_weight,
            self.encoder_hidden_linear_bias,
            self.encoder_local_attn_params,
            device=self.mesh_device,
            input_mesh_mapper=self.input_mesh_mapper,
            output_mesh_composer=self.output_mesh_composer,
        )

        # Transfer back to CPU
        enc_out_cpu = ttnn.to_torch(enc_out_tt, mesh_composer=self.output_mesh_composer)
        if enc_out_cpu.dim() == 4:
            enc_out_cpu = enc_out_cpu.squeeze(1)
        ttnn.deallocate(enc_out_tt)

        return enc_out_cpu, padded_token_masks

    def _decode_wav(self, encoded: torch.Tensor, time_before: torch.Tensor) -> torch.Tensor:
        """
        Expand acoustic features by duration and decode to waveform.

        Args:
            encoded: (N, 512) acoustic features for one batch element
            time_before: (N+1,) duration values
        Returns:
            (1, 1, T) waveform tensor
        """
        time_before = time_before[: encoded.shape[0] + 1]
        if time_before.shape[0] == 0:
            return torch.zeros(1, 1, 0)

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            # Gap fill: repeat zeros for (time_before[pos] - 1) frames
            gap_len = max(0, int(time_before[pos].item()) - 1)
            if gap_len > 0:
                encoded_expanded.append(torch.zeros(gap_len, encoded.shape[-1], dtype=encoded.dtype))
            # Actual acoustic features
            encoded_expanded.append(encoded[pos].unsqueeze(0))

        # Trailing gap
        trail_len = int(time_before[-1].item())
        if trail_len > 0:
            encoded_expanded.append(torch.zeros(trail_len, encoded.shape[-1], dtype=encoded.dtype))

        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)  # (1, T_expanded, 512)
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

        # Decode through decoder transformer + DAC CNN
        waveform = decoder_forward(
            encoded_expanded,
            token_masks,
            self.decoder_proj_weight,
            self.decoder_proj_bias,
            self.decoder_local_attn_params,
            self.wav_decoder,
            device=self.mesh_device,
            input_mesh_mapper=self.input_mesh_mapper,
            output_mesh_composer=self.output_mesh_composer,
        )
        return waveform

    @torch.no_grad()
    def _decode_wav_reference(self, encoded: torch.Tensor, time_before: torch.Tensor) -> torch.Tensor:
        """Run the full reference decoder on CPU for comparison."""
        from models.demos.audio.tada.reference.tada_reference import create_decoder_segment_attention_mask

        time_before = time_before[: encoded.shape[0] + 1]
        if time_before.shape[0] == 0:
            return torch.zeros(1, 1, 0)

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            gap_len = max(0, int(time_before[pos].item()) - 1)
            if gap_len > 0:
                encoded_expanded.append(torch.zeros(gap_len, encoded.shape[-1], dtype=encoded.dtype))
            encoded_expanded.append(encoded[pos].unsqueeze(0))

        trail_len = int(time_before[-1].item())
        if trail_len > 0:
            encoded_expanded.append(torch.zeros(trail_len, encoded.shape[-1], dtype=encoded.dtype))

        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)  # (1, T, 512)
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

        # Run fully on CPU with reference models
        decoder_input = self.ref_decoder_proj(encoded_expanded.float())
        attn_mask = create_decoder_segment_attention_mask(token_masks, version="v2")
        decoded = self.ref_dec_local_attn(decoder_input, mask=attn_mask)
        x_rec = self.wav_decoder(decoded.transpose(1, 2).float())
        return x_rec

    @torch.no_grad()
    def generate(
        self,
        prompt_audio: torch.Tensor | None = None,
        prompt_text: str = "",
        generation_text: str = "Hello, this is a test.",
        inference_options: TadaInferenceOptions | None = None,
        max_steps: int = 1024,
    ) -> dict:
        """
        Generate speech from text, optionally conditioned on a prompt.

        Args:
            prompt_audio: (1, 1, T) audio prompt at 24kHz, or None for unconditional
            prompt_text: text spoken in the prompt audio
            generation_text: text to synthesize
            inference_options: generation parameters
            max_steps: maximum AR steps
        Returns:
            dict with "audio" (waveform tensor), "text_token_ids", etc.
        """
        from models.tt_transformers.tt.common import Mode

        opts = inference_options or TadaInferenceOptions()
        B = 1

        # Encode prompt if provided
        prompt_acoustic_features = None
        prompt_acoustic_masks = None
        prompt_time_len_before = None
        prompt_time_len_after = None

        if prompt_audio is not None and prompt_audio.numel() > 0:
            # Get encoder outputs (acoustic features)
            enc_features, token_masks_enc = self._encode_prompt(prompt_audio, prompt_text)
            # For simplicity in bringup, treat all encoder frames as prompt acoustic features
            prompt_acoustic_features = enc_features  # (B, N_frames, 512)
            prompt_acoustic_masks = torch.ones(prompt_acoustic_features.shape[:2], dtype=torch.long)

            # Compute time gaps from token positions (simplified for bringup)
            n_tokens = prompt_acoustic_features.shape[1]
            # Default uniform duration (will be refined with aligner)
            prompt_time_len_before = torch.ones(B, n_tokens + 1, dtype=torch.long) * 10
            prompt_time_len_before[:, 0] = 0
            prompt_time_len_after = torch.ones(B, n_tokens + 1, dtype=torch.long) * 10

        # Build input IDs
        input_ids, prefix_len = self._build_input_ids(prompt_text, generation_text)
        prompt_len = input_ids.shape[1]
        num_steps = min(max_steps, prompt_len + 100)  # Safety cap

        # Log token sequence for debugging
        logger.info(f"Input sequence ({input_ids.shape[1]} tokens):")
        for pos in range(min(input_ids.shape[1], 30)):
            tok_id = input_ids[0, pos].item()
            tok_str = self.tokenizer.decode([tok_id])
            region = "PREFIX" if pos < prefix_len else ("TEXT" if pos < prompt_len - TADA_NUM_EOS_TOKENS else "EOT")
            logger.info(f"  pos {pos}: {tok_id} = {repr(tok_str)} [{region}]")

        # For unconditional generation (no prompt audio), create zero prompt features
        # for the prefix positions, matching reference behavior:
        # Reference pads prompt_acoustic_features with prefix_len zeros, then trims
        # num_transition_steps from the end.  For no-prompt case, this creates a small
        # set of zero features that get fed back as acoustic input during prefix steps.
        # Reference also shifts mask left by 1 (last position gets mask=1).
        has_real_prompt = prompt_acoustic_features is not None
        if not has_real_prompt:
            ref_prefix_len = prefix_len - 1  # Exclude BOS (reference prefix_len doesn't include BOS)
            n_prompt_pad = max(0, ref_prefix_len - TADA_SHIFT_ACOUSTIC)
            if n_prompt_pad > 0:
                prompt_acoustic_features = torch.zeros(B, n_prompt_pad, TADA_ACOUSTIC_DIM)
                # Reference shifts mask left by 1: cat([masks[:, 1:], ones[:, :1]])
                # For no-prompt with n_prompt_pad=2: original masks=[0,0] → shifted=[0,1]
                prompt_acoustic_masks = torch.zeros(B, n_prompt_pad, dtype=torch.long)
                if n_prompt_pad > 0:
                    prompt_acoustic_masks[:, -1] = 1  # Last position gets mask=1
                prompt_time_len_before = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
                prompt_time_len_after = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
                logger.info(f"Created {n_prompt_pad} zero prompt features for prefix (mask shifted)")

        # Number of prefix acoustic features to discard from the output:
        # For no-prompt: discard fake prompt features + 1 structural token feature.
        #   n_prompt_pad(=2) + 1 = 3, keeping features from "This" onwards.
        #   (Reference uses larger trim for voice cloning, but for unconditional
        #   generation we want all text token features.)
        # For prompt: reference formula = num_prompt_tokens + num_transition_steps - 1
        if not has_real_prompt:
            n_prompt_pad_val = max(0, (prefix_len - 1) - TADA_SHIFT_ACOUSTIC)
            n_prefix_acoustic = n_prompt_pad_val + 1  # fake prompt + 1 structural
        else:
            n_prefix_acoustic = max(0, prefix_len - TADA_SHIFT_ACOUSTIC)
        logger.info(f"Prefix length: {prefix_len}, will discard first {n_prefix_acoustic} acoustic features")

        # Initialize AR state
        acoustic_features = torch.zeros(B, TADA_ACOUSTIC_DIM)
        acoustic_masks = torch.zeros(B, dtype=torch.long)
        time_len_before = torch.zeros(B, dtype=torch.long)
        time_len_after = torch.zeros(B, dtype=torch.long)

        # Build parameter namespace for tada_embed_inputs
        tada_params = self._build_tada_parameters_namespace()

        all_acoustic_features = []
        all_time_before = []
        all_output_token_ids = []
        all_hidden_states = []  # Save hidden states for debugging

        shift_acoustic = TADA_SHIFT_ACOUSTIC
        use_cfg = opts.acoustic_cfg_scale != 1.0

        # Structural token IDs for negative conditioning (kept as-is, rest → pad)
        start_header_id = self.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")

        logger.info(f"Starting AR generation for {num_steps} steps (CFG={'on' if use_cfg else 'off'})...")

        for step in range(num_steps):
            # Get current text token
            if step < input_ids.shape[1]:
                input_slice = input_ids[:, step]
            else:
                input_slice = input_ids[:, -1]

            # Build embedding on TT
            inputs_embeds = tada_embed_inputs(
                input_slice,
                acoustic_features,
                acoustic_masks,
                time_len_before,
                time_len_after,
                parameters=tada_params,
                device=self.mesh_device,
                input_mesh_mapper=self.input_mesh_mapper,
            )

            # Llama decode step — ensure 4D shape (1, 1, B_padded, dim)
            mode = Mode.DECODE
            inputs_embeds = ttnn.unsqueeze_to_4D(inputs_embeds)
            current_pos_tt, rope_idxs = self._prepare_decode_pos(step)
            hidden_tt = self._run_llama_step(inputs_embeds, current_pos_tt, rope_idxs, mode)

            # Negative conditioning: run second Llama pass with pad token (same acoustic)
            # Structural tokens (header markers, eot) are kept; content tokens → pad
            if use_cfg:
                is_structural = (
                    (input_slice == start_header_id) | (input_slice == end_header_id) | (input_slice == self.eot_id)
                )
                neg_input_slice = torch.where(
                    is_structural, input_slice, torch.full_like(input_slice, self.pad_token_id)
                )
                neg_embeds = tada_embed_inputs(
                    neg_input_slice,
                    acoustic_features,
                    acoustic_masks,
                    time_len_before,
                    time_len_after,
                    parameters=tada_params,
                    device=self.mesh_device,
                    input_mesh_mapper=self.input_mesh_mapper,
                )
                neg_embeds = ttnn.unsqueeze_to_4D(neg_embeds)
                neg_hidden_tt = self._run_llama_step(
                    neg_embeds, current_pos_tt, rope_idxs, mode, kv_cache=self.neg_kv_cache
                )

            # hidden_tt is 4D (1, 1, B_padded, dim) from Llama — reshape to 3D (B, 1, dim)
            hidden_cpu = ttnn.to_torch(hidden_tt, mesh_composer=self.output_mesh_composer)
            if hidden_cpu.dim() == 4:
                hidden_cpu = hidden_cpu.squeeze(0).squeeze(0)  # (B_padded, dim)
                hidden_cpu = hidden_cpu[:B].unsqueeze(1)  # (B, 1, dim)
            elif hidden_cpu.dim() == 3:
                hidden_cpu = hidden_cpu[:B]
            ttnn.deallocate(hidden_tt)
            all_hidden_states.append(hidden_cpu.clone())
            hidden_3d = ttnn.from_torch(
                hidden_cpu,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=self.input_mesh_mapper,
            )

            # Extract negative condition
            if use_cfg:
                neg_cpu = ttnn.to_torch(neg_hidden_tt, mesh_composer=self.output_mesh_composer)
                if neg_cpu.dim() == 4:
                    neg_cpu = neg_cpu.squeeze(0).squeeze(0)[:B].unsqueeze(1)
                elif neg_cpu.dim() == 3:
                    neg_cpu = neg_cpu[:B]
                ttnn.deallocate(neg_hidden_tt)
                neg_cond_tt = ttnn.from_torch(
                    neg_cpu,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=self.input_mesh_mapper,
                )
            else:
                neg_cond_tt = hidden_3d  # unused when cfg=1.0, pass same tensor

            # LM head for text logits
            logits_tt = tada_lm_head(hidden_3d, parameters=tada_params)
            logits_cpu = ttnn.to_torch(logits_tt, mesh_composer=self.output_mesh_composer)
            if logits_cpu.dim() == 3:
                logits_cpu = logits_cpu.squeeze(1)  # (B, vocab_size)
            ttnn.deallocate(logits_tt)

            # Sample next text token if past prompt
            if step >= input_ids.shape[1] - 1:
                if opts.random_seed is not None:
                    torch.manual_seed(opts.random_seed + 10000 + step)
                next_token = sample_text_token(logits_cpu, input_ids, opts, self.pad_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                all_output_token_ids.append(next_token)

                # Stop on EOS
                if next_token.item() == self.eos_token_id or next_token.item() == self.eot_id:
                    logger.info(f"EOS at step {step}")
                    break
            else:
                all_output_token_ids.append(input_ids[:, step + 1 : step + 2])

            # Flow matching ODE to predict acoustic features + duration
            speech = self._solve_flow_matching(hidden_3d, neg_cond_tt, opts, step_idx=step)
            ttnn.deallocate(hidden_3d)
            if use_cfg:
                ttnn.deallocate(neg_cond_tt)

            # Extract time from gray code
            time_gray = speech[..., -TADA_TIME_DIM:]
            predicted_time_before = decode_gray_code_to_time(time_gray[..., :TADA_NUM_TIME_BITS], TADA_NUM_TIME_BITS)
            predicted_time_after = decode_gray_code_to_time(time_gray[..., TADA_NUM_TIME_BITS:], TADA_NUM_TIME_BITS)

            # Update acoustic state for next step (with shift_acoustic delay)
            if step >= shift_acoustic:
                if prompt_acoustic_features is not None and step - shift_acoustic < prompt_acoustic_features.shape[1]:
                    acoustic_features = prompt_acoustic_features[:, step - shift_acoustic]
                    acoustic_masks = prompt_acoustic_masks[:, step - shift_acoustic]
                else:
                    acoustic_features = speech[..., :TADA_ACOUSTIC_DIM]
                    acoustic_masks = torch.ones(B, dtype=torch.long)
                all_acoustic_features.append(
                    acoustic_features.unsqueeze(1) if acoustic_features.dim() == 2 else acoustic_features
                )

                if prompt_time_len_before is not None and step - shift_acoustic < prompt_time_len_before.shape[1] - 1:
                    time_len_before = prompt_time_len_before[:, step - shift_acoustic + 1]
                    time_len_after = prompt_time_len_after[:, step - shift_acoustic + 1]
                else:
                    time_len_before = predicted_time_before
                    time_len_after = predicted_time_after
                all_time_before.append(time_len_before.unsqueeze(1) if time_len_before.dim() == 1 else time_len_before)
            else:
                acoustic_features = torch.zeros(B, TADA_ACOUSTIC_DIM)
                acoustic_masks = torch.zeros(B, dtype=torch.long)
                time_len_before = torch.zeros(B, dtype=torch.long)
                time_len_after = torch.zeros(B, dtype=torch.long)

            tok_name = (
                repr(self.tokenizer.decode([input_slice[0].item()])) if input_slice[0].item() < TADA_VOCAB_SIZE else "?"
            )
            src = (
                "prompt"
                if (
                    prompt_acoustic_features is not None
                    and step >= shift_acoustic
                    and step - shift_acoustic < prompt_acoustic_features.shape[1]
                )
                else ("predicted" if step >= shift_acoustic else "zeros")
            )
            if step % 5 == 0 or step < 15:
                logger.info(
                    f"  Step {step}/{num_steps} [{tok_name}]: "
                    f"speech_norm={speech.norm().item():.3f}, "
                    f"af_src={src}, af_norm={acoustic_features.norm().item():.3f}, "
                    f"t_before={time_len_before.tolist()}, "
                    f"t_after={time_len_after.tolist()}, "
                    f"mask={acoustic_masks.tolist()}"
                )

        logger.info("AR generation complete. Decoding waveform...")

        # Collect results
        result = {
            "text_token_ids": torch.cat(all_output_token_ids, dim=1) if all_output_token_ids else None,
            "text": self.tokenizer.decode(input_ids[0]) if input_ids is not None else "",
        }

        if all_acoustic_features:
            acoustic_cat = torch.cat([f if f.dim() == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1)
            # Un-normalize acoustic features
            acoustic_cat = acoustic_cat * TADA_ACOUSTIC_STD + TADA_ACOUSTIC_MEAN

            time_before_cat = torch.cat([t if t.dim() == 2 else t.unsqueeze(1) for t in all_time_before], dim=1)
            # Add trailing time_before
            if all_time_before:
                time_before_cat = torch.cat(
                    [
                        time_before_cat,
                        all_time_before[-1] if all_time_before[-1].dim() == 2 else all_time_before[-1].unsqueeze(1),
                    ],
                    dim=1,
                )

            logger.info(
                f"Acoustic features (raw): shape={acoustic_cat.shape}, "
                f"norm={acoustic_cat.norm().item():.3f}, "
                f"min={acoustic_cat.min().item():.3f}, max={acoustic_cat.max().item():.3f}"
            )
            logger.info(f"Time before (raw): shape={time_before_cat.shape}, values={time_before_cat[0].tolist()}")

            # Slice off prefix acoustic features (structural tokens produce garbage)
            # The first n_prefix_acoustic features correspond to non-text prefix tokens
            if n_prefix_acoustic > 0 and acoustic_cat.shape[1] > n_prefix_acoustic:
                logger.info(
                    f"Slicing off {n_prefix_acoustic} prefix acoustic features "
                    f"(keeping {acoustic_cat.shape[1] - n_prefix_acoustic} text features)"
                )
                acoustic_cat = acoustic_cat[:, n_prefix_acoustic:, :]
                time_before_cat = time_before_cat[:, n_prefix_acoustic:]

            logger.info(
                f"Acoustic features (trimmed): shape={acoustic_cat.shape}, " f"norm={acoustic_cat.norm().item():.3f}"
            )
            logger.info(f"Time before (trimmed): values={time_before_cat[0].tolist()}")

            # Decode waveform (TTNN decoder transformer + DACDecoder CNN)
            try:
                wav = self._decode_wav(acoustic_cat[0], time_before_cat[0])
                # Remove leading silence (from first time_before gap)
                leading_silence_samples = int(time_before_cat[0, 0].item() * 480)
                if leading_silence_samples > 0 and wav.shape[-1] > leading_silence_samples:
                    wav = wav[..., leading_silence_samples:]
                    logger.info(f"Trimmed {leading_silence_samples} leading silence samples")
                result["audio"] = wav
                logger.info(f"Decoded waveform: shape={wav.shape}, norm={wav.norm().item():.4f}")
            except Exception as e:
                logger.error(f"Waveform decoding failed: {e}")
                import traceback

                traceback.print_exc()
                result["audio"] = torch.zeros(1, 1, 0)

            # Also run reference decoder (fully on CPU) for comparison
            try:
                wav_ref = self._decode_wav_reference(acoustic_cat[0], time_before_cat[0])
                if leading_silence_samples > 0 and wav_ref.shape[-1] > leading_silence_samples:
                    wav_ref = wav_ref[..., leading_silence_samples:]
                result["audio_reference"] = wav_ref
                logger.info(f"Reference waveform: shape={wav_ref.shape}, norm={wav_ref.norm().item():.4f}")
            except Exception as e:
                logger.warning(f"Reference decoder comparison failed: {e}")

            result["acoustic_features"] = acoustic_cat
            result["time_before"] = time_before_cat
        else:
            result["audio"] = torch.zeros(1, 1, 0)
            result["acoustic_features"] = None
            result["time_before"] = None

        # Save hidden states for debugging
        if all_hidden_states:
            result["hidden_states"] = all_hidden_states
            output_dir = os.path.join(os.path.dirname(__file__), "..", "demo", "output")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                {"hidden_states": [h.float() for h in all_hidden_states], "input_ids": input_ids},
                os.path.join(output_dir, "debug_hidden_states.pt"),
            )
            logger.info(f"Saved {len(all_hidden_states)} hidden states to debug_hidden_states.pt")

        return result
