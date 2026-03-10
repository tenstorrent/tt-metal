# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Bark Small Pipeline Orchestrator.

Loads HuggingFace suno/bark-small weights and runs the full 3-stage pipeline:
    Text -> Semantic Tokens -> Coarse Codebooks -> Fine Codebooks -> Audio Waveform

Usage:
    model = TtBarkModel(device)
    audio = model.generate("Hello, this is Bark speaking!")
"""

import time

import numpy as np
import torch

import ttnn
from models.demos.wormhole.bark.tt.bark_fine import TtBarkFineModel, preprocess_fine_model_parameters
from models.demos.wormhole.bark.tt.bark_gpt import BarkConfig, TtBarkGPT, preprocess_model_parameters


# ----- Bark upstream token constants (from HF BarkSemanticGenerationConfig / BarkCoarseGenerationConfig) -----
SEMANTIC_VOCAB_SIZE = 10_000
SEMANTIC_PAD_TOKEN = 10_000  # EOS for semantic stage
TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_INFER_TOKEN = 129_599
COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050
CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2


class TtBarkModel:
    """Complete Bark Small pipeline: text-to-audio on Tenstorrent hardware.

    Stages:
    1. Text-to-Semantic: tokenized text -> 10k semantic token vocabulary
    2. Semantic-to-Coarse: semantic tokens -> 2 coarse EnCodec codebooks
    3. Coarse-to-Fine: 2 codebooks -> 8 codebooks
    4. EnCodec Decoder: 8 codebooks -> 24kHz mono audio waveform
    """

    def __init__(self, device, model_name="suno/bark-small"):
        """Initialize the Bark pipeline by loading HuggingFace weights.

        Args:
            device: TTNN device
            model_name: HuggingFace model name (default: suno/bark-small)
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading Bark model from {model_name}...")
        self._load_model(model_name)
        print("Bark model loaded successfully!")

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _load_model(self, model_name):
        """Load all components from HuggingFace."""
        from transformers import AutoProcessor, BarkModel

        # Load the full HF model
        hf_model = BarkModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Store generation configs (compatible with old + new HF versions)
        gen = hf_model.generation_config
        self.semantic_generation_config = getattr(gen, "semantic_generation_config", gen)
        self.coarse_generation_config = getattr(gen, "coarse_generation_config", gen)
        self.fine_generation_config = getattr(gen, "fine_generation_config", gen)

        # Get device grid size for Stage 3 optimizations
        compute_grid = self.device.compute_with_storage_grid_size()

        # --- Stage 1: Semantic model ---
        semantic_config = BarkConfig(
            hidden_size=hf_model.semantic.config.hidden_size,
            num_heads=hf_model.semantic.config.num_heads,
            num_layers=hf_model.semantic.config.num_layers,
            block_size=hf_model.semantic.config.block_size,
            input_vocab_size=hf_model.semantic.config.input_vocab_size,
            output_vocab_size=hf_model.semantic.config.output_vocab_size,
            bias=getattr(hf_model.semantic.config, "bias", False),
            use_lofi=False,
            grid_size=compute_grid,
        )
        semantic_params = preprocess_model_parameters(hf_model.semantic, self.device)
        self.semantic_model = TtBarkGPT(self.device, semantic_params, semantic_config, is_causal=True)

        # --- Stage 2: Coarse model ---
        coarse_config = BarkConfig(
            hidden_size=hf_model.coarse_acoustics.config.hidden_size,
            num_heads=hf_model.coarse_acoustics.config.num_heads,
            num_layers=hf_model.coarse_acoustics.config.num_layers,
            block_size=hf_model.coarse_acoustics.config.block_size,
            input_vocab_size=hf_model.coarse_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.coarse_acoustics.config.output_vocab_size,
            bias=getattr(hf_model.coarse_acoustics.config, "bias", False),
            use_lofi=False,
            grid_size=compute_grid,
        )
        coarse_params = preprocess_model_parameters(hf_model.coarse_acoustics, self.device)
        self.coarse_model = TtBarkGPT(self.device, coarse_params, coarse_config, is_causal=True)

        # --- Stage 3: Fine model ---
        fine_config = BarkConfig(
            hidden_size=hf_model.fine_acoustics.config.hidden_size,
            num_heads=hf_model.fine_acoustics.config.num_heads,
            num_layers=hf_model.fine_acoustics.config.num_layers,
            block_size=hf_model.fine_acoustics.config.block_size,
            input_vocab_size=hf_model.fine_acoustics.config.input_vocab_size,
            output_vocab_size=hf_model.fine_acoustics.config.output_vocab_size,
            bias=True,  # Fine model uses bias for LayerNorm
            use_lofi=False,
            grid_size=compute_grid,
        )
        fine_params = preprocess_fine_model_parameters(hf_model.fine_acoustics, self.device)
        self.fine_model = TtBarkFineModel(
            self.device,
            fine_params,
            fine_config,
            n_codes_total=hf_model.fine_acoustics.config.n_codes_total,
            n_codes_given=hf_model.fine_acoustics.config.n_codes_given,
        )
        self.hf_fine = hf_model.fine_acoustics

        # --- EnCodec decoder ---
        self.codec_model = hf_model.codec_model
        self.codec_model.eval()

        # --- Defensive: verify hardcoded constants match the loaded checkpoint ---
        expected_infer = hf_model.semantic.config.input_vocab_size - 1
        assert expected_infer == SEMANTIC_INFER_TOKEN, (
            f"SEMANTIC_INFER_TOKEN mismatch: checkpoint has {expected_infer}, "
            f"code has {SEMANTIC_INFER_TOKEN}"
        )

        # Clean up full model reference
        del hf_model

    def generate_semantic_tokens(self, text: str, voice_preset=None) -> torch.Tensor:
        """Stage 1: Generate semantic tokens using optimized TTNN model with KV caching.

        Follows the upstream HF BarkSemanticModel.generate() contract:
        1. Offsets tokenizer IDs into the semantic embedding range (+TEXT_ENCODING_OFFSET)
        2. Masks padding positions to TEXT_ENCODING_OFFSET (text_pad_token)
        3. Appends the SEMANTIC_INFER_TOKEN to the prompt
        4. Suppresses logits above SEMANTIC_PAD_TOKEN (allows EOS at index 10000)
        5. Returns only the newly generated semantic tokens (prompt stripped)

        Args:
            text: Input text string
            voice_preset: Optional voice preset dict

        Returns:
            semantic_tokens: [1, seq_len] semantic token IDs in [0, SEMANTIC_VOCAB_SIZE)
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask", None)
        if input_ids.shape[0] != 1:
            raise ValueError("Bark TTNN implementation currently only supports batch size 1")

        # --- Upstream contract: offset text IDs and mask padding ---
        input_ids = input_ids + TEXT_ENCODING_OFFSET
        if attention_mask is not None:
            # Mask padding positions back to text_pad_token (= TEXT_ENCODING_OFFSET)
            input_ids = input_ids.masked_fill((1 - attention_mask).bool(), TEXT_ENCODING_OFFSET)
        infer_token = torch.tensor([[SEMANTIC_INFER_TOKEN]], dtype=torch.long)
        input_ids = torch.cat([input_ids, infer_token], dim=-1)

        # Initial pre-fill (process entire prompt) — use DRAM to avoid L1 overflow with KV cache
        gen_mem = ttnn.DRAM_MEMORY_CONFIG
        logits, layer_past = self.semantic_model(input_ids=input_ids, use_cache=True, memory_config=gen_mem)

        # --- Unified autoregressive loop with logits suppression on host ---
        max_new_tokens = getattr(self.semantic_generation_config, "max_new_tokens", None) or 768
        generated_tokens = []
        next_token_torch = None
        tt_next_token = None

        for step in range(max_new_tokens):
            if step == 0:
                # Use prefill logits
                logits_torch = ttnn.to_torch(logits).squeeze(0)
                ttnn.deallocate(logits)
            else:
                logits, layer_past = self.semantic_model(
                    input_ids=tt_next_token,
                    layer_past=layer_past,
                    use_cache=True,
                    memory_config=gen_mem,
                )
                ttnn.deallocate(tt_next_token)
                logits_torch = ttnn.to_torch(logits).squeeze(0)
                ttnn.deallocate(logits)

            last_logits = logits_torch[:, -1, :]
            # Allow EOS at SEMANTIC_PAD_TOKEN (10000); suppress everything above it
            last_logits[:, SEMANTIC_PAD_TOKEN + 1 :] = -float("inf")
            next_token_torch = torch.argmax(last_logits, dim=-1)  # [1]

            # EOS check BEFORE appending — EOS itself is never included in output
            if next_token_torch.item() == SEMANTIC_PAD_TOKEN:
                break

            generated_tokens.append(next_token_torch.unsqueeze(-1))
            tt_next_token = ttnn.from_torch(
                next_token_torch.unsqueeze(0).to(torch.int32),
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        if tt_next_token is not None:
            ttnn.deallocate(tt_next_token)

        if not generated_tokens:
            return torch.zeros((1, 1), dtype=torch.long)
        return torch.cat(generated_tokens, dim=-1)

    def generate_coarse_tokens(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        """Stage 2: Generate coarse EnCodec tokens using optimized TTNN model with KV caching.

        Follows the upstream HF BarkCoarseModel.generate() contract:
        1. Offsets semantic tokens by SEMANTIC_VOCAB_SIZE
        2. Appends COARSE_INFER_TOKEN to the prompt
        3. Applies alternating-codebook logits masking per step (allows EOS always)
        4. Generates interleaved codebook0/codebook1 tokens
        5. Remaps output IDs: (token - SEMANTIC_VOCAB_SIZE) % CODEBOOK_SIZE

        Args:
            semantic_tokens: [1, seq_len] semantic token IDs from Stage 1

        Returns:
            coarse_tokens: [1, coarse_seq_len * 2] interleaved, values in [0, CODEBOOK_SIZE)
        """
        if semantic_tokens.shape[0] != 1:
            raise ValueError("Bark TTNN implementation currently only supports batch size 1")

        # --- Upstream contract: offset semantic tokens and append infer token ---
        input_ids = semantic_tokens.to(torch.long) + SEMANTIC_VOCAB_SIZE
        infer_token = torch.tensor([[COARSE_INFER_TOKEN]], dtype=torch.long)
        input_ids = torch.cat([input_ids, infer_token], dim=-1)

        # Initial pre-fill — use DRAM to avoid L1 overflow with KV cache
        gen_mem = ttnn.DRAM_MEMORY_CONFIG
        logits, layer_past = self.coarse_model(input_ids=input_ids, use_cache=True, memory_config=gen_mem)

        # --- Unified loop with alternating-codebook logits suppression on host ---
        max_new_tokens = getattr(self.coarse_generation_config, "max_new_tokens", None) or 768
        generated_tokens = []
        tt_next_token = None

        for step in range(max_new_tokens):
            if step == 0:
                logits_torch = ttnn.to_torch(logits).squeeze(0)
                ttnn.deallocate(logits)
            else:
                logits, layer_past = self.coarse_model(
                    input_ids=tt_next_token,
                    layer_past=layer_past,
                    use_cache=True,
                    memory_config=gen_mem,
                )
                ttnn.deallocate(tt_next_token)
                logits_torch = ttnn.to_torch(logits).squeeze(0)
                ttnn.deallocate(logits)

            last_logits = logits_torch[:, -1, :]  # [1, vocab]

            # Alternating codebook suppression — critical for correct coarse generation
            codebook_idx = step % N_COARSE_CODEBOOKS
            allowed_start = SEMANTIC_VOCAB_SIZE + codebook_idx * CODEBOOK_SIZE
            allowed_end = allowed_start + CODEBOOK_SIZE
            mask = torch.full_like(last_logits, -float("inf"))
            mask[:, allowed_start:allowed_end] = 0.0
            mask[:, COARSE_SEMANTIC_PAD_TOKEN] = 0.0  # always allow EOS
            next_token_torch = torch.argmax(last_logits + mask, dim=-1)

            # EOS check BEFORE appending — EOS never enters output
            if next_token_torch.item() == COARSE_SEMANTIC_PAD_TOKEN:
                break

            generated_tokens.append(next_token_torch.unsqueeze(-1))
            tt_next_token = ttnn.from_torch(
                next_token_torch.unsqueeze(0).to(torch.int32),
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        if tt_next_token is not None:
            ttnn.deallocate(tt_next_token)

        coarse_output = torch.cat(generated_tokens, dim=-1)  # [1, n_tokens]
        # Remap to [0, CODEBOOK_SIZE) — fine model embedding tables expect this range
        coarse_output = (coarse_output - SEMANTIC_VOCAB_SIZE) % CODEBOOK_SIZE
        return coarse_output

    def generate_fine_tokens(self, coarse_tokens: torch.Tensor) -> torch.Tensor:
        """Stage 3: Generate fine EnCodec tokens (on-device loop).

        Maintains all codebooks as separate TTNN tensors on the device to avoid
        repeated data movement between host and device.

        Args:
            coarse_tokens: [batch, coarse_seq_len * 2] interleaved

        Returns:
            fine_tokens: [batch, seq_len, 8] all codebooks on host
        """
        n_coarse = self.fine_model.n_codes_given  # 2
        batch_size = coarse_tokens.shape[0]
        coarse_seq_len = coarse_tokens.shape[1] // n_coarse

        # De-interleave: [batch, seq*2] -> [batch, seq, 2]
        coarse_tokens_reshaped = coarse_tokens.reshape(batch_size, coarse_seq_len, n_coarse)

        # Move initial 2 codebooks to device as a list
        tt_codebooks = []
        for i in range(n_coarse):
            # Shape: [1, batch, seq, 1]
            cb_i = ttnn.from_torch(
                coarse_tokens_reshaped[:, :, i].unsqueeze(0).unsqueeze(-1).to(torch.int32),
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_codebooks.append(cb_i)

        # Predict codebooks 2-7 autoregressively on device
        with torch.no_grad():
            for codebook_idx in range(n_coarse, self.fine_model.n_codes_total):
                # logits: [1, batch, seq, vocab]
                logits = self.fine_model(codebook_idx, tt_codebooks)

                # argmax on device: [1, batch, seq, 1]
                preds = ttnn.argmax(logits, dim=-1)
                tt_codebooks.append(preds)

                # Optimization: deallocate logits immediately
                ttnn.deallocate(logits)

        # Gather all codebooks from device to host
        # shape [batch, seq, 8]
        fine_tokens = torch.zeros(batch_size, coarse_seq_len, self.fine_model.n_codes_total, dtype=torch.long)
        for i, tt_cb in enumerate(tt_codebooks):
            cb_torch = ttnn.to_torch(tt_cb).squeeze(0).squeeze(-1).to(torch.long)
            cb_torch = cb_torch.clamp(0, CODEBOOK_SIZE - 1)  # Safety clamp to [0, 1023]
            fine_tokens[:, :, i] = cb_torch
            ttnn.deallocate(tt_cb)

        return fine_tokens

    def decode_audio(self, fine_tokens: torch.Tensor) -> np.ndarray:
        """Stage 4: Decode EnCodec tokens to audio waveform.

        Uses the correct upstream two-step decode path:
        1. quantizer.decode() to get continuous embeddings from codebook indices
        2. decoder() to produce audio from embeddings

        Args:
            fine_tokens: [batch, seq_len, 8] codebook tokens

        Returns:
            audio: numpy array of 24kHz mono audio
        """
        # Transpose to [n_codebooks, batch, seq_len] for quantizer.decode
        # fine_tokens: [batch, seq, 8] -> [8, batch, seq]
        fine_output = fine_tokens.transpose(0, 2).transpose(1, 2)  # [8, batch, seq]
        assert fine_output.shape[0] == 8 and fine_output.ndim == 3, (
            f"EnCodec input shape wrong: expected [8, batch, seq], got {fine_output.shape}"
        )

        with torch.no_grad():
            # Step 1: Decode codebook indices to continuous embeddings
            emb = self.codec_model.quantizer.decode(fine_output)
            # Step 2: Decode embeddings to audio waveform
            audio_values = self.codec_model.decoder(emb)

        # audio_values: [batch, 1, samples] -> squeeze to [samples]
        audio = audio_values.squeeze().cpu().numpy()
        return audio

    def generate(
        self,
        text: str,
        voice_preset=None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Full text-to-audio pipeline.

        Args:
            text: Input text string
            voice_preset: Optional voice preset dict
            verbose: Print timing info

        Returns:
            audio: numpy array of 24kHz mono audio waveform
        """
        timings = {}

        # Stage 1: Text -> Semantic
        t0 = time.time()
        semantic_tokens = self.generate_semantic_tokens(text, voice_preset)
        timings["semantic"] = time.time() - t0
        if verbose:
            print(
                f"Stage 1 (Semantic): {semantic_tokens.shape[1]} tokens in {timings['semantic']:.2f}s "
                f"({semantic_tokens.shape[1] / timings['semantic']:.1f} tok/s)"
            )

        # Stage 2: Semantic -> Coarse
        t0 = time.time()
        coarse_tokens = self.generate_coarse_tokens(semantic_tokens)
        timings["coarse"] = time.time() - t0
        if verbose:
            print(
                f"Stage 2 (Coarse): {coarse_tokens.shape[1]} tokens in {timings['coarse']:.2f}s "
                f"({coarse_tokens.shape[1] / timings['coarse']:.1f} tok/s)"
            )

        # Stage 3: Coarse -> Fine
        t0 = time.time()
        fine_tokens = self.generate_fine_tokens(coarse_tokens)
        timings["fine"] = time.time() - t0
        if verbose:
            print(
                f"Stage 3 (Fine): {fine_tokens.shape[1]}x{fine_tokens.shape[2]} codebooks " f"in {timings['fine']:.2f}s"
            )

        # Stage 4: Decode audio
        t0 = time.time()
        audio = self.decode_audio(fine_tokens)
        timings["decode"] = time.time() - t0
        if verbose:
            total = sum(timings.values())
            duration = len(audio) / 24000
            rtf = total / duration if duration > 0 else float("inf")
            print(f"Stage 4 (Decode): {duration:.2f}s audio in {timings['decode']:.2f}s")
            print(f"Total: {total:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.2f}")

        return audio
