# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN SpeechT5 Autoregressive Generator.
Implements the generation loop for text-to-speech.
"""

import torch
import ttnn
from typing import Optional


class TTNNSpeechT5Generator:
    """
    Autoregressive generator for SpeechT5 TTS using TTNN.

    This class implements the generation loop that produces mel spectrograms
    from text token IDs autoregressively (one step at a time).
    """

    def __init__(
        self,
        encoder,
        decoder,
        postnet,
        device,
        reduction_factor: int = 2,
        max_decoder_steps: int = 200,
        stop_threshold: float = 0.5,
        num_mel_bins: int = 80,
    ):
        """
        Initialize generator.

        Args:
            encoder: TTNN encoder model
            decoder: TTNN decoder model
            postnet: TTNN post-net model
            device: TTNN device
            reduction_factor: Number of mel frames generated per decoder step (default: 2)
            max_decoder_steps: Maximum number of decoder steps (default: 200)
            stop_threshold: Threshold for stop token prediction (default: 0.5)
            num_mel_bins: Number of mel bins (default: 80 for SpeechT5)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.device = device
        self.reduction_factor = reduction_factor
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = stop_threshold
        self.num_mel_bins = num_mel_bins

    def generate(
        self,
        input_ids: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate mel spectrogram from text token IDs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            speaker_embeddings: Speaker embeddings [batch, 1, hidden_size] or [batch, hidden_size]
            verbose: Print progress messages

        Returns:
            mel_spectrogram: Generated mel spectrogram [batch, frames, num_mel_bins]
        """
        batch_size = input_ids.shape[0]

        if verbose:
            print(f"\n{'='*80}")
            print(f"TTNN AUTOREGRESSIVE GENERATION")
            print(f"{'='*80}")

        # Step 1: Encode input text (run once)
        if verbose:
            print(f"\n[1/4] Encoding text...")
            print(f"      Input: {batch_size} samples, {input_ids.shape[1]} tokens")

        encoder_hidden_states = self._encode(input_ids)

        if verbose:
            print(
                f"      ✓ Encoder output shape: {encoder_hidden_states.shape if hasattr(encoder_hidden_states, 'shape') else 'N/A'}"
            )

        # Step 2: Speaker embeddings (shape handling done in _generate_mel_frames)
        if speaker_embeddings is not None and verbose:
            if hasattr(speaker_embeddings, "shape"):
                print(f"\n[2/4] Speaker embeddings: {speaker_embeddings.shape}")

        # Step 3: Autoregressive generation loop
        if verbose:
            print(f"\n[3/4] Generating mel frames (autoregressive)...")
            print(f"      Max steps: {self.max_decoder_steps}")
            print(f"      Reduction factor: {self.reduction_factor}")

        refined_mel = self._generate_mel_frames(encoder_hidden_states, speaker_embeddings, batch_size, verbose)

        # Mel frames are already refined by post-net in the loop
        if verbose:
            print(f"      ✓ Final mel shape: {refined_mel.shape}")
            print(f"      ✓ Total frames: {refined_mel.shape[1]}")
            print(f"      ✓ Estimated duration: {refined_mel.shape[1] * 0.0125:.2f} seconds")
            print(f"\n{'='*80}")
            print(f"✓ GENERATION COMPLETE")
            print(f"{'='*80}\n")

        return refined_mel

    def _encode(self, input_ids) -> ttnn.Tensor:
        """Run encoder on input IDs (accepts torch.Tensor or ttnn.Tensor)."""
        # Convert to TTNN if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        else:
            input_ids_ttnn = input_ids

        # Run encoder
        encoder_output = self.encoder(input_ids_ttnn)

        # Extract hidden states (encoder returns tuple)
        if isinstance(encoder_output, tuple):
            encoder_hidden_states = encoder_output[0]
        else:
            encoder_hidden_states = encoder_output

        return encoder_hidden_states

    def _generate_mel_frames(
        self,
        encoder_hidden_states: ttnn.Tensor,
        speaker_embeddings,
        batch_size: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Generate mel frames autoregressively (accepts torch.Tensor or ttnn.Tensor for speaker_embeddings)."""

        # Initialize decoder input (start with zeros)
        decoder_input = self._get_initial_decoder_input(batch_size)

        # Convert speaker embeddings to TTNN if provided and needed
        speaker_embeddings_ttnn = None
        if speaker_embeddings is not None:
            if isinstance(speaker_embeddings, torch.Tensor):
                # Ensure correct shape [batch, 1, dim] or [batch, dim]
                if speaker_embeddings.ndim == 2:
                    speaker_embeddings = speaker_embeddings.unsqueeze(1)

                speaker_embeddings_ttnn = ttnn.from_torch(
                    speaker_embeddings,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
            else:
                # Already TTNN tensor
                speaker_embeddings_ttnn = speaker_embeddings

        mel_frames_list = []

        for step in range(self.max_decoder_steps):
            # Run decoder for one step - outputs hidden states [B, T, 768]
            decoder_hidden_states = self.decoder(
                decoder_input,
                encoder_hidden_states,
                speaker_embeddings_ttnn,
            )

            # Post-net: hidden states → mel frames
            # Returns: (mel_before_postnet, mel_after_postnet, stop_logits)
            _, mel_refined, stop_logits_ttnn = self.postnet(decoder_hidden_states)

            # Convert outputs to PyTorch
            mel_refined_torch = ttnn.to_torch(mel_refined)
            stop_logits_torch = ttnn.to_torch(stop_logits_ttnn)

            # Store refined mel frames (already processed by post-net)
            mel_frames_list.append(mel_refined_torch)

            # Check stop condition using stop token prediction
            # stop_logits shape: [batch, seq_len * reduction_factor]
            # HF checks the LAST frame's stop probability (not all frames)
            stop_probs = torch.sigmoid(stop_logits_torch)
            last_frame_stop_prob = stop_probs[0, -1]  # Check last frame only
            should_stop = last_frame_stop_prob > self.stop_threshold

            if should_stop:
                if verbose:
                    print(f"      ✓ Stop token detected at step {step + 1} (prob: {last_frame_stop_prob:.4f})")
                break

            # Check max steps as fallback
            if step >= self.max_decoder_steps - 1:
                if verbose:
                    print(f"      ✓ Reached max steps ({self.max_decoder_steps})")
                break

            # Update decoder input for next step
            # Use the last reduction_factor mel frames as input
            decoder_input = ttnn.from_torch(
                mel_refined_torch[:, -self.reduction_factor :, :],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            if verbose and (step + 1) % 20 == 0:
                frames_generated = (step + 1) * self.reduction_factor
                print(f"      Step {step + 1}/{self.max_decoder_steps} - Generated {frames_generated} frames")

        # Concatenate all mel frames
        all_mel_frames = torch.cat(mel_frames_list, dim=1)

        if verbose:
            total_steps = len(mel_frames_list)
            total_frames = all_mel_frames.shape[1]
            print(f"      ✓ Generated {total_steps} decoder steps")
            print(f"      ✓ Total mel frames: {total_frames}")

        return all_mel_frames

    def _get_initial_decoder_input(self, batch_size: int) -> ttnn.Tensor:
        """Get initial decoder input (zeros for cold start)."""
        # Start with reduction_factor frames of zeros
        initial_input = torch.zeros(batch_size, self.reduction_factor, self.num_mel_bins)

        # Convert to TTNN
        initial_input_ttnn = ttnn.from_torch(
            initial_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        return initial_input_ttnn

    def _postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply post-net to convert hidden states to refined mel spectrogram."""
        # Convert to TTNN
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Run post-net: hidden states → mel frames
        # Returns: (mel_before_postnet, mel_after_postnet, stop_logits)
        _, refined_mel_ttnn, _ = self.postnet(hidden_states_ttnn)

        # Convert back to PyTorch
        refined_mel = ttnn.to_torch(refined_mel_ttnn)

        return refined_mel
