#!/usr/bin/env python3
"""
Basic Vocoder for ChatTTS Semantic Tokens

Converts semantic speech tokens to audio waveforms using formant synthesis.
This is a simplified vocoder for demonstration purposes.
"""

import torch
import io
import soundfile as sf


class BasicVocoder:
    """
    Simple formant-based vocoder that converts semantic tokens to speech-like audio.

    This vocoder uses the semantic tokens to control:
    - Fundamental frequency (F0)
    - Formant frequencies
    - Amplitude envelope
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 256, n_formants: int = 3):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_formants = n_formants

        # Formant frequencies for different vowel sounds (Hz)
        self.formants = {
            "a": [700, 1200, 2500],  # /a/ as in "father"
            "e": [500, 1700, 2500],  # /e/ as in "bed"
            "i": [300, 2200, 2800],  # /i/ as in "machine"
            "o": [500, 800, 2500],  # /o/ as in "boat"
            "u": [300, 800, 2200],  # /u/ as in "boot"
        }

        # Default formants for unknown tokens
        self.default_formants = [500, 1500, 2500]

    def tokens_to_formants(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert semantic tokens to formant frequencies.

        Args:
            tokens: [batch_size, seq_len] token indices

        Returns:
            formants: [batch_size, seq_len, n_formants] formant frequencies in Hz
        """
        batch_size, seq_len = tokens.shape

        # Map tokens to formant sets (simplified: use token value modulo number of vowel types)
        formants_list = list(self.formants.values())
        n_vowel_types = len(formants_list)

        formants = torch.zeros(batch_size, seq_len, self.n_formants)

        for b in range(batch_size):
            for t in range(seq_len):
                token_idx = tokens[b, t].item()
                vowel_idx = token_idx % n_vowel_types
                formants[b, t] = torch.tensor(formants_list[vowel_idx])

        return formants

    def synthesize_formant_audio(
        self,
        formants: torch.Tensor,
        duration: float = 0.08,  # Duration per token in seconds
        f0_base: float = 100.0,  # Base fundamental frequency in Hz
    ) -> torch.Tensor:
        """
        Synthesize audio from formant frequencies using enhanced additive synthesis.

        Args:
            formants: [batch_size, seq_len, n_formants] formant frequencies
            duration: Duration per token in seconds
            f0_base: Base fundamental frequency

        Returns:
            audio: [batch_size, n_samples] synthesized audio
        """
        batch_size, seq_len, n_formants = formants.shape
        samples_per_token = int(duration * self.sample_rate)
        total_samples = seq_len * samples_per_token

        # Generate time axis
        t = torch.linspace(0, seq_len * duration, total_samples)

        # More natural fundamental frequency with prosody
        # Add some variation and slight upward inflection
        slow_modulation = torch.sin(2 * torch.pi * 0.3 * t) * 15  # Slow prosody changes
        fast_modulation = torch.sin(2 * torch.pi * 5.0 * t) * 5  # Natural jitter
        f0 = f0_base + slow_modulation + fast_modulation

        # Initialize audio signal
        audio = torch.zeros(total_samples)

        # Smooth transitions between tokens (crossfade)
        fade_samples = int(0.02 * self.sample_rate)  # 20ms crossfade

        # Additive synthesis with multiple harmonics and better envelopes
        for i in range(seq_len):
            start_sample = i * samples_per_token
            end_sample = (i + 1) * samples_per_token

            if end_sample > total_samples:
                end_sample = total_samples

            t_segment = t[start_sample:end_sample]
            segment_length = len(t_segment)

            if segment_length == 0:
                continue

            # Get formants for this token
            token_formants = formants[0, i]  # Use first batch

            # Create amplitude envelope with smoother attack/decay
            envelope = torch.ones(segment_length)

            # Attack (fade in)
            attack_samples = min(fade_samples, segment_length // 4)
            if attack_samples > 0:
                attack_env = torch.linspace(0, 1, attack_samples)
                envelope[:attack_samples] *= attack_env

            # Decay (fade out)
            decay_samples = min(fade_samples, segment_length // 4)
            if decay_samples > 0:
                decay_env = torch.linspace(1, 0, decay_samples)
                envelope[-decay_samples:] *= decay_env

            # Crossfade with previous token
            if i > 0 and start_sample > 0:
                crossfade_start = max(0, start_sample - fade_samples)
                crossfade_length = min(fade_samples, segment_length)
                if crossfade_length > 0:
                    crossfade_env = torch.linspace(0, 1, crossfade_length)
                    envelope[:crossfade_length] *= crossfade_env

            # Generate formant signals with harmonics
            for formant_freq in token_formants:
                if formant_freq > 0:
                    # Fundamental formant
                    formant_signal = torch.sin(2 * torch.pi * formant_freq * t_segment)

                    # Add some harmonics for richness (reduced amplitude)
                    harmonic2 = torch.sin(2 * torch.pi * (formant_freq * 2) * t_segment) * 0.3
                    harmonic3 = torch.sin(2 * torch.pi * (formant_freq * 3) * t_segment) * 0.1

                    # Combine harmonics
                    combined_signal = formant_signal + harmonic2 + harmonic3

                    # Apply envelope
                    audio[start_sample:end_sample] += combined_signal * envelope * 0.4

        # Add fundamental frequency component with natural voice characteristics
        fundamental = torch.sin(2 * torch.pi * f0 * t) * 0.08

        # Add some breathiness (noise component)
        noise = torch.randn(total_samples) * 0.02
        breathy_fundamental = fundamental + noise

        audio += breathy_fundamental

        # Add some spectral shaping (simple high-frequency rolloff)
        # This simulates the natural frequency response of speech
        if len(audio) > 100:
            # Simple low-pass filter effect
            alpha = 0.98
            filtered_audio = torch.zeros_like(audio)
            filtered_audio[0] = audio[0]
            for i in range(1, len(audio)):
                filtered_audio[i] = alpha * filtered_audio[i - 1] + (1 - alpha) * audio[i]
            audio = filtered_audio

        # Normalize to prevent clipping
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val * 0.8  # Leave some headroom

        return audio.unsqueeze(0)  # Add batch dimension

    def __call__(self, semantic_tokens: torch.Tensor, duration_per_token: float = 0.08) -> bytes:
        """
        Convert semantic tokens to audio bytes.

        Args:
            semantic_tokens: [batch_size, seq_len] token indices
            duration_per_token: Duration per token in seconds

        Returns:
            audio_bytes: WAV audio data as bytes
        """
        # Convert tokens to formants
        formants = self.tokens_to_formants(semantic_tokens)

        # Synthesize audio
        audio_waveform = self.synthesize_formant_audio(formants, duration=duration_per_token)

        # Convert to numpy and create WAV bytes
        audio_np = audio_waveform.squeeze(0).numpy()

        # Create WAV buffer
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np, self.sample_rate, format="WAV")
        wav_buffer.seek(0)

        return wav_buffer.getvalue()


def create_chattts_vocoder(sample_rate: int = 22050) -> BasicVocoder:
    """
    Create a vocoder instance for ChatTTS semantic tokens.

    Args:
        sample_rate: Audio sample rate in Hz

    Returns:
        BasicVocoder instance
    """
    return BasicVocoder(sample_rate=sample_rate, hop_length=256, n_formants=3)


# Test function
def test_vocoder():
    """Test the vocoder with sample tokens"""
    print("🎵 Testing Basic Vocoder...")

    vocoder = BasicVocoder()

    # Sample semantic tokens (simulating ChatTTS output)
    sample_tokens = torch.randint(0, 100, (1, 20))  # 20 tokens

    print(f"Input tokens shape: {sample_tokens.shape}")
    print(f"Sample tokens: {sample_tokens[0][:10].tolist()}")

    # Convert to audio
    audio_bytes = vocoder(sample_tokens)

    print(f"Generated audio size: {len(audio_bytes)} bytes")

    # Load and check audio properties
    with io.BytesIO(audio_bytes) as buf:
        audio_data, sample_rate = sf.read(buf)

    duration = len(audio_data) / sample_rate
    print(f"Audio duration: {duration:.2f}s")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Audio shape: {audio_data.shape}")

    print("✅ Vocoder test completed!")
    return audio_bytes


if __name__ == "__main__":
    test_vocoder()
