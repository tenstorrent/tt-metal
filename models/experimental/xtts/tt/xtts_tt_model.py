"""End-to-end XTTS-v2 pipeline driven by the ported TTNN modules.

`XttsTTModel` initializes the four ported TTNN modules and runs the full demo
pipeline (text + speaker reference wav -> waveform), executing the heavy compute
on the Tenstorrent device:

    ported to TTNN (used in the forward path):
      * conditioning encoder + perceiver   -> gpt_cond_latent
      * GPT2 decoder stack (+ ln_f)         -> continuous GPT latents
      * HiFiGAN vocoder (waveform_decoder)  -> waveform

    still on host (not yet ported; reused from the reference model):
      * VoiceBpeTokenizer, mel-spectrogram front-ends
      * GPT autoregressive `generate` loop (produces discrete mel codes)
      * ResNet speaker encoder (Phase 4)

It does this by constructing the reference Xtts once and temporarily patching
the three module call-sites to route through TTNN, then calling the reference's
own `get_conditioning_latents` / `inference` so all the un-ported glue is reused.

Device must be opened with l1_small_size>0 (vocoder conv ops need it):
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
"""

import contextlib
import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
import gpt2_block as GB  # noqa: E402
import conditioning_encoder as CE  # noqa: E402
import perceiver_resampler as PR  # noqa: E402
import hifigan_generator as HG  # noqa: E402
from model_config import load_reference_state_dict  # noqa: E402

N_LAYERS = 30


class XttsTTModel:
    def __init__(self, device, model=None):
        """device: an opened ttnn device (l1_small_size>0).
        model: a reference Xtts (loaded if None)."""
        self.device = device
        if model is None:
            model, _ = load_reference_state_dict("cpu")
        self.model = model
        sd = model.state_dict()

        # --- initialize the four ported TTNN modules ---
        self.gpt_layers = GB.load_stack_params(sd, device, N_LAYERS)
        self.ln_f_w = GB._to_device(sd["gpt.gpt.ln_f.weight"], device)
        self.ln_f_b = GB._to_device(sd["gpt.gpt.ln_f.bias"], device)
        self.enc_params = CE.load_encoder_params(sd, device)
        self.perceiver_params = PR.load_perceiver_params(sd, device)
        self.vocoder_params = HG.load_generator_params(model.hifigan_decoder.waveform_decoder, device)

    # ------------------------------------------------------------------ #
    # TTNN implementations of the patched reference call-sites
    # ------------------------------------------------------------------ #
    def _tt_get_style_emb(self, cond_input, return_latent=False):
        """Reference gpt.get_style_emb: mel (b,80,s) -> cond latents (b,1024,32)."""
        if return_latent:
            return cond_input.unsqueeze(1)
        if cond_input.ndim == 4:
            cond_input = cond_input.squeeze(1)
        mel = ttnn.from_torch(
            cond_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16
        )
        conds = CE.conditioning_encoder(mel, self.enc_params)  # [1, S, 1024]
        latents = PR.perceiver_resampler(conds, self.perceiver_params)  # [1, 32, 1024]
        out = ttnn.to_torch(latents).float().transpose(1, 2)  # [1, 1024, 32]
        return out.to(cond_input.dtype)

    def _tt_gpt_forward(self, inputs_embeds=None, return_dict=True, **kwargs):
        """Reference gpt.gpt (GPT2Model) forward: emb -> last_hidden_state via the
        TTNN 30-layer stack + ln_f. (Causal; assumes the demo's unpadded sequence.)

        Returns a HF ModelOutput so both `.last_hidden_state` (get_logits) and
        `[0]` indexing work."""
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        emb = inputs_embeds
        x = ttnn.from_torch(emb.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        h = GB.stack_prefill(x, self.gpt_layers)
        h = GB.layer_norm(h, self.ln_f_w, self.ln_f_b)
        lhs = ttnn.to_torch(h).float()[:, : emb.shape[1], :].to(emb.dtype)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=lhs)

    def _tt_waveform_decoder(self, z, g=None):
        """Reference waveform_decoder forward: GPT latents z + g -> wav.
        z may arrive 2D [C,T] (the decoder squeezes it before this call)."""
        z = z.float().cpu()
        if z.dim() == 2:
            z = z.unsqueeze(0)  # [1, 1024, T]
        g = g.float().cpu().reshape(1, -1, 1)  # [1, 512, 1]
        wav = HG.hifigan_generator(z, g, self.vocoder_params, self.device)
        return wav.to(z.dtype)

    # ------------------------------------------------------------------ #
    # patching + the full pipeline
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def _route_through_tt(self):
        m = self.model
        gpt, wd = m.gpt, m.hifigan_decoder.waveform_decoder
        orig_style, orig_wrapper, orig_inner, orig_wd = (gpt.get_style_emb, gpt.forward, gpt.gpt.forward, wd.forward)

        def tt_gpt_wrapper(*args, **kwargs):
            # Run only the latent-extraction forward through TT. gpt.gpt is shared
            # with the AR `generate` path (gpt_inference), so swap it in ONLY for
            # the duration of this wrapper call, then restore.
            gpt.gpt.forward = self._tt_gpt_forward
            try:
                return orig_wrapper(*args, **kwargs)
            finally:
                gpt.gpt.forward = orig_inner

        gpt.get_style_emb = self._tt_get_style_emb
        gpt.forward = tt_gpt_wrapper
        wd.forward = self._tt_waveform_decoder
        try:
            yield
        finally:
            gpt.get_style_emb, gpt.forward, wd.forward = orig_style, orig_wrapper, orig_wd
            gpt.gpt.forward = orig_inner

    @torch.inference_mode()
    def forward(self, text, speaker_wav, language="en"):
        """Full demo pipeline: returns the synthesized waveform (numpy, 24 kHz)."""
        with self._route_through_tt():
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav)
            out = self.model.inference(text, language, gpt_cond_latent, speaker_embedding)
        return out["wav"]

    __call__ = forward


def main():
    import argparse

    import numpy as np
    import soundfile as sf

    parser = argparse.ArgumentParser(description="XTTS-v2 TTNN pipeline demo")
    parser.add_argument("--text", default="It took me quite a long time to develop a voice.")
    parser.add_argument("--speaker_wav", required=True, help="reference speaker clip (wav)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default="tt_pipeline_out.wav")
    args = parser.parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=32768)  # conv ops need L1-small
    try:
        model = XttsTTModel(device)
        wav = np.asarray(model.forward(args.text, args.speaker_wav, args.language)).reshape(-1)
        sf.write(args.output, wav, 24000)
        print(f"wrote {args.output}  ({len(wav) / 24000:.2f}s, 24 kHz)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
