#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 0.7 — Generate golden reference fixtures for CosyVoice

Runs the CosyVoice reference pipeline on CPU with:
  - seed = 1986 (the 4 global seed calls the yaml itself makes: random,
    numpy, torch, torch.cuda.all — replicated here for determinism), AND
  - RAS (repetition-aware sampling) with the fixed seed for deterministic
    token sequences. Pure greedy (argmax) causes degenerate period-2 loops
    in this model; RAS with a seeded torch RNG is fully reproducible.

Produces, per mode, under `model_data/golden/`:
  wav/<mode>_<i>.wav            — generated audio (torchaudio, 24 kHz)
  llm/<mode>.pt                 — lm_input embeddings, per-step logp, token seq
  flow/<mode>.pt                — encoder mu, mask, spks, cond, per-step dphi_dt, final mel
  hift/<mode>.pt                — input mel, f0, source s, output waveform

Instrumentation is via class-level monkey-patches on the three on-device
components' inference methods (no source edits to the reference repo). Run from
the demo root or pass --demo-root:

    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal/models/demos/cosyvoice
    PYTHONPATH=model_data/CosyVoice_src:model_data/CosyVoice_src/third_party/Matcha-TTS \
        python scripts/gen_golden.py --modes zero_shot

CPU runtime is ~minutes per mode (Qwen2.5-0.5B on CPU). Start with
`--modes zero_shot` to prove the harness, then run all 4.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
MATCHA = CV_SRC / "third_party" / "Matcha-TTS"
ASSET = CV_SRC / "asset"
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden"
SPK2INFO = CKPT_DIR / "spk2info.pt"

SEED = 1986


def _setup_imports():
    """Put CosyVoice + Matcha on PYTHONPATH and chdir to the repo root.

    `example.py` uses relative `./asset/*.wav` paths, so the process must run
    from `CosyVoice_src/`. `AutoModel(model_dir=...)` accepts an absolute path,
    sidestepping the relative `pretrained_models/CosyVoice2-0.5B` assumption.
    """
    for p in (str(CV_SRC), str(MATCHA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(CV_SRC)


def _stub_pyworld():
    """Stub `pyworld` so the yaml's training-only `!name:` processor tags load.

    `cosyvoice/dataset/processor.py` does `import pyworld as pw` at module top,
    and the yaml binds `parquet_opener`/`compute_f0`/etc. via `!name:` which
    imports that module during `CosyVoice2.__init__`. pyworld is excluded from
    `requirements-cosyvoice.txt` (training-only); inference never calls these
    processors, so a no-op stub is sufficient and policy-compliant.
    """
    if "pyworld" not in sys.modules:
        stub = types.ModuleType("pyworld")
        _noop = lambda *a, **k: None
        for n in (
            "wave_to_world",
            "world_to_wave",
            "pythonworld",
            "dio",
            "stft",
            "harvest",
            "cheaptrick",
            "d4c",
            "star",
            "vocoder",
        ):
            setattr(stub, n, _noop)
        sys.modules["pyworld"] = stub


def _patch_load_wav():
    """Work around a torch 2.11 / torchaudio incompatibility.

    The reference `cosyvoice.utils.file_utils.load_wav` calls
    `torchaudio.load(wav, backend='soundfile')`. Under torchaudio shipping with
    torch 2.11, `load()` routes through `load_with_torchcodec` which requires the
    uninstalled `torchcodec` package (it ignores the explicit `soundfile`
    backend arg). `soundfile` 0.14.0 IS installed, so reimplement `load_wav` to
    read via `soundfile.read` + `torchaudio.transforms.Resample` — identical
    behavior (mono mean, resample to target_sr) without the broken code path.
    Installing `torchcodec` is an alternative but is a new package; this patch
    keeps the env untouched.
    """
    import cosyvoice.utils.file_utils as fu
    import soundfile
    import torchaudio

    def load_wav(wav, target_sr, min_sr=16000):
        data, sample_rate = soundfile.read(str(wav), dtype="float32")
        # data: (samples,) or (samples, channels) -> (1, samples) mono
        t = torch.from_numpy(data)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t().mean(dim=0, keepdim=True)
        speech = t
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, f"wav sample rate {sample_rate} must be >= {min_sr}"
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    fu.load_wav = load_wav


def _patch_qwen2_encoder():
    """Fix two transformers 5.10 incompatibilities in Qwen2Encoder.

    1. Force attn_implementation='eager'. Transformers 5.10 defaults to SDPA,
       which mishandles CosyVoice's custom 1D attention mask (masks[:, -1, :]).
       SDPA applies its own causal masking on top, producing divergent outputs
       (max diff ~124 vs eager on a 10-token input).

    2. Fix decode-step attention mask. CosyVoice passes attention_mask=[1,1]
       during single-token decode, but transformers 5.x requires the mask to
       cover the full KV-cache length (all cached positions + current token).
       Without this, decode tokens only attend to the last position → gibberish.
    """
    from cosyvoice.llm import llm as llm_module
    from transformers import Qwen2ForCausalLM

    _OrigQwen2Encoder = llm_module.Qwen2Encoder

    class Qwen2EncoderCompat(_OrigQwen2Encoder):
        def __init__(self, pretrain_path):
            torch.nn.Module.__init__(self)
            self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path, attn_implementation="eager")

        def forward_one_step(self, xs, masks, cache=None):
            input_masks = masks[:, -1, :]
            if cache is not None:
                cache_len = cache.get_seq_length()
                prefix = torch.ones(1, cache_len, dtype=torch.bool, device=xs.device)
                input_masks = torch.cat([prefix, input_masks], dim=1)
            outs = self.model(
                inputs_embeds=xs,
                attention_mask=input_masks,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=cache,
            )
            return outs.hidden_states[-1], outs.past_key_values

    llm_module.Qwen2Encoder = Qwen2EncoderCompat


def set_seed(seed: int = SEED):
    """Replicate `cosyvoice2.yaml` lines 1-5 (U2 confirmed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Capture state (module-level; the LLM decode runs in a worker thread, but the
# main thread joins before reading, so no locking is needed).
# ---------------------------------------------------------------------------
_CAPTURE: dict = {"mode": None, "llm": None, "flow": None, "hift": None}


def _new_capture():
    return {"lm_input": None, "logps": [], "tokens": []}


def _new_flow_capture():
    return {
        "mu": None,
        "mask": None,
        "spks": None,
        "cond": None,
        "dphi_dt": [],
        "mel": None,
        "x_init": None,
        "t_span": None,
        "token": None,
        "token_len": None,
        "prompt_token": None,
        "prompt_token_len": None,
        "embedding": None,
        "prompt_feat": None,
        "prompt_feat_len": None,
    }


def _new_hift_capture():
    return {"mel_in": None, "f0": None, "source": None, "waveform": None}


# ---------------------------------------------------------------------------
# Instrumentation: monkey-patch the three on-device inference paths.
# ---------------------------------------------------------------------------
_ORIG = {}


def install_instrumentation():
    """Patch LLM/flow/hift inference to capture fixtures.

    Sampling uses the ORIGINAL RAS (repetition-aware sampling) with the fixed
    seed (1986) for determinism. Pure greedy (argmax) causes degenerate period-2
    token loops in this model — it was trained with RAS and requires the
    repetition penalty window to produce coherent speech.
    """
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.hifigan.generator import HiFTGenerator
    from cosyvoice.llm.llm import Qwen2LM

    # ---- LLM: per-step logp/token capture (keep original RAS sampling) ----
    _ORIG["Qwen2LM.inference_wrapper"] = Qwen2LM.inference_wrapper

    _orig_inference_wrapper = Qwen2LM.inference_wrapper

    def inference_wrapper_capture(self, lm_input, sampling, min_len, max_len, uuid):
        cap = _CAPTURE["llm"]
        if cap is not None:
            cap["lm_input"] = lm_input.detach().cpu().clone()
            cap["rng_state"] = torch.random.get_rng_state()
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(
                    torch.bool
                ),
                cache=cache,
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=(i < min_len))
            if cap is not None:
                cap["logps"].append(logp.squeeze(0).detach().cpu().clone())
                cap["tokens"].append(int(top_ids))
            if top_ids in self.stop_token_ids:
                break
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    Qwen2LM.inference_wrapper = inference_wrapper_capture

    # ---- Flow: capture mu/mask/spks/cond + per-step dphi_dt + final mel ----
    _ORIG["flow.inference"] = CausalMaskedDiffWithXvec.inference
    _orig_flow_inference = CausalMaskedDiffWithXvec.inference

    def flow_inference_capture(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming,
        finalize,
    ):
        _CAPTURE["flow"] = _new_flow_capture()
        cap = _CAPTURE["flow"]
        if cap is not None:
            cap["token"] = token.detach().cpu().clone()
            cap["token_len"] = token_len.detach().cpu().clone()
            cap["prompt_token"] = prompt_token.detach().cpu().clone()
            cap["prompt_token_len"] = prompt_token_len.detach().cpu().clone()
            cap["embedding"] = embedding.detach().cpu().clone()
            cap["prompt_feat"] = prompt_feat.detach().cpu().clone()
            cap["prompt_feat_len"] = prompt_feat_len.detach().cpu().clone()
        return _orig_flow_inference(
            self,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming,
            finalize,
        )

    CausalMaskedDiffWithXvec.inference = flow_inference_capture

    _ORIG["CFM.solve_euler"] = ConditionalCFM.solve_euler
    _orig_solve_euler = ConditionalCFM.solve_euler

    def solve_euler_capture(self, x, t_span, mu, mask, spks, cond, streaming=False):
        cap = _CAPTURE["flow"]
        if cap is not None:
            cap["x_init"] = x.detach().cpu().clone()
            cap["t_span"] = t_span.detach().cpu().clone()
            cap["mu"] = mu.detach().cpu().clone()
            cap["mask"] = mask.detach().cpu().clone()
            cap["spks"] = spks.detach().cpu().clone()
            cap["cond"] = cond.detach().cpu().clone()
        # Wrap forward_estimator to grab per-step velocities.
        orig_fe = self.forward_estimator

        def fe_capture(x_, mask_, mu_, t_, spks_, cond_, streaming_=False):
            out = orig_fe(x_, mask_, mu_, t_, spks_, cond_, streaming=streaming_)
            if cap is not None:
                cap["dphi_dt"].append(out.detach().cpu().clone())
            return out

        self.forward_estimator = fe_capture
        try:
            result = _orig_solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=streaming)
        finally:
            self.forward_estimator = orig_fe
        if cap is not None:
            cap["mel"] = result.detach().cpu().clone()
        return result

    ConditionalCFM.solve_euler = solve_euler_capture

    # ---- HiFT: capture input mel, f0, source, waveform ----
    _ORIG["HiFT.inference"] = HiFTGenerator.inference
    _orig_hift_inference = HiFTGenerator.inference

    def hift_inference_capture(self, speech_feat, cache_source=torch.zeros(1, 1, 0)):
        # CV2's CosyVoice2Model.token2wav calls the BASE HiFTGenerator.inference
        # signature (cache_source=...), NOT CausalHiFTGenerator's finalize= form.
        _CAPTURE["hift"] = _new_hift_capture()
        cap = _CAPTURE["hift"]
        if cap is not None:
            cap["mel_in"] = speech_feat.detach().cpu().clone()
        generated_speech, s = _orig_hift_inference(self, speech_feat, cache_source=cache_source)
        if cap is not None:
            cap["waveform"] = generated_speech.detach().cpu().clone()
            cap["source"] = s.detach().cpu().clone()
            # f0 is computed inside the base inference but not returned;
            # recompute via the predictor for the Phase-2 f0_predictor PCC
            # fixture (U17). IMPORTANT: the BASE HiFTGenerator.inference (the CV2
            # path) computes f0 in the predictor's native dtype — it does NOT
            # cast to float64 (that's only CausalHiFTGenerator). Match that
            # exactly and do NOT mutate the predictor dtype (doing so would
            # break subsequent modes: base inference feeds float32 speech_feat
            # to a float64 predictor → dtype mismatch). Use the CPU-float32
            # snapshot (cap["mel_in"]) — the live speech_feat may have been
            # moved to a non-CPU device by the base inference.
            try:
                f0 = self.f0_predictor(cap["mel_in"].float())
                cap["f0"] = f0.detach().cpu().clone()
            except Exception as exc:
                cap["f0"] = None
        return generated_speech, s

    HiFTGenerator.inference = hift_inference_capture


# ---------------------------------------------------------------------------
# Per-mode text + prompt config (U5).
# ---------------------------------------------------------------------------
ZERO_SHOT_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福" "让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
CROSS_LINGUAL_TEXT = "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，" "因为他自己也被逗笑了[laughter]。"
INSTRUCT_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福" "让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
INSTRUCT_INSTRUCT = "用四川话说这句话<|endofprompt|>"
SFT_TEXT = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
SFT_SPK_ID = "my_zero_shot_spk"
PROMPT_WAV = "./asset/zero_shot_prompt.wav"


def run_zero_shot(cv, mode_dir):
    _CAPTURE["llm"] = _new_capture()
    outputs = []
    for i, j in enumerate(cv.inference_zero_shot(ZERO_SHOT_TEXT, ZERO_SHOT_PROMPT_TEXT, PROMPT_WAV, stream=False)):
        _save_wav(mode_dir, i, j["tts_speech"], cv.sample_rate)
        outputs.append(j["tts_speech"])
    return _finalize_llm_capture()


def run_cross_lingual(cv, mode_dir):
    _CAPTURE["llm"] = _new_capture()
    outputs = []
    for i, j in enumerate(cv.inference_cross_lingual(CROSS_LINGUAL_TEXT, PROMPT_WAV, stream=False)):
        _save_wav(mode_dir, i, j["tts_speech"], cv.sample_rate)
        outputs.append(j["tts_speech"])
    return _finalize_llm_capture()


def run_instruct2(cv, mode_dir):
    _CAPTURE["llm"] = _new_capture()
    outputs = []
    for i, j in enumerate(cv.inference_instruct2(INSTRUCT_TEXT, INSTRUCT_INSTRUCT, PROMPT_WAV, stream=False)):
        _save_wav(mode_dir, i, j["tts_speech"], cv.sample_rate)
        outputs.append(j["tts_speech"])
    return _finalize_llm_capture()


def run_sft(cv, mode_dir):
    # SFT = bootstrap a zero-shot speaker, keep it in-memory, then inference_sft.
    # We deliberately do NOT call save_spkinfo() (U3): keeping the checkpoint
    # dir pristine matters more than persisting the SFT speaker across runs.
    if SFT_SPK_ID not in cv.frontend.spk2info:
        assert cv.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, PROMPT_WAV, SFT_SPK_ID) is True
        # Reference-repo quirk (U5): `add_zero_shot_spk` stores llm_embedding/
        # flow_embedding, but `frontend_sft` reads a singular `embedding` key.
        # example.py never calls inference_sft, so the mismatch is latent.
        # Bridge it so the SFT path works:
        cv.frontend.spk2info[SFT_SPK_ID]["embedding"] = cv.frontend.spk2info[SFT_SPK_ID]["llm_embedding"]
    _CAPTURE["llm"] = _new_capture()
    outputs = []
    for i, j in enumerate(cv.inference_sft(SFT_TEXT, SFT_SPK_ID, stream=False)):
        _save_wav(mode_dir, i, j["tts_speech"], cv.sample_rate)
        outputs.append(j["tts_speech"])
    return _finalize_llm_capture()


MODES = {
    "zero_shot": run_zero_shot,
    "cross_lingual": run_cross_lingual,
    "instruct2": run_instruct2,
    "sft": run_sft,
}


def _save_wav(mode_dir, i, speech, sample_rate):
    # torchaudio.save also routes through torchcodec under torchaudio 2.11
    # (same incompatibility as load); write via soundfile directly.
    import soundfile

    wav_dir = GOLDEN_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    # speech: (1, samples) or (channels, samples) float32
    audio = speech.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio[0]  # mono
    soundfile.write(str(wav_dir / f"{mode_dir}_{i}.wav"), audio, sample_rate)


def _finalize_llm_capture():
    cap = _CAPTURE["llm"]
    if cap is None:
        return None
    cap["tokens"] = torch.tensor(cap["tokens"], dtype=torch.long)
    cap["logps"] = torch.stack(cap["logps"]) if cap["logps"] else torch.empty(0)
    return cap


def save_fixtures(mode):
    """Dump the captured LLM/flow/hift tensors to .pt fixtures."""
    for comp in ("llm", "flow", "hift"):
        cap = _CAPTURE.get(comp)
        if cap is None:
            continue
        out_dir = GOLDEN_DIR / comp
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cap, out_dir / f"{mode}.pt")
        print(f"  saved {comp}/{mode}.pt " f"(keys: {sorted(cap.keys()) if isinstance(cap, dict) else type(cap)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes", default="zero_shot", help="comma-separated subset of: " + ",".join(MODES) + " (default: zero_shot)"
    )
    args = ap.parse_args()

    _setup_imports()
    _stub_pyworld()
    _patch_load_wav()
    _patch_qwen2_encoder()
    set_seed(SEED)
    print(f"[gen_golden] seed={SEED}, RAS sampling (seeded), CWD={os.getcwd()}")
    print(f"[gen_golden] model_dir={CKPT_DIR}")

    spk2info_existed = SPK2INFO.exists()
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        install_instrumentation()
        cv = CosyVoice2(model_dir=str(CKPT_DIR))
        # CPU golden-gen dtype fix: the bundled Qwen2.5-0.5B (CosyVoice-BlankEN)
        # loads as bfloat16, but the CosyVoice-specific heads (speech_embedding,
        # llm_embedding) are float32, so the assembled lm_input is float32 and
        # the Qwen2 layers mismatch. On CPU there is no bf16 matmul benefit, so
        # cast the whole pipeline to float32 for deterministic golden fixtures.
        # (This is a golden-gen-only concern; the TTNN port uses bf16 on device.)
        cv.model.llm.float()
        cv.model.flow.float()
        cv.model.hift.float()

        wanted = [m.strip() for m in args.modes.split(",") if m.strip()]
        for m in wanted:
            if m not in MODES:
                print(f"[gen_golden] unknown mode {m!r}; skip", file=sys.stderr)
                continue
            print(f"\n[gen_golden] === mode: {m} ===")
            set_seed(SEED)  # re-seed per mode for intra-run determinism
            _CAPTURE["llm"] = _new_capture()
            _CAPTURE["flow"] = None
            _CAPTURE["hift"] = None
            MODES[m](cv, m)
            save_fixtures(m)
            llm_tok = _CAPTURE["llm"]["tokens"] if _CAPTURE["llm"] else None
            print(
                f"[gen_golden] {m}: generated " f"{(llm_tok.numel() if llm_tok is not None else 0)} speech " f"tokens"
            )
    finally:
        # U3 cleanup: remove spk2info.pt if this run created it, so the HF
        # checkpoint dir stays pristine (matches §1.1 "not shipped").
        if not spk2info_existed and SPK2INFO.exists():
            try:
                SPK2INFO.unlink()
                print("[gen_golden] removed generated spk2info.pt " "(checkpoint dir left pristine)")
            except OSError:
                pass

    print(f"\n[gen_golden] done. fixtures under {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
