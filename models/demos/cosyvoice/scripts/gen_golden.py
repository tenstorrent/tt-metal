# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Capture PyTorch reference goldens for every CosyVoice-300M submodule.

Runs the *unmodified* reference pipeline once with a pinned RNG and records the
real tensors crossing each module boundary, rather than reconstructing plausible
inputs by hand. The .npz files it writes are what `tests/pcc/` compares TTNN
against, so they are the definition of correct for this bring-up.

RUN THIS IN THE CosyVoice VENV, NOT THE TT-METAL ONE:

    PYTHONPATH=$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS \
    $COSYVOICE_ENV/bin/python gen_golden.py --out <dir>

Why the reference is not deterministic, and what this script does about it
-------------------------------------------------------------------------
Three modules draw from the global RNG *inside* their forward pass:

  * ConditionalCFM.forward   z = torch.randn_like(mu)          (the flow prior)
  * SineGen.forward          phase_vec ~ U(-pi, pi), plus randn noise
  * SourceModuleHnNSF.forward  noise = randn_like(uv)

A TTNN port cannot consume the torch RNG stream in the same order, so seeding
alone does not make TTNN and PyTorch comparable -- it only makes PyTorch
reproducible against itself. Every such draw is therefore captured here as a
named array, and the TTNN modules must accept them as explicit inputs in PCC
tests. This generalises the DD-5 lesson (pin the DDIM noise) into the stronger
rule the vocoder needs: capture the noise, then inject it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.distributions.uniform import Uniform

DEFAULT_COSYVOICE = os.environ.get("COSYVOICE_REPO", "/mnt/CosyVoice")
SEED = 1986  # the seed cosyvoice.yaml itself sets, kept for continuity


# --------------------------------------------------------------------------
# capture plumbing
# --------------------------------------------------------------------------
# Modules whose per-call tensors are huge. The AR decoder's att_cache is
# [n_layers, n_heads, T, 2*head_dim] = 24 MB at T=209, and it is re-fed every
# step, so capturing all of them costs ~430 MB for one short utterance.
# Two calls is prefill + one decode step, which is what a KV-cache PCC test needs.
CALL_CAPS = {"llm.ar_forward_chunk": 2}

# Float arrays above this many elements are stored fp16. That is 11 mantissa
# bits -- strictly more precision than the bfloat16 the device will carry, and
# three orders of magnitude finer than the PCC >= 0.99 gates consume.
LARGE_ARRAY = 1 << 20


class Recorder:
    """Collects named tensor bundles, one bundle per module invocation."""

    def __init__(self, max_calls_per_key: int = 4, compact: bool = True):
        self.store: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
        self.max_calls = max_calls_per_key
        self.compact = compact
        self.handles: list = []
        self.patched: list[tuple[object, str, object]] = []

    def _cap(self, key: str) -> int:
        return min(self.max_calls, CALL_CAPS.get(key, self.max_calls))

    # -- recording -------------------------------------------------------
    def add(self, key: str, **arrays) -> None:
        if len(self.store[key]) >= self._cap(key):
            return
        bundle = {}
        for k, v in arrays.items():
            arr = _np(v)
            if arr is None:
                continue
            if self.compact and arr.dtype == np.float32 and arr.size > LARGE_ARRAY:
                arr = arr.astype(np.float16)
            bundle[k] = arr
        self.store[key].append(bundle)

    # -- nn.Module forward hooks -----------------------------------------
    def hook_module(self, key: str, module: torch.nn.Module, in_names, out_names) -> None:
        def _hook(_mod, args, output):
            bundle = {}
            for i, name in enumerate(in_names):
                if i < len(args):
                    bundle[f"in_{name}"] = args[i]
            outs = output if isinstance(output, tuple) else (output,)
            for i, name in enumerate(out_names):
                if i < len(outs):
                    bundle[f"out_{name}"] = outs[i]
            self.add(key, **bundle)

        self.handles.append(module.register_forward_hook(_hook, with_kwargs=False))

    # -- method-level boundaries (not nn.Modules) -------------------------
    def patch(self, key, obj, meth, in_names, out_names):
        """Wrap a bound method so its args and returns are recorded."""
        original = getattr(obj, meth)

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            bundle = {}
            for i, name in enumerate(in_names):
                if i < len(args):
                    bundle[f"in_{name}"] = args[i]
                elif name in kwargs:
                    bundle[f"in_{name}"] = kwargs[name]
            outs = out if isinstance(out, tuple) else (out,)
            for i, name in enumerate(out_names):
                if i < len(outs):
                    bundle[f"out_{name}"] = outs[i]
            self.add(key, **bundle)
            return out

        setattr(obj, meth, wrapper)
        self.patched.append((obj, meth, original))

    def close(self):
        for h in self.handles:
            h.remove()
        for obj, meth, original in self.patched:
            setattr(obj, meth, original)
        self.handles, self.patched = [], []

    # -- output ----------------------------------------------------------
    def save(self, out_dir: str) -> dict[str, dict]:
        """Write one .npz per module, deduplicating byte-identical arrays.

        Step N's input KV cache is the *same tensor* step N-1 returned, so a
        content hash collapses the dominant cost with no loss of fidelity. The
        alias map travels inside the .npz under `__aliases__`; load_golden()
        resolves it so callers never see the difference.
        """
        import hashlib

        os.makedirs(out_dir, exist_ok=True)
        manifest = {}
        for key, calls in sorted(self.store.items()):
            flat, shapes, aliases, by_hash = {}, {}, {}, {}
            for ci, bundle in enumerate(calls):
                for name, arr in bundle.items():
                    full = f"call{ci}.{name}"
                    shapes[full] = [list(arr.shape), str(arr.dtype)]
                    if arr.size:
                        h = hashlib.blake2b(
                            np.ascontiguousarray(arr).tobytes() + str(arr.dtype).encode(), digest_size=16
                        ).hexdigest()
                        if h in by_hash:
                            aliases[full] = by_hash[h]
                            continue
                        by_hash[h] = full
                    flat[full] = arr
            if aliases:
                flat["__aliases__"] = np.frombuffer(json.dumps(aliases).encode(), dtype=np.uint8)
            path = os.path.join(out_dir, f"{key}.npz")
            np.savez_compressed(path, **flat)
            size = os.path.getsize(path)
            manifest[key] = {
                "file": os.path.basename(path),
                "calls": len(calls),
                "bytes": size,
                "aliases": aliases,
                "arrays": shapes,
            }
            dedup = f"  ({len(aliases)} aliased)" if aliases else ""
            print(f"  {key:<28} {len(calls)} call(s)  {size/1e6:7.2f} MB{dedup}")
        return manifest


def load_golden(path: str) -> dict[str, np.ndarray]:
    """Read a golden .npz, resolving the alias map save() may have written.

    Tests should use this rather than np.load, otherwise deduplicated arrays
    (e.g. call1.in_att_cache, which aliases call0.out_att_cache) look missing.
    """
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != "__aliases__"}
        if "__aliases__" in z.files:
            for alias, target in json.loads(bytes(z["__aliases__"]).decode()).items():
                data[alias] = data[target]
    return data


def _np(x):
    if isinstance(x, torch.Tensor):
        # clone() lifts the tensor out of inference_mode, which several of these
        # boundaries run under; numpy() on an inference tensor is a trap.
        t = x.detach().clone().cpu()
        # bool/int survive as-is; everything float becomes fp32 so the golden is
        # exact and the TTNN side decides its own dtype.
        if t.dtype in (torch.float16, torch.bfloat16, torch.float64):
            t = t.float()
        return t.numpy()
    if isinstance(x, (int, float, bool)):
        return np.asarray(x)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
        return np.asarray(x)
    return None


# --------------------------------------------------------------------------
# RNG interception: capture the draws the reference makes internally
# --------------------------------------------------------------------------
def install_rng_capture(rec: Recorder, cosyvoice_pkg) -> None:
    """Record z / phase_vec / noise so PCC tests can inject them into TTNN."""
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.hifigan.generator import SineGen, SourceModuleHnNSF

    # --- ConditionalCFM: the flow prior z ---------------------------------
    cfm_forward = ConditionalCFM.forward

    def cfm_wrapper(
        self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)
    ):
        # Re-derive z exactly as the original does, record it, then hand the
        # same tensor to the original by seeding a fork so the draw repeats.
        state = torch.random.get_rng_state()
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        rec.add(
            "flow.cfm",
            in_mu=mu,
            in_mask=mask,
            in_spks=spks,
            in_cond=cond,
            in_n_timesteps=n_timesteps,
            in_prompt_len=prompt_len,
            rng_z=z,
        )
        torch.random.set_rng_state(state)  # rewind so the original draws the same z
        return cfm_forward(self, mu, mask, n_timesteps, temperature, spks, cond, prompt_len, cache)

    ConditionalCFM.forward = cfm_wrapper

    # --- SineGen: per-harmonic phase offset and additive noise ------------
    sine_forward = SineGen.forward

    def sine_wrapper(self, f0):
        state = torch.random.get_rng_state()
        out = sine_forward(self, f0)

        # `phase_vec` never leaves SineGen.forward, but the device needs it: without
        # it the harmonic bank is phase-shifted per harmonic and no amount of
        # correct arithmetic reproduces the reference. It is the FIRST draw the
        # forward pass makes (Uniform, before the randn for noise), so rewinding
        # and replaying that one sample recovers exactly the value used.
        torch.random.set_rng_state(state)
        phase_vec = Uniform(low=-np.pi, high=np.pi).sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1))
        phase_vec[:, 0, :] = 0  # the fundamental is unshifted, as upstream

        torch.random.set_rng_state(state)
        out2 = sine_forward(self, f0)  # identical draw, used to expose internals
        rec.add(
            "hift.sinegen",
            in_f0=f0,
            # transposed to the [B, 1, H+1] the channels-last device layout wants
            in_phase_vec=phase_vec.transpose(1, 2),
            out_sine=out2[0],
            out_uv=out2[1],
            out_noise=out2[2],
        )
        return out

    SineGen.forward = sine_wrapper

    # --- SourceModuleHnNSF: the noise branch ------------------------------
    src_forward = SourceModuleHnNSF.forward

    def src_wrapper(self, x):
        sine_merge, noise, uv = src_forward(self, x)
        rec.add("hift.m_source", in_f0_upsampled=x, out_sine_merge=sine_merge, out_noise=noise, out_uv=uv)
        return sine_merge, noise, uv

    SourceModuleHnNSF.forward = src_wrapper

    return (ConditionalCFM, cfm_forward), (SineGen, sine_forward), (SourceModuleHnNSF, src_forward)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_recorder(model, rec: Recorder) -> None:
    llm, flow, hift = model.llm, model.flow, model.hift

    # ---- LLM ------------------------------------------------------------
    rec.hook_module("llm.text_encoder", llm.text_encoder, ["xs", "xs_lens"], ["xs", "masks"])
    rec.hook_module("llm.text_encoder_affine", llm.text_encoder_affine_layer, ["x"], ["y"])
    rec.hook_module("llm.spk_embed_affine", llm.spk_embed_affine_layer, ["x"], ["y"])
    rec.hook_module("llm.decoder_head", llm.llm_decoder, ["x"], ["logits"])
    # AR decoder: forward_chunk carries the KV cache, so patch the method
    rec.patch(
        "llm.ar_forward_chunk",
        llm.llm,
        "forward_chunk",
        ["xs", "offset", "required_cache_size", "att_cache", "cnn_cache", "att_mask"],
        ["ys", "att_cache", "cnn_cache"],
    )

    # ---- Flow -----------------------------------------------------------
    rec.hook_module("flow.spk_embed_affine", flow.spk_embed_affine_layer, ["x"], ["y"])
    rec.hook_module("flow.input_embedding", flow.input_embedding, ["tokens"], ["emb"])
    rec.hook_module("flow.encoder", flow.encoder, ["xs", "xs_lens"], ["xs", "masks"])
    # One RelPositionMultiHeadedAttention layer, with its weights. ESPnet rel-pos
    # attention is NOT standard SDPA (02_plan.md sec.3.3) and is the second-hardest
    # risk in the bring-up, so it gets a golden of its own rather than being
    # validated only through the encoder it sits inside.
    attn = flow.encoder.encoders[0].self_attn
    rec.patch(
        "flow.rel_pos_attention",
        attn,
        "forward",
        ["query", "key", "value", "mask", "pos_emb", "cache"],
        ["out", "new_cache"],
    )
    rec.add(
        "flow.rel_pos_attention_weights",
        w_query=attn.linear_q.weight,
        b_query=attn.linear_q.bias,
        w_key=attn.linear_k.weight,
        b_key=attn.linear_k.bias,
        w_value=attn.linear_v.weight,
        b_value=attn.linear_v.bias,
        w_out=attn.linear_out.weight,
        b_out=attn.linear_out.bias,
        w_pos=attn.linear_pos.weight,
        pos_bias_u=attn.pos_bias_u,
        pos_bias_v=attn.pos_bias_v,
        n_head=attn.h,
        d_k=attn.d_k,
    )
    rec.hook_module("flow.encoder_proj", flow.encoder_proj, ["x"], ["y"])
    rec.patch(
        "flow.length_regulator",
        flow.length_regulator,
        "inference",
        ["x1", "x2", "mel_len1", "mel_len2", "input_frame_rate"],
        ["y", "y_lens"],
    )
    rec.hook_module("flow.cfm_estimator", flow.decoder.estimator, ["x", "mask", "mu", "t", "spks", "cond"], ["dphi_dt"])
    rec.patch(
        "flow.solve_euler", flow.decoder, "solve_euler", ["x", "t_span", "mu", "mask", "spks", "cond"], ["sample"]
    )

    # ---- HiFT vocoder ---------------------------------------------------
    rec.hook_module("hift.f0_predictor", hift.f0_predictor, ["mel"], ["f0"])
    rec.patch("hift.decode", hift, "decode", ["x", "s"], ["speech"])
    rec.patch("hift.inference", hift, "inference", ["speech_feat", "cache_source"], ["speech", "source"])
    # THE iSTFT boundary -- the single most important golden in this file.
    # P1 of the plan reproduces exactly this mapping on silicon.
    rec.patch("hift.istft", hift, "_istft", ["magnitude", "phase"], ["waveform"])
    rec.patch("hift.stft", hift, "_stft", ["x"], ["real", "imag"])
    for i, rb in enumerate(hift.resblocks[:2]):
        rec.hook_module(f"hift.resblock{i}", rb, ["x"], ["y"])
    for i, up in enumerate(hift.ups):
        rec.hook_module(f"hift.upsample{i}", up, ["x"], ["y"])
    rec.hook_module("hift.conv_pre", hift.conv_pre, ["x"], ["y"])
    rec.hook_module("hift.conv_post", hift.conv_post, ["x"], ["y"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice-root", default=DEFAULT_COSYVOICE)
    ap.add_argument("--model-dir", default=None, help="default <root>/pretrained_models/CosyVoice-300M")
    ap.add_argument("--out", default=None, help="default <this file>/../tests/golden")
    ap.add_argument("--mode", default="zero_shot", choices=["zero_shot", "sft", "cross_lingual", "instruct"])
    ap.add_argument("--text", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--full-precision",
        action="store_true",
        help="keep every array fp32 (much larger; use when auditing " "the iSTFT identity at its full 1e-7 agreement)",
    )
    ap.add_argument(
        "--max-calls",
        type=int,
        default=10,
        help="cap on recorded invocations per module (AR decode runs hundreds). "
        "10 captures the whole 10-step Euler trajectory of the CFM solver.",
    )
    args = ap.parse_args()

    root = args.cosyvoice_root
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "third_party", "Matcha-TTS"))
    model_dir = args.model_dir or os.path.join(root, "pretrained_models", "CosyVoice-300M")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    out_dir = os.path.abspath(out_dir)

    import cosyvoice  # noqa: F401
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.common import set_all_random_seed

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    print(f"loading {model_dir} ...")
    t0 = time.time()
    cv = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False)
    print(f"loaded in {time.time()-t0:.1f}s")

    rec = Recorder(max_calls_per_key=args.max_calls, compact=not args.full_precision)
    install_rng_capture(rec, cosyvoice)
    build_recorder(cv.model, rec)

    text = args.text or "希望你以后能够做的比我还好呦。"
    prompt_wav = os.path.join(root, "asset", "zero_shot_prompt.wav")
    prompt_text = "希望你以后能够做的比我还好呦。"

    set_all_random_seed(args.seed)
    t0 = time.time()
    chunks = []
    if args.mode == "zero_shot":
        gen = cv.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)
    elif args.mode == "cross_lingual":
        gen = cv.inference_cross_lingual(text, os.path.join(root, "asset", "cross_lingual_prompt.wav"), stream=False)
    elif args.mode == "sft":
        gen = cv.inference_sft(text, cv.list_available_spks()[0], stream=False)
    else:
        gen = cv.inference_instruct(
            text, cv.list_available_spks()[0], "Theo 'Crimson', is a fiery, passionate rebel " "leader.<|endofprompt|>"
        )
    for out in gen:
        chunks.append(out["tts_speech"])
    elapsed = time.time() - t0
    rec.close()

    wav = torch.concat(chunks, dim=1)
    dur = wav.shape[1] / cv.sample_rate
    print(f"\nsynthesised {dur:.2f}s in {elapsed:.1f}s  (RTF {elapsed/dur:.2f})")

    print(f"\nwriting goldens to {out_dir}")
    manifest = rec.save(out_dir)

    np.savez_compressed(os.path.join(out_dir, "e2e.npz"), waveform=wav.detach().cpu().numpy())
    meta = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cosyvoice_commit": _git_head(root),
        "model_dir": model_dir,
        "mode": args.mode,
        "text": text,
        "seed": args.seed,
        "torch": torch.__version__,
        "sample_rate": cv.sample_rate,
        "audio_seconds": dur,
        "cpu_seconds": elapsed,
        "rtf": elapsed / dur,
        "modules": manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"\nmanifest -> {os.path.join(out_dir, 'manifest.json')}")
    return 0


def _git_head(path: str) -> str:
    """The checked-out commit of the CosyVoice tree, read straight off disk.

    This deliberately does not shell out to `git`. `path` arrives from `--cosyvoice-root`
    or `$COSYVOICE_REPO`, and `git -C <path> ...` would parse a value beginning with `-`
    as an option rather than a directory -- argument injection, which the argv-list form
    does nothing to prevent (it only rules out *shell* metacharacters). Reading `.git`
    directly removes the process spawn, so there is no argument vector at all.

    Handles the two HEAD forms and the `gitdir:` indirection a submodule checkout uses.
    """
    try:
        dot = os.path.join(path, ".git")
        if os.path.isfile(dot):  # submodule / worktree: "gitdir: <path>"
            with open(dot) as fh:
                dot = os.path.join(path, fh.read().split(":", 1)[1].strip())
        with open(os.path.join(dot, "HEAD")) as fh:
            head = fh.read().strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD -- the raw SHA, which is how CosyVoice is pinned
        ref = head.split(":", 1)[1].strip()
        loose = os.path.join(dot, *ref.split("/"))
        if os.path.isfile(loose):
            with open(loose) as fh:
                return fh.read().strip()
        with open(os.path.join(dot, "packed-refs")) as fh:  # ref was packed away by gc
            for line in fh:
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha.strip()
        return "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
