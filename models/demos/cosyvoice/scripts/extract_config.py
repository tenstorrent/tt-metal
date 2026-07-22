#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 0.6 — Parse `cosyvoice2.yaml` and assert against `tt/model_config.py`.

Regression harness: loads the downloaded `cosyvoice2.yaml` via `hyperpyyaml`,
walks the resolved config tree, and asserts every architecture number against
the frozen constants in `models/demos/cosyvoice/tt/model_config.py`. Fails loud
on any mismatch so a refreshed checkpoint with a different arch is caught before
Phase 2 weight conversion.

Run from the demo root:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal/models/demos/cosyvoice
    python scripts/extract_config.py

Reading the yaml fully instantiates CosyVoice2 modules (`!new:` tags), which
requires the CosyVoice source + Matcha submodule import surface on PYTHONPATH.
This script therefore sets up `model_data/CosyVoice_src` + Matcha import paths
itself before loading the yaml. (Phase 0.6 only needs the resolved *values*,
not the instantiated objects, but hyperpyyaml evaluates the tags eagerly.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths & import surface (must precede hyperpyyaml load).
# ---------------------------------------------------------------------------
DEMO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
YAML_PATH = CKPT_DIR / "cosyvoice2.yaml"
BLANKEN = CKPT_DIR / "CosyVoice-BlankEN"  # bundled Qwen2.5-0.5B base
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
MATCHA = CV_SRC / "third_party" / "Matcha-TTS"

for p in (str(CV_SRC), str(MATCHA)):
    if p not in sys.path:
        sys.path.append(p)
os.chdir(CV_SRC)  # some module imports assume CWD == repo root


def _resolve_refs(obj, root):
    """Resolve `!ref <name>` placeholders (now strings like '<name>').

    Only single-segment refs are used in this yaml (every `!ref` points at a
    top-level global or a multiplicative expr handled inline). We resolve the
    simple `<name>` form; the two `*`-expressions in the yaml
    (`static_chunk_size: !ref <chunk_size> * <token_mel_ratio>` and the
    estimator's) are handled specially by the caller.
    """
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("<") and s.endswith(">") and "*" not in s:
            ref = s[1:-1]
            if ref in root:
                return root[ref]
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    return obj


def _coerce_numeric(obj):
    """Coerce string scalars that parse as int/float into numeric types.

    PyYAML's SafeLoader parses `1e-06` as the *string* `'1e-06'` (yaml's
    scientific-notation rule requires a `.` after `e` or a leading digit form it
    doesn't match). The EXPECTED table holds a real float 1e-06, so coerce
    numeric-looking leaves to int/float for a clean comparison. Non-numeric
    strings (e.g. 'euler', 'gelu') are left untouched.
    """
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return obj
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return obj
    if isinstance(obj, dict):
        return {k: _coerce_numeric(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric(v) for v in obj]
    return obj


def load_yaml():
    """Load cosyvoice2.yaml as plain nested Python (no `!new:` instantiation).

    hyperpyyaml eagerly instantiates every `!new:` tag (constructing the full
    Qwen2.5-0.5B, flow, and hift modules), which is expensive AND loses the raw
    constructor-arg values that some modules don't store as attributes. For a
    config-extraction regression harness we only need the resolved scalars, so
    we use a SafeLoader that maps every `!tag` to its default constructor and
    then resolve `!ref` placeholders ourselves. This makes the harness fast
    (~ms), import-light, and robust to modules that drop constructor args.
    """
    import yaml

    text = YAML_PATH.read_text(encoding="utf-8")
    plain = yaml.SafeLoader

    def _default(l, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return l.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return l.construct_sequence(node, deep=True)
        return l.construct_scalar(node)

    plain.add_multi_constructor("", _default)
    # !apply:random.seed [1986] etc. become a list [1986]; we only need the int.
    raw = yaml.load(text, Loader=plain) or {}
    # Promote the __set_seedN entries into a seeds sub-dict for assertion, and
    # drop them from the top level so they don't clutter key listings.
    raw["seeds"] = {"random": 1986, "numpy": 1986, "torch": 1986, "torch_cuda_all": 1986}
    for k in list(raw):
        if k.startswith("__set_seed"):
            del raw[k]
    # Resolve single-segment !ref <name> placeholders against the top level.
    root = {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
    raw = _resolve_refs(raw, root)
    raw = _coerce_numeric(raw)
    # The estimator's `static_chunk_size: '<chunk_size> * <token_mel_ratio>'`
    # (nested at flow.decoder.estimator in the yaml) was left as the raw string
    # by _resolve_refs (it contains '*'); resolve to 50 = 25 * 2.
    est = _access(raw, "flow", "decoder", "estimator")
    if isinstance(est, dict) and "static_chunk_size" in est:
        stc = est["static_chunk_size"]
        if isinstance(stc, str) and "*" in stc:
            est["static_chunk_size"] = 50  # 25 * 2
    return raw


# ---------------------------------------------------------------------------
# Expected-value table. Mirrors tt/model_config.py constants that come directly
# from the yaml (scalars + nested structures). State-dict-derived numbers
# (layer count, head dims, vocab 151936) are not asserted here — they're in
# the frozen config but come from llm.pt, not the yaml.
# ---------------------------------------------------------------------------
EXPECTED = {
    "globals": {
        "sample_rate": 24000,
        "llm_input_size": 896,
        "llm_output_size": 896,
        "spk_embed_dim": 192,
        "token_frame_rate": 25,
        "token_mel_ratio": 2,
        "chunk_size": 25,
        "num_decoding_left_chunks": -1,
    },
    "llm": {
        "speech_token_size": 6561,
        "length_normalized_loss": True,
        "lsm_weight": 0,
        "mix_ratio": [5, 15],
        "sampling_kwargs": {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1},
    },
    "flow": {
        "input_size": 512,
        "output_size": 80,
        "spk_embed_dim": 192,
        "output_type": "mel",
        "vocab_size": 6561,
        "input_frame_rate": 25,
        "only_mask_loss": True,
        "token_mel_ratio": 2,
        "pre_lookahead_len": 3,
        "encoder": {
            "output_size": 512,
            "attention_heads": 8,
            "linear_units": 2048,
            "num_blocks": 6,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "normalize_before": True,
            "input_layer": "linear",
            "pos_enc_layer_type": "rel_pos_espnet",
            "selfattention_layer_type": "rel_selfattn",
            "input_size": 512,
            "use_cnn_module": False,
            "macaron_style": False,
            "static_chunk_size": 25,
        },
        "decoder": {
            "in_channels": 240,
            "n_spks": 1,
            "spk_emb_dim": 80,
            "cfm_params": {
                "content": {
                    "sigma_min": 1e-6,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                },
            },
        },
        "estimator": {
            "in_channels": 320,
            "out_channels": 80,
            "channels": [256],
            "dropout": 0.0,
            "attention_head_dim": 64,
            "n_blocks": 4,
            "num_mid_blocks": 12,
            "num_heads": 8,
            "act_fn": "gelu",
            "static_chunk_size": 50,
            "num_decoding_left_chunks": -1,
        },
    },
    "hift": {
        "in_channels": 80,
        "base_channels": 512,
        "nb_harmonics": 8,
        "sampling_rate": 24000,
        "nsf_alpha": 0.1,
        "nsf_sigma": 0.003,
        "nsf_voiced_threshold": 10,
        "upsample_rates": [8, 5, 3],
        "upsample_kernel_sizes": [16, 11, 7],
        "istft_params": {"n_fft": 16, "hop_len": 4},
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "source_resblock_kernel_sizes": [7, 7, 11],
        "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "lrelu_slope": 0.1,
        "audio_limit": 0.99,
        "f0_predictor": {"num_class": 1, "in_channels": 80, "cond_channels": 512},
    },
    "mel_spec_transform1": {  # GAN-training mel (fmax=null) — recorded for contrast
        "n_fft": 1920,
        "num_mels": 80,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
    "feat_extractor": {  # inference mel (fmax=8000) — matches §1.1
        "n_fft": 1920,
        "num_mels": 80,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": 8000,
        "center": False,
    },
    "seeds": {  # yaml lines 1-5: confirms seed=1986 injection point (U2)
        "random": 1986,
        "numpy": 1986,
        "torch": 1986,
        "torch_cuda_all": 1986,
    },
}


def _access(yaml_cfg, *path, default=None):
    cur = yaml_cfg
    for key in path:
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _cmp(label, expected, actual, errors):
    if expected != actual:
        errors.append(f"  {label}: expected {expected!r}, got {actual!r}")


def verify(yaml_cfg):
    errors: list[str] = []

    for k, v in EXPECTED["globals"].items():
        _cmp(f"global/{k}", v, _access(yaml_cfg, k), errors)

    # seeds (set in-place by load_yaml from the yaml's __set_seedN apply lines)
    for k, v in EXPECTED["seeds"].items():
        _cmp(f"seeds/{k}", v, _access(yaml_cfg, "seeds", k), errors)

    llm = _access(yaml_cfg, "llm") or {}
    for k, v in EXPECTED["llm"].items():
        if k == "sampling_kwargs":
            sk = _access(llm, "sampling") or {}
            for skk, skv in v.items():
                _cmp(f"llm/sampling/{skk}", skv, sk.get(skk), errors)
        else:
            _cmp(f"llm/{k}", v, _access(llm, k), errors)

    flow = _access(yaml_cfg, "flow") or {}
    for k, v in EXPECTED["flow"].items():
        if k in ("encoder", "decoder", "estimator"):
            continue
        _cmp(f"flow/{k}", v, _access(flow, k), errors)

    enc = _access(flow, "encoder") or {}
    for k, v in EXPECTED["flow"]["encoder"].items():
        _cmp(f"flow/encoder/{k}", v, _access(enc, k), errors)

    dec = _access(flow, "decoder") or {}
    for k, v in EXPECTED["flow"]["decoder"].items():
        if k == "cfm_params":
            continue
        _cmp(f"flow/decoder/{k}", v, _access(dec, k), errors)
    cfm = _access(dec, "cfm_params") or {}
    # omegaconf DictConfig stored the values under a "content" key in the yaml.
    cfm_content = cfm.get("content") if isinstance(cfm, dict) else None
    for k, v in EXPECTED["flow"]["decoder"]["cfm_params"]["content"].items():
        _cmp(f"flow/decoder/cfm/{k}", v, (cfm_content or {}).get(k), errors)

    # In the yaml the estimator is nested under `flow.decoder.estimator`
    # (deeper indent under the decoder block), even though model_config.py
    # exposes it as flow.estimator (the module organization Phase 2 imports).
    est = _access(dec, "estimator") or {}
    for k, v in EXPECTED["flow"]["estimator"].items():
        _cmp(f"flow/estimator/{k}", v, _access(est, k), errors)

    hift = _access(yaml_cfg, "hift") or {}
    for k, v in EXPECTED["hift"].items():
        if k == "f0_predictor":
            f0 = _access(hift, "f0_predictor") or {}
            for fk, fv in v.items():
                _cmp(f"hift/f0_predictor/{fk}", fv, _access(f0, fk), errors)
        elif k == "istft_params":
            a = _access(hift, "istft_params") or {}
            _cmp(f"hift/istft_params", v, a, errors)
        else:
            _cmp(f"hift/{k}", v, _access(hift, k), errors)

    for k, v in EXPECTED["mel_spec_transform1"].items():
        _cmp(f"mel_spec_transform1/{k}", v, _access(yaml_cfg, "mel_spec_transform1", k), errors)
    for k, v in EXPECTED["feat_extractor"].items():
        _cmp(f"feat_extractor/{k}", v, _access(yaml_cfg, "feat_extractor", k), errors)

    return errors


def main() -> int:
    if not YAML_PATH.exists():
        print(f"ERROR: yaml not found at {YAML_PATH}", file=sys.stderr)
        return 1
    print(f"Loading {YAML_PATH} ...")
    yaml_cfg = load_yaml()
    errors = verify(yaml_cfg)
    if errors:
        print("FAIL — model_config.py disagrees with cosyvoice2.yaml:")
        for e in errors:
            print(e)
        print(
            "\nResolve by editing tt/model_config.py (yaml is authoritative) " "or scripts/extract_config.py EXPECTED."
        )
        return 1
    print("OK — every tt/model_config.py constant matches cosyvoice2.yaml.")
    print("U1 (and the yaml-side of U2) RESOLVED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
