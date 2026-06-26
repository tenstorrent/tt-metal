#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch RVC v2 48 kHz assets from public sources and convert to the
# .safetensors layout the demo loads. Idempotent — re-running skips
# completed steps.
#
# Sources (all public on HuggingFace):
#   lj1995/VoiceConversionWebUI : pretrained_v2/f0G48k.pth, hubert_base.pt, rmvpe.pt
#   facebook/hubert-base-ls960  : config.json (for hubert_cfg.json fallback)
#
# Final on-disk layout (matches models/demos/rvc/utils/config.py):
#   data/assets/pretrained_v2/f0G48k.safetensors
#   data/assets/hubert.safetensors
#   data/configs/v2/48k.json
#   data/configs/hubert_cfg.json
#   data/rmvpe.safetensors
#   data/speech/<your-input>.wav   (user-provided)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ASSETS_DIR="${DATA_DIR}/assets"
PRETRAINED_DIR="${ASSETS_DIR}/pretrained_v2"
CONFIGS_DIR="${DATA_DIR}/configs"
CONFIGS_V2_DIR="${CONFIGS_DIR}/v2"
SPEECH_DIR="${DATA_DIR}/speech"
TMP_DIR="${SCRIPT_DIR}/.assets_tmp"

LJ1995="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

mkdir -p "${PRETRAINED_DIR}" "${ASSETS_DIR}" "${CONFIGS_V2_DIR}" "${SPEECH_DIR}" "${TMP_DIR}"

fetch() {
  local url="$1" out="$2"
  if [[ -s "${out}" ]]; then
    echo "  [skip] ${out} exists"
    return
  fi
  echo "  [fetch] ${url}"
  curl --location --fail --silent --show-error --output "${out}.part" "${url}"
  mv "${out}.part" "${out}"
}

convert_pt_to_safetensors() {
  local pt_path="$1" st_path="$2" extract_mode="$3"  # rvc | hubert | rmvpe
  if [[ -s "${st_path}" ]]; then
    echo "  [skip] ${st_path} exists"
    return
  fi
  echo "  [convert] ${pt_path} -> ${st_path}  (mode=${extract_mode})"
  python3 - "${pt_path}" "${st_path}" "${extract_mode}" <<'PYEOF'
import sys, pickle, torch
from safetensors.torch import save_file
pt_path, st_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]

# hubert_base.pt is a fairseq checkpoint with non-torch classes in the top-level
# dict. We only need state_dict["model"]; bypass fairseq by stubbing find_class.
class _Dummy:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state): pass
    def __reduce__(self): return (_Dummy, ())

class _SkipFairseqUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("torch", "collections", "numpy", "builtins", "argparse")):
            return super().find_class(module, name)
        return _Dummy

class _SkipPickleModule:
    Unpickler = _SkipFairseqUnpickler
    @staticmethod
    def load(file, **kw): return _SkipFairseqUnpickler(file).load()

try:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
except Exception:
    if mode != "hubert":
        raise
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False, pickle_module=_SkipPickleModule)

if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
    sd = ckpt["model"]
elif isinstance(ckpt, dict) and "weight" in ckpt and isinstance(ckpt["weight"], dict):
    sd = ckpt["weight"]
elif isinstance(ckpt, dict):
    sd = ckpt
else:
    raise SystemExit(f"unexpected checkpoint root: {type(ckpt).__name__}")

def squeeze_kernel1(w):
    """Conv1d(k=1) [out, in, 1] -> Linear [out, in]. No-op for k>1."""
    return w.squeeze(-1).contiguous() if w.ndim == 3 and w.shape[-1] == 1 else w.contiguous()

def wn_to_w(wg, wv):
    """Collapse PyTorch weight_norm (dim=0) and squeeze if k=1."""
    reduce_dims = list(range(1, wv.ndim))
    norm = (wv.float() ** 2).sum(dim=reduce_dims, keepdim=True).sqrt()
    w = (wg.float() * wv.float() / (norm + 1e-12)).contiguous()
    return squeeze_kernel1(w)

if mode == "rvc":
    # Upstream RVC f0G48k uses VITS-style nesting: 4 ResidualCouplingLayers
    # interleaved with Flip layers (no-param), and weight_norm on every WN conv.
    # The demo's torch_impl/vc/synthesizer.py refactors this to use Linear (not
    # Conv1d k=1) for pre/post and removes weight_norm everywhere, so the demo
    # checkpoint must mirror that flattening.
    new = {}
    # Flow: renumber 0,2,4,6 -> 0,1,2,3; rename pre/post -> pre_linear/post_linear;
    # collapse weight_norm on cond_layer / in_layers / res_skip_layers.
    for old_idx, new_idx in [(0, 0), (2, 1), (4, 2), (6, 3)]:
        op = f"flow.flows.{old_idx}"
        np_ = f"flow.flows.{new_idx}"
        new[f"{np_}.pre_linear.weight"] = squeeze_kernel1(sd[f"{op}.pre.weight"])
        new[f"{np_}.pre_linear.bias"]   = sd[f"{op}.pre.bias"].contiguous()
        new[f"{np_}.post_linear.weight"] = squeeze_kernel1(sd[f"{op}.post.weight"])
        new[f"{np_}.post_linear.bias"]   = sd[f"{op}.post.bias"].contiguous()
        new[f"{np_}.enc.cond_layer.weight"] = wn_to_w(
            sd[f"{op}.enc.cond_layer.weight_g"], sd[f"{op}.enc.cond_layer.weight_v"])
        new[f"{np_}.enc.cond_layer.bias"] = sd[f"{op}.enc.cond_layer.bias"].contiguous()
        for i in range(3):
            new[f"{np_}.enc.in_layers.{i}.weight"] = wn_to_w(
                sd[f"{op}.enc.in_layers.{i}.weight_g"], sd[f"{op}.enc.in_layers.{i}.weight_v"])
            new[f"{np_}.enc.in_layers.{i}.bias"] = sd[f"{op}.enc.in_layers.{i}.bias"].contiguous()
            new[f"{np_}.enc.res_skip_layers.{i}.weight"] = wn_to_w(
                sd[f"{op}.enc.res_skip_layers.{i}.weight_g"], sd[f"{op}.enc.res_skip_layers.{i}.weight_v"])
            new[f"{np_}.enc.res_skip_layers.{i}.bias"] = sd[f"{op}.enc.res_skip_layers.{i}.bias"].contiguous()

    # Generator: rename dec.cond -> dec.cond_linear; collapse weight_norm
    # on resblocks.*.convs1/2 and ups.*; pass-through everything else under dec.*.
    new["dec.cond_linear.weight"] = squeeze_kernel1(sd["dec.cond.weight"])
    new["dec.cond_linear.bias"]   = sd["dec.cond.bias"].contiguous()
    pass_through_dec = ("dec.conv_pre", "dec.conv_post", "dec.noise_convs", "dec.m_source")
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith(pass_through_dec):
            # noise_convs[3] is Conv1d(k=1) and the demo expects Linear shape;
            # squeeze_kernel1 is a no-op for k>1 convs.
            new[k] = squeeze_kernel1(v.detach())
    # weight_norm collapse for ups.*, resblocks.*.convs1/2.*
    seen = set()
    for k in sd:
        if not k.endswith(".weight_g"):
            continue
        if not (k.startswith("dec.ups.") or k.startswith("dec.resblocks.")):
            continue
        base = k[:-len(".weight_g")]
        if base in seen:
            continue
        seen.add(base)
        new[f"{base}.weight"] = wn_to_w(sd[f"{base}.weight_g"], sd[f"{base}.weight_v"])
        bias_key = f"{base}.bias"
        if bias_key in sd and torch.is_tensor(sd[bias_key]):
            new[bias_key] = sd[bias_key].detach().contiguous()
    # TextEncoder (enc_p): same Conv1d(k=1) -> Linear pattern as flow's pre/post.
    # Upstream has conv_{q,k,v,o} in attention (k=1) and a top-level proj (k=1);
    # the demo's MultiHeadAttention has linear_{q,k,v,o} and TextEncoder has
    # proj_linear. FFN's conv_{1,2} stay as Conv1d (k=3, no rename/squeeze).
    # norm_layers use {beta, gamma} param names which match.
    for k, v in sd.items():
        if not torch.is_tensor(v) or not k.startswith("enc_p."):
            continue
        # attn_layers conv_{q,k,v,o} -> linear_{q,k,v,o}
        import re as _re
        m = _re.match(r"^(enc_p\.encoder\.attn_layers\.\d+)\.(conv_[qkvo])\.(weight|bias)$", k)
        if m:
            new_name = "linear_" + m.group(2).split("_")[1]
            new_key = f"{m.group(1)}.{new_name}.{m.group(3)}"
            new[new_key] = squeeze_kernel1(v.detach())
            continue
        # top-level enc_p.proj -> enc_p.proj_linear (after enc_p. is stripped by demo loader)
        if k == "enc_p.proj.weight":
            new["enc_p.proj_linear.weight"] = squeeze_kernel1(v.detach())
            continue
        if k == "enc_p.proj.bias":
            new["enc_p.proj_linear.bias"] = v.detach().contiguous()
            continue
        # pass-through: emb_phone, emb_pitch, ffn_layers.conv_1/2 (k=3),
        # norm_layers.{beta,gamma}, attn_layers.emb_rel_{k,v}
        new[k] = v.detach().contiguous()

    # emb_g (speaker embedding) pass-through.
    for k, v in sd.items():
        if torch.is_tensor(v) and k.startswith("emb_g."):
            new[k] = v.detach().contiguous()

    # enc_q.* (PosteriorEncoder) is training-only — not loaded by demo. Skip.
    sd_out = new
elif mode == "hubert":
    # fairseq HuBERT-base -> demo's refactored HubertModel:
    #   - collapse weight_norm on encoder.pos_conv.0
    #   - rename feature_extractor.conv_layers.0.2.* -> .0.1.*  (fairseq has
    #     Dropout at index 1; demo skips Dropout so GroupNorm is at index 1)
    #   - drop pretraining-only heads (label_embs_concat, mask_emb, final_proj,
    #     target_glu) — not present in the demo's HubertModel
    sd_out = {}
    drop_prefixes = ("label_embs_concat", "mask_emb", "target_glu")
    pos_conv_handled = set()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith(drop_prefixes):
            continue
        # weight_norm on encoder.pos_conv.0
        if k.startswith("encoder.pos_conv.0.weight_g") or k.startswith("encoder.pos_conv.0.weight_v"):
            base = "encoder.pos_conv.0"
            if base in pos_conv_handled:
                continue
            pos_conv_handled.add(base)
            sd_out[f"{base}.weight"] = wn_to_w(sd[f"{base}.weight_g"], sd[f"{base}.weight_v"])
            continue
        # feature_extractor.conv_layers.0.2 -> .0.1 (GroupNorm reindex)
        if k.startswith("feature_extractor.conv_layers.0.2."):
            nk = "feature_extractor.conv_layers.0.1." + k[len("feature_extractor.conv_layers.0.2."):]
            sd_out[nk] = v.detach().contiguous().clone()
            continue
        sd_out[k] = v.detach().contiguous().clone()
else:
    # rmvpe: pass-through.
    sd_out = {k: v.detach().contiguous().clone() for k, v in sd.items() if torch.is_tensor(v)}

if not sd_out:
    raise SystemExit(f"no tensors extracted from {pt_path}")

# Storage dtype matters for perf: the demo's TTNN runtime uses bfloat16
# internally. Saving as bfloat16 halves the file size and (more importantly)
# halves the mmap'd safetensors lazy-load bytes per .float() call inside
# preprocess_linear_weight(), which measurably speeds up Generator forward
# (~30% on N300 in benchmarks). Float32 storage is functionally correct but
# wastes precision since it's downcast to bfloat16 on-device anyway.
sd_out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd_out.items()}
save_file(sd_out, st_path)
print(f"  wrote {len(sd_out)} tensors (bfloat16)")
PYEOF
}

echo "=== 1. Download .pt files from lj1995/VoiceConversionWebUI ==="
fetch "${LJ1995}/pretrained_v2/f0G48k.pth"  "${TMP_DIR}/f0G48k.pth"
fetch "${LJ1995}/hubert_base.pt"            "${TMP_DIR}/hubert_base.pt"
fetch "${LJ1995}/rmvpe.pt"                  "${TMP_DIR}/rmvpe.pt"

echo "=== 2. Convert to .safetensors ==="
convert_pt_to_safetensors "${TMP_DIR}/f0G48k.pth"     "${PRETRAINED_DIR}/f0G48k.safetensors"  rvc
convert_pt_to_safetensors "${TMP_DIR}/hubert_base.pt" "${ASSETS_DIR}/hubert.safetensors"      hubert
convert_pt_to_safetensors "${TMP_DIR}/rmvpe.pt"       "${DATA_DIR}/rmvpe.safetensors"         rmvpe

echo "=== 3. Install config files ==="
# Configs are committed at scripts/configs/ (small JSONs, not in data/ which
# is gitignored). The HuBERT config is the full fairseq cfg the demo expects
# (cfg["model"] sub-dict, plus cfg["task"], etc.) — too large to inline cleanly.
SCRIPT_CONFIGS="${SCRIPT_DIR}/scripts/configs"
if [[ ! -f "${SCRIPT_CONFIGS}/hubert_cfg.json" || ! -f "${SCRIPT_CONFIGS}/v2/48k.json" ]]; then
  echo "ERROR: expected configs missing from ${SCRIPT_CONFIGS}/"
  echo "  hubert_cfg.json or v2/48k.json not found"
  exit 1
fi
cp "${SCRIPT_CONFIGS}/hubert_cfg.json" "${CONFIGS_DIR}/hubert_cfg.json"
cp "${SCRIPT_CONFIGS}/v2/48k.json"     "${CONFIGS_V2_DIR}/48k.json"
echo "  installed ${CONFIGS_DIR}/hubert_cfg.json"
echo "  installed ${CONFIGS_V2_DIR}/48k.json"

echo "=== 4. Cleanup ==="
rm -rf "${TMP_DIR}"

echo "=== Done. Final layout: ==="
( cd "${DATA_DIR}" && find . -maxdepth 4 -type f -not -path "./speech/*" -not -path "./output/*" | sort | sed 's/^/  /' )
echo ""
echo "Next: place a 16+ kHz mono speech .wav at:"
echo "  ${SPEECH_DIR}/sample-speech-0.wav"
echo "Then run: python -m models.demos.rvc.demo"
