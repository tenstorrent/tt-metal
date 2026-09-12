"""Torch golden for the DFlash2 draft-model ttnn port.

Loads the reference DFlash2 draft (z-lab/dflash `dflash/model.py`) with the real
incoai/Qwen3.8-27B-DFlash2 weights, runs its forward on SYNTHETIC inputs (fixed seed),
and dumps per-component intermediates (fc/hidden_norm, per-layer, per-conv, draft_hidden,
selector) to an .npz the ttnn bring-up tests PCC against. No 27B target needed: the draft
layers take a synthetic noise_embedding (1,8,H) + synthetic target_hidden (1,C,25600).

Run:  DFLASH_REPO=/tmp/.../dflash_src DFLASH_W=/home/.../dflash_weights \
      tt-metal/python_env/bin/python models/demos/blackhole/qwen36/tests/dflash2_golden.py
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

REPO = os.environ.get("DFLASH_REPO", "/tmp/dflash_src")
W = os.environ.get("DFLASH_W", "/home/ttuser/experiments/qwen36_27b/dflash_weights")
OUT = os.environ.get("DFLASH_GOLDEN", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_golden.npz")
C = int(os.environ.get("DFLASH_CTX", "40"))  # synthetic context length
SEED = 0


def _load_reference():
    sys.path.insert(0, REPO)
    import json

    from dflash.model import DFlash2DraftModel  # noqa: E402
    from transformers import Qwen3Config

    cfg_d = json.load(open(f"{W}/config.json"))
    cfg = Qwen3Config(**{k: v for k, v in cfg_d.items() if k != "dflash_config"})
    for k, v in cfg_d["dflash_config"].items():
        setattr(cfg, k, v)
    cfg.target_layer_ids = cfg_d["dflash_config"]["target_layer_ids"]
    cfg.torch_dtype = torch.float32
    model = DFlash2DraftModel(cfg).to(torch.float32).eval()

    # Load weights (safetensors shards). Selector codebooks are stored without a .weight suffix.
    sd = {}
    import glob

    for f in sorted(glob.glob(f"{W}/*.safetensors")):
        sd.update(load_file(f))
    remap = {}
    for k in list(sd):
        if k.endswith("candidate_selector.predecessor_codebook") or k.endswith("candidate_selector.successor_codebook"):
            remap[k] = k + ".weight"
    for old, new in remap.items():
        sd[new] = sd.pop(old)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[golden] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  e.g. missing:", missing[:6])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:6])
    return model, cfg


def main():
    torch.manual_seed(SEED)
    model, cfg = _load_reference()
    H = cfg.hidden_size
    B = cfg.block_size
    taps = len(cfg.target_layer_ids)

    REAL = os.environ.get("DFLASH_REAL")
    if REAL and os.path.exists(REAL):
        # Real target taps captured from an on-device prefill (Phase B).
        r = np.load(REAL)
        target_hidden_cat = torch.from_numpy(r["target_hidden_cat"]).float()  # (1,C,25600)
        C = target_hidden_cat.shape[1]
        anchor = int(r["anchor"])
        # noise_embedding = target.embed_tokens([anchor, MASK*7]) (the block).
        from safetensors import safe_open

        tgt = os.environ.get("TGT_W", "/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B")
        with safe_open(f"{tgt}/model-00001-of-00015.safetensors", framework="pt") as f:
            emb = f.get_tensor("model.language_model.embed_tokens.weight").float()  # (V,H)
        blk = torch.tensor([[anchor] + [cfg.mask_token_id] * (B - 1)])  # (1,8)
        noise_embedding = torch.nn.functional.embedding(blk, emb)  # (1,8,H)
        print(f"[golden] REAL inputs: C={C} anchor={anchor}")
    else:
        noise_embedding = torch.randn(1, B, H, dtype=torch.float32) * 0.1
        target_hidden_cat = torch.randn(1, C, taps * H, dtype=torch.float32) * 0.1  # concat of 5 taps
    position_ids = torch.arange(C + B, dtype=torch.long).unsqueeze(0)

    dump = {
        "noise_embedding": noise_embedding.numpy(),
        "target_hidden_cat": target_hidden_cat.numpy(),
        "position_ids": position_ids.numpy(),
    }

    # Capture intermediates via hooks.
    caught = {}
    h1 = model.fc.register_forward_hook(lambda m, i, o: caught.__setitem__("fc_out", o.detach()))
    h2 = model.hidden_norm.register_forward_hook(lambda m, i, o: caught.__setitem__("hidden_norm_out", o.detach()))
    for li, layer in enumerate(model.layers):
        layer.register_forward_hook(
            (
                lambda idx: lambda m, i, o: caught.__setitem__(
                    f"layer{idx}_out", o[0].detach() if isinstance(o, tuple) else o.detach()
                )
            )(li)
        )

    # Capture layer0 self_attn I/O (for gate A3) — the EXACT inputs the real model feeds it.
    acap = {}

    def _attn_pre(mod, args, kwargs):
        acap["hidden"] = kwargs["hidden_states"].detach()
        acap["ctx"] = kwargs["target_hidden"].detach()
        cos, sin = kwargs["position_embeddings"]
        acap["cos"] = cos.detach()
        acap["sin"] = sin.detach()

    model.layers[0].self_attn.register_forward_pre_hook(_attn_pre, with_kwargs=True)
    model.layers[0].self_attn.register_forward_hook(lambda m, i, o: acap.__setitem__("out", o[0].detach()))

    with torch.no_grad():
        draft_hidden_full = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_cat,
        )
    dump["fc_out"] = caught["fc_out"].numpy()
    dump["hidden_norm_out"] = caught["hidden_norm_out"].numpy()
    for li in range(len(model.layers)):
        dump[f"layer{li}_out"] = caught[f"layer{li}_out"].numpy()
    dump["draft_hidden_full"] = draft_hidden_full.detach().numpy()  # (1,B,H)
    dump["draft_hidden"] = draft_hidden_full[:, 1 - B :, :].detach().numpy()  # (1,B-1,H) drafted positions
    for k in ("hidden", "ctx", "cos", "sin", "out"):
        dump[f"attn0_{k}"] = acap[k].numpy()

    h1.remove()
    h2.remove()

    # bf16 floor: the reference's OWN fp32->bf16 loss (the best any bf16 impl, incl. ttnn, can do).
    from models.common.utility_functions import comp_pcc

    m16 = model.to(torch.bfloat16)
    with torch.no_grad():
        dh16 = m16(
            position_ids=position_ids,
            noise_embedding=noise_embedding.bfloat16(),
            target_hidden=target_hidden_cat.bfloat16(),
        ).float()
    print(
        "[golden] bf16-FLOOR draft_hidden_full PCC (ref-bf16 vs ref-fp32):",
        comp_pcc(torch.from_numpy(dump["draft_hidden_full"]), dh16, 0.9)[1],
    )

    out = OUT.replace(".npz", "_real.npz") if REAL else OUT
    np.savez(out, **dump)
    print(f"[golden] wrote {out}: keys={list(dump)}")
    print(f"[golden] draft_hidden shape {dump['draft_hidden'].shape}")


if __name__ == "__main__":
    main()
