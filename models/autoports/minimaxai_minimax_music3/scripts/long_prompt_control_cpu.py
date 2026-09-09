# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU half of the long-prompt precision control: HF Qwen3 in bf16 and in fp32 on the same prompts,
PCC of TT vs fp32, TT vs bf16 and bf16 vs fp32 (the reference's own noise floor), for every
generated/long_prompt_control/tt_outputs*.pt (one per dtype policy, from long_prompt_control_device.py
and policy_probe_device.py). HF outputs are cached in hf_outputs.pt. Writes
doc/llm/pcc/long_prompt_control.json. No device needed."""
import json
import os
import pathlib
import time

import torch
from loguru import logger

from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.common.utility_functions import comp_pcc


def pcc(a, b):
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


torch.set_num_threads(max(16, os.cpu_count() or 16))
model_dir = pathlib.Path(os.environ["MM3_MODEL_DIR"])
ctrl = model_dir / "generated" / "long_prompt_control"
tt_files = sorted(ctrl.glob("tt_outputs*.pt"))
tts = {f.stem.replace("tt_outputs", "").lstrip("_") or "functional": torch.load(f) for f in tt_files}
base = tts["functional"]
hf_cache = ctrl / "hf_outputs.pt"
if hf_cache.is_file():
    out = torch.load(hf_cache)
    logger.info("loaded cached HF outputs")
else:
    out = {}
    for dtype in (torch.bfloat16, torch.float32):
        t0 = time.time()
        hf = R.load_hf_qwen3(dtype)
        logger.info(f"HF {dtype} loaded {time.time()-t0:.0f}s")
        for name in ("104", "5000"):
            ids = base[f"ids{name}"]
            t0 = time.time()
            past, h, l = R.hf_prefill(hf, hf.model.embed_tokens(ids))
            _, h2, l2 = R.hf_decode_step(hf, past, base["step_in"].to(dtype).unsqueeze(1))
            logger.info(f"HF {dtype} {name}: {time.time()-t0:.0f}s")
            out[f"hf_{str(dtype).split('.')[-1]}_{name}"] = {
                "hidden": h.float(),
                "logits": l,
                "dec_hidden": h2.float(),
                "dec_logits": l2,
            }
        del hf
    torch.save(out, hf_cache)


def compare(tt):
    report = {}
    for name in ("104", "5000"):
        b = out[f"hf_bfloat16_{name}"]
        f = out[f"hf_float32_{name}"]
        rows = {}
        for r in range(2):
            rows[f"row{r}"] = {
                "prefill_hidden": {
                    "tt_vs_fp32": pcc(f["hidden"][r], tt[f"tt_hidden_{name}"][r]),
                    "tt_vs_bf16": pcc(b["hidden"][r], tt[f"tt_hidden_{name}"][r]),
                    "bf16_vs_fp32": pcc(f["hidden"][r], b["hidden"][r]),
                },
                "prefill_logits": {
                    "tt_vs_fp32": pcc(f["logits"][r], tt[f"tt_logits_{name}"][r]),
                    "tt_vs_bf16": pcc(b["logits"][r], tt[f"tt_logits_{name}"][r]),
                    "bf16_vs_fp32": pcc(f["logits"][r], b["logits"][r]),
                },
                "decode_hidden": {
                    "tt_vs_fp32": pcc(f["dec_hidden"][r], tt[f"tt_dec_hidden_{name}"][r]),
                    "tt_vs_bf16": pcc(b["dec_hidden"][r], tt[f"tt_dec_hidden_{name}"][r]),
                    "bf16_vs_fp32": pcc(f["dec_hidden"][r], b["dec_hidden"][r]),
                },
                "decode_logits": {
                    "tt_vs_fp32": pcc(f["dec_logits"][r], tt[f"tt_dec_logits_{name}"][r]),
                    "tt_vs_bf16": pcc(b["dec_logits"][r], tt[f"tt_dec_logits_{name}"][r]),
                    "bf16_vs_fp32": pcc(f["dec_logits"][r], b["dec_logits"][r]),
                },
                "argmax": {
                    "fp32": int(f["logits"][r].argmax()),
                    "bf16": int(b["logits"][r].argmax()),
                    "tt": int(tt[f"tt_logits_{name}"][r].argmax()),
                },
            }
        report[f"prompt_{name}"] = rows
    return report


full = {policy: compare(tt) for policy, tt in tts.items()}
for policy, rep in full.items():
    logger.info(f"{policy}: {json.dumps(rep, indent=1)}")
full["_meta"] = {
    "prompt_5000": "golden text_ids repeated to 5000 tokens",
    "prompt_104": "golden text_ids",
    "decode_input": "golden frame 0 codes via _embed_audio_frame",
    "hf": "Qwen3ForCausalLM transformers 5.15 CPU, bf16 and fp32",
    "tt": "MusicLLM, one entry per dtype policy",
    "policies": {p: t.get("dtypes") for p, t in tts.items()},
}
(model_dir / "doc" / "llm" / "pcc").mkdir(parents=True, exist_ok=True)
(model_dir / "doc" / "llm" / "pcc" / "long_prompt_control.json").write_text(json.dumps(full, indent=2) + "\n")
logger.info("written long_prompt_control.json")
