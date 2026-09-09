# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device half of the long-prompt precision control: prefill the golden prompt repeated to 5000
tokens (and the 104-token golden prompt) and save the TT hidden/logits for the CPU half
(long_prompt_control_cpu.py), which compares them against HF in bf16 *and* fp32."""
import os
import pathlib

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import MAX_PROMPT_TOKENS
from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM

out_dir = pathlib.Path(os.environ["MM3_MODEL_DIR"]) / "generated" / "long_prompt_control"
out_dir.mkdir(parents=True, exist_ok=True)
text_ids = torch.load(R.reference_dir() / "text_ids.pt")
reps = -(-MAX_PROMPT_TOKENS // text_ids.shape[1])
ids5000 = text_ids.repeat(1, reps)[:, :MAX_PROMPT_TOKENS]
codes = torch.load(R.reference_dir() / "sampled_codes.pt")
embed_w = R.load_embed_weight()
audio = R.load_audio_embeddings()
step_in = R.embed_audio_frame(embed_w, audio, codes[0].unsqueeze(0).expand(2, -1))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90_000_000)
mesh.enable_program_cache()
try:
    llm = MusicLLM(mesh)
    res = {"ids5000": ids5000, "ids104": text_ids, "step_in": step_in}
    for name, ids in (("104", text_ids), ("5000", ids5000)):
        llm.reset_cache()
        h, l = llm.prefill(llm.embed_tokens(ids))
        h2, l2 = llm.decode(step_in, ids.shape[1])
        res[f"tt_hidden_{name}"] = h
        res[f"tt_logits_{name}"] = l
        res[f"tt_dec_hidden_{name}"] = h2
        res[f"tt_dec_logits_{name}"] = l2
        logger.info(f"{name}: saved")
    torch.save(res, out_dir / "tt_outputs.pt")
finally:
    ttnn.close_mesh_device(mesh)
