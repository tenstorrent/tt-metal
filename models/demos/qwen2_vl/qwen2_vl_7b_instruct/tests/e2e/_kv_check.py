# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the Tier-2 fixed-capacity KV-cache decode.

Runs `pipe.generate_kv` (prefill once + seq=1 decode steps) and checks it against
the HF golden the e2e gate uses: token stream must match, and stacked next-token
logits PCC must clear 0.95. Also cross-checks it agrees with the eager full-seq
`generate()` so the KV path is a faithful drop-in.

Run:
    ./python_env/bin/python -m \
      models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._kv_check
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e import _golden
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tt.pipeline import GRADUATED_STUBS, build_pipeline

N = 16


def main():
    from transformers import Qwen2VLForConditionalGeneration

    g = _golden()
    inputs = {k: g[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    hf_tokens = g["man_tokens"][:N].tolist()
    hf_logits = g["man_logits"][:N].float()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        pipe = build_pipeline(device, model)
        kv_tokens, kv_logits = pipe.generate_kv(inputs, max_new_tokens=N, return_logits=True)
        eager_tokens, _ = pipe.generate(inputs, max_new_tokens=N)

        per_step = [float(comp_pcc(hf_logits[i], kv_logits[i], 0.0)[1]) for i in range(N)]
        _, e2e_pcc = comp_pcc(hf_logits.reshape(-1), kv_logits.reshape(-1), 0.95)
        e2e_pcc = float(e2e_pcc)
        n_match_hf = sum(int(a == b) for a, b in zip(kv_tokens, hf_tokens))
        n_match_eager = sum(int(a == b) for a, b in zip(kv_tokens, eager_tokens))

        missing = set(GRADUATED_STUBS) - pipe.invoked
        print("=" * 72)
        print(f"HF    tokens: {hf_tokens}")
        print(f"KV    tokens: {kv_tokens}")
        print(f"eager tokens: {eager_tokens}")
        print(f"per-step logits PCC: {[round(p, 4) for p in per_step]}")
        print(f"KV vs HF token match   : {n_match_hf}/{N}")
        print(f"KV vs eager token match: {n_match_eager}/{N}")
        print(f"invoked graduated stubs: {sorted(pipe.invoked)}")
        print(f"e2e stacked-logits PCC (KV vs HF): {e2e_pcc}")
        print("=" * 72)
        assert not missing, f"graduated stubs never invoked: {sorted(missing)}"
        assert e2e_pcc >= 0.95, f"PCC {e2e_pcc} < 0.95"
        assert kv_tokens == hf_tokens, f"token mismatch vs HF: {n_match_hf}/{N}"
        print("KV-cache decode PASS (tokens == HF golden, PCC >= 0.95)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
