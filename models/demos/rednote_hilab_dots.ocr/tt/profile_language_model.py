# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr Qwen2 language-model assembly.

Profiles :class:`TtLanguageModel` -- the full Qwen2ForCausalLM assembly
(TtEmbedding -> N x TtDecoderLayer -> final TtRMSNorm -> TtLMHead) in isolation
under metal trace so the CSV reflects device-kernel time rather than host
dispatch.

Production-representative shapes come from the seed-0 golden (the same golden the
PCC test consumes): seq_len 64, hidden 1536, 12 query / 2 KV heads, head_dim 128,
rope_theta 1e6, rms_norm_eps 1e-6, attention_bias True, vocab 151936, reduced
num_layers (2) standing in for the full 28-layer trunk (the per-layer device
profile scales linearly with layer count, so 2 layers is a representative-per-layer
sample and keeps the trace small -- matching the golden config).

language_model COMPOSES already-optimized leaf/composite modules: TtDecoderLayer
carries the attention (-21.8%) + mlp (-7.5%) + residual L1 wins, the final
TtRMSNorm is at-ceiling, and TtLMHead carries the bf8 (-35%) win. So it inherits
those wins. This harness exists to check the COMPOSITE boundaries that only appear
at the assembly level: the embed -> decoder-stack handoff (the reshape that flattens
[1, seq, hidden] -> [seq, hidden]), any inter-layer reshard landing DRAM, and the
final-norm -> lm_head handoff.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_language_model.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "dots_tt_language_model_profile", os.path.join(_TT_DIR, "language_model.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtLanguageModel = _mod.TtLanguageModel

_GOLDEN_PATH = os.path.normpath(os.path.join(_TT_DIR, "..", "reference", "golden", "language_model.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    golden = torch.load(_GOLDEN_PATH, map_location="cpu", weights_only=False)
    input_ids = golden["input"].to(torch.int64)  # [1, 64]
    state_dict = {k: v.to(torch.float32) for k, v in golden["state_dict"].items()}
    cfg = golden["config"]

    _, seq_len = input_ids.shape

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        model = TtLanguageModel(
            device=device,
            state_dict=state_dict,
            num_layers=int(cfg["num_layers"]),
            seq_len=seq_len,
            num_heads=int(cfg["num_attention_heads"]),
            num_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            rope_theta=float(cfg["rope_theta"]),
            eps=float(cfg["rms_norm_eps"]),
            bias=bool(cfg["attention_bias"]),
        )

        # ttnn.embedding consumes row-major uint32 indices (the production input).
        tt_input = ttnn.from_torch(
            input_ids.to(torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = model(tt_input)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = model(tt_input)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = model(tt_input)
            ttnn.synchronize_device(device)

        print("profile_language_model done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
