# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record checkpoint tensor stats and real decoder-input activation stats for synthetic tests.

python models/autoports/google_gemma_4_12b_it/tests/stats/collect_stats.py

* weight_stats.json: for every text decoder layer and every tensor: shape, dtype, mean, std.
* activation_stats.json: per-channel mean/std of the real inputs to layers 0 (sliding) and 5 (full),
  from a 1024-token chat-formatted prompt of repository text through the float32 HF layers.
"""

import json

from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    ACTIVATION_STATS,
    HF_REVISION,
    LAYER_KINDS,
    WEIGHT_STATS,
    real_hidden,
    real_layer_state_dict,
    real_prompt_ids,
    text_config,
)


def main():
    cfg = text_config()
    stats = {}
    for i in range(cfg.num_hidden_layers):
        sd = real_layer_state_dict(i)
        stats[str(i)] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype).replace("torch.", ""),
                "mean": float(v.float().mean()),
                "std": float(v.float().std()) if v.numel() > 1 else 0.0,
            }
            for k, v in sorted(sd.items())
        }
    stats["_provenance"] = {"hf_revision": HF_REVISION}
    WEIGHT_STATS.write_text(json.dumps(stats, indent=1) + "\n")

    ids = real_prompt_ids(1024)
    act = {"_provenance": f"{len(ids)} chat-formatted prompt tokens, revision {HF_REVISION}"}
    for idx in sorted(LAYER_KINDS.values()):
        x = real_hidden(cfg, idx, ids)[0]
        act[f"{idx}.mean"] = [float(f"{v:.6g}") for v in x.mean(0).tolist()]
        act[f"{idx}.std"] = [float(f"{v:.6g}") for v in x.std(0).tolist()]
        print(f"layer {idx}: global mean {x.mean():.4f} std {x.std():.4f} absmax {x.abs().max():.1f}")
    ACTIVATION_STATS.write_text(json.dumps(act) + "\n")


if __name__ == "__main__":
    main()
