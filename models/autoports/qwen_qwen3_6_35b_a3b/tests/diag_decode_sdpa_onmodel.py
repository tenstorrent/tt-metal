# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On-model control for the decode-SDPA chunk/decomposition choice (README section 3.8).

`diag_sdpa_decode.py` measures the op in isolation with random K/V over a 2-D
(`k_chunk_size` x `max_cores_per_head_batch`) grid, and shows that long-context decode accuracy is
governed by **`k_chunk_size`** -- the bf16 accumulation depth -- while `max_cores_per_head_batch`
must stay at 1 because every larger value is silently wrong below some context. This script
re-measures the candidates on the **whole decoder layer** against HF, off a real prefilled cache, so
the shipping decision rests on the real path:

    A  no program config     -- *not* neutral: the paged op substitutes `k_chunk_size=32`,
                                1 core/head (`sdpa_decode.cpp:122-129`), the worst measured
                                setting. This is what the layer ran before any sweep existed.
    B  k_chunk 128, 1 core   -- what `k_chunk_size=0` (dynamic) resolves to here. The setting the
                                previous, wrongly-attributed review round shipped.
    C  k_chunk 512, 1 core   -- **what the layer now ships**
                                (`DecoderConfig.decode_sdpa_k_chunk_size = 512`). Largest legal
                                chunk: 1024 exceeds L1.
    D  k_chunk 256, 16 cores -- the fastest setting in the whole op sweep (1.39 ms/call vs C's
                                7.45) and the most accurate at the advertised context. Included to
                                measure on-model what the op sweep says is unshippable: it is
                                silently wrong at 257, 1024 and 4096 keys.

One layer is built per context and prefilled **once**; every setting then decodes from that same
cache, so the comparison is same-input (an earlier version compared across processes with different
seeds, which was weaker).

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_decode_sdpa_onmodel.py
"""

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import (
    build_layer_pair,
    from_tt,
    to_tt_decode,
    to_tt_positions,
    to_tt_prefill,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

#: decode positions: short contexts where the op sweep shows multi-core settings breaking, the
#: mid range, and the advertised context.
POSITIONS = [257, 1023, 4095, 32767, 262143]


def settings(device):
    """(label, SDPAProgramConfig or None) for each candidate."""
    grid = device.compute_with_storage_grid_size()

    def cfg(k_chunk, max_cores):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_cores_per_head_batch=max_cores,
        )

    return [
        ("A no-config (k32, 1 core)", None),
        ("B k128, 1 core", cfg(128, 1)),
        ("C k512, 1 core (shipped)", cfg(512, 1)),
        ("D k256, 16 cores (fastest)", cfg(256, 16)),
    ]


def run_position(device, position):
    context = position + 1
    pair = build_layer_pair(
        device,
        kind="full",
        max_batch_size=1,
        supported_context=max(context, 1024),
    )
    pair.tt.reset_state()

    x = ref.synthetic_hidden_states(pair.hf_config, 1, position, seed=2)
    tt_x = to_tt_prefill(device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table))
    ttnn.deallocate(tt_x)
    hf_cache = ref.hf_fill_full_attention_cache(pair.hf, pair.hf_config, x)
    del x

    token = ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=3)
    want = ref.hf_decode(pair.hf, pair.hf_config, token, positions=torch.tensor([position]), cache=hf_cache)

    out = {}
    for label, program_config in settings(device):
        # decode-only knob, so the same prefilled cache serves every setting. Set the built
        # config directly: `_decode_sdpa_program_config` runs at construction time.
        pair.tt.decode_sdpa_config = program_config
        tt_tok = to_tt_decode(device, token.reshape(1, 1, -1))
        tt_pos = to_tt_positions(device, torch.tensor([position]))
        tt_out = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=pair.page_table)
        got = from_tt(tt_out).reshape(1, 1, pair.cfg.hidden_size)
        for t in (tt_tok, tt_pos, tt_out):
            ttnn.deallocate(t)
        value = ref.pcc(got, want)
        out[label] = value
        print(f"ONMODEL context={context:>7}  {label:<30} layer_pcc={value:.7f}", flush=True)
    pair.tt.release()
    return out


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        labels = [label for label, _ in settings(device)]
        table = {p: run_position(device, p) for p in POSITIONS}
        print("\nONMODEL SUMMARY  context " + "  ".join(f"{label:>30}" for label in labels), flush=True)
        for position, row in table.items():
            print(
                f"ONMODEL SUMMARY {position + 1:>8} " + "  ".join(f"{row[label]:>30.7f}" for label in labels),
                flush=True,
            )
        worst = {label: min(row[label] for row in table.values()) for label in labels}
        print("\nONMODEL WORST-OVER-CONTEXTS " + "  ".join(f"{label}={worst[label]:.7f}" for label in labels))
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
