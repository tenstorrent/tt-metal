# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On-model control for the decode-SDPA parallel decomposition (README section 3.8).

`diag_sdpa_decode.py` measures the op in isolation with random K/V and shows that decode accuracy
at long context is governed by `SDPAProgramConfig::max_cores_per_head_batch` -- the number of cores
the op splits each KV head's keys across. This script re-measures the candidate settings on the
**whole decoder layer** against HF, with a real prefilled cache, so the shipping decision rests on
the real path:

    A  no program config             -- the factory then uses `num_cores_available`, i.e. 55
                                        cores/head on this part. What the layer shipped before
                                        this sweep existed.
    B  explicit config, struct default (`max_cores_per_head_batch = 16`)
    C  explicit config, 1 core/head  -- the only setting the op sweep finds correct at *every*
                                        context, and **what the layer now ships**
                                        (`DecoderConfig.decode_sdpa_max_cores_per_head = 1`)

One layer is built per context and prefilled **once**; the three settings then decode from that
same cache, so the comparison is same-input (an earlier version of this script compared across
processes with different seeds, which was weaker).

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

    def cfg(max_cores):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=0,  # dynamic, same as the no-config path
            exp_approx_mode=False,
            max_cores_per_head_batch=max_cores,
        )

    return [
        ("A no-config (55 cores/head)", None),
        ("B explicit, default 16/head", cfg(16)),
        ("C explicit, 1 core/head", cfg(1)),
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
