# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Classify the real-weight `prefill[linear]` max-abs outlier (README section 3.2).

`pcc_real_weights.jsonl` records `maxabs = 1.2687` for `prefill[linear] seq=1024` while the
same case with synthetic weights is 0.0449 and HF's own bf16-vs-fp32 divergence on these real
weights is 0.0706 — 18-28x, even though rel-RMS is only 1.06% and PCC 0.9999435. A max-abs
outlier that large is either expected (the error sits on a large-magnitude element, so it is small
*relatively*) or it is a real localized defect, and quoting only PCC and rel-RMS cannot tell them
apart. This measures which.

For the worst element it reports: the reference magnitude there, the relative error there, how the
reference magnitude at that position compares to the tensor as a whole, and how much of the total
squared error the worst token accounts for. It also checks the one mechanism the stage already
documents as a candidate (section 5.1: bf16 `ttnn.topk` swaps a low-weight expert for ~6% of
tokens) by comparing fp32 and bf16 top-k expert sets for the worst token.

Needs the real checkpoint and the device.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_real_weight_maxabs.py
"""

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import build_layer_pair, from_tt, to_tt_prefill
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

SEQ = 1024
SEED = 55  # the seed test_real_weights_prefill_and_decode uses, so this is the same input
#: The decisive control. `linear_attention` carries a running recurrent state across
#: `delta_chunk_size`=64 chunks, so if the outlier is state accumulation the error must grow with
#: sequence length; at seq 64 there is exactly one chunk and no carry at all. `full` is the
#: no-recurrence control at the same weights and length.
#: 2048 and 4096 are the ones that test *boundedness* rather than growth: if the error kept
#: compounding with chunk count the maxabs would keep climbing past 1024's 1.27, whereas a decaying
#: recurrence ages old error out and the curve flattens. The earlier write-up argued boundedness from
#: a *synthetic*-weight 262143-token tail number, which is a different weight set and a different
#: comparison window, so it did not support the claim.
SEQ_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096]


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pair = build_layer_pair(
            device, kind="linear", max_batch_size=2, supported_context=max(SEQ_SWEEP), real_weights=True
        )
        pair.tt.reset_state()
        x = ref.synthetic_hidden_states(pair.hf_config, 1, SEQ, seed=SEED)

        tt_x = to_tt_prefill(device, x)
        out = pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table)
        got = from_tt(out).float().reshape(1, SEQ, -1)
        ttnn.deallocate(out)
        ttnn.deallocate(tt_x)

        want = ref.hf_prefill(pair.hf, pair.hf_config, x, start_pos=0, cache=ref.make_cache(pair.hf_config)).output
        want = want.float().reshape(1, SEQ, -1)

        err = (got - want).abs()
        flat = int(err.argmax())
        tok, ch = divmod(flat, err.shape[-1])
        w = float(want[0, tok, ch])
        g = float(got[0, tok, ch])

        print(f"MAXABS  worst element: token {tok} channel {ch}")
        print(
            f"MAXABS  want {w:+.4f}  got {g:+.4f}  abs_err {abs(g - w):.4f}  rel_err {abs(g - w) / max(abs(w), 1e-9):.4%}"
        )
        print(
            f"MAXABS  |want| at that element {abs(w):.4f} vs tensor max |want| {float(want.abs().max()):.4f} "
            f"({abs(w) / float(want.abs().max()):.1%} of it)"
        )
        print(
            f"MAXABS  channel {ch} max |want| over tokens: {float(want[0, :, ch].abs().max()):.4f}; "
            f"median channel max: {float(want.abs().amax(dim=1).median()):.4f}"
        )

        # how concentrated is the error?
        tok_sq = (err[0] ** 2).sum(dim=-1)
        total = float(tok_sq.sum())
        print(
            f"MAXABS  worst token holds {float(tok_sq[tok]) / total:.1%} of the total squared error; "
            f"top-8 tokens hold {float(tok_sq.topk(8).values.sum()) / total:.1%}"
        )
        big = (err[0] > 0.2).nonzero()
        print(
            f"MAXABS  elements with abs err > 0.2: {len(big)} of {err[0].numel()} "
            f"across {len(set(int(i) for i in big[:, 0]))} tokens"
        )

        # the mechanism the stage already documents: bf16 top-k picking a different expert set
        with torch.no_grad():
            normed = pair.hf.post_attention_layernorm(x)
            logits, _, idx32 = pair.hf.mlp.gate(normed.reshape(-1, pair.hf_config.hidden_size))
            k = pair.hf_config.num_experts_per_tok
            sel32 = idx32.sort(dim=-1).values
            sel16 = logits.to(torch.bfloat16).float().topk(k, dim=-1).indices.sort(dim=-1).values
        differs = (sel32 != sel16).any(dim=-1)
        worst_tokens = [int(i) for i in tok_sq.topk(8).indices]
        print(
            f"MAXABS  bf16 top-k differs for {int(differs.sum())} of {differs.numel()} tokens "
            f"({float(differs.double().mean()):.1%})"
        )
        print(
            f"MAXABS  worst-8 tokens with a differing expert set: "
            f"{sum(1 for t in worst_tokens if bool(differs[t]))} of 8  (tokens {worst_tokens})"
        )
        print(f"MAXABS  worst token {tok} expert set differs: {bool(differs[tok])}")
        # ---- does the error grow with the recurrence? ----
        tok_err = err[0].amax(dim=-1)
        n = SEQ // 8
        print(
            "MAXABS  per-eighth max abs err: " + "  ".join(f"{float(tok_err[i*n:(i+1)*n].max()):.3f}" for i in range(8))
        )

        print("MAXABS  --- sequence-length sweep, real weights, same seed ---")
        for seq in SEQ_SWEEP:
            pair.tt.reset_state()
            xs = ref.synthetic_hidden_states(pair.hf_config, 1, seq, seed=SEED)
            tt_xs = to_tt_prefill(device, xs)
            o = pair.tt.prefill_forward(tt_xs, user_id=0, page_table=pair.page_table)
            g = from_tt(o).float().reshape(1, seq, -1)
            ttnn.deallocate(o)
            ttnn.deallocate(tt_xs)
            w = ref.hf_prefill(pair.hf, pair.hf_config, xs, start_pos=0, cache=ref.make_cache(pair.hf_config)).output
            w = w.float().reshape(1, seq, -1)
            chunks = -(-seq // pair.cfg.delta_chunk_size)
            print(
                f"MAXABS  seq={seq:5} ({chunks:2} delta chunks)  maxabs={float((g-w).abs().max()):.4f}  "
                f"pcc={ref.pcc(g, w):.7f}  max|want|={float(w.abs().max()):.3f}"
            )
        pair.tt.release()

        # the no-recurrence control at the same weights/length
        fpair = build_layer_pair(
            device, kind="full", max_batch_size=2, supported_context=max(SEQ_SWEEP), real_weights=True
        )
        fpair.tt.reset_state()
        tt_fx = to_tt_prefill(device, x)
        o = fpair.tt.prefill_forward(tt_fx, user_id=0, page_table=fpair.page_table)
        g = from_tt(o).float().reshape(1, SEQ, -1)
        ttnn.deallocate(o)
        ttnn.deallocate(tt_fx)
        w = ref.hf_prefill(fpair.hf, fpair.hf_config, x, start_pos=0, cache=ref.make_cache(fpair.hf_config)).output
        w = w.float().reshape(1, SEQ, -1)
        print(
            f"MAXABS  full (no recurrence) seq={SEQ}  maxabs={float((g-w).abs().max()):.4f}  "
            f"pcc={ref.pcc(g, w):.7f}  max|want|={float(w.abs().max()):.3f}"
        )
        fpair.tt.release()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
