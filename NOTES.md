# t25 notes (not committed) — DONE

Commit 39b5ca67409: DiffVAEOptions.production() builds the fused stage-5 qkv (5607 -> 5481 ms mean, bit-identical).
Rejected: DIFFVAE_NA_KV_BF8=1 (typecast re-impl: PCC 0.15, 5581 ms, broken + slower),
DIFFVAE_S5_MLP_FIDELITY=LoFi (45.49 dB, PCC 0.9995, 5540 ms vs 5535 fused, no gain). Knob diff kept in tmp/rejected_knobs.patch.
Logs tmp/ab_*.log. Pixel dumps and tmp/tt-metal-cache deleted.
