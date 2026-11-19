# Integrated8 cumulative candidate (not run)

Requires reviewed batched-cache ownership integration and its dedicated watcher test before selection. Six ordinary TP4 cases retain tiled_conv7 + affine6 + packed SwiGLU, current B1 legacy-prefill-convolution guard and source-construction caching, and enable batched cache fill. The flag is inert for GDN. The reviewed runner must prove initial and timed full-attention prefill actually execute bulk fills and reproduce the oracle-gated output/cache.

`integrated8_l1_compare.json` adds three full-attention L1 MLP cases against the matching new controls. The flat-prefill guard now permits rows>32 where this decode-only policy does not apply; this compares the whole retained L1 family without deliberately undoing prefill compaction. Neither manifest selects defaults. Preserve all old results. Final default gates remain separate.
