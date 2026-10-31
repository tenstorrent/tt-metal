# Prefill input-L1 advice

Real pinned weights, 4096 tokens, TP4, selected precision, whole-layer warmed host timing. Baselines are the final default acceptance runs for the same batch and layer. Device headline results remain in performance.md.

| Trial artifact | Default prefill us | Trial prefill us | Prefill PCC | Decode PCC | Decision |
|---|---:|---:|---:|---:|---|
| input_l1_chunks_layer0_batch16.json | 160360.19 | 179023.37 | 0.99992388 | 0.99994234 | Slower; rejected |
| input_l1_chunks_layer0_batch32.json | 319634.42 | 388000.96 | 0.99992065 | 0.99994962 | Slower; rejected |
| input_l1_chunks_layer0_batch8.json | 80180.19 | 88946.96 | 0.99992365 | 0.99995400 | Slower; rejected |
| input_l1_chunks_layer5_batch16.json | 192066.60 | 211467.38 | 0.99998994 | 0.99998847 | Slower; rejected |
| input_l1_chunks_layer5_batch32.json | 386401.83 | 541158.66 | 0.99998903 | 0.99999075 | Slower; rejected |
| input_l1_chunks_layer5_batch8.json | 96867.56 | 105056.64 | 0.99998992 | 0.99999244 | Slower; rejected |
| input_l1_down_m1_layer0_batch8.json | 80180.19 | 98288.31 | 0.99992323 | 0.99995395 | Slower; rejected |
| input_l1_down_m1_layer5_batch8.json | 96867.56 | 116665.95 | 0.99998995 | 0.99999242 | Slower; rejected |
| input_l1_down_small_layer0_batch16.json | 160360.19 | 212229.74 | 0.99992386 | 0.99994163 | Slower; rejected |
| input_l1_down_small_layer5_batch16.json | 192066.60 | 246389.82 | 0.99998988 | 0.99998850 | Slower; rejected |
| input_l1_mlp_m1n8_layer0_batch8.json | 80180.19 | 114918.24 | 0.99992314 | 0.99995395 | Slower; rejected |
| input_l1_mlp_m1n8_layer5_batch8.json | 96867.56 | 130591.56 | 0.99998991 | 0.99999242 | Slower; rejected |
| input_l1_o_layer0_batch8.json | 80180.19 | 80644.90 | 0.99992323 | 0.99995395 | Slower; rejected |
| input_l1_o_m1_layer0_batch16.json | 160360.19 | 171253.85 | 0.99992404 | 0.99994163 | Slower; rejected |
| input_l1_o_m1_layer5_batch8.json | 96867.56 | 109197.14 | 0.99998995 | 0.99999242 | Slower; rejected |
| input_l1_q_m1n8_layer0_batch8.json | 80180.19 | 91390.88 | 0.99991890 | 0.99994483 | Slower; rejected |
| input_l1_q_m1n8_layer5_batch8.json | 96867.56 | 110253.57 | 0.99998980 | 0.99999253 | Slower; rejected |
| batch_l1_chunks_layer0_batch16.json | 160360.19 | 189855.88 | 0.99991884 | 0.99993128 | Slower; rejected |
| batch_l1_chunks_layer0_batch32.json | 319634.42 | 422325.39 | 0.99991609 | 0.99993910 | Slower; rejected |
| batch_l1_chunks_layer0_batch8.json | 80180.19 | 87990.80 | 0.99991899 | 0.99995539 | Slower; rejected |
| batch_l1_chunks_layer5_batch16.json | 192066.60 | 269850.10 | 0.99998901 | 0.99998823 | Slower; rejected |
| batch_l1_chunks_layer5_batch32.json | 386401.83 | 607410.58 | 0.99998849 | 0.99999158 | Slower; rejected |
| batch_l1_chunks_layer5_batch8.json | 96867.56 | 106391.49 | 0.99998984 | 0.99999163 | Slower; rejected |

Single-role trials retain the default residual contract and charge conversion overhead. Shared MLP input is converted once for both gate and up. Q/MLP at B8 initially collide with static circular buffers, so M4/K8/N16 is adapted to M1/K8/N8; down uses M1 or M1/K4/N4 at B16. Full O at B8 is retried at M1 after M4 fails. All adapted runnable controls pass and are slower. Exact overlaps and the warmed B16 full-O 128-MiB allocation failure are in input_l1_failures.json.

The 110-bank allocator has 1461120 bytes per bank. B16 Q/MLP and B32 Q/MLP/full-O/down input-only tile-rounded requirements exceed that physical capacity. Full O at B16 (equivalently sliding O at B32) needs 1220608 bytes per bank; the warmed probe's largest contiguous block is only 1114752 bytes, despite 1329792 total free bytes. This is a contiguous allocation limit, not a claim that total bytes alone exceed capacity. input_l1_capacity.json records every role/batch bound.

Two coherent streamed families adapt around those limits at all B8/B16/B32 and both layer kinds. batch_l1_chunks limits physical chunk rows to 1024 and carries activations/norms through the L1 family. input_l1_chunks uses 2048 rows, original M4/K8/N16 matmul geometry with all five projection inputs in L1 and shared gate/up input, retaining DRAM output/norm contract. Both own internal subdivision and concatenate logical results; neither requires aligned public inputs or changes the production path. Both pass PCC but lose whole-layer latency. No input-L1 optimization is rejected solely at its first allocation/API error.
