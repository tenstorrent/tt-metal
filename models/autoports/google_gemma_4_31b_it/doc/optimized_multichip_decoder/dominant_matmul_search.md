# Dominant decode matmul search

All rows use real TP4 shapes and BFP4/LoFi. Percentages are diagnostic matmul contributions to the final whole-layer device window, averaged across devices; gate/up is their sum. Trial latency is whole-layer traced host time, so it is not compared numerically to a device subtotal. Full candidate layouts and source hashes are in the named JSON/manifests.

| Kind | Role (% of layer) | Sender cores / K block tried | Layer trace us | Prefill / decode PCC | Decision / evidence |
|---|---|---|---:|---|---|
| sliding_attention | QKV (7.39%) | 28 / 6, 2 readers | 427.06 | 0.99991582 / 0.99992573 | Kept final default; final_layer0_batch1.json |
| sliding_attention | QKV (7.39%) | 14 / 12, 2 readers | 438.36 | 0.99991959 / 0.99993603 | Rejected: slower than final cumulative family; q12_layer0_batch1.json |
| sliding_attention | QKV (7.39%) | 7 / 24, 2 readers | 429.13 | 0.99991959 / 0.99993465 | Rejected: slower than final cumulative family; q24_layer0_batch1.json |
| sliding_attention | O (4.91%) | 8 / 8, 2 readers | 427.06 | 0.99991582 / 0.99992573 | Kept final default; final_layer0_batch1.json |
| sliding_attention | O (4.91%) | 4 / 16, 2 readers | 435.43 | 0.99991959 / 0.99992352 | Rejected: slower than final cumulative family; o16_layer0_batch1.json |
| sliding_attention | gate/up (19.45%) | 28 / 6, 2 readers | 427.06 | 0.99991582 / 0.99992573 | Kept final default; final_layer0_batch1.json |
| sliding_attention | gate/up (19.45%) | 14 / 12, 2 readers | 472.51 | 0.99991959 / 0.99992750 | Rejected: slower than final cumulative family; gate12_layer0_batch1.json |
| sliding_attention | gate/up (19.45%) | 7 / 24, 2 readers | 476.87 | 0.99991959 / 0.99992656 | Rejected: slower than final cumulative family; gate24_layer0_batch1.json |
| sliding_attention | down (10.91%) | 8 / 11, 2 readers | 427.06 | 0.99991582 / 0.99992573 | Kept final default; final_layer0_batch1.json |
| sliding_attention | down (10.91%) | 8 / 22, 2 readers | 430.16 | 0.99991959 / 0.99992963 | Rejected: slower than final cumulative family; down22_layer0_batch1.json |
| full_attention | QK (7.98%) | 8 / 21, 2 readers | 475.06 | 0.99998989 / 0.99999201 | Kept final default; final_layer5_batch1.json |
| full_attention | QK (7.98%) | 14 / 12, 2 readers | 486.28 | 0.99998997 / 0.99999269 | Rejected: slower than final cumulative family; q12_layer5_batch1.json |
| full_attention | QK (7.98%) | 7 / 24, 2 readers | 480.19 | 0.99998997 / 0.99999273 | Rejected: slower than final cumulative family; q24_layer5_batch1.json |
| full_attention | O (7.56%) | 8 / 16, 2 readers | 475.06 | 0.99998989 / 0.99999201 | Kept final default; final_layer5_batch1.json |
| full_attention | O (7.56%) | 4 / 32, 2 readers | 499.25 | 0.99998997 / 0.99998904 | Rejected: slower than final cumulative family; o16_layer5_batch1.json |
| full_attention | gate/up (17.11%) | 28 / 6, 2 readers | 475.06 | 0.99998989 / 0.99999201 | Kept final default; final_layer5_batch1.json |
| full_attention | gate/up (17.11%) | 14 / 12, 2 readers | 528.63 | 0.99998997 / 0.99999265 | Rejected: slower than final cumulative family; gate12_layer5_batch1.json |
| full_attention | gate/up (17.11%) | 7 / 24, 2 readers | 541.84 | 0.99998997 / 0.99999126 | Rejected: slower than final cumulative family; gate24_layer5_batch1.json |
| full_attention | down (9.66%) | 8 / 11, 2 readers | 475.06 | 0.99998989 / 0.99999201 | Kept final default; final_layer5_batch1.json |
| full_attention | down (9.66%) | 8 / 22, 2 readers | 494.44 | 0.99998997 / 0.99999225 | Rejected: slower than final cumulative family; down22_layer5_batch1.json |

Full QK7/K24 additionally crossed the selected norm/pool/L1-slice family in norm8poolq24_layer5_batch1.json:478.77us vs475.04us, passing PCC; rejected. Reader counts have separate alternating-order microprofiles and stage04 whole-layer controls. Prefill M4/K8/N16/subblock1×8, larger-K/grid/padding trials and exact K21 L1 failure are recorded in optimization_audit.md and the stage04 artifacts.
