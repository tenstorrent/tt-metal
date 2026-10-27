# ttnn SDPA: wrong output at an odd q-chunk count > 1 (B = 16 / 32)

Found during the Tafsut bring-up (chip 3, Blackhole, tt-metal `fae9a1ea287`). Not filed upstream: filing is the
user's call (D1 ruling 6). This is the minimal repro to decide with.

## Call

`ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False, scale=1.0,
program_config=SDPAProgramConfig(grid, q_chunk_size=c, k_chunk_size=c, exp_approx_mode=False),
compute_kernel_config=HiFi4)`, with q, k, v `[B, 12, s_pad, 64]` bf16 tile layout, `randn * 0.6`, and `bias`
`[B, 1, s_pad, s_pad]` (0 / -1e9 key mask, or all zeros). Reference: torch SDPA on the same bf16-rounded inputs.
Metric: worst per-batch-row PCC over the first S valid rows.

## Result (job 932, script `/home/andy/bringup/scratch/tafsut/diag_sdpa2.py`; masked / unmasked)

| s_pad / chunk | count | B = 2 | B = 16 | B = 32 | B = 64 |
|---|---|---|---|---|---|
| 160 / 32 | 5 | 1.0000 | 0.9421 / 0.9725 | 0.9631 / 0.9845 | 1.0000 |
| 384 / 128 | 3 | 1.0000 | 0.9701 / 0.9854 | 0.9132 / 0.9623 | 1.0000 |
| 640 / 128 | 5 | 1.0000 | 0.9024 / 0.9790 | 0.9551 / 0.9892 | 1.0000 |
| 1088 / 64 | 17 | 1.0000 | 0.9547 / 0.9972 | 0.9560 / 0.9985 | 1.0000 |
| 1152 / 128 | 9 | 1.0000 | 0.9326 / 0.9918 | 0.9558 / 0.9945 | 1.0000 |

Every even count (or a single chunk) in the same sweep is 1.0000 at every B. The mask isn't needed to trigger it
(the unmasked column fails too), so the trigger is the odd chunk count together with the batch. A guess, not
checked in the source: the B x NH x q-chunk work split over the core grid leaves a remainder at these B that some
cores handle wrong.

Confirmed separately in jobs 933 and 939, and kept as a positive control in
`tests/pcc/test_sdpa_mask.py::test_sdpa_odd_chunk_count_bug` (384/128 and 1088/64 at B = 16; job 943: worst row
0.9796 and 0.9982, so it still reproduces).

## Workaround in the model

`reference/dataflow.py:PAD_TABLE` only uses chunks whose count is 1 or even, and `chunk_count_ok` asserts it
(CONTRACT §7 item 1, PLAN v1.21). Job 943 checks every table row at every batch the model issues (1 to 1024).
