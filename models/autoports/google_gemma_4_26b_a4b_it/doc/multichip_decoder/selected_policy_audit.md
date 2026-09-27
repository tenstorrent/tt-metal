# Selected runtime policy

Runtime a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255 reproduces the passing candidates with no experimental flags. Headline evidence is sliding_final_policy_raw_stress.json and full_final_policy_stress.json; both exercise 4096 input tokens and 128 advancing decode positions at batch/concurrency 1, eight duplicate replays per position, local paged cache PCC and all-rank equality. Stack evidence is stack_final_policy.json. Wider acceptance is recorded in the README.

| Role | Sliding | Full |
| --- | --- | --- |
| Local Q/KV heads, width | 4/2, 256 | 4/1, 512 (KV duplicated rank pairs) |
| QKV decode | BFP8, LoFi, FP32 accumulation/output; 8x4 N2 K22 | BFP8, LoFi, FP32 accumulation/output; 8x6 N2 K22 |
| WO decode | BFP8, LoFi, 11x8 N1 K32 | BFP8, LoFi, 11x8 N1 K8 |
| Attention collective | BF16 | BFP8 |
| Decode RoPE | homogeneous BF16 sharded D256 | original FP32 interleaved |
| Indexed active8 experts | BFP4 GU/down, BFP8 input, LoFi | BFP4 GU/down, BFP8 input, LoFi |
| Expert packing | raw checkpoint host BFP4 GU; device converted down | original device conversion |
| Expert GU/down geometry | 6x2 N1 K44 / 11x8 N1 K6 | same |
| Shared decode GU/down | BFP4 / BFP8, LoFi | BFP4 / BFP4, LoFi |
| Shared decode geometry | GU9x2 N2 K88, down11x4 N2 K17 | GU11x4 N1 K44, down11x4 N2 K17 |
| Grouped shared/routed collective | BFP8 | BF16 |
| Prefill experts | EP4 dynamic active union, BFP8 GU/BFP4 down | EP4 dynamic active union, BFP4 GU/down |
| Shared prefill | retained BF16 | retained BF16 |
| Interface | replicated BF16 [1,1,S,2816] | same |

Both kinds use the fixed 1x4 Linear fabric mesh, router buffers at (10,9), local BFP8 absolute paged caches, grouped independent shared/routed reductions and fused tail norms. The minimum worker grid is explicitly 11x10. The router core lies outside fixed compute grids; it is not globally reserved against automatic layout kernels.

Factory defaults resolve layer-specific geometry, RoPE and grouped-CCL policy when arguments are None. All three validation harnesses preserve those defaults. Explicit flags remain reproducible experiments. Constructor packing is outside forward and trace. The selected hybrid retains separate EP prefill weights; the indexed TP prefill slot aliases the BFP4 decode gate because it is unused. The nonhybrid experiment still retains its separate sliding BFP8 prefill gate and cannot use the selected hybrid capacity plan.

See AUTOFIX_final_policy_packing.md, AUTOFIX_sharded_rope.md, AUTOFIX_full_router1_bfp8.md for measured corrections. selected_policy_proposal_historical.md and selected_policy.patch retain the earlier proposal; they do not describe current defaults. Matching native profiler rows and final checks are required before stage acceptance.
