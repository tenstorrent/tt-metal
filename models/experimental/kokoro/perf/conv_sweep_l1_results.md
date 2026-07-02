# TextEncoder Conv1d — L1-config sweep (in=512 out=512 k=5 B=2 T=48, M=96, K=2560, N=512)

All rows use the production program config (block-sharded, abh=32, split_reader, both double buffers, LoFi + fp32_acc). `prod_dram` = current production (config tensors in DRAM); every other row pins `config_tensors_in_dram=False` (config tensors in L1). Each `L1_*` row flips ONE extra knob the original DRAM sweep never varied (or a combo).

- `conv_us` = dominant conv2d op; `total_us` = all device ops the conv emits (halo+conv2d+reshard).
- `wt_dtype` = conv weight precision. `cfg_dram` = config_tensors_in_dram. `reuse` = enable_activation_reuse. `pk_l1` = packer_l1_acc. `full_k` = full_inner_dim. `abw` = act_block_w_div (auto=default 1). `resh` = reshard_if_not_optimal. `tps` = transpose_shards.

**Fastest PCC-passing: `L1_bf8_packer` — 11.40µs conv (25.56µs total), PCC=0.9999 vs prod_dram 16.15µs**


**Findings:** `bf8_b` conv weights are the only real lever — 16.1→11.4µs (−29%) at PCC 0.99988 vs 0.99989 (weight re-read bandwidth dominates this M=96 conv, so halving the weight bytes wins; `packer_l1_acc` adds ~0 on top). Moving config tensors to L1 (`config_tensors_in_dram=False`, `L1_base`) is neutral vs DRAM (16.4 vs 16.3µs). `enable_activation_reuse` FATALs — not supported for block sharding. `full_inner_dim`, `packer_l1_acc`, `act_block_w_div`, `reshard_if_not_optimal` are all within noise; `transpose_shards` regresses.

**Shard layout (L1):** BLOCK stays the clear winner. WIDTH is ~29µs at best (`L1w_bf8`) — ~2.5× slower than block+bf8; bf8 barely helps width (it replicates spatial work across cores rather than re-reading weights), and `abh=32` DOUBLES width to ~53µs (opposite of block, where abh=32 helps), while `act_block_w_div` FATALs on width. HEIGHT sharding FATALs on every config in the sliding-window/halo op — a hard op-level limitation for this Conv1d-as-Conv2d shape, NOT L1- or bf8-fixable (matches the DRAM sweep).

| config | shard | wt_dtype | cfg_dram | abh | reuse | pk_l1 | full_k | abw | resh | tps | cores | #ops | conv_us | total_us | PCC | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_bf8_packer **(fastest)** | block | bf8_b | False | 32 | False | True | False | auto | False | False | 0 | 7 | 11.40 | 25.56 | 0.9999 | PASS |
| L1_wt_bf8 | block | bf8_b | False | 32 | False | False | False | auto | False | False | 0 | 7 | 11.51 | 25.68 | 0.9999 | PASS |
| L1_packer_l1acc | block | bf16 | False | 32 | False | True | False | auto | False | False | 0 | 7 | 16.09 | 30.70 | 0.9999 | PASS |
| L1_abwdiv4 | block | bf16 | False | 32 | False | False | False | 4 | False | False | 0 | 7 | 16.14 | 30.23 | 0.9999 | PASS |
| prod_dram | block | bf16 | True | 32 | False | False | False | auto | False | False | 0 | 7 | 16.15 | 30.64 | 0.9999 | PASS |
| L1_abwdiv2 | block | bf16 | False | 32 | False | False | False | 2 | False | False | 0 | 7 | 16.16 | 30.49 | 0.9999 | PASS |
| L1_reshard | block | bf16 | False | 32 | False | False | False | auto | True | False | 0 | 7 | 16.16 | 30.30 | 0.9999 | PASS |
| L1_base | block | bf16 | False | 32 | False | False | False | auto | False | False | 0 | 7 | 16.36 | 30.52 | 0.9999 | PASS |
| L1_full_inner | block | bf16 | False | 32 | False | False | True | auto | False | False | 0 | 7 | 16.50 | 30.65 | 0.9999 | PASS |
| L1_transpose | block | bf16 | False | 32 | False | False | False | auto | False | True | 0 | 7 | 26.71 | 41.08 | 0.9999 | PASS |
| L1w_bf8 | width | bf8_b | False | auto | False | False | False | auto | False | False | 0 | 7 | 29.16 | 43.36 | 0.9999 | PASS |
| L1w_bf8_packer | width | bf8_b | False | auto | False | True | False | auto | False | False | 0 | 7 | 29.30 | 43.38 | 0.9999 | PASS |
| L1w_bf16 | width | bf16 | False | auto | False | False | False | auto | False | False | 0 | 7 | 29.37 | 43.87 | 0.9999 | PASS |
| L1w_bf8_abh32 | width | bf8_b | False | 32 | False | False | False | auto | False | False | 0 | 7 | 52.64 | 66.94 | 0.9999 | PASS |
| L1w_bf16_abh32 | width | bf16 | False | 32 | False | False | False | auto | False | False | 0 | 7 | 52.85 | 67.03 | 0.9999 | PASS |
| L1_act_reuse | block | bf16 | False | 32 | True | False | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/ |
| L1_bf8_reuse | block | bf8_b | False | 32 | True | False | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/ |
| L1_bf8_reuse_packer | block | bf8_b | False | 32 | True | True | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/ |
| L1w_bf8_abwdiv2 | width | bf8_b | False | auto | False | False | False | 2 | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/ |
| L1h_bf16 | height | bf16 | False | auto | False | False | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| L1h_bf16_abh32 | height | bf16 | False | 32 | False | False | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
| L1h_bf8 | height | bf8_b | False | auto | False | False | False | auto | False | False | - | - | - | - | - | ERROR: TT_FATAL @ /home/ubuntu/samyuktha/tt-metal/ttnn/cpp/ttnn/operations/sliding_wind |
