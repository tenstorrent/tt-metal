# Primary K10 L1 repair for trial181

Failure retained in `prefill171_primary_k10_b32_s33.log`: the static dataflow-buffer region ended at1,632,192 bytes, exceeding1,572,864 bytes. Earlier main-CB estimate1,519,616 excluded reserved address space and ancillary allocations; it was explicitly marked tight/pending measurement. This is a physical allocation failure, not evidence against K10 arithmetic.

Use `prefill181_primary_k10_bh1_policy.json` on candidate178. It sets primary `in0_block_w=10,out_block_h=1` and explicitly preserves selected residual `in0_block_w=10,subblock_w=3`. Policies shallow-merge at construction: retain the residual entry when replacing the entire projection-config mapping. To isolate the original171 primary-only trial instead, set the exact residual entry to `{}`; that deliberately returns residual toK8 and is not the selected178 baseline.

Primary physical width16,640 on11×10 grid gives per-coreN48. At640 rows, per-coreM2. Repaired output block is1×48 with1×4 subblock, K10. Thus full worker output width remains intact; no partial-width DRAM reader path is introduced. At every shorter tail, the height ceiling remains legal. K5120 is160 tiles and divisible by10.

Native source evidence:

- `matmul_multicore_reuse_mcast_2d_program_factory.cpp:411-434`: A entries are `BH*K*2`; B entries are `PN*K*2`; output/partials each describe `BH*PN` entries.
- The same factory at758–803 shares output and intermediate storage for this interleaved FP32-output path. BF16 A tiles2048B, BFP8 B tiles1088B, FP32 output tiles4096B.
- `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:391-409`: the outer height loop explicitly resets DRAM K-read offset for each height block. Retaining full width satisfies the accompanying width-loop constraint.
- `dataflow_buffer.cpp:3077-3085`: the failure value is the region's ending address, not simply the sum of major CB sizes.

| Configuration | A double buffer | B double buffer | Shared output/partial | Main CB total |
|---|---:|---:|---:|---:|
| K10 BH2, failed | 81,920 | 1,044,480 | 393,216 | 1,519,616 |
| K10 BH1, proposed | 40,960 | 1,044,480 | 196,608 | 1,282,048 |

The observed failed end minus its main CB sum is112,576B, representing the fixed region base and other allocations. Applying that measured difference gives a repaired estimated end1,394,624B and178,240B headroom. This is a calibrated estimate, not a measured repaired allocation. Other simultaneously resident L1 tensors can still cause a distinct collision check; keep inputL1 disabled as in the control.

CPU geometry/estimated-capacity checks pass all1..640 row lengths for both projection roles (1,280 cases), saved in `prefill181_primary_k10_bh1_cpu_checks.json`. No decoder or frozen candidate edits and no hardware execution occurred. Performance is not guaranteed: BH1 rereads B for each row block and can offset the lower K-block count. Parent must run181 for correctness and timing before selection. Precision/fidelity, packer setting, weights, decode, and recurrence are unchanged.


## Trial181 measured disposition

Parent-reported181 measurements confirm the BH1 repair is physically legal. Compare against171 selected residual-only K10/SW3 (milliseconds):

| Shape | Residual-only selected | Primary K10/BH1 + residual K10/SW3 | Relative latency change |
|---|---:|---:|---:|
| B32 S33 | 24.379934 | 24.297887 | -0.337% |
| B32 S512 | 174.322562 | 174.669556 | +0.199% |
| B1 S512 | 6.270639 | 6.213501 | -0.911% |

The repaired primary config offers a bounded tradeoff: a small B1 gain and no B32 long-prefill gain. These are separate-run measurements; no repeated statistical significance is claimed. Residual-only K10/SW3 remains selected. Long decode correctness subsequently passed in primary181_sequence.json (see the prefill181 exit ledger); all four whole-layer gates passed. The candidate remains rejected on the measured prefill tradeoff. No additional candidate is justified by these timings. The failed171 allocation and legal181 repair are both retained as evidence that the optimization was repaired and measured before rejection.
