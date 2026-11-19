# Initial operation topology (before tuning)

Read at baseline 5ebe6996 with prior raw reports and source. Target is B32 on 1x4 Blackhole, regression B1. Detailed independent audit follows in topology_audit.md.

| Site | Existing sequence | Material repeated work | Candidate family / constraints | First action |
|---|---|---|---|---|
| Full attention input | per-row local norm -> packed QKVG DRAM-sharded MM -> head split/norm/RoPE/SDPA | 32 norm and MM launches; Q/K/V/gate already packed | batch norm and packed projection; BF16/BFP8 HiFi2, preserve independent cache/positions | measure batch projection boundary |
| Linear input | per-row ordered norm -> ordered K/V/a/b projection and Q/Z projection | same activation read twice; FP32 ordered accumulation sensitive | combined ordered scan weight packing, preserve K-block512 | compare after CCL placement |
| Attention output | per-row output MM -> AR -> add -> concat | 32 ARs, each padded to32 rows; 32x128KiB BF16 or 32x256KiB FP32 carrier | concatenate local partials before one AR, or RS into hidden512 residual -> next norm | measure coherent batch AR and RS families |
| MoE | batch norm -> FP32 router/top8 -> local grouped dispatch -> active experts -> combine + shared packed GU/scalar -> reduce -> add | batch CCL RS+AG; combine writes top8 slots | persistent direct AR for <=32 logical rows; keep EP4 active work | integrate same collective family |
| Layer return | batched result ->32 singleton slices | 94 operations, ~200us in raw B32 profile | one row-major conversion, then singleton slices/tilize; or tensor residual contract | try shared conversion after batch collective |
| Sharded residual | RS -> hidden512 add -> distributed stats norm + normalized-input AG or ordered gather norm | AG before projections, not an immediate restore after RS | adapted fused MM-RS and AG-MM with persistent buffers; inherited B1 data exists, B32 remains to measure | compare after batch placement |

P=32*2048*2=131072 bytes/chip BF16; linear FP32 partial is 2P. RS rank traffic roughly .75P, RS+AG 1.5P; direct AR topology-dependent. Combining 32 rows before reduction collapses32 physical tiles to one; does not combine users mathematically. Existing shared BF16/FP32 direct-AR scratch is1.5MiB/chip; reuse serially, no extra persistent allocation. All output contracts must be accepted by next norm/attention/MLP; harness gather only outside timing. No dense all-expert execution is proposed.
