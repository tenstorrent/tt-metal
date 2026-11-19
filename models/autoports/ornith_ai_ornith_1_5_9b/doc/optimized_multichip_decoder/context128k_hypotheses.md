# 128K accuracy hypothesis ledger

Inspected by AutoFix hypothesis agents, no device activity in agents.

- Outer chunk restored1024 fails earlier: decode1,row16 PCC0.994835; final larger chunks fail decode6,row11 PCC0.991427. Not sufficient repair.
- No17-bit narrowing found. Every row starts131088–131091, already beyond131072. Failed positions131097/131090. SDPA interleaved DRAM page tables and position values are uint32; uint16 table reads require sharded tables, not this path.
- SDPA program factory uses BF16 im_df/stats_df despite fp32_dest_acc_en. B32/110cores assigns3cores/request; K64 yields about684 local online-softmax accumulation rounds at128K versus342 at64K, repeatedly repacking BF16 sums/output. This is a plausible numerical mechanism, not yet proven.
- K128-only vs HiFi4-only distinguishes accumulation depth from multiplication fidelity. K256/512 need8/16-page attention-table alignment, owned internally. Cache-update tables must remain unmodified.
- Outer chunk1024 also changes final projection geometry; does not restore whole stage4 arithmetic. Layerwise output or predecode state comparison is needed if SDPA controls fail.
