<!-- lb2 176985 (bh-lb-120-a07u28, 2026-11-05): the 1D leftover order, port-alloc 9ce0acbeaba staged against tier; alloc_report.py --prefix p1d; SLOT_ALLOCATOR.md section 12.15 -->
Slot rows per run (each checked against the mock vet):
   ##### p1da_staged_pval1d_4352 (1 tests x1; job at 3 s)
   === p1da_staged_pval1d_4352: fabric_exp/port_slots/yamls/pval1d_4352.yaml (timeout 300 s)
   ##### p1da_staged_pval1d_3072 (1 tests x1; job at 18 s)
   === p1da_staged_pval1d_3072: fabric_exp/port_slots/yamls/pval1d_3072.yaml (timeout 300 s)
   ##### p1da_staged_pvline_4352 (38 tests x1; job at 30 s)
   === p1da_staged_pvline_4352: fabric_exp/port_slots/yamls/pvline_4352.yaml (timeout 300 s)
   ##### p1da_staged_pvline_3072 (38 tests x1; job at 47 s)
   === p1da_staged_pvline_3072: fabric_exp/port_slots/yamls/pvline_3072.yaml (timeout 300 s)
   ##### p1da_staged_pvring_4352 (19 tests x1; job at 56 s)
   === p1da_staged_pvring_4352: fabric_exp/port_slots/yamls/pvring_4352.yaml (timeout 300 s)
   ##### p1da_staged_pvring_3072 (19 tests x1; job at 74 s)
   === p1da_staged_pvring_3072: fabric_exp/port_slots/yamls/pvring_3072.yaml (timeout 300 s)
   ##### p1da_staged_1d_4352 (6 tests x1; job at 87 s)
   === p1da_staged_1d_4352: fabric_exp/slotstudy/sls_1d_p150.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {26, 25} receiver slots {32}
   ##### p1da_tier_1d_4352 (6 tests x1; job at 96 s)
   === p1da_tier_1d_4352: fabric_exp/slotstudy/sls_1d_p150.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {16, 16} receiver slots {32}
   ##### p1da_staged_1d_3072 (6 tests x1; job at 112 s)
   === p1da_staged_1d_3072: fabric_exp/consolidation/slotalloc/yamls/sls_1d_p150_3072.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {43, 42} receiver slots {32}
   ##### p1da_tier_1d_3072 (6 tests x1; job at 119 s)
   === p1da_tier_1d_3072: fabric_exp/consolidation/slotalloc/yamls/sls_1d_p150_3072.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {32, 32} receiver slots {32}
   ##### p1da_staged_ring_4352 (14 tests x1; job at 130 s)
   === p1da_staged_ring_4352: fabric_exp/v4/exp_ring.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {26, 25} receiver slots {32}
   ##### p1da_tier_ring_4352 (14 tests x1; job at 141 s)
   === p1da_tier_ring_4352: fabric_exp/v4/exp_ring.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {16, 16} receiver slots {32}
   ##### p1da_staged_ring_3072 (14 tests x1; job at 157 s)
   === p1da_staged_ring_3072: fabric_exp/consolidation/slotalloc/yamls/exp_ring_3072.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {43, 42} receiver slots {32}
   ##### p1da_tier_ring_3072 (14 tests x1; job at 166 s)
   === p1da_tier_ring_3072: fabric_exp/consolidation/slotalloc/yamls/exp_ring_3072.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {32, 32} receiver slots {32}
   ##### p1db_tier_ring_3072 (14 tests x1; job at 179 s)
   === p1db_tier_ring_3072: fabric_exp/consolidation/slotalloc/yamls/exp_ring_3072.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {32, 32} receiver slots {32}
   ##### p1db_staged_ring_3072 (14 tests x1; job at 189 s)
   === p1db_staged_ring_3072: fabric_exp/consolidation/slotalloc/yamls/exp_ring_3072.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {43, 42} receiver slots {32}
   ##### p1db_tier_ring_4352 (14 tests x1; job at 198 s)
   === p1db_tier_ring_4352: fabric_exp/v4/exp_ring.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {16, 16} receiver slots {32}
   ##### p1db_staged_ring_4352 (14 tests x1; job at 207 s)
   === p1db_staged_ring_4352: fabric_exp/v4/exp_ring.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {26, 25} receiver slots {32}
   ##### p1db_tier_1d_3072 (6 tests x1; job at 216 s)
   === p1db_tier_1d_3072: fabric_exp/consolidation/slotalloc/yamls/sls_1d_p150_3072.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {32, 32} receiver slots {32}
   ##### p1db_staged_1d_3072 (6 tests x1; job at 223 s)
   === p1db_staged_1d_3072: fabric_exp/consolidation/slotalloc/yamls/sls_1d_p150_3072.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {32, 0} receiver slots {32} | VC0 sender slots {43, 42} receiver slots {32}
   ##### p1db_tier_1d_4352 (6 tests x1; job at 230 s)
   === p1db_tier_1d_4352: fabric_exp/slotstudy/sls_1d_p150.yaml (timeout 600 s) TT_FABRIC_EXP_TIER_SLOTS=1
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {16, 16} receiver slots {32}
   ##### p1db_staged_1d_4352 (6 tests x1; job at 237 s)
   === p1db_staged_1d_4352: fabric_exp/slotstudy/sls_1d_p150.yaml (timeout 600 s)
      rows as vetted: VC0 sender slots {16, 0} receiver slots {32} | VC0 sender slots {26, 25} receiver slots {32}

## 1d_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 114.7 | 114.8 | 94.7 | 94.7 | -17.5% | -17.5% | -17.4% |
| SlsLin1HopEW_size_4352_ntype_unicast_write dev0 E | 129.8 | 129.9 | 124.5 | 124.5 | -4.1% | -4.1% | -4.2% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 132.4 | 132.4 | 118.4 | 118.4 | -10.5% | -10.5% | -10.5% |
| SlsLin3HopEW_size_4352_ntype_unicast_write dev0 E | 142.0 | 141.9 | 124.5 | 124.5 | -12.3% | -12.3% | -12.2% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 109.7 | 109.7 | 108.8 | 108.8 | -0.8% | -0.8% | -0.8% |
| SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | 126.9 | 126.9 | 125.7 | 125.8 | -0.9% | -0.9% | -0.9% |

## ring_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 224.3 | 224.4 | 224.0 | 224.0 | -0.1% | -0.1% | -0.2% |
| HalfRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 172.1 | 172.1 | 171.8 | 171.8 | -0.2% | -0.2% | -0.2% |
| Ring1HopEWBidir_size_4352_ntype_unicast_scatter_write dev1 W | 179.3 | 179.4 | 179.3 | 179.3 | -0.0% | -0.0% | -0.0% |
| Ring1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 140.0 | 140.1 | 140.0 | 140.1 | +0.0% | +0.0% | +0.0% |
| Ring1HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.0 | 179.0 | 179.0 | 179.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopEW_size_4352_ntype_unicast_write dev0 E | 140.1 | 140.1 | 140.1 | 140.1 | -0.0% | -0.0% | -0.0% |
| Ring1HopNS_size_4352_ntype_unicast_scatter_write dev0 S | 179.0 | 179.0 | 179.0 | 179.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopNS_size_4352_ntype_unicast_write dev0 S | 140.2 | 140.2 | 140.2 | 140.2 | +0.0% | +0.0% | +0.0% |
| Ring3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.6 | 179.6 | 179.7 | 179.6 | +0.0% | +0.1% | -0.0% |
| Ring3HopEW_size_4352_ntype_unicast_write dev0 E | 165.3 | 165.4 | 162.9 | 162.9 | -1.5% | -1.5% | -1.5% |
| RingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 538.8 | 538.8 | 538.8 | 538.8 | -0.0% | -0.0% | -0.0% |
| RingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 440.1 | 440.0 | 437.3 | 437.4 | -0.6% | -0.6% | -0.6% |
| UnidirRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 529.8 | 529.7 | 527.9 | 527.9 | -0.3% | -0.3% | -0.3% |
| UnidirRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 439.9 | 440.0 | 437.6 | 437.6 | -0.5% | -0.5% | -0.5% |

## 1d_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 93.1 | 93.1 | 92.8 | 92.8 | -0.3% | -0.3% | -0.3% |
| SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | 93.5 | 93.5 | 93.2 | 93.2 | -0.3% | -0.3% | -0.3% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 115.0 | 115.0 | 114.1 | 114.1 | -0.8% | -0.8% | -0.8% |
| SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | 115.5 | 115.5 | 114.4 | 114.4 | -0.9% | -0.9% | -0.9% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 107.3 | 107.3 | 107.2 | 107.1 | -0.2% | -0.1% | -0.2% |
| SlsLinRowMerge_size_3072_ntype_unicast_write dev2 E | 107.6 | 107.6 | 107.5 | 107.5 | -0.1% | -0.1% | -0.1% |

## ring_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 211.9 | 211.9 | 211.9 | 211.9 | -0.0% | -0.0% | -0.0% |
| HalfRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 172.4 | 172.4 | 172.3 | 172.3 | -0.1% | -0.1% | -0.1% |
| Ring1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 180.0 | 180.0 | 179.9 | 180.0 | -0.0% | -0.1% | +0.0% |
| Ring1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 141.0 | 141.0 | 141.0 | 141.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.7 | 179.8 | 179.7 | 179.7 | -0.0% | -0.0% | -0.0% |
| Ring1HopEW_size_3072_ntype_unicast_write dev0 E | 141.3 | 141.2 | 141.2 | 141.2 | -0.1% | -0.1% | -0.0% |
| Ring1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 179.7 | 179.7 | 179.7 | 179.7 | -0.0% | -0.0% | -0.0% |
| Ring1HopNS_size_3072_ntype_unicast_write dev0 S | 141.1 | 141.3 | 141.2 | 141.3 | -0.0% | +0.0% | -0.0% |
| Ring3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.8 | 179.8 | 179.7 | 179.7 | -0.1% | -0.1% | -0.0% |
| Ring3HopEW_size_3072_ntype_unicast_write dev0 E | 164.4 | 164.4 | 164.0 | 164.1 | -0.2% | -0.2% | -0.2% |
| RingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 541.0 | 541.1 | 541.0 | 541.1 | +0.0% | +0.0% | -0.0% |
| RingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 423.7 | 423.6 | 423.7 | 423.7 | +0.0% | -0.0% | +0.0% |
| UnidirRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 477.5 | 477.4 | 477.0 | 477.0 | -0.1% | -0.1% | -0.1% |
| UnidirRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 399.3 | 399.2 | 398.7 | 398.8 | -0.1% | -0.2% | -0.1% |

## Per suite (STAGED against TIER, mean of the passes; mixed tests by link total)

| suite | rows | wins (< -1%) | losses (> +1%) | median | best | worst | acceptance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1d_4352 | 6 | 4 | 0 | -7.3% | -17.5% SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | -0.8% SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | met |
| ring_4352 | 14 | 1 | 0 | -0.0% | -1.5% Ring3HopEW_size_4352_ntype_unicast_write dev0 E | +0.0% Ring3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | met |
| 1d_3072 | 6 | 0 | 0 | -0.3% | -0.9% SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | -0.1% SlsLinRowMerge_size_3072_ntype_unicast_write dev2 E | met |
| ring_3072 | 14 | 0 | 0 | -0.0% | -0.2% Ring3HopEW_size_3072_ntype_unicast_write dev0 E | +0.0% RingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | met |

## The new 1D rows against the capped build's (cross-job, same host)

Same host (a07u28). Old staged = lb2 176714 (24,27,32 / 24,61,32); new staged = lb2 176985 (26,25,32 / 43,42,32). Mean of passes a and b, cycles per packet.

## 1d_4352

| row | tier old | tier new | tier drift | staged old | staged new | new vs old staged | new vs tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 114.8 | 114.8 | -0.0% | 95.3 | 94.7 | -0.6% | -17.5% |
| SlsLin1HopEW_size_4352_ntype_unicast_write dev0 E | 129.9 | 129.8 | -0.1% | 124.5 | 124.5 | -0.0% | -4.1% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 132.4 | 132.4 | -0.1% | 118.3 | 118.4 | +0.1% | -10.5% |
| SlsLin3HopEW_size_4352_ntype_unicast_write dev0 E | 142.0 | 141.9 | -0.1% | 124.5 | 124.5 | +0.0% | -12.3% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 109.7 | 109.7 | -0.0% | 108.7 | 108.8 | +0.1% | -0.8% |
| SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | 126.9 | 126.9 | +0.0% | 125.8 | 125.7 | -0.0% | -0.9% |

## 1d_3072

| row | tier old | tier new | tier drift | staged old | staged new | new vs old staged | new vs tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 93.1 | 93.1 | -0.0% | 94.4 | 92.8 | -1.7% | -0.3% |
| SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | 93.5 | 93.5 | -0.0% | 95.2 | 93.2 | -2.0% | -0.3% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 115.0 | 115.0 | +0.0% | 114.3 | 114.1 | -0.1% | -0.8% |
| SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | 115.5 | 115.5 | -0.0% | 114.7 | 114.4 | -0.2% | -0.9% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 107.3 | 107.3 | -0.0% | 107.0 | 107.1 | +0.1% | -0.2% |
| SlsLinRowMerge_size_3072_ntype_unicast_write dev2 E | 107.7 | 107.6 | -0.1% | 107.5 | 107.5 | +0.0% | -0.1% |

## ring_4352

| row | tier old | tier new | tier drift | staged old | staged new | new vs old staged | new vs tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 224.3 | 224.3 | +0.0% | 224.0 | 224.0 | -0.0% | -0.1% |
| HalfRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 172.1 | 172.1 | -0.0% | 171.7 | 171.8 | +0.1% | -0.2% |
| Ring1HopEWBidir_size_4352_ntype_unicast_scatter_write dev1 W | 179.3 | 179.3 | -0.0% | 179.3 | 179.3 | +0.0% | -0.0% |
| Ring1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 140.1 | 140.1 | -0.0% | 140.0 | 140.1 | +0.0% | +0.0% |
| Ring1HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.0 | 179.0 | -0.0% | 179.0 | 179.0 | -0.0% | -0.0% |
| Ring1HopEW_size_4352_ntype_unicast_write dev0 E | 140.2 | 140.1 | -0.0% | 140.2 | 140.1 | -0.1% | -0.0% |
| Ring1HopNS_size_4352_ntype_unicast_scatter_write dev0 S | 179.0 | 179.0 | +0.0% | 179.0 | 179.0 | -0.0% | -0.0% |
| Ring1HopNS_size_4352_ntype_unicast_write dev0 S | 140.1 | 140.2 | +0.0% | 140.1 | 140.2 | +0.0% | +0.0% |
| Ring3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.7 | 179.6 | -0.1% | 179.5 | 179.6 | +0.1% | +0.0% |
| Ring3HopEW_size_4352_ntype_unicast_write dev0 E | 165.3 | 165.3 | +0.0% | 162.7 | 162.9 | +0.1% | -1.5% |
| RingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 538.8 | 538.8 | -0.0% | 538.8 | 538.8 | +0.0% | -0.0% |
| RingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 440.0 | 440.1 | +0.0% | 437.1 | 437.4 | +0.1% | -0.6% |
| UnidirRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 529.9 | 529.7 | -0.0% | 527.6 | 527.9 | +0.1% | -0.3% |
| UnidirRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 439.9 | 439.9 | -0.0% | 437.4 | 437.6 | +0.0% | -0.5% |

## ring_3072

| row | tier old | tier new | tier drift | staged old | staged new | new vs old staged | new vs tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 211.9 | 211.9 | -0.0% | 211.8 | 211.9 | +0.0% | -0.0% |
| HalfRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 172.4 | 172.4 | +0.0% | 172.2 | 172.3 | +0.1% | -0.1% |
| Ring1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 180.0 | 180.0 | -0.0% | 180.0 | 180.0 | -0.0% | -0.0% |
| Ring1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 141.0 | 141.0 | +0.0% | 141.0 | 141.0 | -0.0% | -0.0% |
| Ring1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.7 | 179.8 | +0.0% | 179.8 | 179.7 | -0.1% | -0.0% |
| Ring1HopEW_size_3072_ntype_unicast_write dev0 E | 141.2 | 141.2 | +0.0% | 141.2 | 141.2 | -0.0% | -0.1% |
| Ring1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 179.7 | 179.7 | +0.0% | 179.8 | 179.7 | -0.0% | -0.0% |
| Ring1HopNS_size_3072_ntype_unicast_write dev0 S | 141.2 | 141.2 | +0.0% | 141.3 | 141.2 | -0.1% | -0.0% |
| Ring3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.7 | 179.8 | +0.0% | 179.5 | 179.7 | +0.1% | -0.1% |
| Ring3HopEW_size_3072_ntype_unicast_write dev0 E | 164.4 | 164.4 | +0.0% | 163.8 | 164.0 | +0.2% | -0.2% |
| RingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 541.1 | 541.0 | -0.0% | 541.1 | 541.1 | -0.0% | +0.0% |
| RingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 423.7 | 423.6 | -0.0% | 423.7 | 423.7 | +0.0% | +0.0% |
| UnidirRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 477.4 | 477.4 | +0.0% | 476.5 | 477.0 | +0.1% | -0.1% |
| UnidirRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 399.3 | 399.3 | -0.0% | 398.0 | 398.7 | +0.2% | -0.1% |

