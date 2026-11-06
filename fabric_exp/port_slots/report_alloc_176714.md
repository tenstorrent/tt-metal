## 2d_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEWBidir_size_1024_ntype_unicast_write dev1 W | 116.5 | 116.5 | 111.8 | 111.4 | -4.2% | -4.1% | -4.4% |
| Sls1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 135.8 | 135.9 | 126.6 | 126.6 | -6.8% | -6.7% | -6.8% |
| Sls1HopEW_size_1024_ntype_unicast_write dev0 E | 116.1 | 116.2 | 111.5 | 111.8 | -3.9% | -4.0% | -3.8% |
| Sls1HopEW_size_4352_ntype_unicast_write dev0 E | 127.5 | 127.5 | 125.8 | 125.8 | -1.3% | -1.3% | -1.3% |
| Sls1HopNS_size_1024_ntype_unicast_write dev0 S | 120.0 | 120.0 | 113.4 | 113.5 | -5.4% | -5.4% | -5.4% |
| Sls1HopNS_size_4352_ntype_unicast_write dev0 S | 129.2 | 129.2 | 125.8 | 125.8 | -2.6% | -2.6% | -2.7% |
| Sls3HopEW_size_1024_ntype_unicast_write dev0 E | 138.5 | 138.6 | 146.1 | 146.3 | +5.5% | +5.4% | +5.6% |
| Sls3HopEW_size_4352_ntype_unicast_write dev0 E | 144.8 | 144.8 | 157.8 | 157.8 | +9.0% | +9.0% | +9.0% |
| SlsMulticast_size_1024_ntype_unicast_write dev7 W | 406.1 | 406.2 | 406.3 | 406.3 | +0.0% | +0.0% | +0.0% |
| SlsMulticast_size_4352_ntype_unicast_write dev7 W | 423.5 | 424.0 | 425.3 | 425.0 | +0.3% | +0.4% | +0.2% |
| SlsRowMerge_size_1024_ntype_unicast_write dev2 E | 119.5 | 119.4 | 119.5 | 119.4 | -0.0% | -0.0% | +0.0% |
| SlsRowMerge_size_4352_ntype_unicast_write dev2 E | 129.5 | 129.5 | 130.0 | 130.0 | +0.4% | +0.4% | +0.4% |
| SlsTurn4_size_1024_ntype_unicast_write dev0 S | 138.5 | 138.4 | 138.9 | 138.8 | +0.3% | +0.3% | +0.3% |
| SlsTurn4_size_4352_ntype_unicast_write dev0 S | 139.6 | 139.5 | 149.0 | 149.0 | +6.7% | +6.7% | +6.7% |

## 1d_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 114.8 | 114.9 | 95.3 | 95.3 | -17.0% | -16.9% | -17.0% |
| SlsLin1HopEW_size_4352_ntype_unicast_write dev0 E | 129.9 | 129.9 | 124.5 | 124.5 | -4.2% | -4.2% | -4.1% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 132.4 | 132.4 | 118.3 | 118.3 | -10.7% | -10.7% | -10.7% |
| SlsLin3HopEW_size_4352_ntype_unicast_write dev0 E | 142.0 | 142.0 | 124.5 | 124.5 | -12.3% | -12.4% | -12.3% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 109.7 | 109.7 | 108.7 | 108.7 | -0.9% | -0.9% | -0.9% |
| SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | 126.9 | 126.9 | 125.8 | 125.7 | -0.9% | -0.8% | -0.9% |

## ring_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 224.3 | 224.3 | 224.0 | 224.0 | -0.1% | -0.1% | -0.2% |
| HalfRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 172.1 | 172.1 | 171.7 | 171.7 | -0.3% | -0.3% | -0.2% |
| Ring1HopEWBidir_size_4352_ntype_unicast_scatter_write dev1 W | 179.4 | 179.3 | 179.3 | 179.3 | -0.0% | -0.0% | +0.0% |
| Ring1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 140.1 | 140.1 | 140.0 | 140.1 | -0.0% | -0.1% | -0.0% |
| Ring1HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.0 | 179.0 | 179.0 | 179.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopEW_size_4352_ntype_unicast_write dev0 E | 140.2 | 140.2 | 140.2 | 140.2 | +0.0% | +0.0% | +0.0% |
| Ring1HopNS_size_4352_ntype_unicast_scatter_write dev0 S | 179.0 | 179.0 | 179.0 | 179.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopNS_size_4352_ntype_unicast_write dev0 S | 140.1 | 140.1 | 140.1 | 140.1 | -0.0% | -0.0% | -0.0% |
| Ring3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 179.7 | 179.7 | 179.5 | 179.5 | -0.1% | -0.1% | -0.1% |
| Ring3HopEW_size_4352_ntype_unicast_write dev0 E | 165.3 | 165.3 | 162.7 | 162.7 | -1.6% | -1.6% | -1.6% |
| RingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 538.9 | 538.8 | 538.8 | 538.7 | -0.0% | -0.0% | -0.0% |
| RingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 440.0 | 440.1 | 437.1 | 437.1 | -0.7% | -0.7% | -0.7% |
| UnidirRingMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 529.8 | 529.9 | 527.6 | 527.6 | -0.4% | -0.4% | -0.4% |
| UnidirRingMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 439.9 | 439.9 | 437.4 | 437.4 | -0.6% | -0.6% | -0.6% |

## vc1_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VC1BurstEW_N32_G500_size_16 dev0 E | 174.4 | 174.4 | 175.0 | 174.9 | +0.3% | +0.4% | +0.3% |
| VC1BurstEW_N32_G500_size_4352 dev0 E | 193.8 | 193.8 | 194.2 | 194.1 | +0.2% | +0.2% | +0.2% |
| VC1BurstEW_N8_G500_size_16 dev0 E | 174.4 | 174.4 | 175.0 | 174.9 | +0.3% | +0.3% | +0.3% |
| VC1BurstEW_N8_G500_size_4352 dev0 E | 193.8 | 193.8 | 194.1 | 194.0 | +0.1% | +0.2% | +0.1% |
| VC1Inter_0_0_to_1_0_size_16_ntype_unicast_write dev0 N | 249.4 | 249.4 | 249.6 | 249.6 | +0.1% | +0.1% | +0.1% |
| VC1Inter_0_0_to_1_0_size_4352_ntype_unicast_write dev0 N | 250.8 | 250.8 | 251.0 | 251.0 | +0.1% | +0.1% | +0.1% |
| VC1Inter_0_0_to_1_1_size_16_ntype_unicast_write dev0 N | 768.7 | 767.9 | 309.6 | 309.6 | -59.7% | -59.7% | -59.7% |
| VC1Inter_0_0_to_1_1_size_4352_ntype_unicast_write dev0 N | 905.7 | 905.1 | 367.5 | 367.5 | -59.4% | -59.4% | -59.4% |
| VC1Inter_0_0_to_1_2_size_16_ntype_unicast_write dev0 N | 787.5 | 787.6 | 312.2 | 312.1 | -60.4% | -60.4% | -60.4% |
| VC1Inter_0_0_to_1_2_size_4352_ntype_unicast_write dev0 N | 903.9 | 904.1 | 377.0 | 376.9 | -58.3% | -58.3% | -58.3% |
| VC1Inter_0_0_to_1_3_size_16_ntype_unicast_write dev0 N | 803.9 | 804.5 | 345.0 | 345.4 | -57.1% | -57.1% | -57.1% |
| VC1Inter_0_0_to_1_3_size_4352_ntype_unicast_write dev0 N | 920.1 | 920.3 | 372.7 | 373.0 | -59.5% | -59.5% | -59.5% |
| VC1Mesh1HopEW_size_16_ntype_unicast_write dev0 E | 174.6 | 174.4 | 174.8 | 174.7 | +0.1% | +0.1% | +0.1% |
| VC1Mesh1HopEW_size_4352_ntype_unicast_write dev0 E | 194.0 | 193.9 | 194.1 | 193.8 | +0.0% | +0.1% | -0.0% |
| VC1Mesh1HopNS_size_16_ntype_unicast_write dev0 S | 181.7 | 181.7 | 180.1 | 180.2 | -0.9% | -0.9% | -0.8% |
| VC1Mesh1HopNS_size_4352_ntype_unicast_write dev0 S | 199.9 | 200.0 | 198.6 | 198.9 | -0.6% | -0.7% | -0.6% |
| VC1Mesh2HopMerge_size_16_ntype_unicast_write dev2 E | 212.1 | 212.1 | 172.3 | 172.4 | -18.8% | -18.8% | -18.7% |
| VC1Mesh2HopMerge_size_4352_ntype_unicast_write dev2 E | 241.4 | 241.5 | 191.0 | 191.1 | -20.9% | -20.9% | -20.9% |
| VC1Mesh2Hop_size_16_ntype_unicast_write dev0 S | 388.2 | 388.3 | 208.9 | 209.1 | -46.2% | -46.2% | -46.2% |
| VC1Mesh2Hop_size_4352_ntype_unicast_write dev0 S | 444.7 | 444.7 | 218.6 | 218.8 | -50.8% | -50.8% | -50.8% |
| VC1MeshMulticast_size_4352_ntype_unicast_write dev3 W | 264.8 | 264.8 | 201.0 | 200.9 | -24.1% | -24.1% | -24.1% |
| VC1MixForward_size_16_ntype_unicast_write dev0 N (flow) | 820.3 | 820.1 | 403.1 | 403.2 | -50.8% | -50.9% | -50.8% |
| VC1MixForward_size_16_ntype_unicast_write dev2 N (flow) | 497.4 | 497.5 | 266.8 | 266.7 | -46.4% | -46.4% | -46.4% |
| VC1MixForward_size_16_ntype_unicast_write LINK TOTAL | 309.6 | 309.6 | 160.5 | 160.5 | -48.2% | -48.1% | -48.2% |
| VC1MixForward_size_4352_ntype_unicast_write dev0 N (flow) | 956.0 | 956.9 | 425.7 | 425.4 | -55.5% | -55.5% | -55.5% |
| VC1MixForward_size_4352_ntype_unicast_write dev2 N (flow) | 549.2 | 549.1 | 263.2 | 263.2 | -52.1% | -52.1% | -52.1% |
| VC1MixForward_size_4352_ntype_unicast_write LINK TOTAL | 348.8 | 348.9 | 162.6 | 162.6 | -53.4% | -53.4% | -53.4% |
| VC1MixWorker_size_16_ntype_unicast_write dev0 E (flow) | 210.0 | 209.9 | 218.0 | 217.9 | +3.8% | +3.8% | +3.8% |
| VC1MixWorker_size_16_ntype_unicast_write dev0 N (flow) | 825.2 | 824.4 | 402.6 | 402.6 | -51.2% | -51.2% | -51.2% |
| VC1MixWorker_size_16_ntype_unicast_write LINK TOTAL | 167.4 | 167.3 | 141.4 | 141.4 | -15.5% | -15.5% | -15.5% |
| VC1MixWorker_size_4352_ntype_unicast_write dev0 E (flow) | 221.9 | 221.9 | 231.2 | 231.2 | +4.2% | +4.2% | +4.2% |
| VC1MixWorker_size_4352_ntype_unicast_write dev0 N (flow) | 955.2 | 955.4 | 434.1 | 433.9 | -54.6% | -54.6% | -54.6% |
| VC1MixWorker_size_4352_ntype_unicast_write LINK TOTAL | 180.1 | 180.1 | 150.9 | 150.9 | -16.2% | -16.2% | -16.2% |
| VC1SoloForward_size_16_ntype_unicast_write dev2 N | 458.0 | 458.2 | 210.9 | 210.9 | -54.0% | -54.0% | -54.0% |
| VC1SoloForward_size_4352_ntype_unicast_write dev2 N | 512.6 | 512.5 | 222.4 | 222.3 | -56.6% | -56.6% | -56.6% |
| VC1SoloWorker_size_16_ntype_unicast_write dev0 E | 174.3 | 174.3 | 174.5 | 174.6 | +0.2% | +0.1% | +0.2% |
| VC1SoloWorker_size_4352_ntype_unicast_write dev0 E | 193.6 | 193.7 | 193.6 | 193.6 | +0.0% | +0.0% | -0.0% |

## vc2_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh1HopEWBidir_size_4352_ntype_unicast_scatter_write dev1 W | 186.1 | 186.1 | 154.1 | 154.1 | -17.2% | -17.2% | -17.2% |
| Mesh1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 155.2 | 155.2 | 132.4 | 132.3 | -14.7% | -14.7% | -14.8% |
| Mesh1HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 185.4 | 185.4 | 153.8 | 153.8 | -17.0% | -17.0% | -17.0% |
| Mesh1HopEW_size_4352_ntype_unicast_write dev0 E | 149.5 | 149.5 | 126.9 | 126.9 | -15.1% | -15.1% | -15.1% |
| Mesh1HopNS_size_4352_ntype_unicast_scatter_write dev0 S | 199.9 | 199.9 | 168.6 | 168.6 | -15.7% | -15.7% | -15.7% |
| Mesh1HopNS_size_4352_ntype_unicast_write dev0 S | 162.0 | 162.0 | 131.9 | 132.0 | -18.6% | -18.6% | -18.6% |
| Mesh3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 298.4 | 298.6 | 185.7 | 185.8 | -37.8% | -37.8% | -37.8% |
| Mesh3HopEW_size_4352_ntype_unicast_write dev0 E | 298.3 | 298.2 | 186.5 | 186.4 | -37.5% | -37.5% | -37.5% |
| MeshMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 718.1 | 718.1 | 555.7 | 555.9 | -22.6% | -22.6% | -22.6% |
| MeshMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 652.9 | 653.2 | 448.2 | 448.3 | -31.4% | -31.3% | -31.4% |
| PcsVC02bEW4352_size_4352_ntype_unicast_write dev0 E (flow) | 148.3 | 148.3 | 127.9 | 127.9 | -13.7% | -13.7% | -13.7% |
| PcsVC2EW4352_size_4352_ntype_unicast_write dev0 E | 870.6 | 870.7 | 442.3 | 442.3 | -49.2% | -49.2% | -49.2% |

## ne_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NeighborExchange_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 128.9 | 128.9 | 128.7 | 128.7 | -0.1% | -0.1% | -0.1% |
| NeighborExchange_size_4352_ntype_unicast_write_num_links_1 dev7 W | 124.8 | 124.8 | 124.8 | 124.8 | -0.0% | -0.0% | -0.0% |

## 2d_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEWBidir_size_1024_ntype_unicast_write dev1 W | 110.1 | 110.5 | 109.8 | 109.6 | -0.6% | -0.3% | -0.9% |
| Sls1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 110.2 | 110.2 | 109.7 | 109.8 | -0.4% | -0.4% | -0.4% |
| Sls1HopEW_size_1024_ntype_unicast_write dev0 E | 109.8 | 109.9 | 109.5 | 109.5 | -0.3% | -0.2% | -0.4% |
| Sls1HopEW_size_3072_ntype_unicast_write dev0 E | 109.8 | 109.8 | 109.6 | 109.6 | -0.2% | -0.2% | -0.2% |
| Sls1HopNS_size_1024_ntype_unicast_write dev0 S | 109.8 | 109.8 | 112.5 | 112.5 | +2.5% | +2.5% | +2.5% |
| Sls1HopNS_size_3072_ntype_unicast_write dev0 S | 109.8 | 109.8 | 113.6 | 113.6 | +3.5% | +3.5% | +3.5% |
| Sls3HopEW_size_1024_ntype_unicast_write dev0 E | 137.0 | 137.0 | 136.9 | 136.8 | -0.1% | -0.1% | -0.1% |
| Sls3HopEW_size_3072_ntype_unicast_write dev0 E | 139.8 | 139.8 | 136.1 | 136.6 | -2.5% | -2.6% | -2.3% |
| SlsMulticast_size_1024_ntype_unicast_write dev7 W | 401.9 | 401.8 | 400.5 | 400.5 | -0.3% | -0.4% | -0.3% |
| SlsMulticast_size_3072_ntype_unicast_write dev7 W | 400.8 | 400.9 | 400.0 | 399.9 | -0.2% | -0.2% | -0.2% |
| SlsRowMerge_size_1024_ntype_unicast_write dev2 E | 117.6 | 117.6 | 118.3 | 118.4 | +0.7% | +0.6% | +0.7% |
| SlsRowMerge_size_3072_ntype_unicast_write dev2 E | 117.4 | 117.4 | 118.3 | 118.6 | +0.9% | +0.7% | +1.0% |
| SlsTurn4_size_1024_ntype_unicast_write dev0 S | 136.9 | 136.8 | 136.7 | 136.8 | -0.1% | -0.2% | +0.0% |
| SlsTurn4_size_3072_ntype_unicast_write dev0 S | 136.1 | 136.1 | 136.1 | 136.5 | +0.2% | +0.0% | +0.3% |

## 1d_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 93.1 | 93.1 | 94.4 | 94.4 | +1.4% | +1.4% | +1.5% |
| SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | 93.5 | 93.5 | 95.2 | 95.2 | +1.7% | +1.7% | +1.7% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 115.0 | 115.0 | 114.3 | 114.3 | -0.6% | -0.6% | -0.6% |
| SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | 115.5 | 115.5 | 114.6 | 114.7 | -0.7% | -0.7% | -0.7% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 107.3 | 107.3 | 107.0 | 107.0 | -0.3% | -0.3% | -0.3% |
| SlsLinRowMerge_size_3072_ntype_unicast_write dev2 E | 107.7 | 107.7 | 107.5 | 107.5 | -0.2% | -0.2% | -0.2% |

## ring_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 211.9 | 211.9 | 211.8 | 211.8 | -0.1% | -0.1% | -0.0% |
| HalfRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 172.4 | 172.4 | 172.2 | 172.2 | -0.1% | -0.1% | -0.1% |
| Ring1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 180.0 | 179.9 | 180.0 | 180.1 | +0.0% | -0.0% | +0.1% |
| Ring1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 141.0 | 141.0 | 141.0 | 141.0 | +0.0% | +0.0% | +0.0% |
| Ring1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.7 | 179.8 | 179.8 | 179.8 | +0.0% | +0.0% | +0.0% |
| Ring1HopEW_size_3072_ntype_unicast_write dev0 E | 141.2 | 141.3 | 141.3 | 141.2 | -0.0% | +0.0% | -0.1% |
| Ring1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 179.7 | 179.7 | 179.8 | 179.8 | +0.0% | +0.0% | +0.0% |
| Ring1HopNS_size_3072_ntype_unicast_write dev0 S | 141.2 | 141.2 | 141.3 | 141.3 | +0.1% | +0.1% | +0.1% |
| Ring3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.7 | 179.7 | 179.5 | 179.5 | -0.1% | -0.1% | -0.1% |
| Ring3HopEW_size_3072_ntype_unicast_write dev0 E | 164.4 | 164.4 | 163.8 | 163.7 | -0.4% | -0.4% | -0.4% |
| RingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 541.1 | 541.1 | 541.0 | 541.1 | -0.0% | -0.0% | +0.0% |
| RingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 423.6 | 423.7 | 423.7 | 423.6 | -0.0% | +0.0% | -0.0% |
| UnidirRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 477.3 | 477.4 | 476.5 | 476.5 | -0.2% | -0.2% | -0.2% |
| UnidirRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 399.3 | 399.3 | 398.1 | 397.9 | -0.3% | -0.3% | -0.3% |

## vc1_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VC1BurstEW_N32_G500_size_16 dev0 E | 173.3 | 173.3 | 138.5 | 138.4 | -20.1% | -20.1% | -20.1% |
| VC1BurstEW_N32_G500_size_3072 dev0 E | 184.3 | 184.3 | 138.5 | 138.6 | -24.8% | -24.8% | -24.8% |
| VC1BurstEW_N8_G500_size_16 dev0 E | 173.5 | 173.5 | 138.5 | 138.5 | -20.2% | -20.2% | -20.2% |
| VC1BurstEW_N8_G500_size_3072 dev0 E | 184.2 | 184.1 | 138.6 | 138.5 | -24.8% | -24.8% | -24.8% |
| VC1Inter_0_0_to_1_0_size_16_ntype_unicast_write dev0 N | 248.4 | 248.4 | 224.5 | 224.6 | -9.6% | -9.6% | -9.6% |
| VC1Inter_0_0_to_1_0_size_3072_ntype_unicast_write dev0 N | 248.9 | 248.9 | 225.1 | 225.1 | -9.6% | -9.6% | -9.6% |
| VC1Inter_0_0_to_1_1_size_16_ntype_unicast_write dev0 N | 769.8 | 769.5 | 264.9 | 264.9 | -65.6% | -65.6% | -65.6% |
| VC1Inter_0_0_to_1_1_size_3072_ntype_unicast_write dev0 N | 827.3 | 827.7 | 266.9 | 266.9 | -67.7% | -67.7% | -67.7% |
| VC1Inter_0_0_to_1_2_size_16_ntype_unicast_write dev0 N | 784.2 | 785.0 | 254.0 | 254.0 | -67.6% | -67.6% | -67.6% |
| VC1Inter_0_0_to_1_2_size_3072_ntype_unicast_write dev0 N | 851.1 | 850.8 | 257.7 | 257.8 | -69.7% | -69.7% | -69.7% |
| VC1Inter_0_0_to_1_3_size_16_ntype_unicast_write dev0 N | 802.6 | 802.5 | 255.7 | 255.7 | -68.1% | -68.1% | -68.1% |
| VC1Inter_0_0_to_1_3_size_3072_ntype_unicast_write dev0 N | 872.9 | 873.1 | 259.7 | 259.7 | -70.2% | -70.2% | -70.3% |
| VC1Mesh1HopEW_size_16_ntype_unicast_write dev0 E | 173.6 | 173.7 | 138.3 | 138.3 | -20.4% | -20.4% | -20.4% |
| VC1Mesh1HopEW_size_3072_ntype_unicast_write dev0 E | 184.6 | 184.6 | 138.4 | 138.3 | -25.0% | -25.0% | -25.0% |
| VC1Mesh1HopNS_size_16_ntype_unicast_write dev0 S | 179.4 | 179.1 | 138.3 | 138.3 | -22.9% | -22.9% | -22.8% |
| VC1Mesh1HopNS_size_3072_ntype_unicast_write dev0 S | 190.2 | 190.0 | 138.4 | 138.4 | -27.2% | -27.3% | -27.2% |
| VC1Mesh2HopMerge_size_16_ntype_unicast_write dev2 E | 211.4 | 211.3 | 130.2 | 130.6 | -38.3% | -38.4% | -38.2% |
| VC1Mesh2HopMerge_size_3072_ntype_unicast_write dev2 E | 231.1 | 231.3 | 130.6 | 130.9 | -43.5% | -43.5% | -43.4% |
| VC1Mesh2Hop_size_16_ntype_unicast_write dev0 S | 387.3 | 387.0 | 175.5 | 175.5 | -54.7% | -54.7% | -54.7% |
| VC1Mesh2Hop_size_3072_ntype_unicast_write dev0 S | 429.5 | 429.4 | 177.1 | 177.1 | -58.8% | -58.8% | -58.8% |
| VC1MeshMulticast_size_3072_ntype_unicast_write dev3 W | 251.2 | 251.3 | 131.2 | 131.4 | -47.7% | -47.8% | -47.7% |
| VC1MixForward_size_16_ntype_unicast_write dev0 N (flow) | 819.2 | 818.5 | 272.4 | 272.3 | -66.7% | -66.7% | -66.7% |
| VC1MixForward_size_16_ntype_unicast_write dev2 N (flow) | 496.1 | 496.1 | 236.7 | 236.7 | -52.3% | -52.3% | -52.3% |
| VC1MixForward_size_16_ntype_unicast_write LINK TOTAL | 309.0 | 308.9 | 126.7 | 126.6 | -59.0% | -59.0% | -59.0% |
| VC1MixForward_size_3072_ntype_unicast_write dev0 N (flow) | 891.4 | 891.5 | 279.6 | 279.3 | -68.7% | -68.6% | -68.7% |
| VC1MixForward_size_3072_ntype_unicast_write dev2 N (flow) | 533.7 | 533.5 | 237.3 | 237.3 | -55.5% | -55.5% | -55.5% |
| VC1MixForward_size_3072_ntype_unicast_write LINK TOTAL | 333.8 | 333.8 | 128.3 | 128.3 | -61.6% | -61.6% | -61.6% |
| VC1MixWorker_size_16_ntype_unicast_write dev0 E (flow) | 209.2 | 209.2 | 182.2 | 182.1 | -12.9% | -12.9% | -12.9% |
| VC1MixWorker_size_16_ntype_unicast_write dev0 N (flow) | 824.6 | 824.0 | 272.6 | 272.6 | -66.9% | -66.9% | -66.9% |
| VC1MixWorker_size_16_ntype_unicast_write LINK TOTAL | 166.9 | 166.8 | 109.2 | 109.2 | -34.6% | -34.6% | -34.6% |
| VC1MixWorker_size_3072_ntype_unicast_write dev0 E (flow) | 217.4 | 217.4 | 182.6 | 182.6 | -16.0% | -16.0% | -16.0% |
| VC1MixWorker_size_3072_ntype_unicast_write dev0 N (flow) | 892.9 | 893.7 | 278.3 | 278.0 | -68.9% | -68.8% | -68.9% |
| VC1MixWorker_size_3072_ntype_unicast_write LINK TOTAL | 174.9 | 174.9 | 110.3 | 110.2 | -37.0% | -36.9% | -37.0% |
| VC1SoloForward_size_16_ntype_unicast_write dev2 N | 458.0 | 457.6 | 178.4 | 178.4 | -61.0% | -61.0% | -61.0% |
| VC1SoloForward_size_3072_ntype_unicast_write dev2 N | 497.5 | 497.6 | 178.9 | 178.9 | -64.0% | -64.0% | -64.0% |
| VC1SoloWorker_size_16_ntype_unicast_write dev0 E | 173.6 | 173.6 | 138.2 | 138.3 | -20.4% | -20.4% | -20.4% |
| VC1SoloWorker_size_3072_ntype_unicast_write dev0 E | 184.2 | 184.2 | 138.3 | 138.3 | -24.9% | -24.9% | -24.9% |

## vc2_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 184.5 | 184.5 | 152.4 | 152.4 | -17.4% | -17.4% | -17.4% |
| Mesh1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 149.3 | 149.2 | 122.5 | 122.5 | -17.9% | -17.9% | -17.9% |
| Mesh1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 184.2 | 184.2 | 152.4 | 152.4 | -17.2% | -17.2% | -17.2% |
| Mesh1HopEW_size_3072_ntype_unicast_write dev0 E | 147.4 | 147.5 | 122.1 | 122.1 | -17.2% | -17.2% | -17.2% |
| Mesh1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 198.6 | 198.6 | 167.5 | 167.5 | -15.7% | -15.7% | -15.7% |
| Mesh1HopNS_size_3072_ntype_unicast_write dev0 S | 160.8 | 160.8 | 129.2 | 129.2 | -19.7% | -19.7% | -19.7% |
| Mesh3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 290.1 | 289.9 | 152.6 | 152.7 | -47.4% | -47.4% | -47.3% |
| Mesh3HopEW_size_3072_ntype_unicast_write dev0 E | 289.5 | 289.5 | 148.3 | 148.3 | -48.8% | -48.8% | -48.8% |
| MeshMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 694.5 | 694.7 | 507.0 | 506.8 | -27.0% | -27.0% | -27.0% |
| MeshMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 628.8 | 628.2 | 426.2 | 426.2 | -32.2% | -32.2% | -32.2% |
| PcsVC02bEW4352_size_3072_ntype_unicast_write dev0 E (flow) | 146.4 | 146.4 | 123.8 | 123.8 | -15.5% | -15.5% | -15.5% |
| PcsVC2EW4352_size_3072_ntype_unicast_write dev0 E | 822.8 | 818.1 | 422.1 | 422.1 | -48.6% | -48.7% | -48.4% |

## ne_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NeighborExchange_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 127.7 | 127.6 | 127.6 | 127.6 | -0.1% | -0.1% | -0.1% |
| NeighborExchange_size_3072_ntype_unicast_write_num_links_1 dev7 W | 88.8 | 88.7 | 88.7 | 88.7 | -0.0% | -0.0% | -0.0% |

## Per suite (STAGED against TIER, mean of the passes; mixed tests by link total)

| suite | rows | wins (< -1%) | losses (> +1%) | median | best | worst | acceptance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2d_4352 | 14 | 6 | 3 | +0.0% | -6.8% Sls1HopEWBidir_size_4352_ntype_unicast_write dev1 W | +9.0% Sls3HopEW_size_4352_ntype_unicast_write dev0 E | NOT met |
| 1d_4352 | 6 | 4 | 0 | -7.4% | -17.0% SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | -0.9% SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | met |
| ring_4352 | 14 | 1 | 0 | -0.1% | -1.6% Ring3HopEW_size_4352_ntype_unicast_write dev0 E | +0.0% Ring1HopEW_size_4352_ntype_unicast_write dev0 E | met |
| vc1_4352 | 29 | 17 | 0 | -18.8% | -60.4% VC1Inter_0_0_to_1_2_size_16_ntype_unicast_write dev0 N | +0.3% VC1BurstEW_N32_G500_size_16 dev0 E | met |
| vc2_4352 | 11 | 11 | 0 | -18.6% | -49.2% PcsVC2EW4352_size_4352_ntype_unicast_write dev0 E | -14.7% Mesh1HopEWBidir_size_4352_ntype_unicast_write dev1 W | met |
| ne_4352 | 2 | 0 | 0 | -0.1% | -0.1% NeighborExchange_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | -0.0% NeighborExchange_size_4352_ntype_unicast_write_num_links_1 dev7 W | met |
| 2d_3072 | 14 | 1 | 2 | -0.1% | -2.5% Sls3HopEW_size_3072_ntype_unicast_write dev0 E | +3.5% Sls1HopNS_size_3072_ntype_unicast_write dev0 S | NOT met |
| 1d_3072 | 6 | 0 | 2 | -0.3% | -0.7% SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | +1.7% SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | NOT met |
| ring_3072 | 14 | 0 | 0 | -0.0% | -0.4% Ring3HopEW_size_3072_ntype_unicast_write dev0 E | +0.1% Ring1HopNS_size_3072_ntype_unicast_write dev0 S | met |
| vc1_3072 | 29 | 29 | 0 | -38.3% | -70.2% VC1Inter_0_0_to_1_3_size_3072_ntype_unicast_write dev0 N | -9.6% VC1Inter_0_0_to_1_0_size_3072_ntype_unicast_write dev0 N | met |
| vc2_3072 | 11 | 11 | 0 | -19.7% | -48.8% Mesh3HopEW_size_3072_ntype_unicast_write dev0 E | -15.7% Mesh1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | met |
| ne_3072 | 2 | 0 | 0 | -0.1% | -0.1% NeighborExchange_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | -0.0% NeighborExchange_size_3072_ntype_unicast_write_num_links_1 dev7 W | met |
