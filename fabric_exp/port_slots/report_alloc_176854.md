## 2d_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEWBidir_size_1024_ntype_unicast_write dev1 W | 116.5 | 116.5 | 116.5 | 116.5 | -0.0% | +0.0% | -0.1% |
| Sls1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 135.8 | 135.7 | 135.6 | 135.7 | -0.0% | -0.1% | +0.1% |
| Sls1HopEW_size_1024_ntype_unicast_write dev0 E | 116.2 | 116.2 | 116.2 | 116.3 | +0.0% | -0.1% | +0.1% |
| Sls1HopEW_size_4352_ntype_unicast_write dev0 E | 127.5 | 127.5 | 127.5 | 127.4 | -0.0% | +0.0% | -0.0% |
| Sls1HopNS_size_1024_ntype_unicast_write dev0 S | 119.9 | 120.0 | 119.9 | 119.9 | -0.0% | -0.0% | -0.0% |
| Sls1HopNS_size_4352_ntype_unicast_write dev0 S | 129.2 | 129.3 | 129.3 | 129.2 | -0.0% | +0.1% | -0.1% |
| Sls3HopEW_size_1024_ntype_unicast_write dev0 E | 138.5 | 138.5 | 138.4 | 138.5 | -0.0% | -0.1% | -0.0% |
| Sls3HopEW_size_4352_ntype_unicast_write dev0 E | 145.0 | 144.9 | 145.0 | 144.8 | -0.0% | +0.0% | -0.1% |
| SlsMulticast_size_1024_ntype_unicast_write dev7 W | 406.1 | 406.1 | 406.2 | 406.2 | +0.0% | +0.0% | +0.0% |
| SlsMulticast_size_4352_ntype_unicast_write dev7 W | 424.3 | 423.6 | 423.6 | 424.5 | +0.0% | -0.2% | +0.2% |
| SlsRowMerge_size_1024_ntype_unicast_write dev2 E | 119.5 | 119.4 | 119.6 | 119.5 | +0.1% | +0.1% | +0.0% |
| SlsRowMerge_size_4352_ntype_unicast_write dev2 E | 129.5 | 129.4 | 129.5 | 129.5 | +0.0% | -0.0% | +0.0% |
| SlsTurn4_size_1024_ntype_unicast_write dev0 S | 138.4 | 138.5 | 138.5 | 138.5 | +0.0% | +0.0% | +0.0% |
| SlsTurn4_size_4352_ntype_unicast_write dev0 S | 139.7 | 139.6 | 139.7 | 139.7 | +0.0% | -0.0% | +0.0% |

## 1d_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 114.7 | 114.8 | 95.3 | 95.3 | -16.9% | -16.9% | -17.0% |
| SlsLin1HopEW_size_4352_ntype_unicast_write dev0 E | 129.8 | 129.8 | 124.5 | 124.5 | -4.1% | -4.1% | -4.1% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 132.5 | 132.5 | 118.2 | 118.3 | -10.8% | -10.8% | -10.7% |
| SlsLin3HopEW_size_4352_ntype_unicast_write dev0 E | 142.0 | 142.1 | 124.5 | 124.5 | -12.4% | -12.4% | -12.4% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 109.7 | 109.7 | 108.7 | 108.6 | -1.0% | -0.9% | -1.0% |
| SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | 126.9 | 126.9 | 125.8 | 125.8 | -0.9% | -0.9% | -0.9% |

## vc2_4352 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh1HopEWBidir_size_4352_ntype_unicast_scatter_write dev1 W | 186.1 | 186.1 | 157.1 | 157.1 | -15.6% | -15.6% | -15.6% |
| Mesh1HopEWBidir_size_4352_ntype_unicast_write dev1 W | 155.1 | 155.2 | 144.4 | 144.3 | -7.0% | -6.9% | -7.0% |
| Mesh1HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 185.4 | 185.4 | 156.3 | 156.4 | -15.7% | -15.7% | -15.7% |
| Mesh1HopEW_size_4352_ntype_unicast_write dev0 E | 149.5 | 149.5 | 137.9 | 137.9 | -7.8% | -7.7% | -7.8% |
| Mesh1HopNS_size_4352_ntype_unicast_scatter_write dev0 S | 199.9 | 199.9 | 169.6 | 169.6 | -15.2% | -15.2% | -15.2% |
| Mesh1HopNS_size_4352_ntype_unicast_write dev0 S | 162.0 | 162.0 | 141.4 | 141.5 | -12.7% | -12.7% | -12.7% |
| Mesh3HopEW_size_4352_ntype_unicast_scatter_write dev0 E | 298.9 | 298.7 | 172.6 | 172.6 | -42.2% | -42.2% | -42.2% |
| Mesh3HopEW_size_4352_ntype_unicast_write dev0 E | 298.6 | 298.3 | 172.3 | 172.3 | -42.3% | -42.3% | -42.2% |
| MeshMulticast_size_4352_ntype_unicast_scatter_write_num_links_1 dev7 W | 718.6 | 718.1 | 555.3 | 555.3 | -22.7% | -22.7% | -22.7% |
| MeshMulticast_size_4352_ntype_unicast_write_num_links_1 dev7 W | 653.1 | 653.3 | 441.1 | 441.3 | -32.5% | -32.5% | -32.5% |
| PcsVC02bEW4352_size_4352_ntype_unicast_write dev0 E (flow) | 148.3 | 148.3 | 137.2 | 137.2 | -7.4% | -7.4% | -7.4% |
| PcsVC2EW4352_size_4352_ntype_unicast_write dev0 E | 869.5 | 870.4 | 585.6 | 585.7 | -32.7% | -32.6% | -32.7% |

## 2d_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEWBidir_size_1024_ntype_unicast_write dev1 W | 110.6 | 110.2 | 110.2 | 110.5 | -0.0% | -0.3% | +0.2% |
| Sls1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 110.1 | 110.2 | 110.3 | 110.3 | +0.2% | +0.2% | +0.1% |
| Sls1HopEW_size_1024_ntype_unicast_write dev0 E | 110.3 | 109.8 | 110.1 | 110.2 | +0.1% | -0.1% | +0.3% |
| Sls1HopEW_size_3072_ntype_unicast_write dev0 E | 109.8 | 109.8 | 110.1 | 110.1 | +0.3% | +0.3% | +0.3% |
| Sls1HopNS_size_1024_ntype_unicast_write dev0 S | 109.7 | 109.8 | 112.1 | 112.1 | +2.1% | +2.2% | +2.1% |
| Sls1HopNS_size_3072_ntype_unicast_write dev0 S | 109.8 | 109.7 | 113.1 | 113.1 | +3.0% | +3.0% | +3.0% |
| Sls3HopEW_size_1024_ntype_unicast_write dev0 E | 137.1 | 137.0 | 136.3 | 136.4 | -0.5% | -0.6% | -0.5% |
| Sls3HopEW_size_3072_ntype_unicast_write dev0 E | 139.8 | 139.8 | 135.9 | 134.9 | -3.1% | -2.8% | -3.5% |
| SlsMulticast_size_1024_ntype_unicast_write dev7 W | 401.8 | 401.8 | 400.5 | 400.6 | -0.3% | -0.3% | -0.3% |
| SlsMulticast_size_3072_ntype_unicast_write dev7 W | 400.9 | 400.9 | 399.5 | 399.4 | -0.4% | -0.4% | -0.4% |
| SlsRowMerge_size_1024_ntype_unicast_write dev2 E | 117.5 | 117.4 | 117.5 | 117.5 | +0.0% | -0.0% | +0.0% |
| SlsRowMerge_size_3072_ntype_unicast_write dev2 E | 117.6 | 117.6 | 117.4 | 117.0 | -0.3% | -0.1% | -0.6% |
| SlsTurn4_size_1024_ntype_unicast_write dev0 S | 136.6 | 136.7 | 136.5 | 136.5 | -0.1% | -0.1% | -0.1% |
| SlsTurn4_size_3072_ntype_unicast_write dev0 S | 136.1 | 136.1 | 135.9 | 135.3 | -0.4% | -0.2% | -0.6% |

## 1d_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 93.1 | 93.1 | 94.3 | 94.3 | +1.3% | +1.4% | +1.3% |
| SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | 93.5 | 93.5 | 95.1 | 95.1 | +1.6% | +1.6% | +1.6% |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 115.0 | 115.0 | 115.9 | 115.9 | +0.8% | +0.8% | +0.8% |
| SlsLin3HopEW_size_3072_ntype_unicast_write dev0 E | 115.5 | 115.5 | 116.4 | 116.4 | +0.8% | +0.8% | +0.8% |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 107.3 | 107.2 | 105.2 | 105.2 | -1.9% | -2.0% | -1.9% |
| SlsLinRowMerge_size_3072_ntype_unicast_write dev2 E | 107.6 | 107.6 | 105.7 | 105.7 | -1.8% | -1.8% | -1.8% |

## ring_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 211.9 | 211.9 | 211.9 | 211.8 | -0.0% | -0.0% | -0.0% |
| HalfRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 172.4 | 172.4 | 172.4 | 172.4 | -0.0% | -0.0% | +0.0% |
| Ring1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 180.0 | 180.0 | 180.0 | 180.0 | -0.0% | -0.0% | -0.0% |
| Ring1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 141.0 | 141.0 | 140.9 | 140.9 | -0.0% | -0.0% | -0.1% |
| Ring1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.8 | 179.7 | 179.6 | 179.7 | -0.1% | -0.1% | -0.0% |
| Ring1HopEW_size_3072_ntype_unicast_write dev0 E | 141.2 | 141.2 | 141.2 | 141.2 | -0.0% | -0.1% | +0.0% |
| Ring1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 179.8 | 179.8 | 179.6 | 179.7 | -0.1% | -0.1% | -0.1% |
| Ring1HopNS_size_3072_ntype_unicast_write dev0 S | 141.3 | 141.2 | 141.1 | 141.1 | -0.1% | -0.1% | -0.0% |
| Ring3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 179.8 | 179.8 | 179.4 | 179.4 | -0.2% | -0.2% | -0.2% |
| Ring3HopEW_size_3072_ntype_unicast_write dev0 E | 164.5 | 164.4 | 164.5 | 164.5 | +0.0% | -0.0% | +0.0% |
| RingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 540.9 | 541.1 | 540.9 | 540.9 | -0.0% | -0.0% | -0.0% |
| RingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 423.6 | 423.7 | 423.2 | 423.3 | -0.1% | -0.1% | -0.1% |
| UnidirRingMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 477.4 | 477.3 | 477.5 | 477.4 | +0.0% | +0.0% | +0.0% |
| UnidirRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 399.4 | 399.3 | 399.5 | 399.6 | +0.0% | +0.0% | +0.1% |

## vc2_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mesh1HopEWBidir_size_3072_ntype_unicast_scatter_write dev1 W | 184.5 | 184.5 | 152.8 | 152.7 | -17.2% | -17.2% | -17.3% |
| Mesh1HopEWBidir_size_3072_ntype_unicast_write dev1 W | 149.3 | 149.2 | 124.8 | 124.9 | -16.4% | -16.4% | -16.3% |
| Mesh1HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 184.2 | 184.2 | 152.9 | 152.8 | -17.0% | -17.0% | -17.0% |
| Mesh1HopEW_size_3072_ntype_unicast_write dev0 E | 147.4 | 147.4 | 122.8 | 122.8 | -16.7% | -16.7% | -16.7% |
| Mesh1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | 198.6 | 198.6 | 167.6 | 167.5 | -15.6% | -15.6% | -15.6% |
| Mesh1HopNS_size_3072_ntype_unicast_write dev0 S | 160.8 | 160.8 | 130.5 | 130.5 | -18.9% | -18.9% | -18.9% |
| Mesh3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | 290.2 | 290.1 | 152.1 | 152.0 | -47.6% | -47.6% | -47.6% |
| Mesh3HopEW_size_3072_ntype_unicast_write dev0 E | 289.8 | 289.6 | 147.9 | 148.1 | -48.9% | -49.0% | -48.9% |
| MeshMulticast_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 695.0 | 695.1 | 504.7 | 504.3 | -27.4% | -27.4% | -27.4% |
| MeshMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | 628.5 | 627.8 | 423.4 | 423.3 | -32.6% | -32.6% | -32.6% |
| PcsVC02bEW4352_size_3072_ntype_unicast_write dev0 E (flow) | 146.4 | 146.4 | 124.2 | 124.2 | -15.2% | -15.2% | -15.2% |
| PcsVC2EW4352_size_3072_ntype_unicast_write dev0 E | 818.3 | 821.4 | 421.9 | 421.8 | -48.5% | -48.4% | -48.7% |

## ne_3072 (runs: a/staged, a/tier, b/staged, b/tier)

| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NeighborExchange_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | 127.7 | 127.7 | 127.1 | 127.1 | -0.4% | -0.4% | -0.4% |
| NeighborExchange_size_3072_ntype_unicast_write_num_links_1 dev7 W | 88.7 | 88.8 | 88.7 | 88.7 | -0.0% | -0.0% | -0.0% |

## Per suite (STAGED against TIER, mean of the passes; mixed tests by link total)

| suite | rows | wins (< -1%) | losses (> +1%) | median | best | worst | acceptance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2d_4352 | 14 | 0 | 0 | +0.0% | -0.0% Sls3HopEW_size_1024_ntype_unicast_write dev0 E | +0.1% SlsRowMerge_size_1024_ntype_unicast_write dev2 E | met |
| 1d_4352 | 6 | 4 | 0 | -7.4% | -16.9% SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | -0.9% SlsLinRowMerge_size_4352_ntype_unicast_write dev2 E | met |
| vc2_4352 | 11 | 11 | 0 | -15.7% | -42.3% Mesh3HopEW_size_4352_ntype_unicast_write dev0 E | -7.0% Mesh1HopEWBidir_size_4352_ntype_unicast_write dev1 W | met |
| 2d_3072 | 14 | 1 | 2 | -0.1% | -3.1% Sls3HopEW_size_3072_ntype_unicast_write dev0 E | +3.0% Sls1HopNS_size_3072_ntype_unicast_write dev0 S | NOT met |
| 1d_3072 | 6 | 2 | 2 | +0.8% | -1.9% SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | +1.6% SlsLin1HopEW_size_3072_ntype_unicast_write dev0 E | NOT met |
| ring_3072 | 14 | 0 | 0 | -0.0% | -0.2% Ring3HopEW_size_3072_ntype_unicast_scatter_write dev0 E | +0.0% UnidirRingMulticast_size_3072_ntype_unicast_write_num_links_1 dev7 W | met |
| vc2_3072 | 11 | 11 | 0 | -18.9% | -48.9% Mesh3HopEW_size_3072_ntype_unicast_write dev0 E | -15.6% Mesh1HopNS_size_3072_ntype_unicast_scatter_write dev0 S | met |
| ne_3072 | 2 | 0 | 0 | -0.2% | -0.4% NeighborExchange_size_3072_ntype_unicast_scatter_write_num_links_1 dev7 W | -0.0% NeighborExchange_size_3072_ntype_unicast_write_num_links_1 dev7 W | met |
