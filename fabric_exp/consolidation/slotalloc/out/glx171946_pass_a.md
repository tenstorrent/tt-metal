### 1d
| row | tier | dyn | dynw24 |
| --- | ---: | ---: | ---: |
| SlsLin1HopEW_size_1024_ntype_unicast_write dev0 E | 121.1 | 95.4 (-21.2%) | 95.9 (-20.8%) |
| SlsLin3HopEW_size_1024_ntype_unicast_write dev0 E | 132.7 | 118.7 (-10.5%) | 118.4 (-10.8%) |
| SlsLin7HopNS_size_1024_ntype_unicast_write dev0 S | 155.8 | 120.9 (-22.4%) | 119.7 (-23.2%) |
| SlsLinColMerge_size_1024_ntype_unicast_write dev24 S | 127.6 | 117.9 (-7.6%) | 117.6 (-7.8%) |
| SlsLinRowMerge_size_1024_ntype_unicast_write dev2 E | 111.0 | 108.8 (-1.9%) | 108.8 (-2.0%) |

dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
dynw24: 0 rows > 1% slower than tier (mixed tests by link total)
dynw24: 0 rows > 2% slower than tier
### vc1
| row | tier | dyn |
| --- | ---: | ---: |
| GlxInter_0_0_to_1_0_size_16_ntype_unicast_write dev0 E | 512.3 | 289.6 (-43.5%) |
| GlxInter_0_0_to_1_0_size_4352_ntype_unicast_write dev0 E | 606.5 | 327.5 (-46.0%) |
| GlxInter_0_0_to_1_1_size_16_ntype_unicast_write dev0 E | 875.7 | 366.2 (-58.2%) |
| GlxInter_0_0_to_1_1_size_4352_ntype_unicast_write dev0 E | 1055.0 | 429.8 (-59.3%) |
| GlxInter_0_0_to_1_2_size_16_ntype_unicast_write dev0 E | 740.9 | 337.7 (-54.4%) |
| GlxInter_0_0_to_1_2_size_4352_ntype_unicast_write dev0 E | 930.4 | 405.2 (-56.4%) |
| GlxInter_0_0_to_1_3_size_16_ntype_unicast_write dev0 E | 805.8 | 364.4 (-54.8%) |
| GlxInter_0_0_to_1_3_size_4352_ntype_unicast_write dev0 E | 982.4 | 411.2 (-58.1%) |
| GlxInter_0_1_to_1_0_size_16_ntype_unicast_write dev1 N | 234.6 | 209.3 (-10.8%) |
| GlxInter_0_1_to_1_0_size_4352_ntype_unicast_write dev1 N | 259.6 | 252.1 (-2.9%) |
| GlxInter_0_1_to_1_1_size_16_ntype_unicast_write dev1 N | 875.3 | 364.7 (-58.3%) |
| GlxInter_0_1_to_1_1_size_4352_ntype_unicast_write dev1 N | 1054.1 | 428.8 (-59.3%) |
| GlxInter_0_1_to_1_2_size_16_ntype_unicast_write dev1 N | 746.8 | 336.3 (-55.0%) |
| GlxInter_0_1_to_1_2_size_4352_ntype_unicast_write dev1 N | 939.1 | 405.0 (-56.9%) |
| GlxInter_0_1_to_1_3_size_16_ntype_unicast_write dev1 N | 811.2 | 365.1 (-55.0%) |
| GlxInter_0_1_to_1_3_size_4352_ntype_unicast_write dev1 N | 981.7 | 410.7 (-58.2%) |
| GlxMixForward_size_16_ntype_unicast_write dev1 N (flow) | 921.0 | 403.6 (-56.2%) |
| GlxMixForward_size_16_ntype_unicast_write dev2 N (flow) | 499.6 | 300.2 (-39.9%) |
| GlxMixForward_size_16_ntype_unicast_write LINK TOTAL | 323.9 | 172.1 (-46.9%) |
| GlxMixForward_size_4352_ntype_unicast_write dev1 N (flow) | 1109.6 | 657.6 (-40.7%) |
| GlxMixForward_size_4352_ntype_unicast_write dev2 N (flow) | 569.1 | 393.9 (-30.8%) |
| GlxMixForward_size_4352_ntype_unicast_write LINK TOTAL | 376.2 | 246.3 (-34.5%) |
| GlxMixWorker_size_16_ntype_unicast_write dev0 E (flow) | 214.2 | 204.4 (-4.6%) |
| GlxMixWorker_size_16_ntype_unicast_write dev1 N (flow) | 928.4 | 405.6 (-56.3%) |
| GlxMixWorker_size_16_ntype_unicast_write LINK TOTAL | 174.0 | 135.9 (-21.9%) |
| GlxMixWorker_size_4352_ntype_unicast_write dev0 E (flow) | 307.7 | 319.1 (+3.7%) |
| GlxMixWorker_size_4352_ntype_unicast_write dev1 N (flow) | 1239.9 | 941.2 (-24.1%) |
| GlxMixWorker_size_4352_ntype_unicast_write LINK TOTAL | 246.5 | 238.3 (-3.3%) |
| GlxSoloForward_size_16_ntype_unicast_write dev2 N | 469.0 | 249.5 (-46.8%) |
| GlxSoloForward_size_4352_ntype_unicast_write dev2 N | 545.6 | 283.2 (-48.1%) |
| GlxSoloWorker_size_16_ntype_unicast_write dev0 E | 180.0 | 142.9 (-20.6%) |
| VC1BurstEW_N32_G500_size_16 dev0 E | 184.1 | 142.9 (-22.4%) |
| VC1BurstEW_N8_G500_size_16 dev0 E | 184.1 | 142.8 (-22.4%) |
| VC1Mesh1HopEW_size_16_ntype_unicast_write dev0 E | 184.0 | 142.6 (-22.5%) |
| VC1Mesh1HopNS_size_16_ntype_unicast_write dev0 S | 181.8 | 143.1 (-21.3%) |
| VC1Mesh2HopMerge_size_16_ntype_unicast_write dev2 E | 239.0 | 139.4 (-41.7%) |
| VC1Mesh2HopMerge_size_4352_ntype_unicast_write dev2 E | 305.7 | 254.0 (-16.9%) |
| VC1Mesh2Hop_size_16_ntype_unicast_write dev0 S | 444.7 | 240.1 (-46.0%) |
| VC1Mesh2Hop_size_4352_ntype_unicast_write dev0 S | 518.2 | 276.8 (-46.6%) |
| VC1MeshMulticast_size_4352_ntype_unicast_write dev3 W | 337.6 | 260.1 (-23.0%) |

dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
### vc2k
| row | tier | def | dyn |
| --- | ---: | ---: | ---: |
| Vc02A_size_2048_ntype_unicast_write dev0 E | 596.5 | 592.7 (-0.7%) | 460.3 (-22.8%) |
| Vc02B_size_2048_ntype_unicast_write dev0 E | 315.3 | 307.7 (-2.4%) | 243.9 (-22.6%) |
| Vc2Alone_size_2048_ntype_unicast_write dev0 E | 877.8 | 877.8 (-0.0%) | 676.6 (-22.9%) |
| Vc2OnVC0_size_2048_ntype_unicast_write dev0 E | 152.6 | 133.9 (-12.3%) | 133.9 (-12.3%) |

def: 0 rows > 1% slower than tier (mixed tests by link total)
def: 0 rows > 2% slower than tier
dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
### 2d3k
| row | tier | dyn | dynw24 | row |
| --- | ---: | ---: | ---: | ---: |
| Sls3HopNS_size_1024_ntype_unicast_write dev0 S | 151.8 | 139.9 (-7.8%) | 139.7 (-8.0%) | 141.9 (-6.5%) |
| Sls3HopNS_size_256_ntype_unicast_write dev0 S | 148.9 | 140.0 (-6.0%) | 139.8 (-6.1%) | 141.4 (-5.1%) |
| Sls7HopNS_size_1024_ntype_unicast_write dev0 S | 167.4 | 141.4 (-15.5%) | 141.3 (-15.6%) | 143.7 (-14.2%) |
| Sls7HopNS_size_256_ntype_unicast_write dev0 S | 162.3 | 140.7 (-13.3%) | 140.7 (-13.3%) | 143.6 (-11.5%) |
| Sls7HopNS_size_3072_ntype_unicast_write dev0 S | 184.2 | 179.6 (-2.5%) | 179.6 (-2.5%) | 179.6 (-2.5%) |
| SlsColMerge_size_1024_ntype_unicast_write dev24 S | 143.5 | 136.6 (-4.8%) | 136.5 (-4.9%) | 135.7 (-5.4%) |
| SlsColMerge_size_256_ntype_unicast_write dev24 S | 143.2 | 136.4 (-4.7%) | 136.2 (-4.9%) | 135.7 (-5.2%) |
| SlsCorner10_size_1024_ntype_unicast_write dev0 S | 165.6 | 142.7 (-13.8%) | 142.7 (-13.8%) | 143.7 (-13.2%) |
| SlsCorner10_size_256_ntype_unicast_write dev0 S | 162.8 | 142.6 (-12.4%) | 143.0 (-12.2%) | 143.4 (-11.9%) |
| SlsCorner10_size_3072_ntype_unicast_write dev0 S | 183.4 | 179.6 (-2.1%) | 179.6 (-2.1%) | 179.6 (-2.1%) |
| SlsMulticast_size_1024_ntype_unicast_write dev31 W | 571.8 | 560.7 (-1.9%) | 560.7 (-1.9%) | 561.0 (-1.9%) |
| SlsMulticast_size_256_ntype_unicast_write dev31 W | 571.2 | 560.8 (-1.8%) | 560.5 (-1.9%) | 560.7 (-1.8%) |
| SlsMulticast_size_3072_ntype_unicast_write dev31 W | 716.9 | 705.8 (-1.6%) | 705.8 (-1.6%) | 705.8 (-1.5%) |

dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
dynw24: 0 rows > 1% slower than tier (mixed tests by link total)
dynw24: 0 rows > 2% slower than tier
row: 0 rows > 1% slower than tier (mixed tests by link total)
row: 0 rows > 2% slower than tier
### vc13k
| row | tier | dyn | dynw24 | row |
| --- | ---: | ---: | ---: | ---: |
| GlxInter_0_0_to_1_0_size_256_ntype_unicast_write dev0 E | 511.4 | 211.7 (-58.6%) | 209.6 (-59.0%) | 210.4 (-58.9%) |
| GlxInter_0_0_to_1_0_size_3072_ntype_unicast_write dev0 E | 568.7 | 215.8 (-62.1%) | 211.0 (-62.9%) | 213.1 (-62.5%) |
| GlxInter_0_0_to_1_1_size_256_ntype_unicast_write dev0 E | 876.8 | 270.2 (-69.2%) | 270.4 (-69.2%) | 251.1 (-71.4%) |
| GlxInter_0_0_to_1_1_size_3072_ntype_unicast_write dev0 E | 996.4 | 296.5 (-70.2%) | 297.1 (-70.2%) | 255.7 (-74.3%) |
| GlxInter_0_0_to_1_2_size_256_ntype_unicast_write dev0 E | 741.6 | 258.7 (-65.1%) | 258.1 (-65.2%) | 249.7 (-66.3%) |
| GlxInter_0_0_to_1_2_size_3072_ntype_unicast_write dev0 E | 860.5 | 267.9 (-68.9%) | 266.9 (-69.0%) | 252.7 (-70.6%) |
| GlxInter_0_0_to_1_3_size_256_ntype_unicast_write dev0 E | 805.6 | 264.1 (-67.2%) | 264.1 (-67.2%) | 248.7 (-69.1%) |
| GlxInter_0_0_to_1_3_size_3072_ntype_unicast_write dev0 E | 928.0 | 290.2 (-68.7%) | 290.3 (-68.7%) | 252.9 (-72.8%) |
| GlxInter_0_1_to_1_0_size_256_ntype_unicast_write dev1 N | 233.7 | 208.3 (-10.9%) | 208.3 (-10.9%) | 208.3 (-10.8%) |
| GlxInter_0_1_to_1_0_size_3072_ntype_unicast_write dev1 N | 244.6 | 209.1 (-14.5%) | 209.0 (-14.5%) | 209.1 (-14.5%) |
| GlxInter_0_1_to_1_1_size_256_ntype_unicast_write dev1 N | 877.4 | 270.8 (-69.1%) | 270.7 (-69.1%) | 250.2 (-71.5%) |
| GlxInter_0_1_to_1_1_size_3072_ntype_unicast_write dev1 N | 988.9 | 297.2 (-69.9%) | 297.4 (-69.9%) | 254.7 (-74.2%) |
| GlxInter_0_1_to_1_2_size_256_ntype_unicast_write dev1 N | 746.6 | 257.6 (-65.5%) | 257.7 (-65.5%) | 248.6 (-66.7%) |
| GlxInter_0_1_to_1_2_size_3072_ntype_unicast_write dev1 N | 868.0 | 266.5 (-69.3%) | 266.6 (-69.3%) | 251.6 (-71.0%) |
| GlxInter_0_1_to_1_3_size_256_ntype_unicast_write dev1 N | 810.6 | 264.1 (-67.4%) | 264.3 (-67.4%) | 247.8 (-69.4%) |
| GlxInter_0_1_to_1_3_size_3072_ntype_unicast_write dev1 N | 930.9 | 290.4 (-68.8%) | 290.2 (-68.8%) | 252.1 (-72.9%) |
| GlxMixForward_size_256_ntype_unicast_write dev1 N (flow) | 921.8 | 316.4 (-65.7%) | 316.5 (-65.7%) | 274.1 (-70.3%) |
| GlxMixForward_size_256_ntype_unicast_write dev2 N (flow) | 499.4 | 246.3 (-50.7%) | 245.8 (-50.8%) | 238.8 (-52.2%) |
| GlxMixForward_size_256_ntype_unicast_write LINK TOTAL | 323.9 | 138.5 (-57.2%) | 138.3 (-57.3%) | 127.6 (-60.6%) |
| GlxMixForward_size_3072_ntype_unicast_write dev1 N (flow) | 1043.8 | 462.8 (-55.7%) | 477.7 (-54.2%) | 451.3 (-56.8%) |
| GlxMixForward_size_3072_ntype_unicast_write dev2 N (flow) | 556.6 | 285.0 (-48.8%) | 269.7 (-51.5%) | 295.5 (-46.9%) |
| GlxMixForward_size_3072_ntype_unicast_write LINK TOTAL | 363.0 | 176.4 (-51.4%) | 172.4 (-52.5%) | 178.6 (-50.8%) |
| GlxMixWorker_size_256_ntype_unicast_write dev0 E (flow) | 213.2 | 192.1 (-9.9%) | 192.8 (-9.6%) | 187.3 (-12.2%) |
| GlxMixWorker_size_256_ntype_unicast_write dev1 N (flow) | 930.4 | 309.9 (-66.7%) | 310.0 (-66.7%) | 260.9 (-72.0%) |
| GlxMixWorker_size_256_ntype_unicast_write LINK TOTAL | 173.5 | 118.6 (-31.6%) | 118.9 (-31.5%) | 109.0 (-37.2%) |
| GlxMixWorker_size_3072_ntype_unicast_write dev0 E (flow) | 232.6 | 251.4 (+8.1%) | 251.4 (+8.1%) | 251.4 (+8.1%) |
| GlxMixWorker_size_3072_ntype_unicast_write dev1 N (flow) | 1049.9 | 598.4 (-43.0%) | 598.3 (-43.0%) | 539.0 (-48.7%) |
| GlxMixWorker_size_3072_ntype_unicast_write LINK TOTAL | 190.4 | 177.0 (-7.0%) | 177.0 (-7.0%) | 171.4 (-10.0%) |
| GlxSoloForward_size_256_ntype_unicast_write dev2 N | 469.0 | 170.6 (-63.6%) | 168.0 (-64.2%) | 168.7 (-64.0%) |
| GlxSoloForward_size_3072_ntype_unicast_write dev2 N | 536.1 | 186.3 (-65.2%) | 179.6 (-66.5%) | 180.9 (-66.3%) |
| GlxSoloWorker_size_256_ntype_unicast_write dev0 E | 180.4 | 140.5 (-22.1%) | 141.9 (-21.3%) | 141.5 (-21.5%) |
| GlxSoloWorker_size_3072_ntype_unicast_write dev0 E | 208.5 | 179.6 (-13.9%) | 179.6 (-13.9%) | 179.6 (-13.9%) |
| VC1BurstEW_N32_G500_size_256 dev0 E | 184.0 | 140.6 (-23.6%) | 141.9 (-22.9%) | 142.0 (-22.8%) |
| VC1BurstEW_N32_G500_size_3072 dev0 E | 208.5 | 179.7 (-13.8%) | 179.7 (-13.8%) | 179.7 (-13.8%) |
| VC1BurstEW_N8_G500_size_256 dev0 E | 184.1 | 140.6 (-23.6%) | 141.6 (-23.1%) | 141.7 (-23.0%) |
| VC1BurstEW_N8_G500_size_3072 dev0 E | 208.4 | 179.7 (-13.8%) | 179.7 (-13.8%) | 179.7 (-13.8%) |
| VC1Mesh1HopEW_size_256_ntype_unicast_write dev0 E | 184.0 | 140.5 (-23.6%) | 141.4 (-23.1%) | 141.8 (-22.9%) |
| VC1Mesh1HopEW_size_3072_ntype_unicast_write dev0 E | 208.9 | 179.6 (-14.0%) | 179.6 (-14.0%) | 179.6 (-14.0%) |
| VC1Mesh1HopNS_size_256_ntype_unicast_write dev0 S | 182.3 | 140.8 (-22.8%) | 141.8 (-22.2%) | 141.9 (-22.1%) |
| VC1Mesh1HopNS_size_3072_ntype_unicast_write dev0 S | 208.7 | 179.6 (-14.0%) | 179.6 (-14.0%) | 179.6 (-13.9%) |
| VC1Mesh2HopMerge_size_256_ntype_unicast_write dev2 E | 239.0 | 130.3 (-45.5%) | 130.2 (-45.5%) | 130.3 (-45.5%) |
| VC1Mesh2HopMerge_size_3072_ntype_unicast_write dev2 E | 266.6 | 179.6 (-32.6%) | 179.6 (-32.6%) | 179.6 (-32.6%) |
| VC1Mesh2Hop_size_256_ntype_unicast_write dev0 S | 444.5 | 168.9 (-62.0%) | 168.4 (-62.1%) | 168.5 (-62.1%) |
| VC1Mesh2Hop_size_3072_ntype_unicast_write dev0 S | 502.9 | 182.0 (-63.8%) | 179.6 (-64.3%) | 179.7 (-64.3%) |
| VC1MeshMulticast_size_3072_ntype_unicast_write dev3 W | 299.1 | 180.9 (-39.5%) | 180.9 (-39.5%) | 180.9 (-39.5%) |

dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
dynw24: 0 rows > 1% slower than tier (mixed tests by link total)
dynw24: 0 rows > 2% slower than tier
row: 0 rows > 1% slower than tier (mixed tests by link total)
row: 0 rows > 2% slower than tier
### vc23k
| row | tier | def | dyn | row |
| --- | ---: | ---: | ---: | ---: |
| Vc02A_size_1024_ntype_unicast_write dev0 E | 573.2 | 569.7 (-0.6%) | 239.7 (-58.2%) | 116.7 (-79.6%) |
| Vc02A_size_3072_ntype_unicast_write dev0 E | 628.2 | 637.9 (+1.5%) | 289.7 (-53.9%) | 179.6 (-71.4%) |
| Vc02B_size_1024_ntype_unicast_write dev0 E | 303.7 | 297.0 (-2.2%) | 132.9 (-56.2%) | 110.4 (-63.7%) |
| Vc02B_size_3072_ntype_unicast_write dev0 E | 343.6 | 362.9 (+5.6%) | 207.9 (-39.5%) | 179.6 (-47.7%) |
| Vc2Alone_size_1024_ntype_unicast_write dev0 E | 842.8 | 842.7 (-0.0%) | 346.3 (-58.9%) | 143.3 (-83.0%) |
| Vc2Alone_size_3072_ntype_unicast_write dev0 E | 912.8 | 912.8 (+0.0%) | 371.4 (-59.3%) | 179.6 (-80.3%) |
| Vc2OnVC0_size_1024_ntype_unicast_write dev0 E | 148.8 | 124.1 (-16.6%) | 124.4 (-16.4%) | 124.4 (-16.4%) |

def: 2 rows > 1% slower than tier (mixed tests by link total); worst +5.6% Vc02B_size_3072_ntype_unicast_write
def: 1 rows > 2% slower than tier; worst +5.6% Vc02B_size_3072_ntype_unicast_write
dyn: 0 rows > 1% slower than tier (mixed tests by link total)
dyn: 0 rows > 2% slower than tier
row: 0 rows > 1% slower than tier (mixed tests by link total)
row: 0 rows > 2% slower than tier
