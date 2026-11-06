
## C pvc1_4352: VC1 forwarding (2, 4, 6, 7; rows 14,7,16,2,8 .. 14,7,16,7,8; passes 2)
| row | size | 2 | 4 | 6 | 7 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VC1Inter_0_0_to_1_0 | 256 | 210.5 | 210.5 | 210.6 | 210.5 | 2 |
| VC1Inter_0_0_to_1_0 | 1024 | 210.7 | 210.7 | 210.7 | 210.7 | 2 |
| VC1Inter_0_0_to_1_0 | 2048 | 211.1 | 211.0 | 211.0 | 211.1 | 2 |
| VC1Inter_0_0_to_1_0 | 4352 | 212.0 | 212.0 | 212.0 | 212.1 | 2 |
| VC1Inter_0_0_to_1_1 | 256 | 834.0 | 433.9 | 300.9 | 264.7 | >6 |
| VC1Inter_0_0_to_1_1 | 1024 | 848.4 | 441.3 | 307.1 | 267.3 | >6 |
| VC1Inter_0_0_to_1_1 | 2048 | 883.7 | 453.4 | 316.4 | 275.6 | >6 |
| VC1Inter_0_0_to_1_1 | 4352 | 932.2 | 486.9 | 335.7 | 292.5 | >6 |
| VC1Inter_0_0_to_1_2 | 256 | 849.1 | 440.7 | 306.0 | 268.2 | >6 |
| VC1Inter_0_0_to_1_2 | 1024 | 865.0 | 447.6 | 312.3 | 271.1 | >6 |
| VC1Inter_0_0_to_1_2 | 2048 | 900.3 | 462.6 | 321.3 | 280.8 | >6 |
| VC1Inter_0_0_to_1_2 | 4352 | 942.7 | 493.8 | 341.1 | 297.4 | >6 |
| VC1Inter_0_0_to_1_3 | 256 | 908.1 | 471.0 | 323.8 | 290.0 | >6 |
| VC1Inter_0_0_to_1_3 | 1024 | 921.5 | 479.2 | 328.7 | 294.3 | >6 |
| VC1Inter_0_0_to_1_3 | 2048 | 933.5 | 488.4 | 337.5 | 300.4 | >6 |
| VC1Inter_0_0_to_1_3 | 4352 | 1008.7 | 519.2 | 356.3 | 316.2 | >6 |
| VC1Mesh2Hop | 256 | 274.4 | 274.4 | 274.4 | 274.4 | 2 |
| VC1Mesh2Hop | 1024 | 282.4 | 282.7 | 282.5 | 282.6 | 2 |
| VC1Mesh2Hop | 2048 | 288.4 | 288.2 | 288.1 | 288.3 | 2 |
| VC1Mesh2Hop | 4352 | 304.7 | 304.4 | 304.6 | 304.5 | 2 |
| VC1MeshMulticast | 256 | 155.7 | 155.7 | 155.7 | 155.7 | 2 |
| VC1MeshMulticast | 1024 | 157.7 | 157.6 | 157.6 | 157.7 | 2 |
| VC1MeshMulticast | 2048 | 161.4 | 161.4 | 161.4 | 161.5 | 2 |
| VC1MeshMulticast | 4352 | 176.1 | 176.1 | 176.1 | 176.0 | 2 |
| VC1MixForward | 256 | 248.2 | 207.9 | 174.9 | 165.1 | >6 |
| VC1MixForward | 1024 | 251.8 | 210.6 | 177.2 | 167.3 | >6 |
| VC1MixForward | 2048 | 257.6 | 215.3 | 181.1 | 170.8 | >6 |
| VC1MixForward | 4352 | 270.9 | 228.1 | 191.9 | 180.7 | >6 |
| VC1MixWorker | 256 | 157.0 | 146.3 | 126.7 | 121.5 | >6 |
| VC1MixWorker | 1024 | 157.2 | 147.1 | 127.6 | 122.4 | >6 |
| VC1MixWorker | 2048 | 159.4 | 148.5 | 129.1 | 124.1 | >6 |
| VC1MixWorker | 4352 | 158.7 | 149.7 | 133.0 | 127.7 | >6 |
| VC1SoloForward | 256 | 307.8 | 307.6 | 307.5 | 307.5 | 2 |
| VC1SoloForward | 1024 | 312.9 | 313.0 | 312.8 | 312.8 | 2 |
| VC1SoloForward | 2048 | 319.5 | 319.2 | 319.1 | 319.2 | 2 |
| VC1SoloForward | 4352 | 335.4 | 335.0 | 335.2 | 335.1 | 2 |
| VC1SoloWorker | 256 | 151.8 | 151.7 | 151.7 | 151.7 | 2 |
| VC1SoloWorker | 1024 | 152.7 | 152.7 | 152.7 | 152.7 | 2 |
| VC1SoloWorker | 2048 | 154.6 | 154.7 | 154.7 | 154.7 | 2 |
| VC1SoloWorker | 4352 | 159.9 | 160.0 | 159.9 | 159.9 | 2 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >6 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >6, >6, >6, >6, >6)
- 1024 B: >6 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >6, >6, >6, >6, >6)
- 2048 B: >6 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >6, >6, >6, >6, >6)
- 4352 B: >6 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >6, >6, >6, >6, >6)

## C pvc1_4352: VC1 receiver (2, 4, 8; rows 14,7,16,6,2 .. 14,7,16,6,8; passes 2)
| row | size | 2 | 4 | 8 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| VC1Inter_0_0_to_1_0 | 256 | 210.5 | 210.6 | 210.6 | 2 |
| VC1Inter_0_0_to_1_0 | 1024 | 210.7 | 210.7 | 210.7 | 2 |
| VC1Inter_0_0_to_1_0 | 2048 | 211.1 | 211.1 | 211.0 | 2 |
| VC1Inter_0_0_to_1_0 | 4352 | 212.1 | 212.0 | 212.0 | 2 |
| VC1Inter_0_0_to_1_1 | 256 | 620.2 | 311.1 | 300.9 | >4 |
| VC1Inter_0_0_to_1_1 | 1024 | 627.4 | 312.3 | 307.1 | >4 |
| VC1Inter_0_0_to_1_1 | 2048 | 637.8 | 319.0 | 316.4 | 3-4 |
| VC1Inter_0_0_to_1_1 | 4352 | 701.9 | 368.8 | 335.7 | >4 |
| VC1Inter_0_0_to_1_2 | 256 | 625.5 | 312.1 | 306.0 | >4 |
| VC1Inter_0_0_to_1_2 | 1024 | 629.1 | 313.6 | 312.3 | 3-4 |
| VC1Inter_0_0_to_1_2 | 2048 | 645.0 | 324.3 | 321.3 | 3-4 |
| VC1Inter_0_0_to_1_2 | 4352 | 724.8 | 374.5 | 341.1 | >4 |
| VC1Inter_0_0_to_1_3 | 256 | 636.9 | 344.0 | 323.8 | >4 |
| VC1Inter_0_0_to_1_3 | 1024 | 646.3 | 347.9 | 328.7 | >4 |
| VC1Inter_0_0_to_1_3 | 2048 | 675.9 | 356.5 | 337.5 | >4 |
| VC1Inter_0_0_to_1_3 | 4352 | 710.5 | 379.1 | 356.3 | >4 |
| VC1Mesh2Hop | 256 | 274.4 | 274.4 | 274.4 | 2 |
| VC1Mesh2Hop | 1024 | 282.8 | 283.0 | 282.5 | 2 |
| VC1Mesh2Hop | 2048 | 288.2 | 288.2 | 288.1 | 2 |
| VC1Mesh2Hop | 4352 | 304.6 | 304.7 | 304.6 | 2 |
| VC1MeshMulticast | 256 | 155.5 | 155.5 | 155.7 | 2 |
| VC1MeshMulticast | 1024 | 157.6 | 157.5 | 157.6 | 2 |
| VC1MeshMulticast | 2048 | 161.4 | 161.3 | 161.4 | 2 |
| VC1MeshMulticast | 4352 | 175.9 | 175.8 | 176.1 | 2 |
| VC1MixForward | 256 | 232.2 | 182.5 | 174.9 | >4 |
| VC1MixForward | 1024 | 234.8 | 184.5 | 177.2 | >4 |
| VC1MixForward | 2048 | 239.2 | 187.7 | 181.1 | >4 |
| VC1MixForward | 4352 | 252.6 | 199.1 | 191.9 | >4 |
| VC1MixWorker | 256 | 158.2 | 133.3 | 126.7 | >4 |
| VC1MixWorker | 1024 | 158.6 | 134.3 | 127.6 | >4 |
| VC1MixWorker | 2048 | 158.8 | 136.0 | 129.1 | >4 |
| VC1MixWorker | 4352 | 159.5 | 140.7 | 133.0 | >4 |
| VC1SoloForward | 256 | 307.5 | 307.7 | 307.5 | 2 |
| VC1SoloForward | 1024 | 312.8 | 312.8 | 312.8 | 2 |
| VC1SoloForward | 2048 | 319.2 | 319.1 | 319.1 | 2 |
| VC1SoloForward | 4352 | 335.3 | 335.3 | 335.2 | 2 |
| VC1SoloWorker | 256 | 151.7 | 151.7 | 151.7 | 2 |
| VC1SoloWorker | 1024 | 152.7 | 152.8 | 152.7 | 2 |
| VC1SoloWorker | 2048 | 154.7 | 154.7 | 154.7 | 2 |
| VC1SoloWorker | 4352 | 159.9 | 159.9 | 159.9 | 2 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >4 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >4, >4, >4, >4, >4)
- 1024 B: >4 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, 3-4, >4, >4, >4, >4)
- 2048 B: >4 (VC1Inter_0_0_to_1_3; rows' knees 2, 2, 2, 2, 2, 3-4, 3-4, >4, >4, >4)
- 4352 B: >4 (VC1Inter_0_0_to_1_1; rows' knees 2, 2, 2, 2, 2, >4, >4, >4, >4, >4)

## Other arms (checks, squeeze, validation): cycles/packet
- F sqw_2d (p2sq_15232, 13,2,4): Sls1HopEW 256 B 270.6; Sls1HopEW 4352 B 333.0; Sls1HopEW 15232 B 448.5; Sls1HopEWBidir 256 B 270.5; Sls1HopEWBidir 4352 B 346.4; Sls1HopEWBidir 15232 B 487.4; Sls3HopEW 256 B 764.6; Sls3HopEW 4352 B 886.7; Sls3HopEW 15232 B 1227.6; SlsMulticast 256 B 1547.5; SlsMulticast 4352 B 1849.2; SlsMulticast 15232 B 2623.9
- F sqf_2d (p2sq_15232, 4,5,4): Sls1HopEW 256 B 371.7; Sls1HopEW 4352 B 426.6; Sls1HopEW 15232 B 552.1; Sls1HopEWBidir 256 B 372.4; Sls1HopEWBidir 4352 B 429.8; Sls1HopEWBidir 15232 B 552.8; Sls3HopEW 256 B 376.7; Sls3HopEW 4352 B 423.9; Sls3HopEW 15232 B 552.1; SlsMulticast 256 B 851.7; SlsMulticast 4352 B 1036.5; SlsMulticast 15232 B 1486.0
- F sqr_2d (p2sq_15232, 9,2,8): Sls1HopEW 256 B 188.2; Sls1HopEW 4352 B 202.9; Sls1HopEW 15232 B 423.3; Sls1HopEWBidir 256 B 188.0; Sls1HopEWBidir 4352 B 211.4; Sls1HopEWBidir 15232 B 426.5; Sls3HopEW 256 B 819.1; Sls3HopEW 4352 B 923.3; Sls3HopEW 15232 B 1232.1; SlsMulticast 256 B 1648.4; SlsMulticast 4352 B 1874.3; SlsMulticast 15232 B 2680.0
- F sqt_2d (p2sq_15232, 7,4,4): Sls1HopEW 256 B 270.7; Sls1HopEW 4352 B 333.3; Sls1HopEW 15232 B 448.7; Sls1HopEWBidir 256 B 270.6; Sls1HopEWBidir 4352 B 346.2; Sls1HopEWBidir 15232 B 487.6; Sls3HopEW 256 B 393.8; Sls3HopEW 4352 B 462.3; Sls3HopEW 15232 B 615.6; SlsMulticast 256 B 910.9; SlsMulticast 4352 B 1109.0; SlsMulticast 15232 B 1600.6
- F sqw_1d (p1sq_15232, 13,2,8): SlsLin1HopEW 256 B 150.1; SlsLin1HopEW 4352 B 174.7; SlsLin1HopEW 15232 B 422.0; SlsLin3HopEW 256 B 813.1; SlsLin3HopEW 4352 B 920.7; SlsLin3HopEW 15232 B 1217.3; SlsLinRowMerge 256 B 483.1; SlsLinRowMerge 4352 B 554.7; SlsLinRowMerge 15232 B 838.2
- F sqf_1d (p1sq_15232, 4,11,8): SlsLin1HopEW 256 B 399.7; SlsLin1HopEW 4352 B 438.9; SlsLin1HopEW 15232 B 532.2; SlsLin3HopEW 256 B 402.8; SlsLin3HopEW 4352 B 440.7; SlsLin3HopEW 15232 B 532.0; SlsLinRowMerge 256 B 186.1; SlsLinRowMerge 4352 B 211.4; SlsLinRowMerge 15232 B 436.3
- F sqr_1d (p1sq_15232, 4,3,16): SlsLin1HopEW 256 B 419.6; SlsLin1HopEW 4352 B 474.0; SlsLin1HopEW 15232 B 532.4; SlsLin3HopEW 256 B 547.4; SlsLin3HopEW 4352 B 617.6; SlsLin3HopEW 15232 B 811.2; SlsLinRowMerge 256 B 320.8; SlsLinRowMerge 4352 B 374.3; SlsLinRowMerge 15232 B 552.7
- F sqt_1d (p1sq_15232, 8,7,8): SlsLin1HopEW 256 B 196.6; SlsLin1HopEW 4352 B 221.9; SlsLin1HopEW 15232 B 422.0; SlsLin3HopEW 256 B 247.8; SlsLin3HopEW 4352 B 276.7; SlsLin3HopEW 15232 B 422.0; SlsLinRowMerge 256 B 169.7; SlsLinRowMerge 4352 B 193.8; SlsLinRowMerge 15232 B 422.0
