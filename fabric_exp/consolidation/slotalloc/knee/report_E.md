
## E p1f_8704: forwarding (2, 4, 9; rows 16,2,16 .. 16,9,16; passes 1)
| row | size | 2 | 4 | 9 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| SlsLin3HopEW | 256 | 748.2 | 413.6 | 195.0 | ≥9 |
| SlsLin3HopEW | 1024 | 767.0 | 427.4 | 201.0 | ≥9 |
| SlsLin3HopEW | 4352 | 838.4 | 475.8 | 221.3 | ≥9 |
| SlsLin3HopEW | 8704 | 932.8 | 476.0 | 255.1 | ≥9 |
| SlsLinRowMerge | 256 | 497.1 | 274.1 | 139.8 | ≥9 |
| SlsLinRowMerge | 1024 | 509.4 | 286.9 | 141.6 | ≥9 |
| SlsLinRowMerge | 4352 | 559.9 | 324.8 | 161.7 | ≥9 |
| SlsLinRowMerge | 8704 | 674.4 | 377.0 | 245.9 | ≥9 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: ≥9 (SlsLin3HopEW; rows' knees ≥9, ≥9)
- 1024 B: ≥9 (SlsLin3HopEW; rows' knees ≥9, ≥9)
- 4352 B: ≥9 (SlsLin3HopEW; rows' knees ≥9, ≥9)
- 8704 B: ≥9 (SlsLin3HopEW; rows' knees ≥9, ≥9)

## E p1w_8704: receiver (8, 16; rows 16,2,8 .. 16,2,16; passes 1)
| row | size | 8 | 16 | knee |
| --- | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 149.8 | 114.2 | ≥16 |
| Lin1HopEWBidir | 1024 | 152.1 | 116.3 | ≥16 |
| Lin1HopEWBidir | 4352 | 192.1 | 140.2 | ≥16 |
| Lin1HopEWBidir | 8704 | 247.3 | 244.8 | ≥16 |
| SlsLin1HopEW | 256 | 150.0 | 114.3 | ≥16 |
| SlsLin1HopEW | 1024 | 151.2 | 116.1 | ≥16 |
| SlsLin1HopEW | 4352 | 174.6 | 130.2 | ≥16 |
| SlsLin1HopEW | 8704 | 243.8 | 243.8 | 8 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: ≥16 (Lin1HopEWBidir; rows' knees ≥16, ≥16)
- 1024 B: ≥16 (Lin1HopEWBidir; rows' knees ≥16, ≥16)
- 4352 B: ≥16 (Lin1HopEWBidir; rows' knees ≥16, ≥16)
- 8704 B: ≥16 (Lin1HopEWBidir; rows' knees 8, ≥16)

## E p1w_8704: worker (4, 12, 20; rows 4,2,16 .. 20,2,16; passes 1)
| row | size | 4 | 12 | 20 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 402.4 | 147.0 | 98.4 | ≥20 |
| Lin1HopEWBidir | 1024 | 413.4 | 149.7 | 98.9 | ≥20 |
| Lin1HopEWBidir | 4352 | 446.6 | 180.4 | 124.9 | ≥20 |
| Lin1HopEWBidir | 8704 | 471.7 | 244.8 | 244.8 | 12 |
| SlsLin1HopEW | 256 | 420.9 | 147.0 | 98.6 | ≥20 |
| SlsLin1HopEW | 1024 | 410.0 | 149.0 | 98.9 | ≥20 |
| SlsLin1HopEW | 4352 | 474.5 | 168.6 | 124.5 | ≥20 |
| SlsLin1HopEW | 8704 | 457.7 | 243.8 | 243.8 | 12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: ≥20 (Lin1HopEWBidir; rows' knees ≥20, ≥20)
- 1024 B: ≥20 (Lin1HopEWBidir; rows' knees ≥20, ≥20)
- 4352 B: ≥20 (Lin1HopEWBidir; rows' knees ≥20, ≥20)
- 8704 B: 12 (Lin1HopEWBidir; rows' knees 12, 12)

## E p2w_8704: worker (4, 8, 12, 16; rows 4,2,16 .. 16,2,16; passes 1)
| row | size | 4 | 8 | 12 | 16 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEW | 256 | 426.0 | 231.5 | 155.9 | 121.2 | ≥16 |
| Sls1HopEW | 1024 | 446.8 | 232.0 | 157.5 | 122.0 | ≥16 |
| Sls1HopEW | 4352 | 514.4 | 254.4 | 173.2 | 133.3 | ≥16 |
| Sls1HopEW | 8704 | 466.9 | 266.8 | 245.1 | 245.1 | 12 |
| Sls1HopEWBidir | 256 | 422.8 | 232.8 | 155.9 | 120.8 | ≥16 |
| Sls1HopEWBidir | 1024 | 448.1 | 232.4 | 158.0 | 122.2 | ≥16 |
| Sls1HopEWBidir | 4352 | 511.2 | 257.0 | 184.4 | 142.5 | ≥16 |
| Sls1HopEWBidir | 8704 | 485.8 | 284.1 | 247.3 | 247.2 | 12 |
| Sls1HopNS | 256 | 423.9 | 231.6 | 156.2 | 121.3 | ≥16 |
| Sls1HopNS | 1024 | 446.8 | 232.7 | 157.9 | 122.3 | ≥16 |
| Sls1HopNS | 4352 | 512.2 | 255.2 | 173.3 | 133.8 | ≥16 |
| Sls1HopNS | 8704 | 467.4 | 267.7 | 245.1 | 245.1 | 12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: ≥16 (Sls1HopEW; rows' knees ≥16, ≥16, ≥16)
- 1024 B: ≥16 (Sls1HopEW; rows' knees ≥16, ≥16, ≥16)
- 4352 B: ≥16 (Sls1HopEW; rows' knees ≥16, ≥16, ≥16)
- 8704 B: 12 (Sls1HopEW; rows' knees 12, 12, 12)
