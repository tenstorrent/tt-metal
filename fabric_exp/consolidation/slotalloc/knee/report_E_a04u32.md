
## E p1f_8704: forwarding (2, 4, 6, 9; rows 16,2,16 .. 16,9,16; passes 2)
| row | size | 2 | 4 | 6 | 9 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin3HopEW | 256 | 747.5 | 413.2 | 290.2 | 195.1 | >6 |
| SlsLin3HopEW | 1024 | 766.0 | 425.9 | 292.5 | 201.0 | >6 |
| SlsLin3HopEW | 4352 | 840.7 | 475.3 | 329.3 | 221.3 | >6 |
| SlsLin3HopEW | 8704 | 932.8 | 475.7 | 336.7 | 255.0 | >6 |
| SlsLinRowMerge | 256 | 497.8 | 274.0 | 194.1 | 139.8 | >6 |
| SlsLinRowMerge | 1024 | 509.4 | 287.1 | 196.5 | 141.6 | >6 |
| SlsLinRowMerge | 4352 | 560.0 | 325.3 | 227.8 | 161.7 | >6 |
| SlsLinRowMerge | 8704 | 674.3 | 376.9 | 287.8 | 245.9 | >6 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 1024 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 4352 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 8704 B: >6 (SlsLin3HopEW; rows' knees >6, >6)

## E p1w_8704: receiver (4, 8, 16; rows 16,2,4 .. 16,2,16; passes 2)
| row | size | 4 | 8 | 16 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 263.3 | 149.8 | 114.2 | >8 |
| Lin1HopEWBidir | 1024 | 268.9 | 152.1 | 116.3 | >8 |
| Lin1HopEWBidir | 4352 | 344.4 | 192.2 | 140.2 | >8 |
| Lin1HopEWBidir | 8704 | 395.3 | 247.3 | 244.8 | >8 |
| SlsLin1HopEW | 256 | 263.1 | 149.9 | 114.3 | >8 |
| SlsLin1HopEW | 1024 | 267.0 | 151.3 | 116.1 | >8 |
| SlsLin1HopEW | 4352 | 321.9 | 174.6 | 130.2 | >8 |
| SlsLin1HopEW | 8704 | 364.8 | 243.8 | 243.8 | 5-8 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 1024 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 4352 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 8704 B: >8 (Lin1HopEWBidir; rows' knees 5-8, >8)

## E p1w_8704: worker (4, 8, 12, 16, 20; rows 4,2,16 .. 20,2,16; passes 2)
| row | size | 4 | 8 | 12 | 16 | 20 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 402.6 | 222.6 | 147.0 | 114.3 | 98.4 | >16 |
| Lin1HopEWBidir | 1024 | 413.2 | 225.8 | 149.7 | 116.4 | 98.9 | >16 |
| Lin1HopEWBidir | 4352 | 446.0 | 252.8 | 180.4 | 140.2 | 124.8 | >16 |
| Lin1HopEWBidir | 8704 | 472.1 | 279.4 | 244.8 | 244.8 | 244.8 | 9-12 |
| SlsLin1HopEW | 256 | 419.4 | 222.5 | 147.0 | 114.4 | 98.6 | >16 |
| SlsLin1HopEW | 1024 | 411.6 | 225.6 | 149.0 | 116.0 | 98.8 | >16 |
| SlsLin1HopEW | 4352 | 476.4 | 250.0 | 168.6 | 130.2 | 124.5 | >16 |
| SlsLin1HopEW | 8704 | 457.7 | 262.4 | 243.8 | 243.8 | 243.8 | 9-12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 1024 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 4352 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 8704 B: 9-12 (Lin1HopEWBidir; rows' knees 9-12, 9-12)

## E p2w_8704: receiver (4, 8, 16; rows 16,2,4 .. 16,2,16; passes 2)
| row | size | 4 | 8 | 16 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEW | 256 | 270.5 | 157.5 | 121.0 | >8 |
| Sls1HopEW | 1024 | 277.3 | 159.8 | 121.9 | >8 |
| Sls1HopEW | 4352 | 332.9 | 179.7 | 133.3 | >8 |
| Sls1HopEW | 8704 | 376.7 | 245.1 | 245.1 | 5-8 |
| Sls1HopEWBidir | 256 | 270.3 | 157.5 | 120.9 | >8 |
| Sls1HopEWBidir | 1024 | 278.2 | 160.4 | 122.3 | >8 |
| Sls1HopEWBidir | 4352 | 346.2 | 190.6 | 142.6 | >8 |
| Sls1HopEWBidir | 8704 | 403.7 | 250.4 | 247.2 | >8 |
| Sls1HopNS | 256 | 270.9 | 158.1 | 121.3 | >8 |
| Sls1HopNS | 1024 | 277.9 | 160.2 | 122.2 | >8 |
| Sls1HopNS | 4352 | 332.8 | 180.0 | 133.8 | >8 |
| Sls1HopNS | 8704 | 379.4 | 245.1 | 245.1 | 5-8 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 1024 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 4352 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 8704 B: >8 (Sls1HopEWBidir; rows' knees 5-8, 5-8, >8)

## E p2w_8704: worker (4, 8, 12, 16, 19; rows 4,2,16 .. 19,2,16; passes 2)
| row | size | 4 | 8 | 12 | 16 | 19 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEW | 256 | 426.5 | 231.4 | 155.9 | 121.1 | 114.5 | >16 |
| Sls1HopEW | 1024 | 446.8 | 232.3 | 157.5 | 121.9 | 114.4 | >16 |
| Sls1HopEW | 4352 | 514.7 | 254.4 | 173.2 | 133.2 | 125.8 | >16 |
| Sls1HopEW | 8704 | 466.9 | 266.8 | 245.1 | 245.1 | 245.1 | 9-12 |
| Sls1HopEWBidir | 256 | 423.5 | 232.7 | 155.9 | 120.8 | 114.5 | >16 |
| Sls1HopEWBidir | 1024 | 448.1 | 232.5 | 158.0 | 122.2 | 114.9 | >16 |
| Sls1HopEWBidir | 4352 | 511.2 | 257.0 | 184.5 | 142.5 | 126.9 | >16 |
| Sls1HopEWBidir | 8704 | 485.7 | 284.1 | 247.3 | 247.2 | 247.2 | 9-12 |
| Sls1HopNS | 256 | 423.9 | 231.3 | 156.3 | 121.3 | 114.3 | >16 |
| Sls1HopNS | 1024 | 447.2 | 232.4 | 157.8 | 122.2 | 114.3 | >16 |
| Sls1HopNS | 4352 | 511.6 | 255.1 | 173.3 | 133.8 | 125.8 | >16 |
| Sls1HopNS | 8704 | 467.4 | 267.7 | 245.1 | 245.1 | 245.1 | 9-12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 1024 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 4352 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 8704 B: 9-12 (Sls1HopEW; rows' knees 9-12, 9-12, 9-12)
