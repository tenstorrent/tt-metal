
## E p1f_8704: forwarding (2, 4, 6, 9; rows 16,2,16 .. 16,9,16; passes 2)
| row | size | 2 | 4 | 6 | 9 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin3HopEW | 256 | 833.1 | 457.3 | 320.2 | 217.2 | >6 |
| SlsLin3HopEW | 1024 | 831.8 | 465.3 | 324.4 | 220.9 | >6 |
| SlsLin3HopEW | 4352 | 927.6 | 513.4 | 349.9 | 243.4 | >6 |
| SlsLin3HopEW | 8704 | 1012.9 | 512.8 | 359.9 | 267.9 | >6 |
| SlsLinRowMerge | 256 | 495.1 | 275.1 | 189.7 | 133.7 | >6 |
| SlsLinRowMerge | 1024 | 500.5 | 278.4 | 192.8 | 136.8 | >6 |
| SlsLinRowMerge | 4352 | 563.1 | 314.1 | 220.3 | 158.7 | >6 |
| SlsLinRowMerge | 8704 | 678.5 | 374.8 | 288.0 | 245.7 | >6 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 1024 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 4352 B: >6 (SlsLin3HopEW; rows' knees >6, >6)
- 8704 B: >6 (SlsLin3HopEW; rows' knees >6, >6)

## E p1w_8704: receiver (4, 8, 16; rows 16,2,4 .. 16,2,16; passes 2)
| row | size | 4 | 8 | 16 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 263.1 | 149.7 | 114.1 | >8 |
| Lin1HopEWBidir | 1024 | 268.4 | 152.0 | 116.3 | >8 |
| Lin1HopEWBidir | 4352 | 343.7 | 191.8 | 140.2 | >8 |
| Lin1HopEWBidir | 8704 | 394.5 | 247.2 | 244.8 | 5-8 |
| SlsLin1HopEW | 256 | 263.0 | 149.8 | 114.3 | >8 |
| SlsLin1HopEW | 1024 | 266.4 | 151.0 | 115.9 | >8 |
| SlsLin1HopEW | 4352 | 321.2 | 174.3 | 130.1 | >8 |
| SlsLin1HopEW | 8704 | 364.3 | 243.8 | 243.8 | 5-8 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 1024 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 4352 B: >8 (Lin1HopEWBidir; rows' knees >8, >8)
- 8704 B: 5-8 (Lin1HopEWBidir; rows' knees 5-8, 5-8)

## E p1w_8704: worker (4, 8, 12, 16, 20; rows 4,2,16 .. 20,2,16; passes 2)
| row | size | 4 | 8 | 12 | 16 | 20 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 402.3 | 222.3 | 146.9 | 114.1 | 98.4 | >16 |
| Lin1HopEWBidir | 1024 | 415.1 | 225.5 | 149.5 | 116.1 | 98.8 | >16 |
| Lin1HopEWBidir | 4352 | 455.4 | 252.5 | 180.6 | 140.1 | 124.8 | >16 |
| Lin1HopEWBidir | 8704 | 471.3 | 279.3 | 244.8 | 244.8 | 244.8 | 9-12 |
| SlsLin1HopEW | 256 | 417.7 | 222.3 | 146.8 | 114.2 | 98.6 | >16 |
| SlsLin1HopEW | 1024 | 413.7 | 225.7 | 148.7 | 115.9 | 98.8 | >16 |
| SlsLin1HopEW | 4352 | 472.4 | 249.9 | 168.2 | 130.0 | 124.5 | >16 |
| SlsLin1HopEW | 8704 | 457.2 | 262.2 | 243.8 | 243.8 | 243.8 | 9-12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 1024 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 4352 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 8704 B: 9-12 (Lin1HopEWBidir; rows' knees 9-12, 9-12)

## E p2w_8704: receiver (4, 8, 16; rows 16,2,4 .. 16,2,16; passes 2)
| row | size | 4 | 8 | 16 | knee |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEW | 256 | 269.8 | 157.2 | 121.0 | >8 |
| Sls1HopEW | 1024 | 276.6 | 159.6 | 121.8 | >8 |
| Sls1HopEW | 4352 | 332.4 | 179.6 | 133.1 | >8 |
| Sls1HopEW | 8704 | 376.1 | 245.1 | 245.1 | 5-8 |
| Sls1HopEWBidir | 256 | 269.5 | 157.0 | 120.8 | >8 |
| Sls1HopEWBidir | 1024 | 277.6 | 160.2 | 122.1 | >8 |
| Sls1HopEWBidir | 4352 | 345.9 | 190.3 | 142.5 | >8 |
| Sls1HopEWBidir | 8704 | 403.0 | 250.3 | 247.2 | >8 |
| Sls1HopNS | 256 | 270.5 | 158.0 | 121.2 | >8 |
| Sls1HopNS | 1024 | 277.6 | 160.1 | 122.2 | >8 |
| Sls1HopNS | 4352 | 332.5 | 180.0 | 133.7 | >8 |
| Sls1HopNS | 8704 | 379.2 | 245.1 | 245.1 | 5-8 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 1024 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 4352 B: >8 (Sls1HopEW; rows' knees >8, >8, >8)
- 8704 B: >8 (Sls1HopEWBidir; rows' knees 5-8, 5-8, >8)

## E p2w_8704: worker (4, 8, 12, 16, 19; rows 4,2,16 .. 19,2,16; passes 2)
| row | size | 4 | 8 | 12 | 16 | 19 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sls1HopEW | 256 | 422.4 | 231.6 | 155.7 | 121.0 | 114.3 | >16 |
| Sls1HopEW | 1024 | 445.7 | 232.2 | 157.4 | 121.7 | 114.3 | >16 |
| Sls1HopEW | 4352 | 514.6 | 254.2 | 173.2 | 133.1 | 125.8 | >16 |
| Sls1HopEW | 8704 | 465.9 | 266.7 | 245.1 | 245.1 | 245.1 | 9-12 |
| Sls1HopEWBidir | 256 | 421.0 | 232.6 | 155.8 | 120.8 | 114.9 | >16 |
| Sls1HopEWBidir | 1024 | 444.1 | 232.4 | 157.9 | 122.3 | 114.6 | >16 |
| Sls1HopEWBidir | 4352 | 511.0 | 256.9 | 184.4 | 142.5 | 126.9 | >16 |
| Sls1HopEWBidir | 8704 | 484.9 | 284.0 | 247.3 | 247.2 | 247.2 | 9-12 |
| Sls1HopNS | 256 | 419.8 | 231.5 | 156.2 | 121.2 | 114.3 | >16 |
| Sls1HopNS | 1024 | 444.5 | 232.1 | 157.7 | 122.1 | 114.4 | >16 |
| Sls1HopNS | 4352 | 511.9 | 254.9 | 173.3 | 133.7 | 125.8 | >16 |
| Sls1HopNS | 8704 | 467.1 | 267.6 | 245.1 | 245.1 | 245.1 | 9-12 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 1024 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 4352 B: >16 (Sls1HopEW; rows' knees >16, >16, >16)
- 8704 B: 9-12 (Sls1HopEW; rows' knees 9-12, 9-12, 9-12)
