
## B p1f_4352: forwarding (2, 4, 8, 12, 16, 24; rows 24,2,32 .. 24,24,32; passes 2)
| row | size | 2 | 4 | 8 | 12 | 16 | 24 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlsLin3HopEW | 256 | 779.2 | 444.2 | 228.9 | 158.3 | 125.4 | 118.3 | >16 |
| SlsLin3HopEW | 1024 | 796.3 | 443.5 | 234.6 | 161.5 | 127.9 | 118.6 | >16 |
| SlsLin3HopEW | 2048 | 828.9 | 458.8 | 242.1 | 166.4 | 130.7 | 118.9 | >16 |
| SlsLin3HopEW | 4352 | 892.8 | 503.5 | 264.7 | 179.7 | 137.7 | 124.5 | >16 |
| SlsLinRowMerge | 256 | 527.4 | 296.5 | 161.1 | 123.3 | 110.8 | 108.8 | >16 |
| SlsLinRowMerge | 1024 | 540.7 | 298.8 | 163.9 | 124.8 | 111.5 | 108.9 | >16 |
| SlsLinRowMerge | 2048 | 558.0 | 310.1 | 167.9 | 126.9 | 112.5 | 109.0 | >16 |
| SlsLinRowMerge | 4352 | 615.1 | 355.3 | 200.0 | 149.0 | 129.2 | 126.1 | >16 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (SlsLin3HopEW; rows' knees >16, >16)
- 1024 B: >16 (SlsLin3HopEW; rows' knees >16, >16)
- 2048 B: >16 (SlsLin3HopEW; rows' knees >16, >16)
- 4352 B: >16 (SlsLin3HopEW; rows' knees >16, >16)

## B p1w_4352: receiver (4, 8, 16, 32; rows 24,8,4 .. 24,8,32; passes 2)
| row | size | 4 | 8 | 16 | 32 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 263.1 | 149.6 | 98.5 | 95.7 | >16 |
| Lin1HopEWBidir | 1024 | 268.7 | 151.9 | 98.8 | 95.7 | >16 |
| Lin1HopEWBidir | 2048 | 305.3 | 159.5 | 100.5 | 95.7 | >16 |
| Lin1HopEWBidir | 4352 | 344.4 | 192.1 | 124.8 | 124.8 | 9-16 |
| SlsLin1HopEW | 256 | 263.1 | 149.7 | 98.6 | 95.6 | >16 |
| SlsLin1HopEW | 1024 | 266.8 | 151.1 | 98.8 | 95.6 | >16 |
| SlsLin1HopEW | 2048 | 298.5 | 155.5 | 99.4 | 95.7 | >16 |
| SlsLin1HopEW | 4352 | 321.7 | 174.4 | 124.5 | 124.5 | 9-16 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 1024 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 2048 B: >16 (Lin1HopEWBidir; rows' knees >16, >16)
- 4352 B: 9-16 (Lin1HopEWBidir; rows' knees 9-16, 9-16)

## B p1w_4352: worker (4, 8, 12, 16, 20, 24, 32; rows 4,2,32 .. 32,2,32; passes 2)
| row | size | 4 | 8 | 12 | 16 | 20 | 24 | 32 | knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lin1HopEWBidir | 256 | 401.8 | 222.5 | 146.9 | 114.0 | 98.2 | 95.8 | 95.6 | 21-24 |
| Lin1HopEWBidir | 1024 | 414.2 | 225.3 | 149.5 | 116.2 | 98.7 | 95.7 | 95.6 | 21-24 |
| Lin1HopEWBidir | 2048 | 433.4 | 226.8 | 155.0 | 119.6 | 100.6 | 95.8 | 95.6 | 21-24 |
| Lin1HopEWBidir | 4352 | 470.7 | 252.7 | 179.8 | 140.1 | 124.8 | 124.8 | 124.8 | 17-20 |
| SlsLin1HopEW | 256 | 419.0 | 222.4 | 146.7 | 114.2 | 98.4 | 95.5 | 95.2 | 21-24 |
| SlsLin1HopEW | 1024 | 413.5 | 225.7 | 148.7 | 115.7 | 98.7 | 95.6 | 95.3 | 21-24 |
| SlsLin1HopEW | 2048 | 433.3 | 226.9 | 152.6 | 118.3 | 99.4 | 95.7 | 95.3 | 21-24 |
| SlsLin1HopEW | 4352 | 474.8 | 249.9 | 168.6 | 130.1 | 124.5 | 124.5 | 124.5 | 17-20 |

Knee per packet size (largest over the rows; the row that sets it):
- 256 B: 21-24 (Lin1HopEWBidir; rows' knees 21-24, 21-24)
- 1024 B: 21-24 (Lin1HopEWBidir; rows' knees 21-24, 21-24)
- 2048 B: 21-24 (Lin1HopEWBidir; rows' knees 21-24, 21-24)
- 4352 B: 17-20 (Lin1HopEWBidir; rows' knees 17-20, 17-20)
