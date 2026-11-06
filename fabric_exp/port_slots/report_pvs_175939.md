Slot-row check (both hosts' logged rows per suite):
   rows match (ns)
   rows match (t42ns)
   rows match (c2d)
   rows match (c1lin)
   rows match (c1ring)
   rows match (c1ne)
   rows match (c1ne)
   rows match (c1ring)
   rows match (c1lin)
   rows match (c2d)
   rows match (t42ns)
   rows match (ns)

## ns (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| EW_Atomic | 4352 | atomic_inc | 130.0 | 130.0 | 130.2 | 130.2 | +0.1% |
| EW_Write | 16 | unicast_write | 116.6 | 116.7 | 120.2 | 120.1 | +3.0% |
| EW_Write | 4352 | unicast_write | 127.2 | 127.2 | 128.3 | 128.4 | +0.9% |
| NS_Atomic | 4352 | atomic_inc | 132.3 | 132.3 | 140.3 | 140.3 | +6.0% |
| NS_Write | 16 | unicast_write | 117.1 | 117.0 | 121.2 | 121.3 | +3.6% |
| NS_Write | 4352 | unicast_write | 127.5 | 127.4 | 130.3 | 130.3 | +2.2% |

## t42ns (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TorusXY4x2_1HopEW | 256 | unicast_scatter_write | 186.7 | 186.8 | 185.7 | 185.7 | -0.5% |
| TorusXY4x2_1HopEW | 1024 | unicast_scatter_write | 186.6 | 186.6 | 185.5 | 185.5 | -0.6% |
| TorusXY4x2_1HopEW | 2048 | unicast_scatter_write | 186.1 | 186.0 | 185.1 | 185.1 | -0.5% |
| TorusXY4x2_1HopEW | 3072 | unicast_scatter_write | 185.9 | 185.9 | 185.0 | 185.0 | -0.5% |
| TorusXY4x2_1HopEW | 4352 | unicast_scatter_write | 186.1 | 186.1 | 185.1 | 185.1 | -0.5% |
| TorusXY4x2_1HopEW | 256 | unicast_write | 146.6 | 146.6 | 145.7 | 145.7 | -0.6% |
| TorusXY4x2_1HopEW | 1024 | unicast_write | 146.6 | 146.6 | 145.8 | 145.8 | -0.6% |
| TorusXY4x2_1HopEW | 2048 | unicast_write | 146.7 | 146.7 | 146.1 | 146.0 | -0.5% |
| TorusXY4x2_1HopEW | 3072 | unicast_write | 146.9 | 146.9 | 146.3 | 146.2 | -0.4% |
| TorusXY4x2_1HopEW | 4352 | unicast_write | 146.9 | 146.8 | 146.0 | 146.0 | -0.6% |
| TorusXY4x2_1HopNS | 256 | unicast_scatter_write | 187.7 | 187.6 | 196.9 | 196.9 | +4.9% |
| TorusXY4x2_1HopNS | 1024 | unicast_scatter_write | 187.7 | 187.7 | 197.0 | 197.0 | +5.0% |
| TorusXY4x2_1HopNS | 2048 | unicast_scatter_write | 187.0 | 187.1 | 197.5 | 197.4 | +5.6% |
| TorusXY4x2_1HopNS | 3072 | unicast_scatter_write | 187.0 | 187.0 | 197.1 | 197.1 | +5.4% |
| TorusXY4x2_1HopNS | 4352 | unicast_scatter_write | 187.1 | 187.1 | 197.1 | 197.1 | +5.3% |
| TorusXY4x2_1HopNS | 256 | unicast_write | 147.6 | 147.5 | 158.6 | 158.6 | +7.5% |
| TorusXY4x2_1HopNS | 1024 | unicast_write | 147.6 | 147.6 | 158.7 | 158.7 | +7.6% |
| TorusXY4x2_1HopNS | 2048 | unicast_write | 147.8 | 147.8 | 157.9 | 158.0 | +6.9% |
| TorusXY4x2_1HopNS | 3072 | unicast_write | 148.0 | 148.0 | 158.1 | 158.1 | +6.8% |
| TorusXY4x2_1HopNS | 4352 | unicast_write | 147.9 | 147.9 | 158.2 | 158.2 | +7.0% |
| TorusXY4x2_1HopNSBidir | 256 | unicast_scatter_write | 187.4 | 187.2 | 196.5 | 196.5 | +4.9% |
| TorusXY4x2_1HopNSBidir | 1024 | unicast_scatter_write | 187.3 | 187.4 | 196.8 | 196.8 | +5.0% |
| TorusXY4x2_1HopNSBidir | 2048 | unicast_scatter_write | 186.7 | 186.8 | 197.2 | 197.2 | +5.6% |
| TorusXY4x2_1HopNSBidir | 3072 | unicast_scatter_write | 187.1 | 187.1 | 196.8 | 196.8 | +5.2% |
| TorusXY4x2_1HopNSBidir | 4352 | unicast_scatter_write | 187.7 | 187.7 | 197.4 | 197.3 | +5.1% |
| TorusXY4x2_1HopNSBidir | 256 | unicast_write | 147.3 | 147.3 | 158.2 | 158.3 | +7.4% |
| TorusXY4x2_1HopNSBidir | 1024 | unicast_write | 147.4 | 147.4 | 158.4 | 158.4 | +7.5% |
| TorusXY4x2_1HopNSBidir | 2048 | unicast_write | 147.8 | 147.8 | 157.8 | 157.9 | +6.8% |
| TorusXY4x2_1HopNSBidir | 3072 | unicast_write | 147.8 | 147.8 | 158.3 | 158.3 | +7.1% |
| TorusXY4x2_1HopNSBidir | 4352 | unicast_write | 148.2 | 148.2 | 158.3 | 158.3 | +6.8% |
| TorusXY4x2_3HopNS | 256 | unicast_scatter_write | 187.7 | 187.7 | 196.9 | 196.9 | +4.9% |
| TorusXY4x2_3HopNS | 1024 | unicast_scatter_write | 187.8 | 187.8 | 197.1 | 197.0 | +4.9% |
| TorusXY4x2_3HopNS | 2048 | unicast_scatter_write | 187.1 | 187.0 | 197.5 | 197.5 | +5.6% |
| TorusXY4x2_3HopNS | 3072 | unicast_scatter_write | 187.0 | 187.0 | 197.0 | 197.0 | +5.4% |
| TorusXY4x2_3HopNS | 4352 | unicast_scatter_write | 187.1 | 187.0 | 197.0 | 197.1 | +5.4% |
| TorusXY4x2_3HopNS | 256 | unicast_write | 147.6 | 147.5 | 158.5 | 158.5 | +7.4% |
| TorusXY4x2_3HopNS | 1024 | unicast_write | 147.5 | 147.5 | 158.6 | 158.6 | +7.5% |
| TorusXY4x2_3HopNS | 2048 | unicast_write | 147.7 | 147.7 | 157.9 | 157.9 | +6.9% |
| TorusXY4x2_3HopNS | 3072 | unicast_write | 148.0 | 148.0 | 158.1 | 158.1 | +6.8% |
| TorusXY4x2_3HopNS | 4352 | unicast_write | 147.9 | 147.9 | 158.2 | 158.2 | +7.0% |
| TorusXY4x2_Wrap1HopNS | 256 | unicast_scatter_write | 187.8 | 187.8 | 196.9 | 196.9 | +4.9% |
| TorusXY4x2_Wrap1HopNS | 1024 | unicast_scatter_write | 187.8 | 187.7 | 197.0 | 197.0 | +5.0% |
| TorusXY4x2_Wrap1HopNS | 2048 | unicast_scatter_write | 187.1 | 187.1 | 197.4 | 197.5 | +5.5% |
| TorusXY4x2_Wrap1HopNS | 3072 | unicast_scatter_write | 187.0 | 186.9 | 197.1 | 197.1 | +5.5% |
| TorusXY4x2_Wrap1HopNS | 4352 | unicast_scatter_write | 187.1 | 187.1 | 197.0 | 197.0 | +5.3% |
| TorusXY4x2_Wrap1HopNS | 256 | unicast_write | 147.6 | 147.6 | 158.5 | 158.5 | +7.4% |
| TorusXY4x2_Wrap1HopNS | 1024 | unicast_write | 147.5 | 147.6 | 158.7 | 158.7 | +7.6% |
| TorusXY4x2_Wrap1HopNS | 2048 | unicast_write | 147.8 | 147.8 | 158.0 | 158.0 | +6.9% |
| TorusXY4x2_Wrap1HopNS | 3072 | unicast_write | 148.0 | 148.0 | 158.2 | 158.1 | +6.8% |
| TorusXY4x2_Wrap1HopNS | 4352 | unicast_write | 147.9 | 147.9 | 158.2 | 158.2 | +6.9% |

## c2d (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Mesh1HopEW | 4352 | unicast_scatter_write | 142.4 | 142.3 | 142.5 | 142.5 | +0.1% |
| Mesh1HopEW | 4352 | unicast_write | 127.2 | 127.2 | 128.3 | 128.4 | +0.9% |
| Mesh1HopEWBidir | 4352 | unicast_scatter_write | 143.6 | 143.7 | 143.9 | 143.9 | +0.1% |
| Mesh1HopEWBidir | 4352 | unicast_write | 135.6 | 135.6 | 136.0 | 136.0 | +0.3% |
| Mesh1HopNS | 4352 | unicast_scatter_write | 145.0 | 145.0 | 152.8 | 152.7 | +5.4% |
| Mesh1HopNS | 4352 | unicast_write | 127.5 | 127.4 | 130.3 | 130.2 | +2.2% |
| Mesh3HopEW | 4352 | unicast_scatter_write | 145.4 | 145.3 | 144.6 | 144.6 | -0.5% |
| Mesh3HopEW | 4352 | unicast_write | 142.5 | 142.6 | 138.3 | 138.3 | -3.0% |
| MeshMulticast | 4352 | unicast_scatter_write_num_links_1 | 522.0 | 522.2 | 522.4 | 522.6 | +0.1% |
| MeshMulticast | 4352 | unicast_write_num_links_1 | 424.4 | 424.5 | 416.9 | 416.9 | -1.8% |

## c1lin (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Linear1HopEW | 4352 | unicast_scatter_write | 133.4 | 133.4 | 133.5 | 133.5 | +0.0% |
| Linear1HopEW | 4352 | unicast_write | 124.5 | 124.5 | 124.5 | 124.5 | +0.0% |
| Linear1HopEWBidir | 4352 | unicast_scatter_write | 133.4 | 133.4 | 133.4 | 133.4 | -0.0% |
| Linear1HopEWBidir | 4352 | unicast_write | 124.8 | 124.8 | 124.8 | 124.8 | +0.0% |
| Linear1HopNS | 4352 | unicast_scatter_write | 133.4 | 133.4 | 133.4 | 133.5 | +0.1% |
| Linear1HopNS | 4352 | unicast_write | 124.5 | 124.5 | 124.5 | 124.5 | +0.0% |
| Linear3HopEW | 4352 | unicast_scatter_write | 133.5 | 133.5 | 133.5 | 133.5 | -0.0% |
| Linear3HopEW | 4352 | unicast_write | 124.5 | 124.5 | 124.5 | 124.5 | +0.0% |
| LinearMulticast | 4352 | unicast_scatter_write_num_links_1 | 507.0 | 507.0 | 506.6 | 506.9 | -0.1% |
| LinearMulticast | 4352 | unicast_write_num_links_1 | 428.4 | 428.3 | 427.9 | 427.6 | -0.1% |
| UnidirLinearMulticast | 4352 | unicast_scatter_write_num_links_1 | 453.4 | 453.2 | 453.3 | 453.3 | -0.0% |
| UnidirLinearMulticast | 4352 | unicast_write_num_links_1 | 374.0 | 373.5 | 373.1 | 373.4 | -0.1% |

## c1ring (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HalfRingMulticast | 4352 | unicast_scatter_write_num_links_1 | 225.3 | 225.2 | 224.0 | 224.0 | -0.6% |
| HalfRingMulticast | 4352 | unicast_write_num_links_1 | 172.9 | 172.9 | 171.7 | 171.7 | -0.7% |

## c1ne (runs: a/port, a/study, b/port, b/study)

| row | size | ntype | study a | study b | port a | port b | port vs study |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| NeighborExchange | 4352 | unicast_scatter_write_num_links_1 | 128.9 | 128.9 | 128.9 | 128.9 | +0.0% |
| NeighborExchange | 4352 | unicast_write_num_links_1 | 124.8 | 124.8 | 124.8 | 124.8 | +0.0% |

