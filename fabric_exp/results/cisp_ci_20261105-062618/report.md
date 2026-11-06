## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| on | a | 48 | 0.9986 | 0.9873 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | NO |
| on | b | 48 | 0.9982 | 0.9861 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | NO |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2

- off pass a: 48.490 GB/s
- on pass a: 47.872 GB/s, ratio 0.9873
- off pass b: 48.492 GB/s
- on pass b: 47.819 GB/s, ratio 0.9861
- off pass-to-pass: +0.0037%

## Per test (ratio per pass)

| test | on a | on b | off b / off a |
| --- | ---: | ---: | ---: |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9988 | 0.9988 | 1.0000 |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9988 | 0.9987 | 1.0001 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9988 | 0.9988 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9987 | 0.9987 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9991 | 0.9992 | 0.9999 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9991 | 0.9990 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9980 | 0.9981 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9977 | 0.9975 | 1.0003 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9986 | 0.9985 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9985 | 0.9986 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9989 | 0.9990 | 0.9999 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9989 | 0.9987 | 1.0001 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9993 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9992 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9992 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9993 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9994 | 0.9995 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9995 | 0.9995 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9992 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9993 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9991 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9992 | 0.9991 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9957 | 0.9949 | 1.0001 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9873 | 0.9861 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9970 | 0.9932 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9952 | 0.9949 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9965 | 0.9965 | 1.0003 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9960 | 0.9963 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9967 | 0.9968 | 1.0004 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 0.9947 | 0.9954 | 0.9994 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9969 | 0.9962 | 1.0003 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9958 | 0.9957 | 1.0001 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9955 | 0.9956 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9953 | 0.9948 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9958 | 0.9952 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9956 | 0.9953 | 1.0005 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0030 | 0.9946 | 1.0080 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0041 | 1.0069 | 0.9971 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0032 | 1.0031 | 1.0002 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9970 | 0.9931 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9989 | 0.9991 | 0.9998 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 0.9972 | 0.9953 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9979 | 0.9979 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9988 | 0.9920 | 1.0048 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9982 | 0.9937 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9958 | 0.9958 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9949 | 0.9949 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9964 | 0.9984 | 1.0000 |
--- router images: off vs on
active_erisc               {'different': 136}; largest A 9884 B 13564 B; changed B - A +3588 to +3720 B; B LINK_SYNC_ROLE counts {'2': 68, '1': 68}
subordinate_active_erisc   {'different': 136}; largest A 6488 B 7528 B; changed B - A +1032 to +1040 B; B LINK_SYNC_ROLE counts {'0': 136}
