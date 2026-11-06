## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| on | a | 48 | 0.9986 | 0.9865 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | NO |
| on | b | 48 | 0.9986 | 0.9884 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | NO |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2

- off pass a: 48.489 GB/s
- on pass a: 47.836 GB/s, ratio 0.9865
- off pass b: 48.489 GB/s
- on pass b: 47.927 GB/s, ratio 0.9884
- off pass-to-pass: +0.0011%

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1

- off pass a: 48.468 GB/s
- on pass a: 48.331 GB/s, ratio 0.9972
- off pass b: 48.467 GB/s
- on pass b: 48.330 GB/s, ratio 0.9972
- off pass-to-pass: -0.0016%

## Per test (ratio per pass)

| test | on a | on b | off b / off a |
| --- | ---: | ---: | ---: |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9988 | 0.9988 | 1.0000 |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9988 | 0.9988 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9988 | 0.9987 | 1.0001 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9987 | 0.9987 | 1.0001 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9992 | 0.9993 | 0.9999 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9991 | 0.9990 | 1.0001 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9981 | 0.9979 | 1.0001 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9974 | 0.9975 | 0.9999 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9987 | 0.9986 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9985 | 0.9985 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9991 | 0.9990 | 1.0001 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9988 | 0.9988 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9992 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9992 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9992 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9993 | 0.9993 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9995 | 0.9995 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9995 | 0.9994 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9992 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9992 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9991 | 0.9992 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9992 | 0.9991 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9972 | 0.9972 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9865 | 0.9884 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9969 | 0.9931 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9969 | 0.9953 | 0.9998 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9964 | 0.9961 | 1.0006 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9961 | 0.9964 | 0.9997 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9965 | 0.9969 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 0.9965 | 0.9964 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9958 | 0.9965 | 0.9997 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9960 | 0.9962 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9949 | 0.9957 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9954 | 0.9953 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9952 | 0.9959 | 0.9998 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9954 | 0.9953 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0033 | 0.9988 | 1.0039 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0061 | 1.0155 | 0.9907 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0027 | 0.9986 | 1.0040 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9949 | 0.9970 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9991 | 0.9989 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0042 | 1.0015 | 1.0049 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9982 | 1.0033 | 0.9947 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9924 | 0.9938 | 1.0026 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9946 | 0.9984 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9958 | 0.9978 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9949 | 0.9946 | 1.0002 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9966 | 0.9988 | 1.0000 |
--- router images: off vs on
active_erisc               {'different': 136}; largest A 9884 B 13564 B; changed B - A +3588 to +3720 B; B LINK_SYNC_ROLE counts {'2': 68, '1': 68}
subordinate_active_erisc   {'different': 136}; largest A 6488 B 7528 B; changed B - A +1032 to +1040 B; B LINK_SYNC_ROLE counts {'0': 136}
--- functions, off vs on, active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x272 link_sync::RouterPort<
    x136 eth_ptp::RxStampRule<63ul, 21ul>::uninstall
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::uninstall
    x136 link_sync::PortBase::take_burst
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::install
    x136 link_sync::UpdatePeriod::remeasure
    x136 eth_ptp::restart_ptp_timer
    x136 kernel_profiler::mark_zone_long
    x136 link_sync::PortBase::record
    x136 eth_ptp::RxStampRule<63ul, 21ul>::install
    x136 link_sync::PortBase::close_round
    x136 link_sync::PortBase::send_dithered
    x136 kernel_profiler::ring_ensure_room_slow
    x 68 link_sync::ReceiverPort::echo
    x 68 link_sync::ReceiverPort::take
    x 68 link_sync::TransmitterPort::burst
  in both, different code:
    x136 _start
    x136 fabric_router::end_round
    x100 fabric_router::sender::SenderChannelInterfaces::start
--- functions, off vs on, subordinate_active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x136 kernel_profiler::ring_ensure_room_slow
    x136 kernel_profiler::mark_zone_long
  in both, different code:
    x136 _start
