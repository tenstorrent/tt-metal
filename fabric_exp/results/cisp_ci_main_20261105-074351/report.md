## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| on | a | 48 | 1.0051 | 1.0011 | FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | yes | yes |
| on | b | 48 | 1.0049 | 1.0011 | FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | yes | yes |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2

- off pass a: 39.900 GB/s
- on pass a: 41.539 GB/s, ratio 1.0411
- off pass b: 39.903 GB/s
- on pass b: 41.541 GB/s, ratio 1.0410
- off pass-to-pass: +0.0090%

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1

- off pass a: 39.778 GB/s
- on pass a: 41.600 GB/s, ratio 1.0458
- off pass b: 39.790 GB/s
- on pass b: 41.597 GB/s, ratio 1.0454
- off pass-to-pass: +0.0292%

## Per test (ratio per pass)

| test | on a | on b | off b / off a |
| --- | ---: | ---: | ---: |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0018 | 1.0017 | 1.0000 |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0017 | 1.0017 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0016 | 1.0013 | 1.0004 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0011 | 1.0011 | 1.0001 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_1 | 1.0023 | 1.0023 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_2 | 1.0023 | 1.0023 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0015 | 1.0016 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0014 | 1.0014 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0022 | 1.0021 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0018 | 1.0016 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_1 | 1.0021 | 1.0021 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_2 | 1.0020 | 1.0020 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0413 | 1.0400 | 1.0011 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0374 | 1.0370 | 1.0005 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0050 | 1.0048 | 1.0003 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0049 | 1.0051 | 0.9999 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_1 | 1.0416 | 1.0413 | 1.0003 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_2 | 1.0385 | 1.0380 | 1.0005 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0442 | 1.0442 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0399 | 1.0399 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0040 | 1.0040 | 0.9999 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0043 | 1.0041 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1 | 1.0458 | 1.0454 | 1.0003 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | 1.0411 | 1.0410 | 1.0001 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0127 | 1.0128 | 0.9998 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0095 | 1.0096 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0052 | 1.0055 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0047 | 1.0048 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 1.0135 | 1.0140 | 0.9997 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0079 | 1.0084 | 0.9994 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0081 | 1.0083 | 0.9997 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0061 | 1.0062 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0082 | 1.0082 | 0.9997 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0064 | 1.0065 | 0.9998 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 1.0091 | 1.0082 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 1.0063 | 1.0063 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0057 | 1.0056 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0044 | 1.0043 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0057 | 1.0059 | 0.9998 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0044 | 1.0042 | 1.0002 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 1.0054 | 1.0054 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0041 | 1.0043 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0054 | 1.0046 | 1.0004 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0036 | 1.0038 | 0.9997 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0042 | 1.0045 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0035 | 1.0033 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 1.0052 | 1.0050 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 1.0037 | 1.0037 | 0.9998 |
--- router images: off vs on
active_erisc               {'different': 136}; largest A 12688 B 16444 B; changed B - A +3532 to +3852 B; B LINK_SYNC_ROLE counts {'1': 65, '2': 71}
subordinate_active_erisc   {'different': 136}; largest A 7700 B 8744 B; changed B - A +1040 to +1044 B; B LINK_SYNC_ROLE counts {'0': 136}
--- functions, off vs on, active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x272 link_sync::RouterPort<
    x136 link_sync::UpdatePeriod::remeasure
    x136 link_sync::PortBase::close_round
    x136 eth_ptp::RxStampRule<63ul, 21ul>::uninstall
    x136 link_sync::PortBase::send_dithered
    x136 kernel_profiler::ring_ensure_room_slow
    x136 link_sync::PortBase::record
    x136 kernel_profiler::mark_zone_long
    x136 eth_ptp::restart_ptp_timer
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::install
    x136 link_sync::PortBase::take_burst
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::uninstall
    x136 eth_ptp::RxStampRule<63ul, 21ul>::install
    x 71 link_sync::ReceiverPort::take
    x 71 link_sync::ReceiverPort::echo
    x 65 link_sync::TransmitterPort::burst
  in both, different code:
    x136 _start
    x136 kernel_main
--- functions, off vs on, subordinate_active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x136 kernel_profiler::mark_zone_long
    x136 kernel_profiler::ring_ensure_room_slow
  in both, different code:
    x136 _start
