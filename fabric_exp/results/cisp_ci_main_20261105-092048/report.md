## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| on | a | 48 | 1.0048 | 1.0012 | FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | yes | yes |
| on | b | 48 | 1.0050 | 1.0014 | FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | yes | yes |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2

- off pass a: 39.902 GB/s
- on pass a: 41.539 GB/s, ratio 1.0410
- off pass b: 39.896 GB/s
- on pass b: 41.538 GB/s, ratio 1.0411
- off pass-to-pass: -0.0153%

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1

- off pass a: 39.788 GB/s
- on pass a: 41.597 GB/s, ratio 1.0455
- off pass b: 39.775 GB/s
- on pass b: 41.597 GB/s, ratio 1.0458
- off pass-to-pass: -0.0330%

## Per test (ratio per pass)

| test | on a | on b | off b / off a |
| --- | ---: | ---: | ---: |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0018 | 1.0019 | 1.0000 |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0017 | 1.0017 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0015 | 1.0014 | 0.9999 |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0012 | 1.0014 | 0.9998 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_1 | 1.0023 | 1.0023 | 1.0000 |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_2 | 1.0023 | 1.0023 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0016 | 1.0015 | 1.0000 |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0014 | 1.0014 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0020 | 1.0022 | 0.9999 |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0017 | 1.0019 | 0.9999 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_1 | 1.0022 | 1.0021 | 1.0000 |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_2 | 1.0020 | 1.0020 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0400 | 1.0399 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0373 | 1.0368 | 1.0002 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0048 | 1.0045 | 1.0001 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0049 | 1.0052 | 0.9998 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_1 | 1.0410 | 1.0409 | 1.0000 |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_2 | 1.0379 | 1.0378 | 1.0000 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0443 | 1.0454 | 0.9997 |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0400 | 1.0401 | 0.9997 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0041 | 1.0039 | 1.0001 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0042 | 1.0040 | 1.0001 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1 | 1.0455 | 1.0458 | 0.9997 |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | 1.0410 | 1.0411 | 0.9998 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0128 | 1.0128 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0097 | 1.0099 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0055 | 1.0053 | 1.0002 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0048 | 1.0047 | 1.0001 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 1.0134 | 1.0141 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0081 | 1.0081 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0083 | 1.0078 | 1.0003 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0062 | 1.0061 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0084 | 1.0083 | 1.0000 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0065 | 1.0064 | 1.0001 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 1.0085 | 1.0090 | 0.9999 |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 1.0062 | 1.0064 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 1.0055 | 1.0056 | 0.9998 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0044 | 1.0046 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0056 | 1.0057 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 1.0043 | 1.0045 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 1.0055 | 1.0054 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0042 | 1.0043 | 1.0000 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 1.0044 | 1.0050 | 0.9995 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 1.0038 | 1.0036 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 1.0042 | 1.0043 | 0.9998 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 1.0031 | 1.0034 | 0.9999 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 1.0052 | 1.0049 | 1.0001 |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 1.0035 | 1.0038 | 0.9997 |
--- router images: off vs on
active_erisc               {'different': 136}; largest A 12688 B 16444 B; changed B - A +3532 to +3852 B; B LINK_SYNC_ROLE counts {'1': 65, '2': 71}
subordinate_active_erisc   {'different': 136}; largest A 7700 B 8744 B; changed B - A +1040 to +1044 B; B LINK_SYNC_ROLE counts {'0': 136}
--- functions, off vs on, active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x272 link_sync::RouterPort<
    x136 link_sync::PortBase::take_burst
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::install
    x136 eth_ptp::RxStampRule<63ul, 21ul>::uninstall
    x136 kernel_profiler::ring_ensure_room_slow
    x136 link_sync::PortBase::send_dithered
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::uninstall
    x136 kernel_profiler::mark_zone_long
    x136 link_sync::PortBase::record
    x136 eth_ptp::RxStampRule<63ul, 21ul>::install
    x136 link_sync::UpdatePeriod::remeasure
    x136 link_sync::PortBase::close_round
    x136 eth_ptp::restart_ptp_timer
    x 71 link_sync::ReceiverPort::take
    x 71 link_sync::ReceiverPort::echo
    x 65 link_sync::TransmitterPort::burst
  in both, different code:
    x136 kernel_main
    x136 _start
--- functions, off vs on, subordinate_active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x136 kernel_profiler::ring_ensure_room_slow
    x136 kernel_profiler::mark_zone_long
  in both, different code:
    x136 _start
