## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| nohook | a | 12 | 1.0000 | 0.9910 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | yes |
| on | a | 48 | 0.9986 | 0.9862 | NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | yes | NO |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2

- nohook pass a: 48.055 GB/s, ratio 0.9910
- off pass a: 48.490 GB/s
- on pass a: 47.820 GB/s, ratio 0.9862

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1

- nohook pass a: 48.315 GB/s, ratio 0.9969
- off pass a: 48.467 GB/s
- on pass a: 48.330 GB/s, ratio 0.9972

## Per test (ratio per pass)

| test | on a | nohook a | off b / off a |
| --- | ---: | ---: | ---: |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9987 | - | - |
| FullRingUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9988 | - | - |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9987 | - | - |
| FullRingUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9986 | - | - |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9991 | - | - |
| FullRingUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9992 | - | - |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9983 | - | - |
| FullRingUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9975 | - | - |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9985 | - | - |
| FullRingUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9986 | - | - |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9989 | - | - |
| FullRingUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9987 | - | - |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9993 | 1.0000 | - |
| NeighborExchangeUnicast_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9992 | 1.0000 | - |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9992 | 1.0001 | - |
| NeighborExchangeUnicast_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9993 | 1.0000 | - |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_1 | 0.9994 | 1.0000 | - |
| NeighborExchangeUnicast_size_2048_ntype_unicast_write_num_links_2 | 0.9995 | 1.0000 | - |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9992 | 1.0000 | - |
| NeighborExchangeUnicast_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9992 | 1.0000 | - |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9991 | 0.9999 | - |
| NeighborExchangeUnicast_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9991 | 1.0000 | - |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_1 | 0.9972 | 0.9969 | - |
| NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2 | 0.9862 | 0.9910 | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9967 | - | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 0.9971 | - | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 0.9926 | - | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9959 | - | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9963 | - | - |
| SingleSenderLinearUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 0.9948 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9959 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9960 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9949 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9946 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9954 | - | - |
| SingleSenderLinearUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9946 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_1 | 0.9990 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_fused_atomic_inc_num_links_2 | 1.0018 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_1 | 1.0049 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_scatter_write_num_links_2 | 0.9990 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_1 | 0.9991 | - | - |
| SingleSenderMeshUnicastAllDevices_size_2048_ntype_unicast_write_num_links_2 | 1.0014 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_1 | 0.9981 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_fused_atomic_inc_num_links_2 | 0.9963 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_1 | 0.9945 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_scatter_write_num_links_2 | 0.9979 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_1 | 0.9950 | - | - |
| SingleSenderMeshUnicastAllDevices_size_4096_ntype_unicast_write_num_links_2 | 0.9985 | - | - |
--- router images: off vs on
active_erisc               {'different': 136}; largest A 9884 B 13564 B; changed B - A +3588 to +3720 B; B LINK_SYNC_ROLE counts {'2': 68, '1': 68}
subordinate_active_erisc   {'different': 136}; largest A 6488 B 7528 B; changed B - A +1032 to +1040 B; B LINK_SYNC_ROLE counts {'0': 136}
--- functions, off vs on, active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x272 link_sync::RouterPort<
    x136 link_sync::PortBase::send_dithered
    x136 eth_ptp::RxStampRule<63ul, 21ul>::uninstall
    x136 kernel_profiler::mark_zone_long
    x136 link_sync::UpdatePeriod::remeasure
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::uninstall
    x136 eth_ptp::restart_ptp_timer
    x136 link_sync::PortBase::record
    x136 link_sync::PortBase::take_burst
    x136 eth_ptp::TxHeaderRow<2ul, 3ul>::install
    x136 kernel_profiler::ring_ensure_room_slow
    x136 link_sync::PortBase::close_round
    x136 eth_ptp::RxStampRule<63ul, 21ul>::install
    x 68 link_sync::ReceiverPort::echo
    x 68 link_sync::ReceiverPort::take
    x 68 link_sync::TransmitterPort::burst
  in both, different code:
    x136 fabric_router::end_round
    x136 _start
    x100 fabric_router::sender::SenderChannelInterfaces::start
--- functions, off vs on, subordinate_active_erisc (code with addresses and immediates masked)
136 pairs; every shared function code-identical in 0
  only in A:
  only in B:
    x136 kernel_profiler::mark_zone_long
    x136 kernel_profiler::ring_ensure_room_slow
  in both, different code:
    x136 _start
--- router images: on vs nohook
active_erisc               {'unpaired': 108, 'different': 28}; largest A 10536 B 8120 B; changed B - A -2456 to -2356 B; B LINK_SYNC_ROLE counts {'1': 11, '2': 17}
subordinate_active_erisc   {'different': 28, 'unpaired': 108}; largest A 6280 B 6280 B; changed B - A +0 to +0 B; B LINK_SYNC_ROLE counts {'0': 28}
--- functions, on vs nohook, active_erisc (code with addresses and immediates masked)
28 pairs; every shared function code-identical in 0
  only in A:
    x 56 link_sync::RouterPort<
    x 28 eth_ptp::restart_ptp_timer
    x 28 eth_ptp::RxStampRule<63ul, 21ul>::install
    x 28 eth_ptp::RxStampRule<63ul, 21ul>::uninstall
    x 28 eth_ptp::TxHeaderRow<2ul, 3ul>::uninstall
    x 28 link_sync::PortBase::send_dithered
    x 28 eth_ptp::TxHeaderRow<2ul, 3ul>::install
    x 28 link_sync::PortBase::take_burst
    x 28 link_sync::UpdatePeriod::remeasure
    x 28 link_sync::PortBase::record
    x 28 link_sync::PortBase::close_round
    x 14 link_sync::TransmitterPort::burst
    x 14 link_sync::ReceiverPort::take
    x 14 link_sync::ReceiverPort::echo
  only in B:
  in both, different code:
    x 28 fabric_router::end_round
    x 28 fabric_router::sender::SenderChannelInterfaces::start
--- functions, on vs nohook, subordinate_active_erisc (code with addresses and immediates masked)
28 pairs; every shared function code-identical in 28
  only in A:
  only in B:
  in both, different code:
