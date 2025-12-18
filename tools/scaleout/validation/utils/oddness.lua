[1,0]<stdout>: Channel Connections found in FSD but missing in GSD (11 connections):
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=11}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=11}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=4}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=4}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=9}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=9}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:

# QB3-1-1 to QB4-1-1 seems impossible to mess up, but GSD does not detect. Faulty cable?
# QB3-3-3 to QB4-3-3 seems odd - cross QB connections are on all cards, QB3-X-1 <> QB4-X-4. There is no QB3-X-3 to QB4-X-3 connection in the wiring diagram, and not sure why this is found in the FSD...
# QB3-1-2 to QB3-4-2 should be an easy connection to make but does not appear for some reason. However, GSD has QB3-1-2 to QB3-3-2, and QB3-2-2 to QB3-4-2. Perhaps the cards are somehow swapped so that tray_id 4 / 3 are interchanged?

[1,0]<stdout>: Port Connections found in FSD but missing in GSD (7 connections):
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=1, port_type=QSFP_DD, port_id=1} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=1, port_type=QSFP_DD, port_id=1}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=1, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=4, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=2, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=3, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=3, port_type=QSFP_DD, port_id=1} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=3, port_type=QSFP_DD, port_id=1}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=3, port_type=QSFP_DD, port_id=4} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=3, port_type=QSFP_DD, port_id=4}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=1, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=4, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=2, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=3, port_type=QSFP_DD, port_id=2}

[1,0]<stdout>:
[1,0]<stdout>: Channel Connections found in GSD but missing in FSD (7 connections):
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-03', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=8}}
[1,0]<stdout>:   - PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=10}} <-> PhysicalChannelEndpoint{hostname='bh-qbae-04', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=10}}
[1,0]<stdout>:


# these would suggest that some connections were made that dont make sense on both machines, from card 1 to 3, and 2 to 4. These connections are not present in the wiring diagram and should not exist
# this kind of issue seems like the tray_ids somehow dont correspond to the cards from top-down order 1,2,3,4, maybe it's 1,2,4,3?

[1,0]<stdout>: Port Connections found in GSD but missing in FSD (4 connections):
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=1, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=3, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=2, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-03', aisle='A', rack=1, shelf_u=30, tray_id=4, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=1, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=3, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>:   - PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=2, port_type=QSFP_DD, port_id=2} <-> PhysicalPortEndpoint{hostname='bh-qbae-04', aisle='A', rack=1, shelf_u=40, tray_id=4, port_type=QSFP_DD, port_id=2}
[1,0]<stdout>: